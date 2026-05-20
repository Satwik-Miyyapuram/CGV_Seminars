from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from gsplat.strategy.base import Strategy
from gsplat.strategy.ops import _update_param_with_optimizer, duplicate, remove, reset_opa, split
from nerfstudio.utils.math import random_quat_tensor
from nerfstudio.utils.spherical_harmonics import num_sh_bases
from torch.nn import Parameter

if TYPE_CHECKING:
    from ges_model import GESModel


@dataclass
class GESStrategy(Strategy):
    """GES strategy for the GS densification.

    This class implements the GES strategy for the GS densification. It defines
    the operations to be performed before and after the `loss.backward()` call.

    The strategy uses absolute gradients (absgrad) by default, following AbsGS
    (arXiv:2404.10484), which prevents positive/negative gradient cancellation
    and produces significantly better densification decisions.
    """

    prune_opa: float = 0.005
    grow_grad2d: float = 0.0002
    grow_scale3d: float = 0.01
    grow_scale2d: float = 0.05
    prune_scale3d: float = 0.1
    prune_scale2d: float = 0.15
    refine_start_iter: int = 500
    # BUG 2 FIX: Separate stop iterations per phase. Surfel densification
    # stops at surfel_density_stop_iter (10k), Gaussian densification runs
    # from gaussian_spawn_iter (20k) to refine_stop_iter_gaussian (30k).
    # The old single refine_stop_iter=15000 blocked ALL densification after
    # 15k, meaning Gaussians spawned at 20k could never be split/duplicated.
    refine_stop_iter: int = 15_000  # kept for surfel phase (but surfel phase ends at 10k anyway)
    refine_stop_iter_gaussian: int = 30_000  # NEW: allow Gaussian densification until training ends
    reset_every: int = 3000
    refine_every: int = 100
    verbose: bool = False
    # BUG 1 FIX: Use absolute gradients for densification. Without absgrad,
    # positive and negative gradients cancel out, severely underestimating
    # which primitives need to be split/duplicated. This matches nerfstudio's
    # Splatfacto default (use_absgrad=True) and gsplat DefaultStrategy.
    absgrad: bool = True

    surfel_density_stop_iter: int = 10000
    surfel_prune_iter: int = 15000
    gaussian_spawn_iter: int = 20000
    
    # Dynamic culling parameters
    surfel_visibility_threshold_real: float = 16.0  # Pixel threshold for real scenes
    surfel_visibility_threshold_synthetic: float = 4.0  # Pixel threshold for synthetic scenes
    use_real_scene: bool = True  # Whether using real scene threshold

    def initialize_state(self, scene_scale: float = 1.0) -> dict[str, Any]:
        """Initialize the strategy state."""
        state = {
            "surfels": {
                "radii": None,  # This will be updated during training with the actual
                # radii of the surfels
                "grad2d": None,
                "count": None,
                "surfel_radii_cache": None,
                "scene_scale": scene_scale,
            },
            "gaussians": {
                "radii": None,  # This will be updated during training with the actual radii
                # of the gaussians
                "grad2d": None,
                "count": None,
                "scene_scale": scene_scale,
            },
        }
        return state

    def _clean_optimizer_states(self, model: GESModel):
        """Clean up all optimizer states to remove any stale parameter entries."""
        for opt_name, optimizer in model.optimizers.items():
            # Build set of parameter IDs currently in param_groups
            valid_ids = set()
            for param_group in optimizer.param_groups:
                for param in param_group.get("params", []):
                    valid_ids.add(id(param))
            
            # Remove stale state entries (keys must be Tensors whose id is not in valid_ids)
            stale_keys = [
                k 
                for k in optimizer.state.keys() 
                if isinstance(k, torch.Tensor) and id(k) not in valid_ids
            ]
            for k in stale_keys:
                del optimizer.state[k]

    def step_pre_backward(self, model: GESModel, step: int):
        """Operations to be performed before the `loss.backward()` call.
        
        Calls retain_grad() on the gradient tensor so its .grad (and .absgrad
        when absgrad=True was passed to the rasterizer) are kept after backward.
        """
        if step <= self.surfel_density_stop_iter:
            surfel_info = model.info.get("surfels")
            if surfel_info is not None and "means2d" in surfel_info:
                surfel_info["means2d"].retain_grad()
        elif step > self.gaussian_spawn_iter:
            gaussians_info = model.info.get("gaussians")
            if gaussians_info is not None and "means2d" in gaussians_info:
                gaussians_info["means2d"].retain_grad()

    def _update_state(self, params: dict[str, Parameter], state: dict[str, Any], info: dict):
        """Helper function to update the strategy state.
        
        BUG 1 FIX: Use absgrad (absolute value of gradients) instead of raw
        .grad when self.absgrad is True. Without this, positive and negative
        gradients cancel during accumulation, leading to an undercount of
        which primitives need densification. This was the primary cause of
        too few surfels being created during steps 500-10000.
        
        The gsplat DefaultStrategy uses:
            if self.absgrad:
                grads = info[key].absgrad.clone()
            else:
                grads = info[key].grad.clone()
        We follow the same pattern here.
        """
        means2d = info["means2d"]
        if self.absgrad:
            # absgrad is set by the rasterizer when absgrad=True is passed.
            # It contains |dL/d(means2d)| — the absolute value of gradients.
            if hasattr(means2d, "absgrad") and means2d.absgrad is not None:
                grads = means2d.absgrad.clone()
            else:
                # Fallback: if absgrad wasn't computed (e.g. first step or
                # rasterizer didn't set it), use regular grad with abs().
                if means2d.grad is not None:
                    grads = means2d.grad.abs().clone()
                else:
                    return  # No gradient available yet
        else:
            if means2d.grad is not None:
                grads = means2d.grad.clone()
            else:
                return  # No gradient available yet

        grads[..., 0] *= info["width"] / 2.0 * info["n_cameras"]
        grads[..., 1] *= info["height"] / 2.0 * info["n_cameras"]

        n_items = len(list(params.values())[0])

        if state["grad2d"] is None:
            state["grad2d"] = torch.zeros(n_items, device=grads.device, dtype=torch.float32)
        if state["count"] is None:
            state["count"] = torch.zeros(n_items, device=grads.device, dtype=torch.float32)
        if state["radii"] is None:
            state["radii"] = torch.zeros(n_items, device=grads.device, dtype=torch.float32)

        # For 2DGS (surfel) rasterization, radii shape is [C, N, 2] (two axis-
        # aligned bounding-box radii). For standard 3DGS, it's [C, N].
        # The .all(dim=-1) collapses the trailing dim: [C,N,2] -> [C,N] (checks
        # both radii > 0). For [C,N] it would wrongly collapse N, but gsplat's
        # own DefaultStrategy uses the same pattern (line 257), so this is the
        # canonical approach for 2DGS radii.
        sel = (info["radii"] > 0.0).all(dim=-1)  # [C, N]
        gs_ids = torch.where(sel)[1]  # [nnz]
        grads = grads[sel]  # [nnz, 2]
        radii = info["radii"][sel].max(dim=-1).values  # [nnz]

        state["grad2d"].index_add_(0, gs_ids, grads.norm(dim=-1))
        state["count"].index_add_(0, gs_ids, torch.ones_like(gs_ids, dtype=torch.float32))

        # ensure radii buffer is float so the indexed assignment matches dtypes
        if state["radii"].dtype != torch.float32:
            state["radii"] = state["radii"].to(dtype=torch.float32, device=grads.device)

        state["radii"][gs_ids] = torch.maximum(
            state["radii"][gs_ids],
            # normalize radii to [0, 1] screen space
            (radii / float(max(info["width"], info["height"]))).to(device=grads.device, dtype=torch.float32),
        )

    def step_post_backward(self, model: GESModel, step: int):
        """Operations to be performed after the `loss.backward()` call.
        
        BUG 2 FIX: Use phase-specific stop iterations instead of a single
        refine_stop_iter. The surfel phase runs densification from
        refine_start_iter to surfel_density_stop_iter (500-10000). The
        Gaussian phase runs from gaussian_spawn_iter to
        refine_stop_iter_gaussian (20000-30000). Previously, the single
        refine_stop_iter=15000 caused step_post_backward to return early
        for ALL steps >= 15000, which meant Gaussians spawned at 20k
        could never be densified (split/duplicated/pruned).
        """
        # Determine what is being densified based on the current phase
        if step <= self.surfel_density_stop_iter:
            # Surfel densification phase: steps 0 - 10k
            if step >= self.refine_stop_iter:
                return  # Surfel refinement has stopped
            target_name = "surfels"
            is_surfel_phase = True
        elif step > self.gaussian_spawn_iter:
            # Gaussian densification phase: steps 20k+
            if step >= self.refine_stop_iter_gaussian:
                return  # Gaussian refinement has stopped
            target_name = "gaussians"
            is_surfel_phase = False
        else:
            # Middle phase (10k-20k): no densification, just refining
            # existing surfel geometry with frozen opacity.
            return

        if not is_surfel_phase and model.gaussian.means.shape[0] == 0:
            return

        params = (
            model.get_surfel_param_dict() if is_surfel_phase else model.get_gaussian_param_dict()
        )
        optimizers = model.get_densification_optimizers(step)
        state = model.strategy_state[target_name]
        info = model.info.get(target_name)

        if info is None or optimizers is None:
            return
        self._update_state(params, state, info)

        mutated = False

        if step >= self.refine_start_iter and step % self.refine_every == 0:
            num_duplicates, n_split = self._grow_gs(params, optimizers, state, step, is_surfel_phase)
            target_type = "Surfels" if is_surfel_phase else "Gaussians"
            if self.verbose or is_surfel_phase:  # Always log surfel densification
                print(
                    f"Step {step}: {num_duplicates} {target_type} duplicated, {n_split} {target_type} split. "
                    f"Now having {len(params['means'])} {target_type}."
                )
            num_prune = self._prune_gs(params, optimizers, state, step, is_surfel_phase)
            if self.verbose or is_surfel_phase:  # Always log surfel pruning
                print(
                    f"Step {step}: {num_prune} {target_type} pruned. Now having {len(params['means'])} {target_type}."
                )

            state["grad2d"].zero_()
            state["count"].zero_()
            state["radii"].zero_()
            torch.cuda.empty_cache()
            mutated = True

        # if step % self.reset_every == 0 and step > 0:
        #     reset_opa(params, optimizers, state, value=self.prune_opa * 2.0)

        if mutated:
            target_obj = model.surfel if is_surfel_phase else model.gaussian
            prefix = "surfel_" if is_surfel_phase else "gaussian_"
            target_obj.means = params["means"]
            model.optimizers[prefix + "means"].param_groups[0]["params"] = [target_obj.means]
            target_obj.quats = params["quats"]
            model.optimizers[prefix + "quats"].param_groups[0]["params"] = [target_obj.quats]
            target_obj.scales = params["scales"]
            model.optimizers[prefix + "scales"].param_groups[0]["params"] = [target_obj.scales]
            target_obj.opacities = params["opacities"]
            model.optimizers[prefix + "opacities"].param_groups[0]["params"] = [
                target_obj.opacities
            ]
            target_obj.features_dc = params["features_dc"]
            model.optimizers[prefix + "features_dc"].param_groups[0]["params"] = [
                target_obj.features_dc
            ]
            target_obj.features_rest = params["features_rest"]
            model.optimizers[prefix + "features_rest"].param_groups[0]["params"] = [
                target_obj.features_rest
            ]
            
            # Clean up optimizer states to remove any stale entries
            self._clean_optimizer_states(model)
            
            # FIX: Sync model.surfel_radii_cache with the state's cache after
            # densification changes the surfel count. The gsplat ops (split,
            # duplicate, remove) update tensors in the state dict, but
            # model.surfel_radii_cache is a separate registered buffer that
            # was never resized. This caused an IndexError at step 10k when
            # execute_discard_phase tried to mask a stale-sized cache
            # (e.g. [4817]) with the current surfel opacity mask (e.g. [4925]).
            if is_surfel_phase:
                new_count = len(params["means"])
                if "surfel_radii_cache" in state and state["surfel_radii_cache"] is not None:
                    model.surfel_radii_cache = state["surfel_radii_cache"].clone()
                else:
                    model.surfel_radii_cache = torch.zeros(
                        new_count, device=model.device
                    )

    @torch.no_grad()
    def _grow_gs(
        self,
        params: dict[str, Parameter],
        optimizers: dict[str, torch.optim.Optimizer],
        state: dict[str, Any],
        step: int,
        is_surfel: bool = False,
    ):
        count = state["count"]
        grads = state["grad2d"] / count.clamp_min(1)
        is_grad_high = grads > self.grow_grad2d
        
        # BUG 10 FIX: For 2DGS (surfels), the 3rd scale (z-axis) is ignored during
        # projection and receives 0 gradient. It never shrinks during optimization.
        # If we include it in the max() check, it will falsely label surfels as "not small"
        # and prevent duplication, forcing splits instead.
        scales_to_check = params["scales"][:, :2] if is_surfel else params["scales"]
        is_small = (
            torch.exp(scales_to_check).max(dim=-1).values
            <= self.grow_scale3d * state["scene_scale"]
        )
        is_duplicate = is_grad_high & is_small
        num_duplicates = is_duplicate.sum().item()

        # is_large = ~is_small
        is_split = is_grad_high & ~is_small
        
        # BUG 12 FIX: Re-enable splitting based on 2D radius. When this was disabled,
        # primitives in flat, low-gradient areas (like chair backrests) would simply 
        # grow until they hit prune_scale3d, at which point they were deleted. 
        # This created unrecoverable holes. Splitting them prevents culling.
        is_split |= state["radii"] > self.grow_scale2d
        n_split = is_split.sum().item()

        if num_duplicates > 0:
            duplicate(params, optimizers, state, is_duplicate)

        is_split = torch.cat(
            [is_split, torch.zeros(num_duplicates, device=grads.device, dtype=torch.bool)]
        )
        if n_split > 0:
            split(
                params=params,
                optimizers=optimizers,
                state=state,
                mask=is_split,
                revised_opacity=False,
            )

        return num_duplicates, n_split

    @torch.no_grad()
    def _prune_gs(
        self,
        params: dict[str, Parameter],
        optimizers: dict[str, torch.optim.Optimizer],
        state: dict[str, Any],
        step: int,
        is_surfel: bool = False,
    ):
        is_prune = torch.sigmoid(params["opacities"].flatten()) < self.prune_opa
        if step > self.reset_every:
            # BUG 10 FIX: For 2DGS (surfels), the 3rd scale (z-axis) is ignored during
            # projection and receives 0 gradient. It never shrinks during optimization.
            # If we include it in the max() check, it will falsely trigger the "too big"
            # threshold and aggressively prune almost all surfels in the scene!
            scales_to_check = params["scales"][:, :2] if is_surfel else params["scales"]
            is_too_big = (
                torch.exp(scales_to_check).max(dim=-1).values
                > self.prune_scale3d * state["scene_scale"]
            )
            is_prune = is_prune | is_too_big
        num_prune = is_prune.sum().item()
        if num_prune > 0:
            remove(params=params, optimizers=optimizers, state=state, mask=is_prune)

        return num_prune

    def execute_discard_phase(self, model, keep_mask: torch.Tensor):
        """
        Callback function to be executed at the 10k iteration to discard the points based
        on the keep_mask.
        """

        def param_fn(name: str, p: torch.Tensor) -> torch.Tensor:
            return Parameter(p[keep_mask], requires_grad=p.requires_grad)

        def optimizer_fn(key: str, v: torch.Tensor) -> torch.Tensor:
            return v[keep_mask]

        params = model.get_surfel_param_dict()
        optimizers = model.get_densification_optimizers(self.surfel_density_stop_iter)
        _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)
        
        # Update model attributes and optimizer param groups
        model.surfel.means = params["means"]
        model.optimizers["surfel_means"].param_groups[0]["params"] = [model.surfel.means]
        model.surfel.quats = params["quats"]
        model.optimizers["surfel_quats"].param_groups[0]["params"] = [model.surfel.quats]
        model.surfel.scales = params["scales"]
        model.optimizers["surfel_scales"].param_groups[0]["params"] = [model.surfel.scales]
        model.surfel.opacities = params["opacities"]
        model.optimizers["surfel_opacities"].param_groups[0]["params"] = [model.surfel.opacities]
        model.surfel.features_dc = params["features_dc"]
        model.optimizers["surfel_features_dc"].param_groups[0]["params"] = [
            model.surfel.features_dc
        ]
        model.surfel.features_rest = params["features_rest"]
        model.optimizers["surfel_features_rest"].param_groups[0]["params"] = [
            model.surfel.features_rest
        ]
        
        # Clean up optimizer states to remove any stale entries
        self._clean_optimizer_states(model)
        
        # Prune the unnormalized radii cache to match the new param shape
        if hasattr(model, "surfel_radii_cache") and model.surfel_radii_cache is not None and model.surfel_radii_cache.shape[0] > 0:
            model.surfel_radii_cache = model.surfel_radii_cache[keep_mask]
        state = model.strategy_state["surfels"]
        if "surfel_radii_cache" in state and state["surfel_radii_cache"] is not None:
            state["surfel_radii_cache"] = state["surfel_radii_cache"][keep_mask]

        print(
            f"Discarded surfels based on the keep_mask at iteration {model.step}. \
            Now having {model.surfel.means.shape[0]} surfels."
        )

    def execute_visibility_prune_phase(self, model):
        """
        Callback function to be executed at the 15k iteration to prune the points based on
        visibility. Implements the dynamic culling algorithm from the GES paper:
        - Compute pixel coverage for each surfel (approximated using 2D radius and opacity)
        - Coverage is π * radius² * opacity
        - Prune surfels with coverage < n_threshold
        """
        # Select threshold based on scene type
        n_threshold = self.surfel_visibility_threshold_real if self.use_real_scene else self.surfel_visibility_threshold_synthetic
        
        # Use the unnormalized pixel-unit radii cache directly from the model,
        # which avoids the strategy state's normalization and periodic zeroing.
        max_2d_radius = model.surfel_radii_cache.detach()
        
        # BUG 3 FIX: surfel_radii_cache can have shape [C, N, 2] or [N, 2] or [N]
        # from 2DGS rasterization. Reduce to [N] scalar per surfel for the
        # area computation. For [C, N, 2], max over cameras (dim 0) then max
        # over the 2 axis radii (dim -1). For [N, 2], just max over axis.
        if max_2d_radius.dim() == 3:
            # [C, N, 2] -> max over cameras -> [N, 2] -> max over axes -> [N]
            max_2d_radius = max_2d_radius.max(dim=0).values.max(dim=-1).values
        elif max_2d_radius.dim() == 2:
            # [N, 2] -> max over axes -> [N]
            max_2d_radius = max_2d_radius.max(dim=-1).values
        # else: already [N], no change needed
        
        opacities = torch.sigmoid(model.surfel.opacities.detach()).squeeze()
        
        # Dynamic culling: approximate pixel coverage for each surfel
        # Coverage approximation: π * max_2d_radius² * opacity
        approx_cover = (3.14159 * max_2d_radius**2) * opacities
        visibility_mask = approx_cover > n_threshold
        num_pruned = torch.sum(~visibility_mask).item()
        print(f"Pruning {num_pruned} surfels based on visibility at iteration {model.step}. "
              f"(threshold: {n_threshold:.1f}, scene: {'real' if self.use_real_scene else 'synthetic'})")
        if num_pruned == 0:
            print("No surfels pruned based on visibility.")
            return

        def param_fn(name: str, p: torch.Tensor) -> torch.Tensor:
            return Parameter(p[visibility_mask], requires_grad=p.requires_grad)

        def optimizer_fn(key: str, v: torch.Tensor) -> torch.Tensor:
            return v[visibility_mask]

        params = model.get_surfel_param_dict()
        optimizers = model.get_densification_optimizers(self.surfel_prune_iter)
        _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)
        
        # Update model attributes and optimizer param groups
        model.surfel.means = params["means"]
        model.optimizers["surfel_means"].param_groups[0]["params"] = [model.surfel.means]
        model.surfel.quats = params["quats"]
        model.optimizers["surfel_quats"].param_groups[0]["params"] = [model.surfel.quats]
        model.surfel.scales = params["scales"]
        model.optimizers["surfel_scales"].param_groups[0]["params"] = [model.surfel.scales]
        model.surfel.opacities = params["opacities"]
        model.optimizers["surfel_opacities"].param_groups[0]["params"] = [model.surfel.opacities]
        model.surfel.features_dc = params["features_dc"]
        model.optimizers["surfel_features_dc"].param_groups[0]["params"] = [
            model.surfel.features_dc
        ]
        model.surfel.features_rest = params["features_rest"]
        model.optimizers["surfel_features_rest"].param_groups[0]["params"] = [
            model.surfel.features_rest
        ]
        
        # Clean up optimizer states to remove any stale entries
        self._clean_optimizer_states(model)
        
        # Prune the unnormalized radii cache to match the new param shape
        if hasattr(model, "surfel_radii_cache") and model.surfel_radii_cache is not None and model.surfel_radii_cache.shape[0] > 0:
            model.surfel_radii_cache = model.surfel_radii_cache[visibility_mask]
        state = model.strategy_state["surfels"]
        if "surfel_radii_cache" in state and state["surfel_radii_cache"] is not None:
            state["surfel_radii_cache"] = state["surfel_radii_cache"][visibility_mask]

        print(
            f"Pruned surfels based on visibility at iteration {model.step}. Now having \
            {model.surfel.means.shape[0]} surfels."
        )

    def clamp_surfel_opacity(self, model, min_opacity: float):
        """Callback function to be executed when the surfel opacity needs to be clamped."""
        target_prob = min(min_opacity / 255.0, 0.9999)
        target_logit = torch.logit(torch.tensor(target_prob, device=model.device))
        model.surfel.opacities.data = torch.clamp_min(model.surfel.opacities.data, target_logit)

    def freeze_surfel_opacity(self, model):
        """Freeze surfel opacity from further optimization (paper: 'keep w_i from optimization' after 10K)."""
        model.surfel.opacities.requires_grad_(False)
        if model.surfel.opacities.grad is not None:
            model.surfel.opacities.grad = None

    def freeze_surfel_geometry(self, model):
        """Callback function to be executed at the 20k iteration to freeze the surfel geometry."""
        for attr in ["means", "quats", "scales", "opacities"]:
            getattr(model.surfel, attr).requires_grad_(False)
            getattr(model.surfel, attr).grad = None

    def spawn_gaussians_from_saved_seeds(self, model, saved_gaussian_seeds: torch.Tensor):
        """
        Callback function to be executed at the 20k iteration to spawn gaussians from the
        saved seeds.
        """
        num_new_gaussians = saved_gaussian_seeds.shape[0]
        
        # Skip if no seeds to spawn from
        if num_new_gaussians == 0:
            print("No Gaussian seeds to spawn from (all surfels were kept). Skipping Gaussian spawning.")
            return
        
        device = model.device
        
        # Find closest surfel for each seed to inherit features (colors)
        # For now, initialize with surfel SH features at similar positions
        # Since we don't have exact position mapping, use mean surfel features as initialization
        surfel_features_dc_mean = model.surfel.features_dc.detach().mean(dim=0, keepdim=True)
        surfel_features_rest_mean = model.surfel.features_rest.detach().mean(dim=0, keepdim=True)

        # BUG 13 FIX: Initialize Gaussian scales SAFELY below the pruning threshold.
        # Previously it was `torch.ones * -1.0`, which evaluates to ~0.367.
        # Since 0.367 > prune_scale3d (0.1), EVERY newly spawned Gaussian was
        # instantly pruned at the very next iteration (step 20,100).
        scene_scale = model.strategy_state["gaussians"]["scene_scale"]
        import math
        safe_scale = math.log(max(1e-5, self.prune_scale3d * scene_scale * 0.5))

        new_data = {
            "means": saved_gaussian_seeds.clone(),
            "quats": random_quat_tensor(num_new_gaussians).to(device),
            "scales": torch.ones((num_new_gaussians, 3), device=device) * safe_scale,
            "opacities": torch.logit(
                0.5 * torch.ones((num_new_gaussians, 1), device=device)
            ),  # Initialize opacities to medium value (0.5 in probability space)
            "features_dc": surfel_features_dc_mean.expand(num_new_gaussians, -1).clone(),  # Inherit surfel colors
            "features_rest": surfel_features_rest_mean.expand(num_new_gaussians, -1, -1).clone(),  # Inherit surfel SH
        }

        def param_fn(name: str, p: torch.Tensor) -> torch.Tensor:
            return Parameter(new_data[name], requires_grad=p.requires_grad)

        def optimizer_fn(key: str, v: torch.Tensor) -> torch.Tensor:
            return torch.zeros((num_new_gaussians, *v.shape[1:]), device=device)

        params = model.get_gaussian_param_dict()
        optimizers = {
            "means": model.optimizers["gaussian_means"],
            "quats": model.optimizers["gaussian_quats"],
            "scales": model.optimizers["gaussian_scales"],
            "opacities": model.optimizers["gaussian_opacities"],
            "features_dc": model.optimizers["gaussian_features_dc"],
            "features_rest": model.optimizers["gaussian_features_rest"],
        }
        _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)
        
        # Update model attributes and optimizer param groups
        model.gaussian.means = params["means"]
        model.optimizers["gaussian_means"].param_groups[0]["params"] = [model.gaussian.means]
        model.gaussian.quats = params["quats"]
        model.optimizers["gaussian_quats"].param_groups[0]["params"] = [model.gaussian.quats]
        model.gaussian.scales = params["scales"]
        model.optimizers["gaussian_scales"].param_groups[0]["params"] = [model.gaussian.scales]
        model.gaussian.opacities = params["opacities"]
        model.optimizers["gaussian_opacities"].param_groups[0]["params"] = [
            model.gaussian.opacities
        ]
        model.gaussian.features_dc = params["features_dc"]
        model.optimizers["gaussian_features_dc"].param_groups[0]["params"] = [
            model.gaussian.features_dc
        ]
        model.gaussian.features_rest = params["features_rest"]
        model.optimizers["gaussian_features_rest"].param_groups[0]["params"] = [
            model.gaussian.features_rest
        ]
        
        # Clean up optimizer states to remove any stale entries
        self._clean_optimizer_states(model)
        
        print(
            f"Spawned {num_new_gaussians} new gaussians from saved seeds at iteration {model.step}."
        )
