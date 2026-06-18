from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from gsplat.strategy.base import Strategy
from gsplat.strategy.ops import _update_param_with_optimizer, duplicate, remove, reset_opa, split
from nerfstudio.utils.math import k_nearest_sklearn
from nerfstudio.utils.spherical_harmonics import RGB2SH, SH2RGB, num_sh_bases
from torch.nn import Parameter

from training_schedule import (
    GAUSSIAN_SPAWN_STEP,
    SURFEL_DENSIFICATION_STOP,
    SURFEL_DISCARD_ITER,
    VISIBILITY_PRUNE_STEP,
)

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

    prune_opa: float = 0.02 / 255.0
    grow_grad2d: float = 0.0002
    grow_scale3d: float = 0.01
    grow_scale2d: float = 0.05
    prune_scale3d: float = 0.1
    prune_scale2d: float = 0.15
    refine_start_iter: int = 500
    refine_stop_iter: int = VISIBILITY_PRUNE_STEP
    refine_stop_iter_gaussian: int = 30000
    reset_every: int = 3000
    refine_every: int = 100
    verbose: bool = False
    absgrad: bool = True

    surfel_density_stop_iter: int = SURFEL_DENSIFICATION_STOP
    surfel_discard_iter: int = SURFEL_DISCARD_ITER
    surfel_prune_iter: int = VISIBILITY_PRUNE_STEP
    gaussian_spawn_iter: int = GAUSSIAN_SPAWN_STEP

    surfel_visibility_threshold_real: float = 16.0
    surfel_visibility_threshold_synthetic: float = 4.0
    use_real_scene: bool = True

    def initialize_state(self, scene_scale: float = 1.0) -> dict[str, Any]:
        """Initialize the strategy state."""
        state = {
            "surfels": {
                "radii": None,
                "grad2d": None,
                "count": None,
                "surfel_radii_cache": None,
                "scene_scale": scene_scale,
            },
            "gaussians": {
                "radii": None,
                "grad2d": None,
                "count": None,
                "scene_scale": scene_scale,
            },
        }
        return state

    def _clean_optimizer_states(self, model: GESModel):
        """Clean up all optimizer states to remove any stale parameter entries."""
        for opt_name, optimizer in model.optimizers.items():
            valid_ids = set()
            for param_group in optimizer.param_groups:
                for param in param_group.get("params", []):
                    valid_ids.add(id(param))

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

        The gsplat DefaultStrategy uses:
            if self.absgrad:
                grads = info[key].absgrad.clone()
            else:
                grads = info[key].grad.clone()
        We follow the same pattern here.
        """
        means2d = info["means2d"]
        if self.absgrad:
            if hasattr(means2d, "absgrad") and means2d.absgrad is not None:
                grads = means2d.absgrad.clone()
            else:
                if means2d.grad is not None:
                    grads = means2d.grad.abs().clone()
                else:
                    return
        else:
            if means2d.grad is not None:
                grads = means2d.grad.clone()
            else:
                return

        grads[..., 0] *= info["width"] / 2.0 * info["n_cameras"]
        grads[..., 1] *= info["height"] / 2.0 * info["n_cameras"]

        n_items = len(list(params.values())[0])

        if state["grad2d"] is None:
            state["grad2d"] = torch.zeros(n_items, device=grads.device, dtype=torch.float32)
        if state["count"] is None:
            state["count"] = torch.zeros(n_items, device=grads.device, dtype=torch.float32)
        if state["radii"] is None:
            state["radii"] = torch.zeros(n_items, device=grads.device, dtype=torch.float32)
        for key in ["grad2d", "count", "radii"]:
            if state[key] is None or state[key].shape[0] < n_items:
                old_tensor = state[key]
                state[key] = torch.zeros(n_items, device=grads.device, dtype=torch.float32)
                if old_tensor is not None:
                    state[key][: old_tensor.shape[0]] = old_tensor
        if info["radii"].dim() == 3:
            sel = (info["radii"] > 0.0).all(dim=-1)
            radii = info["radii"][sel].max(dim=-1).values
        else:
            sel = info["radii"] > 0.0
            radii = info["radii"][sel]

        gs_ids = torch.where(sel)[1]
        grads = grads[sel]

        state["grad2d"].index_add_(0, gs_ids, grads.norm(dim=-1))
        state["count"].index_add_(0, gs_ids, torch.ones_like(gs_ids, dtype=torch.float32))

        if state["radii"].dtype != torch.float32:
            state["radii"] = state["radii"].to(dtype=torch.float32, device=grads.device)

        state["radii"][gs_ids] = torch.maximum(
            state["radii"][gs_ids],
            (radii / float(max(info["width"], info["height"]))).to(
                device=grads.device, dtype=torch.float32
            ),
        )

    def step_post_backward(self, model: GESModel, step: int):
        """Operations to be performed after the `loss.backward()` call."""
        if step <= self.surfel_density_stop_iter:
            if step >= self.refine_stop_iter:
                return
            target_name = "surfels"
            is_surfel_phase = True
        elif step > self.gaussian_spawn_iter:
            if step >= self.refine_stop_iter_gaussian:
                return
            target_name = "gaussians"
            is_surfel_phase = False
        else:
            return

        if not is_surfel_phase and model.gaussian.means.shape[0] == 0:
            return

        params = (
            model.get_surfel_param_dict() if is_surfel_phase else model.get_gaussian_param_dict()
        )
        optimizers = model.get_densification_optimizers(step)
        state = model.strategy_state[target_name]
        info = model.info.get(target_name)

        if not is_surfel_phase and hasattr(model, "gaussian_max_contribution_score"):
            if model.gaussian_max_contribution_score.shape[0] == model.gaussian.means.shape[0]:
                state["gaussian_max_contribution_score"] = model.gaussian_max_contribution_score

        if info is None or optimizers is None:
            return
        self._update_state(params, state, info)

        mutated = False
        if step >= self.refine_start_iter and step % self.refine_every == 0:
            if is_surfel_phase:
                original_num = len(params["means"])
                max_num = getattr(model.config, "max_num_surfels", -1)
                num_duplicates, n_split = self._grow_gs(
                    params, optimizers, state, step, is_surfel_phase, max_num
                )
                target_type = "Surfels"
                if self.verbose or is_surfel_phase:
                    print(
                        f"Step {step}: {num_duplicates} {target_type} duplicated, {n_split} {target_type} split. "
                        f"Now having {len(params['means'])} {target_type}."
                    )
                num_prune = self._prune_gs(
                    params, optimizers, state, step, is_surfel_phase, original_num=original_num
                )
                if self.verbose or is_surfel_phase:
                    print(
                        f"Step {step}: {num_prune} {target_type} pruned. "
                        f"Now having {len(params['means'])} {target_type}."
                    )
            else:
                pass
            state["grad2d"].zero_()
            state["count"].zero_()
            state["radii"].zero_()
            torch.cuda.empty_cache()
            mutated = True

        is_reset_step = step % self.reset_every == 0
        is_surfel_solidification_buffer = (
            is_surfel_phase and step >= self.surfel_density_stop_iter - self.reset_every
        )

        if is_reset_step and not is_surfel_solidification_buffer:
            if is_surfel_phase:
                reset_opa(params=params, optimizers=optimizers, state=state, value=0.1 / 255.0)
            else:
                if step <= self.gaussian_spawn_iter + self.reset_every:
                    reset_opa(params=params, optimizers=optimizers, state=state, value=0.01)
            mutated = True

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

            self._clean_optimizer_states(model)

            if is_surfel_phase:
                new_count = len(params["means"])
                if "surfel_radii_cache" in state and state["surfel_radii_cache"] is not None:
                    model.surfel_radii_cache = state["surfel_radii_cache"].clone()
                else:
                    model.surfel_radii_cache = torch.zeros(new_count, device=model.device)
            else:
                if "gaussian_max_contribution_score" in state:
                    model.gaussian_max_contribution_score = state.pop(
                        "gaussian_max_contribution_score"
                    )
        else:
            if not is_surfel_phase:
                state.pop("gaussian_max_contribution_score", None)

    @torch.no_grad()
    def _grow_gs(
        self,
        params: dict[str, Parameter],
        optimizers: dict[str, torch.optim.Optimizer],
        state: dict[str, Any],
        step: int,
        is_surfel: bool = False,
        max_num: int = -1,
    ):
        """Densify: clone small high-gradient primitives and split large high-gradient ones (surfels use 2D scale checks)."""
        count = state["count"]
        grads = state["grad2d"] / count.clamp_min(1)
        if step >= 500:
            with open("debug_strategy.txt", "a") as f:
                f.write(f"\n--- Step {step} grow ---\n")
                f.write(f"grow_grad2d: {self.grow_grad2d}\n")
                f.write(
                    f"grads: min={grads.min().item():.6f}, max={grads.max().item():.6f}, mean={grads.mean().item():.6f}\n"
                )
                f.write(f"scene_scale: {state['scene_scale']}\n")
        is_grad_high = grads > self.grow_grad2d

        original_num = params["means"].shape[0]
        if max_num > 0 and original_num >= max_num:
            is_grad_high = torch.zeros_like(is_grad_high)

        # Surfels are 2D (2DGS): their 3rd (z) scale is unused, so size checks use only x,y.
        scales_to_check = params["scales"][:, :2] if is_surfel else params["scales"]
        is_small = (
            torch.exp(scales_to_check).max(dim=-1).values
            <= self.grow_scale3d * state["scene_scale"]
        )
        is_duplicate = is_grad_high & is_small
        is_split = is_grad_high & ~is_small

        if max_num > 0:
            allowed_growth = max(0, max_num - original_num)
            if allowed_growth <= 0:
                is_duplicate.zero_()
                is_split.zero_()
            else:
                dup_indices = torch.where(is_duplicate)[0]
                if len(dup_indices) > allowed_growth:
                    is_duplicate[dup_indices[allowed_growth:]] = False
                    allowed_growth = 0
                else:
                    allowed_growth -= len(dup_indices)

                split_indices = torch.where(is_split)[0]
                if len(split_indices) > allowed_growth:
                    is_split[split_indices[allowed_growth:]] = False

        num_duplicates = is_duplicate.sum().item()
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
        original_num: int = -1,
    ):
        """Prune primitives that are too transparent or too large (surfels use 2D scale/radius checks)."""
        prune_opa_thresh = self.prune_opa if is_surfel else 0.01
        sig_opacities = torch.sigmoid(params["opacities"].flatten())
        is_prune = sig_opacities < prune_opa_thresh
        if step >= 500:
            with open("debug_strategy.txt", "a") as f:
                f.write(f"\n--- Step {step} prune ---\n")
                f.write(f"prune_opa_thresh: {prune_opa_thresh}\n")
                f.write(
                    f"opacities (sigmoid): min={sig_opacities.min().item():.6f}, max={sig_opacities.max().item():.6f}, mean={sig_opacities.mean().item():.6f}\n"
                )
        if step > self.reset_every:
            # Surfels are 2D (2DGS): their 3rd (z) scale is unused, so size checks use only x,y.
            scales_to_check = params["scales"][:, :2] if is_surfel else params["scales"]

            is_too_big = (
                torch.exp(scales_to_check).max(dim=-1).values
                > self.prune_scale3d * state["scene_scale"]
            )
            is_prune = is_prune | is_too_big

            if state["radii"] is not None and not is_surfel:
                is_too_big_2d = state["radii"] > self.prune_scale2d
                is_prune = is_prune | is_too_big_2d
        if original_num > 0:
            is_prune[original_num:] = False
        num_prune = is_prune.sum().item()
        if step >= 500:
            with open("debug_strategy.txt", "a") as f:
                f.write(f"num_prune: {num_prune} / {len(is_prune)}\n")
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

        self._clean_optimizer_states(model)

        if (
            hasattr(model, "surfel_radii_cache")
            and model.surfel_radii_cache is not None
            and model.surfel_radii_cache.shape[0] > 0
        ):
            model.surfel_radii_cache = model.surfel_radii_cache[keep_mask]
        state = model.strategy_state["surfels"]
        if "surfel_radii_cache" in state and state["surfel_radii_cache"] is not None:
            state["surfel_radii_cache"] = state["surfel_radii_cache"][keep_mask]
        for key in ["grad2d", "count", "radii"]:
            if state[key] is not None:
                state[key] = state[key][keep_mask]

        print(
            f"Discarded surfels based on the keep_mask at iteration {model.step}. \
            Now having {model.surfel.means.shape[0]} surfels."
        )

    def execute_visibility_prune_phase(self, model):
        """
        Surfel covering-score prune, run every 1k from 15k to 20k (densification already off).
        Implements the GES paper's post-15k 'tiny or invisible surfel' culling:
        - surfel_radii_cache holds the max-over-views ellipse area (rx * ry) per surfel.
        - The covering score is approximated as π * (rx * ry), weighted by opacity.
        - Surfels with score < n_threshold (n_thr) are pruned.
        The accumulator is reset afterwards so the next window measures coverage afresh.
        """
        n_threshold = (
            self.surfel_visibility_threshold_real
            if self.use_real_scene
            else self.surfel_visibility_threshold_synthetic
        )

        # surfel_radii_cache is the per-surfel ellipse area rx * ry (max over views), 1D.
        cover_area = model.surfel_radii_cache.detach()

        approx_cover = 3.14159 * cover_area
        visibility_mask = approx_cover > n_threshold
        num_pruned = torch.sum(~visibility_mask).item()
        print(
            f"Pruning {num_pruned} surfels based on visibility at iteration {model.step}. "
            f"(threshold: {n_threshold:.1f}, scene: {'real' if self.use_real_scene else 'synthetic'})"
        )
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

        self._clean_optimizer_states(model)

        # Keep the covering-area accumulator as a global max over all views (paper's n_i =
        # max_j{n_i,j}): just slice it to the surviving surfels, do not reset. Successive
        # prunes therefore re-check an ever-growing max rather than a fresh 1k-step window.
        state = model.strategy_state["surfels"]
        if (
            hasattr(model, "surfel_radii_cache")
            and model.surfel_radii_cache is not None
            and model.surfel_radii_cache.shape[0] > 0
        ):
            model.surfel_radii_cache = model.surfel_radii_cache[visibility_mask]
        if "surfel_radii_cache" in state and state["surfel_radii_cache"] is not None:
            state["surfel_radii_cache"] = state["surfel_radii_cache"][visibility_mask]
        for key in ["grad2d", "count", "radii"]:
            if state.get(key) is not None:
                state[key] = state[key][visibility_mask]

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

        if num_new_gaussians == 0:
            print(
                "No Gaussian seeds to spawn from (all surfels were kept). Skipping Gaussian spawning."
            )
            return

        device = model.device

        max_num_gaussians = getattr(model.config, "max_num_gaussians", -1)
        if max_num_gaussians > 0:
            current_num_gaussians = model.gaussian.means.shape[0]
            allowed_new_gaussians = max_num_gaussians - current_num_gaussians
            if allowed_new_gaussians <= 0:
                print(f"Cannot spawn Gaussians: limit of {max_num_gaussians} reached.")
                return
            if num_new_gaussians > allowed_new_gaussians:
                print(
                    f"Limiting spawned Gaussians from saved seeds to {allowed_new_gaussians} due to max_num_gaussians limit."
                )
                saved_gaussian_seeds = saved_gaussian_seeds[:allowed_new_gaussians]
                num_new_gaussians = allowed_new_gaussians

                if (
                    hasattr(model, "saved_gaussian_features_dc")
                    and model.saved_gaussian_features_dc.shape[0] > 0
                ):
                    model.saved_gaussian_features_dc = model.saved_gaussian_features_dc[
                        :allowed_new_gaussians
                    ]
                if (
                    hasattr(model, "saved_gaussian_features_rest")
                    and model.saved_gaussian_features_rest.shape[0] > 0
                ):
                    model.saved_gaussian_features_rest = model.saved_gaussian_features_rest[
                        :allowed_new_gaussians
                    ]
                if (
                    hasattr(model, "saved_gaussian_scales")
                    and model.saved_gaussian_scales.shape[0] > 0
                ):
                    model.saved_gaussian_scales = model.saved_gaussian_scales[
                        :allowed_new_gaussians
                    ]
                if (
                    hasattr(model, "saved_gaussian_quats")
                    and model.saved_gaussian_quats.shape[0] > 0
                ):
                    model.saved_gaussian_quats = model.saved_gaussian_quats[:allowed_new_gaussians]

        if (
            hasattr(model, "saved_gaussian_features_dc")
            and model.saved_gaussian_features_dc.shape[0] == num_new_gaussians
        ):
            features_dc = model.saved_gaussian_features_dc.clone()
        else:
            print(
                "Warning: saved_gaussian_features_dc not found or mismatched size. Falling back to mean surfel color."
            )
            surfel_features_dc_mean = model.surfel.features_dc.detach().mean(dim=0, keepdim=True)
            features_dc = surfel_features_dc_mean.expand(num_new_gaussians, -1).clone()

        if (
            hasattr(model, "saved_gaussian_features_rest")
            and model.saved_gaussian_features_rest.shape[0] == num_new_gaussians
        ):
            features_rest = model.saved_gaussian_features_rest.clone()
        else:
            print(
                "Warning: saved_gaussian_features_rest not found or mismatched size. Falling back to mean surfel SH."
            )
            surfel_features_rest_mean = model.surfel.features_rest.detach().mean(
                dim=0, keepdim=True
            )
            features_rest = surfel_features_rest_mean.expand(num_new_gaussians, -1, -1).clone()

        # Spawn with a z-scale safely below the 3D prune threshold so new Gaussians aren't
        # immediately pruned for being "too large".
        scene_scale = model.strategy_state["gaussians"]["scene_scale"]
        safe_scale = math.log(max(1e-5, self.prune_scale3d * scene_scale * 0.5))

        if (
            hasattr(model, "saved_gaussian_scales")
            and model.saved_gaussian_scales.shape[0] == num_new_gaussians
        ):
            scales = model.saved_gaussian_scales.clone()
            z_scale = torch.clamp(scales[:, 2], max=safe_scale)
            scales = scales.clone()
            scales[:, 2] = z_scale
        else:
            print(
                "Warning: saved_gaussian_scales not found or mismatched size. Falling back to global safe scale."
            )
            scales = torch.ones((num_new_gaussians, 3), device=device) * safe_scale

        if (
            hasattr(model, "saved_gaussian_quats")
            and model.saved_gaussian_quats.shape[0] == num_new_gaussians
        ):
            quats = model.saved_gaussian_quats.clone()
        else:
            print(
                "Warning: saved_gaussian_quats not found or mismatched size. Falling back to identity rotation."
            )
            quats = torch.zeros((num_new_gaussians, 4), device=device)
            quats[:, 0] = 1.0

        new_data = {
            "means": saved_gaussian_seeds.clone(),
            "quats": quats,
            "scales": scales,
            "opacities": torch.logit(0.1 * torch.ones((num_new_gaussians, 1), device=device)),
            "features_dc": features_dc,
            "features_rest": features_rest,
        }

        def param_fn(name: str, p: torch.Tensor) -> torch.Tensor:
            return Parameter(torch.cat([p, new_data[name]], dim=0), requires_grad=p.requires_grad)

        def optimizer_fn(key: str, v: torch.Tensor) -> torch.Tensor:
            return torch.cat(
                [v, torch.zeros((num_new_gaussians, *v.shape[1:]), device=device)], dim=0
            )

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

        self._clean_optimizer_states(model)

        if hasattr(model, "gaussian_max_contribution_score"):
            model.gaussian_max_contribution_score = torch.cat(
                [
                    model.gaussian_max_contribution_score,
                    torch.zeros(num_new_gaussians, device=device),
                ]
            )
            state = model.strategy_state["gaussians"]
            if (
                "gaussian_max_contribution_score" in state
                and state["gaussian_max_contribution_score"] is not None
            ):
                state["gaussian_max_contribution_score"] = (
                    model.gaussian_max_contribution_score.clone()
                )

        print(
            f"Spawned {num_new_gaussians} new gaussians from saved seeds at iteration {model.step}."
        )

    def spawn_gaussians_from_error_seeds(
        self, model, spawn_pts: torch.Tensor, spawn_cols: torch.Tensor
    ):
        """
        Spawns new 3D Gaussians at the specified error-based unprojected positions.
        """
        num_new_gaussians = spawn_pts.shape[0]
        if num_new_gaussians == 0:
            return

        device = model.device

        max_num_gaussians = getattr(model.config, "max_num_gaussians", -1)
        if max_num_gaussians > 0:
            current_num_gaussians = model.gaussian.means.shape[0]
            allowed_new_gaussians = max_num_gaussians - current_num_gaussians
            if allowed_new_gaussians <= 0:
                print(
                    f"Cannot spawn Gaussians from error seeds: limit of {max_num_gaussians} reached."
                )
                return
            if num_new_gaussians > allowed_new_gaussians:
                print(
                    f"Limiting spawned Gaussians from error seeds to {allowed_new_gaussians} due to max_num_gaussians limit."
                )
                spawn_pts = spawn_pts[:allowed_new_gaussians]
                spawn_cols = spawn_cols[:allowed_new_gaussians]
                num_new_gaussians = allowed_new_gaussians

        device = model.device
        dim_sh = num_sh_bases(model.config.sh_degree)

        if model.config.sh_degree > 0:
            features_dc = RGB2SH(spawn_cols)
        else:
            features_dc = torch.logit(spawn_cols, eps=1e-10)

        features_rest = torch.zeros((num_new_gaussians, dim_sh - 1, 3), device=device)

        try:
            distances, _ = k_nearest_sklearn(spawn_pts.detach(), 3)
            avg_dist = distances.mean(dim=-1, keepdim=True).clamp(min=1e-7)
            scales = torch.log(avg_dist.repeat(1, 3))
        except Exception as e:
            print(
                f"Warning: k_nearest_sklearn failed during error spawn: {e}. Falling back to scene scale."
            )
            scene_scale = model.strategy_state["gaussians"]["scene_scale"]
            safe_scale = math.log(max(1e-5, self.prune_scale3d * scene_scale * 0.5))
            scales = torch.ones((num_new_gaussians, 3), device=device) * safe_scale

        scene_scale = model.strategy_state["gaussians"]["scene_scale"]
        safe_scale = math.log(max(1e-5, self.prune_scale3d * scene_scale * 0.5))
        scales = torch.clamp(scales, max=safe_scale)

        quats = torch.zeros((num_new_gaussians, 4), device=device)
        quats[:, 0] = 1.0

        new_data = {
            "means": spawn_pts.clone().to(device),
            "quats": quats.to(device),
            "scales": scales.to(device),
            "opacities": torch.logit(0.1 * torch.ones((num_new_gaussians, 1), device=device)),
            "features_dc": features_dc.to(device),
            "features_rest": features_rest.to(device),
        }

        def param_fn(name: str, p: torch.Tensor) -> torch.Tensor:
            return Parameter(torch.cat([p, new_data[name]], dim=0), requires_grad=p.requires_grad)

        def optimizer_fn(key: str, v: torch.Tensor) -> torch.Tensor:
            return torch.cat(
                [v, torch.zeros((num_new_gaussians, *v.shape[1:]), device=device)], dim=0
            )

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

        self._clean_optimizer_states(model)

        if hasattr(model, "gaussian_max_contribution_score"):
            model.gaussian_max_contribution_score = torch.cat(
                [
                    model.gaussian_max_contribution_score,
                    torch.zeros(num_new_gaussians, device=device),
                ]
            )
            state = model.strategy_state["gaussians"]
            if (
                "gaussian_max_contribution_score" in state
                and state["gaussian_max_contribution_score"] is not None
            ):
                state["gaussian_max_contribution_score"] = (
                    model.gaussian_max_contribution_score.clone()
                )

        print(f"Spawned {num_new_gaussians} new Gaussians in high-error regions.")

    def execute_contribution_pruning(self, model, keep_mask: torch.Tensor):
        """
        Prunes 3D Gaussians based on the contribution score keep_mask.
        """
        device = model.device

        def param_fn(name: str, p: torch.Tensor) -> torch.Tensor:
            return Parameter(p[keep_mask], requires_grad=p.requires_grad)

        def optimizer_fn(key: str, v: torch.Tensor) -> torch.Tensor:
            return v[keep_mask]

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

        self._clean_optimizer_states(model)

        state = model.strategy_state["gaussians"]
        for key in ["grad2d", "count", "radii"]:
            if state[key] is not None:
                state[key] = state[key][keep_mask]

        if hasattr(model, "gaussian_max_contribution_score"):
            model.gaussian_max_contribution_score = model.gaussian_max_contribution_score[keep_mask]

        print(f"Pruned redundant Gaussians. Now having {model.gaussian.means.shape[0]} Gaussians.")

    def error_spawn_and_prune(self, model, step: int) -> None:
        """Post-20k joint optimization, run every iteration as a training callback (like the
        surfel densification callback). Each step it (1) accumulates error-map spawn seeds by
        unprojecting high-error pixels through the surfel depth, and (2) updates each Gaussian's
        running max contribution score. Every 1000 steps it prunes low-contribution Gaussians,
        spawns new ones from the accumulated seeds, and resets the score buffer.

        The render data it needs (outputs / gt_img / pred_img) is stashed on the model during
        get_loss_dict, since a training callback doesn't receive it directly.
        """
        if not (20000 < step <= 30000):
            return
        outputs = getattr(model, "_spawn_outputs", None)
        gt_img = getattr(model, "_spawn_gt_img", None)
        pred_img = getattr(model, "_spawn_pred_img", None)
        if outputs is None or gt_img is None or pred_img is None:
            return

        with torch.no_grad():
            device = model.device
            height, width = gt_img.shape[:2]

            error_map = torch.mean((gt_img.detach() - pred_img.detach()) ** 2, dim=-1)
            probs = error_map.flatten() / (error_map.sum() + 1e-8)

            N = 100
            sampled_indices = torch.multinomial(probs, num_samples=N, replacement=True)
            u = sampled_indices % width
            v = sampled_indices // width

            surfel_depth = outputs.get("surfel_depth")
            if surfel_depth is not None:
                z = surfel_depth.squeeze(0)[v, u, 0]
                valid_mask = z < 9.0

                if valid_mask.sum() > 0:
                    u_v = u[valid_mask]
                    v_v = v[valid_mask]
                    z_v = z[valid_mask]

                    intrinsic_mat = outputs["intrinsic_mat"]
                    fx, fy = intrinsic_mat[0, 0, 0], intrinsic_mat[0, 1, 1]
                    cx, cy = intrinsic_mat[0, 0, 2], intrinsic_mat[0, 1, 2]

                    x_cam = (u_v - cx) * z_v / fx
                    y_cam = -(v_v - cy) * z_v / fy
                    pos_cam = torch.stack([x_cam, y_cam, -z_v], dim=-1)

                    c2w = outputs["camera_to_world"]
                    R = c2w[0, :3, :3]
                    T = c2w[0, :3, 3]
                    pos_world = torch.matmul(pos_cam, R.T) + T

                    color = gt_img[v_v, u_v]

                    model.error_spawn_points.append(pos_world.cpu())
                    model.error_spawn_colors.append(color.cpu())

            gaussian_alpha = outputs.get("gaussian_alpha")
            surfel_alpha = outputs.get("surfel_alpha")
            x_screen = outputs.get("x_screen")
            y_screen = outputs.get("y_screen")

            if (
                model.gaussian.means.shape[0] > 0
                and gaussian_alpha is not None
                and surfel_alpha is not None
                and x_screen is not None
                and y_screen is not None
            ):
                num_gaussians = model.gaussian.means.shape[0]

                if model.gaussian_max_contribution_score.shape[0] != num_gaussians:
                    model.gaussian_max_contribution_score = torch.zeros(
                        num_gaussians, device=device
                    )

                c_rgb = torch.clamp(SH2RGB(model.gaussian.features_dc.detach()), 0.0, 1.0)
                c_i = c_rgb.max(dim=-1).values
                alpha_i = torch.sigmoid(model.gaussian.opacities.detach()).flatten()

                x_norm = (x_screen / width) * 2.0 - 1.0
                y_norm = (y_screen / height) * 2.0 - 1.0
                grid = torch.stack((x_norm, y_norm), dim=-1).unsqueeze(0).unsqueeze(0)

                surfel_alpha_map = surfel_alpha.permute(0, 3, 1, 2)
                gaussian_alpha_map = gaussian_alpha.permute(0, 3, 1, 2)

                sampled_W_S = torch.nn.functional.grid_sample(
                    surfel_alpha_map,
                    grid,
                    mode="nearest",
                    padding_mode="border",
                    align_corners=False,
                ).flatten()
                sampled_W_G = torch.nn.functional.grid_sample(
                    gaussian_alpha_map,
                    grid,
                    mode="nearest",
                    padding_mode="border",
                    align_corners=False,
                ).flatten()

                score = (c_i * alpha_i) / (sampled_W_S + sampled_W_G + 1e-5)

                model.gaussian_max_contribution_score = torch.maximum(
                    model.gaussian_max_contribution_score, score
                )

            if step > 20000 and step % 1000 == 0:
                print(
                    f"[Joint Optimization] Periodic step {step} unprojected spawning and contribution pruning..."
                )

                if model.gaussian.means.shape[0] > 0:
                    num_gaussians = model.gaussian.means.shape[0]
                    if model.gaussian_max_contribution_score.shape[0] == num_gaussians:
                        keep_mask = model.gaussian_max_contribution_score >= 0.02
                        num_prune = (~keep_mask).sum().item()
                        if num_prune > 0:
                            print(
                                f"[Contribution Pruning] Pruning {num_prune} Gaussians with contribution score < 0.02."
                            )
                            self.execute_contribution_pruning(model, keep_mask)

                if len(model.error_spawn_points) > 0:
                    spawn_pts = torch.cat(model.error_spawn_points, dim=0).to(device)
                    spawn_cols = torch.cat(model.error_spawn_colors, dim=0).to(device)

                    self.spawn_gaussians_from_error_seeds(model, spawn_pts, spawn_cols)

                    model.error_spawn_points = []
                    model.error_spawn_colors = []

                if model.gaussian.means.shape[0] > 0:
                    model.gaussian_max_contribution_score = torch.zeros(
                        model.gaussian.means.shape[0], device=device
                    )
                    state = model.strategy_state["gaussians"]
                    if "gaussian_max_contribution_score" in state:
                        state["gaussian_max_contribution_score"] = (
                            model.gaussian_max_contribution_score.clone()
                        )
