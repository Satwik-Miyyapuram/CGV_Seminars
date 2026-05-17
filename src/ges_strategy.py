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
    """

    prune_opa: float = 0.005
    grow_grad2d: float = 0.0002
    grow_scale3d: float = 0.01
    grow_scale2d: float = 0.05
    prune_scale3d: float = 0.1
    prune_scale2d: float = 0.15
    refine_start_iter: int = 500
    refine_stop_iter: int = 15_000
    reset_every: int = 3000
    refine_every: int = 100
    verbose: bool = False

    surfel_density_stop_iter: int = 10000
    surfel_prune_iter: int = 15000
    gaussian_spawn_iter: int = 20000

    def initialize_state(self, scene_scale: float = 1.0) -> dict[str, Any]:
        """Initialize the strategy state."""
        state = {
            "surfels": {
                "radii": None,  # This will be updated during training with the actual
                # radii of the surfels
                "grad2d": None,
                "count": None,
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

    def step_pre_backward(self, model: GESModel, step: int):
        """Operations to be performed before the `loss.backward()` call."""
        if step <= self.surfel_density_stop_iter:
            surfel_info = model.info.get("surfels")
            if surfel_info is not None and "means2d" in surfel_info:
                surfel_info["means2d"].retain_grad()
        elif step > self.gaussian_spawn_iter:
            gaussians_info = model.info.get("gaussians")
            if gaussians_info is not None and "means2d" in gaussians_info:
                gaussians_info["means2d"].retain_grad()

    def _update_state(self, params: dict[str, Parameter], state: dict[str, Any], info: dict):
        """Helper function to update the strategy state."""
        grads = info["means2d"].grad.clone()
        grads[..., 0] *= info["width"] / 2.0 * info["n_cameras"]
        grads[..., 1] *= info["height"] / 2.0 * info["n_cameras"]

        n_items = len(list(params.values())[0])

        if state["grad2d"] is None:
            state["grad2d"] = torch.zeros(n_items, device=grads.device)
        if state["count"] is None:
            state["count"] = torch.zeros(n_items, device=grads.device)
        if state["radii"] is None:
            state["radii"] = torch.zeros(n_items, device=grads.device)

        sel = (info["radii"] > 0.0).all(dim=-1)  # [C, N]
        gs_ids = torch.where(sel)[1]  # [nnz]
        grads = grads[sel]  # [nnz, 2]
        radii = info["radii"][sel].max(dim=-1).values  # [nnz]

        state["grad2d"].index_add_(0, gs_ids, grads.norm(dim=-1))
        state["count"].index_add_(0, gs_ids, torch.ones_like(gs_ids, dtype=torch.float32))

        state["radii"][gs_ids] = torch.maximum(
            state["radii"][gs_ids],
            # normalize radii to [0, 1] screen space
            radii / float(max(info["width"], info["height"])),
        )

    def step_post_backward(self, model: GESModel, step: int):
        """Operations to be performed after the `loss.backward()` call."""
        if step >= self.refine_stop_iter:
            return

        # Determine what are getting denisified
        if step <= self.surfel_density_stop_iter:
            target_name = "surfels"
            is_surfel_phase = True
        elif step > self.gaussian_spawn_iter:
            target_name = "gaussians"
            is_surfel_phase = False
        else:
            # we are in the middle phase where we are not doing any densification, just refining
            # the existing points.
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
            num_duplicates, n_split = self._grow_gs(params, optimizers, state, step)
            if self.verbose:
                print(
                    f"Step {step}: {num_duplicates} GSs duplicated, {n_split} GSs split. "
                    f"Now having {len(params['means'])} GSs."
                )
            num_prune = self._prune_gs(params, optimizers, state, step)
            if self.verbose:
                print(
                    f"Step {step}: {num_prune} GSs pruned. Now having {len(params['means'])} GSs."
                )

            state["grad2d"].zero_()
            state["count"].zero_()
            state["radii"].zero_()
            torch.cuda.empty_cache()
            mutated = True

        if step % self.reset_every == 0 and step > 0:
            reset_opa(params, optimizers, state, value=self.prune_opa * 2.0)

        if mutated:
            target_obj = model.surfel if is_surfel_phase else model.gaussian
            prefix = "surfel_" if is_surfel_phase else "gaussian_"
            target_obj.means = Parameter(params["means"])
            model.optimizers[prefix + "means"].param_groups[0]["params"] = [target_obj.means]
            target_obj.quats = Parameter(params["quats"])
            model.optimizers[prefix + "quats"].param_groups[0]["params"] = [target_obj.quats]
            target_obj.scales = Parameter(params["scales"])
            model.optimizers[prefix + "scales"].param_groups[0]["params"] = [target_obj.scales]
            target_obj.opacities = Parameter(params["opacities"])
            model.optimizers[prefix + "opacities"].param_groups[0]["params"] = [
                target_obj.opacities
            ]
            target_obj.features_dc = Parameter(params["features_dc"])
            model.optimizers[prefix + "features_dc"].param_groups[0]["params"] = [
                target_obj.features_dc
            ]
            target_obj.features_rest = Parameter(params["features_rest"])
            model.optimizers[prefix + "features_rest"].param_groups[0]["params"] = [
                target_obj.features_rest
            ]

    @torch.no_grad()
    def _grow_gs(
        self,
        params: dict[str, Parameter],
        optimizers: dict[str, torch.optim.Optimizer],
        state: dict[str, Any],
        step: int,
    ):
        count = state["count"]
        grads = state["grad2d"] / count.clamp_min(1)
        is_grad_high = grads > self.grow_grad2d
        is_small = (
            torch.exp(params["scales"]).max(dim=-1).values
            <= self.grow_scale3d * state["scene_scale"]
        )
        is_duplicate = is_grad_high & is_small
        num_duplicates = is_duplicate.sum().item()

        # is_large = ~is_small
        is_split = is_grad_high & ~is_small
        # is_split |= state["radii"] > self.grow_scale2d # not using for now, since the source says
        # it has bugs
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
    ):
        is_prune = torch.sigmoid(params["opacities"].flatten()) < self.prune_opa
        if step > self.reset_every:
            is_too_big = (
                torch.exp(params["scales"]).max(dim=-1).values
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
        model.surfel.means = Parameter(params["means"])
        model.optimizers["surfel_means"].param_groups[0]["params"] = [model.surfel.means]
        model.surfel.quats = Parameter(params["quats"])
        model.optimizers["surfel_quats"].param_groups[0]["params"] = [model.surfel.quats]
        model.surfel.scales = Parameter(params["scales"])
        model.optimizers["surfel_scales"].param_groups[0]["params"] = [model.surfel.scales]
        model.surfel.opacities = Parameter(params["opacities"])
        model.optimizers["surfel_opacities"].param_groups[0]["params"] = [model.surfel.opacities]
        model.surfel.features_dc = Parameter(params["features_dc"])
        model.optimizers["surfel_features_dc"].param_groups[0]["params"] = [
            model.surfel.features_dc
        ]
        model.surfel.features_rest = Parameter(params["features_rest"])
        model.optimizers["surfel_features_rest"].param_groups[0]["params"] = [
            model.surfel.features_rest
        ]
        print(
            f"Discarded surfels based on the keep_mask at iteration {model.step}. \
            Now having {model.surfel.means.shape[0]} surfels."
        )

    def execute_visibility_prune_phase(self, model):
        """
        Callback function to be executed at the 15k iteration to prune the points based on
        visibility.
        """
        # we will  approximate the visibility since we cannot exactly follow the paper's approach as
        # gsplat uses alpha blending instead of a z buffer for rendering.
        # we approx the cover using radii and opacity,
        n_threshold = 16.0
        max_2d_radius = model.strategy_state["surfels"]["radii"].detach()
        opacities = torch.sigmoid(model.surfel.opacities.detach()).squeeze()
        approx_cover = (3.14159 * max_2d_radius**2) * opacities
        visibility_mask = approx_cover > n_threshold
        num_pruned = torch.sum(~visibility_mask).item()
        print(f"Pruning {num_pruned} surfels based on visibility at iteration {model.step}.")
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
        model.surfel.means = Parameter(params["means"])
        model.optimizers["surfel_means"].param_groups[0]["params"] = [model.surfel.means]
        model.surfel.quats = Parameter(params["quats"])
        model.optimizers["surfel_quats"].param_groups[0]["params"] = [model.surfel.quats]
        model.surfel.scales = Parameter(params["scales"])
        model.optimizers["surfel_scales"].param_groups[0]["params"] = [model.surfel.scales]
        model.surfel.opacities = Parameter(params["opacities"])
        model.optimizers["surfel_opacities"].param_groups[0]["params"] = [model.surfel.opacities]
        model.surfel.features_dc = Parameter(params["features_dc"])
        model.optimizers["surfel_features_dc"].param_groups[0]["params"] = [
            model.surfel.features_dc
        ]
        model.surfel.features_rest = Parameter(params["features_rest"])
        model.optimizers["surfel_features_rest"].param_groups[0]["params"] = [
            model.surfel.features_rest
        ]
        print(
            f"Pruned surfels based on visibility at iteration {model.step}. Now having \
            {model.surfel.means.shape[0]} surfels."
        )

    def clamp_surfel_opacity(self, model, min_opacity: float):
        """Callback function to be executed when the surfel opacity needs to be clamped."""
        target_prob = min(min_opacity / 255.0, 0.9999)
        target_logit = torch.logit(torch.tensor(target_prob, device=model.device))
        model.surfel.opacities.data = torch.clamp_min(model.surfel.opacities.data, target_logit)

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
        device = model.device

        new_data = {
            "means": saved_gaussian_seeds.clone(),
            "quats": random_quat_tensor(num_new_gaussians).to(device),
            "scales": torch.ones((num_new_gaussians, 3), device=device)
            * -2.0,  # Initialize scales to a small value (log scale)
            "opacities": torch.logit(
                0.1 * torch.ones((num_new_gaussians, 1), device=device)
            ),  # Initialize opacities to a low value
            "features_dc": torch.zeros(
                (num_new_gaussians, 3), device=device
            ),  # Initialize DC features to zero
            "features_rest": torch.zeros(
                (num_new_gaussians, num_sh_bases(model.config.sh_degree) - 1, 3), device=device
            ),  # Initialize SH features to zero
        }

        def param_fn(name: str, p: torch.Tensor) -> torch.Tensor:
            return Parameter(new_data[name], requires_grad=p.requires_grad)

        def optimizer_fn(key: str, v: torch.Tensor) -> torch.Tensor:
            return torch.zeros((num_new_gaussians, *v.shape[1:]), device=device)

        params = model.get_gaussian_param_dict()
        optimizers = model.get_densification_optimizers(model.step)
        _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)
        model.gaussian.means = Parameter(params["means"])
        model.optimizers["gaussian_means"].param_groups[0]["params"] = [model.gaussian.means]
        model.gaussian.quats = Parameter(params["quats"])
        model.optimizers["gaussian_quats"].param_groups[0]["params"] = [model.gaussian.quats]
        model.gaussian.scales = Parameter(params["scales"])
        model.optimizers["gaussian_scales"].param_groups[0]["params"] = [model.gaussian.scales]
        model.gaussian.opacities = Parameter(params["opacities"])
        model.optimizers["gaussian_opacities"].param_groups[0]["params"] = [
            model.gaussian.opacities
        ]
        model.gaussian.features_dc = Parameter(params["features_dc"])
        model.optimizers["gaussian_features_dc"].param_groups[0]["params"] = [
            model.gaussian.features_dc
        ]
        model.gaussian.features_rest = Parameter(params["features_rest"])
        model.optimizers["gaussian_features_rest"].param_groups[0]["params"] = [
            model.gaussian.features_rest
        ]
        print(
            f"Spawned {num_new_gaussians} new gaussians from saved seeds at iteration {model.step}."
        )
