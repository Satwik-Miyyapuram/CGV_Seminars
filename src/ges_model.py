"""
ges_model.py
Contains the custom PyTorch Model defining the GES logic.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import torch

# Assuming gsplat is in external_code/gsplat and accessible
from gsplat.rendering import rasterization
from nerfstudio.cameras.camera_optimizers import CameraOptimizer
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.model_components.lib_bilagrid import color_correct
from nerfstudio.models.base_model import Model
from nerfstudio.models.splatfacto import SplatfactoModelConfig, get_viewmat, resize_image
from nerfstudio.utils.colors import get_color
from nerfstudio.utils.math import k_nearest_sklearn, random_quat_tensor
from nerfstudio.utils.spherical_harmonics import RGB2SH, SH2RGB, num_sh_bases
from pytorch_msssim import SSIM

# from surfel_rasterizer_extension.render import rasterize_surfels_to_pixels
from render import rasterization_surfel
from torch.nn import Parameter
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from ges_strategy import GESStrategy
from training_schedule import (
    CLAMP_18K_STEP,
    CLAMP_19K_STEP,
    COMPOSITE_ASSEMBLY_STEP,
    FIXED_VIEW_STEPS,
    GAUSSIAN_SPAWN_STEP,
    LOSS_GRAPH_STEP,
    MILESTONE_STEPS,
    SURFEL_DENSIFICATION_STOP,
    SURFEL_PHASE_END,
    VISIBILITY_PRUNE_STEP,
)


@dataclass
class GESModelConfig(SplatfactoModelConfig):
    """Configuration for the GES Model."""

    _target: type = field(default_factory=lambda: GESModel)
    output_depth_during_training: bool = True  # since we need surfel depth to cull gaussians during
    # rendering, we need to output depth during training as well.
    use_dynamic_epsilon: bool = (
        True  # Use dynamic epsilon for depth offset based on Gaussian scales
    )
    surfel_visibility_threshold_real: int = (
        16  # Threshold for surfel visibility pruning (real scenes)
    )
    surfel_visibility_threshold_synthetic: int = (
        4  # Threshold for surfel visibility pruning (synthetic scenes)
    )
    use_real_scene: bool = True  # Whether to use real scene visibility threshold
    max_num_surfels: int = -1  # Restrict maximum number of surfels during training (-1 to disable)
    max_num_gaussians: int = (
        -1
    )  # Restrict maximum number of gaussians during training (-1 to disable)


class Surfel(torch.nn.Module):
    """Data structure for a surfel (2D)"""

    def __init__(
        self,
        means: Parameter,
        quats: Parameter,
        scales: Parameter,
        opacities: Parameter,
        features_dc: Parameter,
        features_rest: Parameter,
    ):
        super().__init__()
        self.means = means
        self.quats = quats
        self.scales = scales
        self.opacities = opacities
        self.features_dc = features_dc
        self.features_rest = features_rest

    @classmethod
    def from_random(cls, rand_size_init: int, scale_init: float, sh_degree: int):
        means = Parameter((torch.rand((rand_size_init, 3)) - 0.5) * scale_init)
        return cls.from_means(means, sh_degree)

    @classmethod
    def from_means(cls, means: Parameter, sh_degree: int):
        distances, _ = k_nearest_sklearn(means.data, 3)
        # find the average of the three nearest neighbors for each point and use that as the scale
        avg_dist = distances.mean(dim=-1, keepdim=True)
        scales = Parameter(torch.log(avg_dist.repeat(1, 3)))
        num_points = means.shape[0]
        quats = Parameter(random_quat_tensor(num_points))
        dim_sh = num_sh_bases(sh_degree)
        features_dc = Parameter(torch.zeros((num_points, 3)))
        features_rest = Parameter(torch.zeros((num_points, dim_sh - 1, 3)))  # For future use
        opacities = Parameter(torch.ones((num_points, 1)) * 0.1)
        return cls(means, quats, scales, opacities, features_dc, features_rest)


class Gaussian(torch.nn.Module):
    """Data structure for a gaussian (3D)"""

    def __init__(
        self,
        means: Parameter,
        quats: Parameter,
        scales: Parameter,
        opacities: Parameter,
        features_dc: Parameter,
        features_rest: Parameter,
    ):
        super().__init__()
        self.means = means
        self.quats = quats
        self.scales = scales
        self.opacities = opacities
        self.features_dc = features_dc
        self.features_rest = features_rest


class GESModel(Model):
    """
    Gaussian-Surfel Model extending Nerfstudio's base Model.
    """

    config: GESModelConfig  # type: ignore

    def __init__(
        self,
        *args,
        seed_points: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ):
        self.seed_points = seed_points
        super().__init__(*args, **kwargs)
        self.register_buffer(
            "saved_gaussian_seeds", torch.zeros((0, 3))
        )  # buffer to save discarded surfel means for gaussian spawning
        self.register_buffer("saved_gaussian_features_dc", torch.zeros((0, 3)))
        self.register_buffer("saved_gaussian_features_rest", torch.zeros((0, 15, 3)))
        self.register_buffer("saved_gaussian_scales", torch.zeros((0, 3)))
        self.register_buffer("saved_gaussian_quats", torch.zeros((0, 4)))
        self.register_buffer("surfel_radii_cache", torch.zeros((0,)))
        self.register_buffer("gaussian_max_contribution_score", torch.zeros((0,)))

        self.error_spawn_points = []
        self.error_spawn_colors = []

        self.l1_loss_history = []  # For plotting loss curve at the end

    def populate_modules(self):
        """
        Called during initialization.
        Initialize your Surfels and Gaussians here based on the seed point cloud.
        """

        if self.seed_points is not None and not self.config.random_init:
            means = self.seed_points[0]
            colors = self.seed_points[1]
            if self.config.max_num_surfels > 0 and means.shape[0] > self.config.max_num_surfels:
                print(
                    f"Subsampling seed points to max_num_surfels: {self.config.max_num_surfels} (originally {means.shape[0]})"
                )
                perm = torch.randperm(means.shape[0], device=means.device)[
                    : self.config.max_num_surfels
                ]
                means = means[perm]
                colors = colors[perm]
                self.seed_points = (means, colors)

            print("Initializing from seed points...")
            means_param = Parameter(self.seed_points[0])
            self.surfel = Surfel.from_means(means=means_param, sh_degree=self.config.sh_degree)
        else:
            self.surfel = Surfel.from_random(
                rand_size_init=self.config.num_random,
                scale_init=self.config.random_scale,
                sh_degree=self.config.sh_degree,
            )

        device = self.surfel.means.device  # Assuming all parameters are on the same device, we can use this for convenience later when spawning Gaussians on the same device.
        num_points = self.surfel.means.shape[0]
        dim_sh = num_sh_bases(self.config.sh_degree)
        if (
            self.seed_points is not None
            and not self.config.random_init
            and self.seed_points[0].shape[1] > 0
        ):
            print("Initializing from seed points...")
            shs = torch.zeros((self.seed_points[1].shape[0], dim_sh, 3)).float().to(device)
            if self.config.sh_degree > 0:
                shs[:, 0, :3] = RGB2SH(self.seed_points[1] / 255)
                shs[:, 1:, :3] = 0.0
            else:
                shs[:, 0, :3] = torch.logit(self.seed_points[1] / 255, eps=1e-10)
            self.surfel.features_dc = Parameter(shs[:, 0, :])
            self.surfel.features_rest = Parameter(shs[:, 1:, :])
            print(
                f"[INIT] Surfel features_dc range: [{shs[:, 0, :].min().item():.4f}, {shs[:, 0, :].max().item():.4f}]"
            )
            print(
                f"[INIT] Surfel seed colors range: [{self.seed_points[1].min().item():.1f}, {self.seed_points[1].max().item():.1f}]"
            )
        else:
            self.surfel.features_dc = Parameter(torch.rand((num_points, 3)))
            self.surfel.features_rest = Parameter(torch.rand((num_points, dim_sh - 1, 3)))

        # Skeletons for Gaussian parameters (3D) must always be initialized on the proper device
        self.gaussian = Gaussian(
            means=Parameter(torch.empty((0, 3), device=device)),
            quats=Parameter(torch.empty((0, 4), device=device)),
            scales=Parameter(torch.empty((0, 3), device=device)),
            opacities=Parameter(torch.empty((0, 1), device=device)),
            features_dc=Parameter(torch.empty((0, 3), device=device)),
            features_rest=Parameter(torch.empty((0, dim_sh - 1, 3), device=device)),
        )

        self.camera_optimizer: CameraOptimizer = self.config.camera_optimizer.setup(
            num_cameras=self.num_train_data, device="cpu"
        )
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = SSIM(data_range=1.0, size_average=True, channel=3)
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)
        self.step = 0

        self.crop_box: OrientedBox | None = None
        if self.config.background_color == "random":
            self.background_color = torch.tensor(
                [0.1490, 0.1647, 0.2157]
            )  # This color is the same as the default background color in Viser.
            # This would only affect the background color when rendering.
        else:
            self.background_color = get_color(self.config.background_color)
        self.strategy = GESStrategy(
            surfel_visibility_threshold_real=self.config.surfel_visibility_threshold_real,
            surfel_visibility_threshold_synthetic=self.config.surfel_visibility_threshold_synthetic,
            use_real_scene=self.config.use_real_scene,
        )
        self.strategy_state = self.strategy.initialize_state()

    def get_param_groups(self) -> dict[str, list[Parameter]]:
        """
        Group parameters for the optimizer specified in config.py.
        """
        return {
            "surfel_means": [self.surfel.means],
            "surfel_quats": [self.surfel.quats],
            "surfel_scales": [self.surfel.scales],
            "surfel_opacities": [self.surfel.opacities],
            "surfel_features_dc": [self.surfel.features_dc],
            "surfel_features_rest": [self.surfel.features_rest],
            "gaussian_means": [self.gaussian.means],
            "gaussian_quats": [self.gaussian.quats],
            "gaussian_scales": [self.gaussian.scales],
            "gaussian_opacities": [self.gaussian.opacities],
            "gaussian_features_dc": [self.gaussian.features_dc],
            "gaussian_features_rest": [self.gaussian.features_rest],
        }

    def get_surfel_param_dict(self) -> dict[str, Parameter]:
        """Get a dictionary of surfel parameters for easy access in the strategy."""
        return {
            "means": self.surfel.means,
            "quats": self.surfel.quats,
            "scales": self.surfel.scales,
            "opacities": self.surfel.opacities,
            "features_dc": self.surfel.features_dc,
            "features_rest": self.surfel.features_rest,
        }

    def get_gaussian_param_dict(self) -> dict[str, Parameter]:
        """Get a dictionary of gaussian parameters for easy access in the strategy."""
        return {
            "means": self.gaussian.means,
            "quats": self.gaussian.quats,
            "scales": self.gaussian.scales,
            "opacities": self.gaussian.opacities,
            "features_dc": self.gaussian.features_dc,
            "features_rest": self.gaussian.features_rest,
        }

    def get_densification_params(self, step) -> dict[str, Parameter] | None:
        """Get the parameters to be used for densification based on the current step."""
        if step <= SURFEL_DENSIFICATION_STOP:
            return self.get_surfel_param_dict()
        elif step > SURFEL_PHASE_END:
            return self.get_gaussian_param_dict()
        return None

    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
    ) -> Any:
        """
        Override load_state_dict to handle changing parameter
        sizes due to densification/pruning.
        """
        for key in ["surfel", "gaussian"]:
            if f"{key}.means" in state_dict:
                num_pts = state_dict[f"{key}.means"].shape[0]
                getattr(self, key).means = Parameter(torch.zeros(num_pts, 3, device=self.device))
                getattr(self, key).quats = Parameter(torch.zeros(num_pts, 4, device=self.device))
                getattr(self, key).scales = Parameter(torch.zeros(num_pts, 3, device=self.device))
                getattr(self, key).opacities = Parameter(
                    torch.zeros(num_pts, 1, device=self.device)
                )
                getattr(self, key).features_dc = Parameter(
                    torch.zeros(num_pts, 3, device=self.device)
                )

                features_rest_shape = state_dict[f"{key}.features_rest"].shape
                getattr(self, key).features_rest = Parameter(
                    torch.zeros(*features_rest_shape, device=self.device)
                )

        if "saved_gaussian_seeds" in state_dict:
            num_seeds = state_dict["saved_gaussian_seeds"].shape[0]
            self.saved_gaussian_seeds = torch.zeros((num_seeds, 3), device=self.device)
            if "saved_gaussian_features_dc" in state_dict:
                self.saved_gaussian_features_dc = torch.zeros((num_seeds, 3), device=self.device)
            if "saved_gaussian_features_rest" in state_dict:
                rest_shape = state_dict["saved_gaussian_features_rest"].shape
                self.saved_gaussian_features_rest = torch.zeros(*rest_shape, device=self.device)
            if "saved_gaussian_scales" in state_dict:
                self.saved_gaussian_scales = torch.zeros((num_seeds, 3), device=self.device)
            if "saved_gaussian_quats" in state_dict:
                self.saved_gaussian_quats = torch.zeros((num_seeds, 4), device=self.device)
        if "surfel_radii_cache" in state_dict:
            num_radii = state_dict["surfel_radii_cache"].shape[0]
            self.surfel_radii_cache = torch.zeros((num_radii,), device=self.device)
        if "gaussian_max_contribution_score" in state_dict:
            num_scores = state_dict["gaussian_max_contribution_score"].shape[0]
            self.gaussian_max_contribution_score = torch.zeros((num_scores,), device=self.device)

        super().load_state_dict(state_dict, strict=strict, assign=assign)

    def get_densification_optimizers(self, step) -> dict[str, torch.optim.Optimizer] | None:
        """Get the optimizers to be used for densification based on the current step."""
        if step <= SURFEL_PHASE_END:  # Surfel phase: densification through step 20k geometry freeze
            return {
                "means": self.optimizers["surfel_means"],
                "quats": self.optimizers["surfel_quats"],
                "scales": self.optimizers["surfel_scales"],
                "opacities": self.optimizers["surfel_opacities"],
                "features_dc": self.optimizers["surfel_features_dc"],
                "features_rest": self.optimizers["surfel_features_rest"],
            }
        elif step > SURFEL_PHASE_END:
            return {
                "means": self.optimizers["gaussian_means"],
                "quats": self.optimizers["gaussian_quats"],
                "scales": self.optimizers["gaussian_scales"],
                "opacities": self.optimizers["gaussian_opacities"],
                "features_dc": self.optimizers["gaussian_features_dc"],
                "features_rest": self.optimizers["gaussian_features_rest"],
            }
        return None

    def step_callback(self, optimizers, step):
        self.step = step
        self.optimizers = optimizers.optimizers
        self.schedulers = optimizers.schedulers

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> list[TrainingCallback]:
        """
        Register callbacks for the Discard Phase (Iter 10k) and Ramp Phase (Iter 18k-20k).
        """
        callbacks = []
        callbacks.append(
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                func=self.step_callback,
                args=[training_callback_attributes.optimizers],
            )
        )

        def phase_10k_callback(step: int):
            if step == self.strategy.surfel_density_stop_iter:
                print(f"Reached {step} iterations, entering discard phase...")
                opacities = torch.sigmoid(self.surfel.opacities.detach())

                # Perform ablation renders before pruning
                if hasattr(self, "fixed_camera"):
                    try:
                        import os

                        import numpy as np
                        from PIL import Image, ImageDraw

                        os.makedirs("web_assets/ablation_10k", exist_ok=True)

                        original_opacities_data = self.surfel.opacities.data.clone()
                        bg = self.config.background_color
                        self.config.background_color = "white"

                        was_training = self.training
                        self.eval()

                        for thresh in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
                            with torch.no_grad():
                                temp_mask = (opacities >= thresh).view(-1)

                                # Apply temporary mask by pushing opacities of pruned points to -100
                                temp_data = original_opacities_data.clone()
                                temp_data[~temp_mask] = -100.0
                                self.surfel.opacities.data.copy_(temp_data)

                                outputs = self.get_outputs(self.fixed_camera)
                                rgb = outputs["rgb"].detach().cpu().numpy()

                                rgb_img = (rgb * 255).astype(np.uint8)
                                img = Image.fromarray(rgb_img)
                                draw = ImageDraw.Draw(img)
                                draw.text(
                                    (10, 10),
                                    f"Thresh: {thresh:.1f} (Keep: {temp_mask.sum().item()})",
                                    fill=(0, 0, 0),
                                )
                                img.save(f"web_assets/ablation_10k/thresh_{thresh:.1f}.png")

                        if was_training:
                            self.train()

                        # Restore
                        self.surfel.opacities.data.copy_(original_opacities_data)
                        self.config.background_color = bg
                        print("Saved 10k ablation renders to web_assets/ablation_10k/")
                    except Exception as e:
                        print(f"Failed to generate ablation renders: {e}")

                # Final pruning logic: use opacity < 0.9 as the threshold.
                # Since opacities here is already passed through sigmoid, this checks opacity < 90%.
                thresh_val = torch.sigmoid(torch.tensor(0.8)).item()
                mask = (opacities >= thresh_val).view(-1)
                print(f"Using opacity threshold {thresh_val:.5f} for pruning.")

                discard_mask = ~mask
                # we save the surfels discarded here so we can use them to initialize the gaussians
                self.saved_gaussian_seeds = self.surfel.means.detach()[discard_mask].clone()
                self.saved_gaussian_features_dc = self.surfel.features_dc.detach()[
                    discard_mask
                ].clone()
                self.saved_gaussian_features_rest = self.surfel.features_rest.detach()[
                    discard_mask
                ].clone()
                self.saved_gaussian_scales = self.surfel.scales.detach()[discard_mask].clone()
                self.saved_gaussian_quats = self.surfel.quats.detach()[discard_mask].clone()
                print(
                    f"Discarding {discard_mask.sum().item()} surfels, keeping {mask.sum().item()} surfels."
                )
                print(
                    f"[10K] Surfel features_dc range after discard: [{self.surfel.features_dc.min().item():.4f}, {self.surfel.features_dc.max().item():.4f}]"
                )
                self.strategy.execute_discard_phase(self, mask)
                self.strategy.clamp_surfel_opacity(self, min_opacity=30.0)
                # Freeze surfel opacity from further optimization (paper: "keep w_i from optimization")
                self.strategy.freeze_surfel_opacity(self)
                print("Frozen surfel opacity from optimization.")

        def phase_15k_callback(step: int):
            if step == VISIBILITY_PRUNE_STEP:
                print("Reached 15k iterations, Visibility Pruning phase...")
                self.strategy.execute_visibility_prune_phase(self)

        def phase_18k_callback(step: int):
            if step == CLAMP_18K_STEP:
                print("Reached 18k iterations, clamping surfel opacity to 60...")
                self.strategy.clamp_surfel_opacity(self, min_opacity=60.0)

        def phase_19k_callback(step: int):
            if step == CLAMP_19K_STEP:
                print("Reached 19k iterations, clamping surfel opacity to 90...")
                self.strategy.clamp_surfel_opacity(self, min_opacity=90.0)

        def phase_20k_callback(step: int):
            if step == self.strategy.gaussian_spawn_iter:
                print("Reached 20k iterations, solidifying surfels and spawning gaussians...")
                self.strategy.clamp_surfel_opacity(self, min_opacity=255.0)
                if isinstance(self.saved_gaussian_seeds, torch.Tensor):
                    self.strategy.spawn_gaussians_from_saved_seeds(self, self.saved_gaussian_seeds)
                else:
                    print(
                        "Warning: Saved gaussian seeds not found or invalid. Skipping Gaussian spawning."
                    )
                self.strategy.freeze_surfel_geometry(self)

        def densification_post_backward_callback(step: int):
            self.strategy.step_post_backward(self, step)

        def save_milestone_callback(step: int):
            if step in MILESTONE_STEPS:
                filename = f"milestone_{step}.pth"
                torch.save(self.state_dict(), filename)
                print(f"Saved model state at milestone iteration {step} to {filename}")

        def save_loss_graph_callback(step: int):
            if step == LOSS_GRAPH_STEP and len(self.l1_loss_history) > 0:
                try:
                    import os

                    import matplotlib.pyplot as plt

                    os.makedirs("web_assets", exist_ok=True)
                    steps, losses = zip(*self.l1_loss_history)

                    # Compute Exponential Moving Average (EMA) for smoother plotting
                    alpha = 0.05
                    ema_losses = []
                    current_ema = losses[0]
                    for loss in losses:
                        current_ema = alpha * loss + (1 - alpha) * current_ema
                        ema_losses.append(current_ema)

                    plt.figure(figsize=(10, 5))
                    plt.plot(steps, ema_losses, label="L1 Loss (Smoothed)", color="tab:blue")
                    plt.plot(steps, losses, alpha=0.15, color="tab:blue", label="Raw Batch Loss")
                    plt.axvline(
                        x=SURFEL_DENSIFICATION_STOP,
                        color="r",
                        linestyle="--",
                        alpha=0.5,
                        label="Discard Phase",
                    )
                    plt.axvline(
                        x=GAUSSIAN_SPAWN_STEP,
                        color="g",
                        linestyle="--",
                        alpha=0.5,
                        label="Gaussian Spawn",
                    )
                    plt.xlabel("Iterations")
                    plt.ylabel("L1 Loss")
                    plt.title("Training L1 Loss Curve")
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    plt.savefig("web_assets/loss_curve.png", dpi=150)
                    plt.close()
                    print("Saved smoothed loss curve to web_assets/loss_curve.png")
                except Exception as e:
                    print(f"Failed to plot loss curve: {e}")

        def save_fixed_view_callback(step: int):
            target_steps = FIXED_VIEW_STEPS
            if step in target_steps and hasattr(self, "fixed_camera"):
                try:
                    was_training = self.training
                    self.eval()
                    with torch.no_grad():
                        # Save current state to avoid modifying it
                        bg = self.config.background_color
                        self.config.background_color = (
                            "white"  # Force white background for consistency
                        )

                        outputs = self.get_outputs(self.fixed_camera)
                        rgb = outputs["rgb"].detach().cpu().numpy()

                        self.config.background_color = bg  # Restore

                        import os

                        import numpy as np
                        from PIL import Image, ImageDraw

                        os.makedirs("web_assets/progress", exist_ok=True)
                        rgb_img = (rgb * 255).astype(np.uint8)
                        img = Image.fromarray(rgb_img)

                        # Add step text
                        draw = ImageDraw.Draw(img)
                        text = f"Step {step}"
                        # Try to draw text, ignore if font fails
                        draw.text((10, 10), text, fill=(0, 0, 0))

                        img.save(f"web_assets/progress/step_{step}.png")

                    if was_training:
                        self.train()

                    # At the end, composite them
                    if step == COMPOSITE_ASSEMBLY_STEP:
                        images = []
                        for s in target_steps:
                            path = f"web_assets/progress/step_{s}.png"
                            if os.path.exists(path):
                                images.append(Image.open(path))

                        if images:
                            # Create a grid. Calculate rows and cols.
                            cols = 4
                            rows = (len(images) + cols - 1) // cols
                            w, h = images[0].size
                            composite = Image.new("RGB", (w * cols, h * rows), color="white")

                            for i, img in enumerate(images):
                                row = i // cols
                                col = i % cols
                                composite.paste(img, (col * w, row * h))

                            composite.save("web_assets/progression_composite.png")
                            print(
                                "Saved progression composite to web_assets/progression_composite.png"
                            )
                except Exception as e:
                    print(f"Failed to save fixed view at step {step}: {e}")

        callbacks.append(
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                update_every_num_iters=1,
                func=densification_post_backward_callback,
            )
        )
        callbacks.append(
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                iters=(SURFEL_DENSIFICATION_STOP,),
                func=phase_10k_callback,
            )
        )
        callbacks.append(
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                iters=(VISIBILITY_PRUNE_STEP,),
                func=phase_15k_callback,
            )
        )
        callbacks.append(
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                iters=(CLAMP_18K_STEP,),
                func=phase_18k_callback,
            )
        )
        callbacks.append(
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                iters=(CLAMP_19K_STEP,),
                func=phase_19k_callback,
            )
        )
        callbacks.append(
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                iters=(GAUSSIAN_SPAWN_STEP,),
                func=phase_20k_callback,
            )
        )
        callbacks.append(
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                iters=tuple(MILESTONE_STEPS),
                func=save_milestone_callback,
            )
        )
        callbacks.append(
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                iters=(LOSS_GRAPH_STEP,),
                func=save_loss_graph_callback,
            )
        )
        callbacks.append(
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                iters=tuple(FIXED_VIEW_STEPS),
                func=save_fixed_view_callback,
            )
        )
        return callbacks

    def _get_downscale_factor(self):
        if self.training:
            return 2 ** max(
                (self.config.num_downscales - self.step // self.config.resolution_schedule),
                0,
            )
        else:
            return 1

    def _downscale_if_required(self, image):
        d = self._get_downscale_factor()
        if d > 1:
            return resize_image(image, d)
        return image

    def _get_background_color(self):
        if self.config.background_color == "random":
            # Randomize background color every 1000 steps during training
            if self.training:
                background_color = torch.rand(3, device=self.device)
            else:
                background_color = self.background_color.to(self.device)
        elif self.config.background_color == "white":
            background_color = torch.ones(3, device=self.device)
        elif self.config.background_color == "black":
            background_color = torch.zeros(3, device=self.device)
        else:
            raise ValueError(f"Invalid background color option: {self.config.background_color}")
        return background_color.to(self.device)

    @staticmethod
    def get_empty_outputs(
        width: int, height: int, background: torch.Tensor
    ) -> dict[str, torch.Tensor | list]:
        rgb = background.repeat(height, width, 1)
        depth = background.new_ones(*rgb.shape[:2], 1) * 10
        accumulation = background.new_zeros(*rgb.shape[:2], 1)
        return {"rgb": rgb, "depth": depth, "accumulation": accumulation, "background": background}

    def get_outputs(self, ray_bundle: RayBundle | Cameras) -> dict[str, torch.Tensor | list]:
        """
        The core rendering logic for a given camera.
        """
        camera = ray_bundle
        if not isinstance(camera, Cameras):
            print("Called get_outputs with not a camera")
            return {}

        # Log surfel and Gaussian counts (helpful for debugging eval)
        if not self.training:
            print(
                f"[EVAL] Rendering with {self.surfel.means.shape[0]} surfels and {self.gaussian.means.shape[0]} Gaussians at step {self.step}"
            )
            if self.surfel.means.shape[0] > 0:
                print(
                    f"[EVAL] Surfel features_dc range: [{self.surfel.features_dc.min().item():.4f}, {self.surfel.features_dc.max().item():.4f}]"
                )
            if self.gaussian.means.shape[0] > 0:
                print(
                    f"[EVAL] Gaussian features_dc range: [{self.gaussian.features_dc.min().item():.4f}, {self.gaussian.features_dc.max().item():.4f}]"
                )

        if self.training:
            assert camera.shape[0] == 1, "Only one camera at a time"
            optimized_camera_to_world = self.camera_optimizer.apply_to_camera(camera)
            # Save the first training camera to render progress from a fixed view
            if not hasattr(self, "fixed_camera"):
                self.fixed_camera = camera.to("cpu")
        else:
            optimized_camera_to_world = camera.camera_to_worlds

        gaussian_crop_ids = None
        surfel_crop_ids = None
        # cropping
        if self.crop_box is not None and not self.training:
            gaussian_crop_ids = self.crop_box.within(self.gaussian.means).squeeze()
            surfel_crop_ids = self.crop_box.within(self.surfel.means).squeeze()
            if gaussian_crop_ids.sum() == 0 and surfel_crop_ids.sum() == 0:
                return self.get_empty_outputs(
                    int(camera.width.item()), int(camera.height.item()), self.background_color
                )

        surfel_crop = (
            self.surfel
            if surfel_crop_ids is None
            else Surfel(
                means=Parameter(self.surfel.means[surfel_crop_ids]),
                quats=Parameter(self.surfel.quats[surfel_crop_ids]),
                scales=Parameter(self.surfel.scales[surfel_crop_ids]),
                opacities=Parameter(self.surfel.opacities[surfel_crop_ids]),
                features_dc=Parameter(self.surfel.features_dc[surfel_crop_ids]),
                features_rest=Parameter(self.surfel.features_rest[surfel_crop_ids]),
            )
        )
        gaussian_crop = (
            self.gaussian
            if gaussian_crop_ids is None
            else Gaussian(
                means=Parameter(self.gaussian.means[gaussian_crop_ids]),
                quats=Parameter(self.gaussian.quats[gaussian_crop_ids]),
                scales=Parameter(self.gaussian.scales[gaussian_crop_ids]),
                opacities=Parameter(self.gaussian.opacities[gaussian_crop_ids]),
                features_dc=Parameter(self.gaussian.features_dc[gaussian_crop_ids]),
                features_rest=Parameter(self.gaussian.features_rest[gaussian_crop_ids]),
            )
        )
        surfel_color_crop = torch.cat(
            (surfel_crop.features_dc[:, None, :], surfel_crop.features_rest), dim=1
        )
        gaussian_color_crop = torch.cat(
            (gaussian_crop.features_dc[:, None, :], gaussian_crop.features_rest), dim=1
        )

        camera_scale_fac = self._get_downscale_factor()
        camera.rescale_output_resolution(1 / camera_scale_fac)
        viewmat = get_viewmat(optimized_camera_to_world)
        if viewmat.dim() == 2:
            viewmat = viewmat.unsqueeze(0)
        viewmat = viewmat.to(self.device)
        intrinsic_mat = camera.get_intrinsics_matrices()
        if intrinsic_mat.dim() == 2:
            intrinsic_mat = intrinsic_mat.unsqueeze(0)
        intrinsic_mat = intrinsic_mat.to(self.device)
        height = int(camera.height.item())
        width = int(camera.width.item())
        camera.rescale_output_resolution(camera_scale_fac)

        if self.config.output_depth_during_training or not self.training:
            render_mode = "RGB+ED"
        else:
            render_mode = "RGB"
        if self.config.sh_degree > 0:
            sh_degree_to_use = min(
                self.step // self.config.sh_degree_interval, self.config.sh_degree
            )
        else:
            surfel_color_crop = torch.sigmoid(surfel_color_crop).squeeze(1)  # [N, 1, 3] -> [N, 3]
            gaussian_color_crop = torch.sigmoid(gaussian_color_crop).squeeze(
                1
            )  # [N, 1, 3] -> [N, 3]
            sh_degree_to_use = None
        # print(f"devices: means on {surfel_crop.means.device}, viewmat on {veiwmat.device},
        # intrinsic on {intrinsic_mat.device}, color on {surfel_color_crop.device},quats on
        # {surfel_crop.quats.device}, scales on {surfel_crop.scales.device}, opacities on
        # {surfel_crop.opacities.device}")
        # rasterization of surfels
        # rasterization of surfels
        if self.step >= 500:
            with open("debug_strategy.txt", "a") as f:
                f.write(f"\n--- Step {self.step} get_outputs ---\n")
                f.write(f"training: {self.training}\n")
                f.write(f"torch.is_grad_enabled(): {torch.is_grad_enabled()}\n")
                f.write(f"surfel_crop.means.shape: {list(surfel_crop.means.shape)}\n")
                f.write(f"surfel_crop.means.requires_grad: {surfel_crop.means.requires_grad}\n")
        if surfel_crop.means.shape[0] > 0:
            opacities = torch.sigmoid(surfel_crop.opacities.squeeze(-1))
            if self.step >= 500:
                with open("debug_strategy.txt", "a") as f:
                    f.write("Entering surfel rasterization path...\n")
                    f.write(f"surfel_crop.quats.requires_grad: {surfel_crop.quats.requires_grad}\n")
                    f.write(
                        f"surfel_crop.scales.requires_grad: {surfel_crop.scales.requires_grad}\n"
                    )
                    f.write(
                        f"surfel_crop.opacities.requires_grad: {surfel_crop.opacities.requires_grad}\n"
                    )
                    f.write(f"opacities.requires_grad: {opacities.requires_grad}\n")
                    f.write(f"surfel_color_crop.requires_grad: {surfel_color_crop.requires_grad}\n")
                    f.write(f"viewmat.requires_grad: {viewmat.requires_grad}\n")
                    f.write(f"intrinsic_mat.requires_grad: {intrinsic_mat.requires_grad}\n")
            # BUG 4 FIX: Pass absgrad=True so the rasterizer computes the
            # absolute-value gradient (means2d.absgrad) needed for effective
            # densification. Without this, the strategy falls back to .grad
            # which suffers from positive/negative cancellation.
            surfel_rgb, surfel_alpha, _, _, _, _, surfel_info = rasterization_surfel(
                means=surfel_crop.means,
                quats=surfel_crop.quats,
                scales=torch.exp(surfel_crop.scales),
                opacities=opacities,
                colors=surfel_color_crop,
                viewmats=viewmat,
                Ks=intrinsic_mat,
                width=width,
                height=height,
                packed=False,
                render_mode=render_mode,
                sh_degree=sh_degree_to_use,
                absgrad=True,
            )
        else:
            # When there are no surfels, they contribute zero color and opacity.
            # We initialize surfel_rgb to zeros so that its premultiplied color is black (0),
            # and surfel_alpha to zeros. If depth is required, we initialize the depth channel to 10.0.
            if render_mode == "RGB+ED":
                surfel_rgb = torch.zeros((1, height, width, 4), device=self.device)
                surfel_rgb[..., 3:4] = 10.0
            else:
                surfel_rgb = torch.zeros((1, height, width, 3), device=self.device)
            surfel_alpha = torch.zeros((1, height, width, 1), device=self.device)
            surfel_info = None

        if surfel_rgb.shape[-1] == 4:
            surfel_depth = surfel_rgb[..., 3:4]
        else:
            surfel_depth = torch.zeros((height, width, 1), device=self.device) + 10.0

        if self.gaussian.means.shape[0] > 0:
            R = viewmat[0, :3, :3]
            T = viewmat[0, :3, 3]
            cam_space_means = torch.matmul(gaussian_crop.means, R.T) + T
            gaussian_depths = cam_space_means[..., 2]
            # project to 2d screen coords
            fx, fy = intrinsic_mat[0, 0, 0], intrinsic_mat[0, 1, 1]
            cx, cy = intrinsic_mat[0, 0, 2], intrinsic_mat[0, 1, 2]

            x_screen = (cam_space_means[..., 0] * fx) / (gaussian_depths + 1e-5) + cx
            y_screen = (cam_space_means[..., 1] * fy) / (gaussian_depths + 1e-5) + cy

            x_norm = (x_screen / width) * 2.0 - 1.0
            y_norm = (y_screen / height) * 2.0 - 1.0
            grid = torch.stack((x_norm, y_norm), dim=-1).unsqueeze(0).unsqueeze(0)  # [1, 1, N, 2]

            # Surfel depth may already carry a batch dimension depending on the renderer path.
            # Normalize it to [B, 1, H, W] before grid sampling.
            if surfel_depth.dim() == 4:
                surfel_depth_map = surfel_depth.permute(0, 3, 1, 2)
            elif surfel_depth.dim() == 3:
                surfel_depth_map = surfel_depth.permute(2, 0, 1).unsqueeze(0)
            else:
                raise RuntimeError(f"Unexpected surfel_depth shape: {tuple(surfel_depth.shape)}")
            sampled_depths = torch.nn.functional.grid_sample(
                surfel_depth_map, grid, mode="nearest", padding_mode="border", align_corners=False
            ).squeeze()  # [N]

            # Compute dynamic epsilon based on Gaussian scales (as per paper Eq. 2-3)
            # ε_i = (1/D) * Σ(s_{i,j}) where D=3 (dimension) and s_{i,j} are axis lengths of scale
            with torch.no_grad():
                if self.config.use_dynamic_epsilon:
                    # Scales are stored in log space, so exponentiate them
                    scale_magnitudes = torch.exp(gaussian_crop.scales.detach())  # [N, 3]
                    # Average the three axis lengths for each Gaussian
                    delta = scale_magnitudes.mean(dim=1)  # [N]
                else:
                    delta = 0.05 * torch.ones(
                        gaussian_crop.means.shape[0], device=gaussian_depths.device
                    )

            valid_mask = gaussian_depths < sampled_depths + delta

            in_screen_mask = (
                (x_screen >= 0) & (x_screen < width) & (y_screen >= 0) & (y_screen < height)
            )

            valid_mask = valid_mask & in_screen_mask

            culled_opacities = gaussian_crop.opacities.clone()
            culled_opacities[
                ~valid_mask
            ] = -100.0  # set to a very low value so that after sigmoid it becomes near zero and
            # gets culled in rendering

            opacities = torch.sigmoid(culled_opacities.squeeze(-1))
            # BUG 4 FIX: Pass absgrad=True for Gaussian rasterization too,
            # so that Gaussian densification (after 20k) uses absolute gradients.
            gaussian_render, gaussian_alpha, gaussian_info = rasterization(
                means=gaussian_crop.means,
                quats=gaussian_crop.quats,
                scales=torch.exp(gaussian_crop.scales),
                opacities=opacities,
                colors=gaussian_color_crop,
                viewmats=viewmat,
                Ks=intrinsic_mat,
                width=width,
                height=height,
                packed=False,
                render_mode=render_mode,
                sh_degree=sh_degree_to_use,
                rasterize_mode=self.config.rasterize_mode,
                absgrad=True,
            )
        else:
            gaussian_render = torch.zeros_like(surfel_rgb)
            gaussian_alpha = torch.zeros_like(surfel_alpha)
            gaussian_info = None
        self.info = {"surfels": surfel_info, "gaussians": gaussian_info}
        if self.training:
            if self.step <= VISIBILITY_PRUNE_STEP:
                if surfel_info is not None:
                    # BUG 7 FIX: 2DGS rasterization returns radii as [C, N, 2]
                    # (two axis-aligned bounding-box radii per surfel per camera).
                    # Reduce to [N] by taking max over cameras and max over the
                    # 2 axes. Previously the else branch kept [1, N, 2] which
                    # caused shape mismatches in visibility pruning at 15k.
                    radii_tensor = surfel_info["radii"].detach()
                    if radii_tensor.dim() == 3:
                        # [C, N, 2] -> max over cameras -> [N, 2] -> max over axes -> [N]
                        surfel_radii = radii_tensor.max(dim=0).values.max(dim=-1).values
                    elif radii_tensor.dim() == 2:
                        # Could be [C, N] (standard) or [N, 2] (2DGS with C=1 squeezed)
                        # For [C, N]: max over cameras -> [N]
                        # For [N, 2]: max over axes -> [N]
                        surfel_radii = radii_tensor.max(dim=0).values
                    else:
                        surfel_radii = radii_tensor

                    state = self.strategy_state["surfels"]
                    # If this is the first step or we have just loaded a checkpoint, restore from self.surfel_radii_cache if available.
                    # Otherwise, initialize with zeros.
                    if "surfel_radii_cache" not in state or state["surfel_radii_cache"] is None:
                        if (
                            self.surfel_radii_cache is not None
                            and self.surfel_radii_cache.shape[0] == surfel_radii.shape[0]
                        ):
                            state["surfel_radii_cache"] = self.surfel_radii_cache.clone()
                        else:
                            state["surfel_radii_cache"] = torch.zeros_like(surfel_radii)

                    # If there's a shape mismatch (e.g. from manual/external changes), resize to zeros
                    if state["surfel_radii_cache"].shape[0] != surfel_radii.shape[0]:
                        state["surfel_radii_cache"] = torch.zeros_like(surfel_radii)

                    # Accumulate maximum pixel-unit radii over all training steps (never zeroed out by strategy)
                    state["surfel_radii_cache"] = torch.maximum(
                        state["surfel_radii_cache"], surfel_radii
                    )
                    self.surfel_radii_cache = state["surfel_radii_cache"]
                else:
                    surfel_radii = None

            self.strategy.step_pre_backward(self, self.step)

        # gaussian_alpha already has the correct shape from rasterization.
        # (Removed no-op `gaussian_alpha = gaussian_alpha[:, ...]` that was vestigial code.)

        background_color = self._get_background_color()

        gaussian_rgb = gaussian_render[:, ..., :3]
        surfel_rgb_color = surfel_rgb[:, ..., :3]
        C_S_premul = surfel_rgb_color.squeeze(0)
        W_S = surfel_alpha.squeeze(0)
        C_G_premul = gaussian_rgb.squeeze(0)
        W_G = gaussian_alpha.squeeze(0)

        if self.gaussian.means.shape[0] == 0:
            # During the surfel-only phase (steps 0–20k, no Gaussians), we use
            # standard alpha compositing exactly as Splatfacto does:
            #   rgb = C_premul + (1 - alpha) * background
            # This is proven and numerically stable.
            total_alpha = W_S
            rgb = C_S_premul + (1.0 - total_alpha) * background_color
            rgb = torch.clamp(rgb, 0.0, 1.0)

            if render_mode == "RGB+ED":
                surfel_depth_val = surfel_rgb[:, ..., 3:4].squeeze(0)
                depth = surfel_depth_val
                # BUG 8 FIX: Use total_alpha instead of gaussian_alpha
                depth = torch.where(total_alpha > 0, depth, depth.detach().max())
            else:
                depth = None
        else:
            # After Gaussians are spawned at step 20k, we apply the paper's Eq. 5:
            #   C = (C_S + C_G) / (W_S + W_G)
            # where C_S, C_G are premultiplied color maps and W_S, W_G are their alphas.
            total_alpha = torch.clamp(W_S + W_G, 0.0, 1.0)
            combined_color = (C_S_premul + C_G_premul) / (W_S + W_G + 1e-5)
            rgb = combined_color * total_alpha + (1.0 - total_alpha) * background_color
            rgb = torch.clamp(rgb, 0.0, 1.0)

            if render_mode == "RGB+ED":
                gaussian_depth = gaussian_render[:, ..., 3:4].squeeze(0)
                surfel_depth_val = surfel_rgb[:, ..., 3:4].squeeze(0)
                # Depth compositing: weighted average of Gaussian and surfel depths
                depth = (gaussian_depth * W_G + surfel_depth_val * W_S) / (W_S + W_G + 1e-5)
                # BUG 8 FIX: Use total_alpha (already squeezed to [H,W,1]) instead
                # of gaussian_alpha (which still has batch dim from rasterization)
                depth = torch.where(total_alpha > 0, depth, depth.detach().max())
            else:
                depth = None
        if background_color.shape[0] == 3 and not self.training:
            background_color = background_color.expand(height, width, 3)

        # Add annotation text during evaluation
        if not self.training:
            try:
                import numpy as np
                from PIL import Image, ImageDraw

                # Convert tensor to PIL Image
                rgb_np = (rgb.detach().cpu().numpy() * 255).astype(np.uint8)
                img = Image.fromarray(rgb_np)
                draw = ImageDraw.Draw(img)

                # Add text label
                text = f"Render (Surfels: {self.surfel.means.shape[0]}, Gaussians: {self.gaussian.means.shape[0]})"
                draw.text((10, 10), text, fill=(255, 255, 255))

                # Convert back to tensor
                rgb = torch.from_numpy(np.array(img).astype(np.float32) / 255.0).to(rgb.device)
            except Exception as e:
                # If text rendering fails, just continue without annotation
                print(f"Warning: Could not add text annotation: {e}")

        out_dict = {
            "rgb": rgb,
            "depth": depth,  # type: ignore
            "accumulation": total_alpha,
            "background": background_color,
        }
        if self.training:
            out_dict.update(
                {
                    "surfel_depth": surfel_depth,
                    "surfel_alpha": surfel_alpha,
                    "gaussian_alpha": gaussian_alpha if self.gaussian.means.shape[0] > 0 else None,
                    "camera_to_world": optimized_camera_to_world,
                    "intrinsic_mat": intrinsic_mat,
                    "x_screen": x_screen if self.gaussian.means.shape[0] > 0 else None,
                    "y_screen": y_screen if self.gaussian.means.shape[0] > 0 else None,
                }
            )
        return out_dict

    def get_gt_img(self, image: torch.Tensor):
        """
        Compute ground truth image with iteration dependent downscale factor for evaluation
        purpose

        Args:
            image: tensor.Tensor in type uint8 or float32
        """
        if image.dtype == torch.uint8:
            image = image.float() / 255.0
        gt_img = self._downscale_if_required(image)
        return gt_img.to(self.device)

    def composite_with_background(self, image, background) -> torch.Tensor:
        """Composite the ground truth image with a background color when it has an alpha channel.

        Args:
            image: the image to composite
            background: the background color
        """
        if image.shape[2] == 4:
            alpha = image[..., -1].unsqueeze(-1).repeat((1, 1, 3))
            return alpha * image[..., :3] + (1 - alpha) * background
        else:
            return image

    def get_metrics_dict(self, outputs, batch) -> dict[str, torch.Tensor]:
        """Compute and returns metrics.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
        """
        gt_rgb = self.composite_with_background(
            self.get_gt_img(batch["image"]), outputs["background"]
        )
        metrics_dict = {}
        predicted_rgb = outputs["rgb"]

        metrics_dict["psnr"] = self.psnr(predicted_rgb, gt_rgb)
        if self.config.color_corrected_metrics:
            cc_rgb = color_correct(predicted_rgb, gt_rgb)
            metrics_dict["cc_psnr"] = self.psnr(cc_rgb, gt_rgb)

        metrics_dict["surfel_count"] = self.surfel.means.shape[0]
        metrics_dict["gaussian_count"] = self.gaussian.means.shape[0]

        self.camera_optimizer.get_metrics_dict(metrics_dict)
        return metrics_dict

    def set_crop(self, crop_box: OrientedBox | None):
        self.crop_box = crop_box

    @torch.no_grad()
    def get_outputs_for_camera(
        self, camera: Cameras, obb_box: OrientedBox | None = None
    ) -> dict[str, torch.Tensor]:
        """Takes in a camera, generates the ray bundle, and computes the output of the model.
        Overridden for a camera-based gaussian model.

        Args:
            camera: generates ray bundle
        """
        assert camera is not None, "must provide camera to gaussian model"
        self.set_crop(obb_box)
        outs = self.get_outputs(camera.to(self.device))
        return outs  # type: ignore

    def get_image_metrics_and_images(
        self, outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
    ) -> tuple[dict[str, float], dict[str, torch.Tensor]]:
        """Writes the test image outputs.

        Args:
            image_idx: Index of the image.
            step: Current step.
            batch: Batch of data.
            outputs: Outputs of the model.

        Returns:
            A dictionary of metrics.
        """
        gt_rgb = self.composite_with_background(
            self.get_gt_img(batch["image"]), outputs["background"]
        )
        predicted_rgb = outputs["rgb"]
        cc_rgb = None

        combined_rgb = torch.cat([gt_rgb, predicted_rgb], dim=1)

        if self.config.color_corrected_metrics:
            cc_rgb = color_correct(predicted_rgb, gt_rgb)
            cc_rgb = torch.moveaxis(cc_rgb, -1, 0)[None, ...]

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        gt_rgb = torch.moveaxis(gt_rgb, -1, 0)[None, ...]
        predicted_rgb = torch.moveaxis(predicted_rgb, -1, 0)[None, ...]

        psnr = self.psnr(gt_rgb, predicted_rgb)
        ssim = self.ssim(gt_rgb, predicted_rgb)
        lpips = self.lpips(gt_rgb, predicted_rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore
        metrics_dict["lpips"] = float(lpips)

        metrics_dict["surfel_count"] = float(self.surfel.means.shape[0])
        metrics_dict["gaussian_count"] = float(self.gaussian.means.shape[0])

        if self.config.color_corrected_metrics:
            assert cc_rgb is not None
            cc_psnr = self.psnr(gt_rgb, cc_rgb)
            cc_ssim = self.ssim(gt_rgb, cc_rgb)
            cc_lpips = self.lpips(gt_rgb, cc_rgb)
            metrics_dict["cc_psnr"] = float(cc_psnr.item())
            metrics_dict["cc_ssim"] = float(cc_ssim)
            metrics_dict["cc_lpips"] = float(cc_lpips)

        images_dict = {"img": combined_rgb}

        return metrics_dict, images_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> dict[str, torch.Tensor]:
        """Computes and returns the losses dict.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
            metrics_dict: dictionary of metrics, some of which we can use for loss
        """
        gt_img = self.composite_with_background(
            self.get_gt_img(batch["image"]), outputs["background"]
        )
        pred_img = outputs["rgb"]

        # Set masked part of both ground-truth and rendered image to black.
        # This is a little bit sketchy for the SSIM loss.
        if "mask" in batch:
            # batch["mask"] : [H, W, 1]
            mask = self._downscale_if_required(batch["mask"])
            mask = mask.to(self.device)
            assert mask.shape[:2] == gt_img.shape[:2] == pred_img.shape[:2]
            gt_img = gt_img * mask
            pred_img = pred_img * mask

        Ll1 = torch.abs(gt_img - pred_img).mean()

        if not Ll1.requires_grad:
            print(f"\n[DIAGNOSTIC] Step {self.step}: Ll1 does not require grad!")
            print(f"  torch.is_grad_enabled(): {torch.is_grad_enabled()}")
            print(f"  pred_img.requires_grad: {pred_img.requires_grad}")
            print(f"  pred_img.grad_fn: {pred_img.grad_fn}")
            if "surfels" in self.info and self.info["surfels"] is not None:
                print(
                    f"  surfels means2d requires_grad: {self.info['surfels']['means2d'].requires_grad}"
                )
                print(
                    f"  surfels opacities requires_grad: {self.info['surfels']['opacities'].requires_grad}"
                )
            print(f"  surfel means requires_grad: {self.surfel.means.requires_grad}")
            print(
                f"  surfel colors (features_dc) requires_grad: {self.surfel.features_dc.requires_grad}"
            )
            print(f"  gaussian means requires_grad: {self.gaussian.means.requires_grad}")
            print(
                f"  gaussian colors (features_dc) requires_grad: {self.gaussian.features_dc.requires_grad}"
            )
            req_grad_params = [
                name for name, param in self.named_parameters() if param.requires_grad
            ]
            print(f"  All parameters requiring grad: {req_grad_params}\n")

        if self.training and self.step % 100 == 0:
            self.l1_loss_history.append((self.step, Ll1.item()))

        simloss = 1 - self.ssim(
            gt_img.permute(2, 0, 1)[None, ...], pred_img.permute(2, 0, 1)[None, ...]
        )

        # BUG 6 FIX: Guard against empty Gaussian tensors producing NaN.
        # When self.gaussian.scales has shape [0, 3] (before 20k when no
        # Gaussians exist), amax/amin produce empty tensors and .mean()
        # returns nan, which corrupts the loss and all parameter updates.
        if self.config.use_scale_regularization and self.step % 10 == 0:
            surfel_scale_exp = torch.exp(self.surfel.scales)
            if surfel_scale_exp.shape[0] > 0:
                surfel_scale_reg = (
                    torch.maximum(
                        surfel_scale_exp.amax(dim=-1) / surfel_scale_exp.amin(dim=-1),
                        torch.tensor(self.config.max_gauss_ratio),
                    )
                    - self.config.max_gauss_ratio
                )
                surfel_scale_reg = surfel_scale_reg.mean()
            else:
                surfel_scale_reg = torch.tensor(0.0, device=self.device)

            gaussian_scale_exp = torch.exp(self.gaussian.scales)
            if gaussian_scale_exp.shape[0] > 0:
                gaussian_scale_reg = (
                    torch.maximum(
                        gaussian_scale_exp.amax(dim=-1) / gaussian_scale_exp.amin(dim=-1),
                        torch.tensor(self.config.max_gauss_ratio),
                    )
                    - self.config.max_gauss_ratio
                )
                gaussian_scale_reg = gaussian_scale_reg.mean()
            else:
                gaussian_scale_reg = torch.tensor(0.0, device=self.device)

            scale_reg = 0.1 * (surfel_scale_reg + gaussian_scale_reg)
        else:
            scale_reg = torch.tensor(0.0).to(self.device)

        loss_dict = {
            "main_loss": (1 - self.config.ssim_lambda) * Ll1 + self.config.ssim_lambda * simloss,
            "scale_reg": scale_reg,
        }
        if self.training:
            # Add loss from camera optimizer
            self.camera_optimizer.get_loss_dict(loss_dict)

        # --- Error-based Spawning and Score-based Pruning (BUG 16) ---
        if self.training and 20000 < self.step <= 30000:
            device = self.device
            height, width = gt_img.shape[:2]

            # 1. Error-based Spawning: Accumulate high-error points
            error_map = torch.mean((gt_img.detach() - pred_img.detach()) ** 2, dim=-1)  # [H, W]
            probs = error_map.flatten() / (error_map.sum() + 1e-8)

            # Sample N = 100 points
            N = 100
            sampled_indices = torch.multinomial(probs, num_samples=N, replacement=True)
            u = sampled_indices % width
            v = sampled_indices // width

            surfel_depth = outputs.get("surfel_depth")
            if surfel_depth is not None:
                # Extract depth
                z = surfel_depth.squeeze(0)[v, u, 0]  # [N]
                valid_mask = z < 9.0  # depth < 9.0 are valid surfel surfaces

                if valid_mask.sum() > 0:
                    u_v = u[valid_mask]
                    v_v = v[valid_mask]
                    z_v = z[valid_mask]

                    intrinsic_mat = outputs["intrinsic_mat"]
                    fx, fy = intrinsic_mat[0, 0, 0], intrinsic_mat[0, 1, 1]
                    cx, cy = intrinsic_mat[0, 0, 2], intrinsic_mat[0, 1, 2]

                    # Unproject to camera space
                    x_cam = (u_v - cx) * z_v / fx
                    y_cam = (v_v - cy) * z_v / fy
                    pos_cam = torch.stack([x_cam, y_cam, z_v], dim=-1)  # [M, 3]

                    # Transform to world space
                    c2w = outputs["camera_to_world"]
                    R = c2w[0, :3, :3]
                    T = c2w[0, :3, 3]
                    pos_world = torch.matmul(pos_cam, R.T) + T  # [M, 3]

                    # Retrieve ground truth color
                    color = gt_img[v_v, u_v]  # [M, 3]

                    self.error_spawn_points.append(pos_world.cpu())
                    self.error_spawn_colors.append(color.cpu())

            # 2. Score-based Pruning: Update maximum contribution scores
            gaussian_alpha = outputs.get("gaussian_alpha")
            surfel_alpha = outputs.get("surfel_alpha")
            x_screen = outputs.get("x_screen")
            y_screen = outputs.get("y_screen")

            if (
                self.gaussian.means.shape[0] > 0
                and gaussian_alpha is not None
                and surfel_alpha is not None
                and x_screen is not None
                and y_screen is not None
            ):
                num_gaussians = self.gaussian.means.shape[0]

                # Verify max contribution score buffer size
                if self.gaussian_max_contribution_score.shape[0] != num_gaussians:
                    self.gaussian_max_contribution_score = torch.zeros(num_gaussians, device=device)

                # Convert log-features to base RGB color
                c_rgb = torch.clamp(SH2RGB(self.gaussian.features_dc.detach()), 0.0, 1.0)
                c_i = c_rgb.max(dim=-1).values  # [N]
                alpha_i = torch.sigmoid(self.gaussian.opacities.detach()).flatten()  # [N]

                # Sample W_S and W_G at Gaussian center
                x_norm = (x_screen / width) * 2.0 - 1.0
                y_norm = (y_screen / height) * 2.0 - 1.0
                grid = (
                    torch.stack((x_norm, y_norm), dim=-1).unsqueeze(0).unsqueeze(0)
                )  # [1, 1, N, 2]

                # surfel_alpha permuted for grid_sample
                surfel_alpha_map = surfel_alpha.permute(0, 3, 1, 2)  # [1, 1, H, W]
                gaussian_alpha_map = gaussian_alpha.permute(0, 3, 1, 2)  # [1, 1, H, W]

                sampled_W_S = torch.nn.functional.grid_sample(
                    surfel_alpha_map,
                    grid,
                    mode="nearest",
                    padding_mode="border",
                    align_corners=False,
                ).flatten()  # [N]
                sampled_W_G = torch.nn.functional.grid_sample(
                    gaussian_alpha_map,
                    grid,
                    mode="nearest",
                    padding_mode="border",
                    align_corners=False,
                ).flatten()  # [N]

                # Compute contribution score
                score = (c_i * alpha_i) / (sampled_W_S + sampled_W_G + 1e-5)  # [N]

                # Update running maximum score
                self.gaussian_max_contribution_score = torch.maximum(
                    self.gaussian_max_contribution_score, score
                )

            # 3. Periodic Spawning and Pruning Callback (Every 1000 steps)
            if self.step > 20000 and self.step % 1000 == 0:
                print(
                    f"[Joint Optimization] Periodic step {self.step} unprojected spawning and contribution pruning..."
                )

                # 3a. Execute Spawning
                if len(self.error_spawn_points) > 0:
                    spawn_pts = torch.cat(self.error_spawn_points, dim=0).to(device)
                    spawn_cols = torch.cat(self.error_spawn_colors, dim=0).to(device)

                    self.strategy.spawn_gaussians_from_error_seeds(self, spawn_pts, spawn_cols)

                    # Clear lists
                    self.error_spawn_points = []
                    self.error_spawn_colors = []

                # 3b. Execute Pruning based on contribution score < 0.02
                if self.gaussian.means.shape[0] > 0:
                    num_gaussians = self.gaussian.means.shape[0]
                    # Verify max contribution score buffer size
                    if self.gaussian_max_contribution_score.shape[0] == num_gaussians:
                        keep_mask = self.gaussian_max_contribution_score >= 0.02
                        num_prune = (~keep_mask).sum().item()
                        if num_prune > 0:
                            print(
                                f"[Contribution Pruning] Pruning {num_prune} Gaussians with contribution score < 0.02."
                            )
                            self.strategy.execute_contribution_pruning(self, keep_mask)

                        # Reset contribution score buffer to zeros for the next 1000 iterations
                        self.gaussian_max_contribution_score = torch.zeros(
                            self.gaussian.means.shape[0], device=device
                        )
                        # Also keep strategy state in sync
                        state = self.strategy_state["gaussians"]
                        if "gaussian_max_contribution_score" in state:
                            state["gaussian_max_contribution_score"] = (
                                self.gaussian_max_contribution_score.clone()
                            )

        return loss_dict
