"""
ges_model.py
Contains the custom PyTorch Model defining the GES logic.
"""
from pytorch_msssim import SSIM

import torch
from torch.nn import Parameter
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type, Union

from nerfstudio.cameras.camera_optimizers import CameraOptimizer, CameraOptimizerConfig
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.model_components.lib_bilagrid import color_correct, total_variation_loss
from nerfstudio.models.base_model import Model
from nerfstudio.models.splatfacto import SplatfactoModelConfig, get_viewmat, resize_image
from nerfstudio.utils.spherical_harmonics import RGB2SH, SH2RGB, num_sh_bases
from nerfstudio.utils.math import k_nearest_sklearn, random_quat_tensor
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.utils.colors import get_color
from ges_strategy import GESStrategy
# Assuming gsplat is in external_code/gsplat and accessible
from gsplat.rendering import rasterization, rasterization_2dgs

@dataclass
class GESModelConfig(SplatfactoModelConfig):
    """Configuration for the GES Model."""
    _target: Type = field(default_factory=lambda: GESModel)
    

class Surfel:
    """Data structure for a surfel (2D)"""
    def __init__(self, means: Parameter, quats: Parameter, scales: Parameter, opacities: Parameter, features_dc: Parameter, features_rest: Parameter):
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
        num_points =  means.shape[0]
        quats = Parameter(random_quat_tensor(num_points))
        dim_sh = num_sh_bases(sh_degree)
        features_dc = Parameter(torch.zeros((num_points, 3)))
        features_rest = Parameter(torch.zeros((num_points, dim_sh - 1, 3))) # For future use
        opacities = Parameter(torch.zeros((num_points, 1)))
        return cls(means, quats, scales, opacities, features_dc, features_rest)
class Gaussian:
    """Data structure for a gaussian (3D)"""
    def __init__(self, means: Parameter, quats: Parameter, scales: Parameter, opacities: Parameter, features_dc: Parameter, features_rest: Parameter):
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
    config: GESModelConfig
    
    def __init__(
        self,
        *args,
        seed_points: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        self.seed_points = seed_points
        super().__init__(*args, **kwargs)

    def populate_modules(self):
        """
        Called during initialization.
        Initialize your Surfels and Gaussians here based on the seed point cloud.
        """
        # TODO: Load initial point cloud from self.kwargs["seed_points"] if provided by datamanager
        # Skeletons for Surfel parameters (2D)

        if self.seed_points is not None and not self.config.random_init:
            print("Initializing from seed points...")
            means = Parameter(self.seed_points[0])
            self.surfel = Surfel.from_means(
                means=means,
                sh_degree=self.config.sh_degree
            )
        else:
            self.surfel = Surfel.from_random(
                rand_size_init=self.config.num_random,
                scale_init=self.config.random_scale,
                sh_degree=self.config.sh_degree
            )
        num_points = self.surfel.means.shape[0]
        dim_sh = num_sh_bases(self.config.sh_degree)
        if (
            self.seed_points is not None
            and not self.config.random_init
            and self.seed_points[0].shape[1]>0
            ):
            print("Initializing from seed points...")
            shs = torch.zeros((self.seed_points[1].shape[0], dim_sh, 3)).float().to(self.device)
            if self.config.sh_degree > 0:
                shs[:, 0, :3] = RGB2SH(self.seed_points[1]/255)
                shs[:, 1:, 3:] = 0.0
            else:
                shs[:, 0, :3] = torch.logit(self.seed_points[1]/255, eps=1e-10)
            self.surfel.features_dc = Parameter(shs[:, 0, :])
            self.surfel.features_rest = Parameter(shs[:, 1:, :])
        else:
            self.surfel.features_dc = Parameter(torch.rand((num_points, 3)))
            self.surfel.features_rest = Parameter(torch.rand((num_points, dim_sh - 1, 3)))
            # Skeletons for Gaussian parameters (3D)
            self.gaussian = Gaussian(
                means=Parameter(torch.empty((0, 3))),
                quats=Parameter(torch.empty((0, 4))),
                scales=Parameter(torch.empty((0, 3))),
                opacities=Parameter(torch.empty((0, 1))),
                features_dc=Parameter(torch.empty((0, 3))),
                features_rest=Parameter(torch.empty((0, dim_sh - 1, 3)))
            )

        self.camera_optimizer: CameraOptimizer =  self.config.camera_optimizer.setup(
            num_cameras=self.num_train_data, device="cpu"
        )
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = SSIM(data_range=1.0, size_average=True, channel=3)
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)
        self.step = 0

        self.crop_box: Optional[OrientedBox] = None
        if self.config.background_color == "random":
            self.background_color = torch.tensor(
                [0.1490, 0.1647, 0.2157]
            )  # This color is the same as the default background color in Viser. This would only affect the background color when rendering.
        else:
            self.background_color = get_color(self.config.background_color)
        self.strategy = GESStrategy()
        self.strategy_state = self.strategy.initialize_state()

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
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
    def get_surfel_param_dict(self) -> Dict[str, Parameter]:
        """Get a dictionary of surfel parameters for easy access in the strategy."""
        return {
            "means": self.surfel.means,
            "quats": self.surfel.quats,
            "scales": self.surfel.scales,
            "opacities": self.surfel.opacities,
            "features_dc": self.surfel.features_dc,
            "features_rest": self.surfel.features_rest,
        }
    def get_gaussian_param_dict(self) -> Dict[str, Parameter]:
        """Get a dictionary of gaussian parameters for easy access in the strategy."""
        return {
            "means": self.gaussian.means,
            "quats": self.gaussian.quats,
            "scales": self.gaussian.scales,
            "opacities": self.gaussian.opacities,
            "features_dc": self.gaussian.features_dc,
            "features_rest": self.gaussian.features_rest,
        }
    def get_densification_params(self, step) -> Optional[Dict[str, Parameter]]:
        """Get the parameters to be used for densification based on the current step."""
        if step <= 10000:
            return self.get_surfel_param_dict()
        elif step > 20000:
            return self.get_gaussian_param_dict()
        return None
    def get_densification_optimizers(self, step) -> Optional[Dict[str, torch.optim.Optimizer]]:
        """Get the optimizers to be used for densification based on the current step."""
        if step <= 10000:
            return {
                "means": self.optimizers["surfel_means"],
                "quats": self.optimizers["surfel_quats"],
                "scales": self.optimizers["surfel_scales"],
                "opacities": self.optimizers["surfel_opacities"],
                "features_dc": self.optimizers["surfel_features_dc"],
                "features_rest": self.optimizers["surfel_features_rest"]
            }
        elif step > 20000:
            return {
                "means": self.optimizers["gaussian_means"],
                "quats": self.optimizers["gaussian_quats"],
                "scales": self.optimizers["gaussian_scales"],
                "opacities": self.optimizers["gaussian_opacities"],
                "features_dc": self.optimizers["gaussian_features_dc"],
                "features_rest": self.optimizers["gaussian_features_rest"]
            }
        return None

    def step_callback(self, optimizers, step):
        self.step = step
        self.optimizers = optimizers.optimizers
        self.schedulers = optimizers.schedulers
    def save_params(self, step, densification_params):
        """Save the params to the respective vars"""
        if step <= 10000:
            self.surfel.means = densification_params["means"]
            self.surfel.quats = densification_params["quats"]
            self.surfel.scales = densification_params["scales"]
            self.surfel.opacities = densification_params["opacities"]
            self.surfel.features_dc = densification_params["features_dc"]
            self.surfel.features_rest = densification_params["features_rest"]
        elif step > 20000:
            self.gaussian.means = densification_params["means"]
            self.gaussian.quats = densification_params["quats"]
            self.gaussian.scales = densification_params["scales"]
            self.gaussian.opacities = densification_params["opacities"]
            self.gaussian.features_dc = densification_params["features_dc"]
            self.gaussian.features_rest = densification_params["features_rest"]
        
    def get_training_callbacks(self, training_callback_attributes: TrainingCallbackAttributes) -> List[TrainingCallback]:
        """
        Register callbacks for the Discard Phase (Iter 10k) and Ramp Phase (Iter 18k-20k).
        """
        callbacks = []
        #TODO: implement the discard phase and ramp phase logic in the respective callbacks, currently just placeholders
        callbacks.append(
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                func=self.step_callback,
                args=[training_callback_attributes.optimizers]
            )
        )
        def phase_10k_callback(step: int):
            if step == 10000:
                print("Reached 10k iterations, entering discard phase...")
                opacities = torch.sigmoid(self.surfel.opacities.detach())
                mask = (opacities >=0.8).squeeze()
                discard_mask = ~mask
                # we save the surfels discarded here so we can use them to initialize the gaussians
                self.saved_gaussian_seeds = self.surfel.means.detach()[discard_mask].clone()
                self.strategy.execute_10k_discard(self, mask)
                self.strategy.clamp_surfel_opacity(self, min_opacity=30.0)
                
            
        def phase_15k_callback(step: int):
            if step == 15000:
                print("Reached 15k iterations, Visibility Pruning phase...")
                self.strategy.execute_15k_visibility_prune(self)
        def phase_18k_19k_callback(step: int):
            if step == 18000:
                print("Reached 18k iterations, opacity clamp to 60.0 ...")
                self.strategy.clamp_surfel_opacity(self, min_opacity=60.0)
            elif step == 19000:
                print("Reached 20k iterations opacity clamp to 90.0 ...")
                self.strategy.clamp_surfel_opacity(self, min_opacity=90.0)
        def phase_20k_callback(step: int):
            if step == 20000:
                print("Reached 20k iterations, solidifying surfels and spawning gaussians...")
                self.strategy.clamp_surfel_opacity(self, min_opacity=255.0)
                self.strategy.spawn_gaussians_from_saved_seeds(self, self.saved_gaussian_seeds)
                self.strategy.freeze_surfel_geometry(self)
        def densification_post_backward_callback(step: int):
            densification_params = self.get_densification_params(step)
            if densification_params is not None:
                densification_info = self.info["surfels"] if step <= 10000 else self.info["gaussians"]
                densification_state = self.strategy_state["surfels"] if step <= 10000 else self.strategy_state["gaussians"]
                densification_optimizers = self.get_densification_optimizers(step)
                self.strategy.step_post_backward(
                    params=densification_params,
                    optimizers=densification_optimizers,
                    state=densification_state,
                    step=step,
                    info=densification_info,
                )
                self.save_params(step, densification_params)
        def save_milestone_callback(step: int):
            if step in [9999,10001, 15000-1, 15000+1, 17500, 20000-1, 20000+1]:
                filename = f"milestone_{step}.pth"
                torch.save(self.state_dict(), filename)
                print(f"Saved model state at milestone iteration {step} to {filename}")
                
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
                iters=tuple([10000]),
                func=phase_10k_callback,
            )
        )
        callbacks.append(
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                iters=tuple([15000]),
                func=phase_15k_callback,
            )
        )
        callbacks.append(
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                iters=tuple([18000, 19000]),
                func=phase_18k_19k_callback,
            )
        )
        callbacks.append(
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                iters=tuple([20000]),
                func=phase_20k_callback,
            )
        )
        
        # Add another callback for the tau ramp logic
        # ...
        
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
                background_color = self.background_color
        elif self.config.background_color == "white":
            background_color = torch.ones(3, device=self.device)
        elif self.config.background_color == "black":
            background_color = torch.zeros(3, device=self.device)
        else:
            raise ValueError(f"Invalid background color option: {self.config.background_color}")
        return background_color
    @staticmethod
    def get_empty_outputs(width: int, height: int, background: torch.Tensor) -> Dict[str, Union[torch.Tensor, List]]:
        rgb = background.repeat(height, width, 1)
        depth = background.new_ones(*rgb.shape[:2], 1) * 10
        accumulation = background.new_zeros(*rgb.shape[:2], 1)
        return {"rgb": rgb, "depth": depth, "accumulation": accumulation, "background": background}
    
    def get_outputs(self, camera: Cameras) -> Dict[str, Union[torch.Tensor, List]]:
        """
        The core rendering logic for a given camera.
        """
        if not isinstance(camera, Cameras):
            print("Called get_outputs with not a camera")
            return {}

        if self.training:
            assert camera.shape[0] == 1, "Only one camera at a time"
            optimized_camera_to_world = self.camera_optimizer.apply_to_camera(camera)
        else:
            optimized_camera_to_world = camera.camera_to_worlds

        gaussian_crop_ids = None
        surfel_crop_ids = None
        # cropping
        if self.crop_box is not None and not self.training:
            gaussian_crop_ids = self.crop_box.within(self.gaussian.means).squeeze()
            surfel_crop_ids = self.crop_box.within(self.surfel.means).squeeze()
            if gaussian_crop_ids.sum() == 0 and surfel_crop_ids.sum() == 0:
                return self.get_empty_outputs(int(camera.width.item()), int(camera.height.item()), self.background_color)

        surfel_crop = self.surfel if surfel_crop_ids is None else Surfel(
            means=Parameter(self.surfel.means[surfel_crop_ids]),
            quats=Parameter(self.surfel.quats[surfel_crop_ids]),
            scales=Parameter(self.surfel.scales[surfel_crop_ids]),
            opacities=Parameter(self.surfel.opacities[surfel_crop_ids]),
            features_dc=Parameter(self.surfel.features_dc[surfel_crop_ids]),
            features_rest=Parameter(self.surfel.features_rest[surfel_crop_ids])
        )
        gaussian_crop = self.gaussian if gaussian_crop_ids is None else Gaussian(
            means=Parameter(self.gaussian.means[gaussian_crop_ids]),
            quats=Parameter(self.gaussian.quats[gaussian_crop_ids]),
            scales=Parameter(self.gaussian.scales[gaussian_crop_ids]),
            opacities=Parameter(self.gaussian.opacities[gaussian_crop_ids]),
            features_dc=Parameter(self.gaussian.features_dc[gaussian_crop_ids]),
            features_rest=Parameter(self.gaussian.features_rest[gaussian_crop_ids])
        )
        surfel_color_crop = torch.cat((surfel_crop.features_dc[:,None,:], surfel_crop.features_rest), dim=1)
        gaussian_color_crop = torch.cat((gaussian_crop.features_dc[:,None,:], gaussian_crop.features_rest), dim=1)
        
        camera_scale_fac = self._get_downscale_factor()
        camera.rescale_output_resolution(1/camera_scale_fac)
        veiwmat = get_viewmat(optimized_camera_to_world)
        intrinsic_mat = camera.get_intrinsics_matrices()
        
        intrinsic_mat=intrinsic_mat.to(self.device)
        height = int(camera.height.item())
        width = int(camera.width.item())
        camera.rescale_output_resolution(camera_scale_fac)
        
        if self.config.output_depth_during_training or not self.training:
            render_mode = "RGB+ED"
        else:
            render_mode = "RGB"
        if self.config.sh_degree > 0:
            sh_degree_to_use = min(self.step // self.config.sh_degree_interval, self.config.sh_degree)
        else:
            surfel_color_crop = torch.sigmoid(surfel_color_crop).squeeze(1)  # [N, 1, 3] -> [N, 3]
            gaussian_color_crop = torch.sigmoid(gaussian_color_crop).squeeze(1)  # [N, 1, 3] -> [N, 3]
            sh_degree_to_use = None
        # rasterization of surfels
        surfel_rgb, surfel_alpha, _, _,_,_, surfel_info = rasterization_2dgs(
            means=surfel_crop.means,
            quats=surfel_crop.quats,
            scales=surfel_crop.scales,
            opacities=surfel_crop.opacities,
            colors=surfel_color_crop,
            viewmats=veiwmat,
            Ks=intrinsic_mat,
            width=width,
            height=height,
            packed=False,
            render_mode=render_mode,
            sh_degree=sh_degree_to_use,
        )
        
            
        if self.gaussian.means.shape[0] > 0:
            gaussian_render, gaussian_alpha, gaussian_info = rasterization(
                means=gaussian_crop.means,
                quats=gaussian_crop.quats,
                scales=gaussian_crop.scales,
                opacities=gaussian_crop.opacities,
                colors=gaussian_color_crop,
                viewmats=veiwmat,
                Ks=intrinsic_mat,
                width=width,
                height=height,
                packed=False,
                render_mode=render_mode,
                sh_degree=sh_degree_to_use,
                rasterize_mode=self.config.rasterize_mode,
            )
        else:
            gaussian_render = torch.zeros_like(surfel_rgb)
            gaussian_alpha = torch.zeros_like(surfel_alpha)
            gaussian_info = None
        self.info = {
            "surfels": surfel_info,
            "gaussians": gaussian_info
        }
        if self.training:
            if self.step <= 15000:
            
                surfel_radii = surfel_info["radii"].detach().max(dim=-1).values.squeeze(0)
                if self.strategy_state["surfels"]["radii"] is None:
                    self.strategy_state["surfels"]["radii"] = surfel_radii
                else:
                    self.strategy_state["surfels"]["radii"] = torch.maximum(self.strategy_state["surfels"]["radii"], surfel_radii)
                
                
            
            densification_params = self.get_densification_params(self.step)
            if densification_params is not None:
                densification_info = self.info["surfels"] if self.step <= 10000 else self.info["gaussians"]
                densification_state = self.strategy_state["surfels"] if self.step <= 10000 else self.strategy_state["gaussians"]
                densification_optimizers = self.get_densification_optimizers(self.step)
                self.strategy.step_pre_backward(
                    params = densification_params,
                    optimizers = densification_optimizers,
                    state = densification_state,
                    step = self.step,
                    info = densification_info,
                )
            
        gaussian_alpha = gaussian_alpha[:,...]
        
        background_color = self._get_background_color()
        
        gaussian_rgb = gaussian_render[:,...,:3]
        gaussian_rgb = torch.clamp(gaussian_rgb, 0.0, 1.0)
        surfel_rgb = torch.clamp(surfel_rgb, 0.0, 1.0)
        
        #eq. 5 compositing
        C_S = surfel_rgb.squeeze(0)
        W_S = surfel_alpha.squeeze(0)
        C_G = gaussian_rgb.squeeze(0)
        W_G = gaussian_alpha.squeeze(0)
        
        rgb =(C_S + C_G) / (W_S + W_G + 1e-5)
        
        final_alpha = torch.clamp(W_S + W_G, 0.0,1.0)
        rgb = rgb + (1 - final_alpha) * background_color
        rgb = torch.clamp(rgb, 0.0, 1.0)
        
        if render_mode == "RGB+ED":
            gaussian_depth = gaussian_render[:,...,3:4].squeeze(0)
            surfel_depth = surfel_rgb[:,...,3:4].squeeze(0)
            depth = (gaussian_depth * W_G + surfel_depth * W_S) / (W_S + W_G + 1e-5)
            depth = torch.where(gaussian_alpha > 0, depth, depth.detach().max())
        else:
            depth = None
        if background_color.shape[0] == 3 and not self.training:
            background_color = background_color.expand(height, width, 3)
        return {
            "rgb": rgb,
            "depth": depth, # type: ignore
            "accumulation": final_alpha,
            "background": background_color
        }
    def get_gt_img(self, image: torch.Tensor):
        """Compute groundtruth image with iteration dependent downscale factor for evaluation purpose

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
    def get_metrics_dict(self, outputs, batch) -> Dict[str, torch.Tensor]:
        """Compute and returns metrics.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
        """
        gt_rgb = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
        metrics_dict = {}
        predicted_rgb = outputs["rgb"]

        metrics_dict["psnr"] = self.psnr(predicted_rgb, gt_rgb)
        if self.config.color_corrected_metrics:
            cc_rgb = color_correct(predicted_rgb, gt_rgb)
            metrics_dict["cc_psnr"] = self.psnr(cc_rgb, gt_rgb)

        metrics_dict["gaussian_count"] = self.num_points

        self.camera_optimizer.get_metrics_dict(metrics_dict)
        return metrics_dict
    def set_crop(self, crop_box: Optional[OrientedBox]):
        self.crop_box = crop_box
    @torch.no_grad()
    def get_outputs_for_camera(self, camera: Cameras, obb_box: Optional[OrientedBox] = None) -> Dict[str, torch.Tensor]:
        """Takes in a camera, generates the raybundle, and computes the output of the model.
        Overridden for a camera-based gaussian model.

        Args:
            camera: generates raybundle
        """
        assert camera is not None, "must provide camera to gaussian model"
        self.set_crop(obb_box)
        outs = self.get_outputs(camera.to(self.device))
        return outs  # type: ignore

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        """Computes and returns the losses dict.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
            metrics_dict: dictionary of metrics, some of which we can use for loss
        """
        gt_img = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
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
        simloss = 1 - self.ssim(gt_img.permute(2, 0, 1)[None, ...], pred_img.permute(2, 0, 1)[None, ...])
        
        #TODO: differentiate the scales for surfel and gaussian
        if self.config.use_scale_regularization and self.step % 10 == 0:
            surfel_scale_exp = torch.exp(self.surfel.scales)
            surfel_scale_reg = (
                torch.maximum(
                    surfel_scale_exp.amax(dim=-1) / surfel_scale_exp.amin(dim=-1),
                    torch.tensor(self.config.max_gauss_ratio),
                )
                - self.config.max_gauss_ratio
            )
            gaussian_scale_exp = torch.exp(self.gaussian.scales)
            gaussian_scale_reg = (
                torch.maximum(
                    gaussian_scale_exp.amax(dim=-1) / gaussian_scale_exp.amin(dim=-1),
                    torch.tensor(self.config.max_gauss_ratio),
                )
                - self.config.max_gauss_ratio
            )
            
            scale_reg = 0.1 * (surfel_scale_reg.mean()+gaussian_scale_reg.mean())
        else:
            scale_reg = torch.tensor(0.0).to(self.device)

        loss_dict = {
            "main_loss": (1 - self.config.ssim_lambda) * Ll1 + self.config.ssim_lambda * simloss,
            "scale_reg": scale_reg,
        }
        if self.training:
            # Add loss from camera optimizer
            self.camera_optimizer.get_loss_dict(loss_dict)
        return loss_dict
