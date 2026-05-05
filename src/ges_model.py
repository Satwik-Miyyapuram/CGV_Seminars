"""
ges_model.py
Contains the custom PyTorch Model defining the GES logic.
"""
import torch
from torch.nn import Parameter
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Type

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation

# Assuming gsplat is in external_code/gsplat and accessible
from gsplat.rendering import rasterization

@dataclass
class GESModelConfig(ModelConfig):
    """Configuration for the GES Model."""
    _target: Type = field(default_factory=lambda: GESModel)
    num_initial_points: int = 100000
    # Add other hyperparameters like threshold for discard phase, ramp start/end iter, etc.
    tau_ramp_start: int = 18000
    tau_ramp_end: int = 20000

class GESModel(Model):
    """
    Gaussian-Surfel Model extending Nerfstudio's base Model.
    """
    config: GESModelConfig

    def populate_modules(self):
        """
        Called during initialization.
        Initialize your Surfels and Gaussians here based on the seed point cloud.
        """
        # TODO: Load initial point cloud from self.kwargs["seed_points"] if provided by datamanager
        
        # Skeletons for Surfel parameters (2D)
        self.surfel_means = Parameter(torch.zeros((self.config.num_initial_points, 3)))
        self.surfel_quats = Parameter(torch.zeros((self.config.num_initial_points, 4))) # Initialize to identity
        self.surfel_scales = Parameter(torch.zeros((self.config.num_initial_points, 2))) # 2D scale
        self.surfel_features_dc = Parameter(torch.zeros((self.config.num_initial_points, 3)))
        
        # Skeletons for Gaussian parameters (3D)
        self.gaussian_means = Parameter(torch.zeros((self.config.num_initial_points, 3)))
        self.gaussian_quats = Parameter(torch.zeros((self.config.num_initial_points, 4)))
        self.gaussian_scales = Parameter(torch.zeros((self.config.num_initial_points, 3)))
        self.gaussian_opacities = Parameter(torch.zeros((self.config.num_initial_points, 1)))
        
        # We need to return a dictionary of parameters for the optimizer
        pass

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """
        Group parameters for the optimizer specified in config.py.
        """
        return {
            "means": [self.surfel_means, self.gaussian_means],
            "quats": [self.surfel_quats, self.gaussian_quats],
            "scales": [self.surfel_scales, self.gaussian_scales],
            "opacities": [self.gaussian_opacities],
            "features_dc": [self.surfel_features_dc],
            "features_rest": [],
        }

    def get_training_callbacks(self, training_callback_attributes: TrainingCallbackAttributes) -> List[TrainingCallback]:
        """
        Register callbacks for the Discard Phase (Iter 10k) and Ramp Phase (Iter 18k-20k).
        """
        callbacks = []
        
        def discard_phase_callback(step: int):
            if step == 10000:
                print("Running Discard Phase...")
                # TODO: Implement distance-based pruning (Eq. 7)
                pass

        callbacks.append(
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                iters=[10000],
                func=discard_phase_callback,
            )
        )
        
        # Add another callback for the tau ramp logic
        # ...
        
        return callbacks

    def get_outputs(self, camera: Cameras) -> Dict[str, torch.Tensor]:
        """
        The core rendering logic for a given camera.
        """
        # TODO: Implement Two-Pass Rendering logic using gsplat
        
        # 1. Rasterize Surfels (2D) to get depth and color
        # surfel_outputs = rasterization(
        #     self.surfel_means, self.surfel_quats, self.surfel_scales, ...
        # )
        
        # 2. Rasterize Gaussians (3D) to get color, blending based on surfel depth
        # gaussian_outputs = rasterization(
        #     self.gaussian_means, ...
        # )
        
        # 3. Composite (Eq. 5)
        # final_color = surfel_color * (1 - gaussian_alpha) + gaussian_color
        
        # Mock output for now
        height = camera.height.item()
        width = camera.width.item()
        rgb = torch.zeros((height, width, 3), device=self.device)
        
        return {
            "rgb": rgb,
            "depth": torch.zeros((height, width, 1), device=self.device)
        }

    def get_metrics_dict(self, outputs, batch) -> Dict[str, torch.Tensor]:
        """Compute metrics for evaluation."""
        return {}

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        """
        Compute training losses.
        """
        # TODO: Implement L1 loss, D-SSIM loss, Depth regularizers
        return {"main_loss": torch.tensor(0.0, device=self.device, requires_grad=True)}
