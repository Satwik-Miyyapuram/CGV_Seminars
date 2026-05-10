from dataclasses import dataclass
from typing import Any, Dict, Tuple, Union

import torch
from typing_extensions import Literal
from torch.nn import Parameter
from external_code.gsplat.gsplat.strategy.default import DefaultStrategy
from external_code.gsplat.gsplat.strategy.ops import _update_param_with_optimizer
from external_code.nerfstudio.nerfstudio.utils.math import random_quat_tensor
from external_code.nerfstudio.nerfstudio.utils.spherical_harmonics import num_sh_bases
from src.ges_model import GESModel


@dataclass
class GESStrategy(DefaultStrategy):
    """GES strategy for the GS densification.

    This class implements the GES strategy for the GS densification. It defines
    the operations to be performed before and after the `loss.backward()` call.
    """

    
    def initialize_state(self, scene_scale: float = 1.0) -> Dict[str, Any]:
        """Initialize the strategy state."""
        state = {
            "surfels":{
                "radii": None,  # This will be updated during training with the actual radii of the surfels
                "grad2d": None,  
                "count": None,
                "scene_scale": scene_scale
            },
            "gaussians":{
                "radii": None,  # This will be updated during training with the actual radii of the gaussians
                "grad2d": None,
                "count": None,
                "scene_scale": scene_scale
            }
        }
        return state
    
    
    def execute_10k_discard(self, model: GESModel, keep_mask: torch.Tensor):
        """Callback function to be executed at the 10k iteration to discard the points based on the keep_mask."""
        def param_fn(name:str, p: torch.Tensor)-> torch.Tensor:
            return Parameter(p[keep_mask], requires_grad=p.requires_grad)
        def optimizer_fn(key:str, v:torch.Tensor)-> torch.Tensor:
            return v[keep_mask]
        surfel_param_dict = model.get_surfel_param_dict()
        _update_param_with_optimizer(param_fn, optimizer_fn, surfel_param_dict, model.optimizers)
        model.surfel.means = surfel_param_dict["surfel_means"]
        model.surfel.quats = surfel_param_dict["surfel_quats"]
        model.surfel.scales = surfel_param_dict["surfel_scales"]
        model.surfel.opacities = surfel_param_dict["surfel_opacities"]
        model.surfel.features_dc = surfel_param_dict["surfel_features_dc"]
        model.surfel.features_rest = surfel_param_dict["surfel_features_rest"]
    
    def execute_15k_visibility_prune(self, model: GESModel):
        """Callback function to be executed at the 15k iteration to prune the points based on visibility."""
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
        def param_fn(name:str, p: torch.Tensor)-> torch.Tensor:
            return Parameter(p[visibility_mask], requires_grad=p.requires_grad)
        def optimizer_fn(key:str, v:torch.Tensor)-> torch.Tensor:
            return v[visibility_mask]
        surfel_param_dict = model.get_surfel_param_dict()
        _update_param_with_optimizer(param_fn, optimizer_fn, surfel_param_dict, model.optimizers)
        model.surfel.means = surfel_param_dict["surfel_means"]
        model.surfel.quats = surfel_param_dict["surfel_quats"]
        model.surfel.scales = surfel_param_dict["surfel_scales"]
        model.surfel.opacities = surfel_param_dict["surfel_opacities"]
        model.surfel.features_dc = surfel_param_dict["surfel_features_dc"]
        model.surfel.features_rest = surfel_param_dict["surfel_features_rest"]
        
    def clamp_surfel_opacity(self, model: GESModel, min_opacity: float):
        """Callback function to be executed when the surfel opacity needs to be clamped."""
        target_prob = min(min_opacity / 255.0, 0.9999)
        target_logit = torch.logit(torch.tensor(target_prob, device=model.device))
        model.surfel.opacities.data = torch.clamp_min(model.surfel.opacities.data, target_logit)
    
    def freeze_surfel_geometry(self, model: GESModel):
        """Callback function to be executed at the 20k iteration to freeze the surfel geometry."""
        model.surfel.means.requires_grad_(False)
        model.surfel.quats.requires_grad_(False)
        model.surfel.scales.requires_grad_(False)
        model.surfel.opacities.requires_grad_(False)
        model.surfel.means.grad = None
        model.surfel.quats.grad = None
        model.surfel.scales.grad = None
        model.surfel.opacities.grad = None
    
    def spawn_gaussians_from_saved_seeds(self, model: GESModel, saved_gaussian_seeds: torch.Tensor):
        """Callback function to be executed at the 20k iteration to spawn gaussians from the saved seeds."""
        num_new_gaussians = saved_gaussian_seeds.shape[0]
        device = model.device
        
        new_data = {
            "gaussian_means": saved_gaussian_seeds.clone(),
            "gaussian_quats": random_quat_tensor(num_new_gaussians).to(device),
            "gaussian_scales": torch.ones((num_new_gaussians, 3), device=device) * -2.0,  # Initialize scales to a small value (log scale)
            "gaussian_opacities": torch.logit(0.1 * torch.ones((num_new_gaussians, 1), device=device)),  # Initialize opacities to a low value
            "gaussian_features_dc": torch.zeros((num_new_gaussians,3), device=device),  # Initialize DC features to zero
            "gaussian_features_rest": torch.zeros((num_new_gaussians, num_sh_bases(model.config.sh_degree) -1,3), device=device),  # Initialize SH features to zero
        }
        
        def param_fn(name:str, p: torch.Tensor)-> torch.Tensor:
            return Parameter(new_data[name], requires_grad=p.requires_grad)
        
        def optimizer_fn(key:str, v:torch.Tensor)-> torch.Tensor:
            return torch.zeros((num_new_gaussians, *v.shape[1:]), device=device)
        
        gausssian_params_dict = model.get_gaussian_param_dict()
        _update_param_with_optimizer(param_fn, optimizer_fn, gausssian_params_dict, model.optimizers)
        model.gaussian.means = gausssian_params_dict["gaussian_means"]
        model.gaussian.quats = gausssian_params_dict["gaussian_quats"]
        model.gaussian.scales = gausssian_params_dict["gaussian_scales"]
        model.gaussian.opacities = gausssian_params_dict["gaussian_opacities"]
        model.gaussian.features_dc = gausssian_params_dict["gaussian_features_dc"]
        model.gaussian.features_rest = gausssian_params_dict["gaussian_features_rest"]
        print(f"Spawned {num_new_gaussians} new gaussians from saved seeds at iteration {model.step}.")
        
        