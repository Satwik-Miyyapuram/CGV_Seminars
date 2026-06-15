"""
config.py
Registers the GES custom method with Nerfstudio.
"""
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanagerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig

from nerfstudio.configs.base_config import ViewerConfig

from ges_model import GESModelConfig
base_optimizer_config = {
        "means": AdamOptimizerConfig(lr=1.6e-4, weight_decay=1e-15),
        "quats": AdamOptimizerConfig(lr=1e-3, weight_decay=1e-15),
        "scales": AdamOptimizerConfig(lr=5e-3, weight_decay=1e-15),
        "opacities": AdamOptimizerConfig(lr=5e-2, weight_decay=1e-15),
        "features_dc": AdamOptimizerConfig(lr=2.5e-3, weight_decay=1e-15),
        "features_rest": AdamOptimizerConfig(lr=1.25e-4, weight_decay=1e-15),
    }
# BUG 9 FIX: Add exponential decay schedulers for means parameters.
# Without decay, the means LR stays constant (1.6e-4) throughout training,
# causing geometry jitter in later optimization stages when positions should
# have converged. Splatfacto uses exponential decay to 1/100th of the
# initial LR by the end of training.
ges_optimizers = {}
for name, optimizer in base_optimizer_config.items():
    if name == "means":
        # Means get LR decay: start at 1.6e-4, decay to 1.6e-6 over training
        scheduler = ExponentialDecaySchedulerConfig(
            lr_final=1.6e-6,
            max_steps=30000,
        )
    else:
        scheduler = None

    if name == "opacities":
        # Surfel opacities are scaled by 255.0 in the CUDA shader (representing tau in [0, 255]).
        # To prevent boundary-only gradient shrinkage from aggressively pruning all surfels
        # before step 500, we use a 5x smaller learning rate of 0.01 for surfel opacities.
        surfel_opt = AdamOptimizerConfig(lr=1e-2, weight_decay=1e-15)
    else:
        surfel_opt = optimizer

    ges_optimizers[f'surfel_{name}'] = {"optimizer": surfel_opt, "scheduler": scheduler}
    ges_optimizers[f'gaussian_{name}'] = {"optimizer": optimizer, "scheduler": scheduler}
ges_method = MethodSpecification(
    config=TrainerConfig(
        method_name="ges-method",
        steps_per_eval_image=100,
        steps_per_eval_batch=0,
        steps_per_save=2000,
        steps_per_eval_all_images=1000,
        max_num_iterations=30000,
        mixed_precision=False,
        pipeline=VanillaPipelineConfig(
            datamanager=FullImageDatamanagerConfig(
                dataparser=NerfstudioDataParserConfig(),
            ),
            model=GESModelConfig(),
        ),
        optimizers=ges_optimizers,
        viewer=ViewerConfig(), # Use default viewer config
    ),
    description="Gaussian-Surfel representation built on gsplat",
)
