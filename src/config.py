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
ges_optimizers = {}
for name,optimizer in base_optimizer_config.items():
    ges_optimizers[f'surfel_{name}'] = {"optimizer": optimizer, "scheduler": None}
    ges_optimizers[f'gaussian_{name}'] = {"optimizer": optimizer, "scheduler": None}
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
