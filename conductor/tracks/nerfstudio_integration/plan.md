# Nerfstudio & gsplat Integration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create the boilerplate structure and Python function skeletons for a custom Nerfstudio method (`GESModel`) that utilizes `gsplat` for rendering Surfels and Gaussians.

**Architecture:** We will create a standalone `src/` Python module containing the custom Model, Configuration, and a runner script (`main.py`). The `GESModel` will handle the specific parameter initialization and two-pass rendering logic defined in the GES paper, while relying on Nerfstudio's `VanillaPipeline` for data loading and the `Trainer` for the optimization loop.

**Tech Stack:** Python 3.10+, PyTorch, nerfstudio, gsplat

---

### Task 1: Setup the runner script (`main.py`)

**Files:**
- Create: `src/main.py`

- [ ] **Step 1: Write the minimal implementation**

```python
"""
main.py
Entry point for training the GES model using Nerfstudio's trainer.
"""
import sys
import torch
from nerfstudio.scripts.train import main as ns_train_main

# Import our custom config so it gets registered with Nerfstudio
import config

def main():
    """
    To run this:
    python src/main.py ges-method --data <path-to-data>
    """
    print("Initializing GES Training via Nerfstudio...")
    
    # We pass the arguments to nerfstudio's train script.
    # The 'ges-method' matches the method_name in config.py
    sys.exit(ns_train_main())

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add src/main.py
git commit -m "feat(nerfstudio): add main runner script"
```

---

### Task 2: Define the Method Configuration (`src/config.py`)

**Files:**
- Create: `src/config.py`

- [ ] **Step 1: Write the minimal implementation**

```python
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

from ges_model import GESModelConfig

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
        optimizers={
            "means": {
                "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1.6e-6, max_steps=30000),
            },
            "features_dc": {
                "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
                "scheduler": None,
            },
            "features_rest": {
                "optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15),
                "scheduler": None,
            },
            "opacities": {
                "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
                "scheduler": None,
            },
            "scales": {
                "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
                "scheduler": None,
            },
            "quats": {"optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15), "scheduler": None},
        },
        viewer=None, # Use default viewer config
    ),
    description="Gaussian-Surfel representation built on gsplat",
)
```

- [ ] **Step 2: Commit**

```bash
git add src/config.py
git commit -m "feat(nerfstudio): add GES method configuration"
```

---

### Task 3: Create the GES Model Skeletons (`src/ges_model.py`)

**Files:**
- Create: `src/ges_model.py`

- [ ] **Step 1: Write the minimal implementation**

```python
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
```

- [ ] **Step 2: Commit**

```bash
git add src/ges_model.py
git commit -m "feat(nerfstudio): add GESModel skeletons"
```

---

### Task 4: Setup Environment for Nerfstudio Plugins

**Files:**
- Create: `pyproject.toml`

- [ ] **Step 1: Write the minimal implementation**

To make sure Nerfstudio discovers our `ges-method` easily, we expose it as an entry point.

```toml
[project]
name = "ges-explorer"
version = "0.1.0"
dependencies = [
    "nerfstudio",
    "gsplat"
]

[project.entry-points.'nerfstudio.method_configs']
ges-method = "src.config:ges_method"
```

- [ ] **Step 2: Commit**

```bash
git add pyproject.toml
git commit -m "chore(nerfstudio): add pyproject.toml for plugin discovery"
```
