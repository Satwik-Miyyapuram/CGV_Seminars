"""
main.py
Entry point for training the GES model using Nerfstudio's trainer.
"""
import os
os.environ["WANDB_MODE"] = "offline"

from pathlib import Path
import sys

# root_dir = Path(__file__).parent.parent.absolute()
# sys.path.append(str(root_dir))
# sys.path.append(str(root_dir/"external_code"/"nerfstudio"))
# sys.path.append(str(root_dir/"external_code"/"gsplat"))

import torch
from nerfstudio.scripts.train import main as ns_train_main
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.data.dataparsers.blender_dataparser import BlenderDataParserConfig
import tyro


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
    cfg = config.ges_method.config
    cfg.data = Path("nerf_synthetic/chair")
    cfg.pipeline.datamanager.dataparser = BlenderDataParserConfig()
    cfg.logging.writer_names = ["console", "tensorboard"]
    
    # Auto-resume from latest checkpoint if one exists
    base_dir = Path("outputs/chair/ges-method")
    if base_dir.exists():
        # Find all run folders that have a nerfstudio_models folder
        run_folders = [f for f in base_dir.iterdir() if f.is_dir() and (f / "nerfstudio_models").exists()]
        if run_folders:
            # Sort chronologically
            latest_run = sorted(run_folders)[-1]
            ckpt_dir = latest_run / "nerfstudio_models"
            checkpoints = list(ckpt_dir.glob("step-*.ckpt"))
            if checkpoints:
                print(f"Auto-resuming from latest checkpoint in {ckpt_dir}...")
                cfg.load_dir = ckpt_dir
                
    ns_train_main(cfg)

if __name__ == "__main__":
    main()
