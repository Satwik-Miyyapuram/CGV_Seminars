"""
main.py
Entry point for training the GES model using Nerfstudio's trainer.
"""
import os
from pathlib import Path
import sys

# root_dir = Path(__file__).parent.parent.absolute()
# sys.path.append(str(root_dir))
# sys.path.append(str(root_dir/"external_code"/"nerfstudio"))
# sys.path.append(str(root_dir/"external_code"/"gsplat"))

import torch
from nerfstudio.scripts.train import main as ns_train_main
from nerfstudio.engine.trainer import TrainerConfig
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
    ns_train_main(cfg)

if __name__ == "__main__":
    main()
