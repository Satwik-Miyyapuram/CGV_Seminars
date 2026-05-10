"""
main.py
Entry point for training the GES model using Nerfstudio's trainer.
"""
import sys
import torch
from external_code.nerfstudio.nerfstudio.scripts.train import main as ns_train_main
from external_code.nerfstudio.nerfstudio.engine.trainer import TrainerConfig
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
    print(f"Using configuration: {cfg}")
    cfg = tyro.cli(TrainerConfig, default=cfg)
    sys.exit(ns_train_main(cfg))

if __name__ == "__main__":
    main()
