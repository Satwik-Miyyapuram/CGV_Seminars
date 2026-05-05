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
