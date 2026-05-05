import os
import json
import time
import tempfile
import torch # type: ignore
from typing import Dict, Any

# Mocking author's code structure
# from external_code.ges_author_code.optimizer import GESOptimizer

class AtomicWriter:
    """Helper to write a file atomically using a temporary file and os.replace."""
    def __init__(self, filepath, mode="w"):
        self.filepath = filepath
        self.mode = mode
        self.temp_file = None

    def __enter__(self):
        dir_name = os.path.dirname(self.filepath)
        self.temp_file = tempfile.NamedTemporaryFile(
            self.mode, dir=dir_name, delete=False, suffix=".tmp"
        )
        return self.temp_file

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.temp_file.close()
        if exc_type is None:
            os.replace(self.temp_file.name, self.filepath)
        else:
            if os.path.exists(self.temp_file.name):
                os.remove(self.temp_file.name)

def save_checkpoint(directory: str, iteration: int, state: Dict[str, Any], metadata: Dict[str, Any]):
    """Saves a checkpoint binary and JSON metadata atomically."""
    os.makedirs(directory, exist_ok=True)
    
    # 1. Save metadata JSON
    json_path = os.path.join(directory, f"iter_{iteration:05d}.json")
    with AtomicWriter(json_path) as f:
        json.dump(metadata, f, indent=4)
        
    # 2. Save raw state (Surfels + Gaussians)
    # This will be processed by pack_for_web.py later
    bin_path = os.path.join(directory, f"iter_{iteration:05d}.raw.pt")
    torch.save(state, bin_path)

def run_optimization(dataset_path: str, output_dir: str, resume_iter: int = 0):
    """
    Main driver for the GES optimization.
    
    Phases:
    1. Standard (0-10k): Joint Surfel + Gaussian optimization.
    2. Discard (10k): One-time pruning of Gaussians far from Surfels.
    3. Transition (10k-18k): Normal training.
    4. Ramp (18k-20k): Shrink Gaussians toward Surfels (Linear interpolation of parameters).
    5. Fine-tune (20k-30k): Final pass.
    """
    print(f"Starting GES optimization on {dataset_path}...")
    # optimizer = GESOptimizer(dataset_path)

    for iteration in range(resume_iter, 30001):
        # --- PHASE: Discard (Eq. 7 in paper) ---
        if iteration == 10000:
            print("Executing Discard Phase: Pruning detached primitives...")
            # state = optimizer.get_state()
            # mask = calculate_discard_mask(state['surfels'], state['gaussians'], threshold=0.1)
            # optimizer.prune(mask)

        # --- PHASE: Ramp (Eq. 8 in paper) ---
        if 18000 <= iteration <= 20000:
            # Linear ramp for tau coefficient
            tau = (iteration - 18000) / 2000.0
            # optimizer.apply_ramp(tau)
            pass

        # --- Standard Step ---
        # loss, metrics = optimizer.step()
        
        # --- Checkpoint logic ---
        if iteration % 1000 == 0 or iteration == 10000 or iteration == 20000:
            # save_checkpoint(output_dir, iteration, optimizer.get_state(), {"loss": 0})
            pass

def calculate_discard_mask(surfels: Dict, gaussians: Dict, threshold: float) -> torch.Tensor:
    """
    Calculates a boolean mask to prune primitives.
    Logic: Prune if distance between surfel center and gaussian center > threshold.
    """
    # dist = torch.norm(surfels['pos'] - gaussians['pos'], dim=-1)
    # return dist < threshold
    return torch.ones(surfels['pos'].shape[0], dtype=torch.bool)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--output', type=str, default="data/optim_run")
    args = parser.parse_args()
    
    run_optimization(args.dataset, args.output)
