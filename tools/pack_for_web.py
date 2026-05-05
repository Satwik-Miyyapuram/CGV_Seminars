import numpy as np
import zstandard as zstd
import torch # type: ignore
import os
import glob
from typing import Dict

def quantize_sh(sh_coeffs: np.ndarray) -> np.ndarray:
    """Quantizes float32 spherical harmonics to float16 for storage."""
    return sh_coeffs.astype(np.float16)

def pack_checkpoint(output_path: str, surfels: Dict[str, np.ndarray], gaussians: Dict[str, np.ndarray]):
    """
    Packs positions, rotations, scales, and SH coefficients into an interleaved binary file.
    
    GPU-Optimized Interleaved Layout (Per Primitive):
    - [0:12]   Position (float32 x 3)
    - [12:20]  Rotation (float16 x 4)  <-- Quantized
    - [20:26]  Scale    (float16 x 3)  <-- Quantized
    - [26:28]  Opacity  (float16 x 1)  <-- Quantized
    - [28:...] SH Coeffs (float16 x 48) <-- Degree 3, Quantized
    """
    num_n = surfels['pos'].shape[0]
    
    # Pre-quantize all attributes
    pos = surfels['pos'].astype(np.float32)
    rot = surfels['rot'].astype(np.float16)
    scale = surfels['scale'].astype(np.float16)
    opacity = gaussians['opacity'].astype(np.float16)
    sh = quantize_sh(surfels['sh']) # N x 48 (16 * 3)
    
    # Create the interleaved buffer
    # Calculate bytes per primitive (12 + 8 + 6 + 2 + 96 = 124 bytes)
    # Note: 124 is not power-of-2, check if alignment is needed for your shader
    packed_data = bytearray()
    
    for i in range(num_n):
        packed_data.extend(pos[i].tobytes())
        packed_data.extend(rot[i].tobytes())
        packed_data.extend(scale[i].tobytes())
        packed_data.extend(opacity[i].tobytes())
        packed_data.extend(sh[i].tobytes())
    
    # Compress using zstd
    cctx = zstd.ZstdCompressor(level=3)
    compressed = cctx.compress(packed_data)
    
    with open(output_path, 'wb') as f:
        f.write(compressed)
    print(f"Packed {num_n} primitives into {len(compressed)} bytes.")

def process_all_raw_checkpoints(input_dir: str, output_dir: str):
    """Iterates through all .raw.pt files and packs them for the web."""
    os.makedirs(output_dir, exist_ok=True)
    raw_files = glob.glob(os.path.join(input_dir, "*.raw.pt"))
    
    for raw_path in raw_files:
        iter_num = os.path.basename(raw_path).split("_")[1].split(".")[0]
        output_path = os.path.join(output_dir, f"iter_{iter_num}.bin")
        
        # state = torch.load(raw_path)
        # pack_checkpoint(output_path, state['surfels'], state['gaussians'])
        print(f"Packed {raw_path} -> {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()
    
    process_all_raw_checkpoints(args.input, args.output)
