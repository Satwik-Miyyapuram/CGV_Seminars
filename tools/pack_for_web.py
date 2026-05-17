from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData, PlyElement

# Paths (Adjust these as needed)
CHECKPOINT_PATH = "outputs/chair/ges-method/YYYY-MM-DD_HHMMSS/nerfstudio_models/step-000029999.ckpt"
OUT_DIR = Path("web_assets")
OUT_DIR.mkdir(exist_ok=True)


def export_to_ply(tensor_dict, filename):
    """Converts a dictionary of splat tensors to a standard 3DGS .ply file"""
    if "means" not in tensor_dict or tensor_dict["means"].shape[0] == 0:
        print(f"Skipping {filename} (0 points)")
        return

    means = tensor_dict["means"].detach().cpu().numpy()
    scales = tensor_dict["scales"].detach().cpu().numpy()
    quats = tensor_dict["quats"].detach().cpu().numpy()
    opacities = tensor_dict["opacities"].detach().cpu().numpy()

    # 3DGS expects spherical harmonics. We extract DC (base color) and Rest(view-dependent)
    features_dc = tensor_dict["features_dc"].detach().cpu().numpy()

    if "features_rest" in tensor_dict:
        features_rest = tensor_dict["features_rest"].detach().cpu().numpy()
        # Flatten the features_rest from [N, 15, 3] to [N, 45]
        features_rest = features_rest.reshape(features_rest.shape[0], -1)
    else:
        features_rest = np.zeros((means.shape[0], 45))

    num_pts = means.shape[0]

    # Standard 3DGS PLY format arrays
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("f_dc_0", "f4"),
        ("f_dc_1", "f4"),
        ("f_dc_2", "f4"),
    ]

    for i in range(features_rest.shape[1]):
        dtype.append((f"f_rest_{i}", "f4"))

    dtype.extend(
        [
            ("opacity", "f4"),
            ("scale_0", "f4"),
            ("scale_1", "f4"),
            ("scale_2", "f4"),
            ("rot_0", "f4"),
            ("rot_1", "f4"),
            ("rot_2", "f4"),
            ("rot_3", "f4"),
        ]
    )

    elements = np.empty(num_pts, dtype=dtype)
    elements["x"], elements["y"], elements["z"] = means.T
    elements["nx"], elements["ny"], elements["nz"] = np.zeros_like(means).T
    elements["f_dc_0"], elements["f_dc_1"], elements["f_dc_2"] = features_dc.T

    if features_rest.shape[1] > 0:
        for i in range(features_rest.shape[1]):
            elements[f"f_rest_{i}"] = features_rest[:, i]

    elements["opacity"] = opacities.squeeze()
    elements["scale_0"], elements["scale_1"], elements["scale_2"] = scales.T
    elements["rot_0"], elements["rot_1"], elements["rot_2"], elements["rot_3"] = quats.T

    el = PlyElement.describe(elements, "vertex")
    PlyData([el]).write(filename)
    print(f"Saved {num_pts} points to {filename}")


if __name__ == "__main__":
    print(f"Loading checkpoint: {CHECKPOINT_PATH}")
    # Load the checkpoint (map_location='cpu' so we can run this anywhere)
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")

    # Nerfstudio wraps the model inside "pipeline" -> "_model"
    model_state = ckpt["pipeline"]

    # Extract Surfel Parameters
    surfels = {
        "means": model_state["_model.surfel.means"],
        "quats": model_state["_model.surfel.quats"],
        "scales": model_state["_model.surfel.scales"],
        "opacities": model_state["_model.surfel.opacities"],
        "features_dc": model_state["_model.surfel.features_dc"],
        "features_rest": model_state.get("_model.surfel.features_rest"),
    }

    # Extract Gaussian Parameters
    gaussians = {
        "means": model_state["_model.gaussian.means"],
        "quats": model_state["_model.gaussian.quats"],
        "scales": model_state["_model.gaussian.scales"],
        "opacities": model_state["_model.gaussian.opacities"],
        "features_dc": model_state["_model.gaussian.features_dc"],
        "features_rest": model_state.get("_model.gaussian.features_rest"),
    }

    export_to_ply(surfels, OUT_DIR / "surfels.ply")
    export_to_ply(gaussians, OUT_DIR / "gaussians.ply")
    print("Done! Ready for Spark.")
