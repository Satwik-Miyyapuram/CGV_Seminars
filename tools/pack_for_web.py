from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData, PlyElement

# Paths (Adjust these as needed)
CHECKPOINT_PATH = "outputs/chair/ges-method/YYYY-MM-DD_HHMMSS/nerfstudio_models/step-000029999.ckpt"
OUT_DIR = Path("web_assets")
OUT_DIR.mkdir(exist_ok=True)

# prim_type values stored in the combined scene.ply
PRIM_TYPE_GAUSSIAN = 0
PRIM_TYPE_SURFEL = 1


def _extract_arrays(tensor_dict, sh_degree):
    """Pull the raw numpy arrays out of a primitive tensor dict.

    Returns (means, scales, quats, opacities, features_dc, features_rest)
    where features_rest is flattened to [N, R] (R may be 0). Rows containing any
    non-finite value (NaN/inf) — degenerate primitives left over from training —
    are dropped so they don't render as exploded garbage.
    """
    means = tensor_dict["means"].detach().cpu().numpy().astype(np.float32)
    scales = tensor_dict["scales"].detach().cpu().numpy().astype(np.float32)
    quats = tensor_dict["quats"].detach().cpu().numpy().astype(np.float32)
    opacities = tensor_dict["opacities"].detach().cpu().numpy().astype(np.float32)
    features_dc = tensor_dict["features_dc"].detach().cpu().numpy().astype(np.float32)

    if sh_degree > 0 and tensor_dict.get("features_rest") is not None:
        features_rest = tensor_dict["features_rest"].detach().cpu().numpy().astype(np.float32)
        # Flatten the features_rest from [N, 15, 3] to [N, 45]
        features_rest = features_rest.reshape(features_rest.shape[0], -1)
    else:
        features_rest = np.zeros((means.shape[0], 0), dtype=np.float32)

    # Drop primitives with any non-finite value (NaN / +-inf).
    finite = (
        np.isfinite(means).all(1)
        & np.isfinite(scales).all(1)
        & np.isfinite(quats).all(1)
        & np.isfinite(opacities.reshape(opacities.shape[0], -1)).all(1)
        & np.isfinite(features_dc).all(1)
        & np.isfinite(features_rest).all(1)
    )
    num_dropped = int((~finite).sum())
    if num_dropped > 0:
        print(f"  Dropping {num_dropped}/{finite.shape[0]} primitives with non-finite values.")
        means = means[finite]
        scales = scales[finite]
        quats = quats[finite]
        opacities = opacities[finite]
        features_dc = features_dc[finite]
        features_rest = features_rest[finite]
    print("opacity stats after dropping non-finite: ")
    print(
        f"  Opacity stats (raw logits): min {opacities.min():.3f}, "
        f"max {opacities.max():.3f}, "
        f"mean {opacities.mean():.3f}"
    )

    return means, scales, quats, opacities, features_dc, features_rest


def _build_dtype(num_rest, include_prim_type):
    """Construct the structured-array dtype for the 3DGS PLY vertex element."""
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
    for i in range(num_rest):
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
    if include_prim_type:
        dtype.append(("prim_type", "u1"))
    return dtype


def _fill_elements(
    elements,
    start,
    means,
    scales,
    quats,
    opacities,
    features_dc,
    features_rest,
    num_rest,
    prim_type=None,
):
    """Fill rows [start:start+N] of a pre-allocated structured array."""
    num_pts = means.shape[0]
    sl = slice(start, start + num_pts)

    elements["x"][sl], elements["y"][sl], elements["z"][sl] = means.T
    elements["nx"][sl], elements["ny"][sl], elements["nz"][sl] = np.zeros_like(means).T
    elements["f_dc_0"][sl], elements["f_dc_1"][sl], elements["f_dc_2"][sl] = features_dc.T

    # Pad/clip features_rest to the target column count so surfels and gaussians
    # share one dtype even if one set lacks SH rest coefficients.
    for i in range(num_rest):
        if i < features_rest.shape[1]:
            elements[f"f_rest_{i}"][sl] = features_rest[:, i]
        else:
            elements[f"f_rest_{i}"][sl] = 0.0

    # Store RAW (logit) opacity.
    # The standard 3DGS PLY convention applies sigmoid on load.
    # The GES paper defines Surfel Alpha as: min(0.99, 255 * sigmoid(raw_opacity) * exp(-0.5*A))
    # To pass `255 * sigmoid(raw_opacity)` to the web viewer through mkkellogg's sigmoid loader,
    # we compute the desired post-sigmoid opacity, cap it at 0.9999, and inverse-sigmoid (logit) it!
    if prim_type == 1:
        # 1. Compute original sigmoid opacity
        sig_opac = 1.0 / (1.0 + np.exp(-opacities))
        # 2. Apply GES 255x multiplier
        sig_opac = sig_opac * 255.0
        # 3. Cap it slightly below 1.0 to prevent inf logits
        sig_opac = np.clip(sig_opac, 1e-6, 0.9999)
        # 4. Inverse sigmoid (logit) it so mkkellogg's loader reproduces this value exactly
        opacities = np.log(sig_opac / (1.0 - sig_opac))

    elements["opacity"][sl] = opacities.squeeze()
    print(
        f"  Opacity stats (raw logits): min {elements['opacity'][sl].min():.3f}, "
        f"max {elements['opacity'][sl].max():.3f}, "
        f"mean {elements['opacity'][sl].mean():.3f}"
    )

    # 3DGS web viewers render everything as 3D ellipsoids. For surfels to look
    # like perfectly flat 2D discs, their local Z-scale must be virtually zero.
    # The PyTorch 2DGS rasterizer ignores the Z-scale internally, but the web
    # viewer reads it from the PLY. We force the Z-scale to log(1e-7) ≈ -16.0 here.
    scale_0, scale_1, scale_2 = scales.T
    if prim_type == 1:  # PRIM_TYPE_SURFEL
        scale_2 = -16.0 * np.ones_like(scale_2)
    elements["scale_0"][sl], elements["scale_1"][sl], elements["scale_2"][sl] = (
        scale_0,
        scale_1,
        scale_2,
    )
    elements["rot_0"][sl], elements["rot_1"][sl], elements["rot_2"][sl], elements["rot_3"][sl] = (
        quats.T
    )

    if prim_type is not None:
        elements["prim_type"][sl] = prim_type


def export_to_ply(tensor_dict, filename, sh_degree=3):
    """Converts a dictionary of splat tensors to a standard 3DGS .ply file"""
    if "means" not in tensor_dict or tensor_dict["means"].shape[0] == 0:
        print(f"Skipping {filename} (0 points)")
        return

    means, scales, quats, opacities, features_dc, features_rest = _extract_arrays(
        tensor_dict, sh_degree
    )
    num_rest = features_rest.shape[1]
    num_pts = means.shape[0]

    elements = np.empty(num_pts, dtype=_build_dtype(num_rest, include_prim_type=False))
    _fill_elements(
        elements, 0, means, scales, quats, opacities, features_dc, features_rest, num_rest
    )

    el = PlyElement.describe(elements, "vertex")
    PlyData([el]).write(filename)
    print(f"Saved {num_pts} points to {filename}")


def export_combined_ply(surfels, gaussians, filename, sh_degree=3):
    """Combine surfels and gaussians into one scene.ply tagged with prim_type.

    prim_type = 1 for surfels, 0 for gaussians. The web viewer splits the file
    back into two standard 3DGS PLYs on load.
    """
    s_arrays = None
    g_arrays = None
    if surfels is not None and surfels.get("means") is not None and surfels["means"].shape[0] > 0:
        s_arrays = _extract_arrays(surfels, sh_degree)
    if (
        gaussians is not None
        and gaussians.get("means") is not None
        and gaussians["means"].shape[0] > 0
    ):
        g_arrays = _extract_arrays(gaussians, sh_degree)

    if s_arrays is None and g_arrays is None:
        print(f"Skipping {filename} (0 points)")
        return

    num_surfels = s_arrays[0].shape[0] if s_arrays is not None else 0
    num_gaussians = g_arrays[0].shape[0] if g_arrays is not None else 0

    # Both primitive types share one dtype; use the widest SH-rest width.
    num_rest = max(
        s_arrays[5].shape[1] if s_arrays is not None else 0,
        g_arrays[5].shape[1] if g_arrays is not None else 0,
    )

    total = num_surfels + num_gaussians
    elements = np.empty(total, dtype=_build_dtype(num_rest, include_prim_type=True))

    offset = 0
    if s_arrays is not None:
        _fill_elements(elements, offset, *s_arrays, num_rest=num_rest, prim_type=PRIM_TYPE_SURFEL)
        offset += num_surfels
    if g_arrays is not None:
        _fill_elements(elements, offset, *g_arrays, num_rest=num_rest, prim_type=PRIM_TYPE_GAUSSIAN)

    el = PlyElement.describe(elements, "vertex")
    PlyData([el]).write(filename)
    print(f"Saved {total} points ({num_surfels} surfels + {num_gaussians} gaussians) to {filename}")


def restrict_primitives(tensor_dict, max_primitives):
    """Restricts the primitives in the dictionary to the top K based on opacity descending"""
    if max_primitives is None or max_primitives <= 0:
        return tensor_dict

    num_pts = tensor_dict["means"].shape[0]
    if num_pts <= max_primitives:
        return tensor_dict

    opacities = tensor_dict["opacities"].squeeze()
    if opacities.ndim == 0:
        return tensor_dict

    opacities_flat = opacities.view(-1)
    # Get top max_primitives indices
    _, top_indices = torch.topk(opacities_flat, k=max_primitives, largest=True, sorted=True)

    restricted_dict = {}
    for k, v in tensor_dict.items():
        if v is not None:
            restricted_dict[k] = v[top_indices]
        else:
            restricted_dict[k] = None
    return restricted_dict


def load_model_state(checkpoint_path):
    """Load a checkpoint and return (state_dict, prefix).

    Supports both formats:
      * nerfstudio `.ckpt`: tensors live under ckpt["pipeline"] with a "_model." prefix.
      * milestone `.pth`:   torch.save(model.state_dict()) with no prefix.
    """
    try:
        obj = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except RuntimeError as e:
        if "central directory" in str(e) or "failed reading zip archive" in str(e):
            size = Path(checkpoint_path).stat().st_size if Path(checkpoint_path).exists() else 0
            raise RuntimeError(
                f"Could not read '{checkpoint_path}' as a torch checkpoint ({size} bytes). "
                "The file looks truncated or corrupt — if a copy/download is still in "
                "progress, wait for it to finish and retry."
            ) from e
        raise

    # Nerfstudio .ckpt wraps the model state under "pipeline".
    if isinstance(obj, dict) and "pipeline" in obj:
        state = obj["pipeline"]
    else:
        state = obj

    if not isinstance(state, dict):
        raise ValueError(
            f"Unrecognized checkpoint structure in {checkpoint_path}: expected a state dict."
        )

    # Detect the key prefix used for the model parameters.
    if "_model.surfel.means" in state:
        prefix = "_model."
    elif "surfel.means" in state:
        prefix = ""
    else:
        raise KeyError(
            f"Could not find surfel parameters in {checkpoint_path}. "
            "Expected keys like '_model.surfel.means' (.ckpt) or 'surfel.means' (.pth)."
        )

    return state, prefix


def extract_primitives(state, prefix, name):
    """Extract a primitive (surfel/gaussian) tensor dict from a state dict."""
    return {
        "means": state[f"{prefix}{name}.means"],
        "quats": state[f"{prefix}{name}.quats"],
        "scales": state[f"{prefix}{name}.scales"],
        "opacities": state[f"{prefix}{name}.opacities"],
        "features_dc": state[f"{prefix}{name}.features_dc"],
        "features_rest": state.get(f"{prefix}{name}.features_rest"),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Pack GES checkpoints (.ckpt or .pth) for web viewer"
    )
    parser.add_argument(
        "--checkpoint",
        "-c",
        type=str,
        default=CHECKPOINT_PATH,
        help="Path to the checkpoint file (.ckpt or .pth)",
    )
    parser.add_argument("--out-dir", "-o", type=str, default=str(OUT_DIR), help="Output directory")
    parser.add_argument(
        "--out-name", type=str, default="scene.ply", help="Combined output PLY filename"
    )
    parser.add_argument(
        "--max-surfels", type=int, default=None, help="Restrict the number of surfels to this count"
    )
    parser.add_argument(
        "--max-gaussians",
        type=int,
        default=None,
        help="Restrict the number of gaussians to this count",
    )
    parser.add_argument(
        "--sh-degree",
        type=int,
        default=3,
        choices=[0, 3],
        help="Spherical harmonics degree to export (0 or 3, default: 3)",
    )
    parser.add_argument(
        "--separate",
        action="store_true",
        help="Also write legacy surfels.ply / gaussians.ply alongside the combined scene.ply",
    )
    args = parser.parse_args()

    checkpoint_path = args.checkpoint
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    print(f"Loading checkpoint: {checkpoint_path}")
    state, prefix = load_model_state(checkpoint_path)
    print(f"Detected key prefix: '{prefix or '(none)'}'")

    surfels = extract_primitives(state, prefix, "surfel")
    gaussians = extract_primitives(state, prefix, "gaussian")

    if args.max_surfels is not None:
        print(f"Restricting surfels to top {args.max_surfels} by opacity...")
        surfels = restrict_primitives(surfels, args.max_surfels)

    if args.max_gaussians is not None:
        print(f"Restricting gaussians to top {args.max_gaussians} by opacity...")
        gaussians = restrict_primitives(gaussians, args.max_gaussians)

    export_combined_ply(surfels, gaussians, out_dir / args.out_name, sh_degree=args.sh_degree)

    if args.separate:
        export_to_ply(surfels, out_dir / "surfels.ply", sh_degree=args.sh_degree)
        export_to_ply(gaussians, out_dir / "gaussians.ply", sh_degree=args.sh_degree)

    print("Done! Ready for Spark.")
