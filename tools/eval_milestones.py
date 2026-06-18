"""
eval_milestones.py

Post-hoc evaluation of the GES training milestones.

During training, `GESModel.save_milestone_callback` dumps the full model state to
`milestone_<step>.pth` at the iterations listed in `training_schedule.MILESTONE_STEPS`.
This script replays those checkpoints offline to produce two figures used in the report:

  1. A **loss graph** -- the reconstruction L1 loss (the same term the trainer logs to
     `l1_loss_history`) averaged over a set of training views, plotted against iteration.
  2. A **progression composite** -- the model rendered from one fixed camera at each
     milestone, tiled into a single image so the coarse-to-fine sharpening is visible.

It rebuilds the exact pipeline `src/main.py` would (same dataparser auto-detection and
scene settings), so the cameras, ground-truth images, and rendering match training.

Usage:
    python tools/eval_milestones.py --data <path_to_dataset>
    python tools/eval_milestones.py --data nerf_synthetic/chair --milestone-dir outputs/chair/ges-method/<run>

Outputs (under --output-dir, default `web_assets/`):
    milestone_loss_curve.png
    milestone_progression_composite.png
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path

import torch

# src/ holds the model + config modules; make them importable regardless of CWD.
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import config  # noqa: E402  (the GES MethodSpecification)


def _fix_transforms_json(data_path: Path) -> None:
    """Mirror of main.py's transforms patch so the Nerfstudio dataparser accepts the JSON."""
    renames = {"fx": "fl_x", "fy": "fl_y"}
    candidates = [
        "transforms.json",
        "transforms_train.json",
        "transforms_val.json",
        "transforms_test.json",
    ]
    for name in candidates:
        fpath = data_path / name
        if not fpath.exists():
            continue
        with open(fpath) as f:
            data = json.load(f)
        changed = False
        for old, new in renames.items():
            if old in data and new not in data:
                data[new] = data.pop(old)
                changed = True
        for frame in data.get("frames", []):
            for old, new in renames.items():
                if old in frame and new not in frame:
                    frame[new] = frame.pop(old)
                    changed = True
        if ("w" not in data or "h" not in data) and data.get("frames"):
            first = data_path / data["frames"][0].get("file_path", "")
            if first.exists():
                from PIL import Image

                with Image.open(first) as img:
                    w, h = img.size
                data.setdefault("w", w)
                data.setdefault("h", h)
                changed = True
        if changed:
            with open(fpath, "w") as f:
                json.dump(data, f, indent=4)


def detect_dataparser(data_path: Path, requested: str) -> str:
    """Replicate main.py's auto-detection of the dataparser type."""
    if requested != "auto":
        return requested
    if (data_path / "sparse").exists() and not (data_path / "transforms.json").exists():
        return "colmap"
    if (data_path / "transforms_train.json").exists() or "nerf_synthetic" in str(data_path):
        test_json = data_path / "transforms_train.json"
        if test_json.exists():
            try:
                with open(test_json) as f:
                    meta = json.load(f)
                return "blender" if "camera_angle_x" in meta else "nerfstudio"
            except Exception:
                return "nerfstudio"
        return "blender"
    return "nerfstudio"


def build_pipeline(args, device: str):
    """Build the GES pipeline (datamanager + model) exactly as main.py configures it."""
    data_path = Path(args.data)
    dp_type = detect_dataparser(data_path, args.dataparser)
    print(f"[eval] Using dataparser: {dp_type}")

    cfg = config.ges_method.config
    cfg.data = data_path
    cfg.pipeline.datamanager.data = data_path

    if dp_type == "blender":
        from nerfstudio.data.dataparsers.blender_dataparser import BlenderDataParserConfig

        cfg.pipeline.datamanager.dataparser = BlenderDataParserConfig()
        bg_color = "white" if args.background_color == "auto" else args.background_color
        use_real = False if args.use_real_scene == "auto" else (args.use_real_scene == "true")
    elif dp_type == "colmap":
        from nerfstudio.data.dataparsers.colmap_dataparser import ColmapDataParserConfig

        colmap_path = (
            Path("sparse/0") if (data_path / "sparse/0").exists() else Path("colmap/sparse/0")
        )
        cfg.pipeline.datamanager.dataparser = ColmapDataParserConfig(
            colmap_path=colmap_path,
            load_3D_points=True,
            downscale_factor=args.downscale_factor,
            downscale_rounding_mode="round",
        )
        bg_color = "random" if args.background_color == "auto" else args.background_color
        use_real = True if args.use_real_scene == "auto" else (args.use_real_scene == "true")
    else:
        from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig

        _fix_transforms_json(data_path)
        cfg.pipeline.datamanager.dataparser = NerfstudioDataParserConfig(
            load_3D_points=True,
            downscale_factor=args.downscale_factor,
        )
        bg_color = "random" if args.background_color == "auto" else args.background_color
        use_real = True if args.use_real_scene == "auto" else (args.use_real_scene == "true")

    # Keep the dataparser's own data path in sync with the datamanager.
    cfg.pipeline.datamanager.dataparser.data = data_path
    cfg.pipeline.model.use_real_scene = use_real
    cfg.pipeline.model.background_color = bg_color

    # test_mode="inference" still constructs the train dataset/cameras (which is all we need)
    # but skips loading eval data, so this is the lightest setup.
    pipeline = cfg.pipeline.setup(device=device, test_mode="inference", world_size=1, local_rank=0)
    pipeline.eval()
    return pipeline


def find_milestones(milestone_dir: Path) -> list[tuple[int, Path]]:
    """Return (step, path) for every milestone_<step>.pth in the directory, sorted by step."""
    out = []
    for p in milestone_dir.glob("milestone_*.pth"):
        m = re.match(r"milestone_(\d+)\.pth$", p.name)
        if m:
            out.append((int(m.group(1)), p))
    out.sort(key=lambda t: t[0])
    return out


def load_milestone(model, path: Path, device: str) -> None:
    """Load a milestone checkpoint into the model (handles the per-step primitive-count changes)."""
    try:
        state = torch.load(path, map_location=device, weights_only=True)
    except Exception:
        # Older / non-tensor-only checkpoints.
        state = torch.load(path, map_location=device, weights_only=False)
    # GESModel.load_state_dict re-sizes the surfel/gaussian Parameters before loading.
    try:
        model.load_state_dict(state, strict=True)
    except Exception as e:
        print(f"[eval]   strict load failed ({e}); retrying with strict=False")
        model.load_state_dict(state, strict=False)


def pick_loss_indices(num_cameras: int, max_images: int) -> list[int]:
    """Evenly sample up to `max_images` camera indices (0 or negative => use all)."""
    if max_images <= 0 or max_images >= num_cameras:
        return list(range(num_cameras))
    step = num_cameras / max_images
    return sorted({min(num_cameras - 1, int(i * step)) for i in range(max_images)})


@torch.no_grad()
def render_camera(model, camera):
    """Render a single (already device-resident) camera, returning HxWx3 float RGB in [0,1]."""
    # get_outputs mutates the camera's output resolution, so always hand it a private copy.
    cam = copy.deepcopy(camera)
    outputs = model.get_outputs(cam)
    return outputs


@torch.no_grad()
def evaluate(model, dm, loss_indices, device) -> tuple[float, float]:
    """Average L1 loss and PSNR over the chosen training views at the current model state."""
    l1_total = 0.0
    psnr_total = 0.0
    n = 0
    for idx in loss_indices:
        camera = dm.train_cameras[idx : idx + 1].to(device)
        batch = dm.cached_train[idx]
        outputs = render_camera(model, camera)
        gt = model.composite_with_background(
            model.get_gt_img(batch["image"]), outputs["background"]
        )
        pred = outputs["rgb"]
        l1 = torch.abs(gt - pred).mean()
        psnr = model.psnr(pred.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None])
        l1_total += float(l1.item())
        psnr_total += float(psnr.item())
        n += 1
    return (l1_total / max(n, 1)), (psnr_total / max(n, 1))


def label_image(pil_img, text: str):
    """Draw a small dark label with a light backing box in the top-left corner."""
    from PIL import ImageDraw

    draw = ImageDraw.Draw(pil_img)
    draw.rectangle([4, 4, 8 + 7 * len(text), 20], fill=(255, 255, 255))
    draw.text((8, 6), text, fill=(0, 0, 0))
    return pil_img


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate GES training milestones (loss + composite)."
    )
    parser.add_argument(
        "--data", type=str, default="nerf_synthetic/chair", help="Dataset directory"
    )
    parser.add_argument(
        "--dataparser",
        type=str,
        default="auto",
        choices=["auto", "blender", "colmap", "nerfstudio"],
    )
    parser.add_argument("--downscale-factor", type=int, default=None)
    parser.add_argument(
        "--background-color", type=str, default="auto", choices=["auto", "white", "black", "random"]
    )
    parser.add_argument(
        "--use-real-scene", type=str, default="auto", choices=["auto", "true", "false"]
    )
    parser.add_argument(
        "--milestone-dir",
        type=str,
        default=".",
        help="Directory containing milestone_<step>.pth files",
    )
    parser.add_argument("--output-dir", type=str, default="web_assets")
    parser.add_argument(
        "--view-idx",
        type=int,
        default=0,
        help="Training camera index to use for the fixed progression view",
    )
    parser.add_argument(
        "--max-loss-images",
        type=int,
        default=8,
        help="Number of training views to average loss over (0 = all). Higher is slower.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--cols", type=int, default=4, help="Columns in the progression composite")
    args = parser.parse_args()

    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    if device == "cpu":
        print("[eval] WARNING: CUDA not available; rendering on CPU will be very slow / may fail.")

    milestone_dir = Path(args.milestone_dir)
    milestones = find_milestones(milestone_dir)
    if not milestones:
        print(f"[eval] No milestone_*.pth files found in {milestone_dir.resolve()}")
        sys.exit(1)
    print(f"[eval] Found {len(milestones)} milestones: {[s for s, _ in milestones]}")

    pipeline = build_pipeline(args, device)
    model = pipeline.model
    dm = pipeline.datamanager
    model.eval()

    # `train_cameras` is created lazily as a side-effect of the `cached_train` cached-property
    # (it moves the cameras onto the device there), so materialize it before indexing.
    _ = dm.cached_train
    num_cameras = int(dm.train_cameras.shape[0])
    loss_indices = pick_loss_indices(num_cameras, args.max_loss_images)
    view_idx = max(0, min(args.view_idx, num_cameras - 1))
    print(
        f"[eval] {num_cameras} training cameras; averaging loss over {len(loss_indices)} of them; "
        f"progression view = camera {view_idx}"
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import numpy as np
    from PIL import Image

    loss_points: list[tuple[int, float, float]] = []  # (step, l1, psnr)
    progression: list[tuple[int, Image.Image]] = []

    for step, path in milestones:
        print(f"\n[eval] === milestone {step} ({path.name}) ===")
        load_milestone(model, path, device)
        model.step = step

        l1, psnr = evaluate(model, dm, loss_indices, device)
        loss_points.append((step, l1, psnr))
        print(
            f"[eval]   surfels={model.surfel.means.shape[0]} gaussians={model.gaussian.means.shape[0]} "
            f"| L1={l1:.5f} PSNR={psnr:.2f}"
        )

        camera = dm.train_cameras[view_idx : view_idx + 1].to(device)
        outputs = render_camera(model, camera)
        rgb = (outputs["rgb"].clamp(0, 1).detach().cpu().numpy() * 255).astype(np.uint8)
        img = Image.fromarray(rgb)
        label_image(img, f"Step {step}")
        single_path = out_dir / f"milestone_render_{step}.png"
        img.save(single_path)
        progression.append((step, img))

    # ---- Loss graph -------------------------------------------------------
    steps = [p[0] for p in loss_points]
    l1s = [p[1] for p in loss_points]
    psnrs = [p[2] for p in loss_points]

    import matplotlib.pyplot as plt

    from training_schedule import GAUSSIAN_SPAWN_STEP, SURFEL_DISCARD_ITER, VISIBILITY_PRUNE_STEP

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(steps, l1s, "o-", color="tab:blue", label="L1 loss")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("L1 loss", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(steps, psnrs, "s--", color="tab:green", label="PSNR (dB)")
    ax2.set_ylabel("PSNR (dB)", color="tab:green")
    ax2.tick_params(axis="y", labelcolor="tab:green")

    for x, lbl, col in [
        (SURFEL_DISCARD_ITER, "Surfel discard (10k)", "red"),
        (VISIBILITY_PRUNE_STEP, "Visibility prune (15k)", "orange"),
        (GAUSSIAN_SPAWN_STEP, "Gaussian spawn (20k)", "purple"),
    ]:
        if steps and min(steps) <= x <= max(steps):
            ax1.axvline(x=x, color=col, linestyle=":", alpha=0.6)
            ax1.text(
                x,
                ax1.get_ylim()[1],
                lbl,
                rotation=90,
                va="top",
                ha="right",
                fontsize=7,
                color=col,
                alpha=0.8,
            )

    ax1.set_title("Milestone reconstruction loss vs. iteration")
    fig.tight_layout()
    loss_path = out_dir / "milestone_loss_curve.png"
    fig.savefig(loss_path, dpi=150)
    plt.close(fig)
    print(f"\n[eval] Saved loss graph -> {loss_path}")

    # ---- Progression composite -------------------------------------------
    if progression:
        cols = max(1, args.cols)
        rows = (len(progression) + cols - 1) // cols
        w, h = progression[0][1].size
        composite = Image.new("RGB", (w * cols, h * rows), color="white")
        for i, (_, im) in enumerate(progression):
            composite.paste(im, ((i % cols) * w, (i // cols) * h))
        comp_path = out_dir / "milestone_progression_composite.png"
        composite.save(comp_path)
        print(f"[eval] Saved progression composite -> {comp_path}")

    print("\n[eval] Done.")


if __name__ == "__main__":
    main()
