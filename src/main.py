"""
main.py
Entry point for training the GES model using Nerfstudio's trainer.
"""

import os

os.environ["WANDB_MODE"] = "offline"

import argparse
from pathlib import Path

from nerfstudio.scripts.train import main as ns_train_main

import config

def _fix_transforms_json(data_path: Path):
    """
    Patch transforms JSON files so they are compatible with NerfstudioDataParser:
      1. Rename fx/fy → fl_x/fl_y (focal length key naming)
      2. Add w/h if missing (read from the first image on disk)
    """
    import json

    renames = {"fx": "fl_x", "fy": "fl_y"}
    json_candidates = ["transforms.json", "transforms_train.json", "transforms_val.json", "transforms_test.json"]

    for name in json_candidates:
        fpath = data_path / name
        if not fpath.exists():
            continue
        with open(fpath, "r") as f:
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

        if "w" not in data or "h" not in data:
            frames = data.get("frames", [])
            if frames:
                first_path = data_path / frames[0].get("file_path", "")
                if first_path.exists():
                    from PIL import Image
                    with Image.open(first_path) as img:
                        img_w, img_h = img.size
                    if "w" not in data:
                        data["w"] = img_w
                    if "h" not in data:
                        data["h"] = img_h
                    changed = True
                    print(f"  [fix] Added w={img_w}, h={img_h} to {name} (read from {first_path.name})")

        if changed:
            with open(fpath, "w") as f:
                json.dump(data, f, indent=4)
            print(f"  [fix] Patched {fpath.name} for NerfstudioDataParser compatibility")

def main():
    """
    To run this:
    python src/main.py --data <path-to-data>
    """
    parser = argparse.ArgumentParser(description="GES-Explorer Nerfstudio Trainer")
    parser.add_argument(
        "--data", type=str, default="nerf_synthetic/chair", help="Path to the dataset directory"
    )
    parser.add_argument(
        "--dataparser",
        type=str,
        default="auto",
        choices=["auto", "blender", "colmap", "nerfstudio"],
        help="Dataparser type (default: auto-detect)",
    )
    parser.add_argument(
        "--downscale-factor", type=int, default=None, help="Downscale factor for images"
    )
    parser.add_argument(
        "--max-num-iterations", type=int, default=30000, help="Max number of iterations"
    )
    parser.add_argument(
        "--from-scratch",
        action="store_true",
        help="Whether to train from scratch without loading any checkpoints (overrides auto-resume)",
        default=False,
    )
    parser.add_argument(
        "--background-color",
        type=str,
        default="auto",
        choices=["auto", "white", "black", "random"],
        help="Background color for evaluation rendering",
    )
    parser.add_argument(
        "--use-real-scene",
        type=str,
        default="auto",
        choices=["auto", "true", "false"],
        help="Whether to use real-world scene parameters",
    )
    parser.add_argument(
        "--load-dir", type=str, default=None, help="Path to checkpoint directory to load"
    )
    parser.add_argument(
        "--max-num-surfels",
        type=int,
        default=None,
        help="Restrict the maximum number of surfels during training",
    )
    parser.add_argument(
        "--max-num-gaussians",
        type=int,
        default=None,
        help="Restrict the maximum number of gaussians during training",
    )

    args = parser.parse_args()
    data_path = Path(args.data)

    print(f"Initializing GES Training via Nerfstudio for dataset: {data_path}")

    dataparser_type = args.dataparser
    if dataparser_type == "auto":
        if (data_path / "sparse").exists() and not (data_path / "transforms.json").exists():
            dataparser_type = "colmap"
            print("Auto-detected raw COLMAP dataset structure. Using ColmapDataParser.")
        elif (data_path / "transforms_train.json").exists() or "nerf_synthetic" in str(data_path):
            import json
            is_blender = False
            test_json = data_path / "transforms_train.json"
            if test_json.exists():
                try:
                    with open(test_json) as f:
                        meta = json.load(f)
                    is_blender = "camera_angle_x" in meta
                except Exception:
                    pass
            else:
                is_blender = True

            if is_blender:
                dataparser_type = "blender"
                print("Auto-detected Blender synthetic dataset structure. Using BlenderDataParser.")
            else:
                dataparser_type = "nerfstudio"
                print("Found transforms_train.json but no camera_angle_x — using NerfstudioDataParser.")
        else:
            dataparser_type = "nerfstudio"
            print(
                "Auto-detected Nerfstudio preprocessed dataset structure. Using NerfstudioDataParser."
            )

    cfg = config.ges_method.config
    cfg.data = data_path
    cfg.max_num_iterations = args.max_num_iterations
    cfg.logging.writer_names = ["console", "tensorboard"]

    if dataparser_type == "blender":
        from nerfstudio.data.dataparsers.blender_dataparser import BlenderDataParserConfig

        cfg.pipeline.datamanager.dataparser = BlenderDataParserConfig()

        bg_color = "white" if args.background_color == "auto" else args.background_color
        use_real = False if args.use_real_scene == "auto" else (args.use_real_scene == "true")
        print(
            f"Configured Blender dataparser. Scene properties: background_color={bg_color}, use_real_scene={use_real}"
        )

    elif dataparser_type == "colmap":
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
        print(
            f"Configured COLMAP dataparser (colmap_path={colmap_path}). Scene properties: background_color={bg_color}, use_real_scene={use_real}"
        )

    else:
        from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig

        _fix_transforms_json(data_path)

        cfg.pipeline.datamanager.dataparser = NerfstudioDataParserConfig(
            load_3D_points=True,
            downscale_factor=args.downscale_factor,
        )
        bg_color = "random" if args.background_color == "auto" else args.background_color
        use_real = True if args.use_real_scene == "auto" else (args.use_real_scene == "true")
        print(
            f"Configured Nerfstudio dataparser. Scene properties: background_color={bg_color}, use_real_scene={use_real}"
        )

    cfg.pipeline.model.use_real_scene = use_real
    cfg.pipeline.model.background_color = bg_color

    if args.max_num_surfels is not None:
        cfg.pipeline.model.max_num_surfels = args.max_num_surfels
        print(f"Restricting max_num_surfels during training to: {args.max_num_surfels}")
    if args.max_num_gaussians is not None:
        cfg.pipeline.model.max_num_gaussians = args.max_num_gaussians
        print(f"Restricting max_num_gaussians during training to: {args.max_num_gaussians}")

    scene_name = data_path.name if data_path.name else "ges-scene"
    base_dir = Path(f"outputs/{scene_name}/ges-method")

    if args.load_dir:
        cfg.load_dir = Path(args.load_dir)
        print(f"Resuming from explicitly provided checkpoint directory: {cfg.load_dir}")
    elif base_dir.exists() and not args.from_scratch:
        run_folders = [
            f for f in base_dir.iterdir() if f.is_dir() and (f / "nerfstudio_models").exists()
        ]
        if run_folders:
            latest_run = sorted(run_folders)[-1]
            ckpt_dir = latest_run / "nerfstudio_models"
            checkpoints = list(ckpt_dir.glob("step-*.ckpt"))
            if checkpoints:
                print(f"Auto-resuming from latest checkpoint in {ckpt_dir}...")
                cfg.load_dir = ckpt_dir

    ns_train_main(cfg)

if __name__ == "__main__":
    main()
