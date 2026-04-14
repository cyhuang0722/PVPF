from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from new_model.utils.runtime import configure_matplotlib_cache

configure_matplotlib_cache(ROOT / "artifacts")

from new_model.data.dataset import SunConditionedPVDataset
from new_model.models.full_model import MinimalSunConditionedPVModel
from new_model.utils.io import ensure_dir, load_json
from new_model.viz.motion import save_patch_motion_comparison_figure


def _load_config(run_dir: Path, config_arg: str | None) -> dict:
    if config_arg:
        config = load_json(config_arg)
    else:
        run_cfg = run_dir / "run_config.json"
        if not run_cfg.exists():
            raise FileNotFoundError(f"Cannot find config. Pass --config or ensure {run_cfg} exists.")
        config = load_json(run_cfg)
    _normalize_data_paths(config)
    return config


def _normalize_data_paths(config: dict) -> None:
    project_root = ROOT.parent
    data_cfg = config.get("data", {})
    for key in ("samples_csv", "sky_mask_path", "camera_dir", "camera_index_csv", "pv_csv", "calibration_json", "artifact_dir"):
        raw = data_cfg.get(key)
        if not raw:
            continue
        path = Path(raw)
        if path.exists():
            continue
        parts = path.parts
        if "PVPF" in parts:
            idx = parts.index("PVPF")
            candidate = project_root / Path(*parts[idx + 1 :])
            if candidate.exists():
                data_cfg[key] = str(candidate)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export optical-flow supervision comparison video for one day")
    parser.add_argument("--run-dir", required=True, help="Training run directory containing best_model.pt")
    parser.add_argument("--date", required=True, help="Target date like 2026-03-31")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--config", default=None, help="Optional config path; defaults to run_dir/run_config.json")
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--out-dir", default=None, help="Optional output directory")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    config = _load_config(run_dir, args.config)
    device_cfg = str(config.get("device", "auto")).lower()
    device = torch.device("cuda" if torch.cuda.is_available() and device_cfg != "cpu" else "cpu")

    dataset = SunConditionedPVDataset(
        csv_path=config["data"]["samples_csv"],
        split=args.split,
        image_size=tuple(config["data"]["image_size"]),
        sky_mask_path=config["data"].get("sky_mask_path"),
        peak_power_w=float(config["data"]["peak_power_w"]),
        camera_index_csv=config["data"].get("camera_index_csv"),
        image_match_tolerance_sec=int(config["data"].get("image_match_tolerance_sec", 75)),
        motion_teacher_pairs_min=config["data"].get("motion_teacher_pairs_min"),
        patch_grid_size=int(config["model"].get("patch_grid_size", 8)),
        teacher_flow_resolution=int(config["data"].get("teacher_flow_resolution", 64)),
        teacher_max_displacement_px=int(config["data"].get("teacher_max_displacement_px", 2)),
        teacher_conf_threshold=float(config["data"].get("teacher_conf_threshold", 0.25)),
        teacher_min_patch_vectors=int(config["data"].get("teacher_min_patch_vectors", 6)),
        teacher_min_magnitude=float(config["data"].get("teacher_min_magnitude", 0.15)),
    )
    model = MinimalSunConditionedPVModel(config["model"]).to(device)
    model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location=device))
    model.eval()

    df = dataset.dataframe().reset_index(drop=True)
    ts = df["ts_target"].astype(str).str.slice(0, 10)
    selected = df.index[ts == args.date].tolist()
    if not selected:
        raise RuntimeError(f"No samples found for date={args.date} split={args.split}")

    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "videos" / f"{args.split}_{args.date}"
    frames_dir = ensure_dir(out_dir / "frames")

    use_csi = bool(config["data"].get("use_clear_sky_index", True))
    for frame_idx, sample_index in enumerate(selected):
        sample = dataset[sample_index]
        batch = {}
        for key, value in sample.items():
            batch[key] = value.unsqueeze(0).to(device) if torch.is_tensor(value) else value

        with torch.no_grad():
            out = model(
                batch["images"],
                batch["pv_history"],
                batch["solar_vec"],
                sun_xy=batch["sun_xy"],
                sun_angles=batch["sun_angles"],
            )

        pred_value = float(out["prediction"][0, 0].detach().cpu().item())
        pred_value_clip = max(0.0, min(1.0, pred_value)) if use_csi else max(0.0, pred_value)
        clear_sky_w = float(batch["target_clear_sky_w"][0].detach().cpu().item())
        pred_w = pred_value_clip * clear_sky_w if use_csi else pred_value_clip
        target_w = float(batch["target_pv_w"][0].detach().cpu().item())
        ts_target = str(df.iloc[sample_index]["ts_target"])

        title = (
            f"{ts_target} | pred={pred_w:.1f}W | true={target_w:.1f}W | "
            f"pred_value={pred_value_clip:.3f} | target={float(batch['target'][0].cpu().item()):.3f}"
        )
        save_patch_motion_comparison_figure(
            image_current=batch["images"][0, -1].detach().cpu().numpy(),
            image_prev_1=batch["images"][0, -2].detach().cpu().numpy(),
            image_prev_2=batch["images"][0, -3].detach().cpu().numpy(),
            patch_motion_pred=out["patch_motion_pred"][0].detach().cpu().numpy(),
            patch_motion_teacher=batch["patch_motion_teacher"][0].detach().cpu().numpy(),
            patch_motion_mask=batch["patch_motion_mask"][0].detach().cpu().numpy(),
            sun_prior=out["sun_prior"][0, 0].detach().cpu().numpy(),
            out_path=frames_dir / f"frame_{frame_idx:04d}.png",
            title=title,
        )

    video_path = out_dir / f"patch_motion_supervision_{args.split}_{args.date}.mp4"
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(args.fps),
        "-i",
        str(frames_dir / "frame_%04d.png"),
        "-pix_fmt",
        "yuv420p",
        str(video_path),
    ]
    subprocess.run(cmd, check=True)
    print(video_path)


if __name__ == "__main__":
    main()
