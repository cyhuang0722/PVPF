from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scsn_model.data.dataset import SunConditionedRBRDataset
from scsn_model.models.full_model import SunConditionedStochasticRBRModel
from scsn_model.utils.io import ensure_dir, load_json, normalize_config_paths, resolve_project_path
from scsn_model.utils.runtime import configure_matplotlib_cache
from scsn_model.viz.rbr import save_rbr_distribution_figure

configure_matplotlib_cache(ROOT / "artifacts")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export SCSN RBR-distribution video for one day")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--config", default=None)
    parser.add_argument("--fps", type=int, default=4)
    args = parser.parse_args()

    run_dir = resolve_project_path(args.run_dir, must_exist=True)
    config = normalize_config_paths(load_json(args.config) if args.config else load_json(run_dir / "run_config.json"))
    device_cfg = str(config.get("device", "auto")).lower()
    device = torch.device("cuda" if device_cfg == "auto" and torch.cuda.is_available() else device_cfg if device_cfg != "auto" else "cpu")

    dataset = SunConditionedRBRDataset(
        csv_path=config["data"]["samples_csv"],
        split=args.split,
        image_size=tuple(config["data"]["image_size"]),
        sky_mask_path=config["data"].get("sky_mask_path"),
        peak_power_w=float(config["data"]["peak_power_w"]),
    )
    model = SunConditionedStochasticRBRModel(config["model"]).to(device)
    model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location=device))
    model.eval()

    df = dataset.dataframe().reset_index(drop=True)
    selected = df.index[df["ts_target"].astype(str).str.slice(0, 10) == args.date].tolist()
    if not selected:
        raise RuntimeError(f"No samples found for date={args.date} split={args.split}")

    out_dir = ensure_dir(run_dir / "videos" / f"{args.split}_{args.date}")
    frames_dir = ensure_dir(out_dir / "frames")

    for frame_idx, sample_index in enumerate(selected):
        sample = dataset[sample_index]
        batch = {k: v.unsqueeze(0).to(device) if torch.is_tensor(v) else v for k, v in sample.items()}
        with torch.no_grad():
            out = model(batch["images"], batch["pv_history"], batch["solar_vec"], sun_xy=batch["sun_xy"], target_sun_xy=batch["target_sun_xy"])
        pred_quantiles = out["prediction"][0].detach().cpu().numpy()
        save_rbr_distribution_figure(
            image=batch["images"][0, -1, :3].detach().cpu().numpy(),
            past_rbr_mean=batch["images"][0, :, 3].mean(dim=0).detach().cpu().numpy(),
            attention=out["attention_map"][0, 0].detach().cpu().numpy(),
            rbr_mean=out["future_rbr_mean_15min"][0, 0].detach().cpu().numpy(),
            rbr_variance=out["future_rbr_variance_15min"][0, 0].detach().cpu().numpy(),
            past_rbr_variation=batch["past_rbr_variation"][0, 0].detach().cpu().numpy(),
            future_rbr_mean_target=batch["future_rbr_mean"][0, 0].detach().cpu().numpy(),
            future_rbr_variance_target=batch["future_rbr_variance"][0, 0].detach().cpu().numpy(),
            out_path=frames_dir / f"frame_{frame_idx:04d}.png",
            title=str(df.iloc[sample_index]["ts_target"]),
            future_rbr_valid=bool(batch["future_rbr_valid"][0].detach().cpu().item() > 0.0),
            summary_values={
                "q90-q10": float(pred_quantiles[4] - pred_quantiles[0]),
                "pv sigma": float(out["pv_sigma"][0, 0].detach().cpu().item()),
                "sun attn sigma": float(out["sun_attention_sigma"][0, 0].detach().cpu().item()),
                "sun mean": float(out["sun_local_rbr_mean"][0, 0].detach().cpu().item()),
                "sun var": float(out["sun_local_rbr_variance"][0, 0].detach().cpu().item()),
            },
        )

    video_path = out_dir / f"scsn_rbr_distribution_{args.split}_{args.date}.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-framerate", str(args.fps), "-i", str(frames_dir / "frame_%04d.png"), "-pix_fmt", "yuv420p", str(video_path)],
        check=True,
    )
    print(video_path)


if __name__ == "__main__":
    main()
