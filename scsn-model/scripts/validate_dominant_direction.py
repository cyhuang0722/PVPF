from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scsn_model.utils.runtime import configure_matplotlib_cache

configure_matplotlib_cache(ROOT / "artifacts")

from scsn_model.data.dataset import load_mask
from scsn_model.utils.io import ensure_dir, load_json, save_json, timestamped_run_dir
from scsn_model.validation.dominant_direction import (
    load_camera_index,
    save_dominant_direction_figure,
    save_dominant_direction_json,
    validate_sample_dominant_direction,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline dominant-direction validation using 5 recent 1-minute frames")
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--date", default=None)
    parser.add_argument("--date-from", default=None)
    parser.add_argument("--date-to", default=None)
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--samples-per-day", type=int, default=None)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    config = load_json(args.config)
    out_root = Path(args.out_dir) if args.out_dir else timestamped_run_dir(ROOT / "artifacts" / "dominant_direction_validation", prefix="validation")
    figures_dir = ensure_dir(out_root / "figures")
    labels_dir = ensure_dir(out_root / "labels")

    df = pd.read_csv(config["data"]["samples_csv"])
    df["ts_target"] = pd.to_datetime(df["ts_target"])
    df["ts_anchor"] = pd.to_datetime(df["ts_anchor"])
    df["ts_target_local"] = df["ts_target"].dt.tz_localize(None)
    df = df[df["split"] == args.split].reset_index(drop=True)
    if args.date:
        df = df[df["ts_target_local"].astype(str).str.slice(0, 10) == args.date].reset_index(drop=True)
    if args.date_from:
        df = df[df["ts_target_local"] >= pd.Timestamp(args.date_from)].reset_index(drop=True)
    if args.date_to:
        df = df[df["ts_target_local"] < (pd.Timestamp(args.date_to) + pd.Timedelta(days=1))].reset_index(drop=True)
    if args.samples_per_day:
        picked = []
        for _, day_df in df.groupby(df["ts_target_local"].dt.strftime("%Y-%m-%d"), sort=True):
            if len(day_df) <= int(args.samples_per_day):
                picked.append(day_df)
            else:
                idxs = np.linspace(0, len(day_df) - 1, int(args.samples_per_day), dtype=int)
                picked.append(day_df.iloc[idxs])
        df = pd.concat(picked, axis=0).reset_index(drop=True) if picked else df.iloc[0:0].copy()
    df = df.head(int(args.max_samples)).reset_index(drop=True)

    sky_mask_hw = None
    if config["data"].get("sky_mask_path"):
        sky_mask_hw = load_mask(config["data"]["sky_mask_path"], tuple(config["data"]["image_size"]))[0]

    camera_ts_ns, camera_paths = load_camera_index(config["data"]["camera_index_csv"])
    rows = []
    failures = []
    for idx, row in df.iterrows():
        try:
            meta, result, payload = validate_sample_dominant_direction(
                row=row,
                camera_ts_ns=camera_ts_ns,
                camera_paths=camera_paths,
                image_size=tuple(config["data"]["image_size"]),
                sky_mask_hw=sky_mask_hw,
                tolerance_sec=int(config["data"].get("image_match_tolerance_sec", 75)),
            )
        except Exception as exc:
            failures.append({"index": int(idx), "ts_target": str(row["ts_target"]), "error": str(exc)})
            continue
        save_dominant_direction_figure(figures_dir / f"{idx:03d}_{str(row['ts_target'])[:16].replace(':','').replace(' ','_')}.png", meta, result, payload)
        save_dominant_direction_json(labels_dir / f"{idx:03d}.json", meta, result)
        rows.append({
            "ts_anchor": meta["ts_anchor"],
            "ts_target": meta["ts_target"],
            "target_pv_w": meta["target_pv_w"],
            "direction_label": result.direction_label,
            "direction_name": result.direction_name,
            "direction_confidence": result.direction_confidence,
            "direction_consistency": result.direction_consistency,
            "mean_cloud_fraction": result.mean_cloud_fraction,
            "mean_peak_strength": result.mean_peak_strength,
            "dominant_dx": float(result.dominant_vector[0]),
            "dominant_dy": float(result.dominant_vector[1]),
        })

    pd.DataFrame(rows).to_csv(out_root / "dominant_direction_stats.csv", index=False)
    save_json(out_root / "summary.json", {
        "config_path": args.config,
        "split": args.split,
        "date": args.date,
        "date_from": args.date_from,
        "date_to": args.date_to,
        "num_samples": int(len(rows)),
        "num_failures": int(len(failures)),
        "failures": failures,
    })
    print(json.dumps({"out_dir": str(out_root), "num_samples": int(len(rows)), "num_failures": int(len(failures))}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
