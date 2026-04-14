from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scsn_model.utils.runtime import configure_matplotlib_cache

configure_matplotlib_cache(ROOT / "artifacts")

from scsn_model.validation.motion_teacher import (
    PatchTeacherResult,
    load_camera_index,
    save_teacher_package,
    save_validation_figure,
    validate_motion_teacher_for_sample,
)
from scsn_model.data.dataset import load_mask
from scsn_model.utils.io import ensure_dir, load_json, save_json, timestamped_run_dir


def _summarize(rows: list[dict]) -> dict:
    if not rows:
        return {"num_samples": 0}
    df = pd.DataFrame(rows)
    summary = {
        "num_samples": int(len(df)),
        "valid_patch_fraction_mean": float(df["valid_patch_fraction"].mean()),
        "valid_sun_patch_fraction_mean": float(df["valid_sun_patch_fraction"].mean()),
        "toward_sun_score_mean": float(df["toward_sun_score"].mean()),
        "away_from_sun_score_mean": float(df["away_from_sun_score"].mean()),
        "future_delta_w_mean": float(df["future_delta_w"].mean()),
        "clear_like_count": int((df["valid_patch_fraction"] < 0.05).sum()),
        "negative_delta_count": int((df["future_delta_w"] < 0).sum()),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline validation for motion supervision before training")
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--date", default=None, help="Optional date filter like 2026-03-31")
    parser.add_argument("--max-samples", type=int, default=30)
    parser.add_argument("--pair-start-min", type=int, default=-1, help="Use one 1-minute pair, e.g. -1 -> 0")
    parser.add_argument("--pair-end-min", type=int, default=0)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    config = load_json(args.config)
    out_root = Path(args.out_dir) if args.out_dir else timestamped_run_dir(ROOT / "artifacts" / "motion_teacher_validation", prefix="validation")
    figures_dir = ensure_dir(out_root / "figures")
    teachers_dir = ensure_dir(out_root / "teacher_packages")

    samples_df = pd.read_csv(config["data"]["samples_csv"])
    samples_df["ts_anchor"] = pd.to_datetime(samples_df["ts_anchor"])
    samples_df["ts_target"] = pd.to_datetime(samples_df["ts_target"])
    samples_df = samples_df[samples_df["split"] == args.split].reset_index(drop=True)
    if args.date:
        samples_df = samples_df[samples_df["ts_target"].astype(str).str.slice(0, 10) == args.date].reset_index(drop=True)
    samples_df = samples_df.head(int(args.max_samples)).reset_index(drop=True)

    sky_mask_hw = None
    if config["data"].get("sky_mask_path"):
        sky_mask_hw = load_mask(config["data"]["sky_mask_path"], tuple(config["data"]["image_size"]))[0]

    camera_ts_ns, camera_paths = load_camera_index(config["data"]["camera_index_csv"])
    stats_rows: list[dict] = []
    failures: list[dict] = []

    for idx, row in samples_df.iterrows():
        try:
            meta, result, payload = validate_motion_teacher_for_sample(
                row=row,
                camera_ts_ns=camera_ts_ns,
                camera_paths=camera_paths,
                image_size=tuple(config["data"]["image_size"]),
                tolerance_sec=int(config["data"].get("image_match_tolerance_sec", 75)),
                sky_mask_hw=sky_mask_hw,
                teacher_pair_min=(int(args.pair_start_min), int(args.pair_end_min)),
                flow_size=(
                    int(config["data"].get("teacher_flow_resolution", 64)),
                    int(config["data"].get("teacher_flow_resolution", 64)),
                ),
                max_displacement_px=int(config["data"].get("teacher_max_displacement_px", 2)),
                patch_grid_size=int(config["model"].get("patch_grid_size", 8)),
            )
        except Exception as exc:
            failures.append({"index": int(idx), "ts_target": str(row["ts_target"]), "error": str(exc)})
            continue

        title = (
            f"{row['ts_target']} | valid={result.valid_patch_fraction:.2f} | "
            f"sun_valid={result.valid_sun_patch_fraction:.2f} | "
            f"toward={result.toward_sun_score:.2f} | away={result.away_from_sun_score:.2f}"
        )
        save_validation_figure(
            out_path=figures_dir / f"{idx:03d}_{str(row['ts_target'])[:16].replace(':','').replace(' ','_')}.png",
            prev_image=payload["prev_image"],
            curr_image=payload["curr_image"],
            masks=payload["masks"],
            flow=payload["flow"],
            patch_result=result,
            sun_xy=np.asarray([meta["sun_x_px"], meta["sun_y_px"]], dtype=float),
            title=title,
        )
        save_teacher_package(teachers_dir / f"{idx:03d}.json", meta=meta, result=result)

        stats_rows.append({
            **meta,
            "valid_patch_fraction": result.valid_patch_fraction,
            "valid_sun_patch_fraction": result.valid_sun_patch_fraction,
            "toward_sun_score": result.toward_sun_score,
            "away_from_sun_score": result.away_from_sun_score,
            "mean_patch_coherence": float(result.patch_coherence.mean()),
            "mean_patch_cloud_fraction": float(result.patch_cloud_fraction.mean()),
            "mean_teacher_confidence": float(result.teacher_confidence.mean()),
        })

    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(out_root / "sample_stats.csv", index=False)
    save_json(out_root / "summary.json", {
        "config_path": args.config,
        "split": args.split,
        "date": args.date,
        "pair_minutes": [int(args.pair_start_min), int(args.pair_end_min)],
        "summary": _summarize(stats_rows),
        "failures": failures,
    })
    print(json.dumps({
        "out_dir": str(out_root),
        "num_samples": int(len(stats_rows)),
        "num_failures": int(len(failures)),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    import numpy as np

    main()
