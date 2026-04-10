from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_POWER_CSV = Path("data/power/power-LSK_N_0117-0401.csv")
DEFAULT_CAMERA_INDEX_CSV = Path("data/camera_data/index/raw_index_resized_64.csv")
DEFAULT_OUT_CSV = BASE_DIR / "derived" / "samples_t.csv"
INPUT_OFFSETS_MIN = [15, 13, 11, 9, 7, 5, 3, 1]


def _nearest_row(
    desired_ts: pd.Timestamp,
    cam_ts_ns: np.ndarray,
    cam_paths: np.ndarray,
    max_time_diff_sec: int,
) -> tuple[str | None, float | None]:
    desired_ns = desired_ts.value
    pos = int(np.searchsorted(cam_ts_ns, desired_ns))
    candidates: list[int] = []
    if 0 <= pos < len(cam_ts_ns):
        candidates.append(pos)
    if pos - 1 >= 0:
        candidates.append(pos - 1)
    if not candidates:
        return None, None
    best_i = min(candidates, key=lambda i: abs(int(cam_ts_ns[i]) - desired_ns))
    diff_sec = abs(int(cam_ts_ns[best_i]) - desired_ns) / 1e9
    if diff_sec > max_time_diff_sec:
        return None, None
    return str(cam_paths[best_i]), float(diff_sec)


def build_samples(
    power_csv: Path,
    camera_index_csv: Path,
    out_csv: Path,
    max_time_diff_sec: int,
    camera_path_prefix_from: str | None = None,
    camera_path_prefix_to: str | None = None,
) -> dict:
    power_df = pd.read_csv(power_csv)
    if "date" not in power_df.columns or "value" not in power_df.columns:
        raise ValueError("power csv must include columns: date,value")
    power_df["date"] = pd.to_datetime(power_df["date"], errors="coerce")
    power_df["value"] = pd.to_numeric(power_df["value"], errors="coerce")
    power_df = power_df.dropna(subset=["date", "value"]).sort_values("date").reset_index(drop=True)
    n_power_non_null = int(len(power_df))

    power_df = power_df[power_df["date"].dt.minute.isin([0, 15, 30, 45])].reset_index(drop=True)
    n_targets_quarter_hour = int(len(power_df))
    power_df = power_df[power_df["value"] > 0].reset_index(drop=True)

    cam_df = pd.read_csv(camera_index_csv)
    if "timestamp" not in cam_df.columns or "file_path" not in cam_df.columns:
        raise ValueError("camera index csv must include columns: timestamp,file_path")
    cam_df["timestamp"] = pd.to_datetime(cam_df["timestamp"], errors="coerce")
    cam_df = cam_df.dropna(subset=["timestamp", "file_path"]).sort_values("timestamp").reset_index(drop=True)

    cam_ts_ns = cam_df["timestamp"].astype("int64").to_numpy()
    cam_paths = cam_df["file_path"].astype(str).to_numpy()

    def rewrite_path(path: str) -> str:
        if camera_path_prefix_from and camera_path_prefix_to and path.startswith(camera_path_prefix_from):
            return camera_path_prefix_to + path[len(camera_path_prefix_from):]
        return path

    rows: list[dict] = []
    for target_ts, target_value in zip(power_df["date"], power_df["value"]):
        input_ts_list = [target_ts - pd.Timedelta(minutes=o) for o in INPUT_OFFSETS_MIN]
        img_paths: list[str] = []
        match_diffs_sec: list[float] = []
        missing = False
        for ts in input_ts_list:
            p, d = _nearest_row(ts, cam_ts_ns, cam_paths, max_time_diff_sec)
            if p is None or d is None:
                missing = True
                break
            p = rewrite_path(p)
            if not Path(p).exists():
                missing = True
                break
            img_paths.append(p)
            match_diffs_sec.append(float(d))
        if missing:
            continue

        rows.append(
            {
                "ts_target": target_ts,
                "input_offsets_min": json.dumps(INPUT_OFFSETS_MIN),
                "input_timestamps": json.dumps([str(x) for x in input_ts_list]),
                "img_paths": json.dumps(img_paths, ensure_ascii=False),
                "match_diffs_sec": json.dumps(match_diffs_sec),
                "pv_target_w": float(target_value),
            }
        )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    summary = {
        "power_csv": str(power_csv),
        "camera_index_csv": str(camera_index_csv),
        "out_csv": str(out_csv),
        "max_time_diff_sec": max_time_diff_sec,
        "camera_path_prefix_from": camera_path_prefix_from,
        "camera_path_prefix_to": camera_path_prefix_to,
        "input_offsets_min": INPUT_OFFSETS_MIN,
        "n_power_non_null": n_power_non_null,
        "n_targets_quarter_hour": n_targets_quarter_hour,
        "n_targets_positive_pv": int(len(power_df)),
        "n_samples": int(len(rows)),
    }
    (out_csv.parent / "preprocess_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build U-Net image-stack to PV@t samples")
    parser.add_argument("--power-csv", default=str(DEFAULT_POWER_CSV))
    parser.add_argument("--camera-index-csv", default=str(DEFAULT_CAMERA_INDEX_CSV))
    parser.add_argument("--out-csv", default=str(DEFAULT_OUT_CSV))
    parser.add_argument("--max-time-diff-sec", type=int, default=180)
    parser.add_argument("--camera-path-prefix-from", default=None)
    parser.add_argument("--camera-path-prefix-to", default=None)
    args = parser.parse_args()

    root_dir = BASE_DIR.parent.parent
    power_csv = Path(args.power_csv)
    camera_index_csv = Path(args.camera_index_csv)
    out_csv = Path(args.out_csv)
    if not power_csv.is_absolute():
        power_csv = root_dir / power_csv
    if not camera_index_csv.is_absolute():
        camera_index_csv = root_dir / camera_index_csv
    if not out_csv.is_absolute():
        out_csv = BASE_DIR / out_csv

    summary = build_samples(
        power_csv=power_csv,
        camera_index_csv=camera_index_csv,
        out_csv=out_csv,
        max_time_diff_sec=args.max_time_diff_sec,
        camera_path_prefix_from=args.camera_path_prefix_from,
        camera_path_prefix_to=args.camera_path_prefix_to,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
