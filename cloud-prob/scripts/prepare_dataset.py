from __future__ import annotations

import argparse
import json
import re
import sys
from bisect import bisect_left
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
sys.path.insert(0, str(ROOT))

from cloud_prob.solar_geometry import Calibration, build_solar_feature_vector, compute_solar_position, project_sun_to_image


TIMESTAMP_RE = re.compile(r"_(\d{17})_TIMING\.(?:jpg|jpeg|png)$", re.IGNORECASE)
WEATHER_KEEP_COLUMNS = [
    "blue_fraction",
    "gray_fraction",
    "dark_gray_fraction",
    "bright_white_fraction",
    "cloud_fraction_proxy",
    "blue_to_gray_ratio",
    "sky_value_std",
    "sky_edge_strength",
    "center_edge_value_delta",
    "radial_value_corr",
    "bright_spot_delta",
    "bright_spot_radial_corr",
    "mean_saturation",
    "mean_value",
    "rbr_diff_mean",
    "rbr_diff_std",
    "rbr_diff_p95",
    "rbr_change_fraction",
    "zenith_deg",
    "target_zenith_deg",
]


def parse_image_time(path: Path) -> pd.Timestamp | None:
    match = TIMESTAMP_RE.search(path.name)
    if not match:
        return None
    return pd.to_datetime(match.group(1), format="%Y%m%d%H%M%S%f").tz_localize("Asia/Shanghai")


def build_image_index(camera_root: Path) -> tuple[list[pd.Timestamp], list[str]]:
    pairs = []
    for path in camera_root.rglob("*.jpg"):
        ts = parse_image_time(path)
        if ts is not None:
            pairs.append((ts, str(path)))
    pairs.sort(key=lambda item: item[0])
    return [item[0] for item in pairs], [item[1] for item in pairs]


def nearest_image(times: list[pd.Timestamp], paths: list[str], target: pd.Timestamp, tolerance_seconds: float) -> str | None:
    pos = bisect_left(times, target)
    candidates = []
    if pos < len(times):
        candidates.append(pos)
    if pos > 0:
        candidates.append(pos - 1)
    best = None
    best_delta = float("inf")
    for idx in candidates:
        delta = abs((times[idx] - target).total_seconds())
        if delta < best_delta:
            best_delta = delta
            best = paths[idx]
    return best if best is not None and best_delta <= tolerance_seconds else None


def select_history_images(
    times: list[pd.Timestamp],
    paths: list[str],
    anchor: pd.Timestamp,
    steps: int,
    step_minutes: int,
    tolerance_seconds: float,
) -> list[str] | None:
    selected = []
    start = anchor - pd.Timedelta(minutes=(steps - 1) * step_minutes)
    for i in range(steps):
        target = start + pd.Timedelta(minutes=i * step_minutes)
        path = nearest_image(times, paths, target, tolerance_seconds)
        if path is None:
            return None
        selected.append(path)
    return selected


def load_mask(path: Path, image_size: int) -> np.ndarray:
    with Image.open(path) as image:
        mask = image.convert("L").resize((image_size, image_size), resample=Image.NEAREST)
    return (np.asarray(mask, dtype=np.float32) / 255.0 >= 0.5).astype(np.float32)


def add_history_features(out: dict[str, float], rows: pd.DataFrame, target_clear: float, peak_power: float, history_count: int) -> None:
    tail = rows.tail(history_count)
    pv = tail["target_pv_w"].to_numpy(dtype=np.float32)
    csi = np.clip(pv / max(float(target_clear), 1e-6), 0.0, 1.25)
    for idx, value in enumerate(pv):
        out[f"past_pv_{idx}_w"] = float(value)
        out[f"past_pv_{idx}_norm"] = float(value / max(float(peak_power), 1e-6))
    out["past_pv_mean_norm"] = float(np.mean(pv) / max(float(peak_power), 1e-6))
    out["past_pv_std_norm"] = float(np.std(pv) / max(float(peak_power), 1e-6))
    out["past_csi_last"] = float(csi[-1])
    out["past_csi_mean"] = float(np.mean(csi))
    out["past_csi_std"] = float(np.std(csi))
    out["past_csi_trend"] = float(np.polyfit(np.arange(len(csi)), csi, 1)[0]) if len(csi) > 1 else 0.0


def build_samples(args: argparse.Namespace) -> pd.DataFrame:
    index = pd.read_csv(args.weather_index_csv)
    index["interval_start"] = pd.to_datetime(index["interval_start"])
    index["interval_end"] = pd.to_datetime(index["interval_end"])
    index["weather_tag"] = index["weather_tag"].astype(str).str.strip().str.lower()
    index = index[index["weather_tag"].isin(["clear_sky", "cloudy", "overcast"])].copy()
    index = index.sort_values("interval_end").reset_index(drop=True)

    image_times, image_paths = build_image_index(Path(args.camera_root))
    mask = load_mask(Path(args.sky_mask_path), int(args.image_size))
    raw_calib = Calibration.from_json(args.calibration_json)
    calib = raw_calib.rescale(dst_w=int(args.sun_coordinate_size), dst_h=int(args.sun_coordinate_size))
    if args.sun_projection_cx_px is not None:
        calib = Calibration(
            cx=float(args.sun_projection_cx_px),
            cy=float(args.sun_projection_cy_px),
            f_px_per_rad=float(args.sun_projection_f_px_per_rad),
            lat=calib.lat,
            lon=calib.lon,
            timezone=calib.timezone,
            reference_width=int(args.sun_coordinate_size),
            reference_height=int(args.sun_coordinate_size),
        )
    solar_times = pd.concat([index["interval_start"], index["interval_end"]]).drop_duplicates().sort_values()
    solar_lookup = compute_solar_position(solar_times.tolist(), calib).set_index("timestamp")
    rows = []
    for pos, row in index.iterrows():
        prev = index.iloc[:pos]
        prev_same_split = prev[prev["split"] == row["split"]]
        if len(prev_same_split) < int(args.history_intervals):
            continue
        image_seq = select_history_images(
            image_times,
            image_paths,
            row["interval_start"],
            int(args.sequence_steps),
            int(args.sequence_step_minutes),
            float(args.image_tolerance_seconds),
        )
        if image_seq is None:
            continue
        target_clear = float(row["target_clear_sky_w"])
        baseline_pv = float(prev_same_split.iloc[-1]["target_pv_w"])
        sample = {
            "interval_start": str(row["interval_start"]),
            "interval_end": str(row["interval_end"]),
            "split": row["split"],
            "image_paths": json.dumps(image_seq),
            "target_pv_w": float(row["target_pv_w"]),
            "target_clear_sky_w": target_clear,
            "target_csi": float(row["target_csi"]),
            "baseline_pv_w": baseline_pv,
            "baseline_csi": float(np.clip(baseline_pv / max(target_clear, 1e-6), 0.0, 1.25)),
            "weather_tag": row["weather_tag"],
        }
        anchor_solar = solar_lookup.loc[row["interval_start"]]
        target_solar = solar_lookup.loc[row["interval_end"]]
        current_sun_x, current_sun_y = project_sun_to_image(
            azimuth_deg=float(anchor_solar["azimuth_deg"]),
            zenith_deg=float(anchor_solar["zenith_deg"]),
            calib=calib,
            image_width=int(args.sun_coordinate_size),
            image_height=int(args.sun_coordinate_size),
            azimuth_offset_deg=float(args.azimuth_offset_deg),
            azimuth_clockwise=bool(args.azimuth_clockwise),
            image_offset_x_px=float(args.sun_image_offset_x_px),
            image_offset_y_px=float(args.sun_image_offset_y_px),
        )
        target_sun_x, target_sun_y = project_sun_to_image(
            azimuth_deg=float(target_solar["azimuth_deg"]),
            zenith_deg=float(target_solar["zenith_deg"]),
            calib=calib,
            image_width=int(args.sun_coordinate_size),
            image_height=int(args.sun_coordinate_size),
            azimuth_offset_deg=float(args.azimuth_offset_deg),
            azimuth_clockwise=bool(args.azimuth_clockwise),
            image_offset_x_px=float(args.sun_image_offset_x_px),
            image_offset_y_px=float(args.sun_image_offset_y_px),
        )
        target_sun_x_f = float(np.asarray(target_sun_x).item())
        target_sun_y_f = float(np.asarray(target_sun_y).item())
        sample["sun_x_px"] = target_sun_x_f
        sample["sun_y_px"] = target_sun_y_f
        sample["current_sun_x_px"] = float(np.asarray(current_sun_x).item())
        sample["current_sun_y_px"] = float(np.asarray(current_sun_y).item())
        sample["sun_dx_px"] = target_sun_x_f - float(np.asarray(current_sun_x).item())
        sample["sun_dy_px"] = target_sun_y_f - float(np.asarray(current_sun_y).item())
        sample["azimuth_deg"] = float(anchor_solar["azimuth_deg"])
        sample["target_azimuth_deg"] = float(target_solar["azimuth_deg"])
        sample["zenith_deg"] = float(anchor_solar["zenith_deg"])
        sample["target_zenith_deg"] = float(target_solar["zenith_deg"])
        solar_vec = build_solar_feature_vector(
            sun_x_px=target_sun_x_f,
            sun_y_px=target_sun_y_f,
            azimuth_deg=float(target_solar["azimuth_deg"]),
            zenith_deg=float(target_solar["zenith_deg"]),
            image_width=int(args.sun_coordinate_size),
            image_height=int(args.sun_coordinate_size),
        )
        for idx, value in enumerate(solar_vec):
            sample[f"solar_vec_{idx}"] = float(value)
        sample["sun_position_source"] = "calibrated_solar_geometry"
        for col in WEATHER_KEEP_COLUMNS:
            sample[col] = float(row[col]) if col in row and pd.notna(row[col]) else np.nan
        add_history_features(sample, prev_same_split, target_clear, float(args.peak_power_w), int(args.history_intervals))
        rows.append(sample)
    samples = pd.DataFrame(rows)
    if args.max_samples and len(samples) > int(args.max_samples):
        parts = []
        per_split = max(1, int(args.max_samples) // max(samples["split"].nunique(), 1))
        for _, part in samples.groupby("split", sort=False):
            parts.append(part.head(min(per_split, len(part))))
        samples = pd.concat(parts, ignore_index=True).head(int(args.max_samples))
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare cloud-prob training samples from weather interval classification.")
    parser.add_argument("--weather-index-csv", type=Path, default=WORKSPACE / "data/weather_interval_classification/weather_interval_index.csv")
    parser.add_argument("--camera-root", type=Path, default=WORKSPACE / "data/camera_data/resized_256")
    parser.add_argument("--sky-mask-path", type=Path, default=WORKSPACE / "data/sky_mask.png")
    parser.add_argument("--calibration-json", type=Path, default=WORKSPACE / "data/calibration.json")
    parser.add_argument("--sun-coordinate-size", type=int, default=256)
    parser.add_argument("--azimuth-offset-deg", type=float, default=330.71337038338856)
    parser.add_argument("--azimuth-clockwise", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--sun-image-offset-x-px", type=float, default=0.0)
    parser.add_argument("--sun-image-offset-y-px", type=float, default=0.0)
    parser.add_argument("--sun-projection-cx-px", type=float, default=128.76104600689308)
    parser.add_argument("--sun-projection-cy-px", type=float, default=123.91888618003209)
    parser.add_argument("--sun-projection-f-px-per-rad", type=float, default=71.27383562943854)
    parser.add_argument("--out-csv", type=Path, default=ROOT / "artifacts/dataset/samples.csv")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--sequence-steps", type=int, default=16)
    parser.add_argument("--sequence-step-minutes", type=int, default=1)
    parser.add_argument("--image-tolerance-seconds", type=float, default=45.0)
    parser.add_argument("--history-intervals", type=int, default=4)
    parser.add_argument("--peak-power-w", type=float, default=66300.0)
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()

    samples = build_samples(args)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    samples.to_csv(args.out_csv, index=False)
    summary = {
        "samples": int(len(samples)),
        "splits": samples["split"].value_counts().to_dict() if len(samples) else {},
        "weather": samples["weather_tag"].value_counts().to_dict() if len(samples) else {},
    }
    (args.out_csv.parent / "prepare_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"out_csv": str(args.out_csv), **summary}, indent=2))


if __name__ == "__main__":
    main()
