from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from PIL import Image

from .solar_geometry import (
    Calibration,
    build_solar_feature_vector,
    compute_clear_sky_power,
    compute_solar_position,
    project_sun_to_image,
)
from ..utils.io import ensure_dir, normalize_config_paths, resolve_project_path, save_json


TIMESTAMP_PATTERN = re.compile(r"_(\d{17})_")


@dataclass
class PrepareSummary:
    n_images_indexed: int
    n_pv_rows: int
    n_candidate_targets: int
    n_samples: int
    n_samples_before_clear_sky_filter: int
    n_excluded_clear_sky_samples: int
    n_missing_image: int
    n_missing_future_image: int
    n_missing_pv: int
    n_filtered_sun_edge: int
    n_split_days_train: int
    n_split_days_val: int
    n_split_days_test: int
    excluded_clear_sky_dates: list[str]
    image_size: tuple[int, int]
    source_image_size: tuple[int, int]


def parse_timestamp_from_name(name: str) -> datetime | None:
    match = TIMESTAMP_PATTERN.search(name)
    if match is None:
        return None
    ts = match.group(1)
    try:
        return datetime.strptime(ts[:14], "%Y%m%d%H%M%S")
    except ValueError:
        return None


def build_camera_index(camera_dir: Path, timezone: str) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for p in sorted(camera_dir.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}:
            continue
        dt = parse_timestamp_from_name(p.name)
        if dt is None:
            continue
        ts = pd.Timestamp(dt).tz_localize(timezone)
        rows.append({"timestamp": ts, "file_path": str(p.resolve())})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="first").reset_index(drop=True)
    return df


def load_camera_index(camera_dir: Path, camera_index_csv: Path | None, timezone: str) -> pd.DataFrame:
    if camera_index_csv is not None and camera_index_csv.exists():
        df = pd.read_csv(camera_index_csv)
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp", "file_path"]).copy()
        if df["timestamp"].dt.tz is None:
            df["timestamp"] = df["timestamp"].dt.tz_localize(timezone)
        else:
            df["timestamp"] = df["timestamp"].dt.tz_convert(timezone)
        df["file_path"] = df["file_path"].map(lambda value: str(resolve_project_path(value, must_exist=False)))
        return df.sort_values("timestamp").reset_index(drop=True)
    return build_camera_index(camera_dir, timezone)


def load_pv_series(pv_csv: Path, timezone: str) -> pd.Series:
    df = pd.read_csv(pv_csv)
    if {"date", "value"}.issubset(df.columns):
        ts_col, value_col = "date", "value"
    elif {"datetime", "power"}.issubset(df.columns):
        ts_col, value_col = "datetime", "power"
    else:
        raise ValueError("PV csv must contain (date,value) or (datetime,power)")
    df["timestamp"] = pd.to_datetime(df[ts_col], errors="coerce")
    df["power_w"] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=["timestamp"]).copy()
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize(timezone)
    else:
        df["timestamp"] = df["timestamp"].dt.tz_convert(timezone)
    df["timestamp"] = df["timestamp"].dt.round("15min")
    series = df.groupby("timestamp", as_index=True)["power_w"].mean().sort_index()
    return series.astype(np.float32)


def _parse_bool_series(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values.fillna(False)
    text = values.astype(str).str.strip().str.lower()
    return text.isin({"true", "1", "yes", "y", "t"})


def load_clear_sky_dates(clear_sky_csv: Path | None, timezone: str) -> set[pd.Timestamp]:
    if clear_sky_csv is None:
        return set()
    csv_path = resolve_project_path(clear_sky_csv, must_exist=True)
    df = pd.read_csv(csv_path)
    if "date" not in df.columns:
        raise ValueError(f"Clear-sky exclusion CSV must contain a date column: {csv_path}")
    if "is_clear_sky" in df.columns:
        df = df[_parse_bool_series(df["is_clear_sky"])].copy()
    dates = pd.to_datetime(df["date"], errors="coerce")
    dates = dates.dropna()
    if dates.empty:
        return set()
    if dates.dt.tz is None:
        dates = dates.dt.tz_localize(timezone)
    else:
        dates = dates.dt.tz_convert(timezone)
    return set(dates.dt.normalize())


def filter_clear_sky_samples(
    df: pd.DataFrame,
    clear_sky_dates: set[pd.Timestamp],
) -> tuple[pd.DataFrame, int]:
    if not clear_sky_dates:
        return df, 0
    anchor_dates = df["ts_anchor"].dt.normalize()
    target_dates = df["ts_target"].dt.normalize()
    keep = ~(anchor_dates.isin(clear_sky_dates) | target_dates.isin(clear_sky_dates))
    excluded = int((~keep).sum())
    return df.loc[keep].reset_index(drop=True), excluded


def _day_split_counts(n_days: int, ratios: dict[str, float]) -> tuple[int, int, int]:
    if n_days <= 0:
        return 0, 0, 0
    if n_days == 1:
        return 1, 0, 0
    if n_days == 2:
        return 1, 0, 1

    n_train = int(n_days * float(ratios["train"]))
    n_val = int(n_days * float(ratios["val"]))
    n_train = max(1, n_train)
    n_val = max(1, n_val) if float(ratios.get("val", 0.0)) > 0 else 0
    if n_train + n_val >= n_days:
        n_val = max(0, min(n_val, n_days - n_train - 1))
    if n_train + n_val >= n_days:
        n_train = max(1, n_days - n_val - 1)
    return n_train, n_val, n_days - n_train - n_val


def assign_splits(df: pd.DataFrame, ratios: dict[str, float], *, by_day: bool = False) -> pd.DataFrame:
    df = df.sort_values("ts_target").reset_index(drop=True).copy()
    df["sample_date"] = df["ts_target"].dt.date.astype(str)
    if by_day:
        days = pd.Index(sorted(df["ts_target"].dt.normalize().unique()))
        n_train_days, n_val_days, _ = _day_split_counts(len(days), ratios)
        train_days = set(days[:n_train_days])
        val_days = set(days[n_train_days : n_train_days + n_val_days])
        split = np.array(["test"] * len(df), dtype=object)
        target_days = df["ts_target"].dt.normalize()
        split[target_days.isin(train_days).to_numpy()] = "train"
        split[target_days.isin(val_days).to_numpy()] = "val"
        df["split"] = split
        return df

    n = len(df)
    n_train = int(n * ratios["train"])
    n_val = int(n * ratios["val"])
    n_test = n - n_train - n_val
    split = np.array(["train"] * n, dtype=object)
    split[n_train : n_train + n_val] = "val"
    split[n_train + n_val :] = "test"
    if n_test <= 0 and n > 0:
        split[-1] = "test"
    df["split"] = split
    return df


def _nearest_image_paths(
    desired_timestamps: Sequence[pd.Timestamp],
    camera_df: pd.DataFrame,
    tolerance_sec: float,
) -> list[str] | None:
    if camera_df.empty:
        return None
    ts_ns = camera_df["timestamp"].astype("int64").to_numpy()
    paths = camera_df["file_path"].astype(str).to_numpy()
    chosen: list[str] = []
    used: set[int] = set()
    for ts in desired_timestamps:
        ns = ts.value
        pos = int(np.searchsorted(ts_ns, ns))
        candidates = [i for i in (pos - 1, pos, pos + 1) if 0 <= i < len(ts_ns) and i not in used]
        if not candidates:
            return None
        best_idx = min(candidates, key=lambda i: abs(int(ts_ns[i]) - ns))
        delta_sec = abs(int(ts_ns[best_idx]) - ns) / 1e9
        if delta_sec > tolerance_sec:
            return None
        chosen.append(str(paths[best_idx]))
        used.add(best_idx)
    return chosen


def _get_image_size(path: str | Path) -> tuple[int, int]:
    with Image.open(path) as im:
        return im.size


def build_samples(config: dict) -> tuple[pd.DataFrame, PrepareSummary]:
    config = normalize_config_paths(config)
    data_cfg = config["data"]
    timezone = data_cfg["timezone"]
    sequence_offsets = data_cfg["sequence_offsets_min"]
    future_image_offsets = data_cfg.get(
        "future_image_offsets_min",
        list(range(1, int(config.get("model", {}).get("future_steps", 15)) + 1)),
    )
    past_pv_offsets = data_cfg["past_pv_offsets_min"]
    target_offset = int(data_cfg["target_offset_min"])
    tolerance_sec = float(data_cfg["image_match_tolerance_sec"])
    sample_hour_start = int(data_cfg.get("sample_hour_start", 8))
    sample_hour_end = int(data_cfg.get("sample_hour_end", 17))
    drop_zero_input_samples = bool(data_cfg.get("drop_zero_input_samples", True))

    camera_df = load_camera_index(
        resolve_project_path(data_cfg["camera_dir"], must_exist=True),
        resolve_project_path(data_cfg["camera_index_csv"], must_exist=False) if data_cfg.get("camera_index_csv") else None,
        timezone,
    )
    if camera_df.empty:
        raise RuntimeError("No camera images found for dataset preparation.")

    source_w, source_h = _get_image_size(camera_df.iloc[0]["file_path"])
    dst_h, dst_w = int(data_cfg["image_size"][0]), int(data_cfg["image_size"][1])

    pv_series = load_pv_series(resolve_project_path(data_cfg["pv_csv"], must_exist=True), timezone)
    calib_raw = Calibration.from_json(resolve_project_path(data_cfg["calibration_json"], must_exist=True))
    calib = calib_raw.rescale(dst_w=dst_w, dst_h=dst_h)
    if data_cfg.get("sun_projection_cx_px") is not None:
        calib = Calibration(
            cx=float(data_cfg["sun_projection_cx_px"]),
            cy=float(data_cfg["sun_projection_cy_px"]),
            f_px_per_rad=float(data_cfg["sun_projection_f_px_per_rad"]),
            lat=calib.lat,
            lon=calib.lon,
            timezone=calib.timezone,
            reference_width=dst_w,
            reference_height=dst_h,
        )

    target_times = pv_series.index
    solar_df = compute_solar_position(target_times, calib)
    clear_sky = compute_clear_sky_power(
        target_times,
        calib,
        peak_power_w=float(data_cfg["peak_power_w"]),
        floor_w=float(data_cfg["clear_sky_floor_w"]),
    )
    solar_lookup = solar_df.set_index("timestamp")
    clear_lookup = clear_sky.to_dict()

    rows: list[dict] = []
    missing_image = 0
    missing_future_image = 0
    missing_pv = 0
    filtered_sun_edge = 0

    for anchor_ts in target_times:
        target_ts = anchor_ts + pd.Timedelta(minutes=target_offset)
        target_hour = target_ts.hour + target_ts.minute / 60.0 + target_ts.second / 3600.0
        if not (sample_hour_start <= target_hour <= sample_hour_end):
            continue
        required_pv_times = [anchor_ts + pd.Timedelta(minutes=o) for o in past_pv_offsets] + [target_ts]
        if not all(ts in pv_series.index for ts in required_pv_times):
            missing_pv += 1
            continue
        pv_values = pv_series.loc[required_pv_times].to_numpy(dtype=np.float32)
        if not np.all(np.isfinite(pv_values)):
            missing_pv += 1
            continue
        if float(pv_values[-1]) < 0.0:
            missing_pv += 1
            continue
        if drop_zero_input_samples and np.any(pv_values[:-1] <= 0.0):
            missing_pv += 1
            continue

        if target_ts not in solar_lookup.index:
            missing_pv += 1
            continue

        anchor_solar = solar_lookup.loc[anchor_ts]
        if float(anchor_solar["elevation_deg"]) < float(data_cfg["min_daylight_elevation_deg"]):
            continue

        img_times = [anchor_ts + pd.Timedelta(minutes=o) for o in sequence_offsets]
        img_paths = _nearest_image_paths(img_times, camera_df, tolerance_sec)
        if img_paths is None:
            missing_image += 1
            continue
        future_img_times = [anchor_ts + pd.Timedelta(minutes=o) for o in future_image_offsets]
        future_img_paths = _nearest_image_paths(future_img_times, camera_df, tolerance_sec)
        if future_img_paths is None:
            missing_future_image += 1
            continue

        target_solar = solar_lookup.loc[target_ts]
        sun_x, sun_y = project_sun_to_image(
            azimuth_deg=float(anchor_solar["azimuth_deg"]),
            zenith_deg=float(anchor_solar["zenith_deg"]),
            calib=calib,
            image_width=dst_w,
            image_height=dst_h,
            azimuth_offset_deg=float(data_cfg.get("azimuth_offset_deg", 0.0)),
            azimuth_clockwise=bool(data_cfg.get("azimuth_clockwise", True)),
            image_offset_x_px=float(data_cfg.get("sun_image_offset_x_px", 0.0)),
            image_offset_y_px=float(data_cfg.get("sun_image_offset_y_px", 0.0)),
        )
        solar_vec = build_solar_feature_vector(
            sun_x_px=float(np.asarray(sun_x).item()),
            sun_y_px=float(np.asarray(sun_y).item()),
            azimuth_deg=float(anchor_solar["azimuth_deg"]),
            zenith_deg=float(anchor_solar["zenith_deg"]),
            image_width=dst_w,
            image_height=dst_h,
        )
        target_sun_x, target_sun_y = project_sun_to_image(
            azimuth_deg=float(target_solar["azimuth_deg"]),
            zenith_deg=float(target_solar["zenith_deg"]),
            calib=calib,
            image_width=dst_w,
            image_height=dst_h,
            azimuth_offset_deg=float(data_cfg.get("azimuth_offset_deg", 0.0)),
            azimuth_clockwise=bool(data_cfg.get("azimuth_clockwise", True)),
            image_offset_x_px=float(data_cfg.get("sun_image_offset_x_px", 0.0)),
            image_offset_y_px=float(data_cfg.get("sun_image_offset_y_px", 0.0)),
        )
        min_edge_margin = min(
            float(np.asarray(sun_x).item()),
            float(np.asarray(sun_y).item()),
            float(dst_w - 1 - np.asarray(sun_x).item()),
            float(dst_h - 1 - np.asarray(sun_y).item()),
        )
        if min_edge_margin < float(data_cfg.get("min_sun_edge_margin_px", 0.0)):
            filtered_sun_edge += 1
            continue

        past_pv = [float(v) for v in pv_values[:-1]]
        target_pv_w = float(pv_values[-1])
        clear_target_w = float(clear_lookup[target_ts])
        if not np.isfinite(clear_target_w) or clear_target_w <= 0:
            missing_pv += 1
            continue
        target_value = (
            target_pv_w / max(clear_target_w, float(data_cfg["clear_sky_floor_w"]))
            if data_cfg.get("use_clear_sky_index", True)
            else target_pv_w
        )

        rows.append(
            {
                "ts_anchor": anchor_ts.isoformat(),
                "ts_target": target_ts.isoformat(),
                "img_paths": json.dumps(img_paths, ensure_ascii=False),
                "future_img_paths": json.dumps(future_img_paths, ensure_ascii=False),
                "past_pv_w": json.dumps(past_pv),
                "target_pv_w": target_pv_w,
                "target_clear_sky_w": clear_target_w,
                "target_value": target_value,
                "use_clear_sky_index": bool(data_cfg.get("use_clear_sky_index", True)),
                "solar_vec": json.dumps(solar_vec.tolist()),
                "sun_x_px": float(np.asarray(sun_x).item()),
                "sun_y_px": float(np.asarray(sun_y).item()),
                "target_sun_x_px": float(np.asarray(target_sun_x).item()),
                "target_sun_y_px": float(np.asarray(target_sun_y).item()),
                "azimuth_deg": float(anchor_solar["azimuth_deg"]),
                "zenith_deg": float(anchor_solar["zenith_deg"]),
                "target_azimuth_deg": float(target_solar["azimuth_deg"]),
                "target_zenith_deg": float(target_solar["zenith_deg"]),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No valid samples were built. Please check camera/PV coverage and tolerances.")
    df["ts_anchor"] = pd.to_datetime(df["ts_anchor"])
    df["ts_target"] = pd.to_datetime(df["ts_target"])
    n_samples_before_clear_sky_filter = int(len(df))
    exclude_clear_sky_days = bool(data_cfg.get("exclude_clear_sky_days", False))
    if exclude_clear_sky_days and not data_cfg.get("clear_sky_exclusion_csv"):
        raise ValueError("data.exclude_clear_sky_days is true, but data.clear_sky_exclusion_csv is not set.")
    clear_sky_dates = (
        load_clear_sky_dates(data_cfg.get("clear_sky_exclusion_csv"), timezone)
        if exclude_clear_sky_days
        else set()
    )
    if exclude_clear_sky_days:
        df, n_excluded_clear_sky_samples = filter_clear_sky_samples(df, clear_sky_dates)
        if df.empty:
            raise RuntimeError("No valid samples remain after clear-sky day exclusion.")
    else:
        n_excluded_clear_sky_samples = 0
    df = assign_splits(
        df,
        data_cfg["chronological_split"],
        by_day=bool(data_cfg.get("split_by_day", False)),
    )
    split_day_counts = (
        df.groupby("split")["sample_date"].nunique().reindex(["train", "val", "test"], fill_value=0).astype(int)
    )

    summary = PrepareSummary(
        n_images_indexed=int(len(camera_df)),
        n_pv_rows=int(len(pv_series)),
        n_candidate_targets=int(len(target_times)),
        n_samples=int(len(df)),
        n_samples_before_clear_sky_filter=n_samples_before_clear_sky_filter,
        n_excluded_clear_sky_samples=int(n_excluded_clear_sky_samples),
        n_missing_image=int(missing_image),
        n_missing_future_image=int(missing_future_image),
        n_missing_pv=int(missing_pv),
        n_filtered_sun_edge=int(filtered_sun_edge),
        n_split_days_train=int(split_day_counts["train"]),
        n_split_days_val=int(split_day_counts["val"]),
        n_split_days_test=int(split_day_counts["test"]),
        excluded_clear_sky_dates=sorted(ts.date().isoformat() for ts in clear_sky_dates),
        image_size=(dst_h, dst_w),
        source_image_size=(source_h, source_w),
    )
    return df, summary


def save_samples(df: pd.DataFrame, summary: PrepareSummary, config: dict) -> None:
    config = normalize_config_paths(config)
    out_dir = ensure_dir(resolve_project_path(config["data"]["artifact_dir"], must_exist=False))
    samples_csv = resolve_project_path(config["data"]["samples_csv"], must_exist=False)
    samples_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(samples_csv, index=False)
    save_json(
        out_dir / "preprocess_summary.json",
        {
            "n_images_indexed": summary.n_images_indexed,
            "n_pv_rows": summary.n_pv_rows,
            "n_candidate_targets": summary.n_candidate_targets,
            "n_samples": summary.n_samples,
            "n_samples_before_clear_sky_filter": summary.n_samples_before_clear_sky_filter,
            "n_excluded_clear_sky_samples": summary.n_excluded_clear_sky_samples,
            "n_missing_image": summary.n_missing_image,
            "n_missing_future_image": summary.n_missing_future_image,
            "n_missing_pv": summary.n_missing_pv,
            "n_filtered_sun_edge": summary.n_filtered_sun_edge,
            "n_split_days_train": summary.n_split_days_train,
            "n_split_days_val": summary.n_split_days_val,
            "n_split_days_test": summary.n_split_days_test,
            "excluded_clear_sky_dates": summary.excluded_clear_sky_dates,
            "image_size": list(summary.image_size),
            "source_image_size": list(summary.source_image_size),
            "sample_hour_start": int(config["data"].get("sample_hour_start", 8)),
            "sample_hour_end": int(config["data"].get("sample_hour_end", 17)),
            "drop_zero_input_samples": bool(config["data"].get("drop_zero_input_samples", True)),
            "future_image_offsets_min": list(config["data"].get("future_image_offsets_min", [])),
            "exclude_clear_sky_days": bool(config["data"].get("exclude_clear_sky_days", False)),
            "clear_sky_exclusion_csv": str(resolve_project_path(config["data"]["clear_sky_exclusion_csv"], must_exist=False))
            if config["data"].get("clear_sky_exclusion_csv")
            else None,
            "split_by_day": bool(config["data"].get("split_by_day", False)),
            "samples_csv": str(samples_csv),
        },
    )
