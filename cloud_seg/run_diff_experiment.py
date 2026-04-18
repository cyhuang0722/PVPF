from __future__ import annotations

import json
import re
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, time
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import binary_closing as ndi_binary_closing
from scipy.ndimage import binary_dilation as ndi_binary_dilation
from scipy.ndimage import binary_opening as ndi_binary_opening
from scipy.ndimage import gaussian_filter, label


MONTH_MAP = {
    "Jan": 1,
    "Feb": 2,
    "Mar": 3,
    "Apr": 4,
    "May": 5,
    "Jun": 6,
    "Jul": 7,
    "Aug": 8,
    "Sep": 9,
    "Oct": 10,
    "Nov": 11,
    "Dec": 12,
}


@dataclass(frozen=True)
class Config:
    image_root: Path
    weather_csv: Path
    clear_sky_csv: Path
    sky_mask_path: Path
    output_dir: Path
    image_size: int = 256
    cloudy_label: int = 2
    max_cloudy_days: int = 20
    start_hour: int = 8
    end_hour: int = 17
    diff_threshold: float = 0.05
    trend_sigma_px: float = 100.0
    opening_radius: int = 2
    closing_radius: int = 3
    min_component_size: int = 200
    cloud_saturation_threshold: float = 0.45
    cloud_value_threshold: float = 0.16
    blue_saturation_threshold: float = 0.18
    blue_value_threshold: float = 0.12
    blue_fraction_partly_threshold: float = 0.12
    blue_fraction_clear_threshold: float = 0.45
    blue_fraction_overcast_threshold: float = 0.08
    gray_fraction_overcast_threshold: float = 0.65
    blue_fraction_broken_threshold: float = 0.08
    blue_fraction_broken_max: float = 0.40
    gray_fraction_broken_threshold: float = 0.85
    rbr_fraction_clear_threshold: float = 0.08
    local_p95_clear_threshold: float = 0.16
    blue_guard_radius: int = 5
    sun_guard_radius: int = 22
    partly_color_saturation_threshold: float = 0.38
    partly_color_value_threshold: float = 0.18
    partly_color_local_floor: float = -0.03
    bright_cloud_threshold: float = 0.10
    bright_cloud_saturation_threshold: float = 0.58


def parse_camera_timestamp(path: Path) -> datetime | None:
    match = re.search(r"_(\d{17})_TIMING", path.name, flags=re.IGNORECASE)
    if match is None:
        return None
    return datetime.strptime(match.group(1), "%Y%m%d%H%M%S%f")


def infer_dataset_year(image_root: Path) -> int:
    year_dirs = sorted([path for path in image_root.iterdir() if path.is_dir() and path.name.isdigit()])
    if not year_dirs:
        raise RuntimeError(f"No year directories found under {image_root}")
    return int(year_dirs[0].name)


def load_weather_labels(weather_csv: Path, year: int) -> pd.DataFrame:
    weather_df = pd.read_csv(weather_csv)
    parsed_dates = []
    for raw in weather_df["Date"].astype(str):
        day_str, month_str = raw.split("-")
        parsed_dates.append(pd.Timestamp(year=year, month=MONTH_MAP[month_str], day=int(day_str)))
    weather_df["date"] = pd.DatetimeIndex(pd.to_datetime(parsed_dates)).normalize()
    weather_df["weather_label"] = weather_df["Weather"].astype(int)
    return weather_df[["date", "weather_label"]].sort_values("date").reset_index(drop=True)


def load_clear_sky_windows(clear_sky_csv: Path) -> pd.DataFrame:
    clear_df = pd.read_csv(clear_sky_csv)
    required_columns = {"date", "start_time", "end_time", "notes"}
    missing_columns = required_columns - set(clear_df.columns)
    if missing_columns:
        raise RuntimeError(f"{clear_sky_csv} is missing columns: {sorted(missing_columns)}")
    clear_df["date"] = pd.to_datetime(clear_df["date"]).dt.normalize()
    clear_df["start_dt"] = pd.to_datetime(clear_df["date"].dt.strftime("%Y-%m-%d") + " " + clear_df["start_time"].astype(str))
    clear_df["end_dt"] = pd.to_datetime(clear_df["date"].dt.strftime("%Y-%m-%d") + " " + clear_df["end_time"].astype(str))
    return clear_df.sort_values(["date", "start_dt"]).reset_index(drop=True)


def build_manifest(image_root: Path, weather_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in sorted(image_root.rglob("*.jpg")):
        ts = parse_camera_timestamp(path)
        if ts is None:
            continue
        rows.append(
            {
                "image_path": str(path),
                "timestamp": ts,
                "date": pd.Timestamp(ts).normalize(),
                "hour": int(ts.hour),
            }
        )
    if not rows:
        raise RuntimeError(f"No camera images found under {image_root}")
    manifest = pd.DataFrame.from_records(rows).sort_values("timestamp").reset_index(drop=True)
    manifest = manifest.merge(weather_df, on="date", how="left")
    manifest["weather_label"] = manifest["weather_label"].fillna(-1).astype(int)
    return manifest


def load_rgb_image(path: Path, image_size: int) -> np.ndarray:
    with Image.open(path) as im:
        rgb = im.convert("RGB")
        if rgb.size != (image_size, image_size):
            rgb = rgb.resize((image_size, image_size), resample=Image.BILINEAR)
        return np.asarray(rgb, dtype=np.float32) / 255.0


def load_mask(mask_path: Path, image_size: int) -> np.ndarray:
    with Image.open(mask_path) as im:
        mask = im.convert("L")
        if mask.size != (image_size, image_size):
            mask = mask.resize((image_size, image_size), resample=Image.NEAREST)
        return (np.asarray(mask, dtype=np.float32) / 255.0) >= 0.5


def compute_saturation_value(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rgb_min = np.min(rgb, axis=-1)
    rgb_max = np.max(rgb, axis=-1)
    saturation = (rgb_max - rgb_min) / np.clip(rgb_max, 1e-6, None)
    value = rgb_max
    return saturation.astype(np.float32), value.astype(np.float32)


def compute_blue_sky_mask(rgb: np.ndarray, sky_mask: np.ndarray, cfg: Config) -> np.ndarray:
    red = rgb[..., 0]
    green = rgb[..., 1]
    blue = rgb[..., 2]
    saturation, value = compute_saturation_value(rgb)
    blue_dominant = (blue >= red * 1.05) & (blue >= green * 0.92)
    return sky_mask & blue_dominant & (saturation >= cfg.blue_saturation_threshold) & (value >= cfg.blue_value_threshold)


def compute_sky_normalized_rbr(rgb: np.ndarray, stat_mask: np.ndarray) -> np.ndarray:
    red = rgb[..., 0]
    blue = rgb[..., 2]
    red_mean = float(np.mean(red[stat_mask]))
    blue_mean = float(np.mean(blue[stat_mask]))
    red_norm = red / max(red_mean, 1e-5)
    blue_norm = blue / max(blue_mean, 1e-5)
    return (red_norm / np.clip(blue_norm, 1e-6, None)).astype(np.float32)


def match_clear_rbr_to_current(cloudy_rbr: np.ndarray, clear_rbr: np.ndarray, sky_mask: np.ndarray) -> tuple[np.ndarray, float]:
    cloudy_median = float(np.median(cloudy_rbr[sky_mask]))
    clear_median = float(np.median(clear_rbr[sky_mask]))
    scale = cloudy_median / max(clear_median, 1e-6)
    return (clear_rbr * scale).astype(np.float32), scale


def disk_structure(radius: int) -> np.ndarray:
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return (xx * xx + yy * yy) <= radius * radius


def remove_small_components(mask: np.ndarray, min_size: int) -> np.ndarray:
    if min_size <= 1:
        return mask.astype(bool)
    labeled, n_components = label(mask.astype(bool))
    if n_components == 0:
        return mask.astype(bool)
    counts = np.bincount(labeled.ravel())
    keep = counts >= int(min_size)
    keep[0] = False
    return keep[labeled]


def estimate_sun_guard(clear_rgb: np.ndarray, sky_mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0 or not np.any(sky_mask):
        return np.zeros_like(sky_mask, dtype=bool)
    value = np.max(clear_rgb, axis=-1)
    masked_value = np.where(sky_mask, value, -np.inf)
    y, x = np.unravel_index(int(np.argmax(masked_value)), masked_value.shape)
    yy, xx = np.ogrid[: sky_mask.shape[0], : sky_mask.shape[1]]
    return ((yy - y) ** 2 + (xx - x) ** 2 <= radius * radius) & sky_mask


def compute_normalized_gray_diff(cloudy_rgb: np.ndarray, clear_rgb: np.ndarray, sky_mask: np.ndarray) -> np.ndarray:
    cloudy_gray = 0.299 * cloudy_rgb[..., 0] + 0.587 * cloudy_rgb[..., 1] + 0.114 * cloudy_rgb[..., 2]
    clear_gray = 0.299 * clear_rgb[..., 0] + 0.587 * clear_rgb[..., 1] + 0.114 * clear_rgb[..., 2]
    cloudy_norm = cloudy_gray / max(float(np.median(cloudy_gray[sky_mask])), 1e-6)
    clear_norm = clear_gray / max(float(np.median(clear_gray[sky_mask])), 1e-6)
    return (cloudy_norm - clear_norm).astype(np.float32)


def detrend_with_mask(values: np.ndarray, sky_mask: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    weights = sky_mask.astype(np.float32)
    weighted_values = np.where(sky_mask, values, 0.0).astype(np.float32)
    smooth_values = gaussian_filter(weighted_values, sigma=float(sigma), truncate=2.0)
    smooth_weights = gaussian_filter(weights, sigma=float(sigma), truncate=2.0)
    trend = np.divide(smooth_values, smooth_weights, out=np.zeros_like(smooth_values), where=smooth_weights > 1e-6)
    local = values - trend
    return trend.astype(np.float32), local.astype(np.float32)


def clean_cloud_mask(mask: np.ndarray, sky_mask: np.ndarray, cfg: Config) -> np.ndarray:
    opened = ndi_binary_opening(mask & sky_mask, structure=disk_structure(cfg.opening_radius))
    closed = ndi_binary_closing(opened, structure=disk_structure(cfg.closing_radius))
    cleaned = remove_small_components(closed & sky_mask, cfg.min_component_size)
    return cleaned & sky_mask


def segment_cloud_rbr_final(
    cloudy_rgb: np.ndarray,
    clear_rgb: np.ndarray,
    sky_mask: np.ndarray,
    cfg: Config,
) -> dict[str, object]:
    cloudy_rbr = compute_sky_normalized_rbr(cloudy_rgb, sky_mask)
    clear_rbr = compute_sky_normalized_rbr(clear_rgb, sky_mask)
    clear_rbr_norm, clear_scale = match_clear_rbr_to_current(cloudy_rbr, clear_rbr, sky_mask)
    raw_diff = cloudy_rbr - clear_rbr_norm
    trend, local_diff = detrend_with_mask(raw_diff, sky_mask, cfg.trend_sigma_px)
    saturation, value = compute_saturation_value(cloudy_rgb)
    color_mask = sky_mask & (saturation <= cfg.cloud_saturation_threshold) & (value >= cfg.cloud_value_threshold)
    partly_color_mask = (
        sky_mask
        & (saturation <= cfg.partly_color_saturation_threshold)
        & (value >= cfg.partly_color_value_threshold)
        & (local_diff >= cfg.partly_color_local_floor)
    )
    gray_diff = compute_normalized_gray_diff(cloudy_rgb, clear_rgb, sky_mask)
    blue_sky_mask = compute_blue_sky_mask(cloudy_rgb, sky_mask, cfg)
    rbr_raw_mask = sky_mask & (local_diff > cfg.diff_threshold)
    sun_guard = estimate_sun_guard(clear_rgb, sky_mask, cfg.sun_guard_radius)
    bright_cloud_mask = (
        sky_mask
        & (gray_diff > cfg.bright_cloud_threshold)
        & (saturation <= cfg.bright_cloud_saturation_threshold)
        & (value >= cfg.cloud_value_threshold)
    )

    blue_fraction = float(np.mean(blue_sky_mask[sky_mask]))
    gray_fraction = float(np.mean(color_mask[sky_mask]))
    rbr_raw_fraction = float(np.mean(rbr_raw_mask[sky_mask]))
    local_diff_p95 = float(np.quantile(local_diff[sky_mask], 0.95))
    blue_guard = ndi_binary_dilation(blue_sky_mask, structure=disk_structure(cfg.blue_guard_radius)) & sky_mask
    if blue_fraction >= cfg.blue_fraction_clear_threshold and (
        rbr_raw_fraction <= cfg.rbr_fraction_clear_threshold or local_diff_p95 <= cfg.local_p95_clear_threshold
    ):
        scene_type = "clear_sky"
        raw_mask = np.zeros_like(sky_mask, dtype=bool)
    elif blue_fraction <= cfg.blue_fraction_overcast_threshold and gray_fraction >= cfg.gray_fraction_overcast_threshold:
        scene_type = "overcast"
        raw_mask = color_mask & ~blue_guard
    elif (
        cfg.blue_fraction_broken_threshold <= blue_fraction <= cfg.blue_fraction_broken_max
        and gray_fraction >= cfg.gray_fraction_broken_threshold
    ):
        scene_type = "broken_cloudy"
        non_blue_cloud = sky_mask & ~blue_guard & (value >= cfg.cloud_value_threshold)
        raw_mask = (rbr_raw_mask | non_blue_cloud | bright_cloud_mask) & ~blue_guard & ~sun_guard
    elif blue_fraction >= cfg.blue_fraction_partly_threshold:
        scene_type = "partly_cloudy"
        scattered_cloud = partly_color_mask & ~blue_guard & ~sun_guard
        bright_cloud = bright_cloud_mask & ~blue_guard & ~sun_guard
        raw_mask = ((rbr_raw_mask | scattered_cloud | bright_cloud) & ~blue_guard & ~sun_guard)
    else:
        scene_type = "mixed"
        raw_mask = rbr_raw_mask

    cloud_mask = clean_cloud_mask(raw_mask, sky_mask, cfg)
    return {
        "cloudy_rbr": cloudy_rbr,
        "clear_rbr_norm": clear_rbr_norm,
        "raw_diff": raw_diff.astype(np.float32),
        "trend": trend,
        "local_diff": local_diff,
        "blue_sky_mask": blue_sky_mask,
        "color_mask": color_mask,
        "sun_guard": sun_guard,
        "bright_cloud_mask": bright_cloud_mask,
        "rbr_raw_mask": rbr_raw_mask,
        "raw_mask": raw_mask,
        "cloud_mask": cloud_mask,
        "clear_scale": clear_scale,
        "scene_type": scene_type,
        "blue_fraction": blue_fraction,
        "gray_fraction": gray_fraction,
        "rbr_raw_fraction": rbr_raw_fraction,
        "local_diff_p95": local_diff_p95,
    }


def ensure_dirs(output_dir: Path) -> dict[str, Path]:
    for generated_subdir in ["days", "manifests", "pairs", "review_pngs"]:
        stale_dir = output_dir / generated_subdir
        if stale_dir.exists():
            shutil.rmtree(stale_dir)
    dirs = {
        "base": output_dir,
        "review_pngs": output_dir / "review_pngs",
        "manifests": output_dir / "manifests",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def nearest_row_for_hour(day_df: pd.DataFrame, hour: int) -> pd.Series | None:
    candidates = day_df[day_df["hour"] == hour].copy()
    if candidates.empty:
        return None
    target_dt = pd.Timestamp.combine(pd.Timestamp(candidates.iloc[0]["date"]).date(), time(hour=hour))
    candidates["delta_seconds"] = (pd.to_datetime(candidates["timestamp"]) - target_dt).abs().dt.total_seconds()
    return candidates.sort_values(["delta_seconds", "timestamp"]).iloc[0]


def nearest_clear_row_for_window(day_df: pd.DataFrame, hour: int, start_dt: pd.Timestamp, end_dt: pd.Timestamp) -> pd.Series | None:
    candidates = day_df[
        (day_df["hour"] == hour)
        & (pd.to_datetime(day_df["timestamp"]) >= start_dt)
        & (pd.to_datetime(day_df["timestamp"]) <= end_dt)
    ].copy()
    if candidates.empty:
        return None
    target_dt = pd.Timestamp.combine(pd.Timestamp(start_dt).date(), time(hour=hour))
    candidates["delta_seconds"] = (pd.to_datetime(candidates["timestamp"]) - target_dt).abs().dt.total_seconds()
    return candidates.sort_values(["delta_seconds", "timestamp"]).iloc[0]


def find_reference_for_hour(
    manifest: pd.DataFrame,
    clear_windows: pd.DataFrame,
    cloudy_day: pd.Timestamp,
    hour: int,
) -> tuple[pd.Series, pd.Series] | None:
    previous_windows = clear_windows[clear_windows["date"] < cloudy_day].sort_values(["date", "start_dt"], ascending=False)
    for window in previous_windows.itertuples(index=False):
        clear_df = manifest[manifest["date"] == window.date].copy()
        clear_row = nearest_clear_row_for_window(clear_df, hour, window.start_dt, window.end_dt)
        if clear_row is not None:
            return clear_row, pd.Series(window._asdict())
    return None


def assert_reference_within_window(pair_df: pd.DataFrame) -> None:
    clear_ts = pd.to_datetime(pair_df["clear_timestamp"])
    window_start = pd.to_datetime(pair_df["clear_window_start"])
    window_end = pd.to_datetime(pair_df["clear_window_end"])
    invalid = pair_df[(clear_ts < window_start) | (clear_ts > window_end)]
    if not invalid.empty:
        sample = invalid.iloc[0]
        raise RuntimeError(
            "Found a clear reference outside its clear-sky window: "
            f"cloudy_date={sample['cloudy_date']}, hour={sample['hour']}, "
            f"clear_timestamp={sample['clear_timestamp']}, "
            f"window=[{sample['clear_window_start']}, {sample['clear_window_end']}]"
        )


def build_day_pair_table(manifest: pd.DataFrame, clear_windows: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    hourly_range = list(range(cfg.start_hour, cfg.end_hour + 1))
    weather_by_date = manifest[["date", "weather_label"]].drop_duplicates().sort_values("date")
    cloudy_days = weather_by_date[weather_by_date["weather_label"] == cfg.cloudy_label]["date"].tolist()
    paired_rows: list[dict[str, object]] = []
    selected_days = 0

    for cloudy_day in cloudy_days:
        cloudy_df = manifest[manifest["date"] == cloudy_day].copy()

        hour_rows: list[dict[str, object]] = []
        for hour in hourly_range:
            cloudy_row = nearest_row_for_hour(cloudy_df, hour)
            reference = find_reference_for_hour(manifest, clear_windows, cloudy_day, hour)
            if cloudy_row is None or reference is None:
                hour_rows = []
                break
            clear_row, clear_window = reference
            hour_rows.append(
                {
                    "cloudy_date": pd.Timestamp(cloudy_day).strftime("%Y-%m-%d"),
                    "clear_date": pd.Timestamp(clear_window["date"]).strftime("%Y-%m-%d"),
                    "clear_window_start": pd.Timestamp(clear_window["start_dt"]).strftime("%Y-%m-%d %H:%M:%S"),
                    "clear_window_end": pd.Timestamp(clear_window["end_dt"]).strftime("%Y-%m-%d %H:%M:%S"),
                    "clear_window_notes": clear_window["notes"],
                    "hour": hour,
                    "cloudy_timestamp": pd.Timestamp(cloudy_row["timestamp"]).strftime("%Y-%m-%d %H:%M:%S.%f"),
                    "clear_timestamp": pd.Timestamp(clear_row["timestamp"]).strftime("%Y-%m-%d %H:%M:%S.%f"),
                    "cloudy_image_path": cloudy_row["image_path"],
                    "clear_image_path": clear_row["image_path"],
                }
            )
        if not hour_rows:
            continue
        paired_rows.extend(hour_rows)
        selected_days += 1
        if selected_days >= cfg.max_cloudy_days:
            break

    if not paired_rows:
        raise RuntimeError("No cloudy/clear day pairs satisfy the hourly requirements.")
    pair_df = pd.DataFrame.from_records(paired_rows)
    assert_reference_within_window(pair_df)
    return pair_df


def save_pair_figure(
    out_path: Path,
    cloudy_rgb: np.ndarray,
    clear_rgb: np.ndarray,
    cloudy_rbr: np.ndarray,
    clear_rbr_norm: np.ndarray,
    raw_diff: np.ndarray,
    trend: np.ndarray,
    local_diff: np.ndarray,
    blue_sky_mask: np.ndarray,
    color_mask: np.ndarray,
    rbr_raw_mask: np.ndarray,
    raw_mask: np.ndarray,
    diff_mask: np.ndarray,
    sky_mask: np.ndarray,
    title: str,
) -> None:
    masked_local_diff = np.where(sky_mask, local_diff, np.nan)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.ravel()
    axes[0].imshow(np.clip(cloudy_rgb, 0.0, 1.0))
    axes[0].set_title("Cloudy RGB")
    axes[1].imshow(np.clip(clear_rgb, 0.0, 1.0))
    axes[1].set_title("Matched Clear RGB")
    axes[2].imshow(blue_sky_mask, cmap="gray")
    axes[2].set_title("Blue-Sky Mask")
    im2 = axes[3].imshow(masked_local_diff, cmap="coolwarm", vmin=-0.15, vmax=0.15)
    axes[3].set_title("Local RBR Residual")
    fig.colorbar(im2, ax=axes[3], fraction=0.046)
    axes[4].imshow(color_mask, cmap="gray")
    axes[4].set_title("Cloud Color Mask")
    axes[5].imshow(rbr_raw_mask, cmap="gray")
    axes[5].set_title("RBR Residual Mask")
    axes[6].imshow(diff_mask, cmap="gray")
    axes[6].set_title("Decision Mask")
    overlay = np.clip(cloudy_rgb, 0.0, 1.0).copy()
    overlay[diff_mask] = 0.55 * overlay[diff_mask] + 0.45 * np.array([1.0, 0.1, 0.0], dtype=np.float32)
    axes[7].imshow(overlay)
    axes[7].set_title("Overlay")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    root = Path("/Users/huangchouyue/Projects/PVPF")
    cfg = Config(
        image_root=root / "data/camera_data/resized_256",
        weather_csv=root / "data/weather.csv",
        clear_sky_csv=root / "data/clear_sky.csv",
        sky_mask_path=root / "data/sky_mask.png",
        output_dir=root / "cloud_seg2/outputs_decision",
    )
    dirs = ensure_dirs(cfg.output_dir)
    year = infer_dataset_year(cfg.image_root)
    weather_df = load_weather_labels(cfg.weather_csv, year)
    clear_windows = load_clear_sky_windows(cfg.clear_sky_csv)
    manifest = build_manifest(cfg.image_root, weather_df)
    pair_df = build_day_pair_table(manifest, clear_windows, cfg)
    sky_mask = load_mask(cfg.sky_mask_path, cfg.image_size)

    pair_df.to_csv(dirs["manifests"] / "selected_pairs.csv", index=False)

    summary_rows: list[dict[str, object]] = []
    day_rows: list[dict[str, object]] = []

    for cloudy_date, day_df in pair_df.groupby("cloudy_date", sort=True):
        clear_refs = sorted(day_df["clear_date"].unique().tolist())
        clear_ref_label = ",".join(clear_refs)
        hourly_cloud_fraction: list[float] = []
        hourly_raw_cloud_fraction: list[float] = []
        hourly_local_diff_mean: list[float] = []
        hourly_local_diff_p95: list[float] = []
        hourly_raw_diff_mean: list[float] = []
        hourly_scale: list[float] = []

        for row in day_df.sort_values("hour").itertuples(index=False):
            cloudy_rgb = load_rgb_image(Path(row.cloudy_image_path), cfg.image_size)
            clear_rgb = load_rgb_image(Path(row.clear_image_path), cfg.image_size)
            seg = segment_cloud_rbr_final(cloudy_rgb, clear_rgb, sky_mask, cfg)
            cloudy_rbr = seg["cloudy_rbr"]
            clear_rbr_norm = seg["clear_rbr_norm"]
            raw_diff = seg["raw_diff"]
            trend = seg["trend"]
            local_diff = seg["local_diff"]
            blue_sky_mask = seg["blue_sky_mask"]
            color_mask = seg["color_mask"]
            rbr_raw_mask = seg["rbr_raw_mask"]
            raw_mask = seg["raw_mask"]
            diff_mask = seg["cloud_mask"]
            clear_scale = float(seg["clear_scale"])
            scene_type = str(seg["scene_type"])
            blue_fraction = float(seg["blue_fraction"])
            gray_fraction = float(seg["gray_fraction"])

            valid_raw_diff = raw_diff[sky_mask]
            valid_local_diff = local_diff[sky_mask]
            color_fraction = float(np.mean(color_mask[sky_mask]))
            rbr_raw_fraction = float(np.mean(rbr_raw_mask[sky_mask]))
            raw_cloud_fraction = float(np.mean(raw_mask[sky_mask]))
            cloud_fraction = float(np.mean(diff_mask[sky_mask]))
            raw_diff_mean = float(np.mean(valid_raw_diff))
            local_diff_mean = float(np.mean(valid_local_diff))
            local_diff_p95 = float(np.quantile(valid_local_diff, 0.95))
            hourly_raw_cloud_fraction.append(raw_cloud_fraction)
            hourly_cloud_fraction.append(cloud_fraction)
            hourly_raw_diff_mean.append(raw_diff_mean)
            hourly_local_diff_mean.append(local_diff_mean)
            hourly_local_diff_p95.append(local_diff_p95)
            hourly_scale.append(clear_scale)

            hour_stub = f"{row.hour:02d}00"
            out_name = f"{cloudy_date}_{hour_stub}_ref_{row.clear_date}.png"
            save_pair_figure(
                dirs["review_pngs"] / out_name,
                cloudy_rgb=cloudy_rgb,
                clear_rgb=clear_rgb,
                cloudy_rbr=cloudy_rbr,
                clear_rbr_norm=clear_rbr_norm,
                raw_diff=raw_diff,
                trend=trend,
                local_diff=local_diff,
                blue_sky_mask=blue_sky_mask,
                color_mask=color_mask,
                rbr_raw_mask=rbr_raw_mask,
                raw_mask=raw_mask,
                diff_mask=diff_mask,
                sky_mask=sky_mask,
                title=(
                    f"{cloudy_date} {row.hour:02d}:00 | ref {row.clear_date} | "
                    f"{scene_type} | blue={blue_fraction:.2f} gray={gray_fraction:.2f} | tau={cfg.diff_threshold:.2f}"
                ),
            )
            summary_rows.append(
                {
                    "cloudy_date": cloudy_date,
                    "clear_date": row.clear_date,
                    "clear_window_start": row.clear_window_start,
                    "clear_window_end": row.clear_window_end,
                    "clear_window_notes": row.clear_window_notes,
                    "hour": int(row.hour),
                    "cloudy_timestamp": row.cloudy_timestamp,
                    "clear_timestamp": row.clear_timestamp,
                    "clear_rbr_scale": clear_scale,
                    "scene_type": scene_type,
                    "blue_fraction": blue_fraction,
                    "gray_fraction": gray_fraction,
                    "color_fraction": color_fraction,
                    "rbr_raw_fraction": rbr_raw_fraction,
                    "raw_cloud_fraction": raw_cloud_fraction,
                    "cloud_fraction": cloud_fraction,
                    "raw_diff_mean": raw_diff_mean,
                    "local_diff_mean": local_diff_mean,
                    "local_diff_p95": local_diff_p95,
                    "cloudy_image_path": row.cloudy_image_path,
                    "clear_image_path": row.clear_image_path,
                    "png_path": str(dirs["review_pngs"] / out_name),
                }
            )

        day_rows.append(
            {
                "cloudy_date": cloudy_date,
                "clear_references": clear_ref_label,
                "n_hours": int(len(day_df)),
                "mean_raw_cloud_fraction": float(np.mean(hourly_raw_cloud_fraction)),
                "mean_cloud_fraction": float(np.mean(hourly_cloud_fraction)),
                "mean_raw_diff": float(np.mean(hourly_raw_diff_mean)),
                "mean_local_diff": float(np.mean(hourly_local_diff_mean)),
                "mean_local_diff_p95": float(np.mean(hourly_local_diff_p95)),
                "mean_clear_rbr_scale": float(np.mean(hourly_scale)),
            }
        )

    pd.DataFrame.from_records(summary_rows).to_csv(dirs["manifests"] / "hourly_summary.csv", index=False)
    pd.DataFrame.from_records(day_rows).to_csv(dirs["manifests"] / "daily_summary.csv", index=False)

    summary = {
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()},
        "selected_cloudy_days": [row["cloudy_date"] for row in day_rows],
        "n_cloudy_days": len(day_rows),
        "n_hourly_pairs": len(summary_rows),
        "output_dir": str(cfg.output_dir),
    }
    (dirs["base"] / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
