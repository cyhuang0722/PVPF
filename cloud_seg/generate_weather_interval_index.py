from __future__ import annotations

import argparse
import ast
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[1]
TAGS = ("clear_sky", "partly_cloudy", "cloudy", "overcast")


@dataclass(frozen=True)
class Config:
    samples_csv: Path
    sky_mask_path: Path
    clear_sky_days_csv: Path
    output_dir: Path
    image_size: int = 256
    start_time: str = "08:30"
    end_time: str = "16:00"
    analysis_fov_deg: float = 140.0
    camera_fov_deg: float = 180.0
    max_sun_zenith_deg: float = 140.0
    sample_per_class: int = 30
    clear_csi_threshold: float = 0.58
    clear_blue_threshold: float = 0.45
    clear_max_gray_fraction: float = 0.72
    clear_max_bright_white_fraction: float = 0.16
    clear_max_cloud_proxy: float = 0.48
    overcast_csi_threshold: float = 0.28
    overcast_low_blue_threshold: float = 0.12
    overcast_min_gray_fraction: float = 0.88
    overcast_sealed_blue_threshold: float = 0.02
    overcast_sealed_gray_threshold: float = 0.98
    overcast_sealed_cloud_proxy_threshold: float = 0.98
    cloudy_csi_threshold: float = 0.50
    cloudy_min_gray_fraction: float = 0.72
    cloudy_max_blue_fraction: float = 0.35
    partly_min_blue_fraction: float = 0.22
    partly_min_csi: float = 0.32
    partly_low_csi_threshold: float = 0.40
    partly_min_gray_fraction: float = 0.62
    partly_min_bright_white_fraction: float = 0.08
    partly_min_cloud_proxy: float = 0.58
    partly_max_blue_fraction: float = 0.60


def _parse_list(value: object) -> list:
    if isinstance(value, list):
        return value
    if not isinstance(value, str):
        return []
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return ast.literal_eval(value)


def _parse_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})


def _load_mask(mask_path: Path, image_size: int) -> np.ndarray:
    if not mask_path.exists():
        return np.ones((image_size, image_size), dtype=bool)
    with Image.open(mask_path) as im:
        mask = im.convert("L")
        if mask.size != (image_size, image_size):
            mask = mask.resize((image_size, image_size), resample=Image.NEAREST)
        return (np.asarray(mask, dtype=np.float32) / 255.0) >= 0.5


def _limit_mask_to_fov(sky_mask: np.ndarray, analysis_fov_deg: float, camera_fov_deg: float) -> np.ndarray:
    if analysis_fov_deg <= 0 or camera_fov_deg <= 0 or analysis_fov_deg >= camera_fov_deg:
        return sky_mask
    yy, xx = np.indices(sky_mask.shape)
    sky_y, sky_x = np.where(sky_mask)
    if sky_x.size == 0:
        return sky_mask
    cx = (sky_mask.shape[1] - 1) / 2.0
    cy = (sky_mask.shape[0] - 1) / 2.0
    sky_radius = np.hypot(sky_x - cx, sky_y - cy)
    max_radius = float(np.max(sky_radius))
    analysis_radius = max_radius * (analysis_fov_deg / camera_fov_deg)
    fov_mask = np.hypot(xx - cx, yy - cy) <= analysis_radius
    return sky_mask & fov_mask


def _load_rgb(path: Path, image_size: int) -> np.ndarray | None:
    try:
        with Image.open(path) as im:
            rgb = im.convert("RGB")
            if rgb.size != (image_size, image_size):
                rgb = rgb.resize((image_size, image_size), resample=Image.BILINEAR)
            return np.asarray(rgb, dtype=np.float32) / 255.0
    except OSError:
        return None


def _rbr_map(rgb: np.ndarray, sky_mask: np.ndarray) -> np.ndarray:
    red = rgb[..., 0]
    blue = rgb[..., 2]
    rbr = red / np.clip(blue, 0.05, None)
    rbr = np.clip(rbr, 0.0, 4.0)
    sky_values = rbr[sky_mask]
    median = float(np.median(sky_values)) if sky_values.size else 1.0
    return (rbr / max(median, 1e-6)).astype(np.float32)


def _temporal_rbr_metrics(image_paths: list[Path], sky_mask: np.ndarray, image_size: int) -> dict[str, float]:
    frames: list[np.ndarray] = []
    for path in image_paths:
        rgb = _load_rgb(path, image_size)
        if rgb is not None:
            frames.append(_rbr_map(rgb, sky_mask))
    if len(frames) < 2:
        return {
            "rbr_diff_mean": np.nan,
            "rbr_diff_std": np.nan,
            "rbr_diff_p95": np.nan,
            "rbr_change_fraction": np.nan,
        }

    diffs = [np.abs(frames[i + 1] - frames[i])[sky_mask] for i in range(len(frames) - 1)]
    all_diffs = np.concatenate(diffs)
    if all_diffs.size == 0:
        return {
            "rbr_diff_mean": np.nan,
            "rbr_diff_std": np.nan,
            "rbr_diff_p95": np.nan,
            "rbr_change_fraction": np.nan,
        }
    return {
        "rbr_diff_mean": float(np.mean(all_diffs)),
        "rbr_diff_std": float(np.std(all_diffs)),
        "rbr_diff_p95": float(np.quantile(all_diffs, 0.95)),
        "rbr_change_fraction": float(np.mean(all_diffs >= 0.08)),
    }


def _image_metrics(image_path: Path, sky_mask: np.ndarray, image_size: int) -> dict[str, float]:
    rgb = _load_rgb(image_path, image_size)
    if rgb is None:
        return {
            "blue_fraction": np.nan,
            "gray_fraction": np.nan,
            "bright_white_fraction": np.nan,
            "rbr_diff_mean": np.nan,
            "rbr_diff_std": np.nan,
            "rbr_diff_p95": np.nan,
            "rbr_change_fraction": np.nan,
            "sky_value_std": np.nan,
            "sky_edge_strength": np.nan,
            "center_value": np.nan,
            "edge_value": np.nan,
            "center_edge_value_delta": np.nan,
            "radial_value_corr": np.nan,
            "bright_spot_value": np.nan,
            "bright_far_value": np.nan,
            "bright_spot_delta": np.nan,
            "bright_spot_radial_corr": np.nan,
            "mean_saturation": np.nan,
            "mean_value": np.nan,
        }

    red = rgb[..., 0]
    green = rgb[..., 1]
    blue = rgb[..., 2]
    rgb_min = np.min(rgb, axis=-1)
    rgb_max = np.max(rgb, axis=-1)
    saturation = (rgb_max - rgb_min) / np.clip(rgb_max, 1e-6, None)
    value = rgb_max

    blue_dominant = (blue >= red * 1.05) & (blue >= green * 0.92)
    blue_sky = sky_mask & blue_dominant & (saturation >= 0.18) & (value >= 0.12)
    gray_cloud = sky_mask & (saturation <= 0.22) & (value >= 0.35)
    dark_gray_cloud = sky_mask & (saturation <= 0.25) & (value < 0.55)
    bright_white = sky_mask & (red >= 0.72) & (green >= 0.72) & (blue >= 0.72) & (saturation <= 0.18)
    denom = max(int(np.sum(sky_mask)), 1)
    blue_fraction = float(np.sum(blue_sky) / denom)
    gray_fraction = float(np.sum(gray_cloud) / denom)
    bright_white_fraction = float(np.sum(bright_white) / denom)
    cloud_fraction_proxy = float(np.clip(gray_fraction + 0.7 * bright_white_fraction - 0.35 * blue_fraction, 0.0, 1.0))
    sky_value = value[sky_mask]
    grad_y, grad_x = np.gradient(value)
    sky_edge_strength = float(np.mean(np.hypot(grad_x[sky_mask], grad_y[sky_mask])))
    yy, xx = np.indices(value.shape)
    cx = (value.shape[1] - 1) / 2.0
    cy = (value.shape[0] - 1) / 2.0
    radius = np.hypot(xx - cx, yy - cy)
    sky_radius = radius[sky_mask]
    radius_norm = sky_radius / max(float(np.max(sky_radius)), 1e-6)
    center_values = value[sky_mask & (radius <= 0.35 * np.max(sky_radius))]
    edge_values = value[sky_mask & (radius >= 0.72 * np.max(sky_radius))]
    center_value = float(np.mean(center_values)) if center_values.size else np.nan
    edge_value = float(np.mean(edge_values)) if edge_values.size else np.nan
    if sky_value.size >= 2 and float(np.std(radius_norm)) > 1e-6 and float(np.std(sky_value)) > 1e-6:
        radial_value_corr = float(np.corrcoef(radius_norm, sky_value)[0, 1])
    else:
        radial_value_corr = np.nan
    sky_y, sky_x = np.where(sky_mask)
    bright_threshold = float(np.quantile(sky_value, 0.985))
    bright_mask = sky_mask & (value >= bright_threshold)
    if int(np.sum(bright_mask)) >= 3:
        weights = value[bright_mask]
        by, bx = np.where(bright_mask)
        spot_x = float(np.average(bx, weights=weights))
        spot_y = float(np.average(by, weights=weights))
    else:
        max_idx = int(np.argmax(sky_value))
        spot_y = float(sky_y[max_idx])
        spot_x = float(sky_x[max_idx])
    spot_radius = np.hypot(xx - spot_x, yy - spot_y)
    sky_spot_radius = spot_radius[sky_mask]
    spot_radius_norm = sky_spot_radius / max(float(np.max(sky_spot_radius)), 1e-6)
    bright_near_values = value[sky_mask & (spot_radius <= 0.20 * np.max(sky_spot_radius))]
    bright_far_values = value[sky_mask & (spot_radius >= 0.55 * np.max(sky_spot_radius))]
    bright_spot_value = float(np.mean(bright_near_values)) if bright_near_values.size else np.nan
    bright_far_value = float(np.mean(bright_far_values)) if bright_far_values.size else np.nan
    if sky_value.size >= 2 and float(np.std(spot_radius_norm)) > 1e-6 and float(np.std(sky_value)) > 1e-6:
        bright_spot_radial_corr = float(np.corrcoef(spot_radius_norm, sky_value)[0, 1])
    else:
        bright_spot_radial_corr = np.nan

    return {
        "blue_fraction": blue_fraction,
        "gray_fraction": gray_fraction,
        "dark_gray_fraction": float(np.sum(dark_gray_cloud) / denom),
        "bright_white_fraction": bright_white_fraction,
        "cloud_fraction_proxy": cloud_fraction_proxy,
        "blue_to_gray_ratio": float(blue_fraction / max(gray_fraction, 1e-6)),
        "sky_value_std": float(np.std(sky_value)),
        "sky_edge_strength": sky_edge_strength,
        "center_value": center_value,
        "edge_value": edge_value,
        "center_edge_value_delta": float(center_value - edge_value) if np.isfinite(center_value) and np.isfinite(edge_value) else np.nan,
        "radial_value_corr": radial_value_corr,
        "bright_spot_value": bright_spot_value,
        "bright_far_value": bright_far_value,
        "bright_spot_delta": float(bright_spot_value - bright_far_value) if np.isfinite(bright_spot_value) and np.isfinite(bright_far_value) else np.nan,
        "bright_spot_radial_corr": bright_spot_radial_corr,
        "mean_saturation": float(np.mean(saturation[sky_mask])),
        "mean_value": float(np.mean(value[sky_mask])),
    }


def _load_day_flags(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["date", "is_clear_sky_day", "is_overcast_day", "usable_for_cloud_mask_ref"])
    df = pd.read_csv(path)
    out = pd.DataFrame({"date": pd.to_datetime(df["date"]).dt.date.astype(str)})
    out["is_clear_sky_day"] = _parse_bool_series(df["is_clear_sky"]) if "is_clear_sky" in df else False
    out["is_overcast_day"] = _parse_bool_series(df["is_overcast"]) if "is_overcast" in df else False
    out["usable_for_cloud_mask_ref"] = (
        _parse_bool_series(df["usable_for_cloud_mask_ref"]) if "usable_for_cloud_mask_ref" in df else False
    )
    return out.drop_duplicates("date")


def classify_interval(row: pd.Series, cfg: Config) -> tuple[str, str]:
    csi = float(row["target_csi"])
    blue = float(row["blue_fraction"]) if pd.notna(row["blue_fraction"]) else np.nan
    gray = float(row["gray_fraction"]) if pd.notna(row["gray_fraction"]) else np.nan
    dark_gray = float(row["dark_gray_fraction"]) if pd.notna(row["dark_gray_fraction"]) else np.nan
    bright = float(row["bright_white_fraction"]) if pd.notna(row["bright_white_fraction"]) else np.nan
    cloud_proxy = float(row["cloud_fraction_proxy"]) if pd.notna(row["cloud_fraction_proxy"]) else np.nan
    sky_value_std = float(row["sky_value_std"]) if pd.notna(row["sky_value_std"]) else np.nan
    sky_edge_strength = float(row["sky_edge_strength"]) if pd.notna(row["sky_edge_strength"]) else np.nan
    bright_spot_delta = float(row["bright_spot_delta"]) if pd.notna(row["bright_spot_delta"]) else np.nan
    bright_spot_radial_corr = (
        float(row["bright_spot_radial_corr"]) if pd.notna(row["bright_spot_radial_corr"]) else np.nan
    )
    rbr_diff_std = float(row["rbr_diff_std"]) if pd.notna(row["rbr_diff_std"]) else np.nan
    rbr_diff_p95 = float(row["rbr_diff_p95"]) if pd.notna(row["rbr_diff_p95"]) else np.nan
    rbr_change_fraction = float(row["rbr_change_fraction"]) if pd.notna(row["rbr_change_fraction"]) else np.nan
    saturation = float(row["mean_saturation"]) if pd.notna(row["mean_saturation"]) else np.nan
    clear_day = bool(row.get("is_clear_sky_day", False))
    overcast_day = bool(row.get("is_overcast_day", False))
    sun_like_gradient = (
        pd.notna(bright_spot_delta)
        and bright_spot_delta >= 0.20
        and (pd.isna(bright_spot_radial_corr) or bright_spot_radial_corr <= -0.50)
    )
    very_clean_blue = (
        pd.notna(blue)
        and blue >= 0.58
        and (pd.isna(gray) or gray <= 0.52)
        and (pd.isna(cloud_proxy) or cloud_proxy <= 0.36)
    )
    low_texture = (
        (pd.isna(sky_edge_strength) or sky_edge_strength <= 0.012)
        and (pd.isna(sky_value_std) or sky_value_std <= 0.13)
    )
    stable_rbr_window = (
        (pd.isna(rbr_diff_std) or rbr_diff_std <= 0.055)
        and (pd.isna(rbr_diff_p95) or rbr_diff_p95 <= 0.16)
        and (pd.isna(rbr_change_fraction) or rbr_change_fraction <= 0.12)
    )
    temporal_cloud_motion = (
        (pd.notna(rbr_diff_std) and rbr_diff_std >= 0.060)
        or (pd.notna(rbr_diff_p95) and rbr_diff_p95 >= 0.18)
        or (pd.notna(rbr_change_fraction) and rbr_change_fraction >= 0.14)
    )

    closed_overcast = (
        (pd.isna(blue) or blue <= cfg.overcast_low_blue_threshold)
        and (pd.isna(gray) or gray >= cfg.overcast_min_gray_fraction)
        and (pd.isna(cloud_proxy) or cloud_proxy >= 0.90)
    )
    sealed_overcast = (
        (pd.isna(blue) or blue <= cfg.overcast_sealed_blue_threshold)
        and (pd.isna(gray) or gray >= cfg.overcast_sealed_gray_threshold)
        and (pd.isna(cloud_proxy) or cloud_proxy >= cfg.overcast_sealed_cloud_proxy_threshold)
    )
    uniform_overcast = (
        pd.notna(bright_spot_delta)
        and bright_spot_delta <= 0.16
        and low_texture
        and (pd.isna(blue) or blue <= 0.08)
        and (pd.isna(gray) or gray >= 0.88)
        and (pd.isna(cloud_proxy) or cloud_proxy >= 0.88)
    )
    if overcast_day and csi <= 0.35 and closed_overcast:
        return "overcast", "day_overcast_closed_sky"
    if sealed_overcast:
        return "overcast", "visually_sealed_sky"
    if uniform_overcast and csi <= 0.45:
        return "overcast", "uniform_low_gradient_overcast"
    if csi <= cfg.overcast_csi_threshold and closed_overcast:
        return "overcast", "low_csi_closed_overcast"
    if csi <= 0.12 and (pd.isna(blue) or blue <= 0.25) and (pd.isna(gray) or gray >= 0.75):
        return "overcast", "very_low_interval_csi"

    uniform_sky_without_cloud_texture = (
        low_texture
        and (sun_like_gradient or pd.isna(bright_spot_delta) or bright_spot_delta >= 0.18)
        and (pd.isna(blue) or blue >= 0.18)
        and (pd.isna(bright) or bright <= 0.13)
        and (pd.isna(cloud_proxy) or cloud_proxy <= 0.78)
        and csi >= 0.45
        and stable_rbr_window
    )
    if uniform_sky_without_cloud_texture:
        return "clear_sky", "uniform_haze_or_halo_no_cloud_texture"

    image_clear = (
        (pd.isna(blue) or blue >= cfg.clear_blue_threshold)
        and (pd.isna(gray) or gray <= cfg.clear_max_gray_fraction)
        and (pd.isna(bright) or bright <= cfg.clear_max_bright_white_fraction)
        and (pd.isna(cloud_proxy) or cloud_proxy <= cfg.clear_max_cloud_proxy)
    )
    if (
        clear_day
        and csi >= 0.58
        and (pd.isna(blue) or blue >= 0.38)
        and (pd.isna(gray) or gray <= 0.62)
        and (pd.isna(bright) or bright <= 0.11)
        and (pd.isna(cloud_proxy) or cloud_proxy <= 0.50)
        and low_texture
        and stable_rbr_window
        and (sun_like_gradient or very_clean_blue)
    ):
        return "clear_sky", "clear_day_high_csi"
    if (
        clear_day
        and csi >= cfg.clear_csi_threshold
        and image_clear
        and low_texture
        and stable_rbr_window
        and (sun_like_gradient or very_clean_blue)
    ):
        return "clear_sky", "clear_day_high_csi_clean_image"
    if (
        csi >= cfg.clear_csi_threshold
        and image_clear
        and low_texture
        and stable_rbr_window
        and (sun_like_gradient or very_clean_blue)
    ):
        return "clear_sky", "high_csi_clean_blue_image"
    if (
        csi >= 0.54
        and pd.notna(blue)
        and blue >= 0.50
        and (pd.isna(bright) or bright <= 0.10)
        and (pd.isna(cloud_proxy) or cloud_proxy <= 0.40)
        and low_texture
        and stable_rbr_window
        and (sun_like_gradient or very_clean_blue)
    ):
        return "clear_sky", "near_clear_high_blue_low_cloud_proxy"
    if (
        csi >= 0.60
        and pd.notna(blue)
        and blue >= 0.42
        and (pd.isna(gray) or gray <= 0.62)
        and (pd.isna(cloud_proxy) or cloud_proxy <= 0.50)
        and (pd.isna(sky_edge_strength) or sky_edge_strength <= 0.008)
        and (pd.isna(sky_value_std) or sky_value_std <= 0.16)
        and (pd.isna(bright) or bright <= 0.11)
        and sun_like_gradient
        and stable_rbr_window
    ):
        return "clear_sky", "high_csi_smooth_sky"

    image_cloudy = (
        (pd.notna(gray) and gray >= cfg.cloudy_min_gray_fraction)
        or (pd.notna(cloud_proxy) and cloud_proxy >= 0.65)
        or (pd.notna(dark_gray) and dark_gray >= 0.35)
    )
    weak_sun_or_low_csi = csi <= cfg.cloudy_csi_threshold or (pd.notna(saturation) and saturation <= 0.12)
    if image_cloudy and weak_sun_or_low_csi and (pd.isna(blue) or blue <= cfg.cloudy_max_blue_fraction):
        return "cloudy", "cloud_dominant_with_limited_blue"
    if csi <= cfg.partly_low_csi_threshold and image_cloudy:
        return "cloudy", "low_csi_cloud_dominant"

    has_some_blue = pd.notna(blue) and cfg.partly_min_blue_fraction <= blue <= cfg.partly_max_blue_fraction
    has_cloud_evidence = (
        (pd.notna(cloud_proxy) and cloud_proxy >= cfg.partly_min_cloud_proxy)
        or (
            pd.notna(gray)
            and gray >= cfg.partly_min_gray_fraction
            and pd.notna(bright)
            and bright >= cfg.partly_min_bright_white_fraction
        )
        or (pd.notna(sky_edge_strength) and sky_edge_strength >= 0.012 and pd.notna(gray) and gray >= 0.50)
        or (temporal_cloud_motion and pd.notna(gray) and gray >= 0.50)
    )
    too_clean_for_partly = (
        pd.notna(blue)
        and blue >= 0.48
        and (pd.isna(bright) or bright <= 0.085)
        and (pd.isna(cloud_proxy) or cloud_proxy <= 0.50)
    )
    strong_partly_texture_or_motion = (
        (pd.notna(sky_edge_strength) and sky_edge_strength >= 0.010)
        or (pd.notna(rbr_diff_std) and rbr_diff_std >= 0.045)
        or (pd.notna(rbr_change_fraction) and rbr_change_fraction >= 0.12)
    )
    if (
        csi >= cfg.partly_min_csi
        and has_some_blue
        and has_cloud_evidence
        and strong_partly_texture_or_motion
        and not too_clean_for_partly
    ):
        return "partly_cloudy", "mixed_clouds_with_some_blue_or_pv"
    if (
        csi >= 0.56
        and pd.notna(blue)
        and blue >= 0.38
        and (pd.isna(cloud_proxy) or cloud_proxy <= 0.56)
        and (pd.isna(sky_edge_strength) or sky_edge_strength <= 0.010)
        and (pd.isna(bright) or bright <= 0.11)
        and stable_rbr_window
        and (sun_like_gradient or very_clean_blue)
    ):
        return "clear_sky", "fallback_high_csi_smooth_blue"
    if (
        pd.notna(blue)
        and blue >= 0.58
        and (pd.isna(cloud_proxy) or cloud_proxy <= 0.30)
        and (pd.isna(gray) or gray <= 0.45)
        and (pd.isna(sky_edge_strength) or sky_edge_strength <= 0.018)
        and stable_rbr_window
    ):
        return "clear_sky", "fallback_visual_clear_blue"
    if (
        csi >= 0.50
        and pd.notna(blue)
        and blue >= 0.34
        and blue <= 0.58
        and (pd.notna(cloud_proxy) and 0.42 <= cloud_proxy <= 0.62)
        and (pd.notna(gray) and 0.52 <= gray <= 0.68)
        and (
            (pd.notna(sky_edge_strength) and sky_edge_strength >= 0.012)
            or (pd.notna(rbr_change_fraction) and rbr_change_fraction >= 0.18)
        )
    ):
        return "partly_cloudy", "fallback_mid_csi_some_blue"
    if (
        csi >= 0.60
        and pd.notna(blue)
        and 0.25 <= blue <= 0.58
        and (pd.notna(cloud_proxy) and 0.50 <= cloud_proxy <= 0.72)
        and (
            (pd.notna(sky_edge_strength) and sky_edge_strength >= 0.012)
            or temporal_cloud_motion
        )
    ):
        return "partly_cloudy", "fallback_high_csi_cloud_edge"
    return "cloudy", "cloudy_fallback_low_blue_or_low_pv"


def build_index(cfg: Config) -> pd.DataFrame:
    samples = pd.read_csv(cfg.samples_csv)
    samples["ts_anchor"] = pd.to_datetime(samples["ts_anchor"])
    samples["ts_target"] = pd.to_datetime(samples["ts_target"])
    samples = samples.sort_values("ts_target").reset_index(drop=True)

    full_sky_mask = _load_mask(cfg.sky_mask_path, cfg.image_size)
    sky_mask = _limit_mask_to_fov(full_sky_mask, cfg.analysis_fov_deg, cfg.camera_fov_deg)
    rows: list[dict[str, object]] = []
    for sample in samples.itertuples(index=False):
        image_paths = _parse_list(sample.img_paths)
        image_path = Path(str(image_paths[-1])) if image_paths else Path("")
        path_list = [Path(str(path)) for path in image_paths]
        metrics = _image_metrics(image_path, sky_mask, cfg.image_size) if image_paths else {}
        if image_paths:
            metrics.update(_temporal_rbr_metrics(path_list, sky_mask, cfg.image_size))
        ts_target = pd.Timestamp(sample.ts_target)
        target_clear_sky_w = float(sample.target_clear_sky_w)
        target_pv_w = float(sample.target_pv_w)
        target_csi = target_pv_w / max(target_clear_sky_w, 1e-6)
        interval_start = ts_target - pd.Timedelta(minutes=15)
        rows.append(
            {
                "interval_start": interval_start,
                "interval_end": ts_target,
                "date": ts_target.date().isoformat(),
                "time": ts_target.strftime("%H:%M:%S"),
                "split": sample.split,
                "target_pv_w": target_pv_w,
                "target_clear_sky_w": target_clear_sky_w,
                "target_csi": target_csi,
                "zenith_deg": float(sample.zenith_deg) if hasattr(sample, "zenith_deg") else np.nan,
                "target_zenith_deg": float(sample.target_zenith_deg) if hasattr(sample, "target_zenith_deg") else np.nan,
                "image_path": str(image_path),
                **metrics,
            }
        )

    index_df = pd.DataFrame.from_records(rows)
    start_t = pd.to_datetime(cfg.start_time).time()
    end_t = pd.to_datetime(cfg.end_time).time()
    interval_times = index_df["interval_end"].dt.time
    index_df = index_df[(interval_times >= start_t) & (interval_times <= end_t)].copy()
    if "target_zenith_deg" in index_df:
        index_df = index_df[
            index_df["target_zenith_deg"].isna() | (index_df["target_zenith_deg"] <= cfg.max_sun_zenith_deg)
        ].copy()

    day_flags = _load_day_flags(cfg.clear_sky_days_csv)
    index_df = index_df.merge(day_flags, on="date", how="left")
    for col in ["is_clear_sky_day", "is_overcast_day", "usable_for_cloud_mask_ref"]:
        index_df[col] = index_df[col].fillna(False).astype(bool)

    labels = index_df.apply(lambda row: classify_interval(row, cfg), axis=1)
    index_df["weather_tag"] = [label for label, _ in labels]
    index_df["tag_reason"] = [reason for _, reason in labels]
    return index_df.sort_values("interval_end").reset_index(drop=True)


def _sample_rows(index_df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    samples: list[pd.DataFrame] = []
    for tag in TAGS:
        tag_df = index_df[index_df["weather_tag"] == tag].copy()
        if tag_df.empty:
            continue
        if len(tag_df) <= cfg.sample_per_class:
            samples.append(tag_df)
            continue
        quantiles = np.linspace(0.05, 0.95, cfg.sample_per_class)
        selected_indices: list[int] = []
        for quantile in quantiles:
            target_csi = float(tag_df["target_csi"].quantile(quantile))
            candidates = tag_df.assign(csi_distance=(tag_df["target_csi"] - target_csi).abs())
            for idx in candidates.sort_values(["csi_distance", "interval_end"]).index:
                if int(idx) not in selected_indices:
                    selected_indices.append(int(idx))
                    break
        samples.append(tag_df.loc[selected_indices])
    if not samples:
        return pd.DataFrame()
    return pd.concat(samples, ignore_index=True).sort_values(["weather_tag", "interval_end"]).reset_index(drop=True)


def _copy_sample_images(samples: pd.DataFrame, output_dir: Path) -> None:
    image_root = output_dir / "sample_images"
    if image_root.exists():
        shutil.rmtree(image_root)
    image_root.mkdir(parents=True, exist_ok=True)
    for row in samples.itertuples(index=False):
        src = Path(row.image_path)
        if not src.exists():
            continue
        tag_dir = image_root / row.weather_tag
        tag_dir.mkdir(parents=True, exist_ok=True)
        dst = tag_dir / f"{pd.Timestamp(row.interval_end).strftime('%Y%m%d_%H%M%S')}_{src.name}"
        shutil.copy2(src, dst)


def _make_contact_sheet(samples: pd.DataFrame, output_dir: Path, thumb_size: int = 160) -> None:
    sheet_dir = output_dir / "sample_sheets"
    if sheet_dir.exists():
        shutil.rmtree(sheet_dir)
    sheet_dir.mkdir(parents=True, exist_ok=True)
    for tag in TAGS:
        tag_df = samples[samples["weather_tag"] == tag].head(30)
        if tag_df.empty:
            continue
        cols = 5
        rows = int(np.ceil(len(tag_df) / cols))
        label_h = 34
        sheet = Image.new("RGB", (cols * thumb_size, rows * (thumb_size + label_h)), "white")
        draw = ImageDraw.Draw(sheet)
        for i, row in enumerate(tag_df.itertuples(index=False)):
            src = Path(row.image_path)
            if not src.exists():
                continue
            with Image.open(src) as im:
                thumb = im.convert("RGB").resize((thumb_size, thumb_size), resample=Image.BILINEAR)
            x = (i % cols) * thumb_size
            y = (i // cols) * (thumb_size + label_h)
            sheet.paste(thumb, (x, y))
            label = f"{pd.Timestamp(row.interval_end).strftime('%m-%d %H:%M')} csi={row.target_csi:.2f}"
            draw.text((x + 4, y + thumb_size + 4), label, fill=(0, 0, 0))
        sheet.save(sheet_dir / f"{tag}_samples.png")


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Build 15-minute weather classification index.")
    parser.add_argument("--samples-csv", default=str(REPO_ROOT / "new-model/artifacts/dataset/samples.csv"))
    parser.add_argument("--sky-mask-path", default=str(REPO_ROOT / "data/sky_mask.png"))
    parser.add_argument("--clear-sky-days-csv", default=str(REPO_ROOT / "data/clear_sky_generated.csv"))
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "data/weather_interval_classification"))
    parser.add_argument("--start-time", default="08:30")
    parser.add_argument("--end-time", default="16:00")
    parser.add_argument("--analysis-fov-deg", type=float, default=140.0)
    parser.add_argument("--camera-fov-deg", type=float, default=180.0)
    parser.add_argument("--max-sun-zenith-deg", type=float, default=140.0)
    parser.add_argument("--sample-per-class", type=int, default=30)
    parser.add_argument("--clear-csi-threshold", type=float, default=0.58)
    parser.add_argument("--clear-blue-threshold", type=float, default=0.45)
    parser.add_argument("--clear-max-gray-fraction", type=float, default=0.72)
    parser.add_argument("--clear-max-bright-white-fraction", type=float, default=0.16)
    parser.add_argument("--clear-max-cloud-proxy", type=float, default=0.48)
    parser.add_argument("--overcast-csi-threshold", type=float, default=0.28)
    parser.add_argument("--overcast-low-blue-threshold", type=float, default=0.12)
    parser.add_argument("--overcast-min-gray-fraction", type=float, default=0.88)
    parser.add_argument("--overcast-sealed-blue-threshold", type=float, default=0.02)
    parser.add_argument("--overcast-sealed-gray-threshold", type=float, default=0.98)
    parser.add_argument("--overcast-sealed-cloud-proxy-threshold", type=float, default=0.98)
    parser.add_argument("--cloudy-csi-threshold", type=float, default=0.50)
    parser.add_argument("--cloudy-min-gray-fraction", type=float, default=0.72)
    parser.add_argument("--cloudy-max-blue-fraction", type=float, default=0.35)
    parser.add_argument("--partly-min-blue-fraction", type=float, default=0.22)
    parser.add_argument("--partly-min-csi", type=float, default=0.32)
    parser.add_argument("--partly-low-csi-threshold", type=float, default=0.40)
    parser.add_argument("--partly-min-gray-fraction", type=float, default=0.62)
    parser.add_argument("--partly-min-bright-white-fraction", type=float, default=0.08)
    parser.add_argument("--partly-min-cloud-proxy", type=float, default=0.58)
    parser.add_argument("--partly-max-blue-fraction", type=float, default=0.60)
    args = parser.parse_args()
    return Config(
        samples_csv=Path(args.samples_csv).resolve(),
        sky_mask_path=Path(args.sky_mask_path).resolve(),
        clear_sky_days_csv=Path(args.clear_sky_days_csv).resolve(),
        output_dir=Path(args.output_dir).resolve(),
        start_time=str(args.start_time),
        end_time=str(args.end_time),
        analysis_fov_deg=float(args.analysis_fov_deg),
        camera_fov_deg=float(args.camera_fov_deg),
        max_sun_zenith_deg=float(args.max_sun_zenith_deg),
        sample_per_class=int(args.sample_per_class),
        clear_csi_threshold=float(args.clear_csi_threshold),
        clear_blue_threshold=float(args.clear_blue_threshold),
        clear_max_gray_fraction=float(args.clear_max_gray_fraction),
        clear_max_bright_white_fraction=float(args.clear_max_bright_white_fraction),
        clear_max_cloud_proxy=float(args.clear_max_cloud_proxy),
        overcast_csi_threshold=float(args.overcast_csi_threshold),
        overcast_low_blue_threshold=float(args.overcast_low_blue_threshold),
        overcast_min_gray_fraction=float(args.overcast_min_gray_fraction),
        overcast_sealed_blue_threshold=float(args.overcast_sealed_blue_threshold),
        overcast_sealed_gray_threshold=float(args.overcast_sealed_gray_threshold),
        overcast_sealed_cloud_proxy_threshold=float(args.overcast_sealed_cloud_proxy_threshold),
        cloudy_csi_threshold=float(args.cloudy_csi_threshold),
        cloudy_min_gray_fraction=float(args.cloudy_min_gray_fraction),
        cloudy_max_blue_fraction=float(args.cloudy_max_blue_fraction),
        partly_min_blue_fraction=float(args.partly_min_blue_fraction),
        partly_min_csi=float(args.partly_min_csi),
        partly_low_csi_threshold=float(args.partly_low_csi_threshold),
        partly_min_gray_fraction=float(args.partly_min_gray_fraction),
        partly_min_bright_white_fraction=float(args.partly_min_bright_white_fraction),
        partly_min_cloud_proxy=float(args.partly_min_cloud_proxy),
        partly_max_blue_fraction=float(args.partly_max_blue_fraction),
    )


def main() -> None:
    cfg = parse_args()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    index_df = build_index(cfg)
    sample_df = _sample_rows(index_df, cfg)

    index_path = cfg.output_dir / "weather_interval_index.csv"
    sample_path = cfg.output_dir / "weather_interval_samples.csv"
    summary_path = cfg.output_dir / "weather_interval_summary.csv"
    index_df.to_csv(index_path, index=False)
    sample_df.to_csv(sample_path, index=False)
    summary = (
        index_df.groupby("weather_tag")
        .agg(
            n_intervals=("weather_tag", "size"),
            mean_csi=("target_csi", "mean"),
            median_csi=("target_csi", "median"),
            mean_blue_fraction=("blue_fraction", "mean"),
            mean_gray_fraction=("gray_fraction", "mean"),
            mean_sky_edge_strength=("sky_edge_strength", "mean"),
            mean_sky_value_std=("sky_value_std", "mean"),
            mean_bright_spot_delta=("bright_spot_delta", "mean"),
            mean_bright_spot_radial_corr=("bright_spot_radial_corr", "mean"),
            mean_rbr_diff_std=("rbr_diff_std", "mean"),
            mean_rbr_diff_p95=("rbr_diff_p95", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(summary_path, index=False)

    _copy_sample_images(sample_df, cfg.output_dir)
    _make_contact_sheet(sample_df, cfg.output_dir)

    print(f"Wrote interval index to {index_path}")
    print(f"Wrote samples to {sample_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
