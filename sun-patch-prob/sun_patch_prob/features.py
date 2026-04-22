from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .utils import local_path, load_json


def parse_json_list(value: object) -> list:
    if isinstance(value, list):
        return value
    return json.loads(str(value))


def load_mask(path: str | Path, image_size: int) -> np.ndarray:
    with Image.open(path) as image:
        mask = image.convert("L").resize((image_size, image_size), resample=Image.NEAREST)
    return (np.asarray(mask, dtype=np.float32) / 255.0) >= 0.5


def load_rgb(path: str | Path, image_size: int) -> np.ndarray:
    with Image.open(local_path(path)) as image:
        rgb = image.convert("RGB").resize((image_size, image_size), resample=Image.BILINEAR)
    return np.asarray(rgb, dtype=np.float32) / 255.0


def distance_deg_map(x_px_256: float, y_px_256: float, image_size: int, f_px_per_rad_256: float) -> np.ndarray:
    scale = image_size / 256.0
    x = float(x_px_256) * scale
    y = float(y_px_256) * scale
    f_px_per_rad = float(f_px_per_rad_256) * scale
    yy, xx = np.meshgrid(np.arange(image_size, dtype=np.float32), np.arange(image_size, dtype=np.float32), indexing="ij")
    dist_px = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)
    return dist_px / max(f_px_per_rad, 1e-6) * (180.0 / math.pi)


def build_rbr(frames: np.ndarray, clip: float) -> np.ndarray:
    rbr_raw = frames[..., 0] / np.clip(frames[..., 2], 1e-3, None)
    return np.clip(rbr_raw, 0.0, clip) / clip


def temporal_summary(prefix: str, values: np.ndarray, out: dict[str, float]) -> None:
    values = np.asarray(values, dtype=np.float32)
    valid = np.isfinite(values)
    if not np.any(valid):
        for suffix in ("mean", "std", "min", "max", "last", "trend"):
            out[f"{prefix}_{suffix}"] = np.nan
        return
    y = values[valid]
    x = np.arange(len(values), dtype=np.float32)[valid]
    out[f"{prefix}_mean"] = float(np.mean(y))
    out[f"{prefix}_std"] = float(np.std(y))
    out[f"{prefix}_min"] = float(np.min(y))
    out[f"{prefix}_max"] = float(np.max(y))
    out[f"{prefix}_last"] = float(y[-1])
    if len(y) > 1:
        xc = x - float(np.mean(x))
        denom = float(np.sum(xc * xc))
        out[f"{prefix}_trend"] = float(np.sum(xc * (y - float(np.mean(y)))) / denom) if denom > 0 else 0.0
    else:
        out[f"{prefix}_trend"] = 0.0


def masked_series(values: np.ndarray, region: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if not np.any(region):
        return np.full(values.shape[0], np.nan, dtype=np.float32), np.full(values.shape[0], np.nan, dtype=np.float32)
    selected = values[:, region]
    return np.nanmean(selected, axis=1), np.nanstd(selected, axis=1)


def weighted_series(values: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    total = float(np.nansum(weights))
    if total <= 0:
        return np.full(values.shape[0], np.nan, dtype=np.float32), np.full(values.shape[0], np.nan, dtype=np.float32)
    w = weights.astype(np.float32) / total
    mean = np.nansum(values * w[None, :, :], axis=(1, 2))
    var = np.nansum(((values - mean[:, None, None]) ** 2) * w[None, :, :], axis=(1, 2))
    return mean.astype(np.float32), np.sqrt(np.maximum(var, 0.0)).astype(np.float32)


def add_region_features(prefix: str, rbr: np.ndarray, luminance: np.ndarray, regions: dict[str, np.ndarray], out: dict[str, float]) -> None:
    delta_rbr = np.diff(rbr, axis=0) if rbr.shape[0] > 1 else np.zeros_like(rbr[:1])
    for name, region in regions.items():
        rbr_mean, rbr_std = masked_series(rbr, region)
        lum_mean, lum_std = masked_series(luminance, region)
        temporal_summary(f"{prefix}_rbr_mean_{name}", rbr_mean, out)
        temporal_summary(f"{prefix}_rbr_std_{name}", rbr_std, out)
        temporal_summary(f"{prefix}_lum_mean_{name}", lum_mean, out)
        temporal_summary(f"{prefix}_lum_std_{name}", lum_std, out)
        if delta_rbr.shape[0] > 0:
            dr_mean, dr_std = masked_series(delta_rbr, region)
            temporal_summary(f"{prefix}_delta_rbr_mean_{name}", dr_mean, out)
            temporal_summary(f"{prefix}_delta_rbr_std_{name}", dr_std, out)


def extract_row_features(row: pd.Series, mask: np.ndarray, config: dict, source_cfg: dict) -> dict[str, float | str]:
    data_cfg = config["data"]
    image_size = int(data_cfg["image_size"])
    rbr_clip = float(data_cfg.get("rbr_clip", 4.0))
    f_px_per_rad_256 = float(source_cfg["data"]["sun_projection_f_px_per_rad"])

    paths = parse_json_list(row["img_paths"])
    frames = np.stack([load_rgb(path, image_size) for path in paths], axis=0)
    frames[:, ~mask, :] = np.nan
    rbr = build_rbr(frames, rbr_clip)
    luminance = np.mean(np.where(np.isfinite(frames), frames, 0.0), axis=-1)
    luminance[:, ~mask] = np.nan

    dist_target = distance_deg_map(float(row["target_sun_x_px"]), float(row["target_sun_y_px"]), image_size, f_px_per_rad_256)
    regions = {"global": mask}
    if bool(data_cfg.get("include_hard_regions", False)):
        dist_now = distance_deg_map(float(row["sun_x_px"]), float(row["sun_y_px"]), image_size, f_px_per_rad_256)
        for lo, hi in data_cfg.get("rings_deg", [[5, 15], [15, 30]]):
            lo_f, hi_f = float(lo), float(hi)
            if hi_f <= lo_f:
                continue
            regions[f"target_ring_{int(lo_f)}_{int(hi_f)}deg"] = mask & (dist_target > lo_f) & (dist_target <= hi_f)
        regions["target_disk_0_5deg"] = mask & (dist_target <= 5.0)
        regions["current_disk_0_5deg"] = mask & (dist_now <= 5.0)

    out: dict[str, float | str] = {}
    add_region_features("past", rbr, luminance, regions, out)

    for sigma in data_cfg.get("weighted_sigmas_deg", [8, 16, 30]):
        weights = np.exp(-(dist_target**2) / (2.0 * float(sigma) ** 2)).astype(np.float32)
        weights = np.where(mask, weights, 0.0)
        r_mean, r_std = weighted_series(rbr, weights)
        l_mean, l_std = weighted_series(luminance, weights)
        temporal_summary(f"past_weighted_rbr_mean_sigma{int(sigma)}deg", r_mean, out)
        temporal_summary(f"past_weighted_rbr_std_sigma{int(sigma)}deg", r_std, out)
        temporal_summary(f"past_weighted_lum_mean_sigma{int(sigma)}deg", l_mean, out)
        temporal_summary(f"past_weighted_lum_std_sigma{int(sigma)}deg", l_std, out)

    future_paths = parse_json_list(row.get("future_img_paths", []))
    if future_paths:
        future = np.stack([load_rgb(path, image_size) for path in future_paths], axis=0)
        future[:, ~mask, :] = np.nan
        future_rbr = build_rbr(future, rbr_clip)
        disk = mask & (dist_target <= 5.0)
        future_disk_mean, _ = masked_series(future_rbr, disk)
        past_disk_mean, _ = masked_series(rbr, disk)
        out["aux_future_sun_mean"] = float(np.nanmean(future_disk_mean))
        out["aux_future_sun_var"] = float(np.nanvar(future_disk_mean))
        out["aux_future_sun_delta"] = float(np.nanmean(future_disk_mean) - past_disk_mean[-1])
    else:
        out["aux_future_sun_mean"] = np.nan
        out["aux_future_sun_var"] = np.nan
        out["aux_future_sun_delta"] = np.nan

    past_pv = np.asarray(parse_json_list(row["past_pv_w"]), dtype=np.float32)
    peak = float(data_cfg["peak_power_w"])
    for i, value in enumerate(past_pv):
        out[f"past_pv_{i}_w"] = float(value)
        out[f"past_pv_{i}_norm"] = float(value / peak)
    out["past_pv_mean_norm"] = float(np.mean(past_pv) / peak)
    out["past_pv_std_norm"] = float(np.std(past_pv) / peak)
    out["past_pv_trend_norm"] = float(np.polyfit(np.arange(len(past_pv)), past_pv / peak, 1)[0]) if len(past_pv) > 1 else 0.0

    solar_vec = np.asarray(parse_json_list(row["solar_vec"]), dtype=np.float32)
    for i, value in enumerate(solar_vec):
        out[f"solar_vec_{i}"] = float(value)
    out["solar_elevation_deg"] = float(90.0 - float(row["zenith_deg"]))
    out["target_solar_elevation_deg"] = float(90.0 - float(row["target_zenith_deg"]))
    out["sun_dx_px"] = float(row["target_sun_x_px"]) - float(row["sun_x_px"])
    out["sun_dy_px"] = float(row["target_sun_y_px"]) - float(row["sun_y_px"])

    target_clear = float(row["target_clear_sky_w"])
    target_csi = float(row["target_value"])
    anchor_clear = float(row["anchor_clear_sky_w"]) if "anchor_clear_sky_w" in row and np.isfinite(float(row["anchor_clear_sky_w"])) else target_clear
    smart_csi = float(np.clip(past_pv[-1] / max(anchor_clear, 1e-6), 0.0, 1.2))
    target_clear_persistence_csi = float(np.clip(past_pv[-1] / max(target_clear, 1e-6), 0.0, 1.2))
    out["baseline_csi"] = smart_csi
    out["baseline_pv_w"] = float(smart_csi * target_clear)
    out["target_csi"] = target_csi
    out["target_pv_w"] = float(row["target_pv_w"])
    out["target_clear_sky_w"] = target_clear
    out["anchor_clear_sky_w"] = anchor_clear
    out["smart_persistence_csi"] = smart_csi
    out["smart_persistence_pv_w"] = float(smart_csi * target_clear)
    out["persistence_pv_w"] = float(past_pv[-1])
    out["persistence_csi_target_clear"] = target_clear_persistence_csi
    out["ts_anchor"] = str(row["ts_anchor"])
    out["ts_target"] = str(row["ts_target"])
    out["split"] = str(row["split"])
    weather_tag = str(row.get("weather_tag", "unknown")).strip().lower() or "unknown"
    out["weather_tag"] = weather_tag
    for tag in ("clear_sky", "overcast", "cloudy", "partly_cloudy", "unknown"):
        out[f"weather_{tag}"] = 1.0 if weather_tag == tag else 0.0
    return out


def add_anchor_clear_sky(samples: pd.DataFrame, data_cfg: dict, source_cfg: dict) -> pd.DataFrame:
    samples = samples.copy()
    if "anchor_clear_sky_w" in samples.columns:
        return samples
    source_config_path = local_path(data_cfg["source_config_json"])
    scsn_root = source_config_path.parents[1]
    if str(scsn_root) not in sys.path:
        sys.path.insert(0, str(scsn_root))
    from scsn_model.data.solar_geometry import Calibration, compute_clear_sky_power

    source_data = source_cfg["data"]
    calib = Calibration.from_json(local_path(source_data["calibration_json"]))
    anchor_times = pd.to_datetime(samples["ts_anchor"]).tolist()
    clear = compute_clear_sky_power(
        anchor_times,
        calib,
        peak_power_w=float(source_data["peak_power_w"]),
        floor_w=float(source_data["clear_sky_floor_w"]),
    )
    samples["anchor_clear_sky_w"] = clear.to_numpy(dtype=np.float32)
    return samples


def build_feature_table(config: dict, max_samples: int = 0, progress_every: int = 100) -> pd.DataFrame:
    data_cfg = config["data"]
    source_cfg = load_json(data_cfg["source_config_json"])
    samples = pd.read_csv(data_cfg["samples_csv"])
    if max_samples > 0:
        parts = []
        per_split = max(1, max_samples // max(samples["split"].nunique(), 1))
        for _, part in samples.groupby("split", sort=False):
            parts.append(part.head(min(per_split, len(part))))
        samples = pd.concat(parts, ignore_index=True).head(max_samples).copy()
    samples = add_anchor_clear_sky(samples, data_cfg, source_cfg)
    mask = load_mask(data_cfg["sky_mask_path"], int(data_cfg["image_size"]))
    rows = []
    total = len(samples)
    for idx, row in samples.iterrows():
        if idx == 0 or (idx + 1) % progress_every == 0 or idx + 1 == total:
            print(f"building sun-patch features {idx + 1}/{total}", flush=True)
        rows.append(extract_row_features(row, mask, config, source_cfg))
    return pd.DataFrame(rows)
