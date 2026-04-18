from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

try:
    from scipy.ndimage import binary_closing as ndi_binary_closing
    from scipy.ndimage import binary_dilation as ndi_binary_dilation
    from scipy.ndimage import binary_opening as ndi_binary_opening
    from scipy.ndimage import gaussian_filter, label
except ModuleNotFoundError:  # pragma: no cover - training env should provide scipy.
    ndi_binary_closing = None
    ndi_binary_dilation = None
    ndi_binary_opening = None
    gaussian_filter = None
    label = None


@dataclass(frozen=True)
class CloudMaskConfig:
    image_size: int = 256
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


def _resolve_existing_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.exists():
        return candidate
    text = str(candidate)
    legacy_prefix = "/home/chuangbn/projects/PVPF"
    local_prefix = "/Users/huangchouyue/Projects/PVPF"
    if text.startswith(legacy_prefix):
        remapped = Path(local_prefix + text[len(legacy_prefix) :])
        if remapped.exists():
            return remapped
    return candidate


def _require_scipy() -> None:
    if gaussian_filter is None or label is None:
        raise RuntimeError("cloud mask weak supervision requires scipy to be installed.")


def _load_rgb_image(path: Path, image_size: int) -> np.ndarray:
    path = _resolve_existing_path(path)
    with Image.open(path) as im:
        rgb = im.convert("RGB")
        if rgb.size != (image_size, image_size):
            rgb = rgb.resize((image_size, image_size), resample=Image.BILINEAR)
        return np.asarray(rgb, dtype=np.float32) / 255.0


def _load_sky_mask(path: Path, image_size: int) -> np.ndarray:
    path = _resolve_existing_path(path)
    with Image.open(path) as im:
        mask = im.convert("L")
        if mask.size != (image_size, image_size):
            mask = mask.resize((image_size, image_size), resample=Image.NEAREST)
        return (np.asarray(mask, dtype=np.float32) / 255.0) >= 0.5


def _compute_saturation_value(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rgb_min = np.min(rgb, axis=-1)
    rgb_max = np.max(rgb, axis=-1)
    saturation = (rgb_max - rgb_min) / np.clip(rgb_max, 1e-6, None)
    return saturation.astype(np.float32), rgb_max.astype(np.float32)


def _compute_blue_sky_mask(rgb: np.ndarray, sky_mask: np.ndarray, cfg: CloudMaskConfig) -> np.ndarray:
    red = rgb[..., 0]
    green = rgb[..., 1]
    blue = rgb[..., 2]
    saturation, value = _compute_saturation_value(rgb)
    blue_dominant = (blue >= red * 1.05) & (blue >= green * 0.92)
    return sky_mask & blue_dominant & (saturation >= cfg.blue_saturation_threshold) & (value >= cfg.blue_value_threshold)


def _compute_sky_normalized_rbr(rgb: np.ndarray, stat_mask: np.ndarray) -> np.ndarray:
    red = rgb[..., 0]
    blue = rgb[..., 2]
    red_norm = red / max(float(np.mean(red[stat_mask])), 1e-5)
    blue_norm = blue / max(float(np.mean(blue[stat_mask])), 1e-5)
    return (red_norm / np.clip(blue_norm, 1e-6, None)).astype(np.float32)


def _disk_structure(radius: int) -> np.ndarray:
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return (xx * xx + yy * yy) <= radius * radius


def _remove_small_components(mask: np.ndarray, min_size: int) -> np.ndarray:
    _require_scipy()
    if min_size <= 1:
        return mask.astype(bool)
    labeled, n_components = label(mask.astype(bool))
    if n_components == 0:
        return mask.astype(bool)
    counts = np.bincount(labeled.ravel())
    keep = counts >= int(min_size)
    keep[0] = False
    return keep[labeled]


def _estimate_sun_guard(clear_rgb: np.ndarray, sky_mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0 or not np.any(sky_mask):
        return np.zeros_like(sky_mask, dtype=bool)
    value = np.max(clear_rgb, axis=-1)
    masked_value = np.where(sky_mask, value, -np.inf)
    y, x = np.unravel_index(int(np.argmax(masked_value)), masked_value.shape)
    yy, xx = np.ogrid[: sky_mask.shape[0], : sky_mask.shape[1]]
    return ((yy - y) ** 2 + (xx - x) ** 2 <= radius * radius) & sky_mask


def _compute_normalized_gray_diff(cloudy_rgb: np.ndarray, clear_rgb: np.ndarray, sky_mask: np.ndarray) -> np.ndarray:
    cloudy_gray = 0.299 * cloudy_rgb[..., 0] + 0.587 * cloudy_rgb[..., 1] + 0.114 * cloudy_rgb[..., 2]
    clear_gray = 0.299 * clear_rgb[..., 0] + 0.587 * clear_rgb[..., 1] + 0.114 * clear_rgb[..., 2]
    cloudy_norm = cloudy_gray / max(float(np.median(cloudy_gray[sky_mask])), 1e-6)
    clear_norm = clear_gray / max(float(np.median(clear_gray[sky_mask])), 1e-6)
    return (cloudy_norm - clear_norm).astype(np.float32)


def _detrend_with_mask(values: np.ndarray, sky_mask: np.ndarray, sigma: float) -> np.ndarray:
    _require_scipy()
    weights = sky_mask.astype(np.float32)
    weighted_values = np.where(sky_mask, values, 0.0).astype(np.float32)
    smooth_values = gaussian_filter(weighted_values, sigma=float(sigma), truncate=2.0)
    smooth_weights = gaussian_filter(weights, sigma=float(sigma), truncate=2.0)
    trend = np.divide(smooth_values, smooth_weights, out=np.zeros_like(smooth_values), where=smooth_weights > 1e-6)
    return (values - trend).astype(np.float32)


def _clean_cloud_mask(mask: np.ndarray, sky_mask: np.ndarray, cfg: CloudMaskConfig) -> np.ndarray:
    _require_scipy()
    opened = ndi_binary_opening(mask & sky_mask, structure=_disk_structure(cfg.opening_radius))
    closed = ndi_binary_closing(opened, structure=_disk_structure(cfg.closing_radius))
    return _remove_small_components(closed & sky_mask, cfg.min_component_size) & sky_mask


def compute_cloud_mask_from_pair(cloudy_rgb: np.ndarray, clear_rgb: np.ndarray, sky_mask: np.ndarray, cfg: CloudMaskConfig) -> np.ndarray:
    _require_scipy()
    cloudy_rbr = _compute_sky_normalized_rbr(cloudy_rgb, sky_mask)
    clear_rbr = _compute_sky_normalized_rbr(clear_rgb, sky_mask)
    scale = float(np.median(cloudy_rbr[sky_mask])) / max(float(np.median(clear_rbr[sky_mask])), 1e-6)
    raw_diff = cloudy_rbr - clear_rbr * scale
    local_diff = _detrend_with_mask(raw_diff.astype(np.float32), sky_mask, cfg.trend_sigma_px)

    saturation, value = _compute_saturation_value(cloudy_rgb)
    color_mask = sky_mask & (saturation <= cfg.cloud_saturation_threshold) & (value >= cfg.cloud_value_threshold)
    partly_color_mask = (
        sky_mask
        & (saturation <= cfg.partly_color_saturation_threshold)
        & (value >= cfg.partly_color_value_threshold)
        & (local_diff >= cfg.partly_color_local_floor)
    )
    gray_diff = _compute_normalized_gray_diff(cloudy_rgb, clear_rgb, sky_mask)
    blue_sky_mask = _compute_blue_sky_mask(cloudy_rgb, sky_mask, cfg)
    rbr_raw_mask = sky_mask & (local_diff > cfg.diff_threshold)
    sun_guard = _estimate_sun_guard(clear_rgb, sky_mask, cfg.sun_guard_radius)
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
    blue_guard = ndi_binary_dilation(blue_sky_mask, structure=_disk_structure(cfg.blue_guard_radius)) & sky_mask

    if blue_fraction >= cfg.blue_fraction_clear_threshold and (
        rbr_raw_fraction <= cfg.rbr_fraction_clear_threshold or local_diff_p95 <= cfg.local_p95_clear_threshold
    ):
        raw_mask = np.zeros_like(sky_mask, dtype=bool)
    elif blue_fraction <= cfg.blue_fraction_overcast_threshold and gray_fraction >= cfg.gray_fraction_overcast_threshold:
        raw_mask = color_mask & ~blue_guard
    elif (
        cfg.blue_fraction_broken_threshold <= blue_fraction <= cfg.blue_fraction_broken_max
        and gray_fraction >= cfg.gray_fraction_broken_threshold
    ):
        non_blue_cloud = sky_mask & ~blue_guard & (value >= cfg.cloud_value_threshold)
        raw_mask = (rbr_raw_mask | non_blue_cloud | bright_cloud_mask) & ~blue_guard & ~sun_guard
    elif blue_fraction >= cfg.blue_fraction_partly_threshold:
        raw_mask = (rbr_raw_mask | (partly_color_mask & ~blue_guard & ~sun_guard) | (bright_cloud_mask & ~blue_guard & ~sun_guard)) & ~blue_guard & ~sun_guard
    else:
        raw_mask = rbr_raw_mask

    return _clean_cloud_mask(raw_mask, sky_mask, cfg).astype(np.float32)


class CloudMaskSupervisor:
    def __init__(
        self,
        manifest_path: str | Path,
        sky_mask_path: str | Path,
        image_size: tuple[int, int],
        cfg: CloudMaskConfig | None = None,
    ) -> None:
        self.manifest_path = _resolve_existing_path(manifest_path)
        self.sky_mask_path = _resolve_existing_path(sky_mask_path)
        self.height, self.width = int(image_size[0]), int(image_size[1])
        if self.height != self.width:
            raise ValueError("Cloud mask supervision currently expects square cloud_seg masks.")
        self.cfg = cfg or CloudMaskConfig(image_size=self.height)
        self.sky_mask = _load_sky_mask(self.sky_mask_path, self.cfg.image_size)
        manifest = pd.read_csv(self.manifest_path)
        required = {"cloudy_image_path", "clear_image_path"}
        missing = required - set(manifest.columns)
        if missing:
            raise ValueError(f"{self.manifest_path} is missing columns: {sorted(missing)}")
        self._pairs = {
            Path(row.cloudy_image_path).name: (_resolve_existing_path(row.cloudy_image_path), _resolve_existing_path(row.clear_image_path))
            for row in manifest.itertuples(index=False)
        }
        self._cache: dict[str, np.ndarray] = {}

    def lookup(self, image_path: str | Path) -> np.ndarray | None:
        key = Path(str(image_path)).name
        pair = self._pairs.get(key)
        if pair is None:
            return None
        if key not in self._cache:
            cloudy_path, clear_path = pair
            cloudy_rgb = _load_rgb_image(cloudy_path, self.cfg.image_size)
            clear_rgb = _load_rgb_image(clear_path, self.cfg.image_size)
            mask = compute_cloud_mask_from_pair(cloudy_rgb, clear_rgb, self.sky_mask, self.cfg)
            self._cache[key] = mask[None, ...].astype(np.float32)
        return self._cache[key]

    def available_keys(self) -> list[str]:
        return list(self._pairs.keys())
