from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from ..utils.io import resolve_project_path
from .cloud_mask_supervision import CloudMaskSupervisor


def load_mask(mask_path: str | Path, size: tuple[int, int]) -> np.ndarray:
    mask_path = resolve_existing_path(mask_path)
    mask = Image.open(mask_path).convert("L").resize((size[1], size[0]), resample=Image.NEAREST)
    arr = (np.asarray(mask, dtype=np.float32) / 255.0 >= 0.5).astype(np.float32)
    return arr[None, ...]


def load_rgb_image(path: str | Path, size: tuple[int, int]) -> np.ndarray:
    path = resolve_existing_path(path)
    with Image.open(path) as im:
        rgb = im.convert("RGB").resize((size[1], size[0]), resample=Image.BILINEAR)
    arr = np.asarray(rgb, dtype=np.float32) / 255.0
    return np.transpose(arr, (2, 0, 1))


def resolve_existing_path(path: str | Path) -> Path:
    candidate = resolve_project_path(path, must_exist=False)
    if candidate.exists():
        return candidate
    return candidate


def _parse_jsonish(value: object) -> list[float] | list[str]:
    if isinstance(value, list):
        return value
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if not isinstance(value, str):
        raise TypeError(f"Unsupported sequence payload: {type(value)!r}")
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return ast.literal_eval(value)


class SunConditionedCloudDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        split: str,
        image_size: tuple[int, int],
        sky_mask_path: str | Path | None = None,
        peak_power_w: float | None = None,
        cloud_mask_manifest_path: str | Path | None = None,
        cloud_mask_sky_mask_path: str | Path | None = None,
    ) -> None:
        self.df = pd.read_csv(resolve_existing_path(csv_path))
        self.df["ts_anchor"] = pd.to_datetime(self.df["ts_anchor"])
        self.df["ts_target"] = pd.to_datetime(self.df["ts_target"])
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        self.image_size = tuple(int(v) for v in image_size)
        self.rbr_hotspot_min_p95 = 0.015
        resolved_sky_mask_path = resolve_existing_path(sky_mask_path) if sky_mask_path else None
        if resolved_sky_mask_path and not resolved_sky_mask_path.exists():
            raise FileNotFoundError(f"Sky mask not found: {sky_mask_path} (resolved to {resolved_sky_mask_path})")
        self.mask = load_mask(resolved_sky_mask_path, self.image_size) if resolved_sky_mask_path else None
        self.peak_power_w = float(peak_power_w) if peak_power_w is not None else 1.0
        if self.peak_power_w <= 0:
            raise ValueError("peak_power_w must be positive.")
        self.cloud_mask_supervisor = None
        if cloud_mask_manifest_path and cloud_mask_sky_mask_path:
            manifest_path = resolve_existing_path(cloud_mask_manifest_path)
            mask_path = resolve_existing_path(cloud_mask_sky_mask_path)
            if not manifest_path.exists():
                raise FileNotFoundError(f"Cloud mask manifest not found: {cloud_mask_manifest_path} (resolved to {manifest_path})")
            if not mask_path.exists():
                raise FileNotFoundError(f"Cloud mask sky mask not found: {cloud_mask_sky_mask_path} (resolved to {mask_path})")
            self.cloud_mask_supervisor = CloudMaskSupervisor(
                manifest_path=manifest_path,
                sky_mask_path=mask_path,
                image_size=self.image_size,
            )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> dict:
        row = self.df.iloc[index]
        img_paths = _parse_jsonish(row["img_paths"])
        future_img_paths = _parse_jsonish(row.get("future_img_paths", []))
        past_pv = np.asarray(_parse_jsonish(row["past_pv_w"]), dtype=np.float32)
        solar_vec = np.asarray(_parse_jsonish(row["solar_vec"]), dtype=np.float32)

        current_sun_xy = np.asarray([float(row["sun_x_px"]), float(row["sun_y_px"])], dtype=np.float32)
        target_sun_xy = np.asarray(
            [
                float(row.get("target_sun_x_px", row["sun_x_px"])),
                float(row.get("target_sun_y_px", row["sun_y_px"])),
            ],
            dtype=np.float32,
        )

        frames = np.stack([load_rgb_image(p, self.image_size) for p in img_paths], axis=0)
        if self.mask is not None:
            frames = frames * self.mask
        future_frames = None
        future_hotspot_valid = 0.0
        if future_img_paths:
            future_frames = np.stack([load_rgb_image(p, self.image_size) for p in future_img_paths], axis=0)
            if self.mask is not None:
                future_frames = future_frames * self.mask
            future_hotspot_valid = 1.0
        cloud_mask, cloud_mask_valid = self._load_cloud_mask(img_paths[-1])

        input_frames = self._build_input_channels(frames=frames, sun_xy=current_sun_xy)
        target_rbr = input_frames[-1, 3:4]
        prev_rbr = input_frames[-2, 3:4] if input_frames.shape[0] > 1 else target_rbr.copy()
        past_rbr_change_hotspot = self._rbr_change_hotspot(frames)
        future_rbr_change_seq, future_rbr_change_hotspot = self._future_rbr_change_hotspot(frames[-1:], future_frames)
        opacity_proxy, gap_proxy, transmission_proxy = self._build_state_proxies(frames[-1], target_rbr, target_sun_xy)

        azimuth_rad = np.deg2rad(float(row["azimuth_deg"]))
        elevation_rad = np.deg2rad(90.0 - float(row["zenith_deg"]))
        sun_angles = np.asarray([azimuth_rad / np.pi, elevation_rad / (0.5 * np.pi)], dtype=np.float32)

        target_azimuth_rad = np.deg2rad(float(row.get("target_azimuth_deg", row["azimuth_deg"])))
        target_elevation_rad = np.deg2rad(90.0 - float(row.get("target_zenith_deg", row["zenith_deg"])))
        target_sun_angles = np.asarray(
            [target_azimuth_rad / np.pi, target_elevation_rad / (0.5 * np.pi)],
            dtype=np.float32,
        )

        return {
            "images": torch.from_numpy(input_frames.astype(np.float32)),
            "pv_history": torch.tensor(past_pv / self.peak_power_w, dtype=torch.float32),
            "solar_vec": torch.tensor(solar_vec, dtype=torch.float32),
            "sun_angles": torch.tensor(sun_angles, dtype=torch.float32),
            "target_sun_angles": torch.tensor(target_sun_angles, dtype=torch.float32),
            "target": torch.tensor(float(row["target_value"]), dtype=torch.float32),
            "target_pv_w": torch.tensor(float(row["target_pv_w"]), dtype=torch.float32),
            "target_clear_sky_w": torch.tensor(float(row["target_clear_sky_w"]), dtype=torch.float32),
            "sun_xy": torch.tensor(current_sun_xy, dtype=torch.float32),
            "target_sun_xy": torch.tensor(target_sun_xy, dtype=torch.float32),
            "target_rbr": torch.from_numpy(target_rbr.astype(np.float32)),
            "prev_rbr": torch.from_numpy(prev_rbr.astype(np.float32)),
            "past_rbr_change_hotspot": torch.from_numpy(past_rbr_change_hotspot.astype(np.float32)),
            "future_rbr_change_hotspot": torch.from_numpy(future_rbr_change_hotspot.astype(np.float32)),
            "future_rbr_change_seq": torch.from_numpy(future_rbr_change_seq.astype(np.float32)),
            "future_hotspot_valid": torch.tensor(future_hotspot_valid, dtype=torch.float32),
            "opacity_proxy": torch.from_numpy(opacity_proxy.astype(np.float32)),
            "gap_proxy": torch.from_numpy(gap_proxy.astype(np.float32)),
            "transmission_proxy": torch.from_numpy(transmission_proxy.astype(np.float32)),
            "cloud_mask": torch.from_numpy(cloud_mask.astype(np.float32)),
            "cloud_mask_valid": torch.tensor(cloud_mask_valid, dtype=torch.float32),
            "meta_index": torch.tensor(index, dtype=torch.long),
        }

    def dataframe(self) -> pd.DataFrame:
        return self.df.copy()

    def cloud_mask_coverage(self) -> tuple[int, int]:
        if self.cloud_mask_supervisor is None:
            return 0, len(self.df)
        keys = set(self.cloud_mask_supervisor.available_keys())
        valid = 0
        for row in self.df.itertuples(index=False):
            img_paths = _parse_jsonish(row.img_paths)
            if Path(str(img_paths[-1])).name in keys:
                valid += 1
        return valid, len(self.df)

    def _build_input_channels(self, frames: np.ndarray, sun_xy: np.ndarray) -> np.ndarray:
        seq_len, _, height, width = frames.shape
        mask = self.mask if self.mask is not None else np.ones((1, height, width), dtype=np.float32)
        sun_distance = self._sun_distance_map(sun_xy=sun_xy, height=height, width=width)
        repeated_mask = np.repeat(mask[None, ...], seq_len, axis=0)
        repeated_distance = np.repeat(sun_distance[None, ...], seq_len, axis=0)
        rbr = self._build_rbr(frames)
        return np.concatenate([frames, rbr, repeated_distance, repeated_mask], axis=1)

    @staticmethod
    def _build_rbr(frames: np.ndarray) -> np.ndarray:
        rbr = frames[:, 0:1] / np.clip(frames[:, 2:3], a_min=1e-3, a_max=None)
        return np.clip(rbr, 0.0, 4.0) / 4.0

    def _rbr_change_hotspot(self, frames: np.ndarray) -> np.ndarray:
        rbr = self._build_rbr(frames)
        if rbr.shape[0] <= 1:
            hotspot = np.zeros_like(rbr[0])
        else:
            hotspot = np.mean(np.abs(rbr[1:] - rbr[:-1]), axis=0)
        return self._normalize_hotspot(hotspot)

    def _future_rbr_change_hotspot(
        self,
        current_frame: np.ndarray,
        future_frames: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if future_frames is None or future_frames.size == 0:
            seq = np.zeros((1, 1, self.image_size[0], self.image_size[1]), dtype=np.float32)
            return seq, seq[0]
        all_frames = np.concatenate([current_frame, future_frames], axis=0)
        rbr = self._build_rbr(all_frames)
        seq = np.abs(rbr[1:] - rbr[:-1])
        seq = np.stack([self._normalize_hotspot(step) for step in seq], axis=0)
        return seq.astype(np.float32), np.mean(seq, axis=0).astype(np.float32)

    def _normalize_hotspot(self, hotspot: np.ndarray) -> np.ndarray:
        hotspot = np.asarray(hotspot, dtype=np.float32)
        if self.mask is not None:
            hotspot = hotspot * self.mask
        scale = float(np.percentile(hotspot, 95))
        if scale < self.rbr_hotspot_min_p95:
            return np.zeros_like(hotspot, dtype=np.float32)
        return np.clip(hotspot / scale, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def _sun_distance_map(sun_xy: np.ndarray, height: int, width: int) -> np.ndarray:
        yy, xx = np.meshgrid(
            np.arange(height, dtype=np.float32),
            np.arange(width, dtype=np.float32),
            indexing="ij",
        )
        dist = np.sqrt((xx - float(sun_xy[0])) ** 2 + (yy - float(sun_xy[1])) ** 2)
        dist = dist / max(np.sqrt(height**2 + width**2), 1.0)
        return dist[None, ...].astype(np.float32)

    def _build_state_proxies(
        self,
        frame_rgb: np.ndarray,
        rbr_map: np.ndarray,
        sun_xy: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        red = frame_rgb[0:1]
        green = frame_rgb[1:2]
        blue = frame_rgb[2:3]
        brightness = np.clip(0.299 * red + 0.587 * green + 0.114 * blue, 0.0, 1.0)
        whiteness = 1.0 - np.clip(np.abs(red - green) + np.abs(green - blue) + np.abs(red - blue), 0.0, 1.0)
        rbr_cloud = np.clip((rbr_map - 0.28) / 0.45, 0.0, 1.0)
        opacity = np.clip(0.55 * whiteness + 0.45 * rbr_cloud, 0.0, 1.0)

        blueness = np.clip(blue - 0.5 * (red + green), 0.0, 1.0)
        clarity = np.clip(0.65 * blueness + 0.35 * brightness, 0.0, 1.0)
        gap = np.clip((1.0 - opacity) * 0.5 + clarity * 0.5, 0.0, 1.0)

        sun_distance = self._sun_distance_map(sun_xy=sun_xy, height=frame_rgb.shape[1], width=frame_rgb.shape[2])
        sun_focus = np.exp(-np.square(sun_distance / 0.22)).astype(np.float32)
        transmission = np.clip((1.0 - opacity) * 0.55 + gap * 0.45, 0.0, 1.0)
        transmission = np.clip(transmission * (0.8 + 0.2 * sun_focus), 0.0, 1.0)

        if self.mask is not None:
            opacity = opacity * self.mask
            gap = gap * self.mask
            transmission = transmission * self.mask
        return opacity, gap, transmission

    def _load_cloud_mask(self, current_image_path: str | Path) -> tuple[np.ndarray, float]:
        empty = np.zeros((1, self.image_size[0], self.image_size[1]), dtype=np.float32)
        if self.cloud_mask_supervisor is None:
            return empty, 0.0
        mask = self.cloud_mask_supervisor.lookup(resolve_existing_path(current_image_path))
        if mask is None:
            return empty, 0.0
        if self.mask is not None:
            mask = mask * self.mask
        return mask.astype(np.float32), 1.0
