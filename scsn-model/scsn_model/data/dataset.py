from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from .cloud_mask_supervision import CloudMaskSupervisor


def load_mask(mask_path: str | Path, size: tuple[int, int]) -> np.ndarray:
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


def _parse_jsonish(value: object) -> list[float] | list[str]:
    if isinstance(value, list):
        return value
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
        self.df = pd.read_csv(csv_path)
        self.df["ts_anchor"] = pd.to_datetime(self.df["ts_anchor"])
        self.df["ts_target"] = pd.to_datetime(self.df["ts_target"])
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        self.image_size = tuple(int(v) for v in image_size)
        self.mask = load_mask(sky_mask_path, self.image_size) if sky_mask_path and Path(sky_mask_path).exists() else None
        self.peak_power_w = float(peak_power_w) if peak_power_w is not None else 1.0
        if self.peak_power_w <= 0:
            raise ValueError("peak_power_w must be positive.")
        self.cloud_mask_supervisor = None
        if cloud_mask_manifest_path and cloud_mask_sky_mask_path:
            manifest_path = Path(cloud_mask_manifest_path)
            mask_path = Path(cloud_mask_sky_mask_path)
            if manifest_path.exists() and mask_path.exists():
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
        cloud_mask, cloud_mask_valid = self._load_cloud_mask(img_paths[-1])

        input_frames = self._build_input_channels(frames=frames, sun_xy=current_sun_xy)
        target_rbr = input_frames[-1, 3:4]
        prev_rbr = input_frames[-2, 3:4] if input_frames.shape[0] > 1 else target_rbr.copy()
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
            "opacity_proxy": torch.from_numpy(opacity_proxy.astype(np.float32)),
            "gap_proxy": torch.from_numpy(gap_proxy.astype(np.float32)),
            "transmission_proxy": torch.from_numpy(transmission_proxy.astype(np.float32)),
            "cloud_mask": torch.from_numpy(cloud_mask.astype(np.float32)),
            "cloud_mask_valid": torch.tensor(cloud_mask_valid, dtype=torch.float32),
            "meta_index": torch.tensor(index, dtype=torch.long),
        }

    def dataframe(self) -> pd.DataFrame:
        return self.df.copy()

    def _build_input_channels(self, frames: np.ndarray, sun_xy: np.ndarray) -> np.ndarray:
        seq_len, _, height, width = frames.shape
        mask = self.mask if self.mask is not None else np.ones((1, height, width), dtype=np.float32)
        sun_distance = self._sun_distance_map(sun_xy=sun_xy, height=height, width=width)
        repeated_mask = np.repeat(mask[None, ...], seq_len, axis=0)
        repeated_distance = np.repeat(sun_distance[None, ...], seq_len, axis=0)
        rbr = frames[:, 0:1] / np.clip(frames[:, 2:3], a_min=1e-3, a_max=None)
        rbr = np.clip(rbr, 0.0, 4.0) / 4.0
        return np.concatenate([frames, rbr, repeated_distance, repeated_mask], axis=1)

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
