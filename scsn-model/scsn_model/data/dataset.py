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


class SunConditionedRBRDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        split: str,
        image_size: tuple[int, int],
        sky_mask_path: str | Path | None = None,
        peak_power_w: float | None = None,
    ) -> None:
        self.df = pd.read_csv(resolve_existing_path(csv_path))
        self.df["ts_anchor"] = pd.to_datetime(self.df["ts_anchor"])
        self.df["ts_target"] = pd.to_datetime(self.df["ts_target"])
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        self.image_size = tuple(int(v) for v in image_size)
        self.rbr_variation_min_p95 = 0.015
        resolved_sky_mask_path = resolve_existing_path(sky_mask_path) if sky_mask_path else None
        if resolved_sky_mask_path and not resolved_sky_mask_path.exists():
            raise FileNotFoundError(f"Sky mask not found: {sky_mask_path} (resolved to {resolved_sky_mask_path})")
        self.mask = load_mask(resolved_sky_mask_path, self.image_size) if resolved_sky_mask_path else None
        self.peak_power_w = float(peak_power_w) if peak_power_w is not None else 1.0
        if self.peak_power_w <= 0:
            raise ValueError("peak_power_w must be positive.")

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
        future_rbr_valid = 0.0
        if future_img_paths:
            future_frames = np.stack([load_rgb_image(p, self.image_size) for p in future_img_paths], axis=0)
            if self.mask is not None:
                future_frames = future_frames * self.mask
            future_rbr_valid = 1.0
        input_frames = self._build_input_channels(frames=frames, sun_xy=current_sun_xy)
        target_rbr = input_frames[-1, 3:4]
        prev_rbr = input_frames[-2, 3:4] if input_frames.shape[0] > 1 else target_rbr.copy()
        past_rbr_variation = self._rbr_variation(frames)
        future_rbr_variation = self._future_rbr_variation(frames[-1:], future_frames)

        return {
            "images": torch.from_numpy(input_frames.astype(np.float32)),
            "pv_history": torch.tensor(past_pv / self.peak_power_w, dtype=torch.float32),
            "solar_vec": torch.tensor(solar_vec, dtype=torch.float32),
            "target": torch.tensor(float(row["target_value"]), dtype=torch.float32),
            "target_pv_w": torch.tensor(float(row["target_pv_w"]), dtype=torch.float32),
            "target_clear_sky_w": torch.tensor(float(row["target_clear_sky_w"]), dtype=torch.float32),
            "sun_xy": torch.tensor(current_sun_xy, dtype=torch.float32),
            "target_sun_xy": torch.tensor(target_sun_xy, dtype=torch.float32),
            "target_rbr": torch.from_numpy(target_rbr.astype(np.float32)),
            "prev_rbr": torch.from_numpy(prev_rbr.astype(np.float32)),
            "past_rbr_variation": torch.from_numpy(past_rbr_variation.astype(np.float32)),
            "future_rbr_variation": torch.from_numpy(future_rbr_variation.astype(np.float32)),
            "future_rbr_valid": torch.tensor(future_rbr_valid, dtype=torch.float32),
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
        rbr = self._build_rbr(frames)
        return np.concatenate([frames, rbr, repeated_distance, repeated_mask], axis=1)

    @staticmethod
    def _build_rbr(frames: np.ndarray) -> np.ndarray:
        rbr = frames[:, 0:1] / np.clip(frames[:, 2:3], a_min=1e-3, a_max=None)
        return np.clip(rbr, 0.0, 4.0) / 4.0

    def _rbr_variation(self, frames: np.ndarray) -> np.ndarray:
        rbr = self._build_rbr(frames)
        if rbr.shape[0] <= 1:
            variation = np.zeros_like(rbr[0])
        else:
            variation = np.mean(np.abs(rbr[1:] - rbr[:-1]), axis=0)
        return self._normalize_variation(variation)

    def _future_rbr_variation(
        self,
        current_frame: np.ndarray,
        future_frames: np.ndarray | None,
    ) -> np.ndarray:
        if future_frames is None or future_frames.size == 0:
            return np.zeros((1, self.image_size[0], self.image_size[1]), dtype=np.float32)
        all_frames = np.concatenate([current_frame, future_frames], axis=0)
        rbr = self._build_rbr(all_frames)
        seq = np.abs(rbr[1:] - rbr[:-1])
        seq = np.stack([self._normalize_variation(step) for step in seq], axis=0)
        return np.mean(seq, axis=0).astype(np.float32)

    def _normalize_variation(self, variation: np.ndarray) -> np.ndarray:
        variation = np.asarray(variation, dtype=np.float32)
        if self.mask is not None:
            variation = variation * self.mask
        scale = float(np.percentile(variation, 95))
        if scale < self.rbr_variation_min_p95:
            return np.zeros_like(variation, dtype=np.float32)
        return np.clip(variation / scale, 0.0, 1.0).astype(np.float32)

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
