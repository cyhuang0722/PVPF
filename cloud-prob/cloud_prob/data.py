from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

WEATHER_TO_INDEX = {"clear_sky": 0, "cloudy": 1, "overcast": 2}
INDEX_TO_WEATHER = {value: key for key, value in WEATHER_TO_INDEX.items()}

META_COLUMNS = {
    "interval_start",
    "interval_end",
    "split",
    "image_paths",
    "target_pv_w",
    "target_clear_sky_w",
    "target_csi",
    "baseline_pv_w",
    "baseline_csi",
    "weather_tag",
    "weather_idx",
    "sun_x_px",
    "sun_y_px",
}


@dataclass
class FeatureSpec:
    feature_columns: list[str]
    mean: np.ndarray
    std: np.ndarray


@dataclass
class DataFrames:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    feature_spec: FeatureSpec


def parse_json_list(value: object) -> list:
    if isinstance(value, list):
        return value
    return json.loads(str(value))


def require_file(path: str | Path, label: str) -> Path:
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved


def load_mask(path: str | Path, image_size: int) -> np.ndarray:
    with Image.open(require_file(path, "sky_mask_path")) as image:
        mask = image.convert("L").resize((image_size, image_size), resample=Image.NEAREST)
    return (np.asarray(mask, dtype=np.float32) / 255.0 >= 0.5).astype(np.float32)


def load_rgb(path: str | Path, image_size: int) -> np.ndarray:
    with Image.open(require_file(path, "image_path")) as image:
        rgb = image.convert("RGB").resize((image_size, image_size), resample=Image.BILINEAR)
    return np.asarray(rgb, dtype=np.float32) / 255.0


def infer_feature_columns(df: pd.DataFrame) -> list[str]:
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    return [col for col in numeric if col not in META_COLUMNS]


def fit_feature_spec(df: pd.DataFrame, feature_columns: list[str]) -> FeatureSpec:
    x = df[feature_columns].to_numpy(dtype=np.float32)
    mean = np.nanmean(x, axis=0).astype(np.float32)
    std = np.nanstd(x, axis=0).astype(np.float32)
    mean = np.where(np.isfinite(mean), mean, 0.0).astype(np.float32)
    std = np.where(np.isfinite(std) & (std > 1e-6), std, 1.0).astype(np.float32)
    return FeatureSpec(feature_columns=feature_columns, mean=mean, std=std)


def prepare_frames(samples_csv: str | Path, max_samples: int = 0) -> tuple[DataFrames, list[str]]:
    df = pd.read_csv(require_file(samples_csv, "samples_csv"))
    df["weather_tag"] = df["weather_tag"].astype(str).str.strip().str.lower()
    df["weather_idx"] = df["weather_tag"].map(WEATHER_TO_INDEX).fillna(1).astype(int)
    if max_samples > 0:
        parts = []
        per_split = max(1, max_samples // max(df["split"].nunique(), 1))
        for _, part in df.groupby("split", sort=False):
            parts.append(part.head(min(per_split, len(part))))
        df = pd.concat(parts, ignore_index=True).head(max_samples)

    train = df[df["split"] == "train"].reset_index(drop=True)
    val = df[df["split"] == "val"].reset_index(drop=True)
    test = df[df["split"] == "test"].reset_index(drop=True)
    feature_columns = infer_feature_columns(train)
    spec = fit_feature_spec(train, feature_columns)
    return DataFrames(train=train, val=val, test=test, feature_spec=spec), feature_columns


def build_patch_channels(rgb: np.ndarray, mask: np.ndarray, sun_xy_256: np.ndarray, rbr_clip: float) -> np.ndarray:
    height, width, _ = rgb.shape
    rgb_masked = rgb * mask[..., None]
    rbr = rgb_masked[..., 0:1] / np.clip(rgb_masked[..., 2:3], 1e-3, None)
    rbr = np.clip(rbr, 0.0, float(rbr_clip)) / float(rbr_clip)
    scale = width / 256.0
    sun_x = float(sun_xy_256[0]) * scale
    sun_y = float(sun_xy_256[1]) * scale
    yy, xx = np.meshgrid(np.arange(height, dtype=np.float32), np.arange(width, dtype=np.float32), indexing="ij")
    dist = np.sqrt((xx - sun_x) ** 2 + (yy - sun_y) ** 2)
    dist = dist / max(np.sqrt(height * height + width * width), 1.0)
    return np.transpose(np.concatenate([rgb_masked, rbr, dist[..., None], mask[..., None]], axis=-1), (2, 0, 1)).astype(np.float32)


def crop_center(chw: np.ndarray, center_xy_256: np.ndarray, image_size: int, patch_size: int) -> np.ndarray:
    scale = image_size / 256.0
    center_x = int(round(float(center_xy_256[0]) * scale))
    center_y = int(round(float(center_xy_256[1]) * scale))
    half = patch_size // 2
    channels, height, width = chw.shape
    padded = np.zeros((channels, height + 2 * half, width + 2 * half), dtype=np.float32)
    padded[:, half : half + height, half : half + width] = chw
    x = center_x + half
    y = center_y + half
    return padded[:, y - half : y - half + patch_size, x - half : x - half + patch_size]


class CloudSequenceDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        spec: FeatureSpec,
        image_size: int,
        patch_size: int,
        sky_mask_path: str | Path,
        rbr_clip: float,
        weather_weights: dict[str, float] | None = None,
    ) -> None:
        self.df = df.reset_index(drop=True).copy()
        self.spec = spec
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.mask = load_mask(sky_mask_path, self.image_size)
        self.rbr_clip = float(rbr_clip)

        x = self.df[spec.feature_columns].to_numpy(dtype=np.float32)
        x = np.where(np.isfinite(x), x, spec.mean)
        self.global_x = ((x - spec.mean) / spec.std).astype(np.float32)
        self.target = self.df["target_csi"].to_numpy(dtype=np.float32)[:, None]
        self.baseline = self.df["baseline_csi"].to_numpy(dtype=np.float32)[:, None]
        self.weather_idx = self.df["weather_idx"].to_numpy(dtype=np.int64)
        weights = weather_weights or {}
        self.weight = (
            self.df["weather_tag"].astype(str).str.strip().str.lower().map(weights).fillna(1.0).to_numpy(dtype=np.float32)[:, None]
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.df.iloc[index]
        paths = parse_json_list(row["image_paths"])
        sun_xy = np.asarray([float(row["sun_x_px"]), float(row["sun_y_px"])], dtype=np.float32)
        patches = []
        for path in paths:
            rgb = load_rgb(path, self.image_size)
            chw = build_patch_channels(rgb, self.mask, sun_xy, self.rbr_clip)
            patches.append(crop_center(chw, sun_xy, self.image_size, self.patch_size))
        return {
            "patch_seq": torch.from_numpy(np.stack(patches, axis=0).astype(np.float32)),
            "global_x": torch.from_numpy(self.global_x[index]),
            "weather_idx": torch.tensor(self.weather_idx[index], dtype=torch.long),
            "target": torch.from_numpy(self.target[index]),
            "baseline": torch.from_numpy(self.baseline[index]),
            "weight": torch.from_numpy(self.weight[index]),
            "index": torch.tensor(index, dtype=torch.long),
        }
