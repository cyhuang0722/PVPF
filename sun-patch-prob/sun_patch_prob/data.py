from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from .utils import local_path


META_COLUMNS = {
    "ts_anchor",
    "ts_target",
    "split",
    "target_pv_w",
    "target_clear_sky_w",
    "anchor_clear_sky_w",
    "target_csi",
    "baseline_csi",
    "baseline_pv_w",
    "smart_persistence_csi",
    "smart_persistence_pv_w",
    "persistence_pv_w",
    "persistence_csi_target_clear",
    "aux_future_sun_mean",
    "aux_future_sun_var",
    "aux_future_sun_delta",
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


def infer_feature_columns(df: pd.DataFrame) -> list[str]:
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    return [col for col in numeric if col not in META_COLUMNS and not col.startswith("future_")]


def fit_feature_spec(df: pd.DataFrame, feature_columns: list[str]) -> FeatureSpec:
    x = df[feature_columns].to_numpy(dtype=np.float32)
    mean = np.nanmean(x, axis=0).astype(np.float32)
    std = np.nanstd(x, axis=0).astype(np.float32)
    mean = np.where(np.isfinite(mean), mean, 0.0).astype(np.float32)
    std = np.where(np.isfinite(std) & (std > 1e-6), std, 1.0).astype(np.float32)
    return FeatureSpec(feature_columns=feature_columns, mean=mean, std=std)


def prepare_frames(samples_csv: str | Path, feature_csv: str | Path, max_samples: int = 0) -> tuple[DataFrames, list[str]]:
    samples = pd.read_csv(samples_csv)
    features = pd.read_csv(feature_csv)
    for frame in (samples, features):
        frame["ts_anchor"] = pd.to_datetime(frame["ts_anchor"]).astype(str)
        frame["ts_target"] = pd.to_datetime(frame["ts_target"]).astype(str)

    keys = ["ts_anchor", "ts_target", "split"]
    path_cols = [
        "img_paths",
        "future_img_paths",
        "sun_x_px",
        "sun_y_px",
        "target_sun_x_px",
        "target_sun_y_px",
        "past_pv_w",
    ]
    keep_cols = keys + [col for col in path_cols if col in samples.columns]
    merged = features.merge(samples[keep_cols], on=keys, how="left", suffixes=("", "_sample"))

    if max_samples > 0:
        parts = []
        per_split = max(1, max_samples // max(merged["split"].nunique(), 1))
        for _, part in merged.groupby("split", sort=False):
            parts.append(part.head(min(per_split, len(part))))
        merged = pd.concat(parts, ignore_index=True).head(max_samples)

    train = merged[merged["split"] == "train"].reset_index(drop=True)
    val = merged[merged["split"] == "val"].reset_index(drop=True)
    test = merged[merged["split"] == "test"].reset_index(drop=True)

    all_features = infer_feature_columns(train)
    feature_columns = [
        col
        for col in all_features
        if col.startswith("past_pv_")
        or col.startswith("solar_")
        or col in {"target_solar_elevation_deg", "sun_dx_px", "sun_dy_px"}
        or col.startswith("weather_")
        or "_global_" in col
    ]
    spec = fit_feature_spec(train, feature_columns)
    return DataFrames(train=train, val=val, test=test, feature_spec=spec), feature_columns


def load_mask(path: str | Path, image_size: int) -> np.ndarray:
    with Image.open(path) as image:
        mask = image.convert("L").resize((image_size, image_size), resample=Image.NEAREST)
    return (np.asarray(mask, dtype=np.float32) / 255.0 >= 0.5).astype(np.float32)


def load_rgb(path: str | Path, image_size: int) -> np.ndarray:
    with Image.open(local_path(path)) as image:
        rgb = image.convert("RGB").resize((image_size, image_size), resample=Image.BILINEAR)
    return np.asarray(rgb, dtype=np.float32) / 255.0


def build_patch_channels(rgb: np.ndarray, mask: np.ndarray, target_xy_256: np.ndarray) -> np.ndarray:
    height, width, _ = rgb.shape
    rgb_masked = rgb * mask[..., None]
    rbr = rgb_masked[..., 0:1] / np.clip(rgb_masked[..., 2:3], 1e-3, None)
    rbr = np.clip(rbr, 0.0, 4.0) / 4.0
    scale = width / 256.0
    target_x = float(target_xy_256[0]) * scale
    target_y = float(target_xy_256[1]) * scale
    yy, xx = np.meshgrid(np.arange(height, dtype=np.float32), np.arange(width, dtype=np.float32), indexing="ij")
    dist = np.sqrt((xx - target_x) ** 2 + (yy - target_y) ** 2)
    dist = dist / max(np.sqrt(height * height + width * width), 1.0)
    channels = np.concatenate([rgb_masked, rbr, dist[..., None], mask[..., None]], axis=-1)
    return np.transpose(channels, (2, 0, 1)).astype(np.float32)


def crop_center(chw: np.ndarray, center_xy_256: np.ndarray, image_size: int, patch_size: int) -> np.ndarray:
    scale = image_size / 256.0
    center_x = int(round(float(center_xy_256[0]) * scale))
    center_y = int(round(float(center_xy_256[1]) * scale))
    half = patch_size // 2
    channels, height, width = chw.shape
    padded = np.zeros((channels, height + 2 * half, width + 2 * half), dtype=np.float32)
    padded[:, half : half + height, half : half + width] = chw
    padded_x = center_x + half
    padded_y = center_y + half
    return padded[:, padded_y - half : padded_y - half + patch_size, padded_x - half : padded_x - half + patch_size]


class SunPatchSequenceDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        spec: FeatureSpec,
        image_size: int,
        patch_size: int,
        sky_mask_path: str | Path,
        weather_weights: dict[str, float] | None = None,
    ) -> None:
        self.df = df.reset_index(drop=True).copy()
        self.spec = spec
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.mask = load_mask(sky_mask_path, self.image_size)

        x = self.df[spec.feature_columns].to_numpy(dtype=np.float32)
        x = np.where(np.isfinite(x), x, spec.mean)
        self.global_x = ((x - spec.mean) / spec.std).astype(np.float32)
        self.target = self.df["target_csi"].to_numpy(dtype=np.float32)[:, None]
        self.baseline = self.df["baseline_csi"].to_numpy(dtype=np.float32)[:, None]
        self.aux = self.df[["aux_future_sun_mean", "aux_future_sun_var", "aux_future_sun_delta"]].to_numpy(dtype=np.float32)
        self.aux = np.where(np.isfinite(self.aux), self.aux, 0.0).astype(np.float32)
        weights = weather_weights or {}
        self.weight = (
            self.df["weather_tag"].astype(str).str.strip().str.lower().map(weights).fillna(1.0).to_numpy(dtype=np.float32)[:, None]
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.df.iloc[index]
        paths = parse_json_list(row["img_paths"])
        target_xy = np.asarray([float(row["target_sun_x_px"]), float(row["target_sun_y_px"])], dtype=np.float32)
        patches = []
        for path in paths:
            rgb = load_rgb(path, self.image_size)
            chw = build_patch_channels(rgb, self.mask, target_xy)
            patches.append(crop_center(chw, target_xy, self.image_size, self.patch_size))
        patch_seq = np.stack(patches, axis=0).astype(np.float32)
        return {
            "patch_seq": torch.from_numpy(patch_seq),
            "global_x": torch.from_numpy(self.global_x[index]),
            "target": torch.from_numpy(self.target[index]),
            "baseline": torch.from_numpy(self.baseline[index]),
            "aux": torch.from_numpy(self.aux[index]),
            "weight": torch.from_numpy(self.weight[index]),
            "index": torch.tensor(index, dtype=torch.long),
        }
