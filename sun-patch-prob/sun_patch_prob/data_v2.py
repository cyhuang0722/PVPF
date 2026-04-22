from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from .data import FeatureSpec, fit_feature_spec, infer_feature_columns
from .features import parse_json_list
from .utils import local_path


def load_mask(path: str | Path, image_size: int) -> np.ndarray:
    with Image.open(path) as image:
        mask = image.convert("L").resize((image_size, image_size), resample=Image.NEAREST)
    return (np.asarray(mask, dtype=np.float32) / 255.0 >= 0.5).astype(np.float32)


def load_rgb(path: str | Path, image_size: int) -> np.ndarray:
    with Image.open(local_path(path)) as image:
        rgb = image.convert("RGB").resize((image_size, image_size), resample=Image.BILINEAR)
    return np.asarray(rgb, dtype=np.float32) / 255.0


def build_patch_channels(rgb: np.ndarray, mask: np.ndarray, target_xy_256: np.ndarray) -> np.ndarray:
    h, w, _ = rgb.shape
    rgb_masked = rgb * mask[..., None]
    rbr = rgb_masked[..., 0:1] / np.clip(rgb_masked[..., 2:3], 1e-3, None)
    rbr = np.clip(rbr, 0.0, 4.0) / 4.0
    scale = w / 256.0
    tx = float(target_xy_256[0]) * scale
    ty = float(target_xy_256[1]) * scale
    yy, xx = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
    dist = np.sqrt((xx - tx) ** 2 + (yy - ty) ** 2)
    dist = dist / max(np.sqrt(h * h + w * w), 1.0)
    channels = np.concatenate([rgb_masked, rbr, dist[..., None], mask[..., None]], axis=-1)
    return np.transpose(channels, (2, 0, 1)).astype(np.float32)


def crop_center(chw: np.ndarray, center_xy_256: np.ndarray, image_size: int, patch_size: int) -> np.ndarray:
    scale = image_size / 256.0
    cx = int(round(float(center_xy_256[0]) * scale))
    cy = int(round(float(center_xy_256[1]) * scale))
    half = patch_size // 2
    c, h, w = chw.shape
    padded = np.zeros((c, h + 2 * half, w + 2 * half), dtype=np.float32)
    padded[:, half : half + h, half : half + w] = chw
    px = cx + half
    py = cy + half
    return padded[:, py - half : py - half + patch_size, px - half : px - half + patch_size]


@dataclass
class V2DataFrames:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    feature_spec: FeatureSpec


def prepare_v2_frames(samples_csv: str | Path, feature_csv: str | Path, max_samples: int = 0) -> tuple[V2DataFrames, list[str]]:
    samples = pd.read_csv(samples_csv)
    features = pd.read_csv(feature_csv)
    for df in (samples, features):
        df["ts_anchor"] = pd.to_datetime(df["ts_anchor"]).astype(str)
        df["ts_target"] = pd.to_datetime(df["ts_target"]).astype(str)
    keys = ["ts_anchor", "ts_target", "split"]
    path_cols = ["img_paths", "future_img_paths", "sun_x_px", "sun_y_px", "target_sun_x_px", "target_sun_y_px", "past_pv_w"]
    keep_cols = keys + [c for c in path_cols if c in samples.columns]
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
        c
        for c in all_features
        if c.startswith("past_pv_")
        or c.startswith("solar_")
        or c in {"target_solar_elevation_deg", "sun_dx_px", "sun_dy_px"}
        or c.startswith("weather_")
        or "_global_" in c
    ]
    spec = fit_feature_spec(train, feature_columns)
    return V2DataFrames(train=train, val=val, test=test, feature_spec=spec), feature_columns


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
        self.weight = self.df["weather_tag"].astype(str).str.strip().str.lower().map(weights).fillna(1.0).to_numpy(dtype=np.float32)[:, None]

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

