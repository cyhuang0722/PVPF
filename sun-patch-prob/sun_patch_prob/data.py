from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


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


class SunPatchFeatureDataset(Dataset):
    def __init__(self, df: pd.DataFrame, spec: FeatureSpec, weather_weights: dict[str, float] | None = None) -> None:
        self.df = df.reset_index(drop=True).copy()
        x = self.df[spec.feature_columns].to_numpy(dtype=np.float32)
        x = np.where(np.isfinite(x), x, spec.mean)
        self.x = ((x - spec.mean) / spec.std).astype(np.float32)
        self.target = self.df["target_csi"].to_numpy(dtype=np.float32)[:, None]
        self.baseline = self.df["baseline_csi"].to_numpy(dtype=np.float32)[:, None]
        self.aux = self.df[["aux_future_sun_mean", "aux_future_sun_var", "aux_future_sun_delta"]].to_numpy(dtype=np.float32)
        self.aux = np.where(np.isfinite(self.aux), self.aux, 0.0).astype(np.float32)
        weights = weather_weights or {}
        if "weather_tag" in self.df:
            self.weight = self.df["weather_tag"].astype(str).str.strip().str.lower().map(weights).fillna(1.0).to_numpy(dtype=np.float32)[:, None]
        else:
            self.weight = np.ones((len(self.df), 1), dtype=np.float32)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "x": torch.from_numpy(self.x[index]),
            "target": torch.from_numpy(self.target[index]),
            "baseline": torch.from_numpy(self.baseline[index]),
            "aux": torch.from_numpy(self.aux[index]),
            "weight": torch.from_numpy(self.weight[index]),
            "index": torch.tensor(index, dtype=torch.long),
        }
