from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .satellite_common import SatellitePatchExtractor, load_stats


class SatelliteForecastDataset(Dataset):
    def __init__(
        self,
        csv_path: Path,
        split: str | None = None,
        stats_path: Path | None = None,
        peak_power_w: float = 66300.0,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.df = pd.read_csv(self.csv_path)
        if "sat_paths" in self.df.columns and len(self.df) and isinstance(self.df["sat_paths"].iloc[0], str):
            self.df["sat_paths"] = self.df["sat_paths"].apply(ast.literal_eval)
        if "targets" in self.df.columns and len(self.df) and isinstance(self.df["targets"].iloc[0], str):
            self.df["targets"] = self.df["targets"].apply(ast.literal_eval)
        if split and "split" in self.df.columns:
            self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        if self.df.empty:
            raise ValueError(f"No samples found in {self.csv_path} for split={split}")

        first = self.df.iloc[0]
        self.extractor = SatellitePatchExtractor(
            channels=tuple(ast.literal_eval(first["channels"]) if isinstance(first["channels"], str) else first["channels"]),
            center_lat=float(first["center_lat"]),
            center_lon=float(first["center_lon"]),
            patch_size=int(first["patch_size"]),
        )
        self.mean = None
        self.std = None
        if stats_path is not None and Path(stats_path).exists():
            self.mean, self.std = load_stats(Path(stats_path))
        self.peak_power_w = float(peak_power_w)

    def __len__(self) -> int:
        return len(self.df)

    @property
    def ts_pred_series(self) -> pd.Series:
        return pd.to_datetime(self.df["ts_pred"])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        sat = np.stack([self.extractor.read_patch(path) for path in row["sat_paths"]], axis=0)
        if self.mean is not None and self.std is not None:
            sat = (sat - self.mean[None, :, None, None]) / self.std[None, :, None, None]
        targets = np.asarray(row["targets"], dtype=np.float32) / np.float32(self.peak_power_w)
        return {
            "satellite": torch.from_numpy(sat.astype(np.float32, copy=False)),
            "targets": torch.from_numpy(targets),
        }


class PackedSatelliteDataset(Dataset):
    def __init__(self, pack_dir: Path, split: str | None = None) -> None:
        self.pack_dir = Path(pack_dir)
        shards = sorted(self.pack_dir.glob("batch_*.npz"))
        if not shards:
            raise FileNotFoundError(f"No batch_*.npz found in {self.pack_dir}")

        sat_list, target_list, ts_list, split_list = [], [], [], []
        for shard in shards:
            data = np.load(shard, allow_pickle=True)
            sat_list.append(data["satellite"])
            target_list.append(data["targets"])
            ts_list.append(data["ts_pred"])
            if "split" in data:
                split_list.append(data["split"])
        self.satellite = np.concatenate(sat_list, axis=0)
        self.targets = np.concatenate(target_list, axis=0)
        self.ts_pred = np.concatenate(ts_list, axis=0)
        self.split = np.concatenate(split_list, axis=0) if split_list else None

        if split and self.split is not None:
            mask = self.split == split
            self.satellite = self.satellite[mask]
            self.targets = self.targets[mask]
            self.ts_pred = self.ts_pred[mask]
            self.split = self.split[mask]
        if len(self.satellite) == 0:
            raise ValueError(f"No packed samples found for split={split} in {self.pack_dir}")

    def __len__(self) -> int:
        return len(self.satellite)

    @property
    def ts_pred_series(self) -> pd.Series:
        return pd.to_datetime(pd.Series(self.ts_pred))

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "satellite": torch.from_numpy(self.satellite[idx].copy()),
            "targets": torch.from_numpy(self.targets[idx].copy()),
        }
