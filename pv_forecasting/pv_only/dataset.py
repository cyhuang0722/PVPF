"""
PV-only dataset loader.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class PVPastDataset(Dataset):
    """
    Each sample:
    - pv_past: (8, 1) -> [t-120, t-105, ..., t-15]
    - target: scalar -> t+15
    """

    def __init__(self, csv_path: Path):
        self.df = pd.read_csv(csv_path)
        if len(self.df) and isinstance(self.df["past_pv"].iloc[0], str):
            self.df["past_pv"] = self.df["past_pv"].apply(ast.literal_eval)

    def __len__(self) -> int:
        return len(self.df)

    @property
    def ts_anchor_series(self) -> pd.Series:
        return pd.to_datetime(self.df["ts_anchor"])

    @property
    def ts_target_series(self) -> pd.Series:
        return pd.to_datetime(self.df["ts_target"])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        pv_past = np.array(row["past_pv"], dtype=np.float32).reshape(8, 1)
        target = np.array([row["target"]], dtype=np.float32)
        return {
            "pv_past": torch.from_numpy(pv_past),
            "target": torch.from_numpy(target),
        }

