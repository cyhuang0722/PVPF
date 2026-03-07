"""
Forecasting Dataset：从 forecast_windows.csv 读取按需加载，或从预打包 .npz 分片加载。
支持 sky_mask（与 pv_output_prediction 一致：二值 mask 乘到图像上）。
"""
import ast
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image


def load_and_resize_sky_mask(mask_path: Path, out_h: int, out_w: int) -> np.ndarray:
    """与 pv_output_prediction 一致：灰度图 resize、二值化，返回 (1, H, W) float32 {0, 1}。"""
    if not Path(mask_path).exists():
        raise FileNotFoundError(f"Sky mask not found: {mask_path}")
    m = Image.open(mask_path).convert("L")
    m = m.resize((out_w, out_h), resample=Image.NEAREST)
    m = np.asarray(m, dtype=np.float32) / 255.0
    m = (m >= 0.5).astype(np.float32)
    return m[None, ...]


def load_image(path: str, size=(128, 128)) -> np.ndarray:
    """加载并 resize，返回 (C,H,W) float32 [0,1]。"""
    with Image.open(path) as im:
        im = im.convert("RGB")
    arr = np.asarray(im.resize(size), dtype=np.float32) / 255.0
    return np.transpose(arr, (2, 0, 1))


class ForecastDataset(Dataset):
    """每样本: 60 张图 (60,3,H,W)，过去 PV (4,1)，目标 (5,)。可选 sky_mask 乘到每帧。"""

    def __init__(
        self,
        csv_path: Path,
        img_size=(128, 128),
        base_dir: Path | None = None,
        sky_mask_path: Optional[Path] = None,
    ):
        self.df = pd.read_csv(csv_path)
        for col in ("img_paths", "past_pv", "targets"):
            if col in self.df.columns and len(self.df) and isinstance(self.df[col].iloc[0], str):
                self.df[col] = self.df[col].apply(ast.literal_eval)
        self.img_size = img_size
        self.base_dir = Path(base_dir) if base_dir else Path(csv_path).resolve().parent.parent
        self.sky_mask_1hw: Optional[np.ndarray] = None
        if sky_mask_path and Path(sky_mask_path).exists():
            self.sky_mask_1hw = load_and_resize_sky_mask(
                Path(sky_mask_path), img_size[0], img_size[1]
            )

    def __len__(self):
        return len(self.df)

    @property
    def ts_pred_series(self):
        return pd.to_datetime(self.df["ts_pred"])

    def __getitem__(self, i):
        row = self.df.iloc[i]
        paths = row["img_paths"]
        past_pv = np.array(row["past_pv"], dtype=np.float32).reshape(4, 1)
        targets = np.array(row["targets"], dtype=np.float32)

        imgs = []
        for p in paths:
            path = Path(p)
            if not path.is_absolute():
                path = self.base_dir / path
            imgs.append(load_image(str(path), self.img_size))
        sky = np.stack(imgs, axis=0)
        if self.sky_mask_1hw is not None:
            sky = sky * self.sky_mask_1hw

        return {
            "sky": torch.from_numpy(sky),
            "pv_past": torch.from_numpy(past_pv),
            "targets": torch.from_numpy(targets),
        }


class PackedForecastDataset(Dataset):
    """从预打包的 .npz 分片加载，无需按需读图。数据已在打包时 resize/归一化/可选 mask。"""

    def __init__(self, pack_dir: Path):
        self.pack_dir = Path(pack_dir)
        shards = sorted(self.pack_dir.glob("batch_*.npz"))
        if not shards:
            raise FileNotFoundError(f"No batch_*.npz in {self.pack_dir}")
        self.sky_list = []
        self.past_pv_list = []
        self.targets_list = []
        self.ts_list = []
        for p in shards:
            d = np.load(p, allow_pickle=True)
            self.sky_list.append(d["sky"])
            self.past_pv_list.append(d["past_pv"])
            self.targets_list.append(d["targets"])
            self.ts_list.append(d["ts_pred"])
        self.sky = np.concatenate(self.sky_list, axis=0)
        self.past_pv = np.concatenate(self.past_pv_list, axis=0)
        self.targets = np.concatenate(self.targets_list, axis=0)
        self.ts = np.concatenate(self.ts_list, axis=0)

    def __len__(self):
        return len(self.sky)

    def __getitem__(self, i):
        return {
            "sky": torch.from_numpy(self.sky[i].copy()),
            "pv_past": torch.from_numpy(self.past_pv[i].copy()),
            "targets": torch.from_numpy(self.targets[i].copy()),
        }

    @property
    def ts_pred_series(self):
        """用于保存 predictions_all 时对齐 ts_pred。"""
        return pd.to_datetime(pd.Series(self.ts))
