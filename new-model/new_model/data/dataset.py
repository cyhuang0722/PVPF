from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


def load_mask(mask_path: str | Path, size: tuple[int, int]) -> np.ndarray:
    mask = Image.open(mask_path).convert("L").resize((size[1], size[0]), resample=Image.NEAREST)
    arr = (np.asarray(mask, dtype=np.float32) / 255.0 >= 0.5).astype(np.float32)
    return arr[None, ...]


def load_rgb_image(path: str | Path, size: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as im:
        rgb = im.convert("RGB").resize((size[1], size[0]), resample=Image.BILINEAR)
    arr = np.asarray(rgb, dtype=np.float32) / 255.0
    return np.transpose(arr, (2, 0, 1))


class SunConditionedPVDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        split: str,
        image_size: tuple[int, int],
        sky_mask_path: str | Path | None = None,
    ) -> None:
        self.df = pd.read_csv(csv_path)
        self.df["ts_anchor"] = pd.to_datetime(self.df["ts_anchor"])
        self.df["ts_target"] = pd.to_datetime(self.df["ts_target"])
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        self.image_size = image_size
        self.mask = load_mask(sky_mask_path, image_size) if sky_mask_path and Path(sky_mask_path).exists() else None

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> dict:
        row = self.df.iloc[index]
        img_paths = row["img_paths"]
        if isinstance(img_paths, str):
            img_paths = json.loads(img_paths)
        past_pv = row["past_pv_w"]
        if isinstance(past_pv, str):
            try:
                past_pv = json.loads(past_pv)
            except json.JSONDecodeError:
                past_pv = ast.literal_eval(past_pv)
        solar_vec = row["solar_vec"]
        if isinstance(solar_vec, str):
            solar_vec = json.loads(solar_vec)

        frames = np.stack([load_rgb_image(p, self.image_size) for p in img_paths], axis=0)
        if self.mask is not None:
            frames = frames * self.mask

        return {
            "images": torch.from_numpy(frames.astype(np.float32)),
            "pv_history": torch.tensor(np.asarray(past_pv, dtype=np.float32)),
            "solar_vec": torch.tensor(np.asarray(solar_vec, dtype=np.float32)),
            "target": torch.tensor(float(row["target_value"]), dtype=torch.float32),
            "target_pv_w": torch.tensor(float(row["target_pv_w"]), dtype=torch.float32),
            "target_clear_sky_w": torch.tensor(float(row["target_clear_sky_w"]), dtype=torch.float32),
            "sun_xy": torch.tensor([float(row["sun_x_px"]), float(row["sun_y_px"])], dtype=torch.float32),
            "meta_index": torch.tensor(index, dtype=torch.long),
        }

    def dataframe(self) -> pd.DataFrame:
        return self.df.copy()
