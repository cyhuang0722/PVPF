from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


def load_and_resize_sky_mask(mask_path: Path, out_h: int, out_w: int) -> np.ndarray:
    with Image.open(mask_path) as m:
        m = m.convert("L")
        m = m.resize((out_w, out_h), resample=Image.NEAREST)
    arr = np.asarray(m, dtype=np.float32) / 255.0
    arr = (arr >= 0.5).astype(np.float32)
    return arr[None, ...]


class PreprocessedUNetDataset(Dataset):
    """
    8 historical sky images sampled every 2 minutes from t-15 to t-1.
    The 8 frames are stacked along channel dimension and mapped to PV at t.
    """

    def __init__(
        self,
        samples_csv: Path,
        image_size: tuple[int, int] = (64, 64),
        channels: int = 3,
        sky_mask_path: Path | None = None,
        camera_path_prefix_from: str | None = None,
        camera_path_prefix_to: str | None = None,
    ):
        super().__init__()
        self.samples_csv = Path(samples_csv)
        self.image_size = image_size
        self.channels = channels
        self.camera_path_prefix_from = camera_path_prefix_from
        self.camera_path_prefix_to = camera_path_prefix_to

        self.df = pd.read_csv(self.samples_csv)
        required = ["ts_target", "img_paths", "pv_target_w"]
        missing = [c for c in required if c not in self.df.columns]
        if missing:
            raise ValueError(f"samples csv missing columns: {missing}")

        self.sky_mask = None
        if sky_mask_path is not None and Path(sky_mask_path).exists():
            h, w = self.image_size
            mask_1hw = load_and_resize_sky_mask(Path(sky_mask_path), h, w)
            self.sky_mask = mask_1hw if self.channels == 1 else np.repeat(mask_1hw, repeats=3, axis=0)

    def _rewrite_path(self, path: str) -> str:
        if self.camera_path_prefix_from and self.camera_path_prefix_to:
            if path.startswith(self.camera_path_prefix_from):
                return self.camera_path_prefix_to + path[len(self.camera_path_prefix_from):]
        return path

    def _load_image(self, path: str) -> np.ndarray:
        h, w = self.image_size
        with Image.open(path) as im:
            if self.channels == 1:
                im = im.convert("L")
                arr = np.asarray(im.resize((w, h), resample=Image.BILINEAR), dtype=np.float32) / 255.0
                return arr[None, ...]
            im = im.convert("RGB")
            arr = np.asarray(im.resize((w, h), resample=Image.BILINEAR), dtype=np.float32) / 255.0
            return np.transpose(arr, (2, 0, 1))

    def __len__(self) -> int:
        return len(self.df)

    @property
    def ts_target_series(self):
        return pd.to_datetime(self.df["ts_target"])

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        paths_raw = row["img_paths"]
        paths = list(ast.literal_eval(paths_raw)) if isinstance(paths_raw, str) else list(paths_raw)
        frames = [self._load_image(self._rewrite_path(str(path))) for path in paths]
        x = np.concatenate(frames, axis=0)
        if self.sky_mask is not None:
            x = x * np.repeat(self.sky_mask, repeats=len(frames), axis=0)

        y = np.float32(row["pv_target_w"])
        return {
            "image_stack": torch.from_numpy(x),
            "y": torch.tensor(y),
            "t_target": str(row["ts_target"]),
        }
