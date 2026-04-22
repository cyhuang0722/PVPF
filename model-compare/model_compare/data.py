from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

WEATHER_TO_INDEX = {"clear_sky": 0, "cloudy": 1, "overcast": 2}
INDEX_TO_WEATHER = {value: key for key, value in WEATHER_TO_INDEX.items()}


def parse_json_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(item) for item in json.loads(str(value))]


def require_file(path: str | Path, label: str) -> Path:
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved


def load_mask(path: str | Path, image_size: int) -> np.ndarray:
    with Image.open(require_file(path, "sky_mask_path")) as image:
        mask = image.convert("L").resize((image_size, image_size), resample=Image.NEAREST)
    return (np.asarray(mask, dtype=np.float32) / 255.0 >= 0.5).astype(np.float32)


def load_rgb(path: str | Path, image_size: int, mask: np.ndarray | None) -> np.ndarray:
    with Image.open(require_file(path, "image_path")) as image:
        rgb = image.convert("RGB").resize((image_size, image_size), resample=Image.BILINEAR)
    arr = np.asarray(rgb, dtype=np.float32) / 255.0
    if mask is not None:
        arr = arr * mask[..., None]
    return np.transpose(arr, (2, 0, 1)).astype(np.float32)


def load_frames(samples_csv: str | Path, max_samples: int = 0) -> dict[str, pd.DataFrame]:
    df = pd.read_csv(require_file(samples_csv, "samples_csv"))
    df["weather_tag"] = df["weather_tag"].astype(str).str.strip().str.lower()
    df["weather_idx"] = df["weather_tag"].map(WEATHER_TO_INDEX).fillna(1).astype(int)
    if max_samples > 0:
        parts = []
        per_split = max(1, max_samples // max(df["split"].nunique(), 1))
        for _, part in df.groupby("split", sort=False):
            parts.append(part.head(min(per_split, len(part))))
        df = pd.concat(parts, ignore_index=True).head(max_samples)
    return {
        split: df[df["split"] == split].reset_index(drop=True).copy()
        for split in ["train", "val", "test"]
    }


class ImageSequenceDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        image_size: int,
        sky_mask_path: str | Path,
        sequence_mode: str,
        max_steps: int,
        use_sky_mask: bool,
    ) -> None:
        self.df = df.reset_index(drop=True).copy()
        self.image_size = int(image_size)
        self.sequence_mode = str(sequence_mode)
        self.max_steps = int(max_steps)
        self.mask = load_mask(sky_mask_path, self.image_size) if use_sky_mask else None
        self.target_csi = self.df["target_csi"].to_numpy(dtype=np.float32)[:, None]
        self.target_pv_w = self.df["target_pv_w"].to_numpy(dtype=np.float32)[:, None]
        self.clear_sky_w = self.df["target_clear_sky_w"].to_numpy(dtype=np.float32)[:, None]
        self.baseline_pv_w = self.df["baseline_pv_w"].to_numpy(dtype=np.float32)[:, None]
        self.weather_idx = self.df["weather_idx"].to_numpy(dtype=np.int64)

    def __len__(self) -> int:
        return len(self.df)

    def _select_paths(self, paths: list[str]) -> list[str]:
        if self.sequence_mode == "latest":
            return [paths[-1]]
        if self.max_steps > 0 and len(paths) > self.max_steps:
            indices = np.linspace(0, len(paths) - 1, self.max_steps).round().astype(int)
            return [paths[int(idx)] for idx in indices]
        return paths

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.df.iloc[index]
        paths = self._select_paths(parse_json_list(row["image_paths"]))
        images = [load_rgb(path, self.image_size, self.mask) for path in paths]
        return {
            "images": torch.from_numpy(np.stack(images, axis=0).astype(np.float32)),
            "target": torch.from_numpy(self.target_csi[index]),
            "clear_sky_w": torch.from_numpy(self.clear_sky_w[index]),
            "target_pv_w": torch.from_numpy(self.target_pv_w[index]),
            "baseline_pv_w": torch.from_numpy(self.baseline_pv_w[index]),
            "weather_idx": torch.tensor(self.weather_idx[index], dtype=torch.long),
            "index": torch.tensor(index, dtype=torch.long),
        }
