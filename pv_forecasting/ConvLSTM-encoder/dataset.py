from __future__ import annotations

import ast
import json
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
    return arr[None, ...]  # (1, H, W)


class CameraPVConvLSTMDataset(Dataset):
    """
    Input sequence timestamps: [t-15, t-13, t-1] (minutes).
    Target timestamp: t+15 (minutes).
    """

    def __init__(
        self,
        power_csv: Path,
        camera_index_csv: Path,
        image_size: tuple[int, int] = (128, 128),
        channels: int = 1,
        max_time_diff_sec: int = 90,
        sky_mask_path: Path | None = None,
        camera_path_prefix_from: str | None = None,
        camera_path_prefix_to: str | None = None,
    ):
        super().__init__()
        self.power_csv = Path(power_csv)
        self.camera_index_csv = Path(camera_index_csv)
        self.image_size = image_size
        self.channels = channels
        self.max_time_diff_sec = max_time_diff_sec
        self.camera_path_prefix_from = camera_path_prefix_from
        self.camera_path_prefix_to = camera_path_prefix_to
        self.sky_mask = None
        if sky_mask_path is not None and Path(sky_mask_path).exists():
            h, w = self.image_size
            mask_1hw = load_and_resize_sky_mask(Path(sky_mask_path), h, w)
            self.sky_mask = mask_1hw if self.channels == 1 else np.repeat(mask_1hw, repeats=3, axis=0)
        self.input_offsets_min = [15, 13, 11, 9, 7, 5, 3, 1]
        self.target_offset_min = 15
        self.samples = self._build_samples()

    def _rewrite_path(self, path: str) -> str:
        if self.camera_path_prefix_from and self.camera_path_prefix_to:
            if path.startswith(self.camera_path_prefix_from):
                return self.camera_path_prefix_to + path[len(self.camera_path_prefix_from):]
        return path

    def _get_nearest_path(
        self,
        desired_ts: pd.Timestamp,
        cam_ts_ns: np.ndarray,
        cam_paths: np.ndarray,
    ) -> str | None:
        desired_ns = desired_ts.value
        pos = int(np.searchsorted(cam_ts_ns, desired_ns))
        candidate_idx: list[int] = []
        if 0 <= pos < len(cam_ts_ns):
            candidate_idx.append(pos)
        if pos - 1 >= 0:
            candidate_idx.append(pos - 1)
        if not candidate_idx:
            return None
        best_i = min(candidate_idx, key=lambda i: abs(int(cam_ts_ns[i]) - desired_ns))
        diff_sec = abs(int(cam_ts_ns[best_i]) - desired_ns) / 1e9
        if diff_sec > self.max_time_diff_sec:
            return None
        return str(cam_paths[best_i])

    def _build_samples(self) -> list[dict]:
        power_df = pd.read_csv(self.power_csv)
        if "date" not in power_df.columns or "value" not in power_df.columns:
            raise ValueError("power csv must include columns: date,value")
        power_df["date"] = pd.to_datetime(power_df["date"], errors="coerce")
        power_df["value"] = pd.to_numeric(power_df["value"], errors="coerce")
        power_df = power_df.dropna(subset=["date", "value"]).sort_values("date").reset_index(drop=True)
        power_df = power_df[power_df["value"] > 0].reset_index(drop=True)
        pv_map = dict(zip(power_df["date"], power_df["value"]))

        cam_df = pd.read_csv(self.camera_index_csv)
        if "timestamp" not in cam_df.columns:
            raise ValueError("camera index must include timestamp column")
        if "file_path" not in cam_df.columns:
            raise ValueError("camera index must include file_path column")
        cam_df["timestamp"] = pd.to_datetime(cam_df["timestamp"], errors="coerce")
        cam_df = cam_df.dropna(subset=["timestamp", "file_path"]).copy()
        cam_df = cam_df.sort_values("timestamp").reset_index(drop=True)
        cam_ts_ns = cam_df["timestamp"].astype("int64").to_numpy()
        cam_paths = cam_df["file_path"].astype(str).to_numpy()

        samples: list[dict] = []
        # Build samples from PV timestamps to guarantee target existence.
        for target_ts in power_df["date"].tolist():
            t = target_ts - pd.Timedelta(minutes=self.target_offset_min)
            img_ts = [
                t - pd.Timedelta(minutes=offset)
                for offset in self.input_offsets_min
            ]
            paths: list[str] = []
            missing = False
            for ts in img_ts:
                path = self._get_nearest_path(ts, cam_ts_ns, cam_paths)
                if path is None:
                    missing = True
                    break
                path = self._rewrite_path(path)
                if not Path(path).exists():
                    missing = True
                    break
                paths.append(path)
            if missing:
                continue
            samples.append(
                {
                    "t_anchor": t,
                    "t_input_0": img_ts[0],
                    "t_input_1": img_ts[1],
                    "t_input_2": img_ts[2],
                    "t_target": target_ts,
                    "img_paths": paths,
                    "pv_target_w": float(pv_map[target_ts]),
                }
            )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, path: str) -> np.ndarray:
        h, w = self.image_size
        with Image.open(path) as im:
            if self.channels == 1:
                im = im.convert("L")
                arr = np.asarray(im.resize((w, h), resample=Image.BILINEAR), dtype=np.float32) / 255.0
                return arr[None, ...]  # (1, H, W)
            im = im.convert("RGB")
            arr = np.asarray(im.resize((w, h), resample=Image.BILINEAR), dtype=np.float32) / 255.0
            return np.transpose(arr, (2, 0, 1))  # (3, H, W)

    def __getitem__(self, idx: int):
        row = self.samples[idx]
        frames = [self._load_image(p) for p in row["img_paths"]]
        x = np.stack(frames, axis=0)  # (T=3, C, H, W)
        if self.sky_mask is not None:
            x = x * self.sky_mask[None, ...]
        y = np.float32(row["pv_target_w"])
        return {
            "x_seq": torch.from_numpy(x),
            "y": torch.tensor(y),
            "t_anchor": str(row["t_anchor"]),
            "t_target": str(row["t_target"]),
        }


class PreprocessedConvLSTMDataset(Dataset):
    def __init__(
        self,
        samples_csv: Path,
        split: str | None = None,
        image_size: tuple[int, int] = (128, 128),
        channels: int = 1,
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
        self.sky_mask = None
        if sky_mask_path is not None and Path(sky_mask_path).exists():
            h, w = self.image_size
            mask_1hw = load_and_resize_sky_mask(Path(sky_mask_path), h, w)
            self.sky_mask = mask_1hw if self.channels == 1 else np.repeat(mask_1hw, repeats=3, axis=0)
        self.df = pd.read_csv(self.samples_csv)
        if split is not None:
            if "split" not in self.df.columns:
                raise ValueError("samples csv missing required column: split")
            self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        required = ["ts_anchor", "ts_target", "img_paths"]
        missing = [c for c in required if c not in self.df.columns]
        if missing:
            raise ValueError(f"samples csv missing columns: {missing}")
        if "target_value" not in self.df.columns:
            if "pv_target_w" in self.df.columns:
                self.df["target_value"] = self.df["pv_target_w"].astype(np.float32)
            elif "target_pv_w" in self.df.columns:
                self.df["target_value"] = self.df["target_pv_w"].astype(np.float32)
            else:
                raise ValueError("samples csv must include target_value or pv_target_w/target_pv_w")
        if "target_pv_w" not in self.df.columns:
            if "pv_target_w" in self.df.columns:
                self.df["target_pv_w"] = self.df["pv_target_w"].astype(np.float32)
            else:
                raise ValueError("samples csv must include target_pv_w or pv_target_w")
        if "target_clear_sky_w" not in self.df.columns:
            self.df["target_clear_sky_w"] = np.ones(len(self.df), dtype=np.float32)
        if "past_pv_w" not in self.df.columns:
            raise ValueError("samples csv missing required column: past_pv_w")
        self.df = self.df.reset_index(drop=True)

    def _rewrite_path(self, path: str) -> str:
        if self.camera_path_prefix_from and self.camera_path_prefix_to:
            if path.startswith(self.camera_path_prefix_from):
                return self.camera_path_prefix_to + path[len(self.camera_path_prefix_from):]
        return path

    def __len__(self) -> int:
        return len(self.df)

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

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        paths_raw = row["img_paths"]
        if isinstance(paths_raw, str):
            try:
                paths = list(json.loads(paths_raw))
            except json.JSONDecodeError:
                paths = list(ast.literal_eval(paths_raw))
        else:
            paths = list(paths_raw)
        paths = [self._rewrite_path(str(p)) for p in paths]
        frames = [self._load_image(p) for p in paths]
        x = np.stack(frames, axis=0)
        if self.sky_mask is not None:
            x = x * self.sky_mask[None, ...]
        target_value = np.float32(row["target_value"])
        target_pv_w = np.float32(row["target_pv_w"])
        target_clear_sky_w = np.float32(row["target_clear_sky_w"])
        past_pv_raw = row["past_pv_w"]
        if isinstance(past_pv_raw, str):
            try:
                past_pv = json.loads(past_pv_raw)
            except json.JSONDecodeError:
                past_pv = ast.literal_eval(past_pv_raw)
        else:
            past_pv = list(past_pv_raw)
        past_pv_w = np.asarray(past_pv, dtype=np.float32)
        denom = max(float(target_clear_sky_w), 1e-6)
        past_pv_csi = np.clip(past_pv_w / denom, a_min=0.0, a_max=1.0)
        return {
            "x_seq": torch.from_numpy(x),
            "pv_history": torch.tensor(past_pv_csi, dtype=torch.float32),
            "pv_history_w": torch.tensor(past_pv_w, dtype=torch.float32),
            "target": torch.tensor(target_value),
            "target_pv_w": torch.tensor(target_pv_w),
            "target_clear_sky_w": torch.tensor(target_clear_sky_w),
            "meta_index": torch.tensor(idx, dtype=torch.long),
            "t_anchor": str(row["ts_anchor"]),
            "t_target": str(row["ts_target"]),
        }
