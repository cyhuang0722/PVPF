from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
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
        peak_power_w: float | None = None,
    ) -> None:
        self.df = pd.read_csv(csv_path)
        self.df["ts_anchor"] = pd.to_datetime(self.df["ts_anchor"])
        self.df["ts_target"] = pd.to_datetime(self.df["ts_target"])
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        self.image_size = image_size
        self.mask = load_mask(sky_mask_path, image_size) if sky_mask_path and Path(sky_mask_path).exists() else None
        self.peak_power_w = float(peak_power_w) if peak_power_w is not None else 1.0
        self.flow_size = (
            max(1, image_size[0] // 8),
            max(1, image_size[1] // 8),
        )
        if self.peak_power_w <= 0:
            raise ValueError("peak_power_w must be positive.")

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

        azimuth_rad = np.deg2rad(float(row["azimuth_deg"]))
        elevation_rad = np.deg2rad(90.0 - float(row["zenith_deg"]))
        sun_angles = np.asarray([azimuth_rad / np.pi, elevation_rad / (0.5 * np.pi)], dtype=np.float32)
        flow_gt, flow_mask = self._build_pseudo_flow(frames, np.asarray([float(row["sun_x_px"]), float(row["sun_y_px"])], dtype=np.float32))

        return {
            "images": torch.from_numpy(frames.astype(np.float32)),
            "pv_history": torch.tensor(np.asarray(past_pv, dtype=np.float32) / self.peak_power_w),
            "solar_vec": torch.tensor(np.asarray(solar_vec, dtype=np.float32)),
            "sun_angles": torch.tensor(sun_angles, dtype=torch.float32),
            "target": torch.tensor(float(row["target_value"]), dtype=torch.float32),
            "target_pv_w": torch.tensor(float(row["target_pv_w"]), dtype=torch.float32),
            "target_clear_sky_w": torch.tensor(float(row["target_clear_sky_w"]), dtype=torch.float32),
            "sun_xy": torch.tensor([float(row["sun_x_px"]), float(row["sun_y_px"])], dtype=torch.float32),
            "flow_gt": torch.from_numpy(flow_gt),
            "flow_mask": torch.from_numpy(flow_mask),
            "meta_index": torch.tensor(index, dtype=torch.long),
        }

    def dataframe(self) -> pd.DataFrame:
        return self.df.copy()

    def _build_pseudo_flow(self, frames: np.ndarray, sun_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        motion_frames = torch.from_numpy(frames[-3:].astype(np.float32))
        gray = motion_frames.mean(dim=1)
        gray = F.interpolate(gray.unsqueeze(1), size=self.flow_size, mode="bilinear", align_corners=False).squeeze(1)

        flows = []
        masks = []
        for idx in range(2):
            flow, conf = self._estimate_pair_flow(gray[idx], gray[idx + 1])
            mask = self._build_flow_mask(confidence=conf, sun_xy=sun_xy)
            flows.append(flow)
            masks.append(mask)
        return np.stack(flows, axis=0), np.stack(masks, axis=0)

    def _estimate_pair_flow(self, prev: torch.Tensor, curr: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        best_cost = None
        best_dx = torch.zeros_like(prev)
        best_dy = torch.zeros_like(prev)
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                shifted = torch.zeros_like(curr)
                src_y0 = max(0, -dy)
                src_y1 = min(curr.shape[0], curr.shape[0] - dy) if dy >= 0 else curr.shape[0]
                dst_y0 = max(0, dy)
                dst_y1 = dst_y0 + (src_y1 - src_y0)
                src_x0 = max(0, -dx)
                src_x1 = min(curr.shape[1], curr.shape[1] - dx) if dx >= 0 else curr.shape[1]
                dst_x0 = max(0, dx)
                dst_x1 = dst_x0 + (src_x1 - src_x0)
                if src_y1 > src_y0 and src_x1 > src_x0:
                    shifted[dst_y0:dst_y1, dst_x0:dst_x1] = curr[src_y0:src_y1, src_x0:src_x1]
                cost = F.avg_pool2d((prev - shifted).pow(2)[None, None], kernel_size=3, stride=1, padding=1)[0, 0]
                if best_cost is None:
                    best_cost = cost
                    best_dx.fill_(dx)
                    best_dy.fill_(dy)
                else:
                    update = cost < best_cost
                    best_cost = torch.where(update, cost, best_cost)
                    best_dx = torch.where(update, torch.full_like(best_dx, float(dx)), best_dx)
                    best_dy = torch.where(update, torch.full_like(best_dy, float(dy)), best_dy)
        conf = torch.exp(-best_cost / 0.05).clamp(0.0, 1.0)
        flow = torch.stack([best_dx, best_dy], dim=0)
        return flow.numpy().astype(np.float32), conf.numpy().astype(np.float32)

    def _build_flow_mask(self, confidence: np.ndarray, sun_xy: np.ndarray) -> np.ndarray:
        h, w = self.flow_size
        mask = np.ones((1, h, w), dtype=np.float32)
        if self.mask is not None:
            mask_img = torch.from_numpy(self.mask.astype(np.float32))
            mask = (F.interpolate(mask_img.unsqueeze(0), size=self.flow_size, mode="nearest")[0].numpy() >= 0.5).astype(np.float32)
        scale_x = w / float(self.image_size[1])
        scale_y = h / float(self.image_size[0])
        sx = float(sun_xy[0]) * scale_x
        sy = float(sun_xy[1]) * scale_y
        yy, xx = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
        sun_valid = (((xx - sx) ** 2 + (yy - sy) ** 2) >= 3.0 ** 2).astype(np.float32)
        mask = mask * sun_valid[None, ...] * (confidence[None, ...] >= 0.25).astype(np.float32)
        mask[:, 0, :] = 0.0
        mask[:, -1, :] = 0.0
        mask[:, :, 0] = 0.0
        mask[:, :, -1] = 0.0
        return mask.astype(np.float32)
