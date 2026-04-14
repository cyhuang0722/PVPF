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
        camera_index_csv: str | Path | None = None,
        image_match_tolerance_sec: int = 75,
        motion_teacher_pairs_min: list[list[int]] | None = None,
        patch_grid_size: int = 8,
        teacher_flow_resolution: int = 64,
        teacher_max_displacement_px: int = 2,
        teacher_conf_threshold: float = 0.25,
        teacher_min_patch_vectors: int = 6,
        teacher_min_magnitude: float = 0.15,
    ) -> None:
        self.df = pd.read_csv(csv_path)
        self.df["ts_anchor"] = pd.to_datetime(self.df["ts_anchor"])
        self.df["ts_target"] = pd.to_datetime(self.df["ts_target"])
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        self.image_size = image_size
        self.mask = load_mask(sky_mask_path, image_size) if sky_mask_path and Path(sky_mask_path).exists() else None
        self.peak_power_w = float(peak_power_w) if peak_power_w is not None else 1.0
        self.image_match_tolerance_sec = int(image_match_tolerance_sec)
        self.motion_teacher_pairs_min = motion_teacher_pairs_min or [[-5, -4], [-3, -2], [-1, 0]]
        self.patch_grid_size = int(patch_grid_size)
        self.teacher_flow_resolution = int(teacher_flow_resolution)
        self.teacher_flow_size = (self.teacher_flow_resolution, self.teacher_flow_resolution)
        self.teacher_max_displacement_px = int(teacher_max_displacement_px)
        self.teacher_conf_threshold = float(teacher_conf_threshold)
        self.teacher_min_patch_vectors = int(teacher_min_patch_vectors)
        self.teacher_min_magnitude = float(teacher_min_magnitude)
        self.camera_index_csv = Path(camera_index_csv) if camera_index_csv else None
        self.camera_ts_ns = None
        self.camera_paths = None
        if self.camera_index_csv is not None and self.camera_index_csv.exists():
            camera_df = pd.read_csv(self.camera_index_csv)
            camera_df["timestamp"] = pd.to_datetime(camera_df["timestamp"], errors="coerce")
            camera_df = camera_df.dropna(subset=["timestamp", "file_path"]).sort_values("timestamp").reset_index(drop=True)
            self.camera_ts_ns = camera_df["timestamp"].astype("int64").to_numpy()
            self.camera_paths = camera_df["file_path"].astype(str).to_numpy()
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
        patch_motion_teacher, patch_motion_mask = self._build_patch_motion_teacher(
            anchor_ts=row["ts_anchor"],
            fallback_frames=frames[-3:],
            sun_xy=np.asarray([float(row["sun_x_px"]), float(row["sun_y_px"])], dtype=np.float32),
        )

        return {
            "images": torch.from_numpy(frames.astype(np.float32)),
            "pv_history": torch.tensor(np.asarray(past_pv, dtype=np.float32) / self.peak_power_w),
            "solar_vec": torch.tensor(np.asarray(solar_vec, dtype=np.float32)),
            "sun_angles": torch.tensor(sun_angles, dtype=torch.float32),
            "target": torch.tensor(float(row["target_value"]), dtype=torch.float32),
            "target_pv_w": torch.tensor(float(row["target_pv_w"]), dtype=torch.float32),
            "target_clear_sky_w": torch.tensor(float(row["target_clear_sky_w"]), dtype=torch.float32),
            "sun_xy": torch.tensor([float(row["sun_x_px"]), float(row["sun_y_px"])], dtype=torch.float32),
            "patch_motion_teacher": torch.from_numpy(patch_motion_teacher),
            "patch_motion_mask": torch.from_numpy(patch_motion_mask),
            "meta_index": torch.tensor(index, dtype=torch.long),
        }

    def dataframe(self) -> pd.DataFrame:
        return self.df.copy()

    def _build_patch_motion_teacher(
        self,
        anchor_ts: pd.Timestamp,
        fallback_frames: np.ndarray,
        sun_xy: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        teacher_pairs = self._load_teacher_pairs(anchor_ts=anchor_ts, fallback_frames=fallback_frames)
        flows = []
        confidences = []
        for prev_frame, curr_frame in teacher_pairs:
            prev = self._prepare_teacher_frame(prev_frame)
            curr = self._prepare_teacher_frame(curr_frame)
            flow, conf = self._estimate_pair_flow(prev, curr)
            flows.append(flow)
            confidences.append(conf)
        return self._aggregate_patch_teacher(
            dense_flows=np.stack(flows, axis=0),
            dense_conf=np.stack(confidences, axis=0),
            sun_xy=sun_xy,
        )

    def _load_teacher_pairs(self, anchor_ts: pd.Timestamp, fallback_frames: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        pairs: list[tuple[np.ndarray, np.ndarray]] = []
        if self.camera_ts_ns is not None and self.camera_paths is not None:
            for prev_offset, curr_offset in self.motion_teacher_pairs_min:
                prev_ts = anchor_ts + pd.Timedelta(minutes=int(prev_offset))
                curr_ts = anchor_ts + pd.Timedelta(minutes=int(curr_offset))
                prev_path = self._get_nearest_path(prev_ts)
                curr_path = self._get_nearest_path(curr_ts)
                if prev_path is None or curr_path is None:
                    continue
                pairs.append((load_rgb_image(prev_path, self.image_size), load_rgb_image(curr_path, self.image_size)))
        if not pairs:
            pairs = [(fallback_frames[0], fallback_frames[1]), (fallback_frames[1], fallback_frames[2])]
        return pairs

    def _get_nearest_path(self, desired_ts: pd.Timestamp) -> str | None:
        desired_ns = desired_ts.value
        pos = int(np.searchsorted(self.camera_ts_ns, desired_ns))
        candidate_idx: list[int] = []
        if 0 <= pos < len(self.camera_ts_ns):
            candidate_idx.append(pos)
        if pos - 1 >= 0:
            candidate_idx.append(pos - 1)
        if not candidate_idx:
            return None
        best_i = min(candidate_idx, key=lambda i: abs(int(self.camera_ts_ns[i]) - desired_ns))
        diff_sec = abs(int(self.camera_ts_ns[best_i]) - desired_ns) / 1e9
        if diff_sec > self.image_match_tolerance_sec:
            return None
        return str(self.camera_paths[best_i])

    def _prepare_teacher_frame(self, frame: np.ndarray) -> torch.Tensor:
        gray = torch.from_numpy(frame.astype(np.float32)).mean(dim=0, keepdim=True)
        return F.interpolate(gray.unsqueeze(0), size=self.teacher_flow_size, mode="bilinear", align_corners=False)[0, 0]

    def _estimate_pair_flow(self, prev: torch.Tensor, curr: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        best_cost = None
        best_dx = torch.zeros_like(prev)
        best_dy = torch.zeros_like(prev)
        for dy in range(-self.teacher_max_displacement_px, self.teacher_max_displacement_px + 1):
            for dx in range(-self.teacher_max_displacement_px, self.teacher_max_displacement_px + 1):
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

    def _aggregate_patch_teacher(
        self,
        dense_flows: np.ndarray,
        dense_conf: np.ndarray,
        sun_xy: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        h, w = self.teacher_flow_size
        dense_mask = self._build_teacher_mask(sun_xy)
        patch_h = h // self.patch_grid_size
        patch_w = w // self.patch_grid_size
        patch_vectors = np.zeros((self.patch_grid_size * self.patch_grid_size, 2), dtype=np.float32)
        patch_mask = np.zeros((self.patch_grid_size * self.patch_grid_size, 1), dtype=np.float32)

        patch_idx = 0
        for gy in range(self.patch_grid_size):
            for gx in range(self.patch_grid_size):
                y0 = gy * patch_h
                y1 = h if gy == self.patch_grid_size - 1 else (gy + 1) * patch_h
                x0 = gx * patch_w
                x1 = w if gx == self.patch_grid_size - 1 else (gx + 1) * patch_w

                local_flow = dense_flows[:, :, y0:y1, x0:x1].transpose(0, 2, 3, 1).reshape(-1, 2)
                local_weight = (dense_conf[:, y0:y1, x0:x1] * dense_mask[y0:y1, x0:x1][None, ...]).reshape(-1)
                valid = local_weight >= self.teacher_conf_threshold
                if int(valid.sum()) < self.teacher_min_patch_vectors:
                    patch_idx += 1
                    continue
                vecs = local_flow[valid]
                weights = local_weight[valid]
                mean_vec = (vecs * weights[:, None]).sum(axis=0) / max(weights.sum(), 1e-6)
                magnitude = float(np.linalg.norm(mean_vec))
                if magnitude < self.teacher_min_magnitude:
                    patch_idx += 1
                    continue
                patch_vectors[patch_idx] = (mean_vec / max(magnitude, 1e-6)).astype(np.float32)
                patch_mask[patch_idx, 0] = 1.0
                patch_idx += 1

        return patch_vectors, patch_mask

    def _build_teacher_mask(self, sun_xy: np.ndarray) -> np.ndarray:
        h, w = self.teacher_flow_size
        mask = np.ones((h, w), dtype=np.float32)
        if self.mask is not None:
            mask_img = torch.from_numpy(self.mask.astype(np.float32))
            mask = (F.interpolate(mask_img.unsqueeze(0), size=self.teacher_flow_size, mode="nearest")[0, 0].numpy() >= 0.5).astype(np.float32)
        scale_x = w / float(self.image_size[1])
        scale_y = h / float(self.image_size[0])
        sx = float(sun_xy[0]) * scale_x
        sy = float(sun_xy[1]) * scale_y
        yy, xx = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
        sun_valid = (((xx - sx) ** 2 + (yy - sy) ** 2) >= 4.5 ** 2).astype(np.float32)
        mask = mask * sun_valid
        mask[0, :] = 0.0
        mask[-1, :] = 0.0
        mask[:, 0] = 0.0
        mask[:, -1] = 0.0
        return mask.astype(np.float32)
