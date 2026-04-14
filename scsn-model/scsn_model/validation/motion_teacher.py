from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from ..data.dataset import load_mask, load_rgb_image


@dataclass
class PatchTeacherResult:
    teacher_direction: np.ndarray
    teacher_valid_mask: np.ndarray
    teacher_confidence: np.ndarray
    sun_weight: np.ndarray
    patch_cloud_fraction: np.ndarray
    patch_coherence: np.ndarray
    toward_sun_score: float
    away_from_sun_score: float
    valid_patch_fraction: float
    valid_sun_patch_fraction: float


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_camera_index(camera_index_csv: str | Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(camera_index_csv)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp", "file_path"]).sort_values("timestamp").reset_index(drop=True)
    return df["timestamp"].astype("int64").to_numpy(), df["file_path"].astype(str).to_numpy()


def nearest_image_path(desired_ts: pd.Timestamp, camera_ts_ns: np.ndarray, camera_paths: np.ndarray, tolerance_sec: int) -> str | None:
    if getattr(desired_ts, "tzinfo", None) is not None:
        desired_ts = desired_ts.tz_localize(None)
    desired_ns = desired_ts.value
    pos = int(np.searchsorted(camera_ts_ns, desired_ns))
    candidate_idx: list[int] = []
    if 0 <= pos < len(camera_ts_ns):
        candidate_idx.append(pos)
    if pos - 1 >= 0:
        candidate_idx.append(pos - 1)
    if not candidate_idx:
        return None
    best_i = min(candidate_idx, key=lambda i: abs(int(camera_ts_ns[i]) - desired_ns))
    diff_sec = abs(int(camera_ts_ns[best_i]) - desired_ns) / 1e9
    if diff_sec > tolerance_sec:
        return None
    return str(camera_paths[best_i])


def _local_variance_mask(gray_hw: np.ndarray, threshold: float = 0.0025, kernel_size: int = 9) -> np.ndarray:
    gray = torch.from_numpy(gray_hw.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    pad = kernel_size // 2
    mean = F.avg_pool2d(gray, kernel_size=kernel_size, stride=1, padding=pad)
    mean_sq = F.avg_pool2d(gray * gray, kernel_size=kernel_size, stride=1, padding=pad)
    var = (mean_sq - mean * mean)[0, 0].numpy()
    return (var > threshold).astype(np.float32)


def simple_cloud_mask(image_chw: np.ndarray, texture_threshold: float = 0.0025) -> np.ndarray:
    rgb = np.transpose(image_chw, (1, 2, 0))
    r = rgb[..., 0]
    b = rgb[..., 2]
    brightness = rgb.mean(axis=-1)
    rb_ratio = r / np.clip(b, 1e-6, None)
    cloud = (rb_ratio > 0.9) & (brightness < 0.95)
    return cloud.astype(np.float32)


def texture_mask(image_chw: np.ndarray, threshold: float = 0.04) -> np.ndarray:
    gray = torch.from_numpy(image_chw.astype(np.float32)).mean(dim=0, keepdim=True).unsqueeze(0)
    sobel_x = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], dtype=torch.float32)
    sobel_y = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], dtype=torch.float32)
    grad_x = F.conv2d(gray, sobel_x, padding=1)
    grad_y = F.conv2d(gray, sobel_y, padding=1)
    grad = torch.sqrt(grad_x.pow(2) + grad_y.pow(2))[0, 0].numpy()
    grad = grad / max(float(np.quantile(grad, 0.98)), 1e-6)
    return (grad >= threshold).astype(np.float32)


def non_saturated_mask(image_chw: np.ndarray, sun_xy: np.ndarray, sun_disk_radius_px: float = 16.0, halo_radius_px: float = 28.0) -> np.ndarray:
    rgb = np.transpose(image_chw, (1, 2, 0))
    intensity = rgb.max(axis=-1)
    h, w = intensity.shape
    yy, xx = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
    sx, sy = float(sun_xy[0]), float(sun_xy[1])
    dist = np.sqrt((xx - sx) ** 2 + (yy - sy) ** 2)
    mask = np.ones((h, w), dtype=np.float32)
    mask[dist <= sun_disk_radius_px] = 0.0
    halo_bad = (dist <= halo_radius_px) & (intensity >= 0.92)
    mask[halo_bad] = 0.0
    return mask


def estimate_dense_flow(prev_chw: np.ndarray, curr_chw: np.ndarray, flow_size: tuple[int, int], max_displacement_px: int = 2) -> tuple[np.ndarray, np.ndarray]:
    prev = torch.from_numpy(prev_chw.astype(np.float32)).mean(dim=0, keepdim=True).unsqueeze(0)
    curr = torch.from_numpy(curr_chw.astype(np.float32)).mean(dim=0, keepdim=True).unsqueeze(0)
    prev = F.interpolate(prev, size=flow_size, mode="bilinear", align_corners=False)[0, 0]
    curr = F.interpolate(curr, size=flow_size, mode="bilinear", align_corners=False)[0, 0]

    best_cost = None
    best_dx = torch.zeros_like(prev)
    best_dy = torch.zeros_like(prev)
    for dy in range(-max_displacement_px, max_displacement_px + 1):
        for dx in range(-max_displacement_px, max_displacement_px + 1):
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
    confidence = torch.exp(-best_cost / 0.05).clamp(0.0, 1.0).numpy()
    flow = torch.stack([best_dx, best_dy], dim=0).numpy().astype(np.float32)
    return flow, confidence.astype(np.float32)


def build_valid_motion_mask(
    image_chw: np.ndarray,
    sky_mask_hw: np.ndarray | None,
    sun_xy: np.ndarray,
    texture_threshold: float = 0.0025,
) -> dict[str, np.ndarray]:
    sky = sky_mask_hw if sky_mask_hw is not None else np.ones(image_chw.shape[1:], dtype=np.float32)
    cloud = simple_cloud_mask(image_chw, texture_threshold=texture_threshold)
    non_sat = non_saturated_mask(image_chw, sun_xy=sun_xy)
    texture = _local_variance_mask(np.transpose(image_chw, (1, 2, 0)).mean(axis=-1), threshold=texture_threshold)
    valid = sky * cloud * non_sat
    return {
        "sky": sky.astype(np.float32),
        "cloud": cloud.astype(np.float32),
        "non_sat": non_sat.astype(np.float32),
        "texture": texture.astype(np.float32),
        "valid": valid.astype(np.float32),
    }


def aggregate_patch_teacher(
    flow: np.ndarray,
    valid_mask: np.ndarray,
    confidence: np.ndarray,
    sun_xy: np.ndarray,
    image_size: tuple[int, int],
    patch_grid_size: int = 8,
    min_valid_pixels: int = 8,
    min_magnitude: float = 0.15,
    min_coherence: float = 0.45,
) -> PatchTeacherResult:
    h, w = valid_mask.shape
    ph = h // patch_grid_size
    pw = w // patch_grid_size
    teacher_direction = np.zeros((patch_grid_size * patch_grid_size, 2), dtype=np.float32)
    teacher_valid = np.zeros((patch_grid_size * patch_grid_size, 1), dtype=np.float32)
    teacher_confidence = np.zeros((patch_grid_size * patch_grid_size, 1), dtype=np.float32)
    sun_weight = np.zeros((patch_grid_size * patch_grid_size, 1), dtype=np.float32)
    patch_cloud_fraction = np.zeros((patch_grid_size * patch_grid_size, 1), dtype=np.float32)
    patch_coherence = np.zeros((patch_grid_size * patch_grid_size, 1), dtype=np.float32)

    scale_x = w / float(image_size[1])
    scale_y = h / float(image_size[0])
    sx, sy = float(sun_xy[0]) * scale_x, float(sun_xy[1]) * scale_y

    global_vecs = flow.transpose(1, 2, 0).reshape(-1, 2)
    global_valid = valid_mask.reshape(-1) > 0.5
    if global_valid.any():
        global_mean = global_vecs[global_valid].mean(axis=0)
    else:
        global_mean = np.array([0.0, -1.0], dtype=np.float32)
    global_norm = np.linalg.norm(global_mean)
    if global_norm < 1e-6:
        global_mean = np.array([0.0, -1.0], dtype=np.float32)
    else:
        global_mean = global_mean / global_norm
    upwind = -global_mean

    toward_sun_score = 0.0
    away_from_sun_score = 0.0
    patch_idx = 0
    for gy in range(patch_grid_size):
        for gx in range(patch_grid_size):
            y0 = gy * ph
            y1 = h if gy == patch_grid_size - 1 else (gy + 1) * ph
            x0 = gx * pw
            x1 = w if gx == patch_grid_size - 1 else (gx + 1) * pw

            patch_mask = valid_mask[y0:y1, x0:x1]
            patch_conf = confidence[y0:y1, x0:x1]
            patch_vecs = flow[:, y0:y1, x0:x1].transpose(1, 2, 0).reshape(-1, 2)
            valid = patch_mask.reshape(-1) > 0.5
            patch_cloud_fraction[patch_idx, 0] = float(patch_mask.mean())

            center_x = (x0 + x1) * 0.5
            center_y = (y0 + y1) * 0.5
            rel = np.array([sx - center_x, sy - center_y], dtype=np.float32)
            rel_norm = np.linalg.norm(rel)
            rel_unit = rel / max(rel_norm, 1e-6)
            spot = math.exp(-(rel_norm ** 2) / (2.0 * (w / 5.0) ** 2))
            along = float(rel_unit @ upwind)
            corridor = max(along, 0.0)
            sun_weight[patch_idx, 0] = min(1.0, 0.7 * spot + 0.6 * corridor)

            if int(valid.sum()) < min_valid_pixels:
                patch_idx += 1
                continue
            vecs = patch_vecs[valid]
            weights = patch_conf.reshape(-1)[valid]
            mean_vec = (vecs * weights[:, None]).sum(axis=0) / max(weights.sum(), 1e-6)
            mag = float(np.linalg.norm(mean_vec))
            if mag < min_magnitude:
                patch_idx += 1
                continue
            unit_vecs = vecs / np.clip(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-6, None)
            mean_unit = (unit_vecs * weights[:, None]).sum(axis=0) / max(weights.sum(), 1e-6)
            coherence = float(np.linalg.norm(mean_unit))
            patch_coherence[patch_idx, 0] = coherence
            if coherence < min_coherence:
                patch_idx += 1
                continue

            dominant = mean_vec / mag
            teacher_direction[patch_idx] = dominant.astype(np.float32)
            teacher_valid[patch_idx, 0] = 1.0
            teacher_confidence[patch_idx, 0] = float(min(1.0, patch_mask.mean() * coherence * mag))
            dot_to_sun = float(dominant @ rel_unit)
            if dot_to_sun > 0:
                toward_sun_score += dot_to_sun * teacher_confidence[patch_idx, 0] * sun_weight[patch_idx, 0]
            else:
                away_from_sun_score += (-dot_to_sun) * teacher_confidence[patch_idx, 0] * sun_weight[patch_idx, 0]
            patch_idx += 1

    valid_patch_fraction = float(teacher_valid.mean())
    sun_valid = (teacher_valid.squeeze(-1) > 0.5) & (sun_weight.squeeze(-1) > 0.3)
    valid_sun_patch_fraction = float(sun_valid.mean()) if len(sun_valid) else 0.0
    return PatchTeacherResult(
        teacher_direction=teacher_direction,
        teacher_valid_mask=teacher_valid,
        teacher_confidence=teacher_confidence,
        sun_weight=sun_weight,
        patch_cloud_fraction=patch_cloud_fraction,
        patch_coherence=patch_coherence,
        toward_sun_score=float(toward_sun_score),
        away_from_sun_score=float(away_from_sun_score),
        valid_patch_fraction=valid_patch_fraction,
        valid_sun_patch_fraction=valid_sun_patch_fraction,
    )


def save_validation_figure(
    out_path: str | Path,
    prev_image: np.ndarray,
    curr_image: np.ndarray,
    masks: dict[str, np.ndarray],
    flow: np.ndarray,
    patch_result: PatchTeacherResult,
    sun_xy: np.ndarray,
    title: str,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def to_rgb(img: np.ndarray) -> np.ndarray:
        return np.transpose(img, (1, 2, 0))

    prev_rgb = to_rgb(prev_image)
    curr_rgb = to_rgb(curr_image)
    flow_valid = masks["valid"]
    if flow_valid.shape != flow[0].shape:
        flow_valid = F.interpolate(
            torch.from_numpy(flow_valid).unsqueeze(0).unsqueeze(0),
            size=flow[0].shape,
            mode="nearest",
        )[0, 0].numpy()
    flow_mag = np.sqrt(flow[0] ** 2 + flow[1] ** 2) * flow_valid

    h, w = masks["valid"].shape
    flow_h, flow_w = flow[0].shape
    grid = int(np.sqrt(patch_result.teacher_direction.shape[0]))
    ph = h / grid
    pw = w / grid
    cx, cy = np.meshgrid((np.arange(grid) + 0.5) * pw, (np.arange(grid) + 0.5) * ph)
    teacher = patch_result.teacher_direction.reshape(grid, grid, 2)
    valid = patch_result.teacher_valid_mask.reshape(grid, grid)
    sx = sun_xy[0] * (w / prev_image.shape[2])
    sy = sun_xy[1] * (h / prev_image.shape[1])

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes[0, 0].imshow(prev_rgb)
    axes[0, 0].set_title("1-min Prev")
    axes[0, 0].axis("off")
    axes[0, 1].imshow(curr_rgb)
    axes[0, 1].set_title("1-min Curr")
    axes[0, 1].axis("off")
    axes[0, 2].imshow(masks["valid"], cmap="gray")
    axes[0, 2].set_title("Valid Pixel Mask")
    axes[0, 2].axis("off")
    axes[0, 3].imshow(flow_mag, cmap="magma")
    step = max(flow_h // 16, 1)
    yy, xx = np.mgrid[0:flow_h:step, 0:flow_w:step]
    axes[0, 3].quiver(xx, yy, flow[0][::step, ::step], flow[1][::step, ::step], color="white")
    axes[0, 3].set_title("Dense Flow After Mask")
    axes[0, 3].axis("off")

    axes[1, 0].imshow(valid, cmap="gray")
    axes[1, 0].set_title("Patch Valid Mask")
    axes[1, 0].axis("off")
    axes[1, 1].imshow(curr_rgb)
    axes[1, 1].quiver(cx, cy, teacher[..., 0], teacher[..., 1], color="cyan", scale=18)
    axes[1, 1].set_title("Patch Teacher Directions")
    axes[1, 1].axis("off")
    axes[1, 2].imshow(patch_result.sun_weight.reshape(grid, grid), cmap="viridis")
    axes[1, 2].set_title("Sun Relevance")
    axes[1, 2].axis("off")
    axes[1, 3].imshow(curr_rgb)
    axes[1, 3].quiver(cx, cy, teacher[..., 0], teacher[..., 1], color="lime", scale=18)
    axes[1, 3].scatter([sx], [sy], c="red", s=60)
    axes[1, 3].set_title("Teacher Overlay + Sun")
    axes[1, 3].axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def validate_motion_teacher_for_sample(
    row: pd.Series,
    camera_ts_ns: np.ndarray,
    camera_paths: np.ndarray,
    image_size: tuple[int, int],
    tolerance_sec: int,
    sky_mask_hw: np.ndarray | None,
    teacher_pair_min: tuple[int, int] = (-1, 0),
    flow_size: tuple[int, int] = (64, 64),
    max_displacement_px: int = 2,
    patch_grid_size: int = 8,
) -> tuple[dict, PatchTeacherResult, dict[str, np.ndarray]]:
    anchor_ts = pd.to_datetime(row["ts_anchor"])
    prev_path = nearest_image_path(anchor_ts + pd.Timedelta(minutes=teacher_pair_min[0]), camera_ts_ns, camera_paths, tolerance_sec)
    curr_path = nearest_image_path(anchor_ts + pd.Timedelta(minutes=teacher_pair_min[1]), camera_ts_ns, camera_paths, tolerance_sec)
    if prev_path is None or curr_path is None:
        raise RuntimeError(f"Missing 1-minute pair around {anchor_ts}")

    prev_img = load_rgb_image(prev_path, image_size)
    curr_img = load_rgb_image(curr_path, image_size)
    sun_xy = np.asarray([float(row["sun_x_px"]), float(row["sun_y_px"])], dtype=np.float32)
    masks = build_valid_motion_mask(curr_img, sky_mask_hw=sky_mask_hw, sun_xy=sun_xy)

    flow, confidence = estimate_dense_flow(prev_img, curr_img, flow_size=flow_size, max_displacement_px=max_displacement_px)
    valid_rs = F.interpolate(torch.from_numpy(masks["valid"]).unsqueeze(0).unsqueeze(0), size=flow_size, mode="nearest")[0, 0].numpy()
    flow = flow * valid_rs[None, ...]
    confidence = confidence * valid_rs

    result = aggregate_patch_teacher(
        flow=flow,
        valid_mask=valid_rs,
        confidence=confidence,
        sun_xy=sun_xy,
        image_size=image_size,
        patch_grid_size=patch_grid_size,
    )
    meta = {
        "ts_anchor": str(row["ts_anchor"]),
        "ts_target": str(row["ts_target"]),
        "target_pv_w": float(row["target_pv_w"]),
        "past_pv_w_last": float(json.loads(row["past_pv_w"])[-1]) if isinstance(row["past_pv_w"], str) else float(row["past_pv_w"][-1]),
        "future_delta_w": float(row["target_pv_w"]) - (
            float(json.loads(row["past_pv_w"])[-1]) if isinstance(row["past_pv_w"], str) else float(row["past_pv_w"][-1])
        ),
        "sun_x_px": float(sun_xy[0]),
        "sun_y_px": float(sun_xy[1]),
        "prev_path": prev_path,
        "curr_path": curr_path,
    }
    payload = {
        "prev_image": prev_img,
        "curr_image": curr_img,
        "masks": masks,
        "flow": flow,
        "confidence": confidence,
    }
    return meta, result, payload


def save_teacher_package(path: str | Path, meta: dict, result: PatchTeacherResult) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": meta,
        "teacher_direction": result.teacher_direction.tolist(),
        "teacher_valid_mask": result.teacher_valid_mask.tolist(),
        "teacher_confidence": result.teacher_confidence.tolist(),
        "sun_weight": result.sun_weight.tolist(),
        "patch_cloud_fraction": result.patch_cloud_fraction.tolist(),
        "patch_coherence": result.patch_coherence.tolist(),
        "toward_sun_score": result.toward_sun_score,
        "away_from_sun_score": result.away_from_sun_score,
        "valid_patch_fraction": result.valid_patch_fraction,
        "valid_sun_patch_fraction": result.valid_sun_patch_fraction,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
