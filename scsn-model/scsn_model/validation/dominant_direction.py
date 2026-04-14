from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .motion_teacher import load_camera_index, nearest_image_path, non_saturated_mask
from ..data.dataset import load_mask, load_rgb_image


@dataclass
class DominantDirectionResult:
    vectors: np.ndarray
    peak_strengths: np.ndarray
    cloud_fractions: np.ndarray
    roi_boxes: np.ndarray
    dominant_vector: np.ndarray
    direction_label: int
    direction_name: str
    direction_confidence: float
    direction_consistency: float
    mean_cloud_fraction: float
    mean_peak_strength: float


DIRECTION_NAMES = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]


def build_rbr_cloud_mask(image_chw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rgb = np.transpose(image_chw, (1, 2, 0))
    r = rgb[..., 0]
    b = rgb[..., 2]
    brightness = rgb.mean(axis=-1)
    rbr = r / np.clip(b, 1e-6, None)
    cloud = ((rbr > 0.9) & (brightness < 0.95)).astype(np.float32)
    return rbr.astype(np.float32), cloud


def build_masked_rbr(image_chw: np.ndarray, sky_mask_hw: np.ndarray | None, sun_xy: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    rbr, cloud = build_rbr_cloud_mask(image_chw)
    sky = sky_mask_hw if sky_mask_hw is not None else np.ones_like(cloud, dtype=np.float32)
    non_sat = non_saturated_mask(image_chw, sun_xy=sun_xy)
    valid = sky * cloud * non_sat
    masked_rbr = rbr * valid
    return masked_rbr.astype(np.float32), {
        "rbr": rbr,
        "cloud": cloud,
        "sky": sky.astype(np.float32),
        "non_sat": non_sat.astype(np.float32),
        "valid": valid.astype(np.float32),
    }


def resize_map(map_hw: np.ndarray, size: tuple[int, int], mode: str) -> np.ndarray:
    return F.interpolate(
        torch.from_numpy(map_hw).unsqueeze(0).unsqueeze(0),
        size=size,
        mode=mode,
        align_corners=False if mode in {"bilinear", "bicubic"} else None,
    )[0, 0].numpy()


def _cloud_roi(mask_a: np.ndarray, mask_b: np.ndarray, min_size: int = 24, margin: int = 6) -> tuple[int, int, int, int]:
    union = (mask_a > 0.5) | (mask_b > 0.5)
    ys, xs = np.where(union)
    h, w = union.shape
    if len(xs) == 0 or len(ys) == 0:
        return 0, h, 0, w
    y0 = max(int(ys.min()) - margin, 0)
    y1 = min(int(ys.max()) + margin + 1, h)
    x0 = max(int(xs.min()) - margin, 0)
    x1 = min(int(xs.max()) + margin + 1, w)
    if (y1 - y0) < min_size:
        extra = (min_size - (y1 - y0)) // 2 + 1
        y0 = max(y0 - extra, 0)
        y1 = min(y1 + extra, h)
    if (x1 - x0) < min_size:
        extra = (min_size - (x1 - x0)) // 2 + 1
        x0 = max(x0 - extra, 0)
        x1 = min(x1 + extra, w)
    return y0, y1, x0, x1


def phase_correlation_shift(prev_hw: np.ndarray, curr_hw: np.ndarray) -> tuple[np.ndarray, float]:
    a = prev_hw.astype(np.float32)
    b = curr_hw.astype(np.float32)
    a = a - a.mean()
    b = b - b.mean()
    fa = np.fft.fft2(a)
    fb = np.fft.fft2(b)
    cps = fa * np.conj(fb)
    cps /= np.clip(np.abs(cps), 1e-6, None)
    corr = np.fft.ifft2(cps)
    corr = np.abs(corr)
    peak_idx = np.unravel_index(np.argmax(corr), corr.shape)
    peak = float(corr[peak_idx] / max(corr.sum(), 1e-6))

    dy, dx = peak_idx
    h, w = corr.shape
    if dy > h // 2:
        dy -= h
    if dx > w // 2:
        dx -= w
    return np.asarray([float(dx), float(dy)], dtype=np.float32), peak


def aggregate_dominant_direction(vectors: np.ndarray, cloud_fractions: np.ndarray, peak_strengths: np.ndarray, roi_boxes: np.ndarray) -> DominantDirectionResult:
    magnitudes = np.linalg.norm(vectors, axis=1)
    nonzero = magnitudes > 1e-6
    if np.any(nonzero):
        unit = vectors[nonzero] / magnitudes[nonzero, None]
        dominant = np.median(unit, axis=0)
    else:
        dominant = np.asarray([0.0, 0.0], dtype=np.float32)
    dom_norm = float(np.linalg.norm(dominant))
    if dom_norm > 1e-6:
        dominant = dominant / dom_norm

    if np.any(nonzero) and dom_norm > 1e-6:
        consistency = float(np.mean(np.clip(unit @ dominant, -1.0, 1.0)))
    else:
        consistency = 0.0

    angle = math.atan2(float(dominant[1]), float(dominant[0])) if dom_norm > 1e-6 else 0.0
    angle = (angle + 2 * math.pi) % (2 * math.pi)
    label = int(np.floor((angle + math.pi / 8) / (math.pi / 4))) % 8
    mean_cloud = float(cloud_fractions.mean()) if len(cloud_fractions) else 0.0
    mean_peak = float(peak_strengths.mean()) if len(peak_strengths) else 0.0
    confidence = float(np.clip(consistency * mean_cloud * min(1.0, mean_peak * 64.0), 0.0, 1.0))

    return DominantDirectionResult(
        vectors=vectors.astype(np.float32),
        peak_strengths=peak_strengths.astype(np.float32),
        cloud_fractions=cloud_fractions.astype(np.float32),
        roi_boxes=roi_boxes.astype(np.int32),
        dominant_vector=dominant.astype(np.float32),
        direction_label=label,
        direction_name=DIRECTION_NAMES[label],
        direction_confidence=confidence,
        direction_consistency=consistency,
        mean_cloud_fraction=mean_cloud,
        mean_peak_strength=mean_peak,
    )


def validate_sample_dominant_direction(
    row: pd.Series,
    camera_ts_ns: np.ndarray,
    camera_paths: np.ndarray,
    image_size: tuple[int, int],
    sky_mask_hw: np.ndarray | None,
    tolerance_sec: int = 75,
    flow_size: tuple[int, int] = (128, 128),
) -> tuple[dict, DominantDirectionResult, dict]:
    anchor_ts = pd.to_datetime(row["ts_anchor"])
    offsets = [-4, -3, -2, -1, 0]
    frame_paths: list[str] = []
    frames: list[np.ndarray] = []
    for off in offsets:
        ts = anchor_ts + pd.Timedelta(minutes=off)
        path = nearest_image_path(ts, camera_ts_ns, camera_paths, tolerance_sec)
        if path is None:
            raise RuntimeError(f"Missing 1-minute frame at offset {off} for {anchor_ts}")
        frame_paths.append(path)
        frames.append(load_rgb_image(path, image_size))

    sun_xy = np.asarray([float(row["sun_x_px"]), float(row["sun_y_px"])], dtype=np.float32)
    masked_rbr_list = []
    mask_list = []
    resized_mask_list = []
    for frame in frames:
        masked_rbr, masks = build_masked_rbr(frame, sky_mask_hw=sky_mask_hw, sun_xy=sun_xy)
        masked_rbr_rs = resize_map(masked_rbr, flow_size, mode="bilinear")
        valid_rs = resize_map(masks["valid"], flow_size, mode="nearest")
        cloud_rs = resize_map(masks["cloud"], flow_size, mode="nearest")
        masked_rbr_list.append(masked_rbr_rs * valid_rs)
        mask_list.append(masks)
        resized_mask_list.append({
            "valid": valid_rs.astype(np.float32),
            "cloud": cloud_rs.astype(np.float32),
        })

    vectors = []
    peaks = []
    cloud_fractions = []
    roi_boxes = []
    for idx in range(4):
        roi = _cloud_roi(resized_mask_list[idx]["cloud"], resized_mask_list[idx + 1]["cloud"])
        y0, y1, x0, x1 = roi
        shift, peak = phase_correlation_shift(
            masked_rbr_list[idx][y0:y1, x0:x1],
            masked_rbr_list[idx + 1][y0:y1, x0:x1],
        )
        vectors.append(shift)
        peaks.append(peak)
        cloud_fractions.append(float(mask_list[idx + 1]["cloud"].mean()))
        roi_boxes.append(np.asarray([y0, y1, x0, x1], dtype=np.int32))

    result = aggregate_dominant_direction(
        vectors=np.stack(vectors, axis=0),
        cloud_fractions=np.asarray(cloud_fractions, dtype=np.float32),
        peak_strengths=np.asarray(peaks, dtype=np.float32),
        roi_boxes=np.stack(roi_boxes, axis=0),
    )
    meta = {
        "ts_anchor": str(row["ts_anchor"]),
        "ts_target": str(row["ts_target"]),
        "target_pv_w": float(row["target_pv_w"]),
        "sun_x_px": float(sun_xy[0]),
        "sun_y_px": float(sun_xy[1]),
        "frame_paths": frame_paths,
    }
    payload = {
        "frames": frames,
        "masks": mask_list,
        "resized_masks": resized_mask_list,
        "masked_rbr": masked_rbr_list,
    }
    return meta, result, payload


def save_dominant_direction_figure(out_path: str | Path, meta: dict, result: DominantDirectionResult, payload: dict) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 3, figsize=(16, 14))
    frame_titles = ["t-4", "t-3", "t-2", "t-1", "t"]
    for i in range(5):
        r, c = divmod(i, 3)
        axes[r, c].imshow(np.transpose(payload["frames"][i], (1, 2, 0)))
        if i < 4:
            y0, y1, x0, x1 = result.roi_boxes[i]
            axes[r, c].add_patch(plt.Rectangle((x0 * payload["frames"][i].shape[2] / payload["masked_rbr"][i].shape[1],
                                                y0 * payload["frames"][i].shape[1] / payload["masked_rbr"][i].shape[0]),
                                               (x1 - x0) * payload["frames"][i].shape[2] / payload["masked_rbr"][i].shape[1],
                                               (y1 - y0) * payload["frames"][i].shape[1] / payload["masked_rbr"][i].shape[0],
                                               fill=False, edgecolor="yellow", linewidth=2))
        axes[r, c].set_title(frame_titles[i])
        axes[r, c].axis("off")

    axes[1, 2].imshow(payload["masks"][-1]["valid"], cmap="gray")
    axes[1, 2].set_title("Valid Mask (t)")
    axes[1, 2].axis("off")

    axes[2, 0].imshow(payload["masked_rbr"][-1], cmap="viridis")
    axes[2, 0].set_title("Masked RBR (t)")
    axes[2, 0].axis("off")

    vecs = result.vectors
    origin_x = np.arange(4)
    origin_y = np.zeros(4)
    axes[2, 1].axhline(0.0, color="lightgray", linewidth=1)
    axes[2, 1].quiver(origin_x, origin_y, vecs[:, 0], -vecs[:, 1], angles="xy", scale_units="xy", scale=1, color="tab:blue")
    axes[2, 1].quiver([1.5], [0.0], [result.dominant_vector[0]], [-result.dominant_vector[1]], angles="xy", scale_units="xy", scale=1, color="red")
    axes[2, 1].set_title("4 pair shifts + dominant")
    axes[2, 1].set_xlim(-0.5, 3.5)
    axes[2, 1].set_ylim(-4, 4)
    axes[2, 1].grid(True, alpha=0.3)

    axes[2, 2].axis("off")
    txt = (
        f"label={result.direction_label} ({result.direction_name})\n"
        f"confidence={result.direction_confidence:.3f}\n"
        f"consistency={result.direction_consistency:.3f}\n"
        f"mean_cloud={result.mean_cloud_fraction:.3f}\n"
        f"mean_peak={result.mean_peak_strength:.5f}\n"
        f"vectors={np.array2string(result.vectors, precision=2)}"
    )
    axes[2, 2].text(0.0, 1.0, txt, va="top", family="monospace")

    fig.suptitle(f"{meta['ts_target']} | dominant {result.direction_name} | conf={result.direction_confidence:.3f}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_dominant_direction_json(path: str | Path, meta: dict, result: DominantDirectionResult) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": meta,
        "vectors": result.vectors.tolist(),
        "peak_strengths": result.peak_strengths.tolist(),
        "cloud_fractions": result.cloud_fractions.tolist(),
        "roi_boxes": result.roi_boxes.tolist(),
        "dominant_vector": result.dominant_vector.tolist(),
        "direction_label": int(result.direction_label),
        "direction_name": result.direction_name,
        "direction_confidence": float(result.direction_confidence),
        "direction_consistency": float(result.direction_consistency),
        "mean_cloud_fraction": float(result.mean_cloud_fraction),
        "mean_peak_strength": float(result.mean_peak_strength),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
