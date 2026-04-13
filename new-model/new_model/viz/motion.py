from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_motion_attention_figure(
    image: np.ndarray,
    motion: np.ndarray,
    attention: np.ndarray,
    out_path: str | Path,
    title: str,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    motion_u = motion[0]
    motion_v = motion[1]
    motion_mag = np.sqrt(motion_u ** 2 + motion_v ** 2)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(np.transpose(image, (1, 2, 0)))
    axes[0].set_title("Last Frame")
    axes[0].axis("off")

    axes[1].imshow(motion_mag, cmap="magma")
    step = max(motion_mag.shape[0] // 16, 1)
    yy, xx = np.mgrid[0 : motion_mag.shape[0] : step, 0 : motion_mag.shape[1] : step]
    axes[1].quiver(xx, yy, motion_u[::step, ::step], motion_v[::step, ::step], color="white")
    axes[1].set_title("Motion Field")
    axes[1].axis("off")

    axes[2].imshow(np.transpose(image, (1, 2, 0)))
    axes[2].imshow(attention, cmap="jet", alpha=0.45)
    axes[2].set_title("Sun Attention")
    axes[2].axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_motion_flow_comparison_figure(
    image_current: np.ndarray,
    image_prev_1: np.ndarray,
    image_prev_2: np.ndarray,
    flow_pred_recent: np.ndarray,
    flow_gt_recent: np.ndarray,
    flow_mask_recent: np.ndarray,
    sun_prior: np.ndarray,
    out_path: str | Path,
    title: str,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _to_rgb(img: np.ndarray) -> np.ndarray:
        if img.shape[0] == 1:
            return np.repeat(np.transpose(img, (1, 2, 0)), 3, axis=2)
        return np.transpose(img, (1, 2, 0))

    def _resize_overlay(field: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
        h, w = target_hw
        if field.shape == (h, w):
            return field
        scale_y = max(h // field.shape[0], 1)
        scale_x = max(w // field.shape[1], 1)
        return np.kron(field, np.ones((scale_y, scale_x), dtype=np.float32))[:h, :w]

    cur_rgb = _to_rgb(image_current)
    prev1_rgb = _to_rgb(image_prev_1)
    prev2_rgb = _to_rgb(image_prev_2)
    img_h, img_w = cur_rgb.shape[:2]

    pred_u, pred_v = flow_pred_recent[0], flow_pred_recent[1]
    gt_u, gt_v = flow_gt_recent[0], flow_gt_recent[1]
    pred_mag = np.sqrt(pred_u ** 2 + pred_v ** 2)
    gt_mag = np.sqrt(gt_u ** 2 + gt_v ** 2)
    mask = flow_mask_recent[0]
    sun_prior_vis = _resize_overlay(sun_prior, (img_h, img_w))

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    axes[0, 0].imshow(prev2_rgb)
    axes[0, 0].set_title("t-4 Frame")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(prev1_rgb)
    axes[0, 1].set_title("t-2 Frame")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(cur_rgb)
    axes[0, 2].set_title("t Frame")
    axes[0, 2].axis("off")

    axes[0, 3].imshow(cur_rgb)
    axes[0, 3].imshow(sun_prior_vis, cmap="jet", alpha=0.35)
    axes[0, 3].set_title("Sun Prior")
    axes[0, 3].axis("off")

    axes[1, 0].imshow(pred_mag, cmap="magma")
    step = max(pred_mag.shape[0] // 16, 1)
    yy, xx = np.mgrid[0 : pred_mag.shape[0] : step, 0 : pred_mag.shape[1] : step]
    axes[1, 0].quiver(xx, yy, pred_u[::step, ::step], pred_v[::step, ::step], color="white")
    axes[1, 0].set_title("Learned Motion (t-2 -> t)")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(gt_mag, cmap="magma")
    axes[1, 1].quiver(xx, yy, gt_u[::step, ::step], gt_v[::step, ::step], color="white")
    axes[1, 1].set_title("Pseudo Flow (t-2 -> t)")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(mask, cmap="gray")
    axes[1, 2].set_title("Flow Valid Mask")
    axes[1, 2].axis("off")

    diff_mag = np.abs(pred_mag - gt_mag) * mask
    axes[1, 3].imshow(diff_mag, cmap="inferno")
    axes[1, 3].set_title("|Motion|-|Flow| Difference")
    axes[1, 3].axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
