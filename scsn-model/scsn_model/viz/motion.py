from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def _add_colorbar(fig: plt.Figure, image, ax: plt.Axes, label: str, ticks: list[float] | None = None) -> None:
    colorbar = fig.colorbar(image, ax=ax, fraction=0.052, pad=0.035, ticks=ticks)
    colorbar.set_label(label, rotation=270, labelpad=12)


def _resize_map(arr: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if arr.shape == shape:
        return arr
    im = Image.fromarray(np.asarray(arr, dtype=np.float32), mode="F")
    return np.asarray(im.resize((shape[1], shape[0]), resample=Image.BILINEAR), dtype=np.float32)


def save_scsn_state_figure(
    image: np.ndarray,
    attention: np.ndarray,
    current_cloud_prob: np.ndarray,
    future_cloud_prob: np.ndarray,
    motion_hotspot: np.ndarray,
    past_rbr_change_hotspot: np.ndarray,
    future_rbr_change_hotspot: np.ndarray,
    future_sun_cloud_prob: np.ndarray,
    out_path: str | Path,
    title: str,
    cloud_mask: np.ndarray | None = None,
    cloud_mask_valid: bool = False,
    future_hotspot_valid: bool = False,
    future_cloud_uncertainty: np.ndarray | None = None,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rgb = np.transpose(image, (1, 2, 0))
    image_shape = rgb.shape[:2]
    attention_overlay = _resize_map(attention, image_shape)
    past_change_overlay = _resize_map(past_rbr_change_hotspot, image_shape)
    motion_overlay = _resize_map(motion_hotspot, image_shape)
    if future_cloud_uncertainty is None:
        future_cloud_uncertainty = 4.0 * future_cloud_prob * (1.0 - future_cloud_prob)

    fig, axes = plt.subplots(3, 4, figsize=(18, 12), constrained_layout=True)
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("Last RGB Frame")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(rgb)
    attention_im = axes[0, 1].imshow(attention_overlay, cmap="jet", alpha=0.4, vmin=0.0, vmax=max(float(np.max(attention_overlay)), 1e-6))
    axes[0, 1].set_title("Sun Attention")
    axes[0, 1].axis("off")
    _add_colorbar(fig, attention_im, axes[0, 1], "attention")

    current_cloud_im = axes[0, 2].imshow(current_cloud_prob, cmap="Blues", vmin=0.0, vmax=1.0)
    axes[0, 2].set_title("Current Cloud Probability")
    axes[0, 2].axis("off")
    _add_colorbar(fig, current_cloud_im, axes[0, 2], "probability", ticks=[0.0, 0.5, 1.0])

    future_cloud_im = axes[0, 3].imshow(future_cloud_prob, cmap="Blues", vmin=0.0, vmax=1.0)
    axes[0, 3].set_title("Predicted 15min Cloud Risk")
    axes[0, 3].axis("off")
    _add_colorbar(fig, future_cloud_im, axes[0, 3], "probability", ticks=[0.0, 0.5, 1.0])

    past_change_im = axes[1, 0].imshow(past_rbr_change_hotspot, cmap="magma", vmin=0.0, vmax=1.0)
    axes[1, 0].set_title("Observed Past RBR Change")
    axes[1, 0].axis("off")
    _add_colorbar(fig, past_change_im, axes[1, 0], "relative change", ticks=[0.0, 0.5, 1.0])

    if future_hotspot_valid:
        future_change_im = axes[1, 1].imshow(future_rbr_change_hotspot, cmap="magma", vmin=0.0, vmax=1.0)
        axes[1, 1].set_title("Observed Future RBR Change")
        _add_colorbar(fig, future_change_im, axes[1, 1], "relative change", ticks=[0.0, 0.5, 1.0])
    else:
        axes[1, 1].imshow(np.zeros_like(motion_hotspot), cmap="magma", vmin=0.0, vmax=1.0)
        axes[1, 1].set_title("Observed Future RBR Change (missing)")
    axes[1, 1].axis("off")

    motion_im = axes[1, 2].imshow(motion_hotspot, cmap="inferno", vmin=0.0, vmax=1.0)
    axes[1, 2].set_title("Predicted 15min Motion Hotspot")
    axes[1, 2].axis("off")
    _add_colorbar(fig, motion_im, axes[1, 2], "relative activity", ticks=[0.0, 0.5, 1.0])

    uncertainty_im = axes[1, 3].imshow(future_cloud_uncertainty, cmap="cividis", vmin=0.0, vmax=1.0)
    axes[1, 3].set_title("Predicted 15min Risk/Uncertainty")
    axes[1, 3].axis("off")
    _add_colorbar(fig, uncertainty_im, axes[1, 3], "uncertainty", ticks=[0.0, 0.5, 1.0])

    if cloud_mask_valid and cloud_mask is not None:
        mask_im = axes[2, 0].imshow(cloud_mask, cmap="gray", vmin=0.0, vmax=1.0)
        axes[2, 0].set_title("Pseudo Cloud Mask")
        _add_colorbar(fig, mask_im, axes[2, 0], "label", ticks=[0.0, 1.0])
    else:
        axes[2, 0].imshow(np.zeros_like(current_cloud_prob), cmap="gray", vmin=0.0, vmax=1.0)
        axes[2, 0].set_title("Pseudo Cloud Mask (missing)")
    axes[2, 0].axis("off")

    axes[2, 1].imshow(rgb)
    axes[2, 1].imshow(past_change_overlay, cmap="magma", alpha=0.45, vmin=0.0, vmax=1.0)
    axes[2, 1].set_title("Past Change Overlay")
    axes[2, 1].axis("off")

    axes[2, 2].imshow(rgb)
    axes[2, 2].imshow(motion_overlay, cmap="inferno", alpha=0.45, vmin=0.0, vmax=1.0)
    axes[2, 2].set_title("Predicted Hotspot Overlay")
    axes[2, 2].axis("off")

    axes[2, 3].plot(np.arange(1, len(future_sun_cloud_prob) + 1), future_sun_cloud_prob, color="tab:blue", linewidth=1.8)
    axes[2, 3].set_ylim(0.0, 1.0)
    axes[2, 3].set_title("Sun-Region Cloud Risk")
    axes[2, 3].set_xlabel("Forecast Step")
    axes[2, 3].grid(alpha=0.25)

    fig.suptitle(title)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
