from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _add_colorbar(fig: plt.Figure, image, ax: plt.Axes, label: str, ticks: list[float] | None = None) -> None:
    colorbar = fig.colorbar(image, ax=ax, fraction=0.052, pad=0.035, ticks=ticks)
    colorbar.set_label(label, rotation=270, labelpad=12)


def save_scsn_state_figure(
    image: np.ndarray,
    attention: np.ndarray,
    current_cloud_prob: np.ndarray,
    future_cloud_prob: np.ndarray,
    motion_u: np.ndarray,
    motion_v: np.ndarray,
    future_sun_cloud_prob: np.ndarray,
    out_path: str | Path,
    title: str,
    cloud_mask: np.ndarray | None = None,
    cloud_mask_valid: bool = False,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rgb = np.transpose(image, (1, 2, 0))
    motion_mag = np.sqrt(motion_u**2 + motion_v**2)
    cloud_uncertainty = 4.0 * future_cloud_prob * (1.0 - future_cloud_prob)
    step = max(motion_mag.shape[0] // 12, 1)
    yy, xx = np.mgrid[0 : motion_mag.shape[0] : step, 0 : motion_mag.shape[1] : step]

    fig, axes = plt.subplots(2, 4, figsize=(18, 9), constrained_layout=True)
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("Last RGB Frame")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(rgb)
    attention_im = axes[0, 1].imshow(attention, cmap="jet", alpha=0.4, vmin=0.0, vmax=max(float(np.max(attention)), 1e-6))
    axes[0, 1].set_title("Sun Attention")
    axes[0, 1].axis("off")
    _add_colorbar(fig, attention_im, axes[0, 1], "attention")

    current_cloud_im = axes[0, 2].imshow(current_cloud_prob, cmap="Blues", vmin=0.0, vmax=1.0)
    axes[0, 2].set_title("Current Cloud Probability")
    axes[0, 2].axis("off")
    _add_colorbar(fig, current_cloud_im, axes[0, 2], "probability", ticks=[0.0, 0.5, 1.0])

    future_cloud_im = axes[0, 3].imshow(future_cloud_prob, cmap="Blues", vmin=0.0, vmax=1.0)
    axes[0, 3].set_title("Future Cloud Probability")
    axes[0, 3].axis("off")
    _add_colorbar(fig, future_cloud_im, axes[0, 3], "probability", ticks=[0.0, 0.5, 1.0])

    if cloud_mask_valid and cloud_mask is not None:
        mask_im = axes[1, 0].imshow(cloud_mask, cmap="gray", vmin=0.0, vmax=1.0)
        axes[1, 0].set_title("Pseudo Cloud Mask")
        _add_colorbar(fig, mask_im, axes[1, 0], "label", ticks=[0.0, 1.0])
    else:
        axes[1, 0].imshow(np.zeros_like(current_cloud_prob), cmap="gray", vmin=0.0, vmax=1.0)
        axes[1, 0].set_title("Pseudo Cloud Mask (missing)")
    axes[1, 0].axis("off")

    motion_im = axes[1, 1].imshow(motion_mag, cmap="inferno", vmin=0.0, vmax=max(float(np.max(motion_mag)), 1e-6))
    axes[1, 1].quiver(xx, yy, motion_u[::step, ::step], motion_v[::step, ::step], color="white")
    axes[1, 1].set_title("Future Motion")
    axes[1, 1].axis("off")
    _add_colorbar(fig, motion_im, axes[1, 1], "pixels")

    axes[1, 2].plot(np.arange(1, len(future_sun_cloud_prob) + 1), future_sun_cloud_prob, color="tab:blue", linewidth=1.8)
    axes[1, 2].set_ylim(0.0, 1.0)
    axes[1, 2].set_title("Sun-Region Cloud Probability")
    axes[1, 2].set_xlabel("Forecast Step")
    axes[1, 2].grid(alpha=0.25)

    uncertainty_im = axes[1, 3].imshow(cloud_uncertainty, cmap="cividis", vmin=0.0, vmax=1.0)
    axes[1, 3].set_title("Future Cloud Uncertainty")
    axes[1, 3].axis("off")
    _add_colorbar(fig, uncertainty_im, axes[1, 3], "uncertainty", ticks=[0.0, 0.5, 1.0])

    fig.suptitle(title)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
