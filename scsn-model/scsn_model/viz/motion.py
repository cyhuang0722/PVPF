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


def _format_summary(values: dict[str, float] | None) -> list[str]:
    if not values:
        return ["diagnostics", "missing"]
    lines = ["diagnostics"]
    for key, value in values.items():
        if np.isfinite(value):
            lines.append(f"{key}: {float(value):.3f}")
    return lines


def save_scsn_state_figure(
    image: np.ndarray,
    attention: np.ndarray,
    rbr_mean: np.ndarray,
    rbr_variance: np.ndarray,
    past_rbr_change_hotspot: np.ndarray,
    future_rbr_change_hotspot: np.ndarray,
    out_path: str | Path,
    title: str,
    future_hotspot_valid: bool = False,
    summary_values: dict[str, float] | None = None,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rgb = np.transpose(image, (1, 2, 0))
    image_shape = rgb.shape[:2]
    attention_overlay = _resize_map(attention, image_shape)
    past_change_overlay = _resize_map(past_rbr_change_hotspot, image_shape)
    mean_overlay = _resize_map(rbr_mean, image_shape)
    variance_overlay = _resize_map(rbr_variance, image_shape)

    fig, axes = plt.subplots(3, 4, figsize=(18, 12), constrained_layout=True)
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("Last RGB Frame")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(rgb)
    attention_im = axes[0, 1].imshow(attention_overlay, cmap="jet", alpha=0.4, vmin=0.0, vmax=max(float(np.max(attention_overlay)), 1e-6))
    axes[0, 1].set_title("Target Sun-Region Weight")
    axes[0, 1].axis("off")
    _add_colorbar(fig, attention_im, axes[0, 1], "attention")

    mean_im = axes[0, 2].imshow(rbr_mean, cmap="viridis", vmin=0.0, vmax=1.0)
    axes[0, 2].set_title("Predicted RBR Mean")
    axes[0, 2].axis("off")
    _add_colorbar(fig, mean_im, axes[0, 2], "mean", ticks=[0.0, 0.5, 1.0])

    variance_im = axes[0, 3].imshow(rbr_variance, cmap="cividis", vmin=0.0, vmax=max(float(np.nanpercentile(rbr_variance, 99)), 1e-6))
    axes[0, 3].set_title("Predicted RBR Variance")
    axes[0, 3].axis("off")
    _add_colorbar(fig, variance_im, axes[0, 3], "variance")

    past_change_im = axes[1, 0].imshow(past_rbr_change_hotspot, cmap="magma", vmin=0.0, vmax=1.0)
    axes[1, 0].set_title("Observed Past RBR Change")
    axes[1, 0].axis("off")
    _add_colorbar(fig, past_change_im, axes[1, 0], "relative change", ticks=[0.0, 0.5, 1.0])

    if future_hotspot_valid:
        future_change_im = axes[1, 1].imshow(future_rbr_change_hotspot, cmap="magma", vmin=0.0, vmax=1.0)
        axes[1, 1].set_title("Observed Future RBR Change")
        _add_colorbar(fig, future_change_im, axes[1, 1], "relative change", ticks=[0.0, 0.5, 1.0])
    else:
        axes[1, 1].imshow(np.zeros_like(rbr_mean), cmap="magma", vmin=0.0, vmax=1.0)
        axes[1, 1].set_title("Observed Future RBR Change (missing)")
    axes[1, 1].axis("off")

    mean2_im = axes[1, 2].imshow(rbr_mean, cmap="viridis", vmin=0.0, vmax=1.0)
    axes[1, 2].set_title("Predicted 15min RBR Mean")
    axes[1, 2].axis("off")
    _add_colorbar(fig, mean2_im, axes[1, 2], "mean", ticks=[0.0, 0.5, 1.0])

    axes[1, 3].axis("off")
    axes[1, 3].set_title("Prediction Diagnostics")
    summary_lines = _format_summary(summary_values)
    axes[1, 3].text(
        0.08,
        0.72,
        "\n".join(summary_lines),
        transform=axes[1, 3].transAxes,
        va="top",
        ha="left",
        fontsize=11,
        family="monospace",
    )

    axes[2, 0].imshow(rgb)
    sun_overlay_im = axes[2, 0].imshow(attention_overlay, cmap="jet", alpha=0.4, vmin=0.0, vmax=max(float(np.max(attention_overlay)), 1e-6))
    axes[2, 0].set_title("Sun Region Overlay")
    axes[2, 0].axis("off")
    _add_colorbar(fig, sun_overlay_im, axes[2, 0], "attention")

    axes[2, 1].imshow(rgb)
    past_overlay_im = axes[2, 1].imshow(past_change_overlay, cmap="magma", alpha=0.45, vmin=0.0, vmax=1.0)
    axes[2, 1].set_title("Past Change Overlay")
    axes[2, 1].axis("off")
    _add_colorbar(fig, past_overlay_im, axes[2, 1], "relative change", ticks=[0.0, 0.5, 1.0])

    axes[2, 2].imshow(rgb)
    mean_overlay_im = axes[2, 2].imshow(mean_overlay, cmap="viridis", alpha=0.45, vmin=0.0, vmax=1.0)
    axes[2, 2].set_title("Predicted Mean Overlay")
    axes[2, 2].axis("off")
    _add_colorbar(fig, mean_overlay_im, axes[2, 2], "mean", ticks=[0.0, 0.5, 1.0])
    axes[2, 3].imshow(rgb)
    variance_overlay_im = axes[2, 3].imshow(variance_overlay, cmap="cividis", alpha=0.45, vmin=0.0, vmax=max(float(np.nanpercentile(rbr_variance, 99)), 1e-6))
    axes[2, 3].set_title("Predicted Variance Overlay")
    axes[2, 3].axis("off")
    _add_colorbar(fig, variance_overlay_im, axes[2, 3], "variance")

    fig.suptitle(title)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
