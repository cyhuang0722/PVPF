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


def save_rbr_distribution_figure(
    image: np.ndarray,
    past_rbr_mean: np.ndarray,
    attention: np.ndarray,
    rbr_mean: np.ndarray,
    rbr_variance: np.ndarray,
    past_rbr_variation: np.ndarray,
    future_rbr_mean_target: np.ndarray,
    future_rbr_variance_target: np.ndarray,
    out_path: str | Path,
    title: str,
    future_rbr_valid: bool = False,
    summary_values: dict[str, float] | None = None,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rgb = np.transpose(image, (1, 2, 0))
    image_shape = rgb.shape[:2]
    target_mean_for_error = _resize_map(future_rbr_mean_target, rbr_mean.shape)
    target_variance_for_error = _resize_map(future_rbr_variance_target, rbr_variance.shape)
    mean_error = np.abs(rbr_mean - target_mean_for_error) if future_rbr_valid else np.zeros_like(rbr_mean)
    variance_error = np.abs(rbr_variance - target_variance_for_error) if future_rbr_valid else np.zeros_like(rbr_variance)
    variance_vmax = max(
        float(np.nanpercentile(rbr_variance, 99)),
        float(np.nanpercentile(future_rbr_variance_target, 99)) if future_rbr_valid else 0.0,
        1e-6,
    )
    variance_error_vmax = max(float(np.nanpercentile(variance_error, 99)), 1e-6)

    fig, axes = plt.subplots(3, 4, figsize=(18, 12), constrained_layout=True)
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("Input: Current RGB")
    axes[0, 0].axis("off")

    past_mean_im = axes[0, 1].imshow(past_rbr_mean, cmap="viridis", vmin=0.0, vmax=1.0)
    axes[0, 1].set_title("Input: Past 15min Avg RBR")
    axes[0, 1].axis("off")
    _add_colorbar(fig, past_mean_im, axes[0, 1], "mean", ticks=[0.0, 0.5, 1.0])

    past_change_im = axes[0, 2].imshow(past_rbr_variation, cmap="magma", vmin=0.0, vmax=1.0)
    axes[0, 2].set_title("Input: Past RBR Variation")
    axes[0, 2].axis("off")
    _add_colorbar(fig, past_change_im, axes[0, 2], "relative change", ticks=[0.0, 0.5, 1.0])

    attention_im = axes[0, 3].imshow(attention, cmap="jet", vmin=0.0, vmax=max(float(np.max(attention)), 1e-6))
    axes[0, 3].set_title("Learned Target Sun Attention")
    axes[0, 3].axis("off")
    _add_colorbar(fig, attention_im, axes[0, 3], "attention")

    pred_mean_im = axes[1, 0].imshow(rbr_mean, cmap="viridis", vmin=0.0, vmax=1.0)
    axes[1, 0].set_title("Output: Pred Future Avg RBR")
    axes[1, 0].axis("off")
    _add_colorbar(fig, pred_mean_im, axes[1, 0], "mean", ticks=[0.0, 0.5, 1.0])

    if future_rbr_valid:
        future_mean_im = axes[1, 1].imshow(future_rbr_mean_target, cmap="viridis", vmin=0.0, vmax=1.0)
        axes[1, 1].set_title("True Future Avg RBR")
        _add_colorbar(fig, future_mean_im, axes[1, 1], "mean", ticks=[0.0, 0.5, 1.0])
    else:
        axes[1, 1].imshow(np.zeros_like(rbr_mean), cmap="viridis", vmin=0.0, vmax=1.0)
        axes[1, 1].set_title("True Future Avg RBR (missing)")
    axes[1, 1].axis("off")

    mean_error_im = axes[1, 2].imshow(mean_error, cmap="inferno", vmin=0.0, vmax=1.0)
    axes[1, 2].set_title("|Pred - True| Avg RBR")
    axes[1, 2].axis("off")
    _add_colorbar(fig, mean_error_im, axes[1, 2], "abs error", ticks=[0.0, 0.5, 1.0])

    axes[1, 3].imshow(rgb)
    mean_overlay = _resize_map(rbr_mean, image_shape)
    mean_overlay_im = axes[1, 3].imshow(mean_overlay, cmap="viridis", alpha=0.45, vmin=0.0, vmax=1.0)
    axes[1, 3].set_title("Pred Avg RBR Overlay")
    axes[1, 3].axis("off")
    _add_colorbar(fig, mean_overlay_im, axes[1, 3], "mean", ticks=[0.0, 0.5, 1.0])

    pred_var_im = axes[2, 0].imshow(rbr_variance, cmap="cividis", vmin=0.0, vmax=variance_vmax)
    axes[2, 0].set_title("Output: Pred Future RBR Variance")
    axes[2, 0].axis("off")
    _add_colorbar(fig, pred_var_im, axes[2, 0], "variance")

    if future_rbr_valid:
        true_var_im = axes[2, 1].imshow(future_rbr_variance_target, cmap="cividis", vmin=0.0, vmax=variance_vmax)
        axes[2, 1].set_title("True Future RBR Variance")
    else:
        true_var_im = axes[2, 1].imshow(np.zeros_like(rbr_variance), cmap="cividis", vmin=0.0, vmax=variance_vmax)
        axes[2, 1].set_title("True Future RBR Variance (missing)")
    axes[2, 1].axis("off")
    _add_colorbar(fig, true_var_im, axes[2, 1], "variance")

    var_error_im = axes[2, 2].imshow(variance_error, cmap="inferno", vmin=0.0, vmax=variance_error_vmax)
    axes[2, 2].set_title("|Pred - True| RBR Variance")
    axes[2, 2].axis("off")
    _add_colorbar(fig, var_error_im, axes[2, 2], "abs error")

    axes[2, 3].axis("off")
    axes[2, 3].set_title("Prediction Diagnostics")
    summary_lines = _format_summary(summary_values)
    axes[2, 3].text(
        0.08,
        0.72,
        "\n".join(summary_lines),
        transform=axes[2, 3].transAxes,
        va="top",
        ha="left",
        fontsize=11,
        family="monospace",
    )

    fig.suptitle(title)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
