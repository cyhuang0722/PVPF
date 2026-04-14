from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_scsn_state_figure(
    image: np.ndarray,
    attention: np.ndarray,
    transmission: np.ndarray,
    opacity: np.ndarray,
    gap: np.ndarray,
    motion_u: np.ndarray,
    motion_v: np.ndarray,
    sun_occlusion: np.ndarray,
    out_path: str | Path,
    title: str,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rgb = np.transpose(image, (1, 2, 0))
    motion_mag = np.sqrt(motion_u**2 + motion_v**2)
    step = max(motion_mag.shape[0] // 12, 1)
    yy, xx = np.mgrid[0 : motion_mag.shape[0] : step, 0 : motion_mag.shape[1] : step]

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("Last RGB Frame")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(rgb)
    axes[0, 1].imshow(attention, cmap="jet", alpha=0.4)
    axes[0, 1].set_title("Sun Attention")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(transmission, cmap="viridis", vmin=0.0, vmax=1.0)
    axes[0, 2].set_title("Transmission")
    axes[0, 2].axis("off")

    axes[0, 3].imshow(opacity, cmap="magma", vmin=0.0, vmax=1.0)
    axes[0, 3].set_title("Opacity")
    axes[0, 3].axis("off")

    axes[1, 0].imshow(gap, cmap="cividis", vmin=0.0, vmax=1.0)
    axes[1, 0].set_title("Gap Probability")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(motion_mag, cmap="inferno")
    axes[1, 1].quiver(xx, yy, motion_u[::step, ::step], motion_v[::step, ::step], color="white")
    axes[1, 1].set_title("Future Motion")
    axes[1, 1].axis("off")

    axes[1, 2].plot(np.arange(1, len(sun_occlusion) + 1), sun_occlusion, color="tab:red", linewidth=1.8)
    axes[1, 2].set_ylim(0.0, 1.0)
    axes[1, 2].set_title("Sun Occlusion Curve")
    axes[1, 2].set_xlabel("Forecast Step")
    axes[1, 2].grid(alpha=0.25)

    effective = 0.5 * transmission + 0.5 * (1.0 - opacity)
    axes[1, 3].imshow(effective, cmap="viridis", vmin=0.0, vmax=1.0)
    axes[1, 3].set_title("Effective Transmission")
    axes[1, 3].axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
