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

