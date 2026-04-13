from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_forecast_band_plot(df: pd.DataFrame, out_path: str | Path, title: str) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["ts_target"], df["target_pv_w"], label="True PV", color="black", linewidth=1.6)
    ax.plot(df["ts_target"], df["pred_w"], label="Pred PV", color="tab:blue", linewidth=1.4)
    ax.set_title(title)
    ax.set_xlabel("Target Timestamp")
    ax.set_ylabel("Power (W)")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
