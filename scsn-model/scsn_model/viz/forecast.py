from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_forecast_band_plot(df: pd.DataFrame, out_path: str | Path, title: str) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["ts_target"], df["target_pv_w"], label="True PV", color="black", linewidth=1.6)
    center_col = "q50_w" if "q50_w" in df.columns else "pred_w"
    lower_col = "q10_w" if "q10_w" in df.columns else None
    upper_col = "q90_w" if "q90_w" in df.columns else None
    inner_lower_col = "q25_w" if "q25_w" in df.columns else None
    inner_upper_col = "q75_w" if "q75_w" in df.columns else None
    ax.plot(df["ts_target"], df[center_col], label="Pred PV", color="tab:blue", linewidth=1.4)
    if lower_col and upper_col:
        ax.fill_between(df["ts_target"], df[lower_col], df[upper_col], color="tab:blue", alpha=0.18, label="q10-q90")
    if inner_lower_col and inner_upper_col:
        ax.fill_between(df["ts_target"], df[inner_lower_col], df[inner_upper_col], color="tab:blue", alpha=0.28, label="q25-q75")
    ax.set_title(title)
    ax.set_xlabel("Target Timestamp")
    ax.set_ylabel("Power (W)")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
