from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_forecast_band(df: pd.DataFrame, out_path: str | Path, title: str) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame = df.sort_values("ts_target").head(240).copy()
    x = pd.to_datetime(frame["ts_target"])
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.fill_between(x, frame["q10_w"], frame["q90_w"], color="#8fb3ff", alpha=0.28, label="q10-q90")
    ax.plot(x, frame["q50_w"], color="#164aa8", linewidth=1.6, label="q50")
    ax.plot(x, frame["target_pv_w"], color="#111111", linewidth=1.2, label="true")
    ax.plot(x, frame["baseline_pv_w"], color="#d97706", linewidth=1.1, alpha=0.9, label="baseline")
    ax.set_title(title)
    ax.set_ylabel("PV Power (W)")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=4, loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_case_plot(row: pd.Series, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    qs = [row["q10_w"], row["q25_w"], row["q50_w"], row["q75_w"], row["q90_w"]]
    labels = ["q10", "q25", "q50", "q75", "q90"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].bar(labels, qs, color=["#9bbcff", "#6f98e8", "#2f5fb3", "#6f98e8", "#9bbcff"])
    axes[0].axhline(float(row["target_pv_w"]), color="black", linewidth=1.4, label="true")
    axes[0].axhline(float(row["baseline_pv_w"]), color="#d97706", linewidth=1.2, linestyle="--", label="baseline")
    axes[0].set_ylabel("PV Power (W)")
    axes[0].set_title(str(row["ts_target"]))
    axes[0].legend()
    axes[1].axis("off")
    lines = [
        f"target: {float(row['target_pv_w']):.0f} W",
        f"q50: {float(row['q50_w']):.0f} W",
        f"baseline: {float(row['baseline_pv_w']):.0f} W",
        f"interval width: {float(row['q90_w'] - row['q10_w']):.0f} W",
        f"df: {float(row['df']):.2f}",
        f"sigma CSI: {float(row['scale']):.3f}",
    ]
    axes[1].text(0.02, 0.95, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=11)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
