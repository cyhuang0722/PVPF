from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import pandas as pd


def save_forecast_band(df: pd.DataFrame, out_path: str | Path, title: str) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame = df.sort_values("interval_end").head(240).copy()
    x = pd.to_datetime(frame["interval_end"])
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.fill_between(x, frame["q10_w"], frame["q90_w"], color="#86a8e7", alpha=0.30, label="q10-q90")
    ax.plot(x, frame["q50_w"], color="#174ea6", linewidth=1.6, label="q50")
    ax.plot(x, frame["target_pv_w"], color="#111111", linewidth=1.2, label="true")
    ax.plot(x, frame["baseline_pv_w"], color="#d97706", linewidth=1.1, alpha=0.9, label="baseline")
    ax.set_title(title)
    ax.set_ylabel("PV Power (W)")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=4, loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def save_weather_gate_plot(df: pd.DataFrame, out_path: str | Path) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame = df.copy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    frame.boxplot(column="cloud_gate", by="weather_tag", ax=axes[0])
    axes[0].set_title("Cloud Gate by Weather")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("gate")
    frame.boxplot(column="scale", by="weather_tag", ax=axes[1])
    axes[1].set_title("Predictive Scale by Weather")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Student-t scale")
    fig.suptitle("")
    fig.savefig(out, dpi=180)
    plt.close(fig)
