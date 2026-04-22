from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

WEATHER_ORDER = ["clear_sky", "cloudy", "overcast"]


def load_metrics(run_dir: Path) -> dict:
    return json.loads((run_dir / "metrics_test.json").read_text(encoding="utf-8"))


def find_latest_run(model_dir: Path) -> Path:
    runs = sorted((model_dir / "runs").glob("run_*"))
    if not runs:
        raise FileNotFoundError(f"no runs found under {model_dir / 'runs'}")
    return runs[-1]


def collect_rows(named_runs: list[str]) -> pd.DataFrame:
    rows = []
    for item in named_runs:
        if "=" in item:
            name, path = item.split("=", 1)
            run_dir = Path(path)
        else:
            run_dir = Path(item)
            name = run_dir.parent.parent.name if run_dir.parent.name == "runs" else run_dir.name
        metrics = load_metrics(run_dir)
        row = {
            "model": name,
            "run_dir": str(run_dir),
            "rmse": metrics["rmse"],
            "mae": metrics["mae"],
            "baseline_rmse": metrics["baseline_rmse"],
            "coverage_80": metrics.get("coverage_80", np.nan),
            "width_80_w": metrics.get("mean_interval_width_w", np.nan),
        }
        for weather in WEATHER_ORDER:
            row[f"{weather}_n"] = metrics.get(f"weather_{weather}_n", 0)
            row[f"{weather}_rmse"] = metrics.get(f"weather_{weather}_rmse", np.nan)
            row[f"{weather}_baseline_rmse"] = metrics.get(f"weather_{weather}_baseline_rmse", np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def save_overall_plot(df: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), dpi=220)
    x = np.arange(len(df))
    axes[0].bar(x - 0.18, df["baseline_rmse"], width=0.36, label="smart persistence", color="#9aa3ad")
    axes[0].bar(x + 0.18, df["rmse"], width=0.36, label="model", color="#3577b8")
    axes[0].set_ylabel("Test RMSE (W)")
    axes[0].set_xticks(x, df["model"], rotation=20, ha="right")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].set_title("Point forecast")

    improvement = 100.0 * (df["baseline_rmse"] - df["rmse"]) / df["baseline_rmse"].replace(0, np.nan)
    axes[1].bar(x, improvement, color="#4f9f72")
    axes[1].axhline(0, color="#222222", linewidth=0.8)
    axes[1].set_ylabel("RMSE improvement (%)")
    axes[1].set_xticks(x, df["model"], rotation=20, ha="right")
    axes[1].set_title("Gain over persistence")

    axes[2].bar(x - 0.18, df["coverage_80"], width=0.36, color="#8267b8", label="coverage")
    ax2 = axes[2].twinx()
    ax2.bar(x + 0.18, df["width_80_w"], width=0.36, color="#d88c3a", label="width")
    axes[2].axhline(0.80, color="#222222", linewidth=0.8, linestyle="--")
    axes[2].set_ylim(0, 1.05)
    axes[2].set_ylabel("q10-q90 coverage")
    ax2.set_ylabel("Mean width (W)")
    axes[2].set_xticks(x, df["model"], rotation=20, ha="right")
    axes[2].set_title("Probabilistic calibration")

    fig.suptitle("Image-only baseline comparison", y=1.02, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def save_weather_plot(df: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, len(WEATHER_ORDER), figsize=(14.5, 3.9), dpi=220, sharey=True)
    for ax, weather in zip(axes, WEATHER_ORDER):
        x = np.arange(len(df))
        ax.bar(x - 0.18, df[f"{weather}_baseline_rmse"], width=0.36, color="#9aa3ad", label="smart persistence")
        ax.bar(x + 0.18, df[f"{weather}_rmse"], width=0.36, color="#3577b8", label="model")
        counts = df[f"{weather}_n"].fillna(0).astype(int).tolist()
        labels = [f"{name}\nn={count}" for name, count in zip(df["model"], counts)]
        ax.set_xticks(x, labels, rotation=20, ha="right")
        ax.set_title(weather.replace("_", " "))
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Test RMSE (W)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Weather-stratified baseline comparison", y=1.03, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def write_summary(df: pd.DataFrame, out: Path) -> None:
    lines = ["# Model Comparison Summary", ""]
    lines.append("## Overall")
    for _, row in df.iterrows():
        improvement = 100.0 * (row["baseline_rmse"] - row["rmse"]) / row["baseline_rmse"]
        lines.append(
            f"- {row['model']}: RMSE {row['rmse']:.0f} W vs persistence {row['baseline_rmse']:.0f} W "
            f"({improvement:+.1f}%), coverage80 {row['coverage_80']:.3f}, width80 {row['width_80_w']:.0f} W."
        )
    lines.append("")
    lines.append("## Weather RMSE")
    for weather in WEATHER_ORDER:
        lines.append(f"- {weather}:")
        for _, row in df.iterrows():
            lines.append(
                f"  - {row['model']}: n={int(row[f'{weather}_n'])}, "
                f"RMSE {row[f'{weather}_rmse']:.0f} W, persistence {row[f'{weather}_baseline_rmse']:.0f} W."
            )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare trained model-compare runs.")
    parser.add_argument("--run", action="append", default=[], help="Either name=/path/to/run or /path/to/run. Can repeat.")
    parser.add_argument("--artifact-root", type=Path, default=Path(__file__).resolve().parents[1] / "artifacts")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    runs = list(args.run)
    if not runs:
        for name in ["convlstm", "cnn_gru", "image_regressor", "vae_regressor"]:
            runs.append(f"{name}={find_latest_run(args.artifact_root / name)}")
    out_dir = args.out_dir or (args.artifact_root / "comparison")
    out_dir.mkdir(parents=True, exist_ok=True)
    df = collect_rows(runs)
    df.to_csv(out_dir / "model_comparison_test.csv", index=False)
    save_overall_plot(df, out_dir / "overall_comparison.png")
    save_weather_plot(df, out_dir / "weather_comparison.png")
    write_summary(df, out_dir / "comparison_summary.md")
    print(json.dumps({"out_dir": str(out_dir), "models": df["model"].tolist()}, indent=2))


if __name__ == "__main__":
    main()
