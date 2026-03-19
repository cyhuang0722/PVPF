#!/usr/bin/env python3
"""
Publication-grade plotting for PV forecasting predictions on val/test.

Input CSV convention (Sky+PV model):
  - ts column: ts_pred
  - predictions: pv_pred_W_{k}
  - ground truth: pv_true_W_{k}

Satellite-only baseline convention (optional):
  - ts_pred
  - pv_pred_t+15_W / pv_true_t+15_W, etc.

Features:
  - Select time range via --start/--end (inclusive).
  - Robust timestamp parsing (tz-aware or naive).
  - Fallback timestamp reconstruction from forecast_windows.csv when ts_pred is missing.
  - Exports high-quality PDF/PNG.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def _default_base_dir() -> Path:
    return Path(__file__).resolve().parent


def _resolve_path(p: str | Path) -> Path:
    p = Path(p)
    if p.is_absolute():
        return p
    # allow both:
    # - pv_forecasting/model_output/run_xxx
    # - model_output/run_xxx (relative to pv_forecasting/)
    base = _default_base_dir().parent if str(p).startswith("pv_forecasting") else _default_base_dir()
    return (base / p).resolve()


def _parse_ts_series(ts: pd.Series, tz: str | None) -> pd.Series:
    # Try parse preserving tz info if present in strings.
    s = pd.to_datetime(ts, errors="coerce", utc=True)
    if tz:
        s = s.dt.tz_convert(tz)
    # For plotting we typically want tz-naive local timestamps.
    s = s.dt.tz_localize(None)
    return s


def _ensure_ts_pred(
    pred_df: pd.DataFrame,
    *,
    tz: str | None,
    window_csv: Path | None,
    split: str | None,
    val_ratio: float,
) -> pd.DataFrame:
    if "ts_pred" in pred_df.columns:
        parsed = _parse_ts_series(pred_df["ts_pred"], tz)
        if parsed.notna().any():
            pred_df = pred_df.copy()
            pred_df["ts_pred"] = parsed
            return pred_df

    if window_csv is None:
        raise ValueError(
            "Column 'ts_pred' is missing/empty in predictions CSV. "
            "Provide --window-csv to reconstruct timestamps (e.g. pv_forecasting/derived/forecast_windows.csv "
            "for val/all, or pv_forecasting/derived/test/test_forecast_windows.csv for test), "
            "or regenerate predictions with pv_forecasting.infer so ts_pred is saved correctly."
        )

    if split is None:
        raise ValueError(
            "Need --split when reconstructing timestamps from --window-csv (use val/test/all)."
        )

    wdf = pd.read_csv(window_csv)
    if "ts_pred" not in wdf.columns:
        raise ValueError(f"Column 'ts_pred' not found in window CSV: {window_csv}")
    ts_all = _parse_ts_series(wdf["ts_pred"], tz)

    n = len(pred_df)
    if split == "all":
        ts = ts_all
    elif split == "val":
        n_val = max(1, int(len(ts_all) * val_ratio))
        ts = ts_all.iloc[len(ts_all) - n_val :]
    elif split == "test":
        ts = ts_all
    else:
        raise ValueError(f"Unknown split: {split}")

    if len(ts) != n:
        raise ValueError(
            f"Timestamp reconstruction length mismatch: predictions has {n} rows, "
            f"but reconstructed ts has {len(ts)} rows (split={split}, window_csv={window_csv}). "
            "This usually means the predictions were generated from a different dataset/order."
        )

    pred_df = pred_df.copy()
    pred_df["ts_pred"] = ts.to_numpy()
    return pred_df


@dataclass(frozen=True)
class HorizonSpec:
    label: str
    pred_col: str
    true_col: str
    step_minutes: int | None


def _detect_horizons(df: pd.DataFrame) -> list[HorizonSpec]:
    # Style A: pv_pred_W_0 / pv_true_W_0 ...
    pred_w = sorted([c for c in df.columns if c.startswith("pv_pred_W_")])
    true_w = {c for c in df.columns if c.startswith("pv_true_W_")}
    if pred_w and any(c.replace("pv_pred_W_", "pv_true_W_") in true_w for c in pred_w):
        hs: list[HorizonSpec] = []
        for c in pred_w:
            suffix = c.replace("pv_pred_W_", "")
            tcol = f"pv_true_W_{suffix}"
            if tcol not in df.columns:
                continue
            try:
                k = int(suffix)
                step = 15 * (k + 1)
                label = f"t+{step}min"
            except ValueError:
                step = None
                label = suffix
            hs.append(HorizonSpec(label=label, pred_col=c, true_col=tcol, step_minutes=step))
        return hs

    # Style B (sat baseline): pv_pred_t+15_W / pv_true_t+15_W ...
    pred_cols = sorted([c for c in df.columns if c.startswith("pv_pred_") and c.endswith("_W")])
    hs2: list[HorizonSpec] = []
    for pc in pred_cols:
        tc = pc.replace("pv_pred_", "pv_true_", 1)
        if tc not in df.columns:
            continue
        label = pc.replace("pv_pred_", "").replace("_W", "")
        step = None
        if label.startswith("t+"):
            try:
                step = int(label.replace("t+", "").replace("min", "").replace("_", "").replace("m", ""))
            except ValueError:
                step = None
        hs2.append(HorizonSpec(label=label, pred_col=pc, true_col=tc, step_minutes=step))
    if hs2:
        return hs2

    raise ValueError(
        "Could not detect pred/true column pairs. Expected columns like "
        "'pv_pred_W_0' + 'pv_true_W_0' (Sky+PV) or 'pv_pred_t+15_W' + 'pv_true_t+15_W' (sat baseline). "
        f"Got columns: {list(df.columns)}"
    )


def _filter_time(df: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    if start is None and end is None:
        return df
    if "ts_pred" not in df.columns:
        raise ValueError("Need 'ts_pred' column for --start/--end filtering.")
    out = df
    if start is not None:
        t0 = pd.to_datetime(start)
        out = out[out["ts_pred"] >= t0]
    if end is not None:
        t1 = pd.to_datetime(end)
        out = out[out["ts_pred"] <= t1]
    return out


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return {"rmse": float("nan"), "mae": float("nan"), "r2": float("nan"), "n": 0.0}
    yt = y_true[mask]
    yp = y_pred[mask]
    err = yp - yt
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    denom = float(np.sum((yt - np.mean(yt)) ** 2))
    r2 = float("nan") if denom <= 0 else float(1.0 - (np.sum((yt - yp) ** 2) / denom))
    return {"rmse": rmse, "mae": mae, "r2": r2, "n": float(mask.sum())}


def _configure_matplotlib(paper: str) -> None:
    import matplotlib as mpl

    # Conservative defaults (no extra deps like SciencePlots).
    if paper == "a4":
        font_size = 10
    else:
        font_size = 9
    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "font.size": font_size,
            "axes.labelsize": font_size,
            "axes.titlesize": font_size + 1,
            "legend.fontsize": font_size - 1,
            "xtick.labelsize": font_size - 1,
            "ytick.labelsize": font_size - 1,
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.6,
            "lines.linewidth": 1.4,
            "lines.markersize": 3.5,
            "axes.grid": True,
            "grid.alpha": 0.28,
            "grid.linestyle": "-",
        }
    )


def plot_predictions(
    df: pd.DataFrame,
    horizons: list[HorizonSpec],
    *,
    out_dir: Path,
    prefix: str,
    fmt: str,
    paper: str,
    title: str | None,
    show_residual: bool,
) -> dict[str, dict[str, float]]:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.lines import Line2D

    _configure_matplotlib(paper)

    out_dir.mkdir(parents=True, exist_ok=True)

    x = df["ts_pred"] if "ts_pred" in df.columns else np.arange(len(df))
    use_time = "ts_pred" in df.columns

    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    metrics_by_h: dict[str, dict[str, float]] = {}

    # 1) Timeseries subplots (truth vs pred)
    n = len(horizons)
    fig_h = 1.95 if paper == "a4" else 1.75
    fig_w = 7.1 if paper == "a4" else 6.4
    span_days = None
    if use_time and len(df) > 1:
        tmin = pd.to_datetime(df["ts_pred"]).min()
        tmax = pd.to_datetime(df["ts_pred"]).max()
        span_days = max((tmax - tmin).total_seconds() / 86400.0, 1e-3)
        # Adaptive figure width for different time spans.
        if span_days <= 1.2:
            fig_w = 8.2 if paper == "a4" else 7.4
        elif span_days <= 3:
            fig_w = 10.0 if paper == "a4" else 9.0
        elif span_days <= 7:
            fig_w = 12.0 if paper == "a4" else 10.8
        else:
            fig_w = 14.0 if paper == "a4" else 12.5
    fig, axes = plt.subplots(nrows=n, ncols=1, sharex=True, figsize=(fig_w, fig_h * n))
    if n == 1:
        axes = [axes]
    for i, h in enumerate(horizons):
        ax = axes[i]
        y_true = df[h.true_col].to_numpy(dtype=float)
        y_pred = df[h.pred_col].to_numpy(dtype=float)
        m = _metrics(y_true, y_pred)
        metrics_by_h[h.label] = m

        c = palette[i % len(palette)]
        ax.plot(x, y_true, color="black", alpha=0.85, label="Truth")
        ax.plot(x, y_pred, color=c, alpha=0.95, label="Pred")
        ax.set_ylabel("PV (W)")
        ax.set_title(f"{h.label}  (RMSE={m['rmse']:.0f}W, MAE={m['mae']:.0f}W, R²={m['r2']:.3f}, n={int(m['n'])})")
        ax.grid(True, which="major")
        if show_residual:
            ax2 = ax.twinx()
            ax2.plot(x, y_pred - y_true, color=c, alpha=0.25, linewidth=1.0, label="Residual")
            ax2.set_ylabel("Pred-Truth (W)", color=c)
            ax2.tick_params(axis="y", colors=c)

    if title:
        fig.suptitle(title, y=1.02)

    # Legend with explicit color mapping for each horizon prediction line.
    legend_handles = [Line2D([0], [0], color="black", lw=1.6, label="Truth")]
    for i, h in enumerate(horizons):
        legend_handles.append(
            Line2D([0], [0], color=palette[i % len(palette)], lw=1.6, label=f"Pred {h.label}")
        )
    legend_cols = min(3, len(legend_handles))
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=legend_cols,
        frameon=False,
        bbox_to_anchor=(0.5, 1.01),
    )

    axes[-1].set_xlabel("Time" if use_time else "Index")
    if use_time:
        # Adaptive date ticks to keep labels readable for 1 day vs multiple days.
        if span_days is not None and span_days <= 1.2:
            locator = mdates.HourLocator(interval=2)
            formatter = mdates.DateFormatter("%m-%d\n%H:%M")
            rotation = 0
        elif span_days is not None and span_days <= 3:
            locator = mdates.HourLocator(interval=6)
            formatter = mdates.DateFormatter("%m-%d\n%H:%M")
            rotation = 0
        elif span_days is not None and span_days <= 10:
            locator = mdates.DayLocator(interval=1)
            formatter = mdates.DateFormatter("%m-%d")
            rotation = 0
        else:
            locator = mdates.DayLocator(interval=2)
            formatter = mdates.DateFormatter("%Y-%m-%d")
            rotation = 20

        axes[-1].xaxis.set_major_locator(locator)
        axes[-1].xaxis.set_major_formatter(formatter)
        fig.autofmt_xdate(rotation=rotation, ha="center")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = out_dir / f"{prefix}_timeseries.{fmt}"
    fig.savefig(out_path)
    plt.close(fig)

    # 2) Pred vs Truth scatter (one panel per horizon)
    cols = 2 if n > 1 else 1
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, 2.8 * rows))
    axes = np.array(axes).reshape(rows, cols)
    for i, h in enumerate(horizons):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        y_true = df[h.true_col].to_numpy(dtype=float)
        y_pred = df[h.pred_col].to_numpy(dtype=float)
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        yt = y_true[mask]
        yp = y_pred[mask]
        m = metrics_by_h[h.label]
        ax.scatter(yt, yp, s=8, alpha=0.35, color=palette[i % len(palette)], edgecolors="none")
        if len(yt):
            mn = float(np.min([yt.min(), yp.min()]))
            mx = float(np.max([yt.max(), yp.max()]))
            ax.plot([mn, mx], [mn, mx], color="gray", linestyle="--", linewidth=1.0, alpha=0.8)
        ax.set_title(f"{h.label} (R²={m['r2']:.3f})")
        ax.set_xlabel("Truth (W)")
        ax.set_ylabel("Pred (W)")
        ax.grid(True, which="major")

    # hide unused axes
    for j in range(n, rows * cols):
        r, c = divmod(j, cols)
        axes[r, c].axis("off")
    fig.tight_layout()
    out_path = out_dir / f"{prefix}_scatter.{fmt}"
    fig.savefig(out_path)
    plt.close(fig)

    return metrics_by_h


def _as_list(csv: str | None) -> list[str] | None:
    if csv is None:
        return None
    xs = [x.strip() for x in csv.split(",")]
    xs = [x for x in xs if x]
    return xs or None


def main() -> None:
    parser = argparse.ArgumentParser(description="Publication-grade plots for pv_forecasting predictions CSV.")
    parser.add_argument(
        "run_dir",
        nargs="?",
        default="pv_forecasting/model_output/run_20260308-004432",
        help="Run directory, e.g. pv_forecasting/model_output/run_xxx",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Predictions CSV name or path. Default depends on --split (val/test/all).",
    )
    parser.add_argument(
        "--split",
        choices=["val", "test", "all"],
        default="test",
        help="Which split the CSV corresponds to (used for defaults and timestamp reconstruction).",
    )
    parser.add_argument("--start", default=None, help="Start time (inclusive), e.g. 2026-02-24 08:00")
    parser.add_argument("--end", default=None, help="End time (inclusive), e.g. 2026-02-24 18:00")
    parser.add_argument(
        "--tz",
        default="Asia/Hong_Kong",
        help="Timezone to interpret/convert timestamps (ts_pred). Use empty to keep UTC.",
    )
    parser.add_argument(
        "--window-csv",
        default=None,
        help="Optional window CSV to reconstruct ts_pred when missing/empty. "
        "For val/all: pv_forecasting/derived/forecast_windows.csv; for test: pv_forecasting/derived/test/test_forecast_windows.csv",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Val ratio used during train/infer (for split=val).")
    parser.add_argument(
        "--horizons",
        default=None,
        help="Comma-separated horizon labels to keep (after auto-detect), e.g. t+15min,t+30min",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory. Default: run_dir/figures",
    )
    parser.add_argument("--prefix", default=None, help="Output filename prefix. Default: predictions_{split}")
    parser.add_argument("--format", dest="fmt", choices=["pdf", "png"], default="pdf", help="Export format.")
    parser.add_argument("--paper", choices=["a4", "ieee"], default="a4", help="Figure sizing preset.")
    parser.add_argument("--title", default=None, help="Optional figure title.")
    parser.add_argument("--show-residual", action="store_true", help="Overlay residual curve on a secondary y-axis.")
    args = parser.parse_args()

    run_dir = _resolve_path(args.run_dir)
    tz = args.tz if args.tz else None
    out_dir = _resolve_path(args.out_dir) if args.out_dir else (run_dir / "figures")
    prefix = args.prefix or f"predictions_{args.split}"

    if args.csv is None:
        csv_name = f"predictions_{args.split}.csv" if args.split in ("val", "test") else "predictions_all.csv"
        csv_path = run_dir / csv_name
    else:
        csv_path = _resolve_path(args.csv)
        if csv_path.is_dir():
            # allow passing run_dir as --csv by mistake
            csv_path = csv_path / f"predictions_{args.split}.csv"
        elif not csv_path.is_absolute():
            csv_path = run_dir / csv_path
    if not csv_path.exists():
        raise FileNotFoundError(f"Predictions CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    window_csv = _resolve_path(args.window_csv) if args.window_csv else None
    df = _ensure_ts_pred(df, tz=tz, window_csv=window_csv, split=args.split, val_ratio=args.val_ratio)
    df = df.sort_values("ts_pred").reset_index(drop=True) if "ts_pred" in df.columns else df
    df = _filter_time(df, args.start, args.end)
    if len(df) == 0:
        raise ValueError("No rows left after --start/--end filtering.")

    horizons = _detect_horizons(df)
    keep = _as_list(args.horizons)
    if keep is not None:
        horizons = [h for h in horizons if h.label in set(keep)]
        if not horizons:
            raise ValueError(f"--horizons filtered everything out. Available labels: {[h.label for h in _detect_horizons(df)]}")

    metrics_by_h = plot_predictions(
        df,
        horizons,
        out_dir=out_dir,
        prefix=prefix,
        fmt=args.fmt,
        paper=args.paper,
        title=args.title,
        show_residual=args.show_residual,
    )

    # Save metrics table for the selected time window.
    metrics_rows = []
    for h in horizons:
        m = metrics_by_h[h.label]
        metrics_rows.append(
            {
                "horizon": h.label,
                "rmse_W": m["rmse"],
                "mae_W": m["mae"],
                "r2": m["r2"],
                "n": int(m["n"]),
            }
        )
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = out_dir / f"{prefix}_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    print(f"Saved figures to: {out_dir}")
    print(f"Saved metrics to: {metrics_path}")


if __name__ == "__main__":
    main()

