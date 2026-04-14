"""
Minimal interactive Plotly viewer for prediction CSV.

Usage example:
python -m pv_forecasting.plot_predictions_interactive \
  --csv pv_forecasting/pv_only/model_output/run_xxx/predictions_val.csv \
  --pred-col-idx 2 \
  --true-col-idx 3 \
  --start "2026-03-01 08:00:00" \
  --end "2026-03-01 18:00:00"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

BASE_DIR = Path(__file__).resolve().parent
# CSV_PATH = BASE_DIR / "pv_only" / "model_output" / "run_20260408-092225_pv_only" / "predictions_val.csv"
# CSV_PATH = Path("/Users/huangchouyue/Projects/PVPF/new-model/artifacts/runs/run_20260413-170933/predictions_test.csv")
CSV_PATH = Path("/Users/huangchouyue/Projects/PVPF/SPM/run_20260413-235432_spm/predictions_test.csv")
start_time = "2025-03-01 00:00:00"
end_time = "2027-04-01 23:59:59"
pred_col_idx = 10
true_col_idx = 3
time_col = "ts_pred"
out_html = CSV_PATH.parent / "forecast_plot_test.html"


def align_bound_to_series_tz(bound: pd.Timestamp, ts_series: pd.Series) -> pd.Timestamp:
    """
    Align start/end bound timezone with the datetime series timezone.
    """
    series_tz = ts_series.dt.tz
    if series_tz is None:
        # Series is tz-naive: remove tz info from bound if needed.
        return bound.tz_localize(None) if bound.tzinfo is not None else bound
    # Series is tz-aware: localize or convert bound.
    if bound.tzinfo is None:
        return bound.tz_localize(series_tz)
    return bound.tz_convert(series_tz)

def auto_pick_time_col(df: pd.DataFrame) -> str:
    for c in ("ts_target", "ts_pred", "ts_anchor", "datetime", "date", "time", "timestamp"):
        if c in df.columns:
            return c
    # fallback: first parseable datetime-like column
    for c in df.columns:
        parsed = pd.to_datetime(df[c], errors="coerce")
        if parsed.notna().mean() > 0.8:
            return c
    raise ValueError("No time column found. Please pass --time-col explicitly.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=CSV_PATH, help="prediction csv path")
    parser.add_argument("--time-col", default=None, help="time column name (optional)")
    parser.add_argument("--pred-col-idx", type=int, default=pred_col_idx, help="prediction column index")
    parser.add_argument("--true-col-idx", type=int, default=true_col_idx, help="ground-truth column index")
    parser.add_argument("--start", default=start_time, help='start time, e.g. "2026-03-01 08:00:00"')
    parser.add_argument("--end", default=end_time, help='end time, e.g. "2026-03-01 18:00:00"')
    parser.add_argument("--title", default="Prediction vs True", help="plot title")
    parser.add_argument("--out-html", default=out_html, help="optional html output path")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        csv_path = Path.cwd() / csv_path
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    cols = df.columns.tolist()
    if args.pred_col_idx < 0 or args.pred_col_idx >= len(cols):
        raise IndexError(f"pred-col-idx out of range: {args.pred_col_idx}, n_cols={len(cols)}")
    if args.true_col_idx < 0 or args.true_col_idx >= len(cols):
        raise IndexError(f"true-col-idx out of range: {args.true_col_idx}, n_cols={len(cols)}")

    time_col = args.time_col if args.time_col else auto_pick_time_col(df)
    if time_col not in df.columns:
        raise ValueError(f"time column not found: {time_col}")

    pred_col = cols[args.pred_col_idx]
    true_col = cols[args.true_col_idx]

    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col]).sort_values(time_col)

    if args.start:
        start = align_bound_to_series_tz(pd.to_datetime(args.start), df[time_col])
        df = df[df[time_col] >= start]
    if args.end:
        end = align_bound_to_series_tz(pd.to_datetime(args.end), df[time_col])
        df = df[df[time_col] <= end]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[time_col], y=df[pred_col], mode="lines", name=f"Pred: {pred_col}"))
    fig.add_trace(go.Scatter(x=df[time_col], y=df[true_col], mode="lines", name=f"True: {true_col}"))
    fig.update_layout(
        title=args.title,
        xaxis_title=time_col,
        yaxis_title="Power (W)",
        hovermode="x unified",
        template="plotly_white",
    )

    if args.out_html:
        out_html.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(out_html), include_plotlyjs="cdn")
        print(f"Saved html: {out_html}")

    print("Columns:")
    for i, c in enumerate(cols):
        print(f"  [{i}] {c}")
    print(f"Using time_col={time_col}, pred_col={pred_col}, true_col={true_col}, n={len(df)}")
    fig.show()


if __name__ == "__main__":
    main()

