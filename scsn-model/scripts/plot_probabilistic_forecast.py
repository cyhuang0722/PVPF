from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

import plotly.graph_objects as go


def _parse_timestamp(value: str) -> datetime:
    value = value.strip()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    return datetime.fromisoformat(value)


def _load_split_csv(path: Path, split: str) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "split": split,
                    "ts_target": _parse_timestamp(row["ts_target"]),
                    "target_pv_w": float(row["target_pv_w"]),
                    "q10_w": float(row["q10_w"]),
                    "q50_w": float(row["q50_w"]),
                    "q90_w": float(row["q90_w"]),
                    "target_value": float(row.get("target_value", 0.0)),
                    "q10": float(row.get("q10", 0.0)),
                    "q50": float(row.get("q50", 0.0)),
                    "q90": float(row.get("q90", 0.0)),
                }
            )
    rows.sort(key=lambda item: item["ts_target"])
    return rows


def _build_hover_text(row: dict) -> str:
    return (
        f"split={row['split']}<br>"
        f"time={row['ts_target']}<br>"
        f"true={row['target_pv_w']:.1f} W<br>"
        f"q10={row['q10_w']:.1f} W ({row['q10']:.3f})<br>"
        f"q50={row['q50_w']:.1f} W ({row['q50']:.3f})<br>"
        f"q90={row['q90_w']:.1f} W ({row['q90']:.3f})"
    )


def _add_split_traces(fig: go.Figure, rows: list[dict], split: str, color: str, show_legend: bool) -> None:
    x = [row["ts_target"] for row in rows]
    q10 = [row["q10_w"] for row in rows]
    q50 = [row["q50_w"] for row in rows]
    q90 = [row["q90_w"] for row in rows]
    target = [row["target_pv_w"] for row in rows]
    hover = [_build_hover_text(row) for row in rows]

    fig.add_trace(
        go.Scatter(
            x=x,
            y=q90,
            mode="lines",
            line={"width": 0},
            hoverinfo="skip",
            showlegend=False,
            legendgroup=split,
            name=f"{split} q90 upper",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=q10,
            mode="lines",
            line={"width": 0},
            fill="tonexty",
            fillcolor=color.replace("rgb", "rgba").replace(")", ", 0.16)"),
            hoverinfo="skip",
            showlegend=show_legend,
            legendgroup=split,
            name=f"{split} q10-q90",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=q10,
            mode="lines",
            line={"color": color, "dash": "dot", "width": 1.2},
            text=hover,
            hovertemplate="%{text}<extra></extra>",
            showlegend=show_legend,
            legendgroup=split,
            name=f"{split} q10",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=q50,
            mode="lines",
            line={"color": color, "width": 2.4},
            text=hover,
            hovertemplate="%{text}<extra></extra>",
            showlegend=show_legend,
            legendgroup=split,
            name=f"{split} q50",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=q90,
            mode="lines",
            line={"color": color, "dash": "dot", "width": 1.2},
            text=hover,
            hovertemplate="%{text}<extra></extra>",
            showlegend=show_legend,
            legendgroup=split,
            name=f"{split} q90",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=target,
            mode="lines",
            line={"color": "black", "width": 1.4},
            text=hover,
            hovertemplate="%{text}<extra></extra>",
            showlegend=show_legend,
            legendgroup=split,
            name=f"{split} true",
        )
    )


def build_figure(run_dir: Path) -> tuple[go.Figure, Path]:
    split_rows = {
        split: _load_split_csv(run_dir / f"predictions_{split}.csv", split)
        for split in ("train", "val", "test")
    }
    colors = {
        "train": "rgb(31, 119, 180)",
        "val": "rgb(255, 127, 14)",
        "test": "rgb(44, 160, 44)",
    }

    fig = go.Figure()
    for idx, split in enumerate(("train", "val", "test")):
        _add_split_traces(
            fig=fig,
            rows=split_rows[split],
            split=split,
            color=colors[split],
            show_legend=True,
        )

    split_boundaries: list[tuple[str, datetime]] = []
    if split_rows["train"] and split_rows["val"]:
        split_boundaries.append(("train / val", split_rows["val"][0]["ts_target"]))
    if split_rows["val"] and split_rows["test"]:
        split_boundaries.append(("val / test", split_rows["test"][0]["ts_target"]))

    for label, boundary in split_boundaries:
        fig.add_vline(x=boundary, line_width=2, line_dash="dash", line_color="rgba(80,80,80,0.8)")
        fig.add_annotation(
            x=boundary,
            y=1.02,
            yref="paper",
            text=label,
            showarrow=False,
            font={"size": 12, "color": "rgba(70,70,70,1)"},
            bgcolor="rgba(255,255,255,0.8)",
        )

    fig.update_layout(
        title=f"SCSN Probabilistic Forecast: {run_dir.name}",
        xaxis_title="Target Time",
        yaxis_title="PV Power (W)",
        hovermode="x unified",
        template="plotly_white",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.08, "xanchor": "left", "x": 0.0},
        margin={"l": 60, "r": 30, "t": 90, "b": 60},
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.08)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.08)")

    out_path = run_dir / "figures" / "probabilistic_forecast_all_splits.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return fig, out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot probabilistic SCSN forecasts for train/val/test in one Plotly figure")
    parser.add_argument("--run-dir", required=True, help="Run directory containing predictions_train/val/test.csv")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    fig, out_path = build_figure(run_dir)
    fig.write_html(str(out_path), include_plotlyjs=True)
    print(out_path)


if __name__ == "__main__":
    main()
