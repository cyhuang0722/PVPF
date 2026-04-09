"""
PV-only preprocessing:
- build windows with input [t-120, t-105, ..., t-15]
- target at t+15
- save intermediate tabular samples for training.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_PV_CSV = BASE_DIR.parent.parent / "data" / "power" / "power-LSK_N_1year_2025_to_2026.csv"
DERIVED_DIR = BASE_DIR / "derived"
DEFAULT_OUT_CSV = DERIVED_DIR / "forecast_windows.csv"
DEFAULT_SUMMARY_JSON = DERIVED_DIR / "preprocess_summary.json"
TZ = "Asia/Singapore"

# Input points: t-120..t-15, every 15 minutes.
PAST_OFFSETS_MIN = [120, 105, 90, 75, 60, 45, 30, 15]
TARGET_OFFSET_MIN = 15


def load_pv_15min(pv_csv: Path, tz: str) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(pv_csv)
    total_rows = len(df)
    if {"datetime", "power"}.issubset(df.columns):
        ts_col, power_col = "datetime", "power"
    elif {"date", "value"}.issubset(df.columns):
        ts_col, power_col = "date", "value"
    else:
        raise ValueError("PV csv must contain either (datetime, power) or (date, value)")

    df["ts_power"] = pd.to_datetime(df[ts_col], errors="coerce")
    df["power"] = pd.to_numeric(df[power_col], errors="coerce")
    invalid_ts_or_nan_power = int(df["ts_power"].isna().sum() + df["power"].isna().sum())
    df = df.dropna(subset=["ts_power", "power"]).copy()
    non_positive_count = int((df["power"] <= 0).sum())
    df = df[df["power"] > 0].copy()
    if df["ts_power"].dt.tz is None:
        df["ts_power"] = df["ts_power"].dt.tz_localize(tz)
    else:
        df["ts_power"] = df["ts_power"].dt.tz_convert(tz)
    df["ts_power"] = df["ts_power"].dt.round("15min")
    df = df.groupby("ts_power", as_index=False)["power"].mean()
    # Safety filter after aggregation, in case duplicates average to non-positive values.
    after_group_non_positive = int((df["power"] <= 0).sum())
    df = df[df["power"] > 0].copy()
    df = df.sort_values("ts_power").reset_index(drop=True)
    stats = {
        "total_rows_raw": int(total_rows),
        "invalid_ts_or_nan_power": int(invalid_ts_or_nan_power),
        "non_positive_power_before_group": int(non_positive_count),
        "non_positive_power_after_group": int(after_group_non_positive),
        "rows_after_filter": int(len(df)),
    }
    return df, stats


def build_windows(df_pv: pd.DataFrame) -> pd.DataFrame:
    if df_pv.empty:
        return pd.DataFrame(columns=["ts_anchor", "ts_target", "past_pv", "target"])

    pv_series = df_pv.set_index("ts_power")["power"].astype(float).sort_index()
    pv_index_set = set(pv_series.index)
    samples: list[dict] = []
    filtered_missing_context = 0

    for t in pv_series.index:
        required_past = [t - pd.Timedelta(minutes=m) for m in PAST_OFFSETS_MIN]
        ts_target = t + pd.Timedelta(minutes=TARGET_OFFSET_MIN)
        required_all = required_past + [ts_target]
        if not all(ts in pv_index_set for ts in required_all):
            filtered_missing_context += 1
            continue

        samples.append(
            {
                "ts_anchor": t,
                "ts_target": ts_target,
                "past_pv": [float(pv_series.loc[ts]) for ts in required_past],
                "target": float(pv_series.loc[ts_target]),
            }
        )

    out = pd.DataFrame(samples)
    if out.empty:
        return out
    out = out.sort_values("ts_anchor").reset_index(drop=True)
    print(
        f"[build_windows] built={len(out)}, "
        f"filtered_missing_context={filtered_missing_context}"
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pv-csv", default=str(DEFAULT_PV_CSV), help="input PV CSV")
    parser.add_argument("--out-csv", default=str(DEFAULT_OUT_CSV), help="output forecast windows CSV")
    parser.add_argument("--summary-json", default=str(DEFAULT_SUMMARY_JSON), help="preprocess summary json")
    args = parser.parse_args()

    pv_csv = Path(args.pv_csv)
    out_csv = Path(args.out_csv)
    summary_json = Path(args.summary_json)
    if not pv_csv.is_absolute():
        pv_csv = (BASE_DIR.parent.parent / pv_csv) if str(pv_csv).startswith(("data/", "pv_forecasting/")) else (BASE_DIR / pv_csv)
    if not out_csv.is_absolute():
        out_csv = (BASE_DIR.parent.parent / out_csv) if str(out_csv).startswith("pv_forecasting/") else (BASE_DIR / out_csv)
    if not summary_json.is_absolute():
        summary_json = (BASE_DIR.parent.parent / summary_json) if str(summary_json).startswith("pv_forecasting/") else (BASE_DIR / summary_json)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)

    print(f"[1] Load PV: {pv_csv}")
    df_pv, pv_filter_stats = load_pv_15min(pv_csv, TZ)
    if df_pv.empty:
        raise RuntimeError("No valid PV rows after parsing.")
    print(
        "  filter stats: "
        f"raw={pv_filter_stats['total_rows_raw']}, "
        f"invalid_ts_or_nan={pv_filter_stats['invalid_ts_or_nan_power']}, "
        f"non_positive_before_group={pv_filter_stats['non_positive_power_before_group']}, "
        f"non_positive_after_group={pv_filter_stats['non_positive_power_after_group']}, "
        f"kept={pv_filter_stats['rows_after_filter']}"
    )
    print(f"  rows={len(df_pv)} | {df_pv['ts_power'].min()} -> {df_pv['ts_power'].max()}")

    print("[2] Build PV-only windows...")
    df_win = build_windows(df_pv)
    if df_win.empty:
        raise RuntimeError("No samples built. Check timestamp continuity in PV data.")

    print(f"[3] Save windows CSV: {out_csv}")
    df_win.to_csv(out_csv, index=False)
    print(f"  saved rows={len(df_win)}")

    summary = {
        "timestamp": datetime.now().isoformat(),
        "pv_csv": str(pv_csv),
        "out_csv": str(out_csv),
        "tz": TZ,
        "past_offsets_min": PAST_OFFSETS_MIN,
        "target_offset_min": TARGET_OFFSET_MIN,
        "num_rows_pv": int(len(df_pv)),
        "pv_filter_stats": pv_filter_stats,
        "num_samples": int(len(df_win)),
        "range_anchor_start": str(df_win["ts_anchor"].iloc[0]),
        "range_anchor_end": str(df_win["ts_anchor"].iloc[-1]),
    }
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[4] Save preprocess summary: {summary_json}")


if __name__ == "__main__":
    main()

