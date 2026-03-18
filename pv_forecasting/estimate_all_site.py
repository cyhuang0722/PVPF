from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


SITE_A_ID = "2245380"
BASE_DIR = Path(__file__).resolve().parent.parent
PEAK_POWER_PATH = BASE_DIR / "data" / "power" / "peakPower.csv"
ALL_SITES_TRUE_PATH = BASE_DIR / "data" / "power" / "all_sites_power_test.csv"
PREDICTIONS_PATH = (
    BASE_DIR
    / "pv_forecasting"
    / "model_output"
    / "run_20260308-004432"
    / "predictions_test.csv"
)


def rmse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def _load_peak_power(peak_power_path: Path) -> pd.Series:
    peak_df = pd.read_csv(peak_power_path, dtype={"SiteID": str})
    peak_df['PeakPower'] = peak_df['PeakPower']*1000
    if "PeakPower" not in peak_df.columns:
        raise ValueError(f"Column 'PeakPower' not found in {peak_power_path}")
    if SITE_A_ID not in set(peak_df["SiteID"]):
        raise ValueError(f"Site A ({SITE_A_ID}) not found in {peak_power_path}")
    return peak_df.set_index("SiteID")["PeakPower"].astype(float)


def _load_all_sites_true(all_sites_true_path: Path) -> pd.DataFrame:
    df = pd.read_csv(all_sites_true_path, dtype={"date": str})
    if "date" not in df.columns:
        raise ValueError(f"Column 'date' not found in {all_sites_true_path}")
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")


def _load_predictions(predictions_path: Path) -> pd.DataFrame:
    pred_df = pd.read_csv(predictions_path)
    if "ts_pred" not in pred_df.columns:
        raise ValueError(f"Column 'ts_pred' not found in {predictions_path}")
    pred_df["ts_pred"] = (
        pd.to_datetime(pred_df["ts_pred"], utc=True)
        .dt.tz_convert("Asia/Hong_Kong")
        .dt.tz_localize(None)
    )
    return pred_df


def _prepare_common_site_columns(all_sites_df: pd.DataFrame, peak_power: pd.Series) -> list[str]:
    common = [sid for sid in all_sites_df.columns if sid in peak_power.index]
    if not common:
        raise ValueError(
            "No common site IDs between all_sites_power_test.csv and peakPower.csv"
        )
    return common


def calc_all_site_accuracy() -> pd.DataFrame:
    peak_power = _load_peak_power(PEAK_POWER_PATH)
    all_sites_true = _load_all_sites_true(ALL_SITES_TRUE_PATH)
    pred_df = _load_predictions(PREDICTIONS_PATH)

    if SITE_A_ID not in all_sites_true.columns:
        raise ValueError(f"Site A ({SITE_A_ID}) not found in {ALL_SITES_TRUE_PATH}")

    site_cols = _prepare_common_site_columns(all_sites_true, peak_power)
    peak_a = float(peak_power[SITE_A_ID])
    peak_by_site = peak_power.reindex(site_cols).astype(float)

    rows: list[dict] = []

    for h in range(4):
        pred_col = f"pv_pred_W_{h}"
        if pred_col not in pred_df.columns:
            raise ValueError(f"Column '{pred_col}' not found in {PREDICTIONS_PATH}")

        step_minutes = 15 * (h + 1)
        target_ts = pred_df["ts_pred"] + pd.to_timedelta(step_minutes, unit="m")

        # Assumption: all sites are proportional to peak power.
        # Therefore the predicted normalized fleet output equals the normalized Site-A prediction.
        pred_norm_all = pred_df[pred_col].astype(float).to_numpy() / peak_a

        aligned_true = all_sites_true.reindex(target_ts)[site_cols].astype(float)

        # Faulty / abnormal sites must be excluded from both numerator and denominator.
        # Abnormal means power <= 0 or NaN at that timestamp.
        valid_mask = aligned_true.notna() & (aligned_true > 0.0)

        valid_peak = pd.DataFrame(
            np.broadcast_to(peak_by_site.to_numpy(), aligned_true.shape),
            index=aligned_true.index,
            columns=aligned_true.columns,
        )

        true_all_sum = aligned_true.where(valid_mask, 0.0).sum(axis=1).to_numpy(dtype=float)
        peak_all_valid_sum = valid_peak.where(valid_mask, 0.0).sum(axis=1).to_numpy(dtype=float)

        true_norm_all = np.full(len(aligned_true), np.nan, dtype=float)
        denom_ok = peak_all_valid_sum > 0.0
        true_norm_all[denom_ok] = true_all_sum[denom_ok] / peak_all_valid_sum[denom_ok]

        eval_mask = np.isfinite(pred_norm_all) & np.isfinite(true_norm_all)
        if not np.any(eval_mask):
            raise ValueError(f"No valid samples for horizon h{h}")

        horizon_rmse = rmse(pred_norm_all[eval_mask], true_norm_all[eval_mask])
        horizon_accuracy = 1.0 - horizon_rmse

        rows.append(
            {
                "horizon": f"h{h}",
                "samples": int(eval_mask.sum()),
                "rmse_normed_all": horizon_rmse,
                "accuracy_normed_all": horizon_accuracy,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    result_df = calc_all_site_accuracy()

    out_dir = PREDICTIONS_PATH.parent
    out_path = out_dir / "metrics_normed_all_4h.csv"
    result_df.to_csv(out_path, index=False)

    print(f"Saved: {out_path}")
    print()
    print(result_df.to_string(index=False))


if __name__ == "__main__":
    main()
