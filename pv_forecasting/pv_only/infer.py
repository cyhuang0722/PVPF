"""
Inference script for PV-only LSTM runs.
Re-generates predictions and metrics from a trained run directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from .dataset import PVPastDataset
from .model import PVPastLSTM


BASE_DIR = Path(__file__).resolve().parent
DERIVED_DIR = BASE_DIR / "derived"
MODEL_OUTPUT_DIR = BASE_DIR / "model_output"
CSV_PATH = DERIVED_DIR / "forecast_windows.csv"
PEAK_POWER_W = 66300.0
VAL_RATIO = 0.2
BATCH_SIZE = 64


def find_latest_run() -> Path:
    runs = sorted(MODEL_OUTPUT_DIR.glob("run_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not runs:
        raise FileNotFoundError(f"No run dir in {MODEL_OUTPUT_DIR}")
    return runs[0]


def load_model_cfg(run_dir: Path) -> dict:
    meta_path = run_dir / "run_metadata.json"
    if not meta_path.exists():
        return {}
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return meta.get("cfg", {}).get("MODEL_CFG", {})


def compute_metrics(pred_w: np.ndarray, true_w: np.ndarray) -> dict:
    err = pred_w - true_w
    mae_w = float(np.abs(err).mean())
    rmse_w = float(np.sqrt((err ** 2).mean()))
    return {
        "mae_W": mae_w,
        "rmse_W": rmse_w,
        "mae_norm_peak": mae_w / PEAK_POWER_W,
        "rmse_norm_peak": rmse_w / PEAK_POWER_W,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", nargs="?", help="e.g. pv_forecasting/pv_only/model_output/run_xxx")
    parser.add_argument("--csv", default=str(CSV_PATH), help="forecast windows csv path")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="batch size")
    parser.add_argument("--val-ratio", type=float, default=VAL_RATIO, help="val split ratio")
    args = parser.parse_args()

    if args.run_dir:
        run_dir = Path(args.run_dir)
        if not run_dir.is_absolute():
            run_dir = (BASE_DIR.parent.parent / run_dir) if str(run_dir).startswith("pv_forecasting/") else (BASE_DIR / run_dir)
    else:
        run_dir = find_latest_run()
        print(f"Using latest run: {run_dir}")

    best_path = run_dir / "best_model.pt"
    if not best_path.exists():
        raise FileNotFoundError(f"best_model.pt not found: {best_path}")

    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        csv_path = (BASE_DIR.parent.parent / csv_path) if str(csv_path).startswith("pv_forecasting/") else (BASE_DIR / csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Device: GPU ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        print("Device: CPU (no GPU detected)")

    dataset = PVPastDataset(csv_path)
    n_val = max(1, int(len(dataset) * args.val_ratio))
    n_train = len(dataset) - n_val
    val_ds = Subset(dataset, range(n_train, len(dataset)))

    model_cfg = load_model_cfg(run_dir)
    model = PVPastLSTM(
        input_dim=model_cfg.get("input_dim", 1),
        hidden_dim=model_cfg.get("hidden_dim", 64),
        num_layers=model_cfg.get("num_layers", 2),
        dropout=model_cfg.get("dropout", 0.2),
    ).to(device)
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()

    def run_predict(ds, base_dataset):
        preds_norm, targets_raw = [], []
        with torch.no_grad():
            for batch in DataLoader(ds, batch_size=args.batch_size, shuffle=False):
                x = batch["pv_past"].to(device) / PEAK_POWER_W
                out = model(x)
                preds_norm.append(out.cpu().numpy().reshape(-1))
                targets_raw.append(batch["target"].numpy().reshape(-1))
        pred_w = np.concatenate(preds_norm, axis=0) * PEAK_POWER_W
        true_w = np.concatenate(targets_raw, axis=0)
        if isinstance(ds, Subset):
            idx = list(ds.indices)
            ts_anchor = base_dataset.ts_anchor_series.iloc[idx]
            ts_target = base_dataset.ts_target_series.iloc[idx]
        else:
            ts_anchor = base_dataset.ts_anchor_series
            ts_target = base_dataset.ts_target_series
        return pred_w, true_w, pd.to_datetime(ts_anchor), pd.to_datetime(ts_target)

    pred_all_w, true_all_w, ts_anchor_all, ts_target_all = run_predict(dataset, dataset)
    pred_val_w, true_val_w, ts_anchor_val, ts_target_val = run_predict(val_ds, dataset)

    pred_all_df = pd.DataFrame(
        {
            "ts_anchor": ts_anchor_all,
            "ts_target": ts_target_all,
            "pv_pred_W_t_plus_15": pred_all_w,
            "pv_true_W_t_plus_15": true_all_w,
        }
    )
    pred_all_df.to_csv(run_dir / "predictions_all.csv", index=False)

    pred_val_df = pd.DataFrame(
        {
            "ts_anchor": ts_anchor_val,
            "ts_target": ts_target_val,
            "pv_pred_W_t_plus_15": pred_val_w,
            "pv_true_W_t_plus_15": true_val_w,
        }
    )
    pred_val_df.to_csv(run_dir / "predictions_val.csv", index=False)

    metrics_val = compute_metrics(pred_val_w, true_val_w)
    metrics_all = compute_metrics(pred_all_w, true_all_w)
    pd.DataFrame([metrics_val]).to_csv(run_dir / "metrics_val.csv", index=False)
    pd.DataFrame([metrics_all]).to_csv(run_dir / "metrics_all.csv", index=False)

    print(f"Regenerated in {run_dir}:")
    print(f"  Val  MAE_W={metrics_val['mae_W']:.2f}, RMSE_W={metrics_val['rmse_W']:.2f}")
    print(f"  All  MAE_W={metrics_all['mae_W']:.2f}, RMSE_W={metrics_all['rmse_W']:.2f}")


if __name__ == "__main__":
    main()

