from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .satellite_common import load_json_config
from .satellite_dataset import PackedSatelliteDataset, SatelliteForecastDataset
from .satellite_model import SatelliteOnlyForecaster


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
MODEL_OUTPUT_DIR = BASE_DIR / "model_output"


def find_latest_run() -> Path:
    runs = sorted(MODEL_OUTPUT_DIR.glob("run_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not runs:
        raise FileNotFoundError(f"No run dirs found in {MODEL_OUTPUT_DIR}")
    return runs[0]


def compute_metrics(pred_w: np.ndarray, true_w: np.ndarray, horizon_names: list[str]) -> dict[str, float]:
    err = pred_w - true_w
    out: dict[str, float] = {}
    rmse_list = []
    for idx, horizon in enumerate(horizon_names):
        out[f"mae_{horizon}_W"] = float(np.abs(err[:, idx]).mean())
        rmse = float(np.sqrt(np.mean(err[:, idx] ** 2)))
        out[f"rmse_{horizon}_W"] = rmse
        rmse_list.append(rmse)
    out["overall_mae_W"] = float(np.abs(err).mean())
    out["overall_rmse_W"] = float(np.sqrt(np.mean(err ** 2)))
    out["mean_rmse_W"] = float(np.mean(rmse_list))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="JSON config path")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    args = parser.parse_args()

    cfg = load_json_config(args.config)
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]
    run_dir = Path(args.run_dir) if args.run_dir else find_latest_run()
    if not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir

    best_path = run_dir / "best_model.pt"
    if not best_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {best_path}")

    pack_dir = PROJECT_ROOT / data_cfg["out_dir"] / "packed_satellite"
    csv_path = PROJECT_ROOT / data_cfg["out_dir"] / "forecast_windows.csv"
    stats_path = PROJECT_ROOT / data_cfg["out_dir"] / data_cfg.get("stats_filename", "satellite_stats.json")

    if pack_dir.exists() and list(pack_dir.glob("batch_*.npz")):
        ds = PackedSatelliteDataset(pack_dir, split=args.split)
    else:
        ds = SatelliteForecastDataset(csv_path, split=args.split, stats_path=stats_path, peak_power_w=float(data_cfg["peak_power_w"]))

    model = SatelliteOnlyForecaster(
        in_channels=len(data_cfg["channels"]),
        encoder_channels=tuple(model_cfg.get("encoder_channels", [32, 64, 128])),
        convlstm_hidden_dim=int(model_cfg.get("convlstm_hidden_dim", 128)),
        convlstm_layers=int(model_cfg.get("convlstm_layers", 1)),
        convlstm_kernel_size=int(model_cfg.get("convlstm_kernel_size", 3)),
        head_hidden_dim=int(model_cfg.get("head_hidden_dim", 128)),
        dropout=float(model_cfg.get("dropout", 0.2)),
        out_dim=len(data_cfg["future_offsets_min"]),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.to(device)
    model.eval()

    loader = DataLoader(ds, batch_size=int(train_cfg["batch_size"]), shuffle=False, num_workers=int(train_cfg.get("num_workers", 0)))
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            out = model(batch["satellite"].to(device))
            preds.append(out.cpu().numpy())
            targets.append(batch["targets"].numpy())
    pred_norm = np.concatenate(preds, axis=0)
    true_norm = np.concatenate(targets, axis=0)
    peak_power_w = float(data_cfg["peak_power_w"])
    pred_w = pred_norm * peak_power_w
    true_w = true_norm * peak_power_w
    horizon_names = [f"t+{m}" for m in data_cfg["future_offsets_min"]]
    metrics = compute_metrics(pred_w, true_w, horizon_names)

    pred_df = pd.DataFrame({"ts_pred": pd.to_datetime(ds.ts_pred_series).reset_index(drop=True)})
    for idx, horizon in enumerate(horizon_names):
        pred_df[f"pv_pred_{horizon}_W"] = pred_w[:, idx]
        pred_df[f"pv_true_{horizon}_W"] = true_w[:, idx]
    pred_df.to_csv(run_dir / f"predictions_{args.split}.csv", index=False)
    (run_dir / f"metrics_{args.split}.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
