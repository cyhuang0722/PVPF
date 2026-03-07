"""
推理脚本：加载已有 best_model.pt，重新生成 predictions 和 metrics。
用法:
  训练/验证数据: python -m pv_forecasting.infer [run_dir]
  测试数据:      python -m pv_forecasting.infer --test-pack-dir derived/test/packed [run_dir]
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from .model import PVForecastModel
from .dataset import ForecastDataset, PackedForecastDataset
from .preprocess import HORIZON


BASE_DIR = Path(__file__).resolve().parent
DERIVED_DIR = BASE_DIR / "derived"
MODEL_OUTPUT_DIR = BASE_DIR / "model_output"
CSV_PATH = DERIVED_DIR / "forecast_windows.csv"
PACK_DIR = DERIVED_DIR / "packed_forecast"
TEST_PACK_DIR = DERIVED_DIR / "test" / "packed"
TEST_CSV_PATH = DERIVED_DIR / "test" / "test_forecast_windows.csv"
SKY_MASK_PATH = BASE_DIR / "sky_mask.png"
IMG_SIZE = (128, 128)
PEAK_POWER_W = 66.3 * 1000.0
BATCH_SIZE = 32
VAL_RATIO = 0.2


def find_latest_run() -> Path:
    runs = sorted(MODEL_OUTPUT_DIR.glob("run_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not runs:
        raise FileNotFoundError(f"No run dir in {MODEL_OUTPUT_DIR}")
    return runs[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", nargs="?", help="e.g. pv_forecasting/model_output/run_xxx")
    parser.add_argument("--test-pack-dir", help="测试数据 packed 目录，如 derived/test/packed")
    parser.add_argument("--test-csv", help="测试数据 CSV（未打包时用）")
    parser.add_argument("--out", help="测试预测输出路径，默认 run_dir/predictions_test.csv")
    args = parser.parse_args()

    if args.run_dir:
        run_dir = Path(args.run_dir)
        if not run_dir.is_absolute():
            run_dir = (BASE_DIR.parent / run_dir) if str(run_dir).startswith("pv_forecasting") else (BASE_DIR / run_dir)
    else:
        run_dir = find_latest_run()
        print(f"Using latest run: {run_dir}")

    best_path = run_dir / "best_model.pt"
    if not best_path.exists():
        raise FileNotFoundError(f"best_model.pt not found: {best_path}")

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    peak = PEAK_POWER_W

    # 选择数据源
    if args.test_pack_dir:
        test_pack = Path(args.test_pack_dir)
        if not test_pack.is_absolute():
            test_pack = BASE_DIR / test_pack
        if not test_pack.exists() or not list(test_pack.glob("batch_*.npz")):
            raise FileNotFoundError(f"Test pack not found: {test_pack}")
        dataset = PackedForecastDataset(test_pack)
        is_test = True
    elif args.test_csv:
        test_csv = Path(args.test_csv)
        if not test_csv.is_absolute():
            test_csv = BASE_DIR / test_csv
        mask_path = SKY_MASK_PATH if SKY_MASK_PATH.exists() else None
        dataset = ForecastDataset(test_csv, img_size=IMG_SIZE, base_dir=BASE_DIR.parent, sky_mask_path=mask_path)
        is_test = True
    elif PACK_DIR.exists() and list(PACK_DIR.glob("batch_*.npz")):
        dataset = PackedForecastDataset(PACK_DIR)
        is_test = False
    elif CSV_PATH.exists():
        mask_path = SKY_MASK_PATH if SKY_MASK_PATH.exists() else None
        dataset = ForecastDataset(CSV_PATH, img_size=IMG_SIZE, base_dir=BASE_DIR.parent, sky_mask_path=mask_path)
        is_test = False
    else:
        raise FileNotFoundError(f"Need {PACK_DIR}, {CSV_PATH}, or --test-pack-dir/--test-csv")

    model = PVForecastModel().to(device)
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()

    def run_predict(ds, base_dataset, has_targets=True):
        preds_norm, targets_raw = [], []
        with torch.no_grad():
            for batch in DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False):
                pv_past_norm = batch["pv_past"].to(device) / peak
                out = model(batch["sky"].to(device), pv_past_norm)
                preds_norm.append(out.cpu().numpy())
                if has_targets:
                    targets_raw.append(batch["targets"].numpy())
        pred_norm = np.concatenate(preds_norm, axis=0)
        pred_w = pred_norm * peak
        if isinstance(ds, Subset):
            ts = base_dataset.ts_pred_series.iloc[list(ds.indices)]
        else:
            ts = base_dataset.ts_pred_series
        true_w = np.concatenate(targets_raw, axis=0) if has_targets else None
        return pred_w, true_w, pd.to_datetime(ts)

    if is_test:
        pred_w, _, ts = run_predict(dataset, dataset, has_targets=True)
        out_path = Path(args.out) if args.out else (run_dir / "predictions_test.csv")
        out_path = out_path if out_path.is_absolute() else (run_dir / out_path)
        pred_df = pd.DataFrame({f"pv_pred_W_{k}": pred_w[:, k] for k in range(HORIZON)})
        pred_df["ts_pred"] = ts
        pred_df.to_csv(out_path, index=False)
        print(f"Test predictions saved: {out_path} ({len(pred_df)} samples)")
        return

    n_val = max(1, int(len(dataset) * VAL_RATIO))
    n_train = len(dataset) - n_val
    val_ds = Subset(dataset, range(n_train, len(dataset)))

    def compute_metrics(pred_w, true_w):
        err = pred_w - true_w
        row = {"mae_W": float(np.abs(err).mean()), "rmse_W": float(np.sqrt((err ** 2).mean()))}
        for k in range(HORIZON):
            row[f"mae_W_h{k}"] = float(np.abs(err[:, k]).mean())
            row[f"rmse_W_h{k}"] = float(np.sqrt((err[:, k] ** 2).mean()))
        return row

    pred_all_w, true_all_w, ts_all = run_predict(dataset, dataset)
    pred_val_w, true_val_w, ts_val = run_predict(val_ds, dataset)

    pred_df_all = pd.DataFrame({f"pv_pred_W_{k}": pred_all_w[:, k] for k in range(HORIZON)})
    pred_df_all["ts_pred"] = ts_all
    for k in range(HORIZON):
        pred_df_all[f"pv_true_W_{k}"] = true_all_w[:, k]
    pred_df_all.to_csv(run_dir / "predictions_all.csv", index=False)

    pred_df_val = pd.DataFrame({f"pv_pred_W_{k}": pred_val_w[:, k] for k in range(HORIZON)})
    pred_df_val["ts_pred"] = ts_val
    for k in range(HORIZON):
        pred_df_val[f"pv_true_W_{k}"] = true_val_w[:, k]
    pred_df_val.to_csv(run_dir / "predictions_val.csv", index=False)

    metrics_val = compute_metrics(pred_val_w, true_val_w)
    pd.DataFrame([metrics_val]).to_csv(run_dir / "metrics_val.csv", index=False)
    metrics_all = compute_metrics(pred_all_w, true_all_w)
    pd.DataFrame([metrics_all]).to_csv(run_dir / "metrics_all.csv", index=False)

    print(f"Regenerated in {run_dir}:")
    print(f"  Val  MAE_W={metrics_val['mae_W']:.2f}, RMSE_W={metrics_val['rmse_W']:.2f}")
    for k in range(HORIZON):
        print(f"    h{k} MAE_W={metrics_val[f'mae_W_h{k}']:.2f}, RMSE_W={metrics_val[f'rmse_W_h{k}']:.2f}")
    print(f"  All  MAE_W={metrics_all['mae_W']:.2f} (reference)")


if __name__ == "__main__":
    main()
