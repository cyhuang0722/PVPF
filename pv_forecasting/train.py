"""
训练 Sky+PV 双分支预测模型，结果保存到 model_output/run_YYYYMMDD-HHMMSS。
"""
import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from .model import PVForecastModel
from .dataset import ForecastDataset, PackedForecastDataset
from .preprocess import HORIZON


# ---------- config ----------
BASE_DIR = Path(__file__).resolve().parent
DERIVED_DIR = BASE_DIR / "derived"
MODEL_OUTPUT_DIR = BASE_DIR / "model_output"
CSV_PATH = DERIVED_DIR / "forecast_windows.csv"
PACK_DIR = DERIVED_DIR / "packed_forecast"
SKY_MASK_PATH = BASE_DIR / "sky_mask.png"

IMG_SIZE = (128, 128)
PEAK_POWER_W = 66.3 * 1000.0
BATCH_SIZE = 32
EPOCHS = 100
LR = 1e-3
VAL_RATIO = 0.2
RANDOM_SEED = 42
SKY_EMBED = 128
PV_HIDDEN = 64
FUSION_HIDDEN = 128
DROPOUT = 0.2


def make_run_dir(run_name: str | None = None) -> Path:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    suffix = f"_{run_name}" if run_name else ""
    run_dir = MODEL_OUTPUT_DIR / f"run_{ts}{suffix}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def setup_logging(run_dir: Path):
    log_path = run_dir / "train.log"
    for h in list(logging.getLogger().handlers):
        logging.getLogger().removeHandler(h)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger().addHandler(sh)
    logging.getLogger().addHandler(fh)


def dump_metadata(run_dir: Path, cfg: dict):
    meta = {"timestamp": datetime.now().isoformat(), "cfg": cfg}
    (run_dir / "run_metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def train_epoch(model, loader, criterion, optimizer, device, peak_w):
    model.train()
    total_loss = 0.0
    for batch in loader:
        sky = batch["sky"].to(device)
        pv_past_norm = batch["pv_past"].to(device) / peak_w
        targets_norm = batch["targets"].to(device) / peak_w
        optimizer.zero_grad()
        out = model(sky, pv_past_norm)
        loss = criterion(out, targets_norm)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, criterion, device, peak_w):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            sky = batch["sky"].to(device)
            pv_past_norm = batch["pv_past"].to(device) / peak_w
            targets_norm = batch["targets"].to(device) / peak_w
            out = model(sky, pv_past_norm)
            total_loss += criterion(out, targets_norm).item()
    return total_loss / len(loader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=str(CSV_PATH), help="训练 CSV 路径")
    parser.add_argument("--pack-dir", default=str(PACK_DIR), help="训练 packed 数据目录")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="训练 batch size")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="训练轮数")
    parser.add_argument("--lr", type=float, default=LR, help="学习率")
    parser.add_argument("--val-ratio", type=float, default=VAL_RATIO, help="验证集比例")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="随机种子")
    parser.add_argument("--sky-embed", type=int, default=SKY_EMBED, help="Sky 分支嵌入维度")
    parser.add_argument("--pv-hidden", type=int, default=PV_HIDDEN, help="PV 分支 hidden 维度")
    parser.add_argument("--fusion-hidden", type=int, default=FUSION_HIDDEN, help="融合层 hidden 维度")
    parser.add_argument("--dropout", type=float, default=DROPOUT, help="融合层 dropout")
    parser.add_argument("--run-name", default=None, help="run 目录后缀，便于区分 ablation")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        csv_path = BASE_DIR.parent / csv_path if str(csv_path).startswith("pv_forecasting") else BASE_DIR / csv_path
    pack_dir = Path(args.pack_dir)
    if not pack_dir.is_absolute():
        pack_dir = BASE_DIR.parent / pack_dir if str(pack_dir).startswith("pv_forecasting") else BASE_DIR / pack_dir

    run_dir = make_run_dir(args.run_name)
    setup_logging(run_dir)
    dump_metadata(run_dir, {
        "CSV_PATH": str(csv_path),
        "PACK_DIR": str(pack_dir),
        "SKY_MASK_PATH": str(SKY_MASK_PATH),
        "IMG_SIZE": IMG_SIZE,
        "PEAK_POWER_W": PEAK_POWER_W,
        "BATCH_SIZE": args.batch_size,
        "EPOCHS": args.epochs,
        "LR": args.lr,
        "VAL_RATIO": args.val_ratio,
        "RANDOM_SEED": args.seed,
        "MODEL_CFG": {
            "sky_embed": args.sky_embed,
            "pv_hidden": args.pv_hidden,
            "fusion_hidden": args.fusion_hidden,
            "dropout": args.dropout,
            "out_dim": HORIZON,
        },
    })

    if pack_dir.exists() and list(pack_dir.glob("batch_*.npz")):
        logging.info(f"Using packed data: {pack_dir}")
        dataset = PackedForecastDataset(pack_dir)
    elif csv_path.exists():
        logging.info(f"Using CSV data: {csv_path}")
        mask_path = SKY_MASK_PATH if SKY_MASK_PATH.exists() else None
        if mask_path:
            logging.info(f"[sky_mask] Loaded {SKY_MASK_PATH} -> applied to sky images")
        dataset = ForecastDataset(
            csv_path, img_size=IMG_SIZE, base_dir=BASE_DIR.parent, sky_mask_path=mask_path
        )
    else:
        raise FileNotFoundError(f"Run preprocess first. Need {pack_dir} or {csv_path}")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info(f"Device: GPU ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        logging.info("Device: CPU (no GPU detected)")
    n_val = max(1, int(len(dataset) * args.val_ratio))
    n_train = len(dataset) - n_val
    train_ds = Subset(dataset, range(n_train))
    val_ds = Subset(dataset, range(n_train, len(dataset)))
    logging.info(f"Split: train={n_train} (chronological first), val={n_val} (chronological last)")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = PVForecastModel(
        sky_embed=args.sky_embed,
        pv_hidden=args.pv_hidden,
        fusion_hidden=args.fusion_hidden,
        out_dim=HORIZON,
        dropout=args.dropout,
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = float("inf")
    history = []

    peak = PEAK_POWER_W
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, peak)
        val_loss = evaluate(model, val_loader, criterion, device, peak)
        history.append({"epoch": epoch + 1, "loss": train_loss, "val_loss": val_loss})
        logging.info(f"Epoch {epoch+1}/{args.epochs} loss={train_loss:.6f} val_loss={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), run_dir / "best_model.pt")

    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)

    # Load best model
    model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location=device))
    model.eval()

    def run_predict(ds, base_dataset) -> Tuple[np.ndarray, np.ndarray, pd.Series]:
        """Run prediction on ds (Subset or full). Return pred_w, true_w (both in W), ts_pred."""
        preds_norm, targets_raw = [], []
        with torch.no_grad():
            for batch in DataLoader(ds, batch_size=args.batch_size, shuffle=False):
                pv_past_norm = batch["pv_past"].to(device) / peak
                out = model(batch["sky"].to(device), pv_past_norm)
                preds_norm.append(out.cpu().numpy())
                targets_raw.append(batch["targets"].numpy())
        pred_norm = np.concatenate(preds_norm, axis=0)
        true_w = np.concatenate(targets_raw, axis=0)
        pred_w = pred_norm * peak
        if isinstance(ds, Subset):
            ts = base_dataset.ts_pred_series.iloc[list(ds.indices)]
        else:
            ts = base_dataset.ts_pred_series
        return pred_w, true_w, pd.to_datetime(ts)

    def compute_metrics(pred_w: np.ndarray, true_w: np.ndarray) -> dict:
        err = pred_w - true_w
        mae_all = np.abs(err).mean()
        rmse_all = np.sqrt((err ** 2).mean())
        row = {"mae_W": float(mae_all), "rmse_W": float(rmse_all)}
        for k in range(HORIZON):
            row[f"mae_W_h{k}"] = float(np.abs(err[:, k]).mean())
            row[f"rmse_W_h{k}"] = float(np.sqrt((err[:, k] ** 2).mean()))
        return row

    # 1) 全量预测（保留，供后续分析）
    pred_all_w, true_all_w, ts_all = run_predict(dataset, dataset)
    pred_df_all = pd.DataFrame({f"pv_pred_W_{k}": pred_all_w[:, k] for k in range(HORIZON)})
    pred_df_all["ts_pred"] = ts_all
    for k in range(HORIZON):
        pred_df_all[f"pv_true_W_{k}"] = true_all_w[:, k]
    pred_df_all.to_csv(run_dir / "predictions_all.csv", index=False)

    # 2) 验证集预测与指标（真正用于评估泛化）
    pred_val_w, true_val_w, ts_val = run_predict(val_ds, dataset)
    pred_df_val = pd.DataFrame({f"pv_pred_W_{k}": pred_val_w[:, k] for k in range(HORIZON)})
    pred_df_val["ts_pred"] = ts_val
    for k in range(HORIZON):
        pred_df_val[f"pv_true_W_{k}"] = true_val_w[:, k]
    pred_df_val.to_csv(run_dir / "predictions_val.csv", index=False)

    metrics_val = compute_metrics(pred_val_w, true_val_w)
    pd.DataFrame([metrics_val]).to_csv(run_dir / "metrics_val.csv", index=False)
    logging.info(f"Val metrics: MAE_W={metrics_val['mae_W']:.2f}, RMSE_W={metrics_val['rmse_W']:.2f}")
    for k in range(HORIZON):
        logging.info(f"  horizon {k} (t+{(k+1)*15}min): MAE_W={metrics_val[f'mae_W_h{k}']:.2f}, RMSE_W={metrics_val[f'rmse_W_h{k}']:.2f}")

    # 3) 全量指标（仅作参考，勿与 val 混淆）
    metrics_all = compute_metrics(pred_all_w, true_all_w)
    pd.DataFrame([metrics_all]).to_csv(run_dir / "metrics_all.csv", index=False)
    logging.info(f"All metrics (reference): MAE_W={metrics_all['mae_W']:.2f}")

    logging.info(f"Saved {run_dir}: history.csv, metrics_val.csv, metrics_all.csv, predictions_val.csv, predictions_all.csv, best_model.pt")


if __name__ == "__main__":
    main()
