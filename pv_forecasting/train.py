"""
训练 Sky+PV 双分支预测模型，结果保存到 model_output/run_YYYYMMDD-HHMMSS。
"""
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


def make_run_dir() -> Path:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = MODEL_OUTPUT_DIR / f"run_{ts}"
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
        pv_past = batch["pv_past"].to(device) / peak_w
        targets = batch["targets"].to(device) / peak_w
        optimizer.zero_grad()
        out = model(sky, pv_past)
        loss = criterion(out, targets)
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
            pv_past = batch["pv_past"].to(device) / peak_w
            targets = batch["targets"].to(device) / peak_w
            out = model(sky, pv_past)
            total_loss += criterion(out, targets).item()
    return total_loss / len(loader)


def main():
    run_dir = make_run_dir()
    setup_logging(run_dir)
    dump_metadata(run_dir, {
        "CSV_PATH": str(CSV_PATH),
        "PACK_DIR": str(PACK_DIR),
        "SKY_MASK_PATH": str(SKY_MASK_PATH),
        "IMG_SIZE": IMG_SIZE,
        "PEAK_POWER_W": PEAK_POWER_W,
        "BATCH_SIZE": BATCH_SIZE,
        "EPOCHS": EPOCHS,
        "LR": LR,
        "VAL_RATIO": VAL_RATIO,
        "RANDOM_SEED": RANDOM_SEED,
    })

    if PACK_DIR.exists() and list(PACK_DIR.glob("batch_*.npz")):
        logging.info(f"Using packed data: {PACK_DIR}")
        dataset = PackedForecastDataset(PACK_DIR)
    elif CSV_PATH.exists():
        logging.info(f"Using CSV data: {CSV_PATH}")
        mask_path = SKY_MASK_PATH if SKY_MASK_PATH.exists() else None
        if mask_path:
            logging.info(f"[sky_mask] Loaded {SKY_MASK_PATH} -> applied to sky images")
        dataset = ForecastDataset(
            CSV_PATH, img_size=IMG_SIZE, base_dir=BASE_DIR.parent, sky_mask_path=mask_path
        )
    else:
        raise FileNotFoundError(f"Run preprocess first. Need {PACK_DIR} or {CSV_PATH}")

    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info(f"Device: GPU ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        logging.info("Device: CPU (no GPU detected)")
    n_val = max(1, int(len(dataset) * VAL_RATIO))
    n_train = len(dataset) - n_val
    train_ds = Subset(dataset, range(n_train))
    val_ds = Subset(dataset, range(n_train, len(dataset)))
    logging.info(f"Split: train={n_train} (chronological first), val={n_val} (chronological last)")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = PVForecastModel().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val = float("inf")
    history = []

    peak = PEAK_POWER_W
    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, peak)
        val_loss = evaluate(model, val_loader, criterion, device, peak)
        history.append({"epoch": epoch + 1, "loss": train_loss, "val_loss": val_loss})
        logging.info(f"Epoch {epoch+1}/{EPOCHS} loss={train_loss:.6f} val_loss={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), run_dir / "best_model.pt")

    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)

    # Load best model
    model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location=device))
    model.eval()

    def run_predict(ds, base_dataset) -> Tuple[np.ndarray, np.ndarray, pd.Series]:
        """Run prediction on ds (Subset or full), return pred_w, true_w, ts_pred."""
        preds, targets = [], []
        with torch.no_grad():
            for batch in DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False):
                pv_past = batch["pv_past"].to(device) / peak
                out = model(batch["sky"].to(device), pv_past)
                preds.append(out.cpu().numpy())
                targets.append(batch["targets"].numpy())
        pred_norm = np.concatenate(preds, axis=0)
        true_norm = np.concatenate(targets, axis=0)
        pred_w = pred_norm * peak
        true_w = true_norm * peak
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
        for k in range(5):
            row[f"mae_W_h{k}"] = float(np.abs(err[:, k]).mean())
            row[f"rmse_W_h{k}"] = float(np.sqrt((err[:, k] ** 2).mean()))
        return row

    # 1) 全量预测（保留，供后续分析）
    pred_all_w, true_all_w, ts_all = run_predict(dataset, dataset)
    pred_df_all = pd.DataFrame({f"pv_pred_W_{k}": pred_all_w[:, k] for k in range(5)})
    pred_df_all["ts_pred"] = ts_all
    for k in range(5):
        pred_df_all[f"pv_true_W_{k}"] = true_all_w[:, k]
    pred_df_all.to_csv(run_dir / "predictions_all.csv", index=False)

    # 2) 验证集预测与指标（真正用于评估泛化）
    pred_val_w, true_val_w, ts_val = run_predict(val_ds, dataset)
    pred_df_val = pd.DataFrame({f"pv_pred_W_{k}": pred_val_w[:, k] for k in range(5)})
    pred_df_val["ts_pred"] = ts_val
    for k in range(5):
        pred_df_val[f"pv_true_W_{k}"] = true_val_w[:, k]
    pred_df_val.to_csv(run_dir / "predictions_val.csv", index=False)

    metrics_val = compute_metrics(pred_val_w, true_val_w)
    pd.DataFrame([metrics_val]).to_csv(run_dir / "metrics_val.csv", index=False)
    logging.info(f"Val metrics: MAE_W={metrics_val['mae_W']:.2f}, RMSE_W={metrics_val['rmse_W']:.2f}")
    for k in range(5):
        logging.info(f"  horizon {k} (t+{k*15}min): MAE_W={metrics_val[f'mae_W_h{k}']:.2f}, RMSE_W={metrics_val[f'rmse_W_h{k}']:.2f}")

    # 3) 全量指标（仅作参考，勿与 val 混淆）
    metrics_all = compute_metrics(pred_all_w, true_all_w)
    pd.DataFrame([metrics_all]).to_csv(run_dir / "metrics_all.csv", index=False)
    logging.info(f"All metrics (reference): MAE_W={metrics_all['mae_W']:.2f}")

    logging.info(f"Saved {run_dir}: history.csv, metrics_val.csv, metrics_all.csv, predictions_val.csv, predictions_all.csv, best_model.pt")


if __name__ == "__main__":
    main()
