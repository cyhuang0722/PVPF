"""
Train PV-only LSTM and save run artifacts.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from .dataset import PVPastDataset
from .model import PVPastLSTM


BASE_DIR = Path(__file__).resolve().parent
DERIVED_DIR = BASE_DIR / "derived"
MODEL_OUTPUT_DIR = BASE_DIR / "model_output"
CSV_PATH = DERIVED_DIR / "forecast_windows.csv"

PEAK_POWER_W = 66300.0
BATCH_SIZE = 64
EPOCHS = 100
LR = 1e-3
VAL_RATIO = 0.2
RANDOM_SEED = 42
HIDDEN_DIM = 64
NUM_LAYERS = 2
DROPOUT = 0.2
WEIGHT_DECAY = 1e-5


def make_run_dir(run_name: str | None = None) -> Path:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    suffix = f"_{run_name}" if run_name else ""
    run_dir = MODEL_OUTPUT_DIR / f"run_{ts}{suffix}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    return run_dir


def setup_logging(run_dir: Path) -> None:
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


def dump_metadata(run_dir: Path, cfg: dict) -> None:
    meta = {"timestamp": datetime.now().isoformat(), "cfg": cfg}
    (run_dir / "run_metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def train_epoch(model, loader, criterion, optimizer, device, peak_w: float) -> float:
    model.train()
    total = 0.0
    for batch in loader:
        x = batch["pv_past"].to(device) / peak_w
        y = batch["target"].to(device) / peak_w
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / max(1, len(loader))


def eval_epoch(model, loader, criterion, device, peak_w: float) -> float:
    model.eval()
    total = 0.0
    with torch.no_grad():
        for batch in loader:
            x = batch["pv_past"].to(device) / peak_w
            y = batch["target"].to(device) / peak_w
            pred = model(x)
            total += criterion(pred, y).item()
    return total / max(1, len(loader))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=str(CSV_PATH), help="forecast windows csv path")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="batch size")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="epochs")
    parser.add_argument("--lr", type=float, default=LR, help="learning rate")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY, help="adamw weight decay")
    parser.add_argument("--val-ratio", type=float, default=VAL_RATIO, help="val split ratio")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="random seed")
    parser.add_argument("--hidden-dim", type=int, default=HIDDEN_DIM, help="lstm hidden dim")
    parser.add_argument("--num-layers", type=int, default=NUM_LAYERS, help="lstm layer count")
    parser.add_argument("--dropout", type=float, default=DROPOUT, help="lstm/head dropout")
    parser.add_argument("--run-name", default="pv_only", help="run dir suffix")
    parser.add_argument("--save-every-epoch", action="store_true", help="save each epoch checkpoint")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        csv_path = (BASE_DIR.parent.parent / csv_path) if str(csv_path).startswith("pv_forecasting/") else (BASE_DIR / csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    run_dir = make_run_dir(args.run_name)
    setup_logging(run_dir)
    dump_metadata(
        run_dir,
        {
            "CSV_PATH": str(csv_path),
            "PEAK_POWER_W": PEAK_POWER_W,
            "BATCH_SIZE": args.batch_size,
            "EPOCHS": args.epochs,
            "LR": args.lr,
            "WEIGHT_DECAY": args.weight_decay,
            "VAL_RATIO": args.val_ratio,
            "RANDOM_SEED": args.seed,
            "MODEL_CFG": {
                "input_dim": 1,
                "hidden_dim": args.hidden_dim,
                "num_layers": args.num_layers,
                "dropout": args.dropout,
            },
        },
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info(f"Device: GPU ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        logging.info("Device: CPU (no GPU detected)")

    dataset = PVPastDataset(csv_path)
    if len(dataset) < 10:
        raise RuntimeError(f"Too few samples: {len(dataset)}")
    n_val = max(1, int(len(dataset) * args.val_ratio))
    n_train = len(dataset) - n_val
    train_ds = Subset(dataset, range(n_train))
    val_ds = Subset(dataset, range(n_train, len(dataset)))
    logging.info(f"Split: train={n_train} (chronological first), val={n_val} (chronological last)")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = PVPastLSTM(
        input_dim=1,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = float("inf")
    history: list[dict] = []
    peak = PEAK_POWER_W
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, peak)
        val_loss = eval_epoch(model, val_loader, criterion, device, peak)
        row = {"epoch": epoch + 1, "loss": train_loss, "val_loss": val_loss}
        history.append(row)
        logging.info(f"Epoch {epoch + 1}/{args.epochs} loss={train_loss:.6f} val_loss={val_loss:.6f}")

        torch.save(model.state_dict(), run_dir / "last_model.pt")
        if args.save_every_epoch:
            torch.save(model.state_dict(), run_dir / "checkpoints" / f"epoch_{epoch + 1:03d}.pt")
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), run_dir / "best_model.pt")

    model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location=device))
    model.eval()

    def run_predict(ds, base_dataset):
        preds_norm, targets_raw = [], []
        with torch.no_grad():
            for batch in DataLoader(ds, batch_size=args.batch_size, shuffle=False):
                x = batch["pv_past"].to(device) / peak
                out = model(x)
                preds_norm.append(out.cpu().numpy().reshape(-1))
                targets_raw.append(batch["target"].numpy().reshape(-1))
        pred_w = np.concatenate(preds_norm, axis=0) * peak
        true_w = np.concatenate(targets_raw, axis=0)
        if isinstance(ds, Subset):
            idx = list(ds.indices)
            ts_anchor = base_dataset.ts_anchor_series.iloc[idx]
            ts_target = base_dataset.ts_target_series.iloc[idx]
        else:
            ts_anchor = base_dataset.ts_anchor_series
            ts_target = base_dataset.ts_target_series
        return pred_w, true_w, pd.to_datetime(ts_anchor), pd.to_datetime(ts_target)

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

    logging.info(f"Val metrics: MAE_W={metrics_val['mae_W']:.2f}, RMSE_W={metrics_val['rmse_W']:.2f}")
    logging.info(f"All metrics (reference): MAE_W={metrics_all['mae_W']:.2f}, RMSE_W={metrics_all['rmse_W']:.2f}")
    logging.info(
        f"Saved {run_dir}: history.csv, metrics_val.csv, metrics_all.csv, "
        "predictions_val.csv, predictions_all.csv, best_model.pt, last_model.pt"
    )


if __name__ == "__main__":
    main()

