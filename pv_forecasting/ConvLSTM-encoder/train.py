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

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from dataset import CameraPVConvLSTMDataset, PreprocessedConvLSTMDataset
from model import ConvLSTMPVRegressor


DEFAULT_POWER_CSV = Path("data/power/power-LSK_N_0117-0401.csv")
DEFAULT_CAMERA_INDEX_CSV = Path("data/camera_data/index/raw_index.csv")
DEFAULT_SAMPLES_CSV = Path("pv_forecasting/ConvLSTM-encoder/derived/samples_t_plus_15.csv")
DEFAULT_SKY_MASK_PATH = Path("data/sky_mask.png")
MODEL_OUTPUT_DIR = CURRENT_DIR / "model_output"


def make_run_dir(run_name: str | None = None) -> Path:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    suffix = f"_{run_name}" if run_name else ""
    run_dir = MODEL_OUTPUT_DIR / f"run_{ts}{suffix}"
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    return run_dir


def setup_logging(run_dir: Path) -> None:
    log_path = run_dir / "train.log"
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(sh)
    root.addHandler(fh)


def dump_metadata(run_dir: Path, cfg: dict) -> None:
    payload = {"timestamp": datetime.now().isoformat(), "cfg": cfg}
    (run_dir / "run_metadata.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def compute_metrics(pred_w: np.ndarray, true_w: np.ndarray, peak_w: float) -> dict:
    err = pred_w - true_w
    mae_w = float(np.abs(err).mean())
    rmse_w = float(np.sqrt((err ** 2).mean()))
    return {
        "mae_W": mae_w,
        "rmse_W": rmse_w,
        "mae_norm_peak": mae_w / peak_w,
        "rmse_norm_peak": rmse_w / peak_w,
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    peak_power_w: float,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    is_train = optimizer is not None
    model.train(is_train)
    loss_sum = 0.0
    for batch in loader:
        x = batch["x_seq"].to(device)
        y = batch["y"].to(device) / peak_power_w
        if is_train:
            optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        if is_train:
            loss.backward()
            optimizer.step()
        loss_sum += loss.item()
    return loss_sum / max(1, len(loader))


def predict(
    model: nn.Module,
    ds,
    device: torch.device,
    batch_size: int,
    peak_power_w: float,
):
    model.eval()
    preds_norm: list[np.ndarray] = []
    y_true: list[np.ndarray] = []
    ts_anchor_all: list[str] = []
    ts_target_all: list[str] = []
    with torch.no_grad():
        for batch in DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0):
            x = batch["x_seq"].to(device)
            out = model(x).cpu().numpy().reshape(-1)
            preds_norm.append(out)
            y_true.append(batch["y"].numpy().reshape(-1))
            ts_anchor_all.extend(batch["t_anchor"])
            ts_target_all.extend(batch["t_target"])
    pred_w = np.concatenate(preds_norm) * peak_power_w
    true_w = np.concatenate(y_true)
    ts_anchor = pd.to_datetime(ts_anchor_all)
    ts_target = pd.to_datetime(ts_target_all)
    return pred_w, true_w, ts_anchor, ts_target


def main() -> None:
    parser = argparse.ArgumentParser(description="ConvLSTM encoder for t+15 PV forecasting")
    parser.add_argument("--power-csv", default=str(DEFAULT_POWER_CSV))
    parser.add_argument("--camera-index-csv", default=str(DEFAULT_CAMERA_INDEX_CSV))
    parser.add_argument("--samples-csv", default=str(DEFAULT_SAMPLES_CSV))
    parser.add_argument("--sky-mask-path", default=str(DEFAULT_SKY_MASK_PATH))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--peak-power-w", type=float, default=66300.0)
    parser.add_argument("--img-h", type=int, default=64)
    parser.add_argument("--img-w", type=int, default=64)
    parser.add_argument("--img-channels", type=int, choices=[1, 3], default=1)
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--fc-hidden", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--max-time-diff-sec", type=int, default=90)
    parser.add_argument("--camera-path-prefix-from", default=None)
    parser.add_argument("--camera-path-prefix-to", default=None)
    parser.add_argument("--run-name", default="convlstm_encoder")
    parser.add_argument("--save-every-epoch", action="store_true")
    args = parser.parse_args()

    root_dir = CURRENT_DIR.parent.parent
    power_csv = Path(args.power_csv)
    camera_index_csv = Path(args.camera_index_csv)
    samples_csv = Path(args.samples_csv)
    sky_mask_path = Path(args.sky_mask_path)
    if not power_csv.is_absolute():
        power_csv = root_dir / power_csv
    if not camera_index_csv.is_absolute():
        camera_index_csv = root_dir / camera_index_csv
    if not samples_csv.is_absolute():
        samples_csv = root_dir / samples_csv
    if not sky_mask_path.is_absolute():
        sky_mask_path = root_dir / sky_mask_path
    if not power_csv.exists():
        raise FileNotFoundError(f"Power csv not found: {power_csv}")
    if not camera_index_csv.exists():
        raise FileNotFoundError(f"Camera index csv not found: {camera_index_csv}")
    use_mask = sky_mask_path.exists()

    run_dir = make_run_dir(args.run_name)
    setup_logging(run_dir)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info("Device: GPU (%s)", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        logging.info("Device: CPU")

    if samples_csv.exists():
        dataset = PreprocessedConvLSTMDataset(
            samples_csv=samples_csv,
            image_size=(args.img_h, args.img_w),
            channels=args.img_channels,
            sky_mask_path=(sky_mask_path if use_mask else None),
            camera_path_prefix_from=args.camera_path_prefix_from,
            camera_path_prefix_to=args.camera_path_prefix_to,
        )
        logging.info("Using preprocessed samples csv: %s", samples_csv)
    else:
        dataset = CameraPVConvLSTMDataset(
            power_csv=power_csv,
            camera_index_csv=camera_index_csv,
            image_size=(args.img_h, args.img_w),
            channels=args.img_channels,
            max_time_diff_sec=args.max_time_diff_sec,
            sky_mask_path=(sky_mask_path if use_mask else None),
            camera_path_prefix_from=args.camera_path_prefix_from,
            camera_path_prefix_to=args.camera_path_prefix_to,
        )
        logging.info("Samples csv not found, fallback online matching from raw index.")
    if len(dataset) < 20:
        raise RuntimeError(f"Too few valid samples ({len(dataset)}). Check camera index and PV coverage.")

    n_val = max(1, int(len(dataset) * args.val_ratio))
    n_train = len(dataset) - n_val
    train_ds = Subset(dataset, range(n_train))
    val_ds = Subset(dataset, range(n_train, len(dataset)))
    logging.info("Dataset size=%d train=%d val=%d", len(dataset), n_train, n_val)
    logging.info("Input offsets are relative to anchor t: [-15,-13,-11,-9,-7,-5,-3,-1], with target at t+15")
    logging.info("Nearest camera tolerance=%d sec", args.max_time_diff_sec)
    logging.info("PV normalization by peak_power_w=%.1f", args.peak_power_w)
    logging.info("Sky mask: %s", str(sky_mask_path) if use_mask else "disabled (file not found)")
    if args.camera_path_prefix_from and args.camera_path_prefix_to:
        logging.info(
            "Camera path remap enabled: '%s' -> '%s'",
            args.camera_path_prefix_from,
            args.camera_path_prefix_to,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = ConvLSTMPVRegressor(
        in_channels=args.img_channels,
        hidden_channels=args.hidden_channels,
        kernel_size=args.kernel_size,
        fc_hidden=args.fc_hidden,
        dropout=args.dropout,
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    dump_metadata(
        run_dir,
        {
            "POWER_CSV": str(power_csv),
            "CAMERA_INDEX_CSV": str(camera_index_csv),
            "BATCH_SIZE": args.batch_size,
            "EPOCHS": args.epochs,
            "LR": args.lr,
            "WEIGHT_DECAY": args.weight_decay,
            "VAL_RATIO": args.val_ratio,
            "RANDOM_SEED": args.seed,
            "PEAK_POWER_W": args.peak_power_w,
            "IMG_SIZE": [args.img_h, args.img_w],
            "IMG_CHANNELS": args.img_channels,
            "MODEL_CFG": {
                "hidden_channels": args.hidden_channels,
                "kernel_size": args.kernel_size,
                "fc_hidden": args.fc_hidden,
                "dropout": args.dropout,
            },
            "MAX_TIME_DIFF_SEC": args.max_time_diff_sec,
            "SKY_MASK_PATH": str(sky_mask_path) if use_mask else None,
            "CAMERA_PATH_PREFIX_FROM": args.camera_path_prefix_from,
            "CAMERA_PATH_PREFIX_TO": args.camera_path_prefix_to,
            "TASK": "predict pv at t+15 from images at t-15..t-1 every 2 minutes (8 frames)",
        },
    )

    history: list[dict] = []
    best_val = float("inf")
    for epoch in range(args.epochs):
        train_loss = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            peak_power_w=args.peak_power_w,
        )
        val_loss = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            optimizer=None,
            device=device,
            peak_power_w=args.peak_power_w,
        )
        history.append({"epoch": epoch + 1, "loss": train_loss, "val_loss": val_loss})
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        torch.save(model.state_dict(), run_dir / "last_model.pt")
        if args.save_every_epoch:
            torch.save(model.state_dict(), run_dir / "checkpoints" / f"epoch_{epoch + 1:03d}.pt")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), run_dir / "best_model.pt")
        logging.info("Epoch %d/%d loss=%.6f val_loss=%.6f", epoch + 1, args.epochs, train_loss, val_loss)

    model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location=device))
    pred_all_w, true_all_w, ts_anchor_all, ts_target_all = predict(
        model=model,
        ds=dataset,
        device=device,
        batch_size=args.batch_size,
        peak_power_w=args.peak_power_w,
    )
    pred_val_w, true_val_w, ts_anchor_val, ts_target_val = predict(
        model=model,
        ds=val_ds,
        device=device,
        batch_size=args.batch_size,
        peak_power_w=args.peak_power_w,
    )

    pred_all_df = pd.DataFrame(
        {
            "ts_anchor": ts_anchor_all,
            "ts_target": ts_target_all,
            "pv_pred_W_t_plus_15": pred_all_w,
            "pv_true_W_t_plus_15": true_all_w,
        }
    )
    pred_val_df = pd.DataFrame(
        {
            "ts_anchor": ts_anchor_val,
            "ts_target": ts_target_val,
            "pv_pred_W_t_plus_15": pred_val_w,
            "pv_true_W_t_plus_15": true_val_w,
        }
    )
    pred_all_df.to_csv(run_dir / "predictions_all.csv", index=False)
    pred_val_df.to_csv(run_dir / "predictions_val.csv", index=False)

    metrics_all = compute_metrics(pred_all_w, true_all_w, args.peak_power_w)
    metrics_val = compute_metrics(pred_val_w, true_val_w, args.peak_power_w)
    pd.DataFrame([metrics_all]).to_csv(run_dir / "metrics_all.csv", index=False)
    pd.DataFrame([metrics_val]).to_csv(run_dir / "metrics_val.csv", index=False)

    logging.info("Val metrics: MAE_W=%.2f RMSE_W=%.2f", metrics_val["mae_W"], metrics_val["rmse_W"])
    logging.info("Saved outputs in %s", run_dir)


if __name__ == "__main__":
    main()
