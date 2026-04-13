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
from torch.utils.data import DataLoader

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from dataset import PreprocessedConvLSTMDataset
from model import ConvLSTMPVRegressor


DEFAULT_POWER_CSV = Path("data/power/power-LSK_N_0117-0401.csv")
DEFAULT_CAMERA_INDEX_CSV = Path("data/camera_data/index/raw_index.csv")
DEFAULT_SAMPLES_CSV = Path("new-model/artifacts/dataset/samples.csv")
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
        "mae": mae_w,
        "rmse": rmse_w,
        "mae_W": mae_w,
        "rmse_W": rmse_w,
        "mae_norm_peak": mae_w / peak_w if peak_w > 0 else float("nan"),
        "rmse_norm_peak": rmse_w / peak_w if peak_w > 0 else float("nan"),
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    is_train = optimizer is not None
    model.train(is_train)
    loss_sum = 0.0
    for batch in loader:
        x = batch["x_seq"].to(device)
        pv_history = batch["pv_history"].to(device)
        y = batch["target"].to(device)
        if is_train:
            optimizer.zero_grad()
        pred = model(x, pv_history)
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
    use_clear_sky_index: bool,
):
    model.eval()
    rows: list[dict] = []
    with torch.no_grad():
        for batch in DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0):
            x = batch["x_seq"].to(device)
            pv_history = batch["pv_history"].to(device)
            pred_raw = model(x, pv_history).cpu().numpy().reshape(-1)
            target_value = batch["target"].cpu().numpy().reshape(-1)
            target_pv_w = batch["target_pv_w"].cpu().numpy().reshape(-1)
            clear_sky_w = batch["target_clear_sky_w"].cpu().numpy().reshape(-1)
            pv_history_w = batch["pv_history_w"].cpu().numpy()
            meta_index = batch["meta_index"].cpu().numpy().reshape(-1)

            pred_value = np.clip(pred_raw, a_min=0.0, a_max=1.0 if use_clear_sky_index else None)
            pred_w = pred_value * clear_sky_w if use_clear_sky_index else pred_value
            pred_w = np.clip(pred_w, a_min=0.0, a_max=None)

            for i in range(len(pred_raw)):
                rows.append(
                    {
                        "ts_anchor": batch["t_anchor"][i],
                        "ts_target": batch["t_target"][i],
                        "target_value": float(target_value[i]),
                        "target_pv_w": float(target_pv_w[i]),
                        "target_clear_sky_w": float(clear_sky_w[i]),
                        "past_pv_w": json.dumps(pv_history_w[i].tolist(), ensure_ascii=False),
                        "pred_value": float(pred_value[i]),
                        "pred_value_raw": float(pred_raw[i]),
                        "pred_w": float(pred_w[i]),
                        "sample_index": int(meta_index[i]),
                    }
                )
    pred_df = pd.DataFrame(rows).sort_values("ts_target").reset_index(drop=True)
    return pred_df


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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--peak-power-w", type=float, default=66300.0)
    parser.add_argument("--img-h", type=int, default=64)
    parser.add_argument("--img-w", type=int, default=64)
    parser.add_argument("--img-channels", type=int, choices=[1, 3], default=1)
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--pv-history-dim", type=int, default=4)
    parser.add_argument("--pv-hidden", type=int, default=32)
    parser.add_argument("--fc-hidden", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--max-time-diff-sec", type=int, default=90)
    parser.add_argument("--camera-path-prefix-from", default=None)
    parser.add_argument("--camera-path-prefix-to", default=None)
    parser.add_argument("--run-name", default="convlstm_encoder")
    parser.add_argument("--save-every-epoch", action="store_true")
    parser.add_argument("--use-clear-sky-index", dest="use_clear_sky_index", action="store_true")
    parser.add_argument("--no-use-clear-sky-index", dest="use_clear_sky_index", action="store_false")
    parser.set_defaults(use_clear_sky_index=True)
    parser.add_argument("--dry-run", action="store_true")
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
    if not samples_csv.exists():
        raise FileNotFoundError(f"Samples csv not found: {samples_csv}")
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

    common_dataset_kwargs = {
        "samples_csv": samples_csv,
        "image_size": (args.img_h, args.img_w),
        "channels": args.img_channels,
        "sky_mask_path": (sky_mask_path if use_mask else None),
        "camera_path_prefix_from": args.camera_path_prefix_from,
        "camera_path_prefix_to": args.camera_path_prefix_to,
    }
    train_ds = PreprocessedConvLSTMDataset(split="train", **common_dataset_kwargs)
    val_ds = PreprocessedConvLSTMDataset(split="val", **common_dataset_kwargs)
    test_ds = PreprocessedConvLSTMDataset(split="test", **common_dataset_kwargs)
    if min(len(train_ds), len(val_ds), len(test_ds)) <= 0:
        raise RuntimeError(
            f"Invalid split sizes from samples csv: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}"
        )
    logging.info("Using preprocessed samples csv: %s", samples_csv)
    logging.info("Dataset sizes train=%d val=%d test=%d", len(train_ds), len(val_ds), len(test_ds))
    logging.info("Input sequence follows samples csv (aligned with new-model dataset preparation).")
    logging.info("Target normalization: %s", "clear-sky index" if args.use_clear_sky_index else "raw PV power")
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
        pv_history_dim=args.pv_history_dim,
        pv_hidden=args.pv_hidden,
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
            "SAMPLES_CSV": str(samples_csv),
            "BATCH_SIZE": args.batch_size,
            "EPOCHS": args.epochs,
            "LR": args.lr,
            "WEIGHT_DECAY": args.weight_decay,
            "RANDOM_SEED": args.seed,
            "PEAK_POWER_W": args.peak_power_w,
            "USE_CLEAR_SKY_INDEX": args.use_clear_sky_index,
            "IMG_SIZE": [args.img_h, args.img_w],
            "IMG_CHANNELS": args.img_channels,
            "MODEL_CFG": {
                "hidden_channels": args.hidden_channels,
                "kernel_size": args.kernel_size,
                "pv_history_dim": args.pv_history_dim,
                "pv_hidden": args.pv_hidden,
                "fc_hidden": args.fc_hidden,
                "dropout": args.dropout,
            },
            "MAX_TIME_DIFF_SEC": args.max_time_diff_sec,
            "SKY_MASK_PATH": str(sky_mask_path) if use_mask else None,
            "CAMERA_PATH_PREFIX_FROM": args.camera_path_prefix_from,
            "CAMERA_PATH_PREFIX_TO": args.camera_path_prefix_to,
            "TASK": "predict pv at t+15 from image sequence defined in shared samples csv",
        },
    )

    if args.dry_run:
        sample_batch = next(iter(val_loader))
        x = sample_batch["x_seq"].to(device)
        pv_history = sample_batch["pv_history"].to(device)
        with torch.no_grad():
            pred = model(x, pv_history)
        logging.info(
            "Dry run succeeded. Batch x=%s pv_history=%s pred=%s",
            tuple(sample_batch["x_seq"].shape),
            tuple(sample_batch["pv_history"].shape),
            tuple(pred.shape),
        )
        return

    history: list[dict] = []
    best_val = float("inf")
    for epoch in range(args.epochs):
        train_loss = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        val_loss = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            optimizer=None,
            device=device,
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
    split_datasets = {"train": train_ds, "val": val_ds, "test": test_ds}
    summary_rows: list[dict] = []
    for split_name, split_ds in split_datasets.items():
        pred_df = predict(
            model=model,
            ds=split_ds,
            device=device,
            batch_size=args.batch_size,
            use_clear_sky_index=args.use_clear_sky_index,
        )
        pred_df.to_csv(run_dir / f"predictions_{split_name}.csv", index=False)
        metrics = compute_metrics(
            pred_df["pred_w"].to_numpy(dtype=np.float32),
            pred_df["target_pv_w"].to_numpy(dtype=np.float32),
            args.peak_power_w,
        )
        (run_dir / f"metrics_{split_name}.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        summary_rows.append({"split": split_name, **metrics})
        logging.info("%s metrics: MAE_W=%.2f RMSE_W=%.2f", split_name, metrics["mae_W"], metrics["rmse_W"])

    pd.DataFrame(summary_rows).to_csv(run_dir / "metrics_summary.csv", index=False)
    logging.info("Saved outputs in %s", run_dir)


if __name__ == "__main__":
    main()
