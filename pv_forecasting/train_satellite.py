from __future__ import annotations

import argparse
import ast
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

from .satellite_common import load_json_config
from .satellite_dataset import PackedSatelliteDataset, SatelliteForecastDataset
from .satellite_model import SatelliteOnlyForecaster


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
MODEL_OUTPUT_DIR = BASE_DIR / "model_output"


def make_run_dir(run_name: str | None = None) -> Path:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    suffix = f"_{run_name}" if run_name else ""
    run_dir = MODEL_OUTPUT_DIR / f"run_{ts}{suffix}"
    run_dir.mkdir(parents=True, exist_ok=True)
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


def compute_metrics(pred_w: np.ndarray, true_w: np.ndarray, horizon_names: list[str]) -> dict[str, float]:
    err = pred_w - true_w
    out: dict[str, float] = {}
    rmse_list = []
    for idx, horizon in enumerate(horizon_names):
        mae = float(np.abs(err[:, idx]).mean())
        rmse = float(np.sqrt(np.mean(err[:, idx] ** 2)))
        out[f"mae_{horizon}_W"] = mae
        out[f"rmse_{horizon}_W"] = rmse
        rmse_list.append(rmse)
    out["overall_mae_W"] = float(np.abs(err).mean())
    out["overall_rmse_W"] = float(np.sqrt(np.mean(err ** 2)))
    out["mean_rmse_W"] = float(np.mean(rmse_list))
    return out


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            sat = batch["satellite"].to(device)
            y = batch["targets"].to(device)
            out = model(sat)
            loss = criterion(out, y)
            total_loss += float(loss.item())
            preds.append(out.cpu().numpy())
            targets.append(y.cpu().numpy())
    return total_loss / max(1, len(loader)), np.concatenate(preds, axis=0), np.concatenate(targets, axis=0)


def resolve_model_kwargs(model_cfg: dict, out_dim: int) -> dict:
    encoder_channels = tuple(model_cfg.get("encoder_channels", [32, 64, 128]))
    base_channels = int(model_cfg.get("base_channels", encoder_channels[0]))
    stage_multipliers = tuple(
        model_cfg.get(
            "stage_multipliers",
            [max(1, int(ch // max(1, base_channels))) for ch in (*encoder_channels, encoder_channels[-1] * 2)],
        )
    )
    return {
        "base_channels": base_channels,
        "stage_multipliers": stage_multipliers,
        "temporal_hidden_dim": int(model_cfg.get("temporal_hidden_dim", model_cfg.get("convlstm_hidden_dim", 128))),
        "head_hidden_dim": int(model_cfg.get("head_hidden_dim", 256)),
        "dropout": float(model_cfg.get("dropout", 0.2)),
        "out_dim": out_dim,
        "debug_shapes": bool(model_cfg.get("debug_shapes", False)),
    }


def validate_preprocessed_artifacts(csv_path: Path, pack_dir: Path, data_cfg: dict) -> None:
    expected_t = int(data_cfg["t_in"])
    expected_c = len(data_cfg["channels"])
    expected_patch = int(data_cfg["patch_size"])

    if csv_path.exists():
        meta = pd.read_csv(csv_path, nrows=1)
        if not meta.empty:
            row = meta.iloc[0]
            sat_paths = ast.literal_eval(row["sat_paths"]) if isinstance(row["sat_paths"], str) else row["sat_paths"]
            channels = ast.literal_eval(row["channels"]) if isinstance(row["channels"], str) else row["channels"]
            patch_size = int(row["patch_size"])
            if len(sat_paths) != expected_t or len(channels) != expected_c or patch_size != expected_patch:
                raise ValueError(
                    "forecast_windows.csv does not match the current config. "
                    "Please rerun pv_forecasting/preprocess_satellite.py."
                )

    shards = sorted(pack_dir.glob("batch_*.npz"))
    if shards:
        with np.load(shards[0], allow_pickle=True) as data:
            sat = data["satellite"]
            if sat.ndim != 5 or sat.shape[1] != expected_t or sat.shape[2] != expected_c or sat.shape[3] != expected_patch or sat.shape[4] != expected_patch:
                raise ValueError(
                    "packed_satellite tensors do not match the current config. "
                    "Please rerun pv_forecasting/preprocess_satellite.py --pack."
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="JSON config path")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    cfg = load_json_config(args.config)
    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    model_cfg = cfg["model"]

    run_dir = make_run_dir(args.run_name or cfg.get("name"))
    setup_logging(run_dir)
    (run_dir / "run_config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")

    pack_dir = PROJECT_ROOT / data_cfg["out_dir"] / "packed_satellite"
    csv_path = PROJECT_ROOT / data_cfg["out_dir"] / "forecast_windows.csv"
    stats_path = PROJECT_ROOT / data_cfg["out_dir"] / data_cfg.get("stats_filename", "satellite_stats.json")
    validate_preprocessed_artifacts(csv_path=csv_path, pack_dir=pack_dir, data_cfg=data_cfg)

    if pack_dir.exists() and list(pack_dir.glob("batch_*.npz")):
        train_ds = PackedSatelliteDataset(pack_dir, split="train")
        val_ds = PackedSatelliteDataset(pack_dir, split="val")
    else:
        train_ds = SatelliteForecastDataset(csv_path, split="train", stats_path=stats_path, peak_power_w=float(data_cfg["peak_power_w"]))
        val_ds = SatelliteForecastDataset(csv_path, split="val", stats_path=stats_path, peak_power_w=float(data_cfg["peak_power_w"]))

    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info(f"Device: GPU ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        logging.info("Device: CPU")

    torch.manual_seed(int(train_cfg.get("seed", 42)))
    train_loader = DataLoader(
        train_ds,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 0)),
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(train_cfg.get("num_workers", 0)),
    )

    model = SatelliteOnlyForecaster(
        in_channels=len(data_cfg["channels"]),
        **resolve_model_kwargs(model_cfg, out_dim=len(data_cfg["future_offsets_min"])),
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(train_cfg["lr"]))

    best_val = float("inf")
    history = []
    horizon_names = [f"t+{m}" for m in data_cfg["future_offsets_min"]]
    peak_power_w = float(data_cfg["peak_power_w"])

    for epoch in range(int(train_cfg["epochs"])):
        model.train()
        total_train = 0.0
        for batch in train_loader:
            sat = batch["satellite"].to(device)
            y = batch["targets"].to(device)
            optimizer.zero_grad()
            out = model(sat)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_train += float(loss.item())

        train_loss = total_train / max(1, len(train_loader))
        val_loss, pred_norm, true_norm = evaluate(model, val_loader, criterion, device)
        pred_w = pred_norm * peak_power_w
        true_w = true_norm * peak_power_w
        metrics = compute_metrics(pred_w, true_w, horizon_names)
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss, **metrics})
        logging.info(
            f"Epoch {epoch + 1}/{train_cfg['epochs']} train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} mean_rmse_W={metrics['mean_rmse_W']:.2f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), run_dir / "best_model.pt")

    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location=device))
    val_loss, pred_norm, true_norm = evaluate(model, val_loader, criterion, device)
    pred_w = pred_norm * peak_power_w
    true_w = true_norm * peak_power_w
    metrics = compute_metrics(pred_w, true_w, horizon_names)
    pd.DataFrame([metrics]).to_csv(run_dir / "metrics_val.csv", index=False)
    ts_val = pd.to_datetime(val_ds.ts_pred_series).reset_index(drop=True)
    pred_df = pd.DataFrame({"ts_pred": ts_val})
    for idx, horizon in enumerate(horizon_names):
        pred_df[f"pv_pred_{horizon}_W"] = pred_w[:, idx]
        pred_df[f"pv_true_{horizon}_W"] = true_w[:, idx]
    pred_df.to_csv(run_dir / "predictions_val.csv", index=False)
    logging.info(f"Saved best model and validation outputs to {run_dir}")


if __name__ == "__main__":
    main()
