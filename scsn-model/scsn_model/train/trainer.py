from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..data.dataset import SunConditionedCloudDataset
from ..losses.scsn import scsn_training_loss
from ..models.full_model import SunConditionedStochasticCloudModel
from ..utils.io import ensure_dir, save_json, set_seed, timestamped_run_dir
from ..viz.forecast import save_forecast_band_plot
from ..viz.motion import save_scsn_state_figure
from .metrics import regression_metrics


def _setup_logger(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger(f"scsn_train_{run_dir.name}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(run_dir / "train.log", encoding="utf-8")
    sh = logging.StreamHandler()
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def _make_loader(dataset: SunConditionedCloudDataset, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def _to_device(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def _target_to_w(values: np.ndarray, clear_sky_w: np.ndarray, use_clear_sky_index: bool) -> np.ndarray:
    clipped = np.clip(values, a_min=0.0, a_max=1.2 if use_clear_sky_index else None)
    if use_clear_sky_index:
        clipped = clipped * clear_sky_w
    return np.clip(clipped, a_min=0.0, a_max=None)


def _select_visual_indices(pred_df: pd.DataFrame, limit: int) -> list[int]:
    if pred_df.empty or limit <= 0:
        return []
    ranked = pred_df.assign(abs_error_w=(pred_df["q50_w"] - pred_df["target_pv_w"]).abs())
    chosen = pd.concat(
        [
            ranked.sort_values("abs_error_w", ascending=False).head(limit // 2 + limit % 2),
            ranked.sort_values("abs_error_w", ascending=True).head(limit // 2),
        ]
    )
    return list(dict.fromkeys(int(v) for v in chosen["sample_index"].tolist()))[:limit]


def _build_visual_item(model: SunConditionedStochasticCloudModel, dataset: SunConditionedCloudDataset, sample_index: int, device: torch.device) -> dict[str, np.ndarray | str]:
    model.eval()
    raw = dataset[sample_index]
    batch = {k: v.unsqueeze(0).to(device) if torch.is_tensor(v) else v for k, v in raw.items()}
    with torch.no_grad():
        out = model(batch["images"], batch["pv_history"], batch["solar_vec"], sun_xy=batch["sun_xy"], target_sun_xy=batch["target_sun_xy"])
    frame = dataset.df.iloc[sample_index]
    return {
        "image": batch["images"][0, -1, :3].detach().cpu().numpy(),
        "attention": out["attention_map"][0, 0].detach().cpu().numpy(),
        "transmission": out["transmission_maps"][0, -1, 0].detach().cpu().numpy(),
        "opacity": out["opacity_maps"][0, -1, 0].detach().cpu().numpy(),
        "gap": out["gap_maps"][0, -1, 0].detach().cpu().numpy(),
        "sun_occlusion": out["sun_occlusion"][0].detach().cpu().numpy(),
        "motion_u": out["motion_fields"][0, -1, 0].detach().cpu().numpy(),
        "motion_v": out["motion_fields"][0, -1, 1].detach().cpu().numpy(),
        "title": str(frame["ts_target"]),
    }


def _evaluate_split(model: SunConditionedStochasticCloudModel, dataset: SunConditionedCloudDataset, config: dict, device: torch.device) -> tuple[pd.DataFrame, dict[str, float]]:
    loader = _make_loader(dataset, batch_size=int(config["train"]["batch_size"]), num_workers=int(config["train"]["num_workers"]), shuffle=False)
    model.eval()
    use_csi = bool(config["data"].get("use_clear_sky_index", True))
    rows: list[dict] = []
    with torch.no_grad():
        for batch in loader:
            data = _to_device(batch, device)
            out = model(data["images"], data["pv_history"], data["solar_vec"], sun_xy=data["sun_xy"], target_sun_xy=data["target_sun_xy"])
            pred = out["prediction"].cpu().numpy()
            clear_sky_w = data["target_clear_sky_w"].cpu().numpy()
            pred_w = _target_to_w(pred, clear_sky_w[:, None], use_csi)
            target_w = data["target_pv_w"].cpu().numpy()
            meta_idx = data["meta_index"].cpu().numpy()
            for i in range(len(meta_idx)):
                frame = dataset.df.iloc[int(meta_idx[i])]
                rows.append(
                    {
                        "ts_anchor": frame["ts_anchor"],
                        "ts_target": frame["ts_target"],
                        "target_value": float(data["target"][i].cpu().item()),
                        "target_pv_w": float(target_w[i]),
                        "target_clear_sky_w": float(clear_sky_w[i]),
                        "q10": float(pred[i, 0]),
                        "q50": float(pred[i, 1]),
                        "q90": float(pred[i, 2]),
                        "q10_w": float(pred_w[i, 0]),
                        "q50_w": float(pred_w[i, 1]),
                        "q90_w": float(pred_w[i, 2]),
                        "sample_index": int(meta_idx[i]),
                    }
                )
    pred_df = pd.DataFrame(rows).sort_values("ts_target").reset_index(drop=True)
    metrics = regression_metrics(pred_df["q50_w"].to_numpy(dtype=np.float32), pred_df["target_pv_w"].to_numpy(dtype=np.float32))
    return pred_df, metrics


def train_model(config: dict) -> Path:
    set_seed(int(config["seed"]))
    run_dir = timestamped_run_dir(config["train"]["artifact_root"])
    ensure_dir(run_dir / "figures")
    logger = _setup_logger(run_dir)
    save_json(run_dir / "run_config.json", config)

    device_cfg = str(config.get("device", "auto")).lower()
    device = torch.device("cuda" if device_cfg == "auto" and torch.cuda.is_available() else device_cfg if device_cfg != "auto" else "cpu")
    logger.info("Using device: %s", device)

    datasets = {
        split: SunConditionedCloudDataset(
            csv_path=config["data"]["samples_csv"],
            split=split,
            image_size=tuple(config["data"]["image_size"]),
            sky_mask_path=config["data"].get("sky_mask_path"),
            peak_power_w=float(config["data"]["peak_power_w"]),
        )
        for split in ("train", "val", "test")
    }
    logger.info("Dataset sizes train=%d val=%d test=%d", len(datasets["train"]), len(datasets["val"]), len(datasets["test"]))

    train_loader = _make_loader(datasets["train"], batch_size=int(config["train"]["batch_size"]), num_workers=int(config["train"]["num_workers"]), shuffle=True)
    model = SunConditionedStochasticCloudModel(config["model"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["train"]["learning_rate"]), weight_decay=float(config["train"]["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(int(config["train"]["epochs"]), 1))

    best_val = float("inf")
    early_delta = float(config["train"].get("early_stopping_min_delta", 20.0))
    min_epochs = int(config["train"].get("early_stopping_min_epochs", 10))
    no_improve = 0
    history: list[dict] = []
    use_csi = bool(config["data"].get("use_clear_sky_index", True))

    for epoch in range(1, int(config["train"]["epochs"]) + 1):
        model.train()
        epoch_stats = {"total": [], "pv": [], "kl": [], "motion": [], "recon": []}
        train_pred_w: list[np.ndarray] = []
        train_target_w: list[np.ndarray] = []

        for batch in train_loader:
            data = _to_device(batch, device)
            optimizer.zero_grad()
            out = model(data["images"], data["pv_history"], data["solar_vec"], sun_xy=data["sun_xy"], target_sun_xy=data["target_sun_xy"])
            target_rbr = F.interpolate(data["target_rbr"], size=out["recon_rbr"].shape[-2:], mode="bilinear", align_corners=False)
            losses = scsn_training_loss(
                prediction=out["prediction"],
                target=data["target"],
                kl_loss=out["kl_loss"],
                motion_reg_loss=out["motion_reg_loss"],
                recon_rbr=out["recon_rbr"],
                target_rbr=target_rbr,
                loss_cfg=config["loss"],
            )
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(config["loss"].get("grad_clip_norm", 5.0)))
            optimizer.step()

            for key in epoch_stats:
                epoch_stats[key].append(float(losses[key].detach().cpu()))

            q50 = out["prediction"][:, 1].detach().cpu().numpy()
            q50_w = _target_to_w(q50, data["target_clear_sky_w"].detach().cpu().numpy(), use_csi)
            train_pred_w.append(q50_w.astype(np.float32))
            train_target_w.append(data["target_pv_w"].detach().cpu().numpy().astype(np.float32))

        scheduler.step()
        val_pred, val_metrics = _evaluate_split(model, datasets["val"], config, device)
        train_metrics = regression_metrics(np.concatenate(train_pred_w, axis=0), np.concatenate(train_target_w, axis=0))
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(epoch_stats["total"])),
            "train_pv_loss": float(np.mean(epoch_stats["pv"])),
            "train_kl_loss": float(np.mean(epoch_stats["kl"])),
            "train_motion_loss": float(np.mean(epoch_stats["motion"])),
            "train_recon_loss": float(np.mean(epoch_stats["recon"])),
            "train_mae_w": float(train_metrics["mae"]),
            "train_rmse_w": float(train_metrics["rmse"]),
            "val_mae_w": float(val_metrics["mae"]),
            "val_rmse_w": float(val_metrics["rmse"]),
            "lr": float(scheduler.get_last_lr()[0]),
        }
        history.append(row)
        logger.info(
            "Epoch %d train_rmse=%.2f val_rmse=%.2f train_mae=%.2f val_mae=%.2f train_loss=%.5f",
            epoch,
            row["train_rmse_w"],
            row["val_rmse_w"],
            row["train_mae_w"],
            row["val_mae_w"],
            row["train_loss"],
        )

        improved = val_metrics["rmse"] < (best_val - early_delta)
        if improved:
            best_val = val_metrics["rmse"]
            no_improve = 0
            torch.save(model.state_dict(), run_dir / "best_model.pt")
        else:
            no_improve += 1
            if epoch >= min_epochs and no_improve >= int(config["train"]["early_stopping_patience"]):
                logger.info("Early stopping triggered at epoch %d", epoch)
                break

    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location=device))
    for split_name, split_ds in datasets.items():
        pred_df, metrics = _evaluate_split(model, split_ds, config, device)
        pred_df.to_csv(run_dir / f"predictions_{split_name}.csv", index=False)
        save_json(run_dir / f"metrics_{split_name}.json", metrics)
        if not pred_df.empty:
            save_forecast_band_plot(pred_df.head(200), run_dir / "figures" / f"forecast_band_{split_name}.png", f"{split_name} forecast band")
        for idx, sample_index in enumerate(_select_visual_indices(pred_df, int(config["train"].get("save_top_k_visualizations", 6)))):
            item = _build_visual_item(model, split_ds, sample_index, device)
            save_scsn_state_figure(
                image=item["image"],
                attention=item["attention"],
                transmission=item["transmission"],
                opacity=item["opacity"],
                gap=item["gap"],
                motion_u=item["motion_u"],
                motion_v=item["motion_v"],
                sun_occlusion=item["sun_occlusion"],
                out_path=run_dir / "figures" / f"cloud_state_{split_name}_{idx:02d}.png",
                title=str(item["title"]),
            )
    return run_dir
