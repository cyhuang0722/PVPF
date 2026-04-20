from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..data.dataset import SunConditionedCloudDataset, _parse_jsonish, resolve_existing_path
from ..data.solar_geometry import Calibration, compute_clear_sky_power
from ..models.full_model import SunConditionedStochasticCloudModel
from ..utils.io import ensure_dir, normalize_config_paths, save_json, set_seed, timestamped_run_dir
from ..viz.forecast import save_forecast_band_plot
from ..viz.motion import save_scsn_state_figure
from .metrics import regression_metrics

try:
    from ..losses.scsn import scsn_training_loss
except ModuleNotFoundError:
    def scsn_training_loss(
        pv_mu: torch.Tensor,
        pv_logvar: torch.Tensor,
        target: torch.Tensor,
        kl_loss: torch.Tensor,
        recon_rbr: torch.Tensor,
        target_rbr: torch.Tensor,
        loss_cfg: dict,
    ) -> dict[str, torch.Tensor]:
        target = target.view_as(pv_mu)
        pv_loss = 0.5 * (torch.exp(-pv_logvar) * (target - pv_mu).pow(2) + pv_logvar)
        pv_loss = pv_loss.mean()
        recon_loss = F.l1_loss(recon_rbr, target_rbr)
        total = (
            float(loss_cfg.get("pv_weight", 1.0)) * pv_loss
            + float(loss_cfg.get("kl_weight", 0.02)) * kl_loss
            + float(loss_cfg.get("reconstruction_weight", 0.2)) * recon_loss
        )
        return {
            "total": total,
            "pv": pv_loss,
            "kl": kl_loss,
            "recon": recon_loss,
        }


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


def _persistence_metrics(dataset: SunConditionedCloudDataset) -> dict[str, float]:
    df = dataset.dataframe()
    pred = []
    target = []
    for row in df.itertuples(index=False):
        past_pv = np.asarray(_parse_jsonish(row.past_pv_w), dtype=np.float32)
        pred.append(float(past_pv[-1]))
        target.append(float(row.target_pv_w))
    metrics = regression_metrics(np.asarray(pred, dtype=np.float32), np.asarray(target, dtype=np.float32))
    metrics["n_samples"] = int(len(target))
    return metrics


def _smart_persistence_predictions(dataset: SunConditionedCloudDataset, config: dict) -> np.ndarray:
    df = dataset.dataframe()
    data_cfg = config["data"]
    calib = Calibration.from_json(resolve_existing_path(data_cfg["calibration_json"]))
    anchor_clear = compute_clear_sky_power(
        pd.to_datetime(df["ts_anchor"]).tolist(),
        calib,
        peak_power_w=float(data_cfg["peak_power_w"]),
        floor_w=float(data_cfg["clear_sky_floor_w"]),
    ).to_numpy(dtype=np.float32)
    prev_pv = np.asarray([float(_parse_jsonish(value)[-1]) for value in df["past_pv_w"]], dtype=np.float32)
    prev_csi = prev_pv / np.clip(anchor_clear, a_min=1e-6, a_max=None)
    target_clear = df["target_clear_sky_w"].to_numpy(dtype=np.float32)
    if bool(data_cfg.get("use_clear_sky_index", True)):
        pred = np.clip(prev_csi, 0.0, 1.0) * target_clear
    else:
        pred = prev_pv * (target_clear / np.clip(anchor_clear, a_min=1e-6, a_max=None))
    return np.clip(pred.astype(np.float32), a_min=0.0, a_max=None)


def _smart_persistence_metrics(dataset: SunConditionedCloudDataset, config: dict) -> dict[str, float]:
    pred = _smart_persistence_predictions(dataset, config)
    target = dataset.dataframe()["target_pv_w"].to_numpy(dtype=np.float32)
    metrics = regression_metrics(pred, target)
    metrics["n_samples"] = int(len(target))
    return metrics


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


def _resize_like(tensor: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return F.interpolate(tensor, size=reference.shape[-2:], mode="bilinear", align_corners=False)


def _compute_auxiliary_losses(output: dict[str, torch.Tensor], data: dict[str, torch.Tensor], loss_cfg: dict) -> dict[str, torch.Tensor]:
    pred_mean = output["future_rbr_mean_15min"]
    pred_logvar = output["future_rbr_logvar_15min"]
    future_rbr_nll = pred_mean.new_zeros(())
    future_rbr_l1 = pred_mean.new_zeros(())
    if "future_rbr_change_hotspot" in data and "future_hotspot_valid" in data:
        valid = data["future_hotspot_valid"].view(-1, 1, 1, 1)
        if torch.any(valid > 0):
            target_rbr = _resize_like(data["future_rbr_change_hotspot"], pred_mean).clamp(0.0, 1.0)
            attention = output["attention_map"].detach()
            attention_scale = attention.flatten(1).amax(dim=1).view(-1, 1, 1, 1).clamp_min(1e-6)
            attention_weight = 1.0 + float(loss_cfg.get("rbr_distribution_sun_weight", loss_cfg.get("future_hotspot_sun_weight", 2.0))) * (attention / attention_scale)
            pixel_nll = 0.5 * (torch.exp(-pred_logvar) * (target_rbr - pred_mean).pow(2) + pred_logvar)
            per_sample_nll = (pixel_nll * attention_weight).mean(dim=(1, 2, 3))
            per_sample_l1 = torch.abs(pred_mean - target_rbr).mean(dim=(1, 2, 3))
            future_rbr_nll = (per_sample_nll * valid.view(-1)).sum() / valid.sum().clamp_min(1.0)
            future_rbr_l1 = (per_sample_l1 * valid.view(-1)).sum() / valid.sum().clamp_min(1.0)

    return {
        "future_rbr_nll": future_rbr_nll,
        "future_rbr_l1": future_rbr_l1,
        "total": float(loss_cfg.get("rbr_distribution_weight", 0.0)) * future_rbr_nll,
    }


def _build_visual_item(model: SunConditionedStochasticCloudModel, dataset: SunConditionedCloudDataset, sample_index: int, device: torch.device) -> dict:
    model.eval()
    raw = dataset[sample_index]
    batch = {k: v.unsqueeze(0).to(device) if torch.is_tensor(v) else v for k, v in raw.items()}
    with torch.no_grad():
        out = model(batch["images"], batch["pv_history"], batch["solar_vec"], sun_xy=batch["sun_xy"], target_sun_xy=batch["target_sun_xy"])
    frame = dataset.df.iloc[sample_index]
    pred_quantiles = out["prediction"][0].detach().cpu().numpy()
    return {
        "image": batch["images"][0, -1, :3].detach().cpu().numpy(),
        "attention": out["attention_map"][0, 0].detach().cpu().numpy(),
        "rbr_mean": out["future_rbr_mean_15min"][0, 0].detach().cpu().numpy(),
        "rbr_variance": out["future_rbr_variance_15min"][0, 0].detach().cpu().numpy(),
        "past_rbr_change_hotspot": batch["past_rbr_change_hotspot"][0, 0].detach().cpu().numpy(),
        "future_rbr_change_hotspot": batch["future_rbr_change_hotspot"][0, 0].detach().cpu().numpy(),
        "future_hotspot_valid": bool(batch["future_hotspot_valid"][0].detach().cpu().item() > 0.0),
        "summary_values": {
            "q90-q10": float(pred_quantiles[4] - pred_quantiles[0]),
            "pv sigma": float(out["pv_sigma"][0, 0].detach().cpu().item()),
            "sun mean": float(out["sun_local_rbr_mean"][0, 0].detach().cpu().item()),
            "sun var": float(out["sun_local_rbr_variance"][0, 0].detach().cpu().item()),
        },
        "title": str(frame["ts_target"]),
    }


def _evaluate_split(model: SunConditionedStochasticCloudModel, dataset: SunConditionedCloudDataset, config: dict, device: torch.device) -> tuple[pd.DataFrame, dict[str, float]]:
    loader = _make_loader(dataset, batch_size=int(config["train"]["batch_size"]), num_workers=int(config["train"]["num_workers"]), shuffle=False)
    model.eval()
    use_csi = bool(config["data"].get("use_clear_sky_index", True))
    smart_persistence_w = _smart_persistence_predictions(dataset, config)
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
            interval_width = pred[:, 4] - pred[:, 0]
            pv_sigma = out["pv_sigma"].cpu().numpy().reshape(-1)
            global_rbr_mean = out["global_rbr_mean"].cpu().numpy().reshape(-1)
            sun_local_rbr_mean = out["sun_local_rbr_mean"].cpu().numpy().reshape(-1)
            global_rbr_variance = out["global_rbr_variance"].cpu().numpy().reshape(-1)
            sun_local_rbr_variance = out["sun_local_rbr_variance"].cpu().numpy().reshape(-1)
            target_hotspot = _resize_like(data["future_rbr_change_hotspot"], out["future_rbr_mean_15min"]).clamp(0.0, 1.0)
            pred_mean = out["future_rbr_mean_15min"].clamp(0.0, 1.0)
            rbr_l1 = torch.abs(pred_mean - target_hotspot).mean(dim=(1, 2, 3)).cpu().numpy()
            pred_flat = pred_mean.flatten(1)
            target_flat = target_hotspot.flatten(1)
            pred_centered = pred_flat - pred_flat.mean(dim=1, keepdim=True)
            target_centered = target_flat - target_flat.mean(dim=1, keepdim=True)
            hotspot_corr = (
                (pred_centered * target_centered).mean(dim=1)
                / (pred_centered.std(dim=1).clamp_min(1e-6) * target_centered.std(dim=1).clamp_min(1e-6))
            ).cpu().numpy()
            hotspot_valid = data["future_hotspot_valid"].cpu().numpy()
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
                        "q25": float(pred[i, 1]),
                        "q50": float(pred[i, 2]),
                        "q75": float(pred[i, 3]),
                        "q90": float(pred[i, 4]),
                        "q10_w": float(pred_w[i, 0]),
                        "q25_w": float(pred_w[i, 1]),
                        "q50_w": float(pred_w[i, 2]),
                        "q75_w": float(pred_w[i, 3]),
                        "q90_w": float(pred_w[i, 4]),
                        "interval_width": float(interval_width[i]),
                        "pv_sigma": float(pv_sigma[i]),
                        "global_rbr_mean": float(global_rbr_mean[i]),
                        "sun_local_rbr_mean": float(sun_local_rbr_mean[i]),
                        "global_rbr_variance": float(global_rbr_variance[i]),
                        "sun_local_rbr_variance": float(sun_local_rbr_variance[i]),
                        "smart_persistence_w": float(smart_persistence_w[int(meta_idx[i])]),
                        "future_rbr_l1": float(rbr_l1[i]) if float(hotspot_valid[i]) > 0 else np.nan,
                        "future_rbr_corr": float(hotspot_corr[i]) if float(hotspot_valid[i]) > 0 else np.nan,
                        "sample_index": int(meta_idx[i]),
                    }
                )
    pred_df = pd.DataFrame(rows).sort_values("ts_target").reset_index(drop=True)
    metrics = regression_metrics(pred_df["q50_w"].to_numpy(dtype=np.float32), pred_df["target_pv_w"].to_numpy(dtype=np.float32))
    metrics["n_samples"] = int(len(pred_df))
    metrics["future_rbr_l1"] = float(pred_df["future_rbr_l1"].mean()) if "future_rbr_l1" in pred_df else float("nan")
    metrics["future_rbr_corr"] = float(pred_df["future_rbr_corr"].mean()) if "future_rbr_corr" in pred_df else float("nan")
    metrics["interval_width_vs_sun_variance_corr"] = float(pred_df["interval_width"].corr(pred_df["sun_local_rbr_variance"])) if len(pred_df) > 1 else float("nan")
    metrics["interval_width_vs_global_variance_corr"] = float(pred_df["interval_width"].corr(pred_df["global_rbr_variance"])) if len(pred_df) > 1 else float("nan")
    return pred_df, metrics


def train_model(config: dict) -> Path:
    config = normalize_config_paths(config)
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
    persistence_by_split = {split_name: _persistence_metrics(split_ds) for split_name, split_ds in datasets.items()}
    smart_persistence_by_split = {split_name: _smart_persistence_metrics(split_ds, config) for split_name, split_ds in datasets.items()}
    for split_name, metrics in persistence_by_split.items():
        logger.info("Persistence baseline %s mae=%.2f rmse=%.2f", split_name, metrics["mae"], metrics["rmse"])
    for split_name, metrics in smart_persistence_by_split.items():
        logger.info("Smart persistence baseline %s mae=%.2f rmse=%.2f", split_name, metrics["mae"], metrics["rmse"])

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
        epoch_stats = {"total": [], "pv": [], "kl": [], "recon": [], "future_rbr_nll": [], "future_rbr_l1": []}
        train_pred_w: list[np.ndarray] = []
        train_target_w: list[np.ndarray] = []

        for batch in train_loader:
            data = _to_device(batch, device)
            optimizer.zero_grad()
            out = model(data["images"], data["pv_history"], data["solar_vec"], sun_xy=data["sun_xy"], target_sun_xy=data["target_sun_xy"])
            target_rbr = F.interpolate(data["target_rbr"], size=out["recon_rbr"].shape[-2:], mode="bilinear", align_corners=False)
            losses = scsn_training_loss(
                pv_mu=out["pv_mu"],
                pv_logvar=out["pv_logvar"],
                target=data["target"],
                kl_loss=out["kl_loss"],
                recon_rbr=out["recon_rbr"],
                target_rbr=target_rbr,
                loss_cfg=config["loss"],
            )
            aux_losses = _compute_auxiliary_losses(out, data, config["loss"])
            losses["total"] = losses["total"] + aux_losses["total"]
            losses.update({key: value for key, value in aux_losses.items() if key != "total"})
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(config["loss"].get("grad_clip_norm", 5.0)))
            optimizer.step()

            for key in epoch_stats:
                epoch_stats[key].append(float(losses[key].detach().cpu()))

            q50 = out["prediction"][:, 2].detach().cpu().numpy()
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
            "train_recon_loss": float(np.mean(epoch_stats["recon"])),
            "train_future_rbr_nll": float(np.mean(epoch_stats["future_rbr_nll"])),
            "train_future_rbr_l1": float(np.mean(epoch_stats["future_rbr_l1"])),
            "train_mae_w": float(train_metrics["mae"]),
            "train_rmse_w": float(train_metrics["rmse"]),
            "val_mae_w": float(val_metrics["mae"]),
            "val_rmse_w": float(val_metrics["rmse"]),
            "train_persistence_mae_w": float(persistence_by_split["train"]["mae"]),
            "train_persistence_rmse_w": float(persistence_by_split["train"]["rmse"]),
            "val_persistence_mae_w": float(persistence_by_split["val"]["mae"]),
            "val_persistence_rmse_w": float(persistence_by_split["val"]["rmse"]),
            "train_smart_persistence_mae_w": float(smart_persistence_by_split["train"]["mae"]),
            "train_smart_persistence_rmse_w": float(smart_persistence_by_split["train"]["rmse"]),
            "val_smart_persistence_mae_w": float(smart_persistence_by_split["val"]["mae"]),
            "val_smart_persistence_rmse_w": float(smart_persistence_by_split["val"]["rmse"]),
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
        metrics = {
            **metrics,
            "persistence_mae": float(persistence_by_split[split_name]["mae"]),
            "persistence_rmse": float(persistence_by_split[split_name]["rmse"]),
            "smart_persistence_mae": float(smart_persistence_by_split[split_name]["mae"]),
            "smart_persistence_rmse": float(smart_persistence_by_split[split_name]["rmse"]),
        }
        save_json(run_dir / f"metrics_{split_name}.json", metrics)
        if not pred_df.empty:
            save_forecast_band_plot(pred_df.head(200), run_dir / "figures" / f"forecast_band_{split_name}.png", f"{split_name} forecast band")
        for idx, sample_index in enumerate(_select_visual_indices(pred_df, int(config["train"].get("save_top_k_visualizations", 6)))):
            item = _build_visual_item(model, split_ds, sample_index, device)
            save_scsn_state_figure(
                image=item["image"],
                attention=item["attention"],
                rbr_mean=item["rbr_mean"],
                rbr_variance=item["rbr_variance"],
                past_rbr_change_hotspot=item["past_rbr_change_hotspot"],
                future_rbr_change_hotspot=item["future_rbr_change_hotspot"],
                out_path=run_dir / "figures" / f"cloud_state_{split_name}_{idx:02d}.png",
                title=str(item["title"]),
                future_hotspot_valid=bool(item["future_hotspot_valid"]),
                summary_values=item["summary_values"],
            )
    return run_dir
