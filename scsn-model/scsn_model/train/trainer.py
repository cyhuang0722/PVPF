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
from ..losses.quantile import quantile_crossing_penalty, quantile_loss
from ..models.full_model import SunConditionedStochasticCloudModel
from ..utils.io import ensure_dir, normalize_config_paths, save_json, set_seed, timestamped_run_dir
from ..viz.forecast import save_forecast_band_plot
from ..viz.motion import save_scsn_state_figure
from .metrics import regression_metrics

try:
    from ..losses.scsn import scsn_training_loss
except ModuleNotFoundError:
    def scsn_training_loss(
        prediction: torch.Tensor,
        target: torch.Tensor,
        kl_loss: torch.Tensor,
        motion_reg_loss: torch.Tensor,
        recon_rbr: torch.Tensor,
        target_rbr: torch.Tensor,
        loss_cfg: dict,
    ) -> dict[str, torch.Tensor]:
        pv_loss = quantile_loss(prediction, target, [0.1, 0.5, 0.9])
        pv_loss = pv_loss + float(loss_cfg.get("crossing_weight", 0.2)) * quantile_crossing_penalty(prediction)
        recon_loss = F.l1_loss(recon_rbr, target_rbr)
        total = (
            float(loss_cfg.get("pv_weight", 1.0)) * pv_loss
            + float(loss_cfg.get("kl_weight", 0.02)) * kl_loss
            + float(loss_cfg.get("motion_weight", 0.01)) * motion_reg_loss
            + float(loss_cfg.get("reconstruction_weight", 0.2)) * recon_loss
        )
        return {
            "total": total,
            "pv": pv_loss,
            "kl": kl_loss,
            "motion": motion_reg_loss,
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


def _make_sampling_grid(flow: torch.Tensor) -> torch.Tensor:
    batch, _, height, width = flow.shape
    device = flow.device
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, height, device=device),
        torch.linspace(-1.0, 1.0, width, device=device),
        indexing="ij",
    )
    base_grid = torch.stack([xx, yy], dim=-1).unsqueeze(0).expand(batch, -1, -1, -1)
    flow_x = flow[:, 0] * (2.0 / max(width - 1, 1))
    flow_y = flow[:, 1] * (2.0 / max(height - 1, 1))
    return torch.stack([base_grid[..., 0] - flow_x, base_grid[..., 1] - flow_y], dim=-1)


def _warp_with_flow(image: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    return F.grid_sample(image, _make_sampling_grid(flow), mode="bilinear", padding_mode="border", align_corners=True)


def _compute_auxiliary_losses(output: dict[str, torch.Tensor], data: dict[str, torch.Tensor], loss_cfg: dict) -> dict[str, torch.Tensor]:
    current_opacity = output["current_opacity"]
    current_gap = output["current_gap"]
    current_transmission = output["current_transmission"]
    current_cloud_prob = output["current_cloud_prob"]

    opacity_proxy = _resize_like(data["opacity_proxy"], current_opacity)
    gap_proxy = _resize_like(data["gap_proxy"], current_gap)
    transmission_proxy = _resize_like(data["transmission_proxy"], current_transmission)

    opacity_proxy_loss = F.l1_loss(current_opacity, opacity_proxy)
    gap_proxy_loss = F.l1_loss(current_gap, gap_proxy)
    transmission_proxy_loss = F.l1_loss(current_transmission, transmission_proxy)

    prev_rbr = _resize_like(data["prev_rbr"], current_opacity)
    curr_rbr = _resize_like(data["target_rbr"], current_opacity)
    flow_now = output["motion_fields"][:, 0]
    warped_prev = _warp_with_flow(prev_rbr, flow_now)
    motion_warp_loss = F.l1_loss(warped_prev, curr_rbr)

    cloud_mask_loss = current_cloud_prob.new_zeros(())
    if "cloud_mask" in data and "cloud_mask_valid" in data:
        valid = data["cloud_mask_valid"].view(-1, 1, 1, 1)
        if torch.any(valid > 0):
            pseudo_cloud = _resize_like(data["cloud_mask"], current_cloud_prob).clamp(0.0, 1.0)
            valid_count = valid.sum().clamp_min(1.0)
            cloud_bce = F.binary_cross_entropy(current_cloud_prob.clamp(1e-4, 1.0 - 1e-4), pseudo_cloud, reduction="none")
            cloud_pixel_loss = (cloud_bce * valid).mean(dim=(1, 2, 3)).sum() / valid_count
            pred_fraction = current_cloud_prob.mean(dim=(1, 2, 3))
            target_fraction = pseudo_cloud.mean(dim=(1, 2, 3))
            cloud_fraction_loss = (torch.abs(pred_fraction - target_fraction) * valid.view(-1)).sum() / valid_count
            cloud_mask_loss = cloud_pixel_loss + float(loss_cfg.get("cloud_fraction_weight", 0.25)) * cloud_fraction_loss

    return {
        "opacity_proxy": opacity_proxy_loss,
        "gap_proxy": gap_proxy_loss,
        "transmission_proxy": transmission_proxy_loss,
        "motion_warp": motion_warp_loss,
        "cloud_mask": cloud_mask_loss,
        "total": (
            float(loss_cfg.get("opacity_proxy_weight", 0.15)) * opacity_proxy_loss
            + float(loss_cfg.get("gap_proxy_weight", 0.10)) * gap_proxy_loss
            + float(loss_cfg.get("transmission_proxy_weight", 0.15)) * transmission_proxy_loss
            + float(loss_cfg.get("motion_warp_weight", 0.10)) * motion_warp_loss
            + float(loss_cfg.get("cloud_mask_weight", 0.0)) * cloud_mask_loss
        ),
    }


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
        "current_cloud_prob": out["current_cloud_prob"][0, 0].detach().cpu().numpy(),
        "future_cloud_prob": out["future_cloud_prob_maps"][0, -1, 0].detach().cpu().numpy(),
        "future_sun_cloud_prob": out["future_sun_cloud_prob"][0].detach().cpu().numpy(),
        "cloud_mask": batch["cloud_mask"][0, 0].detach().cpu().numpy(),
        "cloud_mask_valid": bool(batch["cloud_mask_valid"][0].detach().cpu().item() > 0.0),
        "motion_hotspot": out["future_motion_hotspot_maps"][0, -1, 0].detach().cpu().numpy(),
        "future_cloud_uncertainty": out["future_cloud_uncertainty_maps"][0, -1, 0].detach().cpu().numpy(),
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
                        "smart_persistence_w": float(smart_persistence_w[int(meta_idx[i])]),
                        "sample_index": int(meta_idx[i]),
                    }
                )
    pred_df = pd.DataFrame(rows).sort_values("ts_target").reset_index(drop=True)
    metrics = regression_metrics(pred_df["q50_w"].to_numpy(dtype=np.float32), pred_df["target_pv_w"].to_numpy(dtype=np.float32))
    metrics["n_samples"] = int(len(pred_df))
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
            cloud_mask_manifest_path=config["data"].get("cloud_mask_manifest_path"),
            cloud_mask_sky_mask_path=config["data"].get("cloud_mask_sky_mask_path", config["data"].get("sky_mask_path")),
        )
        for split in ("train", "val", "test")
    }
    logger.info("Dataset sizes train=%d val=%d test=%d", len(datasets["train"]), len(datasets["val"]), len(datasets["test"]))
    for split_name, split_ds in datasets.items():
        if split_ds.cloud_mask_supervisor is not None:
            logger.info("Cloud-mask manifest %s resolved to %s", split_name, split_ds.cloud_mask_supervisor.manifest_path)
        valid_masks, total_masks = split_ds.cloud_mask_coverage()
        logger.info("Cloud-mask coverage %s=%d/%d", split_name, valid_masks, total_masks)
        if split_name == "train" and float(config["loss"].get("cloud_mask_weight", 0.0)) > 0.0 and valid_masks == 0:
            sample_key = "<empty train split>"
            if len(split_ds.df) > 0:
                sample_paths = _parse_jsonish(split_ds.df.iloc[0]["img_paths"])
                sample_key = Path(str(sample_paths[-1])).name
            manifest_keys = []
            if split_ds.cloud_mask_supervisor is not None:
                manifest_keys = split_ds.cloud_mask_supervisor.available_keys()[:5]
            raise RuntimeError(
                "cloud_mask_weight is > 0 but no train samples matched cloud masks. "
                f"Check data.cloud_mask_manifest_path and image path roots. "
                f"First train current-frame key={sample_key!r}; first manifest keys={manifest_keys!r}."
            )
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
        epoch_stats = {"total": [], "pv": [], "kl": [], "motion": [], "recon": [], "opacity_proxy": [], "gap_proxy": [], "transmission_proxy": [], "motion_warp": [], "cloud_mask": []}
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
            aux_losses = _compute_auxiliary_losses(out, data, config["loss"])
            losses["total"] = losses["total"] + aux_losses["total"]
            losses.update({key: value for key, value in aux_losses.items() if key != "total"})
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
            "train_opacity_proxy_loss": float(np.mean(epoch_stats["opacity_proxy"])),
            "train_gap_proxy_loss": float(np.mean(epoch_stats["gap_proxy"])),
            "train_transmission_proxy_loss": float(np.mean(epoch_stats["transmission_proxy"])),
            "train_motion_warp_loss": float(np.mean(epoch_stats["motion_warp"])),
            "train_cloud_mask_loss": float(np.mean(epoch_stats["cloud_mask"])),
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
                current_cloud_prob=item["current_cloud_prob"],
                future_cloud_prob=item["future_cloud_prob"],
                motion_hotspot=item["motion_hotspot"],
                future_sun_cloud_prob=item["future_sun_cloud_prob"],
                out_path=run_dir / "figures" / f"cloud_state_{split_name}_{idx:02d}.png",
                title=str(item["title"]),
                cloud_mask=item["cloud_mask"],
                cloud_mask_valid=bool(item["cloud_mask_valid"]),
                future_cloud_uncertainty=item["future_cloud_uncertainty"],
            )
    return run_dir
