from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..data.dataset import SunConditionedPVDataset
from ..losses.flow import masked_patch_cosine_loss
from ..losses.motion import motion_regularization_loss
from ..models.full_model import MinimalSunConditionedPVModel
from ..utils.io import ensure_dir, save_json, set_seed, timestamped_run_dir
from ..viz.forecast import save_forecast_band_plot
from ..viz.motion import save_motion_attention_figure, save_patch_motion_comparison_figure
from .metrics import regression_metrics


def _setup_logger(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger(f"new_model_train_{run_dir.name}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(run_dir / "train.log", encoding="utf-8")
    sh = logging.StreamHandler()
    fh.setFormatter(fmt)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def _make_loader(dataset: SunConditionedPVDataset, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def _to_device(batch: dict, device: torch.device) -> dict:
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


def _target_to_w(prediction: np.ndarray, clear_sky_w: np.ndarray, use_clear_sky_index: bool) -> np.ndarray:
    prediction = np.clip(prediction, a_min=0.0, a_max=1.0 if use_clear_sky_index else None)
    if use_clear_sky_index:
        values_w = prediction * clear_sky_w
    else:
        values_w = prediction
    return np.clip(values_w, a_min=0.0, a_max=None)


def _visual_category_counts(train_cfg: dict) -> dict[str, int]:
    configured = train_cfg.get("visualization_category_counts")
    if isinstance(configured, dict):
        counts = {
            "best": int(configured.get("best", 0)),
            "largest_error": int(configured.get("largest_error", 0)),
            "representative": int(configured.get("representative", 0)),
            "midday": int(configured.get("midday", 0)),
        }
    else:
        total = int(train_cfg.get("save_top_k_visualizations", 8))
        base = total // 4
        remainder = total % 4
        counts = {
            "best": base + (1 if remainder > 0 else 0),
            "largest_error": base + (1 if remainder > 1 else 0),
            "representative": base + (1 if remainder > 2 else 0),
            "midday": base,
        }
    return counts


def _select_visual_samples(pred_df: pd.DataFrame, train_cfg: dict) -> list[dict[str, int | str]]:
    if pred_df.empty:
        return []

    counts = _visual_category_counts(train_cfg)
    total_requested = sum(counts.values())
    if total_requested <= 0:
        return []

    work_df = pred_df.copy()
    work_df["abs_error_w"] = (work_df["pred_w"] - work_df["target_pv_w"]).abs()
    median_abs_error = float(work_df["abs_error_w"].median())
    ts_target = pd.to_datetime(work_df["ts_target"])
    work_df["seconds_to_noon"] = (
        ts_target.dt.hour * 3600 + ts_target.dt.minute * 60 + ts_target.dt.second - 12 * 3600
    ).abs()
    work_df["representative_gap"] = (work_df["abs_error_w"] - median_abs_error).abs()

    ranked_frames = {
        "best": work_df.sort_values(["abs_error_w", "ts_target"], ascending=[True, True]),
        "largest_error": work_df.sort_values(["abs_error_w", "ts_target"], ascending=[False, True]),
        "representative": work_df.sort_values(["representative_gap", "seconds_to_noon", "ts_target"]),
        "midday": work_df.sort_values(["seconds_to_noon", "abs_error_w", "ts_target"]),
    }

    chosen: list[dict[str, int | str]] = []
    used_indices: set[int] = set()

    for category in ("best", "largest_error", "representative", "midday"):
        remaining = counts.get(category, 0)
        if remaining <= 0:
            continue
        for _, row in ranked_frames[category].iterrows():
            sample_index = int(row["sample_index"])
            if sample_index in used_indices:
                continue
            chosen.append({"sample_index": sample_index, "category": category})
            used_indices.add(sample_index)
            remaining -= 1
            if remaining == 0:
                break

    if len(chosen) < total_requested:
        fallback = ranked_frames["largest_error"]
        for _, row in fallback.iterrows():
            sample_index = int(row["sample_index"])
            if sample_index in used_indices:
                continue
            chosen.append({"sample_index": sample_index, "category": "largest_error"})
            used_indices.add(sample_index)
            if len(chosen) == total_requested:
                break

    return chosen


def _build_visual_item(
    model: MinimalSunConditionedPVModel,
    dataset: SunConditionedPVDataset,
    sample_index: int,
    device: torch.device,
) -> dict[str, np.ndarray | str]:
    model.eval()
    raw = dataset[sample_index]
    batch = {}
    for key, value in raw.items():
        if torch.is_tensor(value):
            batch[key] = value.unsqueeze(0).to(device)
        else:
            batch[key] = value

    with torch.no_grad():
        out = model(
            batch["images"],
            batch["pv_history"],
            batch["solar_vec"],
            sun_xy=batch["sun_xy"],
            sun_angles=batch["sun_angles"],
        )

    frame = dataset.df.iloc[sample_index]
    return {
        "images": batch["images"][0].detach().cpu().numpy(),
        "image": batch["images"][0, -1].detach().cpu().numpy(),
        "patch_motion_pred": out["patch_motion_pred"][0].detach().cpu().numpy(),
        "patch_motion_teacher": batch["patch_motion_teacher"][0].detach().cpu().numpy(),
        "patch_motion_mask": batch["patch_motion_mask"][0].detach().cpu().numpy(),
        "attention": out["attention_map"][0].detach().cpu().numpy(),
        "sun_prior": out["sun_prior"][0, 0].detach().cpu().numpy(),
        "title": str(frame["ts_target"]),
    }


def _evaluate_split(
    model: MinimalSunConditionedPVModel,
    dataset: SunConditionedPVDataset,
    config: dict,
    device: torch.device,
) -> tuple[pd.DataFrame, dict[str, float]]:
    loader = _make_loader(
        dataset,
        batch_size=int(config["train"]["batch_size"]),
        num_workers=int(config["train"]["num_workers"]),
        shuffle=False,
    )
    model.eval()
    use_csi = bool(config["data"].get("use_clear_sky_index", True))
    rows: list[dict] = []

    with torch.no_grad():
        for batch in loader:
            data = _to_device(batch, device)
            out = model(
                data["images"],
                data["pv_history"],
                data["solar_vec"],
                sun_xy=data["sun_xy"],
                sun_angles=data["sun_angles"],
            )
            pred = out["prediction"].squeeze(-1).cpu().numpy()
            clear_sky_w = data["target_clear_sky_w"].cpu().numpy()
            pred_w = _target_to_w(pred, clear_sky_w, use_csi)
            pred_eval = np.clip(pred, a_min=0.0, a_max=1.0 if use_csi else None)
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
                        "pred_value": float(pred_eval[i]),
                        "pred_value_raw": float(pred[i]),
                        "pred_w": float(pred_w[i]),
                        "sample_index": int(meta_idx[i]),
                    }
                )

    pred_df = pd.DataFrame(rows).sort_values("ts_target").reset_index(drop=True)
    true_w = pred_df["target_pv_w"].to_numpy(dtype=np.float32)
    metrics = regression_metrics(pred_df["pred_w"].to_numpy(dtype=np.float32), true_w)
    return pred_df, metrics


def train_model(config: dict) -> Path:
    set_seed(int(config["seed"]))
    run_dir = timestamped_run_dir(config["train"]["artifact_root"])
    ensure_dir(run_dir / "figures")
    logger = _setup_logger(run_dir)
    save_json(run_dir / "run_config.json", config)

    device_cfg = str(config.get("device", "auto")).lower()
    if device_cfg == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_cfg)
    logger.info("Using device: %s", device)

    train_ds = SunConditionedPVDataset(
        csv_path=config["data"]["samples_csv"],
        split="train",
        image_size=tuple(config["data"]["image_size"]),
        sky_mask_path=config["data"].get("sky_mask_path"),
        peak_power_w=float(config["data"]["peak_power_w"]),
        camera_index_csv=config["data"].get("camera_index_csv"),
        image_match_tolerance_sec=int(config["data"].get("image_match_tolerance_sec", 75)),
        motion_teacher_pairs_min=config["data"].get("motion_teacher_pairs_min"),
        patch_grid_size=int(config["model"].get("patch_grid_size", 8)),
        teacher_flow_resolution=int(config["data"].get("teacher_flow_resolution", 64)),
        teacher_max_displacement_px=int(config["data"].get("teacher_max_displacement_px", 2)),
        teacher_conf_threshold=float(config["data"].get("teacher_conf_threshold", 0.25)),
        teacher_min_patch_vectors=int(config["data"].get("teacher_min_patch_vectors", 6)),
        teacher_min_magnitude=float(config["data"].get("teacher_min_magnitude", 0.15)),
    )
    val_ds = SunConditionedPVDataset(
        csv_path=config["data"]["samples_csv"],
        split="val",
        image_size=tuple(config["data"]["image_size"]),
        sky_mask_path=config["data"].get("sky_mask_path"),
        peak_power_w=float(config["data"]["peak_power_w"]),
        camera_index_csv=config["data"].get("camera_index_csv"),
        image_match_tolerance_sec=int(config["data"].get("image_match_tolerance_sec", 75)),
        motion_teacher_pairs_min=config["data"].get("motion_teacher_pairs_min"),
        patch_grid_size=int(config["model"].get("patch_grid_size", 8)),
        teacher_flow_resolution=int(config["data"].get("teacher_flow_resolution", 64)),
        teacher_max_displacement_px=int(config["data"].get("teacher_max_displacement_px", 2)),
        teacher_conf_threshold=float(config["data"].get("teacher_conf_threshold", 0.25)),
        teacher_min_patch_vectors=int(config["data"].get("teacher_min_patch_vectors", 6)),
        teacher_min_magnitude=float(config["data"].get("teacher_min_magnitude", 0.15)),
    )
    test_ds = SunConditionedPVDataset(
        csv_path=config["data"]["samples_csv"],
        split="test",
        image_size=tuple(config["data"]["image_size"]),
        sky_mask_path=config["data"].get("sky_mask_path"),
        peak_power_w=float(config["data"]["peak_power_w"]),
        camera_index_csv=config["data"].get("camera_index_csv"),
        image_match_tolerance_sec=int(config["data"].get("image_match_tolerance_sec", 75)),
        motion_teacher_pairs_min=config["data"].get("motion_teacher_pairs_min"),
        patch_grid_size=int(config["model"].get("patch_grid_size", 8)),
        teacher_flow_resolution=int(config["data"].get("teacher_flow_resolution", 64)),
        teacher_max_displacement_px=int(config["data"].get("teacher_max_displacement_px", 2)),
        teacher_conf_threshold=float(config["data"].get("teacher_conf_threshold", 0.25)),
        teacher_min_patch_vectors=int(config["data"].get("teacher_min_patch_vectors", 6)),
        teacher_min_magnitude=float(config["data"].get("teacher_min_magnitude", 0.15)),
    )
    logger.info("Dataset sizes train=%d val=%d test=%d", len(train_ds), len(val_ds), len(test_ds))

    train_loader = _make_loader(
        train_ds,
        batch_size=int(config["train"]["batch_size"]),
        num_workers=int(config["train"]["num_workers"]),
        shuffle=True,
    )

    model = MinimalSunConditionedPVModel(config["model"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["train"]["learning_rate"]),
        weight_decay=float(config["train"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(int(config["train"]["epochs"]), 1))

    best_val = float("inf")
    no_improve = 0
    history: list[dict] = []
    min_epochs = int(config["train"].get("early_stopping_min_epochs", 12))
    early_delta = float(config["train"].get("early_stopping_min_delta", 50.0))

    for epoch in range(1, int(config["train"]["epochs"]) + 1):
        model.train()
        epoch_losses: list[float] = []
        epoch_reg_losses: list[float] = []
        epoch_motion_losses: list[float] = []

        for batch in train_loader:
            data = _to_device(batch, device)
            optimizer.zero_grad()
            out = model(
                data["images"],
                data["pv_history"],
                data["solar_vec"],
                sun_xy=data["sun_xy"],
                sun_angles=data["sun_angles"],
            )
            pred = out["prediction"].squeeze(-1)
            reg_loss = F.mse_loss(pred, data["target"])
            arch = config["model"].get("architecture", "minimal_sun_conditioned")
            if arch == "dual_timescale_sun_aware":
                motion_loss = masked_patch_cosine_loss(
                    out["patch_motion_pred"],
                    data["patch_motion_teacher"],
                    data["patch_motion_mask"],
                )
                total = reg_loss
                if bool(config["loss"].get("enable_motion_supervision", True)):
                    total = total + float(config["loss"].get("motion_supervision_weight", 0.05)) * motion_loss
            else:
                motion_loss, motion_stats = motion_regularization_loss(
                    out["frame_features"],
                    out["motion_fields"],
                    warp_weight=float(config["loss"]["warp_weight"]),
                    smooth_weight=float(config["loss"]["smooth_weight"]),
                )
                total = reg_loss + float(config["loss"]["motion_loss_weight"]) * motion_loss
            total.backward()
            optimizer.step()

            epoch_losses.append(float(total.detach().cpu()))
            epoch_reg_losses.append(float(reg_loss.detach().cpu()))
            epoch_motion_losses.append(float(motion_loss.detach().cpu()))

        scheduler.step()

        val_pred, val_metrics = _evaluate_split(model, val_ds, config, device)
        history_row = {
            "epoch": epoch,
            "train_loss": float(np.mean(epoch_losses)),
            "train_regression_loss": float(np.mean(epoch_reg_losses)),
            "train_motion_loss": float(np.mean(epoch_motion_losses)),
            "val_mae_w": float(val_metrics["mae"]),
            "val_rmse_w": float(val_metrics["rmse"]),
            "lr": float(scheduler.get_last_lr()[0]),
        }
        history.append(history_row)
        logger.info(
            "Epoch %d train_loss=%.5f val_rmse=%.2f",
            epoch,
            history_row["train_loss"],
            history_row["val_rmse_w"],
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
    for split_name, split_ds in (("train", train_ds), ("val", val_ds), ("test", test_ds)):
        pred_df, metrics = _evaluate_split(model, split_ds, config, device)
        pred_df.to_csv(run_dir / f"predictions_{split_name}.csv", index=False)
        save_json(run_dir / f"metrics_{split_name}.json", metrics)

        if not pred_df.empty:
            save_forecast_band_plot(
                pred_df.head(200),
                run_dir / "figures" / f"forecast_band_{split_name}.png",
                title=f"{split_name} forecast band",
            )
        selected_visuals = _select_visual_samples(pred_df, config["train"])
        for idx, selection in enumerate(selected_visuals):
            item = _build_visual_item(model, split_ds, int(selection["sample_index"]), device)
            if config["model"].get("architecture", "minimal_sun_conditioned") != "dual_timescale_sun_aware":
                save_motion_attention_figure(
                    image=item["image"],
                    motion=np.zeros((2, 8, 8), dtype=np.float32),
                    attention=item["attention"],
                    out_path=run_dir
                    / "figures"
                    / f"motion_attention_{split_name}_{idx:02d}_{selection['category']}.png",
                    title=f"{item['title']} | {selection['category']}",
                )
            save_patch_motion_comparison_figure(
                image_current=item["images"][-1],
                image_prev_1=item["images"][-2],
                image_prev_2=item["images"][-3],
                patch_motion_pred=item["patch_motion_pred"],
                patch_motion_teacher=item["patch_motion_teacher"],
                patch_motion_mask=item["patch_motion_mask"],
                sun_prior=item["sun_prior"],
                out_path=run_dir
                / "figures"
                / f"patch_motion_compare_{split_name}_{idx:02d}_{selection['category']}.png",
                title=f"{item['title']} | {selection['category']}",
            )

    return run_dir
