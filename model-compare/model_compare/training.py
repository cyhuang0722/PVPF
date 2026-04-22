from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import ImageSequenceDataset, load_frames
from .evaluation import calibrate_scale_multiplier, evaluate, make_loader, to_device
from .models import build_model, gaussian_nll, vae_loss
from .utils import resolve_device, save_json, set_seed, timestamped_run_dir


def setup_logger(run_dir: Path, artifact_root: str | Path, model_name: str) -> logging.Logger:
    logger = logging.getLogger(f"model_compare_{model_name}_{run_dir.name}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (
        logging.StreamHandler(),
        logging.FileHandler(run_dir / "train.log", encoding="utf-8"),
        logging.FileHandler(Path(artifact_root) / model_name / "latest_train.log", mode="w", encoding="utf-8"),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def build_datasets(config: dict, model_name: str, max_samples: int = 0) -> dict[str, ImageSequenceDataset]:
    frames = load_frames(config["data"]["samples_csv"], max_samples=max_samples)
    data_cfg = config["data"]
    sequence_mode = "latest" if model_name == "image_regressor" else "sequence"
    return {
        split: ImageSequenceDataset(
            frame,
            image_size=int(data_cfg["image_size"]),
            sky_mask_path=data_cfg["sky_mask_path"],
            sequence_mode=sequence_mode,
            max_steps=int(data_cfg.get("max_steps", 0)),
            use_sky_mask=bool(data_cfg.get("use_sky_mask", True)),
        )
        for split, frame in frames.items()
    }


def train_model(config: dict, model_name: str, epochs_override: int = 0, max_samples: int = 0) -> Path:
    model_name = str(model_name).strip().lower().replace("-", "_")
    set_seed(int(config["seed"]))
    run_dir = timestamped_run_dir(config["train"]["artifact_root"], model_name)
    logger = setup_logger(run_dir, config["train"]["artifact_root"], model_name)
    logger.info("initializing %s baseline", model_name)
    logger.info("run_dir=%s", run_dir)
    device = resolve_device(config.get("device", "auto"))
    logger.info("device=%s", device)

    datasets = build_datasets(config, model_name, max_samples=max_samples)
    logger.info("dataset sizes: train=%d val=%d test=%d", len(datasets["train"]), len(datasets["val"]), len(datasets["test"]))
    model = build_model(model_name, config["model"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["train"]["learning_rate"]),
        weight_decay=float(config["train"]["weight_decay"]),
    )
    save_json(run_dir / "run_config.json", {**config, "model_name": model_name})

    train_loader = make_loader(
        datasets["train"],
        int(config["train"]["batch_size"]),
        shuffle=True,
        num_workers=int(config["train"]["num_workers"]),
    )
    logger.info(
        "train loader: batches=%d batch_size=%d num_workers=%d",
        len(train_loader),
        int(config["train"]["batch_size"]),
        int(config["train"]["num_workers"]),
    )

    epochs = int(epochs_override or config["train"]["epochs"])
    best_val = float("inf")
    best_state = None
    no_improve = 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        mse_losses = []
        epoch_start = time.time()
        total_batches = len(train_loader)
        log_every = int(config["train"].get("log_every_batches", 0))
        for batch_idx, batch in enumerate(train_loader, start=1):
            data = to_device(batch, device)
            optimizer.zero_grad()
            out = model(data["images"])
            nll = gaussian_nll(out["loc"], out["scale"], data["target"])
            mse = torch.nn.functional.mse_loss(out["loc"], data["target"].view_as(out["loc"]))
            gen = vae_loss(
                out,
                data["images"],
                recon_weight=float(config["loss"].get("vae_recon_weight", 0.0)),
                kl_weight=float(config["loss"].get("vae_kl_weight", 0.0)),
            )
            loss = nll + float(config["loss"].get("mse_weight", 0.25)) * mse + gen
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            mse_losses.append(float(mse.detach().cpu()))
            if log_every > 0 and (batch_idx == 1 or batch_idx % log_every == 0 or batch_idx == total_batches):
                elapsed = time.time() - epoch_start
                logger.info(
                    "epoch %03d batch %04d/%04d loss=%.4f mse=%.4f %.2f batch/s",
                    epoch,
                    batch_idx,
                    total_batches,
                    float(np.mean(losses[-min(len(losses), log_every):])),
                    float(np.mean(mse_losses[-min(len(mse_losses), log_every):])),
                    batch_idx / max(elapsed, 1e-6),
                )

        _, val_metrics = evaluate(model, datasets["val"], device)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "train_mse": float(np.mean(mse_losses)), **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        logger.info(
            "epoch %03d loss=%.4f val_rmse=%.1f baseline=%.1f coverage80=%.3f",
            epoch,
            row["train_loss"],
            val_metrics["rmse"],
            val_metrics["baseline_rmse"],
            val_metrics["coverage_80"],
        )

        selection = float(val_metrics["rmse"])
        if selection < best_val:
            best_val = selection
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            no_improve = 0
            torch.save(model.state_dict(), run_dir / "best_model.pt")
        else:
            no_improve += 1
            if epoch >= int(config["train"]["early_stopping_min_epochs"]) and no_improve >= int(config["train"]["early_stopping_patience"]):
                logger.info("early stopping at epoch %d", epoch)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    scale_multiplier = calibrate_scale_multiplier(
        model,
        datasets["val"],
        device,
        float(config["train"].get("target_interval_coverage", 0.80)),
        int(config["train"].get("calibration_steps", 8)),
    )
    save_json(
        run_dir / "calibration.json",
        {
            "scale_multiplier": scale_multiplier,
            "target_coverage": float(config["train"].get("target_interval_coverage", 0.80)),
        },
    )

    for split, dataset in datasets.items():
        pred, metrics = evaluate(model, dataset, device, scale_multiplier=scale_multiplier)
        pred.to_csv(run_dir / f"predictions_{split}.csv", index=False)
        save_json(run_dir / f"metrics_{split}.json", metrics)
    logger.info("finished run_dir=%s", run_dir)
    return run_dir
