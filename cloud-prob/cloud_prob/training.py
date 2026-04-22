from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import CloudSequenceDataset, prepare_frames
from .evaluation import calibrate_scale_multiplier, evaluate, make_loader, to_device
from .model import WeatherConditionedSunAwareModel, interval_width_regularizer, student_t_nll
from .utils import resolve_device, save_json, set_seed, timestamped_run_dir
from .viz import save_forecast_band, save_weather_gate_plot


def train_loss(out: dict[str, torch.Tensor], data: dict[str, torch.Tensor], loss_cfg: dict) -> torch.Tensor:
    nll = student_t_nll(out["loc"], out["scale"], out["df"], data["target"], weight=data["weight"])
    interval = interval_width_regularizer(out["scale"], weight=data["weight"])
    gate = out["cloud_gate"].clamp(1e-5, 1.0 - 1e-5)
    gate_entropy = -(gate * torch.log(gate) + (1.0 - gate) * torch.log(1.0 - gate)).mean()
    return (
        float(loss_cfg["nll_weight"]) * nll
        + float(loss_cfg["interval_weight"]) * interval
        - float(loss_cfg.get("gate_entropy_weight", 0.0)) * gate_entropy
    )


def build_datasets(config: dict, max_samples: int = 0) -> tuple[dict[str, CloudSequenceDataset], list[str]]:
    frames, feature_columns = prepare_frames(config["data"]["samples_csv"], max_samples=max_samples)
    data_cfg = config["data"]
    weather_weights = {str(k).strip().lower(): float(v) for k, v in config["train"].get("weather_weights", {}).items()}
    datasets = {
        split: CloudSequenceDataset(
            getattr(frames, split),
            frames.feature_spec,
            int(data_cfg["image_size"]),
            int(data_cfg["patch_size"]),
            data_cfg["sky_mask_path"],
            float(data_cfg.get("rbr_clip", 4.0)),
            weather_weights,
        )
        for split in ["train", "val", "test"]
    }
    return datasets, feature_columns


def save_run_setup(run_dir: Path, config: dict, feature_columns: list[str], dataset: CloudSequenceDataset) -> None:
    save_json(run_dir / "run_config.json", config)
    (run_dir / "feature_columns.json").write_text(json.dumps(feature_columns, indent=2), encoding="utf-8")
    np.savez(run_dir / "feature_spec.npz", mean=dataset.spec.mean, std=dataset.spec.std)


def train_model(config: dict, epochs_override: int = 0, max_samples: int = 0) -> Path:
    set_seed(int(config["seed"]))
    device = resolve_device(config.get("device", "auto"))
    datasets, feature_columns = build_datasets(config, max_samples=max_samples)
    model = WeatherConditionedSunAwareModel(global_input_dim=len(feature_columns), **config["model"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["train"]["learning_rate"]),
        weight_decay=float(config["train"]["weight_decay"]),
    )

    run_dir = timestamped_run_dir(config["train"]["artifact_root"])
    save_run_setup(run_dir, config, feature_columns, datasets["train"])
    print(f"run_dir={run_dir}", flush=True)

    train_loader = make_loader(
        datasets["train"],
        int(config["train"]["batch_size"]),
        shuffle=True,
        num_workers=int(config["train"]["num_workers"]),
    )
    epochs = int(epochs_override or config["train"]["epochs"])
    best_val = float("inf")
    best_state = None
    no_improve = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            data = to_device(batch, device)
            optimizer.zero_grad()
            out = model(data["patch_seq"], data["global_x"], data["weather_idx"], data["baseline"])
            loss = train_loss(out, data, config["loss"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        _, val_metrics = evaluate(model, datasets["val"], device)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        print(
            f"epoch {epoch:03d} loss={row['train_loss']:.4f} val_rmse={val_metrics['rmse']:.1f} "
            f"base={val_metrics['baseline_rmse']:.1f} gate={val_metrics['cloud_gate_mean']:.3f}",
            flush=True,
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
                print(f"early stopping at epoch {epoch}", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    calibration_steps = int(config["train"].get("calibration_steps", 8))
    print(f"calibrating scale with {calibration_steps} validation passes", flush=True)
    scale_multiplier = calibrate_scale_multiplier(
        model,
        datasets["val"],
        device,
        float(config["train"].get("target_interval_coverage", 0.90)),
        calibration_steps,
    )
    save_json(
        run_dir / "calibration.json",
        {
            "scale_multiplier": scale_multiplier,
            "target_coverage": float(config["train"].get("target_interval_coverage", 0.90)),
        },
    )

    for split, dataset in datasets.items():
        pred, metrics = evaluate(model, dataset, device, scale_multiplier=scale_multiplier)
        pred.to_csv(run_dir / f"predictions_{split}.csv", index=False)
        save_json(run_dir / f"metrics_{split}.json", metrics)
        save_forecast_band(pred, run_dir / "figures" / f"forecast_band_{split}.png", f"{split} forecast band")
        save_weather_gate_plot(pred, run_dir / "figures" / f"weather_gate_{split}.png")

    print(f"run_dir={run_dir}")
    return run_dir
