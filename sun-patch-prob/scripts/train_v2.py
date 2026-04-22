from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import stats
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sun_patch_prob.data_v2 import SunPatchSequenceDataset, prepare_v2_frames
from sun_patch_prob.metrics import interval_coverage, pinball_loss, regression_metrics
from sun_patch_prob.model import interval_width_regularizer, student_t_nll
from sun_patch_prob.model_v2 import SunPatchGatedProbabilisticModel
from sun_patch_prob.utils import load_json, resolve_device, save_json, set_seed, timestamped_run_dir
from sun_patch_prob.viz import save_forecast_band


def make_loader(dataset: SunPatchSequenceDataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=torch.cuda.is_available())


def to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def student_t_quantiles(loc: np.ndarray, scale: np.ndarray, df: np.ndarray, probs: list[float]) -> np.ndarray:
    z = np.stack([stats.t.ppf(p, df=np.clip(df, 2.01, 200.0)) for p in probs], axis=1)
    return loc[:, None] + scale[:, None] * z


def add_subset_metrics(frame: pd.DataFrame, metrics: dict[str, float]) -> None:
    tags = frame["weather_tag"].astype(str).str.strip().str.lower()
    for tag, part in frame.groupby(tags, dropna=False):
        key = str(tag).replace(" ", "_") or "unknown"
        tag_metrics = regression_metrics(part["q50_w"].to_numpy(), part["target_pv_w"].to_numpy())
        tag_smart = regression_metrics(part["smart_persistence_pv_w"].to_numpy(), part["target_pv_w"].to_numpy())
        metrics[f"weather_{key}_n"] = int(len(part))
        metrics[f"weather_{key}_rmse"] = tag_metrics["rmse"]
        metrics[f"weather_{key}_mae"] = tag_metrics["mae"]
        metrics[f"weather_{key}_smart_persistence_rmse"] = tag_smart["rmse"]
        metrics[f"weather_{key}_smart_persistence_mae"] = tag_smart["mae"]
    for name, part in [
        ("hard_weather", frame[tags.isin(["cloudy", "partly_cloudy"])]),
        ("stable_weather", frame[tags.isin(["clear_sky", "overcast"])]),
    ]:
        if part.empty:
            continue
        part_metrics = regression_metrics(part["q50_w"].to_numpy(), part["target_pv_w"].to_numpy())
        part_smart = regression_metrics(part["smart_persistence_pv_w"].to_numpy(), part["target_pv_w"].to_numpy())
        metrics[f"{name}_n"] = int(len(part))
        metrics[f"{name}_rmse"] = part_metrics["rmse"]
        metrics[f"{name}_mae"] = part_metrics["mae"]
        metrics[f"{name}_smart_persistence_rmse"] = part_smart["rmse"]
        metrics[f"{name}_smart_persistence_mae"] = part_smart["mae"]


def evaluate(
    model: SunPatchGatedProbabilisticModel,
    dataset: SunPatchSequenceDataset,
    device: torch.device,
    scale_multiplier: float = 1.0,
) -> tuple[pd.DataFrame, dict[str, float]]:
    loader = make_loader(dataset, batch_size=128, shuffle=False, num_workers=0)
    model.eval()
    arrays: dict[str, list[np.ndarray]] = {k: [] for k in ["loc", "scale", "df", "gate_sun", "residual_limit", "global_residual", "sun_residual", "residual_loc"]}
    indices = []
    with torch.no_grad():
        for batch in loader:
            data = to_device(batch, device)
            out = model(data["patch_seq"], data["global_x"], data["baseline"])
            for key in arrays:
                arrays[key].append(out[key].cpu().numpy().reshape(-1))
            indices.append(data["index"].cpu().numpy())
    values = {key: np.concatenate(parts) for key, parts in arrays.items()}
    values["scale"] = values["scale"] * float(scale_multiplier)
    idx = np.concatenate(indices)
    frame = dataset.df.iloc[idx].reset_index(drop=True).copy()
    q = student_t_quantiles(values["loc"], values["scale"], values["df"], [0.10, 0.25, 0.50, 0.75, 0.90])
    q = np.clip(q, 0.0, 1.25)
    clear = frame["target_clear_sky_w"].to_numpy(dtype=np.float32)
    for i, name in enumerate(["q10", "q25", "q50", "q75", "q90"]):
        frame[name] = q[:, i]
        frame[f"{name}_w"] = q[:, i] * clear
    for key, arr in values.items():
        frame[key] = arr
    metrics = regression_metrics(frame["q50_w"].to_numpy(), frame["target_pv_w"].to_numpy())
    smart = regression_metrics(frame["smart_persistence_pv_w"].to_numpy(), frame["target_pv_w"].to_numpy())
    metrics.update(
        {
            "n_samples": int(len(frame)),
            "smart_persistence_mae": smart["mae"],
            "smart_persistence_rmse": smart["rmse"],
            "coverage_80": interval_coverage(frame["q10_w"].to_numpy(), frame["q90_w"].to_numpy(), frame["target_pv_w"].to_numpy()),
            "pinball_q10_w": pinball_loss(frame["q10_w"].to_numpy(), frame["target_pv_w"].to_numpy(), 0.10),
            "pinball_q90_w": pinball_loss(frame["q90_w"].to_numpy(), frame["target_pv_w"].to_numpy(), 0.90),
            "mean_interval_width_w": float(np.mean(frame["q90_w"].to_numpy() - frame["q10_w"].to_numpy())),
            "scale_multiplier": float(scale_multiplier),
            "gate_sun_mean": float(np.mean(values["gate_sun"])),
        }
    )
    add_subset_metrics(frame, metrics)
    return frame, metrics


def calibrate_scale_multiplier(
    model: SunPatchGatedProbabilisticModel,
    dataset: SunPatchSequenceDataset,
    device: torch.device,
    target_coverage: float,
    steps: int,
) -> float:
    low, high = 0.25, 8.0
    best = high
    for _ in range(max(1, int(steps))):
        mid = (low + high) / 2.0
        _, metrics = evaluate(model, dataset, device, scale_multiplier=mid)
        if float(metrics["coverage_80"]) >= target_coverage:
            best = mid
            high = mid
        else:
            low = mid
    return float(best)


def train_loss(out: dict[str, torch.Tensor], data: dict[str, torch.Tensor], loss_cfg: dict) -> torch.Tensor:
    nll = student_t_nll(out["loc"], out["scale"], out["df"], data["target"], weight=data["weight"])
    aux_loss = F.smooth_l1_loss(out["aux"], data["aux"], reduction="none").mean(dim=1, keepdim=True)
    aux = (aux_loss * data["weight"]).sum() / data["weight"].sum().clamp_min(1e-6)
    interval = interval_width_regularizer(out["scale"], weight=data["weight"])
    gate = out["gate_sun"].clamp(1e-5, 1.0 - 1e-5)
    gate_entropy = -(gate * torch.log(gate) + (1.0 - gate) * torch.log(1.0 - gate)).mean()
    return (
        float(loss_cfg["nll_weight"]) * nll
        + float(loss_cfg["aux_weight"]) * aux
        + float(loss_cfg["interval_weight"]) * interval
        - float(loss_cfg.get("gate_entropy_weight", 0.0)) * gate_entropy
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train v2 sun-patch gated probabilistic model.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/v2_sunpatch_gated.json")
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()

    config = load_json(args.config)
    set_seed(int(config["seed"]))
    device = resolve_device(config.get("device", "auto"))
    frames, feature_columns = prepare_v2_frames(config["data"]["samples_csv"], config["data"]["feature_csv"], max_samples=args.max_samples)
    weather_weights = {str(k).strip().lower(): float(v) for k, v in config["train"].get("weather_weights", {}).items()}
    datasets = {
        "train": SunPatchSequenceDataset(frames.train, frames.feature_spec, int(config["data"]["image_size"]), int(config["data"]["patch_size"]), config["data"]["sky_mask_path"], weather_weights),
        "val": SunPatchSequenceDataset(frames.val, frames.feature_spec, int(config["data"]["image_size"]), int(config["data"]["patch_size"]), config["data"]["sky_mask_path"], weather_weights),
        "test": SunPatchSequenceDataset(frames.test, frames.feature_spec, int(config["data"]["image_size"]), int(config["data"]["patch_size"]), config["data"]["sky_mask_path"], weather_weights),
    }
    model = SunPatchGatedProbabilisticModel(global_input_dim=len(feature_columns), **config["model"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(config["train"]["learning_rate"]), weight_decay=float(config["train"]["weight_decay"]))
    run_dir = timestamped_run_dir(config["train"]["artifact_root"])
    save_json(run_dir / "run_config.json", config)
    (run_dir / "feature_columns.json").write_text(json.dumps(feature_columns, indent=2), encoding="utf-8")
    np.savez(run_dir / "feature_spec.npz", mean=frames.feature_spec.mean, std=frames.feature_spec.std)
    print(f"run_dir={run_dir}", flush=True)
    train_loader = make_loader(datasets["train"], int(config["train"]["batch_size"]), shuffle=True, num_workers=int(config["train"]["num_workers"]))
    epochs = int(args.epochs or config["train"]["epochs"])
    best_val = float("inf")
    best_state = None
    no_improve = 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            data = to_device(batch, device)
            opt.zero_grad()
            out = model(data["patch_seq"], data["global_x"], data["baseline"])
            loss = train_loss(out, data, config["loss"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        _, val_metrics = evaluate(model, datasets["val"], device)
        selection = float(val_metrics.get("hard_weather_rmse", val_metrics["rmse"]))
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        print(
            f"epoch {epoch:03d} loss={row['train_loss']:.4f} val_rmse={val_metrics['rmse']:.1f} "
            f"hard={val_metrics.get('hard_weather_rmse', np.nan):.1f} gate={val_metrics['gate_sun_mean']:.3f}",
            flush=True,
        )
        if selection < best_val:
            best_val = selection
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
            torch.save(model.state_dict(), run_dir / "best_model.pt")
        else:
            no_improve += 1
            if epoch >= int(config["train"]["early_stopping_min_epochs"]) and no_improve >= int(config["train"]["early_stopping_patience"]):
                print(f"early stopping at epoch {epoch}", flush=True)
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    calibration_steps = int(config["train"].get("calibration_steps", 10))
    print(f"calibrating scale with {calibration_steps} validation passes", flush=True)
    scale_multiplier = calibrate_scale_multiplier(
        model,
        datasets["val"],
        device,
        float(config["train"].get("target_interval_coverage", 0.90)),
        calibration_steps,
    )
    save_json(run_dir / "calibration.json", {"scale_multiplier": scale_multiplier, "target_coverage": float(config["train"].get("target_interval_coverage", 0.90))})
    for split, dataset in datasets.items():
        pred, metrics = evaluate(model, dataset, device, scale_multiplier=scale_multiplier)
        pred.to_csv(run_dir / f"predictions_{split}.csv", index=False)
        save_json(run_dir / f"metrics_{split}.json", metrics)
        save_forecast_band(pred, run_dir / "figures" / f"forecast_band_{split}.png", f"{split} v2 sun-patch gated forecast")
    print(f"run_dir={run_dir}")


if __name__ == "__main__":
    main()
