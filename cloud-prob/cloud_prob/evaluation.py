from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from scipy import stats
from torch.utils.data import DataLoader

from .data import CloudSequenceDataset
from .metrics import interval_coverage, pinball_loss, regression_metrics
from .model import WeatherConditionedSunAwareModel


PREDICTION_KEYS = ["loc", "scale", "df", "cloud_gate", "residual_limit", "residual_loc"]


def make_loader(dataset: CloudSequenceDataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=torch.cuda.is_available())


def to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def student_t_quantiles(loc: np.ndarray, scale: np.ndarray, df: np.ndarray, probs: list[float]) -> np.ndarray:
    z = np.stack([stats.t.ppf(p, df=np.clip(df, 2.01, 200.0)) for p in probs], axis=1)
    return loc[:, None] + scale[:, None] * z


def add_weather_metrics(frame: pd.DataFrame, metrics: dict[str, float]) -> None:
    tags = frame["weather_tag"].astype(str).str.strip().str.lower()
    for tag, part in frame.groupby(tags, dropna=False):
        key = str(tag).replace(" ", "_") or "unknown"
        pred = regression_metrics(part["q50_w"].to_numpy(), part["target_pv_w"].to_numpy())
        base = regression_metrics(part["baseline_pv_w"].to_numpy(), part["target_pv_w"].to_numpy())
        metrics[f"weather_{key}_n"] = int(len(part))
        metrics[f"weather_{key}_rmse"] = pred["rmse"]
        metrics[f"weather_{key}_mae"] = pred["mae"]
        metrics[f"weather_{key}_baseline_rmse"] = base["rmse"]
        metrics[f"weather_{key}_baseline_mae"] = base["mae"]


def evaluate(
    model: WeatherConditionedSunAwareModel,
    dataset: CloudSequenceDataset,
    device: torch.device,
    scale_multiplier: float = 1.0,
) -> tuple[pd.DataFrame, dict[str, float]]:
    loader = make_loader(dataset, batch_size=128, shuffle=False, num_workers=0)
    model.eval()
    arrays: dict[str, list[np.ndarray]] = {key: [] for key in PREDICTION_KEYS}
    indices = []
    with torch.no_grad():
        for batch in loader:
            data = to_device(batch, device)
            out = model(data["patch_seq"], data["global_x"], data["weather_idx"], data["baseline"])
            for key in PREDICTION_KEYS:
                arrays[key].append(out[key].cpu().numpy().reshape(-1))
            indices.append(data["index"].cpu().numpy())

    values = {key: np.concatenate(parts) for key, parts in arrays.items()}
    values["scale"] = values["scale"] * float(scale_multiplier)
    frame = dataset.df.iloc[np.concatenate(indices)].reset_index(drop=True).copy()
    quantiles = student_t_quantiles(values["loc"], values["scale"], values["df"], [0.10, 0.25, 0.50, 0.75, 0.90])
    quantiles = np.clip(quantiles, 0.0, 1.25)
    clear = frame["target_clear_sky_w"].to_numpy(dtype=np.float32)
    for idx, name in enumerate(["q10", "q25", "q50", "q75", "q90"]):
        frame[name] = quantiles[:, idx]
        frame[f"{name}_w"] = quantiles[:, idx] * clear
    for key, arr in values.items():
        frame[key] = arr

    metrics = regression_metrics(frame["q50_w"].to_numpy(), frame["target_pv_w"].to_numpy())
    baseline = regression_metrics(frame["baseline_pv_w"].to_numpy(), frame["target_pv_w"].to_numpy())
    metrics.update(
        {
            "n_samples": int(len(frame)),
            "baseline_mae": baseline["mae"],
            "baseline_rmse": baseline["rmse"],
            "coverage_80": interval_coverage(frame["q10_w"].to_numpy(), frame["q90_w"].to_numpy(), frame["target_pv_w"].to_numpy()),
            "pinball_q10_w": pinball_loss(frame["q10_w"].to_numpy(), frame["target_pv_w"].to_numpy(), 0.10),
            "pinball_q90_w": pinball_loss(frame["q90_w"].to_numpy(), frame["target_pv_w"].to_numpy(), 0.90),
            "mean_interval_width_w": float(np.mean(frame["q90_w"].to_numpy() - frame["q10_w"].to_numpy())),
            "scale_multiplier": float(scale_multiplier),
            "cloud_gate_mean": float(np.mean(values["cloud_gate"])),
        }
    )
    add_weather_metrics(frame, metrics)
    return frame, metrics


def calibrate_scale_multiplier(
    model: WeatherConditionedSunAwareModel,
    dataset: CloudSequenceDataset,
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
