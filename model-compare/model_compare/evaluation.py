from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from scipy import stats
from torch.utils.data import DataLoader

from .data import ImageSequenceDataset
from .utils import interval_coverage, regression_metrics


def make_loader(dataset: ImageSequenceDataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=torch.cuda.is_available())


def to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def normal_quantiles(loc: np.ndarray, scale: np.ndarray, probs: list[float]) -> np.ndarray:
    z = np.asarray([stats.norm.ppf(p) for p in probs], dtype=np.float32)
    return loc[:, None] + scale[:, None] * z[None, :]


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


def evaluate(model: torch.nn.Module, dataset: ImageSequenceDataset, device: torch.device, scale_multiplier: float = 1.0) -> tuple[pd.DataFrame, dict[str, float]]:
    loader = make_loader(dataset, batch_size=128, shuffle=False, num_workers=0)
    model.eval()
    locs = []
    scales = []
    indices = []
    with torch.no_grad():
        for batch in loader:
            data = to_device(batch, device)
            out = model(data["images"], data.get("history_x"))
            locs.append(out["loc"].cpu().numpy().reshape(-1))
            scales.append(out["scale"].cpu().numpy().reshape(-1))
            indices.append(data["index"].cpu().numpy())

    loc = np.concatenate(locs)
    scale = np.concatenate(scales) * float(scale_multiplier)
    frame = dataset.df.iloc[np.concatenate(indices)].reset_index(drop=True).copy()
    quantile_probs = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
    quantile_names = ["q05", "q10", "q25", "q50", "q75", "q90", "q95"]
    quantiles = np.clip(normal_quantiles(loc, scale, quantile_probs), 0.0, 1.25)
    clear = frame["target_clear_sky_w"].to_numpy(dtype=np.float32)
    for idx, name in enumerate(quantile_names):
        frame[name] = quantiles[:, idx]
        frame[f"{name}_w"] = quantiles[:, idx] * clear
    frame["loc"] = loc
    frame["scale"] = scale

    metrics = regression_metrics(frame["q50_w"].to_numpy(), frame["target_pv_w"].to_numpy())
    baseline = regression_metrics(frame["baseline_pv_w"].to_numpy(), frame["target_pv_w"].to_numpy())
    metrics.update(
        {
            "n_samples": int(len(frame)),
            "baseline_mae": baseline["mae"],
            "baseline_rmse": baseline["rmse"],
            "coverage_80": interval_coverage(frame["q10_w"].to_numpy(), frame["q90_w"].to_numpy(), frame["target_pv_w"].to_numpy()),
            "coverage_90": interval_coverage(frame["q05_w"].to_numpy(), frame["q95_w"].to_numpy(), frame["target_pv_w"].to_numpy()),
            "mean_interval_width_w": float(np.mean(frame["q90_w"].to_numpy() - frame["q10_w"].to_numpy())),
            "mean_interval_width_90_w": float(np.mean(frame["q95_w"].to_numpy() - frame["q05_w"].to_numpy())),
            "scale_multiplier": float(scale_multiplier),
        }
    )
    add_weather_metrics(frame, metrics)
    return frame, metrics


def calibrate_scale_multiplier(model: torch.nn.Module, dataset: ImageSequenceDataset, device: torch.device, target_coverage: float, steps: int) -> float:
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
