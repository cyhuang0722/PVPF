from __future__ import annotations

import numpy as np


def regression_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    err = pred - target
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err * err))),
        "bias": float(np.mean(err)),
    }


def interval_coverage(lower: np.ndarray, upper: np.ndarray, target: np.ndarray) -> float:
    lower = np.asarray(lower)
    upper = np.asarray(upper)
    target = np.asarray(target)
    return float(np.mean((target >= lower) & (target <= upper)))


def pinball_loss(pred: np.ndarray, target: np.ndarray, quantile: float) -> float:
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    diff = target - pred
    return float(np.mean(np.maximum(quantile * diff, (quantile - 1.0) * diff)))
