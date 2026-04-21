from __future__ import annotations

import numpy as np


def regression_metrics(pred_w: np.ndarray, target_w: np.ndarray) -> dict[str, float]:
    pred = np.asarray(pred_w, dtype=np.float64)
    target = np.asarray(target_w, dtype=np.float64)
    valid = np.isfinite(pred) & np.isfinite(target)
    if not np.any(valid):
        return {"mae": float("nan"), "rmse": float("nan"), "bias": float("nan")}
    err = pred[valid] - target[valid]
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err * err))),
        "bias": float(np.mean(err)),
    }


def interval_coverage(q_low: np.ndarray, q_high: np.ndarray, target: np.ndarray) -> float:
    q_low = np.asarray(q_low, dtype=np.float64)
    q_high = np.asarray(q_high, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    valid = np.isfinite(q_low) & np.isfinite(q_high) & np.isfinite(target)
    if not np.any(valid):
        return float("nan")
    return float(np.mean((target[valid] >= q_low[valid]) & (target[valid] <= q_high[valid])))


def pinball_loss(q: np.ndarray, y: np.ndarray, tau: float) -> float:
    q = np.asarray(q, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    diff = y - q
    return float(np.mean(np.maximum(tau * diff, (tau - 1.0) * diff)))

