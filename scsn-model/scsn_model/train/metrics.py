from __future__ import annotations

import numpy as np


def regression_metrics(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    err = pred - true
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
    }


def pinball_loss_numpy(pred_q: np.ndarray, target: np.ndarray, quantiles: list[float]) -> float:
    vals = []
    for idx, q in enumerate(quantiles):
        err = target - pred_q[:, idx]
        vals.append(np.maximum(q * err, (q - 1.0) * err).mean())
    return float(np.sum(vals))


def interval_metrics(pred_q: np.ndarray, target: np.ndarray) -> dict[str, float]:
    lower = pred_q[:, 0]
    upper = pred_q[:, 2]
    covered = (target >= lower) & (target <= upper)
    return {
        "picp_10_90": float(np.mean(covered)),
        "piw_10_90": float(np.mean(upper - lower)),
    }

