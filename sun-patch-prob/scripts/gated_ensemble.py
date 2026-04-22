from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import pandas as pd


QUANTILE_COLS = ["q10", "q25", "q50", "q75", "q90"]


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


def coverage(q10_w: np.ndarray, q90_w: np.ndarray, target_w: np.ndarray) -> float:
    q10 = np.asarray(q10_w, dtype=np.float64)
    q90 = np.asarray(q90_w, dtype=np.float64)
    target = np.asarray(target_w, dtype=np.float64)
    valid = np.isfinite(q10) & np.isfinite(q90) & np.isfinite(target)
    if not np.any(valid):
        return float("nan")
    return float(np.mean((target[valid] >= q10[valid]) & (target[valid] <= q90[valid])))


def pinball(q_w: np.ndarray, target_w: np.ndarray, tau: float) -> float:
    q = np.asarray(q_w, dtype=np.float64)
    target = np.asarray(target_w, dtype=np.float64)
    diff = target - q
    return float(np.mean(np.maximum(tau * diff, (tau - 1.0) * diff)))


def metrics_for(frame: pd.DataFrame) -> dict[str, float]:
    metrics = regression_metrics(frame["q50_w"].to_numpy(), frame["target_pv_w"].to_numpy())
    smart = regression_metrics(frame["smart_persistence_pv_w"].to_numpy(), frame["target_pv_w"].to_numpy())
    persistence = regression_metrics(frame["persistence_pv_w"].to_numpy(), frame["target_pv_w"].to_numpy())
    metrics.update(
        {
            "n_samples": int(len(frame)),
            "smart_persistence_mae": smart["mae"],
            "smart_persistence_rmse": smart["rmse"],
            "persistence_mae": persistence["mae"],
            "persistence_rmse": persistence["rmse"],
            "coverage_80": coverage(frame["q10_w"].to_numpy(), frame["q90_w"].to_numpy(), frame["target_pv_w"].to_numpy()),
            "pinball_q10_w": pinball(frame["q10_w"].to_numpy(), frame["target_pv_w"].to_numpy(), 0.10),
            "pinball_q90_w": pinball(frame["q90_w"].to_numpy(), frame["target_pv_w"].to_numpy(), 0.90),
            "mean_interval_width_w": float(np.mean(frame["q90_w"].to_numpy() - frame["q10_w"].to_numpy())),
        }
    )
    if "weather_tag" in frame:
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
        hard = frame[tags.isin(["cloudy", "partly_cloudy"])]
        stable = frame[tags.isin(["clear_sky", "overcast"])]
        for name, part in [("hard_weather", hard), ("stable_weather", stable)]:
            if part.empty:
                continue
            part_metrics = regression_metrics(part["q50_w"].to_numpy(), part["target_pv_w"].to_numpy())
            part_smart = regression_metrics(part["smart_persistence_pv_w"].to_numpy(), part["target_pv_w"].to_numpy())
            metrics[f"{name}_n"] = int(len(part))
            metrics[f"{name}_rmse"] = part_metrics["rmse"]
            metrics[f"{name}_mae"] = part_metrics["mae"]
            metrics[f"{name}_smart_persistence_rmse"] = part_smart["rmse"]
            metrics[f"{name}_smart_persistence_mae"] = part_smart["mae"]
    return metrics


def read_branch(ablation_dir: Path, branch: str, split: str) -> pd.DataFrame:
    path = ablation_dir / branch / f"predictions_{split}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def combine_predictions(global_df: pd.DataFrame, weighted_df: pd.DataFrame, alpha_hard: float, alpha_stable: float) -> pd.DataFrame:
    out = global_df.copy()
    tags = out["weather_tag"].astype(str).str.strip().str.lower()
    alpha = np.where(tags.isin(["cloudy", "partly_cloudy"]), float(alpha_hard), float(alpha_stable)).astype(np.float64)
    out["gate_alpha_weighted"] = alpha
    clear = out["target_clear_sky_w"].to_numpy(dtype=np.float64)
    for col in QUANTILE_COLS:
        g = global_df[col].to_numpy(dtype=np.float64)
        w = weighted_df[col].to_numpy(dtype=np.float64)
        q = (1.0 - alpha) * g + alpha * w
        out[col] = np.clip(q, 0.0, 1.25)
        out[f"{col}_w"] = out[col].to_numpy(dtype=np.float64) * clear
    if "loc" in global_df and "loc" in weighted_df:
        out["loc"] = (1.0 - alpha) * global_df["loc"].to_numpy(dtype=np.float64) + alpha * weighted_df["loc"].to_numpy(dtype=np.float64)
    return out


def objective(metrics: dict[str, float], hard_weight: float, stable_penalty_weight: float) -> float:
    overall = metrics["rmse"] / max(metrics["smart_persistence_rmse"], 1e-6)
    hard = metrics.get("hard_weather_rmse", metrics["rmse"]) / max(metrics.get("hard_weather_smart_persistence_rmse", metrics["smart_persistence_rmse"]), 1e-6)
    stable = metrics.get("stable_weather_rmse", metrics["rmse"]) / max(metrics.get("stable_weather_smart_persistence_rmse", metrics["smart_persistence_rmse"]), 1e-6)
    stable_penalty = max(0.0, stable - 1.0)
    return float(overall + hard_weight * hard + stable_penalty_weight * stable_penalty)


def main() -> None:
    parser = argparse.ArgumentParser(description="Search a weather-aware gate between global and weighted ablation branches.")
    parser.add_argument("--ablation-dir", type=Path, required=True)
    parser.add_argument("--global-branch", default="global")
    parser.add_argument("--weighted-branch", default="weighted")
    parser.add_argument("--hard-weight", type=float, default=0.8)
    parser.add_argument("--stable-penalty-weight", type=float, default=1.0)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    out_dir = args.out_dir or args.ablation_dir / f"gated_ensemble_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    val_g = read_branch(args.ablation_dir, args.global_branch, "val")
    val_w = read_branch(args.ablation_dir, args.weighted_branch, "val")
    rows = []
    best = None
    for alpha_hard in np.linspace(0.0, 1.0, 21):
        for alpha_stable in np.linspace(0.0, 1.0, 21):
            pred = combine_predictions(val_g, val_w, alpha_hard=alpha_hard, alpha_stable=alpha_stable)
            metrics = metrics_for(pred)
            score = objective(metrics, args.hard_weight, args.stable_penalty_weight)
            row = {
                "alpha_hard": float(alpha_hard),
                "alpha_stable": float(alpha_stable),
                "score": score,
                **metrics,
            }
            rows.append(row)
            if best is None or score < best["score"]:
                best = row
    search = pd.DataFrame(rows).sort_values("score").reset_index(drop=True)
    search.to_csv(out_dir / "gate_search_val.csv", index=False)
    assert best is not None

    summary = []
    for split in ("train", "val", "test"):
        g = read_branch(args.ablation_dir, args.global_branch, split)
        w = read_branch(args.ablation_dir, args.weighted_branch, split)
        pred = combine_predictions(g, w, alpha_hard=best["alpha_hard"], alpha_stable=best["alpha_stable"])
        metrics = metrics_for(pred)
        pred.to_csv(out_dir / f"predictions_{split}.csv", index=False)
        Path(out_dir / f"metrics_{split}.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        summary.append({"split": split, **metrics})
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(out_dir / "summary.csv", index=False)
    config = {
        "ablation_dir": str(args.ablation_dir),
        "global_branch": args.global_branch,
        "weighted_branch": args.weighted_branch,
        "hard_weight": args.hard_weight,
        "stable_penalty_weight": args.stable_penalty_weight,
        "alpha_hard": best["alpha_hard"],
        "alpha_stable": best["alpha_stable"],
        "val_score": best["score"],
    }
    Path(out_dir / "gate_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(json.dumps(config, indent=2))
    print(f"summary={out_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()

