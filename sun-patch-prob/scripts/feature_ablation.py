from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train import calibrate_scale_multiplier, evaluate, make_loader, to_device
from sun_patch_prob.data import SunPatchFeatureDataset, fit_feature_spec, infer_feature_columns
from sun_patch_prob.model import StudentTResidualModel, interval_width_regularizer, student_t_nll
from sun_patch_prob.utils import load_json, resolve_device, save_json, set_seed


def is_pv_col(col: str) -> bool:
    return col.startswith("past_pv_")


def is_solar_col(col: str) -> bool:
    return col.startswith("solar_") or col in {"sun_dx_px", "sun_dy_px", "target_solar_elevation_deg"}


def is_weather_col(col: str) -> bool:
    return col.startswith("weather_")


def is_global_col(col: str) -> bool:
    return "_global_" in col


def is_ring_col(col: str) -> bool:
    return "target_ring" in col or "target_disk" in col or "current_disk" in col


def is_weighted_col(col: str) -> bool:
    return col.startswith("past_weighted_")


def select_columns(all_cols: list[str], group: str) -> list[str]:
    base = [c for c in all_cols if is_pv_col(c) or is_solar_col(c)]
    base_weather = [c for c in all_cols if c in base or is_weather_col(c)]
    if group == "pv_solar":
        return base
    if group == "pv_solar_weather":
        return base_weather
    if group == "global":
        return [c for c in all_cols if c in base_weather or is_global_col(c)]
    if group == "rings":
        return [c for c in all_cols if c in base_weather or is_ring_col(c)]
    if group == "weighted":
        return [c for c in all_cols if c in base_weather or is_weighted_col(c)]
    if group == "rings_weighted":
        return [c for c in all_cols if c in base_weather or is_ring_col(c) or is_weighted_col(c)]
    if group == "no_hard_rings":
        return [c for c in all_cols if not is_ring_col(c)]
    if group == "weighted_no_hard_rings":
        return [c for c in all_cols if c in base_weather or is_weighted_col(c)]
    if group == "sky_no_weather":
        return [c for c in all_cols if not is_weather_col(c)]
    if group == "all":
        return all_cols
    raise ValueError(f"Unknown ablation group: {group}")


def weighted_mean(values: list[float], weights: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    valid = np.isfinite(arr) & np.isfinite(w)
    if not np.any(valid):
        return float("nan")
    return float(np.sum(arr[valid] * w[valid]) / np.sum(w[valid]).clip(min=1e-6))


def train_one_group(
    config: dict,
    group: str,
    feature_columns: list[str],
    frames: dict[str, pd.DataFrame],
    device: torch.device,
    out_dir: Path,
    epochs: int,
) -> dict[str, float | str | int]:
    weather_weights = {str(k).strip().lower(): float(v) for k, v in config["train"].get("weather_weights", {}).items()}
    spec = fit_feature_spec(frames["train"], feature_columns)
    datasets = {
        split: SunPatchFeatureDataset(frame, spec, weather_weights=weather_weights)
        for split, frame in frames.items()
    }
    model = StudentTResidualModel(input_dim=len(feature_columns), **config["model"]).to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["train"]["learning_rate"]),
        weight_decay=float(config["train"]["weight_decay"]),
    )
    loader = make_loader(datasets["train"], int(config["train"]["batch_size"]), shuffle=True, num_workers=int(config["train"]["num_workers"]))
    best_val = float("inf")
    best_state = None
    no_improve = 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch in loader:
            data = to_device(batch, device)
            opt.zero_grad()
            out = model(data["x"], data["baseline"])
            nll = student_t_nll(out["loc"], out["scale"], out["df"], data["target"], weight=data["weight"])
            aux_loss = F.smooth_l1_loss(out["aux"], data["aux"], reduction="none").mean(dim=1, keepdim=True)
            aux = (aux_loss * data["weight"]).sum() / data["weight"].sum().clamp_min(1e-6)
            interval = interval_width_regularizer(out["scale"], weight=data["weight"])
            loss = (
                float(config["loss"]["nll_weight"]) * nll
                + float(config["loss"]["aux_weight"]) * aux
                + float(config["loss"]["interval_weight"]) * interval
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        _, val_metrics = evaluate(model, datasets["val"], device)
        selection_value = float(val_metrics.get("hard_weather_rmse", val_metrics["rmse"]))
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), **{f"val_{k}": v for k, v in val_metrics.items()}})
        if selection_value < best_val:
            best_val = selection_value
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if epoch >= int(config["train"]["early_stopping_min_epochs"]) and no_improve >= int(config["train"]["early_stopping_patience"]):
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    group_dir = out_dir / group
    group_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(group_dir / "history.csv", index=False)
    (group_dir / "feature_columns.json").write_text(json.dumps(feature_columns, indent=2), encoding="utf-8")
    scale_multiplier = calibrate_scale_multiplier(
        model,
        datasets["val"],
        device,
        target_coverage=float(config["train"].get("target_interval_coverage", 0.90)),
    )
    metrics_by_split = {}
    for split, dataset in datasets.items():
        pred, metrics = evaluate(model, dataset, device, scale_multiplier=scale_multiplier)
        pred.to_csv(group_dir / f"predictions_{split}.csv", index=False)
        save_json(group_dir / f"metrics_{split}.json", metrics)
        metrics_by_split[split] = metrics

    val = metrics_by_split["val"]
    test = metrics_by_split["test"]
    return {
        "group": group,
        "n_features": len(feature_columns),
        "epochs": len(history),
        "scale_multiplier": scale_multiplier,
        "val_rmse": val["rmse"],
        "val_smart_rmse": val["smart_persistence_rmse"],
        "val_hard_rmse": val.get("hard_weather_rmse", np.nan),
        "val_hard_smart_rmse": val.get("hard_weather_smart_persistence_rmse", np.nan),
        "val_stable_rmse": val.get("stable_weather_rmse", np.nan),
        "val_stable_smart_rmse": val.get("stable_weather_smart_persistence_rmse", np.nan),
        "val_coverage_80": val["coverage_80"],
        "test_rmse": test["rmse"],
        "test_smart_rmse": test["smart_persistence_rmse"],
        "test_hard_rmse": test.get("hard_weather_rmse", np.nan),
        "test_hard_smart_rmse": test.get("hard_weather_smart_persistence_rmse", np.nan),
        "test_stable_rmse": test.get("stable_weather_rmse", np.nan),
        "test_stable_smart_rmse": test.get("stable_weather_smart_persistence_rmse", np.nan),
        "test_coverage_80": test["coverage_80"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run feature-family ablations for the sun-patch probabilistic model.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/all_weather_balanced.json")
    parser.add_argument("--feature-csv", type=Path, default=None)
    parser.add_argument("--groups", nargs="*", default=["pv_solar", "pv_solar_weather", "global", "rings", "weighted", "rings_weighted", "sky_no_weather", "all"])
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    config = load_json(args.config)
    set_seed(int(config["seed"]))
    device = resolve_device(config.get("device", "auto"))
    feature_path = args.feature_csv or Path(config["data"]["feature_csv"])
    df = pd.read_csv(feature_path)
    frames = {
        "train": df[df["split"] == "train"].reset_index(drop=True),
        "val": df[df["split"] == "val"].reset_index(drop=True),
        "test": df[df["split"] == "test"].reset_index(drop=True),
    }
    all_cols = infer_feature_columns(frames["train"])
    out_dir = args.out_dir or (ROOT / "artifacts/ablations" / f"ablation_{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(out_dir / "run_config.json", config)
    epochs = int(args.epochs or config["train"]["epochs"])
    rows = []
    for group in args.groups:
        columns = select_columns(all_cols, group)
        print(f"running group={group} n_features={len(columns)}", flush=True)
        row = train_one_group(config, group, columns, frames, device, out_dir, epochs)
        rows.append(row)
        pd.DataFrame(rows).to_csv(out_dir / "summary.csv", index=False)
        print(
            f"{group}: val_hard={row['val_hard_rmse']:.1f} test_hard={row['test_hard_rmse']:.1f} "
            f"test={row['test_rmse']:.1f}",
            flush=True,
        )
    print(f"summary={out_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
