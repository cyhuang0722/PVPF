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

from sun_patch_prob.data import SunPatchFeatureDataset, fit_feature_spec, infer_feature_columns
from sun_patch_prob.metrics import interval_coverage, pinball_loss, regression_metrics
from sun_patch_prob.model import StudentTResidualModel, interval_width_regularizer, student_t_nll
from sun_patch_prob.utils import load_json, resolve_device, save_json, set_seed, timestamped_run_dir
from sun_patch_prob.viz import save_case_plot, save_forecast_band


def make_loader(dataset: SunPatchFeatureDataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def student_t_quantiles(loc: np.ndarray, scale: np.ndarray, df: np.ndarray, probs: list[float]) -> np.ndarray:
    z = np.stack([stats.t.ppf(p, df=np.clip(df, 2.01, 200.0)) for p in probs], axis=1)
    return loc[:, None] + scale[:, None] * z


def evaluate(
    model: StudentTResidualModel,
    dataset: SunPatchFeatureDataset,
    device: torch.device,
    scale_multiplier: float = 1.0,
) -> tuple[pd.DataFrame, dict[str, float]]:
    loader = make_loader(dataset, batch_size=256, shuffle=False, num_workers=0)
    model.eval()
    locs, scales, dfs = [], [], []
    with torch.no_grad():
        for batch in loader:
            data = to_device(batch, device)
            out = model(data["x"], data["baseline"])
            locs.append(out["loc"].cpu().numpy().reshape(-1))
            scales.append(out["scale"].cpu().numpy().reshape(-1))
            dfs.append(out["df"].cpu().numpy().reshape(-1))
    loc = np.concatenate(locs)
    scale = np.concatenate(scales) * float(scale_multiplier)
    df = np.concatenate(dfs)
    probs = [0.10, 0.25, 0.50, 0.75, 0.90]
    q = student_t_quantiles(loc, scale, df, probs)
    q = np.clip(q, 0.0, 1.25)
    frame = dataset.df.copy()
    clear = frame["target_clear_sky_w"].to_numpy(dtype=np.float32)
    for i, name in enumerate(["q10", "q25", "q50", "q75", "q90"]):
        frame[name] = q[:, i]
        frame[f"{name}_w"] = q[:, i] * clear
    frame["loc"] = loc
    frame["scale"] = scale
    frame["df"] = df
    metrics = regression_metrics(frame["q50_w"].to_numpy(), frame["target_pv_w"].to_numpy())
    base = regression_metrics(frame["baseline_pv_w"].to_numpy(), frame["target_pv_w"].to_numpy())
    smart = regression_metrics(frame["smart_persistence_pv_w"].to_numpy(), frame["target_pv_w"].to_numpy()) if "smart_persistence_pv_w" in frame else base
    persistence = regression_metrics(frame["persistence_pv_w"].to_numpy(), frame["target_pv_w"].to_numpy()) if "persistence_pv_w" in frame else base
    metrics.update(
        {
            "n_samples": int(len(frame)),
            "baseline_mae": float(base["mae"]),
            "baseline_rmse": float(base["rmse"]),
            "smart_persistence_mae": float(smart["mae"]),
            "smart_persistence_rmse": float(smart["rmse"]),
            "persistence_mae": float(persistence["mae"]),
            "persistence_rmse": float(persistence["rmse"]),
            "coverage_80": interval_coverage(frame["q10_w"].to_numpy(), frame["q90_w"].to_numpy(), frame["target_pv_w"].to_numpy()),
            "pinball_q10_w": pinball_loss(frame["q10_w"].to_numpy(), frame["target_pv_w"].to_numpy(), 0.10),
            "pinball_q90_w": pinball_loss(frame["q90_w"].to_numpy(), frame["target_pv_w"].to_numpy(), 0.90),
            "mean_interval_width_w": float(np.mean(frame["q90_w"].to_numpy() - frame["q10_w"].to_numpy())),
            "scale_multiplier": float(scale_multiplier),
        }
    )
    if "weather_tag" in frame:
        for tag, part in frame.groupby("weather_tag", dropna=False):
            tag_key = str(tag).strip().lower().replace(" ", "_") or "unknown"
            tag_metrics = regression_metrics(part["q50_w"].to_numpy(), part["target_pv_w"].to_numpy())
            tag_smart = regression_metrics(part["smart_persistence_pv_w"].to_numpy(), part["target_pv_w"].to_numpy()) if "smart_persistence_pv_w" in part else tag_metrics
            metrics[f"weather_{tag_key}_n"] = int(len(part))
            metrics[f"weather_{tag_key}_rmse"] = float(tag_metrics["rmse"])
            metrics[f"weather_{tag_key}_mae"] = float(tag_metrics["mae"])
            metrics[f"weather_{tag_key}_smart_persistence_rmse"] = float(tag_smart["rmse"])
            metrics[f"weather_{tag_key}_smart_persistence_mae"] = float(tag_smart["mae"])
        hard = frame[frame["weather_tag"].astype(str).str.strip().str.lower().isin(["cloudy", "partly_cloudy"])]
        stable = frame[frame["weather_tag"].astype(str).str.strip().str.lower().isin(["clear_sky", "overcast"])]
        for name, part in [("hard_weather", hard), ("stable_weather", stable)]:
            if part.empty:
                continue
            part_metrics = regression_metrics(part["q50_w"].to_numpy(), part["target_pv_w"].to_numpy())
            part_smart = regression_metrics(part["smart_persistence_pv_w"].to_numpy(), part["target_pv_w"].to_numpy())
            metrics[f"{name}_n"] = int(len(part))
            metrics[f"{name}_rmse"] = float(part_metrics["rmse"])
            metrics[f"{name}_mae"] = float(part_metrics["mae"])
            metrics[f"{name}_smart_persistence_rmse"] = float(part_smart["rmse"])
            metrics[f"{name}_smart_persistence_mae"] = float(part_smart["mae"])
    return frame, metrics


def calibrate_scale_multiplier(model: StudentTResidualModel, dataset: SunPatchFeatureDataset, device: torch.device, target_coverage: float = 0.80) -> float:
    low, high = 0.25, 8.0
    best = high
    for _ in range(24):
        mid = (low + high) / 2.0
        _, metrics = evaluate(model, dataset, device, scale_multiplier=mid)
        coverage = float(metrics["coverage_80"])
        if coverage >= target_coverage:
            best = mid
            high = mid
        else:
            low = mid
    return float(best)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Student-t sun-patch probabilistic PV model.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/base.json")
    parser.add_argument("--feature-csv", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=0)
    args = parser.parse_args()

    config = load_json(args.config)
    set_seed(int(config["seed"]))
    device = resolve_device(config.get("device", "auto"))
    feature_path = args.feature_csv or Path(config["data"]["feature_csv"])
    if not feature_path.exists():
        raise FileNotFoundError(f"Feature table missing: {feature_path}. Run scripts/build_features.py first.")

    df = pd.read_csv(feature_path)
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    feature_columns = infer_feature_columns(train_df)
    spec = fit_feature_spec(train_df, feature_columns)
    weather_weights = {str(k).strip().lower(): float(v) for k, v in config["train"].get("weather_weights", {}).items()}
    datasets = {
        "train": SunPatchFeatureDataset(train_df, spec, weather_weights=weather_weights),
        "val": SunPatchFeatureDataset(val_df, spec, weather_weights=weather_weights),
        "test": SunPatchFeatureDataset(test_df, spec, weather_weights=weather_weights),
    }

    model_cfg = config["model"]
    model = StudentTResidualModel(input_dim=len(feature_columns), **model_cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(config["train"]["learning_rate"]), weight_decay=float(config["train"]["weight_decay"]))
    train_cfg = config["train"]
    loss_cfg = config["loss"]
    epochs = int(args.epochs or train_cfg["epochs"])
    run_dir = timestamped_run_dir(train_cfg["artifact_root"])
    save_json(run_dir / "run_config.json", config)
    (run_dir / "feature_columns.json").write_text(json.dumps(feature_columns, indent=2), encoding="utf-8")
    np.savez(run_dir / "feature_spec.npz", mean=spec.mean, std=spec.std)

    loader = make_loader(datasets["train"], int(train_cfg["batch_size"]), shuffle=True, num_workers=int(train_cfg["num_workers"]))
    best_val = float("inf")
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
                float(loss_cfg["nll_weight"]) * nll
                + float(loss_cfg["aux_weight"]) * aux
                + float(loss_cfg["interval_weight"]) * interval
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        _, val_metrics = evaluate(model, datasets["val"], device)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        print(
            f"epoch {epoch:03d} loss={row['train_loss']:.4f} "
            f"val_rmse={val_metrics['rmse']:.1f} baseline={val_metrics['baseline_rmse']:.1f} "
            f"cov80={val_metrics['coverage_80']:.3f}",
            flush=True,
        )
        if val_metrics["rmse"] < best_val:
            best_val = float(val_metrics["rmse"])
            no_improve = 0
            torch.save(model.state_dict(), run_dir / "best_model.pt")
        else:
            no_improve += 1
            if epoch >= int(train_cfg["early_stopping_min_epochs"]) and no_improve >= int(train_cfg["early_stopping_patience"]):
                print(f"early stopping at epoch {epoch}", flush=True)
                break

    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location=device))
    scale_multiplier = calibrate_scale_multiplier(
        model,
        datasets["val"],
        device,
        target_coverage=float(config["train"].get("target_interval_coverage", 0.80)),
    )
    save_json(run_dir / "calibration.json", {"scale_multiplier": scale_multiplier, "target_coverage": float(config["train"].get("target_interval_coverage", 0.80))})
    for split, dataset in datasets.items():
        pred, metrics = evaluate(model, dataset, device, scale_multiplier=scale_multiplier)
        pred.to_csv(run_dir / f"predictions_{split}.csv", index=False)
        save_json(run_dir / f"metrics_{split}.json", metrics)
        save_forecast_band(pred, run_dir / "figures" / f"forecast_band_{split}.png", f"{split} Student-t PV forecast")
        if split in {"val", "test"}:
            ranked = pred.assign(abs_error=(pred["q50_w"] - pred["target_pv_w"]).abs()).sort_values("abs_error", ascending=False)
            for i, (_, row) in enumerate(ranked.head(int(train_cfg["save_case_count"])).iterrows()):
                save_case_plot(row, run_dir / "figures" / f"case_{split}_{i:02d}.png")

    print(f"run_dir={run_dir}")


if __name__ == "__main__":
    main()
