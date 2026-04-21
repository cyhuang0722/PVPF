from __future__ import annotations

import argparse
import ast
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Config:
    samples_csv: Path
    camera_index_csv: Path
    sky_mask_path: Path
    output_csv: Path
    image_size: int = 256
    start_hour: int = 8
    end_hour: int = 17
    min_day_samples: int = 12
    min_image_samples: int = 12
    r2_threshold: float = 0.99
    mae_clear_sky_index_threshold: float = 0.06
    min_blue_fraction: float = 0.35
    max_gray_fraction: float = 0.60
    max_bright_white_fraction: float = 0.35
    overcast_max_mean_target_csi: float = 0.20
    overcast_max_max_target_csi: float = 0.35
    overcast_max_pv_peak_ratio: float = 0.35
    overcast_max_csi_p90: float = 0.30
    image_stride_min: int = 30


def _load_json_list(value: object) -> list:
    if isinstance(value, list):
        return value
    if not isinstance(value, str):
        return []
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return ast.literal_eval(value)


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
    if ss_tot <= 1e-12:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def _load_mask(mask_path: Path, image_size: int) -> np.ndarray | None:
    if not mask_path.exists():
        return None
    with Image.open(mask_path) as im:
        mask = im.convert("L")
        if mask.size != (image_size, image_size):
            mask = mask.resize((image_size, image_size), resample=Image.NEAREST)
        return (np.asarray(mask, dtype=np.float32) / 255.0) >= 0.5


def _load_rgb(path: Path, image_size: int) -> np.ndarray | None:
    try:
        with Image.open(path) as im:
            rgb = im.convert("RGB")
            if rgb.size != (image_size, image_size):
                rgb = rgb.resize((image_size, image_size), resample=Image.BILINEAR)
            return np.asarray(rgb, dtype=np.float32) / 255.0
    except OSError:
        return None


def _full_image_mask(image_size: int) -> np.ndarray:
    return np.ones((image_size, image_size), dtype=bool)


def _image_metrics(image_path: Path, sky_mask: np.ndarray, image_size: int) -> dict[str, float] | None:
    rgb = _load_rgb(image_path, image_size)
    if rgb is None:
        return None

    red = rgb[..., 0]
    green = rgb[..., 1]
    blue = rgb[..., 2]
    rgb_min = np.min(rgb, axis=-1)
    rgb_max = np.max(rgb, axis=-1)
    saturation = (rgb_max - rgb_min) / np.clip(rgb_max, 1e-6, None)
    value = rgb_max

    blue_dominant = (blue >= red * 1.05) & (blue >= green * 0.92)
    blue_sky = sky_mask & blue_dominant & (saturation >= 0.18) & (value >= 0.12)
    gray_cloud = sky_mask & (saturation <= 0.22) & (value >= 0.35)
    bright_white = sky_mask & (red >= 0.72) & (green >= 0.72) & (blue >= 0.72) & (saturation <= 0.18)

    denom = max(int(np.sum(sky_mask)), 1)
    return {
        "blue_fraction": float(np.sum(blue_sky) / denom),
        "gray_fraction": float(np.sum(gray_cloud) / denom),
        "bright_white_fraction": float(np.sum(bright_white) / denom),
        "mean_saturation": float(np.mean(saturation[sky_mask])),
        "mean_value": float(np.mean(value[sky_mask])),
    }


def _compute_spm_predictions(samples: pd.DataFrame) -> pd.DataFrame:
    df = samples.copy()
    df["ts_anchor"] = pd.to_datetime(df["ts_anchor"])
    df["ts_target"] = pd.to_datetime(df["ts_target"])
    df = df.sort_values("ts_target").reset_index(drop=True)
    df["date"] = df["ts_target"].dt.date.astype(str)
    df["hour"] = df["ts_target"].dt.hour
    df["prev_pv_w"] = df["past_pv_w"].apply(lambda x: float(_load_json_list(x)[-1]))
    df["anchor_clear_sky_w"] = df["target_clear_sky_w"].shift(1)
    same_series = df["ts_anchor"].eq(df["ts_target"].shift(1))
    df.loc[~same_series, "anchor_clear_sky_w"] = np.nan

    ratio = df["target_clear_sky_w"] / df["anchor_clear_sky_w"].clip(lower=1e-6)
    df["spm_pred_w"] = (df["prev_pv_w"] * ratio).clip(lower=0.0)
    df["target_csi"] = df["target_pv_w"] / df["target_clear_sky_w"].clip(lower=1e-6)
    df["spm_pred_csi"] = df["spm_pred_w"] / df["target_clear_sky_w"].clip(lower=1e-6)
    return df.dropna(subset=["spm_pred_w", "target_pv_w", "target_clear_sky_w"])


def _daily_pv_metrics(pred_df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    daylight = pred_df[(pred_df["hour"] >= cfg.start_hour) & (pred_df["hour"] <= cfg.end_hour)].copy()
    rows: list[dict[str, object]] = []
    for date, day_df in daylight.groupby("date", sort=True):
        if len(day_df) < cfg.min_day_samples:
            continue
        y_true = day_df["target_pv_w"].to_numpy(dtype=np.float64)
        y_pred = day_df["spm_pred_w"].to_numpy(dtype=np.float64)
        csi_true = day_df["target_csi"].to_numpy(dtype=np.float64)
        csi_pred = day_df["spm_pred_csi"].to_numpy(dtype=np.float64)
        peak_clear_sky_w = float(np.max(day_df["target_clear_sky_w"].to_numpy(dtype=np.float64)))
        peak_pv_w = float(np.max(y_true))
        rows.append(
            {
                "date": date,
                "n_pv_samples": int(len(day_df)),
                "pv_r2": _r2_score(y_true, y_pred),
                "pv_mae_w": float(np.mean(np.abs(y_true - y_pred))),
                "pv_rmse_w": float(math.sqrt(np.mean((y_true - y_pred) ** 2))),
                "csi_mae": float(np.mean(np.abs(csi_true - csi_pred))),
                "mean_target_csi": float(np.mean(csi_true)),
                "p90_target_csi": float(np.quantile(csi_true, 0.90)),
                "min_target_csi": float(np.min(csi_true)),
                "max_target_csi": float(np.max(csi_true)),
                "peak_pv_w": peak_pv_w,
                "peak_clear_sky_w": peak_clear_sky_w,
                "peak_pv_ratio": float(peak_pv_w / max(peak_clear_sky_w, 1e-6)),
            }
        )
    return pd.DataFrame.from_records(rows)


def _select_images_for_day(day_images: pd.DataFrame, stride_min: int) -> pd.DataFrame:
    if day_images.empty:
        return day_images
    work = day_images.sort_values("timestamp").copy()
    work["bucket"] = work["timestamp"].dt.floor(f"{stride_min}min")
    center = work["bucket"] + pd.to_timedelta(stride_min / 2, unit="min")
    work["distance_to_bucket_center"] = (work["timestamp"] - center).abs()
    return (
        work.sort_values(["bucket", "distance_to_bucket_center"])
        .groupby("bucket", as_index=False)
        .head(1)
        .drop(columns=["bucket", "distance_to_bucket_center"])
    )


def _daily_image_metrics(camera_index: pd.DataFrame, sky_mask: np.ndarray, cfg: Config) -> pd.DataFrame:
    index_df = camera_index.copy()
    index_df["timestamp"] = pd.to_datetime(index_df["timestamp"])
    index_df["date"] = index_df["timestamp"].dt.date.astype(str)
    index_df["hour"] = index_df["timestamp"].dt.hour
    index_df = index_df[(index_df["hour"] >= cfg.start_hour) & (index_df["hour"] <= cfg.end_hour)]

    rows: list[dict[str, object]] = []
    for date, day_images in index_df.groupby("date", sort=True):
        selected = _select_images_for_day(day_images, cfg.image_stride_min)
        metrics = []
        for raw_path in selected["file_path"].astype(str):
            metric = _image_metrics(Path(raw_path), sky_mask, cfg.image_size)
            if metric is not None:
                metrics.append(metric)
        if len(metrics) < cfg.min_image_samples:
            continue

        metric_df = pd.DataFrame.from_records(metrics)
        rows.append(
            {
                "date": date,
                "n_image_samples": int(len(metric_df)),
                "blue_fraction_mean": float(metric_df["blue_fraction"].mean()),
                "blue_fraction_p10": float(metric_df["blue_fraction"].quantile(0.10)),
                "gray_fraction_mean": float(metric_df["gray_fraction"].mean()),
                "gray_fraction_p90": float(metric_df["gray_fraction"].quantile(0.90)),
                "bright_white_fraction_mean": float(metric_df["bright_white_fraction"].mean()),
                "mean_saturation": float(metric_df["mean_saturation"].mean()),
                "mean_value": float(metric_df["mean_value"].mean()),
            }
        )
    return pd.DataFrame.from_records(rows)


def generate_clear_sky_table(cfg: Config) -> pd.DataFrame:
    samples = pd.read_csv(cfg.samples_csv)
    camera_index = pd.read_csv(cfg.camera_index_csv)
    sky_mask = _load_mask(cfg.sky_mask_path, cfg.image_size)
    if sky_mask is None:
        sky_mask = _full_image_mask(cfg.image_size)

    pred_df = _compute_spm_predictions(samples)
    pv_daily = _daily_pv_metrics(pred_df, cfg)
    image_daily = _daily_image_metrics(camera_index, sky_mask, cfg)

    if pv_daily.empty:
        raise RuntimeError("No day has enough PV samples to compute daily smart-persistence R2.")

    if image_daily.empty:
        merged = pv_daily.copy()
        merged["n_image_samples"] = 0
        merged["image_clear_sky"] = False
    else:
        merged = pv_daily.merge(image_daily, on="date", how="left")
        merged["image_clear_sky"] = (
            (merged["blue_fraction_mean"] >= cfg.min_blue_fraction)
            & (merged["gray_fraction_mean"] <= cfg.max_gray_fraction)
            & (merged["bright_white_fraction_mean"] <= cfg.max_bright_white_fraction)
        )

    merged["pv_clear_sky"] = (merged["pv_r2"] >= cfg.r2_threshold) & (
        merged["csi_mae"] <= cfg.mae_clear_sky_index_threshold
    )
    merged["is_clear_sky"] = merged["pv_clear_sky"] & merged["image_clear_sky"].fillna(False)
    merged["is_overcast"] = (
        (merged["mean_target_csi"] <= cfg.overcast_max_mean_target_csi)
        & (merged["max_target_csi"] <= cfg.overcast_max_max_target_csi)
        & (merged["p90_target_csi"] <= cfg.overcast_max_csi_p90)
        & (merged["peak_pv_ratio"] <= cfg.overcast_max_pv_peak_ratio)
    ).fillna(False)
    merged["usable_for_cloud_mask_ref"] = (~merged["is_clear_sky"]) & (~merged["is_overcast"])
    merged["start_time"] = f"{cfg.start_hour:02d}:00"
    merged["end_time"] = f"{cfg.end_hour:02d}:59"
    merged["notes"] = np.select(
        [merged["is_clear_sky"], merged["is_overcast"]],
        [
            "auto clear sky by SPM daily R2 and image color heuristic",
            "auto overcast/rain by low CSI and low PV peak heuristic",
        ],
        default="auto non-clear usable for cloud-mask reference",
    )

    preferred_columns = [
        "date",
        "start_time",
        "end_time",
        "is_clear_sky",
        "is_overcast",
        "usable_for_cloud_mask_ref",
        "pv_clear_sky",
        "image_clear_sky",
        "pv_r2",
        "csi_mae",
        "pv_mae_w",
        "pv_rmse_w",
        "mean_target_csi",
        "p90_target_csi",
        "min_target_csi",
        "max_target_csi",
        "peak_pv_w",
        "peak_clear_sky_w",
        "peak_pv_ratio",
        "n_pv_samples",
        "n_image_samples",
        "blue_fraction_mean",
        "blue_fraction_p10",
        "gray_fraction_mean",
        "gray_fraction_p90",
        "bright_white_fraction_mean",
        "mean_saturation",
        "mean_value",
        "notes",
    ]
    existing_columns = [col for col in preferred_columns if col in merged.columns]
    return merged[existing_columns].sort_values("date").reset_index(drop=True)


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Generate an automatic clear-sky day table.")
    parser.add_argument(
        "--samples-csv",
        default=str(REPO_ROOT / "new-model/artifacts/dataset/samples.csv"),
        help="Prepared sample CSV with PV targets and image paths.",
    )
    parser.add_argument(
        "--camera-index-csv",
        default=str(REPO_ROOT / "data/camera_data/index/raw_index_resized_256.csv"),
        help="Camera image index CSV.",
    )
    parser.add_argument(
        "--sky-mask-path",
        default=str(REPO_ROOT / "data/sky_mask.png"),
        help="Sky mask image. If missing, the whole image is used.",
    )
    parser.add_argument(
        "--output-csv",
        default=str(REPO_ROOT / "data/clear_sky_generated.csv"),
        help="Output clear-sky table CSV.",
    )
    parser.add_argument("--r2-threshold", type=float, default=0.99)
    parser.add_argument("--csi-mae-threshold", type=float, default=0.06)
    parser.add_argument("--min-blue-fraction", type=float, default=0.35)
    parser.add_argument("--max-gray-fraction", type=float, default=0.60)
    parser.add_argument("--max-bright-white-fraction", type=float, default=0.35)
    parser.add_argument("--overcast-max-mean-target-csi", type=float, default=0.20)
    parser.add_argument("--overcast-max-max-target-csi", type=float, default=0.35)
    parser.add_argument("--overcast-max-pv-peak-ratio", type=float, default=0.35)
    parser.add_argument("--overcast-max-csi-p90", type=float, default=0.30)
    parser.add_argument("--start-hour", type=int, default=8)
    parser.add_argument("--end-hour", type=int, default=17)
    parser.add_argument("--min-day-samples", type=int, default=12)
    parser.add_argument("--min-image-samples", type=int, default=12)
    parser.add_argument("--image-stride-min", type=int, default=30)
    args = parser.parse_args()
    return Config(
        samples_csv=Path(args.samples_csv).resolve(),
        camera_index_csv=Path(args.camera_index_csv).resolve(),
        sky_mask_path=Path(args.sky_mask_path).resolve(),
        output_csv=Path(args.output_csv).resolve(),
        r2_threshold=float(args.r2_threshold),
        mae_clear_sky_index_threshold=float(args.csi_mae_threshold),
        min_blue_fraction=float(args.min_blue_fraction),
        max_gray_fraction=float(args.max_gray_fraction),
        max_bright_white_fraction=float(args.max_bright_white_fraction),
        overcast_max_mean_target_csi=float(args.overcast_max_mean_target_csi),
        overcast_max_max_target_csi=float(args.overcast_max_max_target_csi),
        overcast_max_pv_peak_ratio=float(args.overcast_max_pv_peak_ratio),
        overcast_max_csi_p90=float(args.overcast_max_csi_p90),
        start_hour=int(args.start_hour),
        end_hour=int(args.end_hour),
        min_day_samples=int(args.min_day_samples),
        min_image_samples=int(args.min_image_samples),
        image_stride_min=int(args.image_stride_min),
    )


def main() -> None:
    cfg = parse_args()
    clear_sky_table = generate_clear_sky_table(cfg)
    cfg.output_csv.parent.mkdir(parents=True, exist_ok=True)
    clear_sky_table.to_csv(cfg.output_csv, index=False)

    n_clear = int(clear_sky_table["is_clear_sky"].sum())
    print(f"Wrote {len(clear_sky_table)} daily rows to {cfg.output_csv}")
    print(f"Clear-sky days: {n_clear}")
    if n_clear:
        print(clear_sky_table.loc[clear_sky_table["is_clear_sky"], ["date", "pv_r2", "csi_mae"]].to_string(index=False))


if __name__ == "__main__":
    main()
