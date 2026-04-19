from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scsn_model.data.solar_geometry import Calibration, compute_solar_position, project_sun_to_image
from scsn_model.utils.io import ensure_dir, load_json, normalize_config_paths, resolve_project_path, save_json
from scsn_model.utils.runtime import configure_matplotlib_cache

configure_matplotlib_cache(ROOT / "artifacts")

TIMESTAMP_PATTERN = re.compile(r"_(\d{17})_")


def parse_ts_from_name(name: str, timezone: str) -> pd.Timestamp | None:
    match = TIMESTAMP_PATTERN.search(name)
    if match is None:
        return None
    ts = match.group(1)
    try:
        dt = datetime.strptime(ts[:14], "%Y%m%d%H%M%S")
    except Exception:
        return None
    return pd.Timestamp(dt).tz_localize(timezone)


def load_mask(mask_path: Path, image_shape: tuple[int, int]) -> np.ndarray:
    if not mask_path.exists():
        raise FileNotFoundError(f"Sky mask not found: {mask_path}")
    mask = Image.open(mask_path).convert("L")
    h, w = image_shape
    if mask.size != (w, h):
        mask = mask.resize((w, h), resample=Image.NEAREST)
    return np.asarray(mask, dtype=np.uint8)


def detect_sun_centroid_simple(img_rgb: np.ndarray, sky_mask: np.ndarray) -> tuple[tuple[float, float] | None, float]:
    arr = img_rgb.astype(np.float32) / 255.0
    mask = sky_mask > 0
    if not np.any(mask):
        return None, 0.0

    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]
    v = np.max(arr, axis=2)
    min_rgb = np.minimum(np.minimum(r, g), b)
    s = np.zeros_like(v)
    valid = v > 1e-6
    s[valid] = (v[valid] - min_rgb[valid]) / v[valid]

    v_masked = v[mask]
    if v_masked.size == 0 or float(v_masked.max()) < 0.7:
        return None, 0.0
    masked_v = np.where(mask, v, -1.0)
    peak_idx = int(np.argmax(masked_v))
    peak_y, peak_x = np.unravel_index(peak_idx, masked_v.shape)
    peak_v = float(masked_v[peak_y, peak_x])
    if peak_v <= 0:
        return None, 0.0

    radius = 14
    y0 = max(0, peak_y - radius)
    y1 = min(v.shape[0], peak_y + radius + 1)
    x0 = max(0, peak_x - radius)
    x1 = min(v.shape[1], peak_x + radius + 1)

    yy, xx = np.mgrid[y0:y1, x0:x1]
    local_mask = mask[y0:y1, x0:x1]
    local_v = v[y0:y1, x0:x1]
    local_s = s[y0:y1, x0:x1]

    # Focus on the immediate bright solar core around the brightest pixel.
    cand = local_mask & (local_v >= max(peak_v * 0.94, float(np.percentile(v_masked, 99.5)))) & (local_s <= 0.5)
    if np.count_nonzero(cand) == 0:
        cand = local_mask & (local_v >= peak_v * 0.97)
    if np.count_nonzero(cand) == 0:
        return (float(peak_x), float(peak_y)), peak_v

    weights = local_v[cand]
    cx = float(np.sum(xx[cand] * weights) / np.sum(weights))
    cy = float(np.sum(yy[cand] * weights) / np.sum(weights))
    score = float(np.mean(weights))
    return (cx, cy), score


def gather_samples(
    config: dict,
    date_str: str,
    stride_min: int,
    max_samples: int | None,
    min_score: float,
) -> tuple[pd.DataFrame, tuple[int, int]]:
    data_cfg = config["data"]
    timezone = data_cfg["timezone"]
    date_prefix = pd.Timestamp(date_str).strftime("%Y-%m-%d")
    camera_root = resolve_project_path(data_cfg["camera_dir"], must_exist=True)

    candidates = sorted(camera_root.rglob("*2026*.jpg"))
    rows: list[dict] = []
    last_bucket: pd.Timestamp | None = None
    image_shape: tuple[int, int] | None = None

    for path in candidates:
        ts = parse_ts_from_name(path.name, timezone)
        if ts is None or ts.strftime("%Y-%m-%d") != date_prefix:
            continue
        bucket = ts.floor(f"{stride_min}min")
        if last_bucket is not None and bucket == last_bucket:
            continue

        with Image.open(path) as im:
            img = np.asarray(im.convert("RGB"))
        image_shape = img.shape[:2]
        mask = load_mask(resolve_project_path(data_cfg["sky_mask_path"], must_exist=True), image_shape)
        centroid, score = detect_sun_centroid_simple(img, mask)
        if centroid is None or float(score) < min_score:
            continue

        rows.append(
            {
                "timestamp": ts,
                "image_path": str(path),
                "u_detect": float(centroid[0]),
                "v_detect": float(centroid[1]),
                "score": float(score),
            }
        )
        last_bucket = bucket
        if max_samples is not None and len(rows) >= max_samples:
            break

    if not rows or image_shape is None:
        raise RuntimeError(f"No valid sun detections found for {date_str}.")

    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    return df, image_shape


def evaluate_orientation(
    sample_df: pd.DataFrame,
    calib: Calibration,
    image_width: int,
    image_height: int,
    offset_deg: float,
    clockwise: bool,
    image_offset_x_px: float = 0.0,
    image_offset_y_px: float = 0.0,
) -> tuple[float, np.ndarray, np.ndarray]:
    solar = compute_solar_position(sample_df["timestamp"].tolist(), calib)
    x_pred, y_pred = project_sun_to_image(
        azimuth_deg=solar["azimuth_deg"].to_numpy(dtype=np.float32),
        zenith_deg=solar["zenith_deg"].to_numpy(dtype=np.float32),
        calib=calib,
        image_width=image_width,
        image_height=image_height,
        azimuth_offset_deg=offset_deg,
        azimuth_clockwise=clockwise,
        image_offset_x_px=image_offset_x_px,
        image_offset_y_px=image_offset_y_px,
    )
    dx = x_pred - sample_df["u_detect"].to_numpy(dtype=np.float32)
    dy = y_pred - sample_df["v_detect"].to_numpy(dtype=np.float32)
    err = np.sqrt(dx ** 2 + dy ** 2)
    return float(np.mean(err)), np.asarray(x_pred), np.asarray(y_pred)


def search_best_projection(
    sample_df: pd.DataFrame,
    calib: Calibration,
    image_width: int,
    image_height: int,
) -> dict:
    solar = compute_solar_position(sample_df["timestamp"].tolist(), calib)
    az = solar["azimuth_deg"].to_numpy(dtype=np.float64)
    zen = solar["zenith_deg"].to_numpy(dtype=np.float64)
    theta = np.deg2rad(zen)
    u_obs = sample_df["u_detect"].to_numpy(dtype=np.float64)
    v_obs = sample_df["v_detect"].to_numpy(dtype=np.float64)

    best: dict | None = None
    for clockwise in (True, False):
        direction = 1.0 if clockwise else -1.0

        def residuals(params: np.ndarray) -> np.ndarray:
            cx, cy, f, offset_deg = params
            ang = np.deg2rad(direction * az + offset_deg)
            r = f * theta
            x = cx + r * np.sin(ang)
            y = cy - r * np.cos(ang)
            return np.concatenate([x - u_obs, y - v_obs])

        x0 = np.array([calib.cx, calib.cy, calib.f_px_per_rad, 0.0], dtype=np.float64)
        bounds_lo = np.array([0.0, 0.0, 10.0, -720.0], dtype=np.float64)
        bounds_hi = np.array([image_width, image_height, 500.0, 720.0], dtype=np.float64)
        res = least_squares(residuals, x0=x0, bounds=(bounds_lo, bounds_hi), loss="soft_l1", f_scale=8.0, max_nfev=4000)

        cx, cy, f, offset_deg = map(float, res.x)
        offset_deg = offset_deg % 360.0
        ang = np.deg2rad(direction * az + offset_deg)
        r = f * theta
        x_pred = cx + r * np.sin(ang)
        y_pred = cy - r * np.cos(ang)
        err = np.sqrt((x_pred - u_obs) ** 2 + (y_pred - v_obs) ** 2)
        mean_err = float(np.mean(err))

        if best is None or mean_err < best["mean_error_px"]:
            best = {
                "azimuth_offset_deg": offset_deg,
                "azimuth_clockwise": bool(clockwise),
                "mean_error_px": mean_err,
                "x_pred": x_pred,
                "y_pred": y_pred,
                "projection_cx_px": cx,
                "projection_cy_px": cy,
                "projection_f_px_per_rad": f,
                "image_offset_x_px": 0.0,
                "image_offset_y_px": 0.0,
            }

    assert best is not None
    return best


def save_overlays(sample_df: pd.DataFrame, best: dict, out_dir: Path) -> None:
    overlay_dir = ensure_dir(out_dir / "overlays")
    x_pred = best["x_pred"]
    y_pred = best["y_pred"]
    sample_df = sample_df.copy()
    sample_df["u_pred"] = x_pred
    sample_df["v_pred"] = y_pred
    sample_df["err_px"] = np.sqrt(
        (sample_df["u_pred"] - sample_df["u_detect"]) ** 2 + (sample_df["v_pred"] - sample_df["v_detect"]) ** 2
    )

    for idx, row in sample_df.iterrows():
        with Image.open(resolve_project_path(row["image_path"], must_exist=True)) as im:
            img = im.convert("RGB")
        draw = ImageDraw.Draw(img)
        ud, vd = float(row["u_detect"]), float(row["v_detect"])
        up, vp = float(row["u_pred"]), float(row["v_pred"])
        draw.ellipse((ud - 8, vd - 8, ud + 8, vd + 8), outline=(0, 255, 0), width=2)
        draw.ellipse((up - 8, vp - 8, up + 8, vp + 8), outline=(255, 0, 0), width=2)
        draw.line((ud, vd, up, vp), fill=(255, 255, 255), width=1)
        text = (
            f"{pd.Timestamp(row['timestamp']).strftime('%Y-%m-%d %H:%M')}  "
            f"err={row['err_px']:.1f}px  det=green pred=red"
        )
        draw.text((10, 10), text, fill=(255, 255, 0))
        out_path = overlay_dir / f"{idx:03d}_{pd.Timestamp(row['timestamp']).strftime('%H%M')}.png"
        img.save(out_path)

    sample_df.to_csv(out_dir / "orientation_fit_samples.csv", index=False)


def maybe_update_config(config_path: Path, best: dict) -> None:
    config_path = config_path.expanduser().resolve()
    config = load_json(config_path)
    config["data"]["azimuth_offset_deg"] = round(float(best["azimuth_offset_deg"]), 3)
    config["data"]["azimuth_clockwise"] = bool(best["azimuth_clockwise"])
    config["data"]["sun_image_offset_x_px"] = round(float(best.get("image_offset_x_px", 0.0)), 3)
    config["data"]["sun_image_offset_y_px"] = round(float(best.get("image_offset_y_px", 0.0)), 3)
    config["data"]["sun_projection_cx_px"] = round(float(best.get("projection_cx_px")), 3)
    config["data"]["sun_projection_cy_px"] = round(float(best.get("projection_cy_px")), 3)
    config["data"]["sun_projection_f_px_per_rad"] = round(float(best.get("projection_f_px_per_rad")), 3)
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate sun azimuth orientation using detected sun centroids")
    parser.add_argument("--config", required=True)
    parser.add_argument("--date", default="2026-01-18")
    parser.add_argument("--stride-min", type=int, default=15)
    parser.add_argument("--max-samples", type=int, default=48)
    parser.add_argument("--min-score", type=float, default=0.15)
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "artifacts" / "sun_orientation_calibration"),
    )
    parser.add_argument("--update-config", action="store_true")
    args = parser.parse_args()

    config = normalize_config_paths(load_json(args.config))
    sample_df, image_shape = gather_samples(
        config=config,
        date_str=args.date,
        stride_min=args.stride_min,
        max_samples=args.max_samples,
        min_score=args.min_score,
    )
    img_h, img_w = image_shape
    calib = Calibration.from_json(resolve_project_path(config["data"]["calibration_json"], must_exist=True)).rescale(dst_w=img_w, dst_h=img_h)
    best = search_best_projection(sample_df, calib, img_w, img_h)

    out_dir = ensure_dir(resolve_project_path(args.out_dir, must_exist=False) / args.date)
    save_overlays(sample_df, best, out_dir)
    summary = {
        "date": args.date,
        "n_samples": int(len(sample_df)),
        "image_width": int(img_w),
        "image_height": int(img_h),
        "azimuth_offset_deg": float(best["azimuth_offset_deg"]),
        "azimuth_clockwise": bool(best["azimuth_clockwise"]),
        "image_offset_x_px": float(best.get("image_offset_x_px", 0.0)),
        "image_offset_y_px": float(best.get("image_offset_y_px", 0.0)),
        "projection_cx_px": float(best.get("projection_cx_px")),
        "projection_cy_px": float(best.get("projection_cy_px")),
        "projection_f_px_per_rad": float(best.get("projection_f_px_per_rad")),
        "mean_error_px": float(best["mean_error_px"]),
        "config_path": str(Path(args.config).expanduser().resolve()),
    }
    save_json(out_dir / "orientation_fit_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.update_config:
        maybe_update_config(Path(args.config), best)
        print(f"Updated config: {args.config}")


if __name__ == "__main__":
    main()
