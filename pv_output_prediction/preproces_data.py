"""preproces_data.py

Step 1 data preparation for PV power mapping using 15-minute mean power labels and
1-minute sky images.

What this script does (simplified):
1) Index images in CAM_DIR and parse their timestamps from filenames.
2) Load PV CSV from power-LSK_N.csv (columns: date, value), treat it as 15-minute
   resolution, and parse timestamps.
3) Build window-level samples: for each PV point with valid positive power, gather
   15 image paths (1-min cadence) for the window ending at ts_power_end.
4) Save all samples (no train/val/test split) to OUT_DIR as parquet (preferred)
   + csv (for inspection).

Notes:
- Assumes PV power peak ~60kW; daylight inference params are tuned accordingly.
- POWER_TS_IS_END=True means the PV timestamp is the end of the 15-min averaging window.
  If your PV timestamp is the window start, set POWER_TS_IS_END=False.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image


# =========================
# CONFIG (edit as needed)
# =========================
@dataclass
class CFG:
    # Paths
    CAM_DIR: Path = Path("../data/cam_dir")
    PV_CSV: Path = Path("../data/power/power-LSK_N.csv")
    OUT_DIR: Path = Path("./derived")

    # Time
    TZ: str = "Asia/Singapore"  # change to "Asia/Hong_Kong" if your data is HK local time

    # PV interpretation
    POWER_TS_IS_END: bool = True  # True: timestamp is window end; False: timestamp is window start

    # Windowing
    WINDOW_LEN: int = 15  # 15 images for a 15-min label

    # Daylight inference (tuned for peak ~60 kW)
    DAYLIGHT_ABS_MIN_KW: float = 2.0       # absolute minimum power to consider daytime (kW)
    DAYLIGHT_FRAC_OF_P95: float = 0.05     # fraction of that day’s P95
    DAYLIGHT_PAD_MIN: int = 30             # pad before/after detected daytime window

    # Label validity within daylight
    # Within daylight: if power is NaN or <= 0 -> treat as missing (drop for now)
    MISSING_POWER_LEQ: float = 0.0

    # Split ratios (chronological)
    TRAIN_RATIO: float = 0.70
    VAL_RATIO: float = 0.15
    TEST_RATIO: float = 0.15

    # Image packing (optional)
    IMG_HEIGHT: int = 64
    IMG_WIDTH: int = 64
    PACK_BATCH_SIZE: int = 64
    PACK_DIR: Path = Path("./derived/packed")
    ENABLE_PACKING: bool = True  # set True to run packing step in main()


cfg = CFG()


# =========================
# Timestamp parsing (use user's function)
# =========================
def parse_timestamp(fname: str) -> Optional[datetime]:
    """Parse timestamp from camera filename.

    Example:
      192.168.10.2_01_20260119093031294_TIMING.jpg

    Extracts the 17-digit block between underscores, then takes first 14 digits as
    YYYYmmddHHMMSS.
    """

    m = re.search(r"_(\d{17})_", fname)
    if m is None:
        return None
    return datetime.strptime(m.group(1)[:14], "%Y%m%d%H%M%S")


# =========================
# Core helpers
# =========================
def build_image_index(cam_dir: Path, tz: str) -> pd.DataFrame:
    rows = []
    for p in sorted(cam_dir.iterdir()):
        if not p.is_file() or p.name.startswith("."):
            continue
        ts_naive = parse_timestamp(p.name)
        if ts_naive is None:
            continue
        ts = pd.Timestamp(ts_naive).tz_localize(tz).round("min")
        rows.append({"ts_img": ts, "img_path": str(p)})

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Keep one image per minute (if duplicates exist, keep the first after sorting)
    df = df.sort_values("ts_img").drop_duplicates(subset=["ts_img"], keep="first").reset_index(drop=True)
    return df


def load_pv_15min(pv_csv: Path, tz: str) -> pd.DataFrame:
    """Load PV CSV that is already at 15-minute resolution.

    This function is made flexible for different column names:
    - Current expected file: power-LSK_N.csv with columns: date, value
    - Also supports older format with columns: datetime, power

    No resampling is performed; we only:
    - parse timestamps
    - apply timezone
    - sort and deduplicate identical timestamps
    """

    df = pd.read_csv(pv_csv)

    # Figure out which columns to use for timestamp & power.
    if {"datetime", "power"}.issubset(df.columns):
        ts_col = "datetime"
        power_col = "power"
    elif {"date", "value"}.issubset(df.columns):
        ts_col = "date"
        power_col = "value"
    else:
        raise ValueError("PV csv must contain either (datetime, power) or (date, value)")

    df["ts_power"] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.dropna(subset=["ts_power"]).copy()

    # timezone
    if df["ts_power"].dt.tz is None:
        df["ts_power"] = df["ts_power"].dt.tz_localize(tz)
    else:
        df["ts_power"] = df["ts_power"].dt.tz_convert(tz)

    # Optional safety: align to minute (should already be on 15-min grid)
    df["ts_power"] = df["ts_power"].dt.round("min")

    # Keep only needed columns and sort.
    # Convert power to numeric and keep the column name as 'power'.
    df["power"] = pd.to_numeric(df[power_col], errors="coerce")
    df = df[["ts_power", "power"]].sort_values("ts_power")

    # If there are duplicate timestamps, average them (rare but happens)
    df = df.groupby("ts_power", as_index=False)["power"].mean()

    return df


def infer_daylight_mask(
    df_15: pd.DataFrame,
    abs_min_kw: float,
    frac_of_p95: float,
    pad_min: int,
    missing_power_leq: float,
) -> pd.DataFrame:
    """Infer daylight window per day using robust thresholding on PV power.

    Returns df with columns:
      - is_daylight: bool
      - is_label_missing: bool (within daylight but power missing/invalid)

    Rationale:
    - Night often has power=0 or NaN.
    - Daytime can also have 0/NaN due to faults/missing data.
    - We infer daylight window by detecting when power is clearly above noise, per day.
    """

    df = df_15.copy()
    df["date"] = df["ts_power"].dt.date

    is_daylight = np.zeros(len(df), dtype=bool)

    for _, g in df.groupby("date"):
        p = g["power"].to_numpy(dtype=float)
        p_valid = p[np.isfinite(p)]

        # Too few points -> skip the day
        if p_valid.size < 4:
            continue

        p95 = float(np.quantile(p_valid, 0.95))
        thr = max(abs_min_kw, frac_of_p95 * p95)

        idx = np.where(p > thr)[0]
        if idx.size == 0:
            continue

        t_start = g.iloc[int(idx[0])]["ts_power"] - pd.Timedelta(minutes=pad_min)
        t_end = g.iloc[int(idx[-1])]["ts_power"] + pd.Timedelta(minutes=pad_min)

        mask = (g["ts_power"] >= t_start) & (g["ts_power"] <= t_end)
        is_daylight[g.index.values] = mask.to_numpy()

    df["is_daylight"] = is_daylight
    df["is_label_missing"] = df["is_daylight"] & (
        (~np.isfinite(df["power"].to_numpy(dtype=float))) | (df["power"].to_numpy(dtype=float) <= missing_power_leq)
    )

    return df.drop(columns=["date"])


def build_windows(df_img: pd.DataFrame, df_pv: pd.DataFrame, window_len: int, power_ts_is_end: bool) -> pd.DataFrame:
    """For each valid PV point, gather 15 image paths at 1-min cadence ending at ts_power_end.

    Window minutes included (len=15): ts_end-14, ..., ts_end

    Drops samples if:
      - label is missing (is_label_missing=True)
      - any required image is missing
    """

    if df_img.empty or df_pv.empty:
        return pd.DataFrame(columns=["ts_power_end", "power", "img_paths"])  # empty

    # Map ts_img -> img_path for fast lookup.
    # Use timezone-aware string keys (all in cfg.TZ, rounded to minute) so that
    # they match the string form of Timestamp objects we create below.
    img_key = df_img["ts_img"].dt.tz_convert(cfg.TZ).dt.round("min").astype(str)
    img_map = dict(zip(img_key, df_img["img_path"].tolist()))

    samples = []
    for _, r in df_pv.iterrows():
        if bool(r.get("is_label_missing", False)):
            continue

        ts_power = r["ts_power"]
        power = r["power"]

        ts_end = ts_power if power_ts_is_end else (ts_power + pd.Timedelta(minutes=15))
        ts_end = ts_end.round("min")

        ts_list = [(ts_end - pd.Timedelta(minutes=i)).round("min") for i in reversed(range(window_len))]

        paths: List[str] = []
        missing = False
        for ts in ts_list:
            # ts is already tz-aware in cfg.TZ and rounded to minute
            key = str(ts)
            p = img_map.get(key)
            if p is None:
                missing = True
                break
            paths.append(p)

        if missing:
            continue

        samples.append({
            "ts_power_end": ts_end,
            "power": float(power),
            "img_paths": paths,
        })

    out = pd.DataFrame(samples)
    if out.empty:
        return out

    out = out.sort_values("ts_power_end").reset_index(drop=True)
    return out


def time_split(df: pd.DataFrame, train_ratio: float, val_ratio: float) -> pd.DataFrame:
    n = len(df)
    if n == 0:
        raise RuntimeError("No window samples were created. Check timestamp parsing, overlap, and missing data.")

    i_train = int(n * train_ratio)
    i_val = int(n * (train_ratio + val_ratio))

    split = np.array(["train"] * n, dtype=object)
    split[i_train:i_val] = "val"
    split[i_val:] = "test"

    df = df.copy()
    df["split"] = split
    return df


# =========================
# Image packing helpers
# =========================


def decode_resize_image(path: str, img_height: int, img_width: int) -> np.ndarray:
    """Decode a JPG/PNG image, convert to RGB, resize and normalize to [0, 1]."""
    with Image.open(path) as im:
        im = im.convert("RGB")
        im = im.resize((img_width, img_height))
        arr = np.asarray(im, dtype=np.float32) / 255.0
    return arr


def pack_windows_to_npz(
    df_win: pd.DataFrame,
    out_dir: Path,
    window_len: int,
    img_height: int,
    img_width: int,
    batch_size: int,
) -> None:
    """Pack window samples into pre-decoded image batches saved as .npz.

    Each window sample has:
      - power scalar label
      - img_paths: list[str] of length WINDOW_LEN

    This function will produce shards:
      out_dir/batch_000000_000063.npz, ...

    Each shard contains:
      - images: (B, T, H, W, 3) float32 in [0,1]
      - power:  (B,) float32
      - ts_power_end: (B,) ISO strings
    """
    if df_win.empty:
        print("[pack_windows_to_npz] df_win is empty, nothing to pack.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    num_samples = len(df_win)
    print(f"[pack_windows_to_npz] Packing {num_samples} samples into batches of {batch_size} ...")

    for start in range(0, num_samples, batch_size):
        end = min(start + batch_size, num_samples)
        batch_df = df_win.iloc[start:end]
        bsz = len(batch_df)

        images = np.zeros((bsz, window_len, img_height, img_width, 3), dtype=np.float32)
        power = np.zeros((bsz,), dtype=np.float32)
        ts_end: List[str] = []

        for i, (_, row) in enumerate(batch_df.iterrows()):
            paths: List[str] = row["img_paths"]
            if len(paths) != window_len:
                raise ValueError(
                    f"Sample {start + i} has {len(paths)} images, expected {window_len}."
                )
            for t, p in enumerate(paths):
                images[i, t] = decode_resize_image(p, img_height, img_width)
            power[i] = float(row["power"])
            ts_end.append(str(row["ts_power_end"]))

        shard_name = f"batch_{start:06d}_{end-1:06d}.npz"
        shard_path = out_dir / shard_name
        np.savez_compressed(
            shard_path,
            images=images,
            power=power,
            ts_power_end=np.array(ts_end, dtype=object),
        )
        print(f"  saved shard {shard_path} (samples {start}..{end-1})")


def main() -> None:
    cfg.OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[1] Build image index...")
    df_img = build_image_index(cfg.CAM_DIR, cfg.TZ)

    if df_img.empty:
        raise RuntimeError(f"No images indexed in {cfg.CAM_DIR}. Check path and filename format.")
    print(f"  images: {len(df_img)} | {df_img['ts_img'].min()} -> {df_img['ts_img'].max()}")

    print("[2] Load PV (already 15min resolution)...")
    df_pv = load_pv_15min(cfg.PV_CSV, cfg.TZ)
    if df_pv.empty:
        raise RuntimeError(f"PV CSV loaded but empty after parsing: {cfg.PV_CSV}")
    print(f"  pv rows (15min): {len(df_pv)} | {df_pv['ts_power'].min()} -> {df_pv['ts_power'].max()}")

    # 简单处理：只保留有有效功率的时间点（>0 且非 NaN），不再做白天推断和 train/val/test 划分
    print("[3] Filter valid PV rows (power > 0)...")
    df_pv = df_pv.copy()
    df_pv["power"] = pd.to_numeric(df_pv["power"], errors="coerce")
    df_pv = df_pv.dropna(subset=["power"])
    df_pv = df_pv[df_pv["power"] > 0].reset_index(drop=True)
    print(f"  valid pv rows: {len(df_pv)}")

    print("[4] Build window samples (15 images -> scalar)...")
    df_win = build_windows(df_img, df_pv, cfg.WINDOW_LEN, cfg.POWER_TS_IS_END)
    print(f"  windows built: {len(df_win)}")

    out_parquet = cfg.OUT_DIR / "pv_windows_simple.parquet"
    out_csv = cfg.OUT_DIR / "pv_windows_simple.csv"

    df_win.to_parquet(out_parquet, index=False)
    df_win.to_csv(out_csv, index=False)

    print("Saved window metadata:")
    print(f"  {out_parquet}")
    print(f"  {out_csv}")

    # Optional: pre-pack image data into .npz shards
    if cfg.ENABLE_PACKING:
        print("[5] Pack window samples into pre-decoded image batches (.npz)...")
        pack_windows_to_npz(
            df_win=df_win,
            out_dir=cfg.PACK_DIR,
            window_len=cfg.WINDOW_LEN,
            img_height=cfg.IMG_HEIGHT,
            img_width=cfg.IMG_WIDTH,
            batch_size=cfg.PACK_BATCH_SIZE,
        )
        print(f"  packed shards saved under: {cfg.PACK_DIR}")


if __name__ == "__main__":
    main()
