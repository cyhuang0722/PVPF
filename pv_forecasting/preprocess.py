"""
Forecasting 数据预处理：60 分钟天空图 + 过去 4 个 15min PV + 预测 4 个 horizon。
样本以 15 分钟为锚点 t：天空图 t-59..t，past_pv=[t-45,t-30,t-15,t]，targets=[t+15,t+30,t+45,t+60]。
支持将图像预打包为 .npz 分片（resize + 可选 mask + [0,1] 归一化）。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image

# 复用 pv_output_prediction 的解析与加载（同目录则直接导入）
_base = Path(__file__).resolve().parent.parent
if str(_base) not in sys.path:
    sys.path.insert(0, str(_base))
try:
    from pv_output_prediction.preproces_data import (
        parse_timestamp,
        build_image_index,
        load_pv_15min,
        CFG as BASE_CFG,
    )
except ImportError:
    from preproces_data import parse_timestamp, build_image_index, load_pv_15min
    BASE_CFG = None

IMG_LEN = 60       # 60 张 1-min 天空图
PAST_PV_LEN = 4    # 过去 4 个 15min
HORIZON = 4        # 4 个预测时刻 [t+15, t+30, t+45, t+60]
# 容差匹配：目标分钟与最近图像的绝对时间差阈值（秒），避免 round 导致秒级偏移误删
IMG_MATCH_TOLERANCE_SEC = 45


def build_image_index_raw(cam_dir: Path, tz: str) -> pd.DataFrame:
    """与 build_image_index 相同，但保留原始时间戳（不 round），用于容差匹配。"""
    rows = []
    for p in sorted(cam_dir.iterdir()):
        if not p.is_file() or p.name.startswith("."):
            continue
        ts_naive = parse_timestamp(p.name)
        if ts_naive is None:
            continue
        ts = pd.Timestamp(ts_naive).tz_localize(tz)
        rows.append({"ts_img": ts, "img_path": str(p)})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("ts_img").reset_index(drop=True)


def print_time_coverage_debug(df_img: pd.DataFrame, df_pv: pd.DataFrame, tz: str) -> None:
    """打印图像/PV 时间覆盖与间隔诊断，便于定位为什么样本只集中在少数小时。"""
    print("[debug] ---- time coverage ----")

    if not df_img.empty:
        img_ts = pd.to_datetime(df_img["ts_img"]).sort_values()
        img_diff = img_ts.diff().dropna().dt.total_seconds()
        print(f"[debug][img] range: {img_ts.iloc[0]} -> {img_ts.iloc[-1]} (n={len(img_ts)})")
        if len(img_diff):
            print(
                "[debug][img] gap_sec: "
                f"min={img_diff.min():.1f}, median={img_diff.median():.1f}, "
                f"p95={img_diff.quantile(0.95):.1f}, max={img_diff.max():.1f}"
            )
        img_hourly = img_ts.dt.hour.value_counts().sort_index()
        print("[debug][img] count by hour:")
        print(img_hourly.to_string())
    else:
        print("[debug][img] empty")

    if not df_pv.empty:
        pv_ts = pd.to_datetime(df_pv["ts_power"])
        if getattr(pv_ts.dt, "tz", None) is None:
            pv_ts = pv_ts.dt.tz_localize(tz)
        else:
            pv_ts = pv_ts.dt.tz_convert(tz)
        pv_ts = pv_ts.sort_values()
        pv_diff = pv_ts.diff().dropna().dt.total_seconds()
        print(f"[debug][pv] range: {pv_ts.iloc[0]} -> {pv_ts.iloc[-1]} (n={len(pv_ts)})")
        if len(pv_diff):
            print(
                "[debug][pv] gap_sec: "
                f"min={pv_diff.min():.1f}, median={pv_diff.median():.1f}, "
                f"p95={pv_diff.quantile(0.95):.1f}, max={pv_diff.max():.1f}"
            )
        pv_hourly = pv_ts.dt.hour.value_counts().sort_index()
        print("[debug][pv] count by hour:")
        print(pv_hourly.to_string())
    else:
        print("[debug][pv] empty")

    print("[debug] -----------------------")


def build_forecast_windows(
    df_img: pd.DataFrame,
    df_pv: pd.DataFrame,
    img_len: int = IMG_LEN,
    past_pv_len: int = PAST_PV_LEN,
    horizon: int = HORIZON,
    tz: str = "Asia/Singapore",
    tolerance_sec: float = IMG_MATCH_TOLERANCE_SEC,
) -> pd.DataFrame:
    """
    对每个有足够历史的 15min 时刻 t，构建：60 张图路径、4 个过去 PV [t-45,t-30,t-15,t]、4 个目标 [t+15..t+60]。
    使用容差匹配：对每个目标分钟找最近图像，若 |Δ| ≤ tolerance_sec 则接受，避免 round 导致秒级偏移误删。
    """
    if df_img.empty or df_pv.empty:
        return pd.DataFrame(columns=["ts_pred", "img_paths", "past_pv", "targets"])

    df_img = df_img.copy()
    df_img["ts_img"] = df_img["ts_img"].dt.tz_convert(tz)
    df_img = df_img.sort_values("ts_img").reset_index(drop=True)
    ts_arr = df_img["ts_img"].values.astype("datetime64[ns]")
    arr_ns = ts_arr.view("i8")
    path_arr = df_img["img_path"].values

    df_pv = df_pv.copy()
    df_pv["ts_power"] = pd.to_datetime(df_pv["ts_power"])
    if df_pv["ts_power"].dt.tz is None:
        df_pv["ts_power"] = df_pv["ts_power"].dt.tz_localize(tz)
    else:
        df_pv["ts_power"] = df_pv["ts_power"].dt.tz_convert(tz)
    df_pv["ts_power"] = df_pv["ts_power"].dt.round("15min")
    df_pv = df_pv.sort_values("ts_power").drop_duplicates(subset=["ts_power"], keep="last").reset_index(drop=True)

    pv_series = df_pv.set_index("ts_power")["power"].astype(np.float32).sort_index()
    pv_index = pv_series.index
    pv_index_set = set(pv_index)

    samples = []
    filtered_missing_img = 0
    filtered_missing_pv = 0
    delta_secs: list[float] = []

    for t in pv_index:
        t = pd.Timestamp(t)
        if t.tz is None:
            t = t.tz_localize(tz)
        else:
            t = t.tz_convert(tz)
        t = t.round("min")

        required_past = [t - pd.Timedelta(minutes=m) for m in (45, 30, 15, 0)]   # [t-45, t-30, t-15, t]
        required_future = [t + pd.Timedelta(minutes=m) for m in (15, 30, 45, 60)]  # [t+15, t+30, t+45, t+60]
        required_all = required_past + required_future
        if not all(ts in pv_index_set for ts in required_all):
            filtered_missing_pv += 1
            continue

        ts_list = [t - pd.Timedelta(minutes=j) for j in range(img_len - 1, -1, -1)]
        paths = []
        used: set[int] = set()
        sample_deltas: list[float] = []

        for ts in ts_list:
            ts_ns = ts.value
            idx = np.searchsorted(arr_ns, ts_ns)
            candidates = []
            if idx > 0:
                candidates.append(idx - 1)
            if idx < len(arr_ns):
                candidates.append(idx)

            best_path, best_delta, best_idx = None, None, -1
            for c in candidates:
                if c in used:
                    continue
                delta_ns = abs(ts_ns - arr_ns[c])
                delta_sec = delta_ns / 1e9
                if delta_sec <= tolerance_sec and (best_delta is None or delta_sec < best_delta):
                    best_path = str(path_arr[c])
                    best_delta = delta_sec
                    best_idx = c

            if best_path is None:
                filtered_missing_img += 1
                break

            paths.append(best_path)
            used.add(best_idx)
            sample_deltas.append(best_delta)

        if len(paths) != img_len:
            continue

        past_pv = [float(pv_series.loc[ts]) for ts in required_past]
        targets = [float(pv_series.loc[ts]) for ts in required_future]
        delta_secs.extend(sample_deltas)
        samples.append(
            {
                "ts_pred": t,
                "img_paths": paths,
                "past_pv": past_pv,
                "targets": targets,
            }
        )

    out = pd.DataFrame(samples)
    if not out.empty:
        out = out.sort_values("ts_pred").reset_index(drop=True)

    print(
        f"  [match] samples built: {len(out)}, "
        f"filtered (missing pv continuity): {filtered_missing_pv}, "
        f"filtered (missing img): {filtered_missing_img}"
    )
    if delta_secs:
        delta_arr = np.array(delta_secs)
        print(
            f"  [match] delta_sec: min={delta_arr.min():.2f}s, max={delta_arr.max():.2f}s, "
            f"mean={delta_arr.mean():.2f}s, p95={np.percentile(delta_arr, 95):.2f}s"
        )
    if not out.empty:
        hour_dist = out["ts_pred"].dt.hour.value_counts().sort_index()
        print("  [match] sample count by hour:")
        print(hour_dist.to_string())

    return out


# ---------- 预打包配置 ----------
PACK_DIR = Path(__file__).resolve().parent / "derived" / "packed_forecast"
IMG_HEIGHT, IMG_WIDTH = 128, 128
PACK_BATCH_SIZE = 64
SKY_MASK_PATH = Path(__file__).resolve().parent / "sky_mask.png"


def load_image_resize(path: str, size: tuple[int, int], mask_1hw: Optional[np.ndarray] = None) -> np.ndarray:
    """加载、resize、[0,1] 归一化，可选乘 mask。返回 (C,H,W) float32。"""
    with Image.open(path) as im:
        im = im.convert("RGB")
    arr = np.asarray(im.resize((size[1], size[0])), dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    if mask_1hw is not None:
        arr = arr * mask_1hw
    return arr


def pack_forecast_to_npz(
    csv_path: Path,
    out_dir: Path,
    img_size: tuple[int, int] = (IMG_HEIGHT, IMG_WIDTH),
    batch_size: int = PACK_BATCH_SIZE,
    mask_path: Optional[Path] = None,
    base_dir: Optional[Path] = None,
) -> None:
    """
    从 forecast_windows.csv 读取样本，将 60 张图做 resize（+ 可选 mask）+ 归一化后写入 .npz 分片。
    每片: sky (B,60,3,H,W), past_pv (B,4,1), targets (B,4), ts_pred (B,) object。
    """
    import ast
    df = pd.read_csv(csv_path)
    for col in ("img_paths", "past_pv", "targets"):
        if col in df.columns and len(df) and isinstance(df[col].iloc[0], str):
            df[col] = df[col].apply(ast.literal_eval)

    base_dir = Path(base_dir) if base_dir else csv_path.resolve().parent.parent
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mask_1hw = None
    if mask_path and Path(mask_path).exists():
        m = Image.open(mask_path).convert("L")
        m = m.resize((img_size[1], img_size[0]), resample=Image.NEAREST)
        m = (np.asarray(m, dtype=np.float32) / 255.0 >= 0.5).astype(np.float32)
        mask_1hw = m[None, ...]
        print(f"  [sky_mask] Loaded {mask_path} -> resized to ({img_size[0]}, {img_size[1]}), applied to packed images")

    n = len(df)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        B = end - start
        sky = np.zeros((B, 60, 3, img_size[0], img_size[1]), dtype=np.float32)
        past_pv = np.zeros((B, 4, 1), dtype=np.float32)
        targets = np.zeros((B, 4), dtype=np.float32)
        ts_pred = []

        for i in range(B):
            row = df.iloc[start + i]
            for t, p in enumerate(row["img_paths"]):
                path = Path(p)
                if not path.is_absolute():
                    path = base_dir / path
                sky[i, t] = load_image_resize(str(path), img_size, mask_1hw)
            past_pv[i] = np.array(row["past_pv"], dtype=np.float32).reshape(4, 1)
            targets[i] = np.array(row["targets"], dtype=np.float32)
            ts_pred.append(str(row["ts_pred"]))

        fname = out_dir / f"batch_{start:06d}_{end-1:06d}.npz"
        np.savez_compressed(
            fname,
            sky=sky,
            past_pv=past_pv,
            targets=targets,
            ts_pred=np.array(ts_pred, dtype=object),
        )
        print(f"  saved {fname.name} (samples {start}..{end-1})")


def main():
    base = Path(__file__).resolve().parent.parent
    cam_dir = base / "data" / "cam_dir"
    pv_csv = base / "data" / "power" / "power-LSK_N.csv"
    out_dir = Path(__file__).resolve().parent / "derived"
    tz = getattr(BASE_CFG, "TZ", "Asia/Singapore") if BASE_CFG else "Asia/Singapore"

    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1] Build image index (raw timestamps for tolerance matching)...")
    df_img = build_image_index_raw(cam_dir, tz)
    if df_img.empty:
        raise RuntimeError(f"No images in {cam_dir}")
    print(f"  images: {len(df_img)}")

    print("[2] Load PV 15min...")
    df_pv = load_pv_15min(pv_csv, tz)
    df_pv = df_pv.dropna(subset=["power"]).reset_index(drop=True)

    if len(df_pv) < PAST_PV_LEN + HORIZON:
        raise RuntimeError("Not enough PV points")
    print(f"  pv rows: {len(df_pv)}")
    print_time_coverage_debug(df_img, df_pv, tz)

    print("[3] Build forecast windows (60 img + 4 past PV + 4 targets)...")
    df = build_forecast_windows(df_img, df_pv, IMG_LEN, PAST_PV_LEN, HORIZON, tz)
    print(f"  samples: {len(df)}")

    out_csv = out_dir / "forecast_windows.csv"
    df.to_csv(out_csv, index=False)
    print(f"  saved: {out_csv}")
    try:
        out_parquet = out_dir / "forecast_windows.parquet"
        df.to_parquet(out_parquet, index=False)
        print(f"  saved: {out_parquet}")
    except Exception as e:
        print(f"  skip parquet: {e}")

    # 打包需单独加 --pack，耗时较长，建议先检查索引 CSV 后再执行
    if "--pack" in sys.argv:
        pack_dir = out_dir / "packed_forecast"
        print("[4] Pack to .npz (resize + optional mask + norm)...")
        pack_forecast_to_npz(
            csv_path=out_csv,
            out_dir=pack_dir,
            img_size=(IMG_HEIGHT, IMG_WIDTH),
            batch_size=PACK_BATCH_SIZE,
            mask_path=SKY_MASK_PATH if SKY_MASK_PATH.exists() else None,
            base_dir=base,
        )
        print(f"  packed dir: {pack_dir}")
    else:
        print("[4] Skip packing (add --pack to run). Inspect forecast_windows.csv first.")


if __name__ == "__main__":
    main()
