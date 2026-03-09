"""
测试数据预处理：从 cam_test 构建样本并打包，与 preprocess 相同逻辑（60 图 + 4 past PV + 4 targets）。
需要：cam_test 图像 + 覆盖测试时段的 PV（含 past 与 future，用于构建完整样本）。

用法: python -m pv_forecasting.preprocess_test [--cam-dir data/cam_test] [--pv-csv ...] [--out-dir derived/test] [--pack]
"""
import argparse
from pathlib import Path

from .preprocess import (
    BASE_CFG,
    IMG_HEIGHT,
    IMG_WIDTH,
    HORIZON,
    IMG_LEN,
    PAST_PV_LEN,
    PACK_BATCH_SIZE,
    SKY_MASK_PATH,
    build_forecast_windows,
    build_image_index_raw,
    load_pv_15min,
    pack_forecast_to_npz,
    print_time_coverage_debug,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam-dir", default="data/cam_test", help="测试图像目录")
    parser.add_argument("--pv-csv", default="data/power/power-LSK_N-test.csv", help="PV 15min 数据（需覆盖测试时段）")
    parser.add_argument("--pack", action="store_true", help="打包为 .npz（否则只生成 CSV）")
    parser.add_argument("--out-dir", default=None, help="输出目录，默认 derived/test")
    args = parser.parse_args()

    base = Path(__file__).resolve().parent.parent
    cam_dir = base / args.cam_dir
    pv_csv = base / args.pv_csv
    out_dir = Path(__file__).resolve().parent / (args.out_dir or "derived/test")
    tz = getattr(BASE_CFG, "TZ", "Asia/Singapore") if BASE_CFG else "Asia/Singapore"

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1] Build image index from {cam_dir}...")
    df_img = build_image_index_raw(cam_dir, tz)
    if df_img.empty:
        raise RuntimeError(f"No images in {cam_dir}")
    print(f"  images: {len(df_img)}")

    print(f"[2] Load PV from {pv_csv}...")
    df_pv = load_pv_15min(pv_csv, tz)
    df_pv = df_pv.dropna(subset=["power"]).reset_index(drop=True)
    if len(df_pv) < PAST_PV_LEN + HORIZON:
        raise RuntimeError(f"Not enough PV points (need at least {PAST_PV_LEN + HORIZON} for past + horizon)")
    print(f"  pv rows: {len(df_pv)}")
    print_time_coverage_debug(df_img, df_pv, tz)

    print("[3] Build forecast windows (60 img + 4 past PV + 4 targets)...")
    df = build_forecast_windows(df_img, df_pv, IMG_LEN, PAST_PV_LEN, HORIZON, tz)
    if df.empty:
        img_range = f"{df_img['ts_img'].min()} ~ {df_img['ts_img'].max()}" if not df_img.empty else "?"
        pv_range = f"{df_pv['ts_power'].min()} ~ {df_pv['ts_power'].max()}" if not df_pv.empty else "?"
        raise RuntimeError(
            f"No test samples built. 图像与 PV 时间需重叠，且每个锚点需有完整 past(4) + future(5) 的 PV。\n"
            f"  图像: {img_range}\n   PV: {pv_range}\n"
            f"  请用 --pv-csv 指定覆盖测试时段的 PV 文件。"
        )
    print(f"  samples: {len(df)}")

    out_csv = out_dir / "test_forecast_windows.csv"
    df.to_csv(out_csv, index=False)
    print(f"  saved: {out_csv}")

    if args.pack:
        pack_dir = out_dir / "packed"
        print("[4] Pack to .npz...")
        pack_forecast_to_npz(
            csv_path=out_csv,
            out_dir=pack_dir,
            img_size=(IMG_HEIGHT, IMG_WIDTH),
            batch_size=PACK_BATCH_SIZE,
            mask_path=SKY_MASK_PATH if SKY_MASK_PATH.exists() else None,
            base_dir=base,
        )
        print(f"  packed: {pack_dir}")
    else:
        print("[4] Skip packing (add --pack to run)")


if __name__ == "__main__":
    main()
