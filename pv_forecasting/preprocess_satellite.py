from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .satellite_common import (
    CloudIndexMapProvider,
    SatellitePatchExtractor,
    build_satellite_index,
    chronological_split,
    compute_channel_stats_from_array,
    dump_json,
    ensure_dir,
    load_json_config,
    load_power_series,
    save_stats,
)


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent


def _match_satellite_paths(
    sat_ns: np.ndarray,
    sat_paths: np.ndarray,
    requested_ts: list[pd.Timestamp],
    tolerance_sec: float,
) -> list[str] | None:
    matched_paths = []
    used = set()
    for ts in requested_ts:
        ts_ns = ts.value
        idx = np.searchsorted(sat_ns, ts_ns)
        candidates = []
        if idx > 0:
            candidates.append(idx - 1)
        if idx < len(sat_ns):
            candidates.append(idx)

        best_idx = None
        best_delta = None
        for cand in candidates:
            if cand in used:
                continue
            delta = abs(ts_ns - sat_ns[cand]) / 1e9
            if delta <= tolerance_sec and (best_delta is None or delta < best_delta):
                best_idx = cand
                best_delta = delta
        if best_idx is None:
            return None
        used.add(best_idx)
        matched_paths.append(str(sat_paths[best_idx]))
    return matched_paths


def build_windows(
    sat_df: pd.DataFrame,
    pv_series: pd.Series,
    t_in: int,
    sat_stride_min: int,
    future_offsets_min: list[int],
    tolerance_sec: float,
) -> pd.DataFrame:
    sat_df = sat_df.copy().sort_values("ts_sat").reset_index(drop=True)
    sat_ns = sat_df["ts_sat"].values.astype("datetime64[ns]").view("i8")
    sat_paths = sat_df["sat_path"].to_numpy()
    pv_index = pv_series.index.sort_values()
    pv_set = set(pv_index)

    rows = []
    for t in pv_index:
        required_future = [t + pd.Timedelta(minutes=m) for m in future_offsets_min]
        if any(ts not in pv_set for ts in required_future):
            continue

        input_sat_ts = [
            t - pd.Timedelta(minutes=sat_stride_min * step)
            for step in range(t_in - 1, -1, -1)
        ]
        input_sat_paths = _match_satellite_paths(sat_ns, sat_paths, input_sat_ts, tolerance_sec)
        future_sat_paths = _match_satellite_paths(sat_ns, sat_paths, required_future, tolerance_sec)
        if input_sat_paths is None or future_sat_paths is None:
            continue
        rows.append(
            {
                "ts_pred": t,
                "sat_paths": input_sat_paths,
                "future_sat_paths": future_sat_paths,
                "targets": [float(pv_series.loc[ts]) for ts in required_future],
            }
        )
    return pd.DataFrame(rows).sort_values("ts_pred").reset_index(drop=True)


def build_input_sequence(
    sat_paths: list[str],
    extractor: SatellitePatchExtractor,
    ci_provider: CloudIndexMapProvider | None,
) -> np.ndarray:
    frames = []
    for path in sat_paths:
        raw = extractor.read_patch(path)
        if ci_provider is not None:
            raw = np.concatenate([raw, ci_provider.get_patch(path)[None, :, :]], axis=0)
        frames.append(raw.astype(np.float32, copy=False))
    return np.stack(frames, axis=0)


def build_cloud_index_targets(
    future_sat_paths: list[str],
    ci_provider: CloudIndexMapProvider,
) -> np.ndarray:
    return np.stack([ci_provider.get_patch(path) for path in future_sat_paths], axis=0).astype(np.float32, copy=False)


def pack_dataset(
    df: pd.DataFrame,
    extractor: SatellitePatchExtractor,
    ci_provider: CloudIndexMapProvider | None,
    out_dir: Path,
    batch_size: int,
    peak_power_w: float,
    mean: np.ndarray,
    std: np.ndarray,
) -> None:
    pack_dir = ensure_dir(out_dir / "packed_satellite")
    all_sat = []
    all_ci = []
    for _, row in df.iterrows():
        seq = build_input_sequence(row["sat_paths"], extractor, ci_provider)
        seq = (seq - mean[None, :, None, None]) / std[None, :, None, None]
        all_sat.append(seq.astype(np.float32, copy=False))
        if ci_provider is not None:
            all_ci.append(build_cloud_index_targets(row["future_sat_paths"], ci_provider))

    all_sat = np.stack(all_sat, axis=0)
    all_targets = np.stack(df["targets"].apply(lambda x: np.asarray(x, dtype=np.float32)).to_list(), axis=0)
    all_targets = all_targets / np.float32(peak_power_w)
    all_ci_targets = np.stack(all_ci, axis=0) if all_ci else None
    ts_pred = df["ts_pred"].astype(str).to_numpy()
    split = df["split"].astype(str).to_numpy()

    for start in range(0, len(df), batch_size):
        end = min(start + batch_size, len(df))
        payload = {
            "satellite": all_sat[start:end],
            "targets": all_targets[start:end],
            "ts_pred": ts_pred[start:end],
            "split": split[start:end],
        }
        if all_ci_targets is not None:
            payload["cloud_index_targets"] = all_ci_targets[start:end]
        np.savez_compressed(pack_dir / f"batch_{start:06d}_{end-1:06d}.npz", **payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="JSON config path")
    parser.add_argument("--pack", action="store_true", help="Read ROI patches and pack normalized tensors into npz shards")
    args = parser.parse_args()

    cfg = load_json_config(args.config)
    data_cfg = cfg["data"]
    split_cfg = cfg["split"]
    roi_dir = PROJECT_ROOT / data_cfg["roi_dir"]
    pv_csv = PROJECT_ROOT / data_cfg["pv_csv"]
    out_dir = ensure_dir(PROJECT_ROOT / data_cfg["out_dir"])
    stats_path = out_dir / data_cfg.get("stats_filename", "satellite_stats.json")

    sat_df = build_satellite_index(roi_dir, data_cfg["timezone"])
    sat_df.to_csv(out_dir / "satellite_index.csv", index=False)
    pv_series = load_power_series(pv_csv, data_cfg["timezone"], data_cfg.get("pv_value_col", "value"))
    windows = build_windows(
        sat_df=sat_df,
        pv_series=pv_series,
        t_in=int(data_cfg["t_in"]),
        sat_stride_min=int(data_cfg.get("sat_stride_min", 15)),
        future_offsets_min=list(data_cfg["future_offsets_min"]),
        tolerance_sec=float(data_cfg.get("match_tolerance_sec", 180.0)),
    )
    if windows.empty:
        raise ValueError("No satellite forecasting windows were built")

    windows["split"] = chronological_split(
        len(windows),
        train_ratio=float(split_cfg["train_ratio"]),
        val_ratio=float(split_cfg["val_ratio"]),
    )
    windows["channels"] = [list(data_cfg["channels"])] * len(windows)
    windows["center_lat"] = float(data_cfg["center_lat"])
    windows["center_lon"] = float(data_cfg["center_lon"])
    windows["patch_size"] = int(data_cfg["patch_size"])
    windows.to_csv(out_dir / "forecast_windows.csv", index=False)

    summary = {
        "num_samples": int(len(windows)),
        "split_counts": windows["split"].value_counts().to_dict(),
        "channels": list(data_cfg["channels"]),
        "future_offsets_min": list(data_cfg["future_offsets_min"]),
        "t_in": int(data_cfg["t_in"]),
        "patch_size": int(data_cfg["patch_size"]),
        "peak_power_w": float(data_cfg["peak_power_w"]),
    }

    extractor = SatellitePatchExtractor(
        channels=tuple(data_cfg["channels"]),
        center_lat=float(data_cfg["center_lat"]),
        center_lon=float(data_cfg["center_lon"]),
        patch_size=int(data_cfg["patch_size"]),
    )
    ci_provider = None
    if bool(data_cfg.get("include_cloud_index_map", False)):
        ci_provider = CloudIndexMapProvider(
            sat_df=sat_df,
            source_channel=int(data_cfg.get("cloud_index_source_channel", data_cfg["channels"][0])),
            lookback_days=int(data_cfg.get("cloud_index_lookback_days", 10)),
            extractor=extractor,
        )
    train_df = windows[windows["split"] == "train"].reset_index(drop=True)
    train_sat = np.stack(
        [build_input_sequence(row["sat_paths"], extractor, ci_provider) for _, row in train_df.iterrows()],
        axis=0,
    )
    mean, std = compute_channel_stats_from_array(train_sat)
    input_channel_labels: list[int | str] = list(extractor.channels)
    if ci_provider is not None:
        input_channel_labels.append("cloud_index_map")
    save_stats(stats_path, mean, std, input_channel_labels)
    summary["stats_path"] = str(stats_path)
    summary["num_input_channels"] = int(train_sat.shape[2])
    summary["include_cloud_index_map"] = bool(ci_provider is not None)
    if ci_provider is not None:
        summary["cloud_index_source_channel"] = int(data_cfg.get("cloud_index_source_channel", data_cfg["channels"][0]))
        summary["cloud_index_lookback_days"] = int(data_cfg.get("cloud_index_lookback_days", 10))

    if args.pack:
        pack_dataset(
            df=windows,
            extractor=extractor,
            ci_provider=ci_provider,
            out_dir=out_dir,
            batch_size=int(data_cfg.get("pack_batch_size", 64)),
            peak_power_w=float(data_cfg["peak_power_w"]),
            mean=mean,
            std=std,
        )
        summary["pack_dir"] = str(out_dir / "packed_satellite")

    dump_json(out_dir / "preprocess_summary.json", summary)
    print(f"Saved windows to {out_dir / 'forecast_windows.csv'}")
    print(f"Saved channel stats to {stats_path}")
    if args.pack:
        print(f"Saved packed dataset to {out_dir / 'packed_satellite'}")


if __name__ == "__main__":
    main()
