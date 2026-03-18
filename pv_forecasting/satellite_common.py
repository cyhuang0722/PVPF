from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd


ROI_TS_PATTERN = re.compile(r"_(\d{14})_(\d{14})_\d{4}M_")


def load_json_config(path: str | Path) -> dict[str, Any]:
    cfg_path = Path(path)
    return json.loads(cfg_path.read_text(encoding="utf-8"))


def resolve_path(path_str: str | None, project_root: Path) -> Path | None:
    if not path_str:
        return None
    path = Path(path_str)
    if path.is_absolute():
        return path
    return project_root / path


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_roi_timerange(path: str | Path, timezone: str) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    m = ROI_TS_PATTERN.search(Path(path).name)
    if m is None:
        raise ValueError(f"Cannot parse ROI timestamp from {path}")
    ts_start = pd.to_datetime(m.group(1), format="%Y%m%d%H%M%S").tz_localize(timezone)
    ts_end = pd.to_datetime(m.group(2), format="%Y%m%d%H%M%S").tz_localize(timezone)
    ts_available = (ts_end + pd.Timedelta(seconds=1)).floor("min")
    return ts_start, ts_end, ts_available


def build_satellite_index(roi_dir: Path, timezone: str) -> pd.DataFrame:
    rows = []
    for path in sorted(roi_dir.glob("*.HDF")):
        ts_start, ts_end, ts_available = parse_roi_timerange(path, timezone)
        rows.append(
            {
                "ts_sat_start": ts_start,
                "ts_sat_end": ts_end,
                "ts_sat": ts_available,
                "sat_path": str(path.resolve()),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        raise FileNotFoundError(f"No ROI HDF found in {roi_dir}")
    return out.sort_values("ts_sat").reset_index(drop=True)


def load_power_series(pv_csv: Path, timezone: str, value_col: str = "value") -> pd.Series:
    df = pd.read_csv(pv_csv)
    if "date" not in df.columns:
        raise ValueError(f"Column 'date' not found in {pv_csv}")
    if value_col not in df.columns:
        raise ValueError(f"Column '{value_col}' not found in {pv_csv}")
    df["date"] = pd.to_datetime(df["date"])
    ts = df["date"]
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize(timezone)
    else:
        ts = ts.dt.tz_convert(timezone)
    series = pd.Series(df[value_col].astype(np.float32).to_numpy(), index=ts, name="power")
    series = series.sort_index()
    series = series[series.notna()]
    return series


def chronological_split(
    n: int,
    train_ratio: float,
    val_ratio: float,
) -> np.ndarray:
    if n <= 0:
        return np.empty(0, dtype=object)
    if not (0.0 < train_ratio < 1.0) or not (0.0 <= val_ratio < 1.0):
        raise ValueError("train_ratio and val_ratio must be within (0,1)")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1")
    n_train = max(1, int(n * train_ratio))
    n_val = max(1, int(n * val_ratio))
    if n_train + n_val >= n:
        n_val = max(1, n - n_train - 1)
    n_test = max(0, n - n_train - n_val)
    split = np.array(["train"] * n, dtype=object)
    split[n_train:n_train + n_val] = "val"
    if n_test > 0:
        split[n_train + n_val:] = "test"
    return split


def _attr_to_py(v, default=None):
    if v is None:
        return default
    if isinstance(v, (bytes, bytearray)):
        return v.decode("utf-8", "ignore")
    arr = np.array(v)
    if arr.ndim == 0:
        return arr.item()
    return arr.tolist()


def _attr_to_scalar(v, default=None):
    if v is None:
        return default
    if isinstance(v, (bytes, bytearray)):
        return v.decode("utf-8", "ignore")
    arr = np.array(v)
    if arr.ndim == 0:
        return arr.item()
    return arr.reshape(-1)[0].item()


def _read_geo_params(f: h5py.File) -> dict[str, float]:
    a = _attr_to_scalar(f.attrs.get("Semimajor axis of ellipsoid"), 6378137.0)
    b = _attr_to_scalar(f.attrs.get("Semiminor axis of ellipsoid"), 6356752.0)
    lon0_deg = _attr_to_scalar(f.attrs.get("NOMCenterLon"), None)
    if lon0_deg is None:
        lon0_deg = _attr_to_scalar(f.attrs.get("RegCenterLon"), None)
    if lon0_deg is None:
        raise KeyError("Cannot find NOMCenterLon/RegCenterLon in file attrs.")
    sat_h = _attr_to_scalar(f.attrs.get("NOMSatHeight"), None)
    if sat_h is None:
        raise KeyError("Cannot find NOMSatHeight in file attrs.")
    dx_urad = _attr_to_scalar(f.attrs.get("dSamplingAngle"), None)
    dy_urad = _attr_to_scalar(f.attrs.get("dSteppingAngle"), None)
    if dx_urad is None or dy_urad is None:
        raise KeyError("Cannot find dSamplingAngle/dSteppingAngle in file attrs.")
    r_sat = float(a) + float(sat_h)
    return {
        "a": float(a),
        "b": float(b),
        "lon0_deg": float(lon0_deg),
        "r_sat": r_sat,
        "dx": float(dx_urad) * 1e-6,
        "dy": float(dy_urad) * 1e-6,
    }


def lonlat_to_rowcol(lon_deg: float, lat_deg: float, H: int, W: int, geo: dict[str, float]) -> tuple[int, int]:
    a, b = geo["a"], geo["b"]
    lon0 = np.deg2rad(geo["lon0_deg"])
    r_sat = geo["r_sat"]
    dx, dy = geo["dx"], geo["dy"]

    lon = np.deg2rad(lon_deg)
    lat = np.deg2rad(lat_deg)

    e2 = (a * a - b * b) / (a * a)
    N = a / np.sqrt(1.0 - e2 * np.sin(lat) ** 2)
    lam = lon - lon0
    x = N * np.cos(lat) * np.cos(lam)
    y = N * np.cos(lat) * np.sin(lam)
    z = (b * b / (a * a)) * N * np.sin(lat)

    alpha = np.arctan2(y, (r_sat - x))
    beta = np.arctan2(z, np.sqrt((r_sat - x) ** 2 + y * y))
    col0 = (W - 1) / 2.0
    row0 = (H - 1) / 2.0

    col = col0 + alpha / dx
    row = row0 - beta / dy
    return int(round(row)), int(round(col))


def load_roi_info(f: h5py.File) -> dict[str, Any]:
    if "ROIInfo" not in f:
        raise KeyError("ROIInfo not found in cropped ROI HDF")
    return {k: _attr_to_py(v) for k, v in f["ROIInfo"].attrs.items()}


def _calibrated_channel_from_file(f: h5py.File, ch: int) -> np.ndarray:
    data_path = f"Data/NOMChannel{ch:02d}"
    cal_path = f"Calibration/CALChannel{ch:02d}"
    if data_path not in f:
        raise KeyError(f"Dataset not found: {data_path}")

    ds = f[data_path]
    dn = ds[...].astype(np.int32, copy=False)
    fill = _attr_to_py(ds.attrs.get("FillValue"), 65535)
    if cal_path in f:
        lut = f[cal_path][...].astype(np.float32, copy=False)
        out = np.full(dn.shape, np.nan, dtype=np.float32)
        valid = (dn != fill) & (dn >= 0) & (dn < lut.shape[0])
        out[valid] = lut[dn[valid]]
    else:
        out = dn.astype(np.float32, copy=False)
        out[out == fill] = np.nan

    # VIS/NIR channels are used as calibrated reflectance-like inputs.
    return out


@lru_cache(maxsize=256)
def load_calibrated_channel(file_path: str, ch: int) -> np.ndarray:
    with h5py.File(file_path, "r") as f:
        return _calibrated_channel_from_file(f, ch)


@dataclass
class SatellitePatchExtractor:
    channels: tuple[int, ...]
    center_lat: float
    center_lon: float
    patch_size: int
    _bounds_cache: dict[str, tuple[int, int, int, int]] = field(default_factory=dict, init=False, repr=False)

    def _rowcol_in_cropped(self, f: h5py.File) -> tuple[int, int]:
        info = load_roi_info(f)
        geo = _read_geo_params(f)
        orig_h, orig_w = info["original_shape"]
        r_full, c_full = lonlat_to_rowcol(self.center_lon, self.center_lat, orig_h, orig_w, geo)
        return int(r_full - info["r0"]), int(c_full - info["c0"])

    def patch_bounds(self, file_path: str | Path) -> tuple[int, int, int, int]:
        path = str(file_path)
        if path in self._bounds_cache:
            return self._bounds_cache[path]

        with h5py.File(path, "r") as f:
            row_c, col_c = self._rowcol_in_cropped(f)
            data_shape = f[f"Data/NOMChannel{self.channels[0]:02d}"].shape
            h, w = data_shape
            half = self.patch_size // 2
            r0 = max(0, row_c - half)
            c0 = max(0, col_c - half)
            r1 = min(h, r0 + self.patch_size)
            c1 = min(w, c0 + self.patch_size)
            r0 = max(0, r1 - self.patch_size)
            c0 = max(0, c1 - self.patch_size)
            if (r1 - r0) != self.patch_size or (c1 - c0) != self.patch_size:
                raise ValueError(f"Cannot extract {self.patch_size}x{self.patch_size} patch from {file_path}")

        bounds = (r0, r1, c0, c1)
        self._bounds_cache[path] = bounds
        return bounds

    def crop_array(self, file_path: str | Path, array: np.ndarray) -> np.ndarray:
        r0, r1, c0, c1 = self.patch_bounds(file_path)
        cut = array[r0:r1, c0:c1]
        cut = np.nan_to_num(cut, nan=0.0, posinf=0.0, neginf=0.0)
        return cut.astype(np.float32, copy=False)

    def read_patch(self, file_path: str | Path) -> np.ndarray:
        patch = []
        for ch in self.channels:
            calibrated = load_calibrated_channel(str(file_path), ch)
            patch.append(self.crop_array(file_path, calibrated))
        return np.stack(patch, axis=0)


class CloudIndexMapProvider:
    def __init__(
        self,
        sat_df: pd.DataFrame,
        source_channel: int,
        lookback_days: int,
        extractor: SatellitePatchExtractor,
    ) -> None:
        self.source_channel = int(source_channel)
        self.lookback_days = int(lookback_days)
        self.extractor = extractor
        self.history_map = self._build_history_map(sat_df)
        self._ci_patch_cache: dict[str, np.ndarray] = {}

    def _build_history_map(self, sat_df: pd.DataFrame) -> dict[str, list[str]]:
        df = sat_df.copy().sort_values("ts_sat").reset_index(drop=True)
        df["time_slot"] = df["ts_sat"].dt.strftime("%H:%M")
        histories: dict[str, list[tuple[pd.Timestamp, str]]] = {}
        out: dict[str, list[str]] = {}

        for row in df.itertuples(index=False):
            slot_history = histories.setdefault(row.time_slot, [])
            cutoff = row.ts_sat - pd.Timedelta(days=self.lookback_days)
            slot_history[:] = [(ts, path) for ts, path in slot_history if ts >= cutoff]
            out[str(row.sat_path)] = [path for _, path in slot_history] + [str(row.sat_path)]
            slot_history.append((row.ts_sat, str(row.sat_path)))
        return out

    def get_patch(self, file_path: str | Path) -> np.ndarray:
        path = str(file_path)
        cached = self._ci_patch_cache.get(path)
        if cached is not None:
            return cached

        current = load_calibrated_channel(path, self.source_channel)
        history_paths = self.history_map.get(path, [path])
        past_patches = [self.extractor.crop_array(hist_path, load_calibrated_channel(hist_path, self.source_channel)) for hist_path in history_paths]
        pmin = np.min(np.stack(past_patches, axis=0), axis=0)
        current_patch = self.extractor.crop_array(path, current)
        pmax = float(np.nanmax(current))
        denom = np.maximum(pmax - pmin, 1e-6)
        ci = np.clip((current_patch - pmin) / denom, 0.0, 1.0).astype(np.float32, copy=False)
        self._ci_patch_cache[path] = ci
        return ci


def compute_channel_stats_from_array(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if x.ndim != 5:
        raise ValueError(f"Expected [N,T,C,H,W], got {x.shape}")
    flat = np.transpose(x, (2, 0, 1, 3, 4)).reshape(x.shape[2], -1)
    mean = flat.mean(axis=1).astype(np.float32)
    std = flat.std(axis=1).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std)
    return mean, std


def save_stats(path: Path, mean: np.ndarray, std: np.ndarray, channels: list[Any]) -> None:
    dump_json(
        path,
        {
            "channels": channels,
            "mean": [float(v) for v in mean],
            "std": [float(v) for v in std],
        },
    )


def load_stats(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mean = np.asarray(payload["mean"], dtype=np.float32)
    std = np.asarray(payload["std"], dtype=np.float32)
    return mean, std
