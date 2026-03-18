from __future__ import annotations

import json
import re
from dataclasses import dataclass
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


@dataclass(frozen=True)
class SatellitePatchExtractor:
    channels: tuple[int, ...]
    center_lat: float
    center_lon: float
    patch_size: int

    def _rowcol_in_cropped(self, f: h5py.File) -> tuple[int, int]:
        info = load_roi_info(f)
        geo = _read_geo_params(f)
        orig_h, orig_w = info["original_shape"]
        r_full, c_full = lonlat_to_rowcol(self.center_lon, self.center_lat, orig_h, orig_w, geo)
        return int(r_full - info["r0"]), int(c_full - info["c0"])

    def _calibrated_channel(self, f: h5py.File, ch: int) -> np.ndarray:
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

        # For VIS/NIR channels, CALChannel lookup already yields reflectance-like values.
        # We keep the LUT output directly as model input instead of converting it to radiance.
        return out

    def read_patch(self, file_path: str | Path) -> np.ndarray:
        with h5py.File(file_path, "r") as f:
            row_c, col_c = self._rowcol_in_cropped(f)
            half = self.patch_size // 2
            data_shape = f[f"Data/NOMChannel{self.channels[0]:02d}"].shape
            h, w = data_shape
            r0 = max(0, row_c - half)
            c0 = max(0, col_c - half)
            r1 = min(h, r0 + self.patch_size)
            c1 = min(w, c0 + self.patch_size)
            r0 = max(0, r1 - self.patch_size)
            c0 = max(0, c1 - self.patch_size)
            if (r1 - r0) != self.patch_size or (c1 - c0) != self.patch_size:
                raise ValueError(f"Cannot extract {self.patch_size}x{self.patch_size} patch from {file_path}")

            patch = []
            for ch in self.channels:
                calibrated = self._calibrated_channel(f, ch)
                cut = calibrated[r0:r1, c0:c1]
                cut = np.nan_to_num(cut, nan=0.0, posinf=0.0, neginf=0.0)
                patch.append(cut.astype(np.float32, copy=False))
        return np.stack(patch, axis=0)


def compute_channel_stats_from_array(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if x.ndim != 5:
        raise ValueError(f"Expected [N,T,C,H,W], got {x.shape}")
    flat = np.transpose(x, (2, 0, 1, 3, 4)).reshape(x.shape[2], -1)
    mean = flat.mean(axis=1).astype(np.float32)
    std = flat.std(axis=1).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std)
    return mean, std


def save_stats(path: Path, mean: np.ndarray, std: np.ndarray, channels: list[int]) -> None:
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
