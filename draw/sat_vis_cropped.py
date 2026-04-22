"""
Helpers to read/plot cropped FY-4 AGRI L1 files produced by process-sat/crop_roi.py.

Key differences from sat_vis.py:
- Uses ROIInfo (written by crop_roi.py) to recover the original grid offsets.
- Supports mapping lon/lat -> (row, col) within the cropped array.
- Keeps calibration logic identical to sat_vis.py (LUT + ESUN for VIS ch<=6).
"""

from math import pi
from pathlib import Path
from typing import Dict, Tuple

import h5py
import numpy as np
import sys

# Make project root importable so we can reuse sat_vis helpers.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sat_vis import _read_geo_params, lonlat_to_rowcol, rowcol_to_lonlat


def _attr_to_py(v, default=None):
    """
    Convert HDF5 attribute to plain python.
    - bytes -> str
    - scalars -> python scalar
    - arrays -> python list
    """
    if v is None:
        return default
    if isinstance(v, (bytes, bytearray)):
        return v.decode("utf-8", "ignore")
    arr = np.array(v)
    if arr.ndim == 0:
        return arr.item()
    return arr.tolist()


def load_roi_info(f: h5py.File) -> Dict:
    if "ROIInfo" not in f:
        raise KeyError("ROIInfo not found; make sure the file is produced by crop_roi.py")
    g = f["ROIInfo"]
    out = {k: _attr_to_py(v) for k, v in g.attrs.items()}
    return out


def gsd_km_at_subsatellite(geo: Dict[str, float]) -> float:
    gsd_km = (geo["r_sat"] * (geo["dx"] + geo["dy"]) * 0.5) / 1000.0
    return max(gsd_km, 1e-6)


def lonlat_to_rowcol_cropped(lon_deg, lat_deg, f: h5py.File):
    """
    Map lon/lat to (row, col) in the cropped array by using original grid offsets.
    """
    info = load_roi_info(f)
    geo = _read_geo_params(f)
    orig_h, orig_w = info["original_shape"]
    r_full, c_full = lonlat_to_rowcol(lon_deg, lat_deg, orig_h, orig_w, geo)
    r = r_full - info["r0"]
    c = c_full - info["c0"]
    return int(r), int(c)


def extent_from_roiinfo(f: h5py.File) -> Tuple[float, float, float, float]:
    info = load_roi_info(f)
    extent_ll = info["extent_ll"]
    # extent_ll is stored as tuple (lon_min, lon_max, lat_min, lat_max)
    return tuple(extent_ll)


def calibrate_channel(f: h5py.File, ch: int) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
    info = load_roi_info(f)
    data_path = f"Data/NOMChannel{ch:02d}"
    if data_path not in f:
        raise KeyError(f"Dataset not found: {data_path}")

    ds = f[data_path]
    dn = ds[...].astype(np.int32, copy=False)
    fill = _attr_to_py(ds.attrs.get("FillValue"), 65535)

    cal_path = f"Calibration/CALChannel{ch:02d}"
    if cal_path in f:
        lut = f[cal_path][...].astype(np.float32, copy=False)
        out = np.full(dn.shape, np.nan, dtype=np.float32)
        m = (dn != fill) & (dn >= 0) & (dn < lut.shape[0])
        out[m] = lut[dn[m]]
    else:
        out = dn.astype(np.float32, copy=False)
        out[out == fill] = np.nan

    if ch <= 6 and "Calibration/ESUN" in f:
        esun = f["Calibration/ESUN"]
        out = out * esun[ch - 1] / pi

    extent_ll = extent_from_roiinfo(f)
    return out, extent_ll


def read_pixel_at_lonlat(file_path: Path, lon_deg: float, lat_deg: float, ch: int = 2):
    with h5py.File(file_path, "r") as f:
        roi = load_roi_info(f)
        geo = _read_geo_params(f)
        r, c = lonlat_to_rowcol_cropped(lon_deg, lat_deg, f)
        data = f[f"Data/NOMChannel{ch:02d}"]
        if r < 0 or c < 0 or r >= data.shape[0] or c >= data.shape[1]:
            raise ValueError("Requested lon/lat is outside the cropped ROI.")
        dn_val = data[r, c]
        cal, _ = calibrate_channel(f, ch)
        rad_val = cal[r, c]
        lon_cell, lat_cell = rowcol_to_lonlat(roi["r0"] + r, roi["c0"] + c, roi["original_shape"][0], roi["original_shape"][1], geo)
        return {"dn": dn_val, "radiance": rad_val, "lon": lon_cell, "lat": lat_cell}


def plot_cropped(
    file_path: Path,
    ch: int = 2,
    save_path: Path = Path("cropped_roi.png"),
):
    """
    Quick viewer for a cropped file; no external basemap dependency.
    """
    import matplotlib.pyplot as plt

    with h5py.File(file_path, "r") as f:
        data, extent_ll = calibrate_channel(f, ch)

    fig, ax = plt.subplots(figsize=(7, 6), dpi=200)
    im = ax.imshow(
        data,
        origin="upper",
        extent=extent_ll,
        cmap="inferno",
    )
    cbar = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label(f"Radiance (Ch{ch:02d})")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.tight_layout()
    plt.savefig(save_path, dpi=240, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    # Example: view a cropped file under down_roi
    example = Path(__file__).resolve().parent.parent / "sat-roi"
    files = sorted(example.glob("*.HDF"))
    if files:
        plot_cropped(files[0], ch=2, save_path=Path("cropped_preview.png"))
    else:
        print("No cropped files found under sat-roi/")

