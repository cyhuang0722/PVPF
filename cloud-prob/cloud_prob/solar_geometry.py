from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pvlib import solarposition


@dataclass(frozen=True)
class Calibration:
    cx: float
    cy: float
    f_px_per_rad: float
    lat: float
    lon: float
    timezone: str
    reference_width: int | None = None
    reference_height: int | None = None

    @staticmethod
    def from_json(path: str | Path) -> "Calibration":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        intr = data["intrinsics"]
        site = data["site"]
        return Calibration(
            cx=float(intr["cx"]),
            cy=float(intr["cy"]),
            f_px_per_rad=float(intr["f_px_per_rad"]),
            lat=float(site["lat"]),
            lon=float(site["lon"]),
            timezone=str(site["timezone"]),
            reference_width=int(data.get("image_width")) if data.get("image_width") else None,
            reference_height=int(data.get("image_height")) if data.get("image_height") else None,
        )

    def inferred_reference_size(self) -> tuple[int, int]:
        if self.reference_width and self.reference_height:
            return self.reference_width, self.reference_height
        return int(round(self.cx * 2.0)), int(round(self.cy * 2.0))

    def rescale(self, dst_w: int, dst_h: int) -> "Calibration":
        ref_w, ref_h = self.inferred_reference_size()
        sx = dst_w / float(ref_w)
        sy = dst_h / float(ref_h)
        scale = 0.5 * (sx + sy)
        return Calibration(
            cx=self.cx * sx,
            cy=self.cy * sy,
            f_px_per_rad=self.f_px_per_rad * scale,
            lat=self.lat,
            lon=self.lon,
            timezone=self.timezone,
            reference_width=dst_w,
            reference_height=dst_h,
        )


def compute_solar_position(
    timestamps: pd.Series | list[pd.Timestamp],
    calib: Calibration,
    altitude_m: float = 0.0,
) -> pd.DataFrame:
    ts = pd.DatetimeIndex(pd.to_datetime(list(timestamps)))
    if ts.tz is None:
        ts = ts.tz_localize(calib.timezone)
    else:
        ts = ts.tz_convert(calib.timezone)
    solpos = solarposition.get_solarposition(ts, calib.lat, calib.lon, altitude_m)
    return pd.DataFrame(
        {
            "timestamp": ts,
            "azimuth_deg": solpos["azimuth"].to_numpy(dtype=float),
            "elevation_deg": solpos["elevation"].to_numpy(dtype=float),
            "zenith_deg": solpos["zenith"].to_numpy(dtype=float),
        }
    )


def project_sun_to_image(
    azimuth_deg: np.ndarray | float,
    zenith_deg: np.ndarray | float,
    calib: Calibration,
    image_width: int,
    image_height: int,
    azimuth_offset_deg: float = 0.0,
    azimuth_clockwise: bool = True,
    image_offset_x_px: float = 0.0,
    image_offset_y_px: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    azimuth_deg = np.asarray(azimuth_deg, dtype=np.float32)
    zenith_deg = np.asarray(zenith_deg, dtype=np.float32)
    radius = calib.f_px_per_rad * np.deg2rad(zenith_deg)
    direction = 1.0 if azimuth_clockwise else -1.0
    angle = np.deg2rad(direction * azimuth_deg + azimuth_offset_deg)
    x = calib.cx + radius * np.sin(angle) + float(image_offset_x_px)
    y = calib.cy - radius * np.cos(angle) + float(image_offset_y_px)
    return np.clip(x, -0.5, image_width - 0.5), np.clip(y, -0.5, image_height - 0.5)


def build_solar_feature_vector(
    sun_x_px: float,
    sun_y_px: float,
    azimuth_deg: float,
    zenith_deg: float,
    image_width: int,
    image_height: int,
) -> np.ndarray:
    x_norm = 2.0 * (sun_x_px / max(image_width - 1, 1)) - 1.0
    y_norm = 2.0 * (sun_y_px / max(image_height - 1, 1)) - 1.0
    az_rad = np.deg2rad(azimuth_deg)
    zen_norm = np.clip(zenith_deg / 90.0, 0.0, 1.0)
    return np.array([x_norm, y_norm, np.sin(az_rad), np.cos(az_rad), zen_norm], dtype=np.float32)
