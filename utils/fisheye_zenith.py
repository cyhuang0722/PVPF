from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class EquidistantCalibration:
    """Equidistant fisheye calibration for zenith mapping.

    Model:
        r = f * theta
        theta = r / f
    where theta is the zenith angle (radians), r is pixel radius from (cx, cy).
    """
    cx: float
    cy: float
    f_px_per_rad: float

    @staticmethod
    def from_json(path: str | Path) -> "EquidistantCalibration":
        p = Path(path)
        data = json.loads(p.read_text(encoding="utf-8"))
        intr = data.get("intrinsics", data)  # allow flat format
        return EquidistantCalibration(
            cx=float(intr["cx"]),
            cy=float(intr["cy"]),
            f_px_per_rad=float(intr["f_px_per_rad"]),
        )


def pixel_to_zenith_rad(u: np.ndarray, v: np.ndarray, calib: EquidistantCalibration) -> np.ndarray:
    """Vectorized pixel -> zenith angle (radians)."""
    r = np.sqrt((u - calib.cx) ** 2 + (v - calib.cy) ** 2)
    return r / calib.f_px_per_rad


def pixel_to_zenith_deg(u: np.ndarray, v: np.ndarray, calib: EquidistantCalibration) -> np.ndarray:
    """Vectorized pixel -> zenith angle (degrees)."""
    return np.degrees(pixel_to_zenith_rad(u, v, calib))


def zenith_deg_to_radius_px(zenith_deg: float, calib: EquidistantCalibration) -> float:
    """Zenith angle (deg) -> radius in pixels, for drawing iso-zenith circles."""
    return float(calib.f_px_per_rad * np.deg2rad(float(zenith_deg)))


def draw_iso_zenith_circles(
    img_bgr: np.ndarray,
    calib: EquidistantCalibration,
    zenith_degs: Tuple[int, ...] = (10, 20, 30, 40, 50, 60, 70, 80),
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    put_text: bool = True,
) -> np.ndarray:
    """Return a copy of image with iso-zenith circles drawn."""
    out = img_bgr.copy()
    cx_i, cy_i = int(round(calib.cx)), int(round(calib.cy))
    cv2.circle(out, (cx_i, cy_i), 6, color, -1)

    for z in zenith_degs:
        rad = zenith_deg_to_radius_px(z, calib)
        cv2.circle(out, (cx_i, cy_i), int(round(rad)), color, thickness)
        if put_text:
            x = int(round(calib.cx + rad)) + 6
            y = int(round(calib.cy))
            cv2.putText(out, f"{z}deg", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

    return out


def make_zenith_map_deg(
    h: int,
    w: int,
    calib: EquidistantCalibration,
    stride: int = 1,
    dtype=np.float32,
) -> np.ndarray:
    """Create a zenith angle map (degrees) for an image grid.

    If stride > 1, computes a coarse map and upsamples (fast for big images).
    """
    if stride < 1:
        raise ValueError("stride must be >= 1")

    hh = (h + stride - 1) // stride
    ww = (w + stride - 1) // stride

    ys = (np.arange(hh) * stride).astype(np.float32)
    xs = (np.arange(ww) * stride).astype(np.float32)
    xx, yy = np.meshgrid(xs, ys)

    zen = pixel_to_zenith_deg(xx, yy, calib).astype(dtype)

    if stride == 1:
        return zen

    # Upsample back to (h, w)
    zen_up = cv2.resize(zen, (w, h), interpolation=cv2.INTER_LINEAR)
    return zen_up.astype(dtype)


# ---------- Example usage ----------
if __name__ == "__main__":
    calib = EquidistantCalibration.from_json("PVPF/data/calibration.json")

    # Example: compute zenith at a single pixel
    u, v = 1600.0, 1500.0
    print("zenith_deg =", float(pixel_to_zenith_deg(np.array([u]), np.array([v]), calib)[0]))

    # Example: draw circles
    # img = cv2.imread("some_frame.jpg")
    # out = draw_iso_zenith_circles(img, calib)
    # cv2.imwrite("debug_circles.png", out)