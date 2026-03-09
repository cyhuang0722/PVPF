
import csv
import os
from math import radians

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

from sample_images_5min import detect_sun_centroid

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import re
from datetime import datetime, time
from zoneinfo import ZoneInfo


def refine_centroid_geometry(
    img_bgr: np.ndarray,
    init_uv: tuple[float, float],
    theta_rad: float,
    cx: float,
    cy: float,
    f: float,
    sun_mask: np.ndarray | None = None,
    band_px: float | None = None,
) -> tuple[float, float]:
    """Refine sun centroid using the equidistant geometry constraint.

    Idea:
      - For a given frame, sun zenith angle theta is known from ephemeris.
      - Under equidistant projection, the sun must lie near the ring r_pred = f*theta.
      - Halo/partial occlusion can bias a plain intensity centroid.
      - We therefore select only bright pixels close to r_pred, and compute a V-weighted centroid.

    Args:
      img_bgr: masked BGR image (non-sky already zeroed is OK).
      init_uv: initial centroid (u,v) from detector.
      theta_rad: zenith angle in radians.
      cx,cy,f: fitted equidistant parameters.
      sun_mask: optional binary mask (uint8 0/255) from detector.
      band_px: optional ring half-width in pixels; if None, chosen adaptively.

    Returns:
      refined (u,v). If refinement fails, returns init_uv.
    """
    if img_bgr is None or img_bgr.size == 0:
        return init_uv

    H, W = img_bgr.shape[:2]
    u0, v0 = float(init_uv[0]), float(init_uv[1])

    # Predicted radius for the sun in this frame
    r_pred = float(f * theta_rad)
    if not np.isfinite(r_pred) or r_pred <= 1.0:
        return init_uv

    # Choose a band width: wider at large r (more halo), but still restrictive
    if band_px is None:
        band_px = max(10.0, 0.012 * r_pred)  # ~1.2% of radius

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1].astype(np.float32) / 255.0
    v = hsv[:, :, 2].astype(np.float32) / 255.0

    # Candidate pixels: prefer the detector mask if available, otherwise use a bright+lowS heuristic.
    if sun_mask is not None and np.any(sun_mask > 0):
        cand = (sun_mask > 0)
    else:
        # Bright tail + near-white helps suppress colored flares.
        thr_v = float(np.percentile(v, 99.7))
        cand = (v >= thr_v) & (s <= 0.45)

    if not np.any(cand):
        return init_uv

    yy, xx = np.where(cand)
    # Radial distance of candidates to fitted center
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    keep = np.abs(rr - r_pred) <= band_px

    if np.count_nonzero(keep) < 30:
        return init_uv

    yy = yy[keep]
    xx = xx[keep]

    # Locality constraint: keep points not too far from initial centroid
    # (prevents selecting a different bright blob on the same ring)
    d0 = np.sqrt((xx - u0) ** 2 + (yy - v0) ** 2)
    keep2 = d0 <= max(80.0, 0.08 * r_pred)

    if np.count_nonzero(keep2) < 20:
        return init_uv

    yy = yy[keep2]
    xx = xx[keep2]

    w = v[yy, xx]
    w_sum = float(np.sum(w))
    if w_sum <= 1e-6:
        return init_uv

    u = float(np.sum(xx * w) / w_sum)
    v_ = float(np.sum(yy * w) / w_sum)

    # Sanity: do not allow giant jumps
    if np.hypot(u - u0, v_ - v0) > 120.0:
        return init_uv

    return (u, v_)

def main() -> None:
    image_dir = "/Users/huangchouyue/Projects/PVPF/data/data_sort"
    csv_in = "/Users/huangchouyue/Projects/PVPF/data/data_sort/sample_with_sun.csv" # sample 的时间和太阳位置
    mask_path = "/Users/huangchouyue/Projects/PVPF/data/sky_mask.png"
    plot_out = "/Users/huangchouyue/Projects/PVPF/data/data_sort/r_vs_theta.png"
    out_dir = os.path.dirname(plot_out)
    overlay_out = os.path.join(out_dir, "overlay_zenith_circles.png")
    worst_dir = os.path.join(out_dir, "worst_samples")
    os.makedirs(worst_dir, exist_ok=True)

    sky_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if sky_mask is None:
        raise FileNotFoundError(f"sky_mask not found or unreadable: {mask_path}")

    r_list = []
    theta_list = []
    u_list = []
    v_list = []
    file_list = []
    theta_deg_list = []
    score_list = []
    sun_mask_list = []
    dt_list = []

    ts_pat = re.compile(r"_(\d{17})_TIMING")
    tz = ZoneInfo("Asia/Singapore")

    def parse_dt_from_filename(name: str) -> datetime | None:
        m = ts_pat.search(name)
        if not m:
            return None
        s = m.group(1)
        try:
            # YYYYMMDDhhmmssmmm
            dt = datetime(
                int(s[0:4]), int(s[4:6]), int(s[6:8]),
                int(s[8:10]), int(s[10:12]), int(s[12:14]),
                int(s[14:17]) * 1000,
                tzinfo=tz,
            )
            return dt
        except Exception:
            return None

    with open(csv_in, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row["filename"]
            zenith_deg = float(row["zenith_deg"])
            img_path = os.path.join(image_dir, filename)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is None:
                continue
            if sky_mask.shape[:2] != img.shape[:2]:
                mask = cv2.resize(sky_mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            else:
                mask = sky_mask
            img_masked = cv2.bitwise_and(img, img, mask=mask)

            centroid, score, sun_mask = detect_sun_centroid(img_masked)
            if centroid is None:
                continue

            h, w = img.shape[:2]
            u, v = centroid
            theta_rad = radians(zenith_deg)

            # Store raw observations; we'll fit cx,cy,f later.
            u_list.append(float(u))
            v_list.append(float(v))
            theta_list.append(theta_rad)
            file_list.append(filename)
            theta_deg_list.append(float(zenith_deg))
            score_list.append(float(score))
            sun_mask_list.append(sun_mask)
            dt_list.append(parse_dt_from_filename(filename))

    u_arr = np.array(u_list, dtype=np.float64)
    v_arr = np.array(v_list, dtype=np.float64)
    theta_arr = np.array(theta_list, dtype=np.float64)

    # Initial guess: image center and a typical f for ~180° fisheye
    # (theta=pi/2 should map near the visible horizon radius)
    cx0, cy0 = w / 2.0, h / 2.0
    f0 = min(w, h) / np.pi

    def residuals(p: np.ndarray) -> np.ndarray:
        cx, cy, f = float(p[0]), float(p[1]), float(p[2])
        r = np.sqrt((u_arr - cx) ** 2 + (v_arr - cy) ** 2)
        return r - f * theta_arr

    res = least_squares(
        residuals,
        x0=np.array([cx0, cy0, f0], dtype=np.float64),
        loss="soft_l1",
        f_scale=20.0,
        max_nfev=200,
    )
    cx, cy, f = map(float, res.x)

    # --- Geometry-constrained centroid refinement (one iteration) ---
    u_ref = []
    v_ref = []
    for i in range(len(u_arr)):
        uv0 = (float(u_arr[i]), float(v_arr[i]))
        theta_rad = float(theta_arr[i])
        # Prepare masked image for refinement
        img_path = os.path.join(image_dir, file_list[int(i)])
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is not None:
            h, w = img.shape[:2]
            if sky_mask.shape[:2] != (h, w):
                mask = cv2.resize(sky_mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            else:
                mask = sky_mask
            img_masked = cv2.bitwise_and(img, img, mask=mask)
        else:
            # fallback to zeros if image unreadable
            h, w = sky_mask.shape[:2]
            img_masked = np.zeros((h, w, 3), np.uint8)
        uv1 = refine_centroid_geometry(
            img_bgr=img_masked,
            init_uv=uv0,
            theta_rad=theta_rad,
            cx=cx,
            cy=cy,
            f=f,
            sun_mask=sun_mask_list[int(i)],
        )
        u_ref.append(float(uv1[0]))
        v_ref.append(float(uv1[1]))

    u_arr = np.array(u_ref, dtype=np.float64)
    v_arr = np.array(v_ref, dtype=np.float64)

    # Re-fit (cx,cy,f) using refined centroids
    res2 = least_squares(
        residuals,
        x0=np.array([cx, cy, f], dtype=np.float64),
        loss="soft_l1",
        f_scale=20.0,
        max_nfev=200,
    )
    cx, cy, f = map(float, res2.x)

    r_arr = np.sqrt((u_arr - cx) ** 2 + (v_arr - cy) ** 2)

    # Per-sample absolute residual in pixels
    err_arr = np.abs(r_arr - f * theta_arr)

    # Export worst samples for debugging
    worst_idx = np.argsort(-err_arr)[: min(20, len(err_arr))]
    for k, i in enumerate(worst_idx):
        filename = file_list[int(i)]
        img_path = os.path.join(image_dir, filename)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            continue

        # Match mask size
        if sky_mask.shape[:2] != img.shape[:2]:
            mask = cv2.resize(sky_mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        else:
            mask = sky_mask 
        img_masked = cv2.bitwise_and(img, img, mask=mask)

        # Current (possibly refined) centroid used for r/theta check
        u = float(u_arr[i])
        v = float(v_arr[i])

        # Original detector centroid for comparison
        u0 = float(u_list[int(i)])
        v0 = float(v_list[int(i)])

        theta_deg = float(theta_deg_list[int(i)])
        r = float(r_arr[i])
        e = float(err_arr[i])

        # Draw centroid
        out = img_masked.copy()
        cv2.circle(out, (int(round(u)), int(round(v))), 10, (0, 0, 255), 2)
        cv2.circle(out, (int(round(u)), int(round(v))), 2, (0, 0, 255), -1)
        cv2.circle(out, (int(round(u0)), int(round(v0))), 10, (255, 0, 0), 2)
        cv2.circle(out, (int(round(u0)), int(round(v0))), 2, (255, 0, 0), -1)
        cv2.putText(out, "red=refined, blue=orig", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

        # Draw sun mask outline if available
        sun_mask = sun_mask_list[int(i)]
        if sun_mask is not None and np.any(sun_mask > 0):
            cnts, _ = cv2.findContours(sun_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                cv2.drawContours(out, cnts, -1, (0, 0, 255), 2)

        txt1 = f"theta={theta_deg:.2f} deg, r={r:.1f} px"
        txt2 = f"|r-f*theta|={e:.1f} px, score={score_list[int(i)]:.3f}"
        cv2.putText(out, txt1, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(out, txt2, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

        out_path = os.path.join(worst_dir, f"{k:02d}_err_{e:.1f}_{filename}")
        cv2.imwrite(out_path, out)

    plt.figure(figsize=(6, 5))
    plt.scatter(r_arr, theta_arr, s=12, alpha=0.7, label="samples")

    # Plot fitted relationship in the same axes (x=r, y=theta): theta = r / f
    r_line = np.linspace(r_arr.min(), r_arr.max(), 200)
    theta_line = r_line / f
    plt.plot(r_line, theta_line, color="red", linewidth=1.5, label=f"fit: cx={cx:.1f}, cy={cy:.1f}, f={f:.2f}")

    plt.xlabel("r (pixels)")
    plt.ylabel("theta (rad)")
    plt.title("Equidistant check: r vs theta")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_out, dpi=150)
    plt.close()

    # Pick a representative image: closest to 12:00 local among valid samples
    target_t = time(12, 0, 0)
    best_i = None
    best_dt_diff = None
    for i, dt in enumerate(dt_list):
        if dt is None:
            continue
        dt0 = dt.replace(hour=target_t.hour, minute=target_t.minute, second=target_t.second, microsecond=0)
        diff = abs((dt - dt0).total_seconds())
        if best_dt_diff is None or diff < best_dt_diff:
            best_dt_diff = diff
            best_i = i

    if best_i is None and len(file_list) > 0:
        best_i = 0

    if best_i is not None:
        rep_name = file_list[int(best_i)]
        rep_path = os.path.join(image_dir, rep_name)
        rep_img = cv2.imread(rep_path, cv2.IMREAD_COLOR)
        if rep_img is not None:
            if sky_mask.shape[:2] != rep_img.shape[:2]:
                mask = cv2.resize(sky_mask, (rep_img.shape[1], rep_img.shape[0]), interpolation=cv2.INTER_NEAREST)
            else:
                mask = sky_mask
            rep = cv2.bitwise_and(rep_img, rep_img, mask=mask)

            # Draw iso-zenith circles
            for z_deg in range(10, 90, 10):
                theta_rad = np.deg2rad(float(z_deg))
                rad = f * theta_rad
                cv2.circle(rep, (int(round(cx)), int(round(cy))), int(round(rad)), (0, 255, 0), 2)
                cv2.putText(rep, f"{z_deg}deg", (int(round(cx + rad)) + 6, int(round(cy))), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

            # Mark fitted center
            cv2.circle(rep, (int(round(cx)), int(round(cy))), 8, (0, 255, 0), -1)
            cv2.putText(rep, f"center ({cx:.1f},{cy:.1f})", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(rep, f"f={f:.2f} px/rad", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imwrite(overlay_out, rep)
            print(f"Saved overlay: {overlay_out}")
            print(f"Representative frame: {rep_name}")

    abs_err = err_arr
    print(f"Samples used: {len(r_arr)}")
    print(f"Estimated cx,cy: ({cx:.2f}, {cy:.2f})")
    print(f"Estimated f: {f:.2f} px/rad")
    print(f"Median |r - f*theta|: {np.median(abs_err):.2f} px")
    print(f"P95   |r - f*theta|: {np.percentile(abs_err, 95):.2f} px")
    print(f"Saved worst samples to: {worst_dir}")


if __name__ == "__main__":
    main()
