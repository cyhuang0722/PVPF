import csv
import os
import re
import shutil
from datetime import datetime, date, time, timezone, timedelta

import cv2
import numpy as np

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    from backports.zoneinfo import ZoneInfo


TIMESTAMP_RE = re.compile(r"_(\d{17})_TIMING")


def parse_local_time(filename: str, tz: ZoneInfo) -> datetime | None:
    match = TIMESTAMP_RE.search(filename)
    if not match:
        return None
    ts = match.group(1)
    dt = datetime.strptime(ts, "%Y%m%d%H%M%S%f")
    return dt.replace(tzinfo=tz)


def in_date_time_range(
    dt_local: datetime,
    date_start: date,
    date_end: date,
    time_start: time,
    time_end: time,
) -> bool:
    if not (date_start <= dt_local.date() <= date_end):
        return False
    return time_start <= dt_local.time() <= time_end


def bucket_5min(dt_local: datetime) -> datetime:
    minute = (dt_local.minute // 5) * 5
    return dt_local.replace(minute=minute, second=0, microsecond=0)


def detect_sun_centroid(
    img_bgr: np.ndarray,
) -> tuple[tuple[float, float] | None, float, np.ndarray | None]:
    """Detect sun center robustly.

    Previous version used a global bright-percentile mask, which can be biased by the
    circumsolar halo/glare. This version anchors on the global maximum (sun core) and
    adaptively grows a connected component around it using relative thresholds.

    Returns:
        (u, v) centroid in pixel coords, score in [0,1], and a binary sun mask (uint8).
    """
    if img_bgr is None or img_bgr.size == 0:
        return None, 0.0, None

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1].astype(np.float32) / 255.0
    v = hsv[:, :, 2].astype(np.float32) / 255.0

    # If the image is mostly black after masking, bail out early.
    v_max = float(np.max(v))
    if not np.isfinite(v_max) or v_max <= 0.05:
        return None, 0.0, None

    # Percentile thresholds are more robust when the global maximum is a tiny flare.
    pct_list = [99.97, 99.95, 99.90, 99.80, 99.70, 99.50]
    thr_list = [float(np.percentile(v, p)) for p in pct_list]

    # Sun core is typically near-white (low saturation). Flare streaks are often colored.
    s_max = 0.35

    # Area constraints help reject single-pixel noise and very large halo blobs.
    H, W = v.shape
    # Reject tiny flare dots; for ~2k-4k images, sun core blob should be much larger.
    min_area = max(500, int(0.00008 * H * W))
    max_area = int(0.02 * H * W)  # sun core should be far smaller than this

    best = None  # (score, cx, cy, mask)

    for thr in thr_list:
        # Bright AND low-saturation pixels (sun core is near-white)
        core = ((v >= thr) & (s <= s_max)).astype(np.uint8) * 255

        # Clean up small speckles; keep compact bright core.
        k = max(3, (min(H, W) // 400) | 1)  # odd kernel, scales mildly with resolution
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        core = cv2.morphologyEx(core, cv2.MORPH_OPEN, kernel, iterations=1)
        core = cv2.morphologyEx(core, cv2.MORPH_CLOSE, kernel, iterations=1)

        # Evaluate all connected components; glare streaks can be brighter than the sun core.
        num, labels, stats, _ = cv2.connectedComponentsWithStats(core, connectivity=8)

        for lab in range(1, num):
            area = int(stats[lab, cv2.CC_STAT_AREA])
            if area < min_area or area > max_area:
                #尺寸不合理
                continue

            bw = int(stats[lab, cv2.CC_STAT_WIDTH])
            bh = int(stats[lab, cv2.CC_STAT_HEIGHT])
            if bw <= 0 or bh <= 0:
                continue
            aspect = max(bw, bh) / max(1.0, float(min(bw, bh)))
            if aspect > 1.25:
                # 联通区细长
                continue

            comp = (labels == lab)
            comp_u8 = (comp.astype(np.uint8)) * 255

            contours, _ = cv2.findContours(comp_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            cnt = max(contours, key=cv2.contourArea)

            # 周长
            perimeter = float(cv2.arcLength(cnt, True))
            if perimeter <= 1e-6:
                continue
            circ = float(4.0 * np.pi * area / (perimeter * perimeter))
            if circ < 0.6:
                # circularity  周长-面积 circ = 4πA / P²圆盘 → 接近 1 细长 / 弯月 / 条纹 → 很低
                continue

            (x_c, y_c), enc_r = cv2.minEnclosingCircle(cnt)
            if enc_r <= 1e-3:
                continue
            fill = float(area) / float(np.pi * enc_r * enc_r)
            if fill < 0.6:
                # 填充率低
                continue

            # Weighted centroid
            yy, xx = np.where(comp)
            w = v[yy, xx]
            w_sum = float(np.sum(w))
            if w_sum <= 1e-6:
                continue
            cx = float(np.sum(xx * w) / w_sum)
            cy = float(np.sum(yy * w) / w_sum)

            mean_v = float(np.mean(w))
            # Prefer bright, round, *and larger* blobs (sun core) over tiny flares.
            size_w = float(np.log1p(area))
            score = mean_v * (0.5 * circ + 0.5 * fill) * size_w

            if best is None or score > best[0]:
                best = (score, cx, cy, comp_u8)

        # Early stop: if we already have a very compact and very bright blob.
        if best is not None:
            score, _, _, _ = best
            if score > 0.85:
                break

    if best is None:
        # Fallback: single-pixel maximum (may be flare; use only for debugging)
        y0, x0 = np.unravel_index(int(np.argmax(v)), v.shape)
        sun_mask = np.zeros(v.shape, dtype=np.uint8)
        sun_mask[y0, x0] = 255
        return (int(x0), int(y0)), float(v[y0, x0]), sun_mask

    score, cx, cy, sun_mask = best
    return (int(round(cx)), int(round(cy))), float(score), sun_mask


def main() -> None:
    image_dir = "/Users/huangchouyue/Projects/PVPF/data/cam_dir"
    output_dir = "/Users/huangchouyue/Projects/PVPF/data/data_sort"
    debug_dir = "/Users/huangchouyue/Projects/PVPF/data/data_sort/debug_sun_detect"
    csv_path = "/Users/huangchouyue/Projects/PVPF/data/data_sort/sample.csv"
    mask_path = "/Users/huangchouyue/Projects/PVPF/data/sky_mask.png"
    tz = ZoneInfo("Asia/Singapore")

    date_start = date(2026, 1, 16)
    date_end = date(2026, 1, 18)
    time_start = time(8, 30, 0)
    time_end = time(15, 30, 0)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(debug_dir, exist_ok=True)

    sky_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if sky_mask is None:
        raise FileNotFoundError(f"sky_mask not found or unreadable: {mask_path}")
    image_paths = [
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.lower().endswith(".jpg")
    ]
    image_paths.sort()

    picked = {}
    for path in image_paths:
        dt_local = parse_local_time(os.path.basename(path), tz)
        if dt_local is None:
            print("skip file: ", path)
            continue
        if not in_date_time_range(dt_local, date_start, date_end, time_start, time_end):
            continue
        key = bucket_5min(dt_local)
        if key not in picked:
            picked[key] = (path, dt_local)

    rows = []
    for key in sorted(picked.keys()):
        path, dt_local = picked[key]
        dt_utc = dt_local.astimezone(timezone.utc)
        filename = os.path.basename(path)
        rows.append((filename, dt_local.isoformat(), dt_utc.isoformat()))
        shutil.copy2(path, os.path.join(output_dir, filename))

        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            continue

        # Ensure sky_mask matches image size
        if sky_mask.shape[:2] != img.shape[:2]:
            sky_mask_resized = cv2.resize(sky_mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        else:
            sky_mask_resized = sky_mask

        img_masked = cv2.bitwise_and(img, img, mask=sky_mask_resized)

        centroid, score, sun_mask = detect_sun_centroid(img_masked)
        if centroid is not None:
            u, v = int(round(centroid[0])), int(round(centroid[1]))
            cv2.circle(img_masked, (u, v), 6, (0, 0, 255), -1)
        if sun_mask is not None and np.any(sun_mask > 0):
            # Draw sun mask outline for debugging (less visually misleading than filling).
            cnts, _ = cv2.findContours(sun_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                cv2.drawContours(img_masked, cnts, -1, (0, 0, 255), 2)
        cv2.imwrite(os.path.join(debug_dir, filename), img_masked)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "local_time", "utc_time"])
        writer.writerows(rows)

    print(f"Selected images: {len(rows)}")
    print(f"Saved images to: {output_dir}")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
