from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scsn_model.utils.io import ensure_dir, load_json, save_json
from scsn_model.utils.runtime import configure_matplotlib_cache

configure_matplotlib_cache(ROOT / "artifacts")


def _sample_day_rows(df: pd.DataFrame, day: str, samples_per_day: int) -> pd.DataFrame:
    day_df = df[df["ts_anchor"].dt.strftime("%Y-%m-%d") == day].sort_values("ts_anchor").reset_index(drop=True)
    if day_df.empty:
        return day_df
    if len(day_df) <= samples_per_day:
        return day_df
    idx = sorted(set(round(float(x)) for x in np.linspace(0, len(day_df) - 1, samples_per_day)))
    return day_df.iloc[idx].reset_index(drop=True)


def _draw_sun_marker(image_path: str, sun_x: float, sun_y: float, out_path: Path, label: str) -> None:
    with Image.open(image_path) as im:
        img = im.convert("RGB")
    draw = ImageDraw.Draw(img)
    r = 8
    draw.ellipse((sun_x - r, sun_y - r, sun_x + r, sun_y + r), outline=(255, 60, 60), width=3)
    draw.line((sun_x - 14, sun_y, sun_x + 14, sun_y), fill=(255, 60, 60), width=2)
    draw.line((sun_x, sun_y - 14, sun_x, sun_y + 14), fill=(255, 60, 60), width=2)
    draw.text((10, 10), label, fill=(255, 255, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create sun projection sanity-check overlays")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dates", nargs="+", default=["2026-03-12", "2026-03-19", "2026-03-25"])
    parser.add_argument("--samples-per-day", type=int, default=5)
    parser.add_argument("--out-dir", default=str(ROOT / "artifacts" / "sanity_check_sun"))
    args = parser.parse_args()

    config = load_json(args.config)
    samples_path = Path(config["data"]["samples_csv"])
    df = pd.read_csv(samples_path)
    df["ts_anchor"] = pd.to_datetime(df["ts_anchor"])
    out_dir = ensure_dir(args.out_dir)

    summary_rows: list[dict] = []
    for day in args.dates:
        sampled = _sample_day_rows(df, day, args.samples_per_day)
        for _, row in sampled.iterrows():
            img_paths = json.loads(row["img_paths"]) if isinstance(row["img_paths"], str) else row["img_paths"]
            image_path = img_paths[-1]
            ts_str = pd.Timestamp(row["ts_anchor"]).strftime("%Y%m%d_%H%M")
            label = (
                f"{pd.Timestamp(row['ts_anchor']).strftime('%Y-%m-%d %H:%M')}  "
                f"az={row['azimuth_deg']:.1f}  zen={row['zenith_deg']:.1f}"
            )
            out_path = out_dir / day / f"sun_overlay_{ts_str}.png"
            _draw_sun_marker(image_path, float(row["sun_x_px"]), float(row["sun_y_px"]), out_path, label)
            summary_rows.append(
                {
                    "date": day,
                    "ts_anchor": str(row["ts_anchor"]),
                    "image_path": image_path,
                    "out_path": str(out_path),
                    "sun_x_px": float(row["sun_x_px"]),
                    "sun_y_px": float(row["sun_y_px"]),
                    "azimuth_deg": float(row["azimuth_deg"]),
                    "zenith_deg": float(row["zenith_deg"]),
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "sanity_check_summary.csv", index=False)
    save_json(
        out_dir / "sanity_check_summary.json",
        {
            "dates": args.dates,
            "samples_per_day": args.samples_per_day,
            "n_outputs": int(len(summary_df)),
            "summary_csv": str(out_dir / "sanity_check_summary.csv"),
        },
    )
    print(f"Saved {len(summary_df)} sanity-check overlays to {out_dir}")


if __name__ == "__main__":
    main()
