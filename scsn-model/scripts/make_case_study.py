from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scsn_model.utils.runtime import configure_matplotlib_cache

configure_matplotlib_cache(ROOT / "artifacts")

from scsn_model.viz.forecast import save_forecast_band_plot


def main() -> None:
    parser = argparse.ArgumentParser(description="Make forecast case-study plot")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    pred_csv = run_dir / f"predictions_{args.split}.csv"
    df = pd.read_csv(pred_csv)
    df["ts_target"] = pd.to_datetime(df["ts_target"])
    if args.start:
        df = df[df["ts_target"] >= pd.Timestamp(args.start)]
    if args.end:
        df = df[df["ts_target"] <= pd.Timestamp(args.end)]
    if df.empty:
        raise RuntimeError("No rows available for the requested case-study range.")
    out_path = run_dir / "figures" / f"case_study_{args.split}.png"
    save_forecast_band_plot(df, out_path, title=f"Case study: {args.split}")
    print(f"Saved case study to {out_path}")


if __name__ == "__main__":
    main()
