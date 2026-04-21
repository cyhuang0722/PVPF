from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sun_patch_prob.features import build_feature_table
from sun_patch_prob.utils import load_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Build sun-patch probabilistic PV feature table.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/base.json")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()

    config = load_json(args.config)
    out_path = args.output or Path(config["data"]["feature_csv"])
    if args.max_samples > 0 and args.output is None:
        out_path = out_path.with_name(f"{out_path.stem}_smoke_{args.max_samples}{out_path.suffix}")
    if out_path.exists() and not args.force:
        print(f"feature table already exists: {out_path}")
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    features = build_feature_table(config, max_samples=args.max_samples, progress_every=args.progress_every)
    features.to_csv(out_path, index=False, compression="gzip")
    print(f"saved {len(features)} rows x {len(features.columns)} columns to {out_path}")


if __name__ == "__main__":
    main()
