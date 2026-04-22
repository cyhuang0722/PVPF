from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
SCSN_ROOT = WORKSPACE / "scsn-model"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCSN_ROOT))

from scsn_model.data.preprocessing import build_samples, save_samples
from sun_patch_prob.utils import load_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare sample CSVs for the sun-patch probabilistic project.")
    parser.add_argument("--source-config", type=Path, default=SCSN_ROOT / "configs/base.json")
    parser.add_argument("--mode", choices=["filtered", "all-weather"], default="all-weather")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "artifacts/dataset_all_weather")
    args = parser.parse_args()

    config = copy.deepcopy(load_json(args.source_config))
    config["data"]["artifact_dir"] = str(args.out_dir)
    config["data"]["samples_csv"] = str(args.out_dir / "samples.csv")

    if args.mode == "all-weather":
        config["data"]["exclude_clear_sky_days"] = False
        config["data"]["allowed_weather_tags"] = ["clear_sky", "overcast", "cloudy", "partly_cloudy"]
    elif args.mode == "filtered":
        config["data"]["exclude_clear_sky_days"] = True
        config["data"]["allowed_weather_tags"] = ["cloudy", "partly_cloudy"]

    df, summary = build_samples(config)
    save_samples(df, summary, config)
    print(f"saved {len(df)} {args.mode} samples to {config['data']['samples_csv']}")


if __name__ == "__main__":
    main()

