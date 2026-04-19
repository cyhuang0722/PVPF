from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scsn_model.data.preprocessing import build_samples, save_samples
from scsn_model.utils.io import load_json, normalize_config_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare sun-conditioned PV forecasting dataset")
    parser.add_argument("--config", required=True, help="Path to config JSON")
    args = parser.parse_args()

    config = normalize_config_paths(load_json(args.config))
    df, summary = build_samples(config)
    save_samples(df, summary, config)
    print(f"Saved {len(df)} samples to {config['data']['samples_csv']}")


if __name__ == "__main__":
    main()
