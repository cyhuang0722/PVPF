from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scsn_model.utils.runtime import configure_matplotlib_cache

configure_matplotlib_cache(ROOT / "artifacts")

from scsn_model.train.trainer import train_model
from scsn_model.utils.io import load_json, normalize_config_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Sun-Conditioned Stochastic Cloud State Network")
    parser.add_argument("--config", required=True, help="Path to config JSON")
    args = parser.parse_args()

    config = normalize_config_paths(load_json(args.config))
    run_dir = train_model(config)
    print(f"Training finished. Outputs saved to {run_dir}")


if __name__ == "__main__":
    main()
