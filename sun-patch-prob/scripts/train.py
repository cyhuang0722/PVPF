from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sun_patch_prob.training import train_model
from sun_patch_prob.utils import load_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the sun-patch gated probabilistic PV model.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/base.json")
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_model(load_json(args.config), epochs_override=args.epochs, max_samples=args.max_samples)


if __name__ == "__main__":
    main()
