from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model_compare.training import train_model
from model_compare.utils import load_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Train image-only PV forecasting baselines.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/base.json")
    parser.add_argument(
        "--model",
        choices=[
            "convlstm",
            "cnn_gru",
            "image_regressor",
            "vae_regressor",
            "convlstm_pv",
            "cnn_gru_pv",
            "image_regressor_pv",
            "vae_regressor_pv",
        ],
        required=True,
        help="Baseline to train.",
    )
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()
    train_model(load_json(args.config), args.model, epochs_override=args.epochs, max_samples=args.max_samples)


if __name__ == "__main__":
    main()
