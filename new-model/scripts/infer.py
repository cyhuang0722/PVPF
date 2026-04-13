from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from new_model.utils.runtime import configure_matplotlib_cache

configure_matplotlib_cache(ROOT / "artifacts")

from new_model.data.dataset import SunConditionedPVDataset
from new_model.models.full_model import MinimalSunConditionedPVModel
from new_model.train.trainer import _evaluate_split  # reuse evaluation path
from new_model.utils.io import load_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Inference for minimal sun-conditioned PV model")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = parser.parse_args()

    config = load_json(args.config)
    run_dir = Path(args.run_dir)
    device = torch.device("cuda" if torch.cuda.is_available() and str(config.get("device", "auto")) != "cpu" else "cpu")

    dataset = SunConditionedPVDataset(
        csv_path=config["data"]["samples_csv"],
        split=args.split,
        image_size=tuple(config["data"]["image_size"]),
        sky_mask_path=config["data"].get("sky_mask_path"),
    )
    model = MinimalSunConditionedPVModel(config["model"]).to(device)
    model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location=device))

    pred_df, metrics = _evaluate_split(model, dataset, config, device)
    out_csv = run_dir / f"infer_{args.split}.csv"
    pred_df.to_csv(out_csv, index=False)
    print(json.dumps({"out_csv": str(out_csv), "metrics": metrics}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
