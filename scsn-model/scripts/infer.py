from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scsn_model.data.dataset import SunConditionedCloudDataset
from scsn_model.models.full_model import SunConditionedStochasticCloudModel
from scsn_model.train.trainer import _evaluate_split
from scsn_model.utils.io import load_json, normalize_config_paths, resolve_project_path
from scsn_model.utils.runtime import configure_matplotlib_cache

configure_matplotlib_cache(ROOT / "artifacts")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inference for SCSN PV forecasting model")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = parser.parse_args()

    config = normalize_config_paths(load_json(args.config))
    run_dir = resolve_project_path(args.run_dir, must_exist=True)
    device_cfg = str(config.get("device", "auto")).lower()
    device = torch.device("cuda" if device_cfg == "auto" and torch.cuda.is_available() else device_cfg if device_cfg != "auto" else "cpu")

    dataset = SunConditionedCloudDataset(
        csv_path=config["data"]["samples_csv"],
        split=args.split,
        image_size=tuple(config["data"]["image_size"]),
        sky_mask_path=config["data"].get("sky_mask_path"),
        peak_power_w=float(config["data"]["peak_power_w"]),
    )
    model = SunConditionedStochasticCloudModel(config["model"]).to(device)
    model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location=device))

    pred_df, metrics = _evaluate_split(model, dataset, config, device)
    out_csv = run_dir / f"infer_{args.split}.csv"
    pred_df.to_csv(out_csv, index=False)
    print(json.dumps({"out_csv": str(out_csv), "metrics": metrics}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
