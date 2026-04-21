from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train import evaluate
from sun_patch_prob.data import FeatureSpec, SunPatchFeatureDataset
from sun_patch_prob.model import StudentTResidualModel
from sun_patch_prob.utils import load_json, resolve_device, save_json
from sun_patch_prob.viz import save_forecast_band


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained sun-patch probabilistic PV model.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    args = parser.parse_args()

    config = load_json(args.run_dir / "run_config.json")
    df = pd.read_csv(config["data"]["feature_csv"])
    split_df = df[df["split"] == args.split].reset_index(drop=True)
    feature_columns = json.loads((args.run_dir / "feature_columns.json").read_text(encoding="utf-8"))
    spec_np = np.load(args.run_dir / "feature_spec.npz")
    spec = FeatureSpec(feature_columns=feature_columns, mean=spec_np["mean"], std=spec_np["std"])
    dataset = SunPatchFeatureDataset(split_df, spec)
    model = StudentTResidualModel(input_dim=len(feature_columns), **config["model"])
    device = resolve_device(config.get("device", "auto"))
    model.load_state_dict(torch.load(args.run_dir / "best_model.pt", map_location=device))
    model.to(device)
    pred, metrics = evaluate(model, dataset, device)
    pred.to_csv(args.run_dir / f"predictions_{args.split}_eval.csv", index=False)
    save_json(args.run_dir / f"metrics_{args.split}_eval.json", metrics)
    save_forecast_band(pred, args.run_dir / "figures" / f"forecast_band_{args.split}_eval.png", f"{args.split} eval forecast")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
