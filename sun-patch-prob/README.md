# Sun Patch Probabilistic PV Forecast

This project contains the current PV forecasting model only. Old RBR decoder
experiments, tabular-only baselines, and legacy project references have been
removed from this directory.

## Model

The model predicts a 15-minute-ahead probabilistic PV forecast around a
smart-persistence CSI baseline.

Inputs:

- 16 recent target-sun-centered image patches
- per-frame channels: masked RGB, RBR, target-sun distance, sky mask
- global PV, solar, weather, and whole-sky summary features

Architecture:

- CNN frame encoder for each sun patch
- GRU temporal encoder over the recent patch sequence
- MLP encoder for global features
- learned gate between global context and sun-patch context
- Student-t probabilistic head for `q10/q25/q50/q75/q90`
- auxiliary future sun-region statistics head for light self-supervision

## Layout

```text
sun-patch-prob/
  configs/base.json
  scripts/train.py
  sun_patch_prob/
    data.py
    evaluation.py
    metrics.py
    model.py
    training.py
    utils.py
    viz.py
  artifacts/
    dataset_all_weather/
    features/
    runs/
```

Expected prepared data files:

- `artifacts/dataset_all_weather/samples.csv`
- `artifacts/features/global_features.csv.gz`

## Train

Use the `torch_h5` environment:

```bash
conda run -n torch_h5 python -u /Users/huangchouyue/Projects/PVPF/sun-patch-prob/scripts/train.py \
  --config /Users/huangchouyue/Projects/PVPF/sun-patch-prob/configs/base.json
```

For a quick smoke run:

```bash
conda run -n torch_h5 python -u /Users/huangchouyue/Projects/PVPF/sun-patch-prob/scripts/train.py \
  --config /Users/huangchouyue/Projects/PVPF/sun-patch-prob/configs/base.json \
  --max-samples 80 \
  --epochs 2
```

## Outputs

Each run writes to:

```text
sun-patch-prob/artifacts/runs/run_YYYYMMDD-HHMMSS/
```

Expected files:

- `best_model.pt`
- `history.csv`
- `predictions_train.csv`
- `predictions_val.csv`
- `predictions_test.csv`
- `metrics_train.json`
- `metrics_val.json`
- `metrics_test.json`
- `figures/forecast_band_*.png`
