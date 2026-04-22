# Sun Patch Probabilistic PV Forecast

This project is a clean successor experiment to `scsn-model`. It keeps the old
model untouched and starts from a smaller, more reliable probabilistic baseline.

The first model avoids explicit cloud motion prediction and avoids full future
image generation. It predicts the 15 minute ahead clear-sky-index residual from:

- PV history
- solar geometry
- sun-relative RBR patch/ring statistics
- global sky RBR/brightness statistics

The main forecast distribution is a Student-t residual distribution around a
smart-persistence-style CSI baseline. Auxiliary heads predict future sun-patch
statistics for light self-supervision.

## Why This Project Exists

The previous RBR decoder produced visually strong edges and weak spatial
correlation with the true future RBR map. This project instead makes the
visualization explainable:

- current RGB with target sun rings
- past sun-patch RBR curves
- PV q10/q50/q90 forecast bands
- predicted distribution diagnostics
- auxiliary future sun-patch statistics

## Layout

```text
sun-patch-prob/
  configs/base.json
  scripts/build_features.py
  scripts/train.py
  scripts/evaluate.py
  sun_patch_prob/
    data.py
    features.py
    metrics.py
    model.py
    viz.py
```

## Recommended Validation Environment

Use the existing `torch_h5` conda environment:

```bash
conda run -n torch_h5 python /Users/huangchouyue/Projects/PVPF/sun-patch-prob/scripts/build_features.py \
  --config /Users/huangchouyue/Projects/PVPF/sun-patch-prob/configs/base.json

conda run -n torch_h5 python /Users/huangchouyue/Projects/PVPF/sun-patch-prob/scripts/train.py \
  --config /Users/huangchouyue/Projects/PVPF/sun-patch-prob/configs/base.json
```

The stronger RMSE candidate is:

```bash
conda run -n torch_h5 python /Users/huangchouyue/Projects/PVPF/sun-patch-prob/scripts/train.py \
  --config /Users/huangchouyue/Projects/PVPF/sun-patch-prob/configs/wide_regularized.json
```

To prepare an all-weather sample table without touching `scsn-model/artifacts`,
write it inside this project:

```bash
conda run -n torch_h5 python /Users/huangchouyue/Projects/PVPF/sun-patch-prob/scripts/prepare_samples.py \
  --mode all-weather \
  --out-dir /Users/huangchouyue/Projects/PVPF/sun-patch-prob/artifacts/dataset_all_weather
```

Feature-family ablation:

```bash
conda run -n torch_h5 python /Users/huangchouyue/Projects/PVPF/sun-patch-prob/scripts/feature_ablation.py \
  --config /Users/huangchouyue/Projects/PVPF/sun-patch-prob/configs/all_weather_balanced.json
```

The cleaned feature pipeline disables hard disk/ring inputs by default and keeps
global sky plus Gaussian weighted sun-region features:

```bash
conda run -n torch_h5 python /Users/huangchouyue/Projects/PVPF/sun-patch-prob/scripts/build_features.py \
  --config /Users/huangchouyue/Projects/PVPF/sun-patch-prob/configs/all_weather_clean_gated.json \
  --force
```

Weather-aware gate between the `global` and `weighted` ablation branches:

```bash
conda run -n torch_h5 python /Users/huangchouyue/Projects/PVPF/sun-patch-prob/scripts/gated_ensemble.py \
  --ablation-dir /Users/huangchouyue/Projects/PVPF/sun-patch-prob/artifacts/ablations/<ablation_run>
```

V2 learned-gate model with target-sun patch CNN/GRU:

```bash
conda run -n torch_h5 python /Users/huangchouyue/Projects/PVPF/sun-patch-prob/scripts/train_v2.py \
  --config /Users/huangchouyue/Projects/PVPF/sun-patch-prob/configs/v2_sunpatch_gated.json
```

For a quick smoke run:

```bash
conda run -n torch_h5 python /Users/huangchouyue/Projects/PVPF/sun-patch-prob/scripts/build_features.py \
  --config /Users/huangchouyue/Projects/PVPF/sun-patch-prob/configs/base.json \
  --max-samples 90 --force

conda run -n torch_h5 python /Users/huangchouyue/Projects/PVPF/sun-patch-prob/scripts/train.py \
  --config /Users/huangchouyue/Projects/PVPF/sun-patch-prob/configs/base.json \
  --feature-csv /Users/huangchouyue/Projects/PVPF/sun-patch-prob/artifacts/features/features.csv_smoke_90.gz \
  --epochs 3
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
- `figures/case_*.png`
