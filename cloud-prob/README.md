# Cloud Probabilistic PV Forecasting

This is a clean seminar-oriented model version:

**Weather-Conditioned Sun-Aware Probabilistic PV Forecasting**

The model avoids rigid cloud motion prediction. It learns a short temporal
representation from recent sky-image patches around the calibrated target sun
position, conditions the forecast on weather regime, and outputs a Student-t
predictive distribution for PV power.

## Model

Input:

- recent sky image sequence
- patch channels: masked RGB, RBR, distance-to-calibrated-sun map, sky mask
- tabular weather/cloud summary features
- weather class: `clear_sky`, `cloudy`, `overcast`
- sun position from calibrated fisheye solar geometry

Architecture:

- CNN frame encoder
- GRU temporal encoder
- weather embedding
- global feature encoder
- Student-t probabilistic head: `loc`, `scale`, `df`

Outputs:

- `q10/q25/q50/q75/q90`
- forecast bands
- weather-wise metrics
- diagnostic `cloud_gate`, `residual_limit`, `scale`

## Prepare Dataset

Before running on a new machine, edit only `configs/base.json`:

```json
"paths": {
  "workspace_root": "/path/to/PVPF",
  "project_root": "/path/to/PVPF/cloud-prob"
}
```

All data, output, mask, calibration, and run paths are expanded from these two
values. If your server data layout is different, adjust the explicit path fields
in the same `prepare`, `data`, and `train` sections of `configs/base.json`.

```bash
cd /path/to/PVPF/cloud-prob
conda run -n torch_h5 python -u scripts/prepare_dataset.py --config configs/base.json
```

By default this uses `${workspace_root}/data/calibration.json`.
The dataset builder computes solar azimuth/zenith with `pvlib`, then projects
the current and target sun positions into the 256x256 sky image using the
historically fitted camera orientation and fisheye projection parameters from
commit `de829bef31c842ff2e35018da87cfa7dbe2a8d50`.

Expected output:

```text
cloud-prob/artifacts/dataset/samples.csv
cloud-prob/artifacts/dataset/prepare_summary.json
```

## Train

```bash
cd /path/to/PVPF/cloud-prob
conda run -n torch_h5 python -u scripts/train.py --config configs/base.json
```

Smoke test:

```bash
cd /path/to/PVPF/cloud-prob

conda run -n torch_h5 python -u scripts/prepare_dataset.py \
  --config configs/base.json \
  --max-samples 90

conda run -n torch_h5 python -u scripts/train.py \
  --config configs/base.json \
  --max-samples 90 \
  --epochs 2
```

## Seminar Story

1. Cloud motion is unstable and not directly supervised.
2. PV response is dominated by cloud state near the sun.
3. Weather regime controls forecast uncertainty.
4. Student-t output gives a calibrated probabilistic forecast band.
