# Model Compare

This directory contains lightweight image-only baselines for comparing against the `cloud-prob`
seminar model. The baselines use the same `samples.csv` split and produce aligned
`predictions_*.csv` / `metrics_*.json` outputs with weather-stratified metrics.

## Models

- `convlstm`: masked RGB image sequence -> ConvLSTM -> Gaussian CSI regression.
- `cnn_gru`: masked RGB image sequence -> frame CNN -> GRU -> Gaussian CSI regression.
- `image_regressor`: latest masked RGB image only -> CNN -> Gaussian CSI regression.
- `vae_regressor`: 8 uniformly sampled masked RGB frames -> VAE latent sequence -> GRU -> Gaussian CSI regression,
  with reconstruction and KL regularization.

These baselines intentionally do not use the hand-crafted sun/path/weather features from
`cloud-prob`. Weather labels are only used for reporting grouped metrics.

`data.max_steps` defaults to `8`, so sequence models uniformly sample 8 frames from each interval
instead of using every available image. This keeps ConvLSTM/VAE runs manageable while preserving
early-to-late temporal context.

## Paths

For a different machine, edit only `configs/base.json`:

- `paths.workspace_root`
- `paths.project_root`
- `data.samples_csv`
- `data.sky_mask_path`

## Train

Recommended server commands:

```bash
cd /home/chuangbn/projects/PVPF/model-compare
conda run --no-capture-output -n torch_h5 python -u scripts/train.py --model convlstm
conda run --no-capture-output -n torch_h5 python -u scripts/train.py --model cnn_gru
conda run --no-capture-output -n torch_h5 python -u scripts/train.py --model image_regressor
conda run --no-capture-output -n torch_h5 python -u scripts/train.py --model vae_regressor
```

Local smoke test:

```bash
cd /Users/huangchouyue/Projects/PVPF/model-compare
conda run --no-capture-output -n torch_h5 python -u scripts/train.py --model cnn_gru --epochs 1 --max-samples 48
```

## Compare Runs

After all three models finish:

```bash
conda run --no-capture-output -n torch_h5 python -u scripts/compare_runs.py
```

Outputs are written to `artifacts/comparison`:

- `overall_comparison.png`
- `weather_comparison.png`
- `model_comparison_test.csv`
- `comparison_summary.md`
