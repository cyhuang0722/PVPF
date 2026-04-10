# U-Net PV Mapping

This module follows the SkyGPT U-Net regression idea, but the task here is:

- Input: 8 historical sky images at `t-15, t-13, ..., t-1`
- Target: PV at `t`
- Implementation: stack the 8 frames along channel dimension, then feed the stacked tensor to a modified U-Net regressor.

## Files

- `preprocess.py`: build training samples csv for the `t` mapping task.
- `dataset.py`: load 8-frame image stacks with optional `sky_mask`.
- `model.py`: modified U-Net for regression.
- `train.py`: train and evaluate the model.

## Typical usage

Build samples:

```bash
conda activate torch_h5
python /Users/huangchouyue/Projects/PVPF/pv_forecasting/u-net/preprocess.py
```

Train:

```bash
conda activate torch_h5
python /Users/huangchouyue/Projects/PVPF/pv_forecasting/u-net/train.py
```
