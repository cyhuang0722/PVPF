# ConvLSTM Encoder Baseline

这个目录现在用于和 `new-model` 做基础模型对标，默认直接复用 `new-model` 生成的样本清单，因此以下几项已经对齐：

- 输入序列：直接读取样本 CSV 里的 `img_paths`，与 `new-model` 完全一致
- 数据切分：直接读取同一份样本 CSV 里的 `split=train/val/test`
- 目标归一化：默认使用 `clear sky index`
- 结果统计：按和 `new-model` 一致的口径导出 `predictions_{train,val,test}.csv` 与 `metrics_{train,val,test}.json`

## 推荐流程

### 1. 先准备共享样本

如果你还没生成 `new-model` 的样本，先在项目根目录执行：

```bash
conda activate torch_h5
python /Users/huangchouyue/Projects/PVPF/new-model/scripts/prepare_dataset.py \
  --config /Users/huangchouyue/Projects/PVPF/new-model/configs/base.json
```

默认会生成：

`/Users/huangchouyue/Projects/PVPF/new-model/artifacts/dataset/samples.csv`

这份文件里包含：

- `img_paths`
- `target_value`
- `target_pv_w`
- `target_clear_sky_w`
- `split`

## 2. 先做不训练的运行检查

下面这条命令只会加载数据、构建模型、跑一个 batch 的前向，不会开始训练：

```bash
conda activate torch_h5
python /Users/huangchouyue/Projects/PVPF/pv_forecasting/ConvLSTM-encoder/train.py \
  --samples-csv /Users/huangchouyue/Projects/PVPF/new-model/artifacts/dataset/samples.csv \
  --sky-mask-path /Users/huangchouyue/Projects/PVPF/data/sky_mask.png \
  --img-h 256 \
  --img-w 256 \
  --img-channels 3 \
  --dry-run
```

如果你的 `samples.csv` 里保存的是另一套相机根目录，可以再加：

```bash
--camera-path-prefix-from "/旧前缀" \
--camera-path-prefix-to "/新前缀"
```

## 3. 正式训练

确认 dry-run 没问题后，再自己启动训练：

```bash
conda activate torch_h5
python /Users/huangchouyue/Projects/PVPF/pv_forecasting/ConvLSTM-encoder/train.py \
  --samples-csv /Users/huangchouyue/Projects/PVPF/new-model/artifacts/dataset/samples.csv \
  --sky-mask-path /Users/huangchouyue/Projects/PVPF/data/sky_mask.png \
  --img-h 256 \
  --img-w 256 \
  --img-channels 3 \
  --batch-size 8 \
  --epochs 30 \
  --run-name convlstm_encoder_baseline
```

默认就是 clear-sky index 目标；如果你临时想退回原始功率目标，可以显式加：

```bash
--no-use-clear-sky-index
```

## 输出文件

训练输出目录：

`/Users/huangchouyue/Projects/PVPF/pv_forecasting/ConvLSTM-encoder/model_output/run_YYYYMMDD-HHMMSS_<run-name>/`

主要文件：

- `history.csv`
- `best_model.pt`
- `last_model.pt`
- `predictions_train.csv`
- `predictions_val.csv`
- `predictions_test.csv`
- `metrics_train.json`
- `metrics_val.json`
- `metrics_test.json`
- `metrics_summary.csv`
- `run_metadata.json`
- `train.log`

其中预测文件的字段与 `new-model` 对齐：

- `ts_anchor`
- `ts_target`
- `target_value`
- `target_pv_w`
- `target_clear_sky_w`
- `pred_value`
- `pred_value_raw`
- `pred_w`
- `sample_index`
