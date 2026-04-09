# ConvLSTM Encoder for PV t+15

目标：使用图像序列 `t-15` 到 `t-1`（每 2 分钟，8 帧）预测 `t+15` 的 PV 功率。

## 数据输入

- PV: `data/power/power-LSK_N_0117-0401.csv`（列：`date,value`）
- Camera index: `data/camera_data/index/raw_index.csv`（列：`timestamp,file_path,...`）
- 图片文件路径来自 index 的 `file_path`（绝对路径）

## 时间对齐规则（在线 preprocess）

- 不需要额外 preprocess 脚本，`train.py` 内会自动构建样本。
- 以 PV 时间 `target=t+15` 为主轴，得到 anchor `t=target-15`。
- 目标输入时刻是 `t-15, t-13, t-11, t-9, t-7, t-5, t-3, t-1`（每 2 分钟）。
- 在相机索引时间轴上，为每个目标时刻找“最近一帧”（默认允许最大时间误差 `90s`，可调）。
- 会过滤 `pv_target_w <= 0` 的样本（仅保留正功率样本）。
- 训练时会对每帧输入图像乘 `data/sky_mask.png`（可通过参数关闭或更换）。

## 先做 preprocess（推荐）

在项目根目录执行：

```bash
python "pv_forecasting/ConvLSTM-encoder/preprocess.py" \
  --power-csv "data/power/power-LSK_N_0117-0401.csv" \
  --camera-index-csv "data/camera_data/index/raw_index_resized_64.csv" \
  --max-time-diff-sec 180 \
  --out-csv "derived/samples_t_plus_15.csv"
```

生成：
- `pv_forecasting/ConvLSTM-encoder/derived/samples_t_plus_15.csv`
- `pv_forecasting/ConvLSTM-encoder/derived/preprocess_summary.json`

## 训练

在项目根目录执行：

```bash
python "pv_forecasting/ConvLSTM-encoder/train.py" \
  --samples-csv "pv_forecasting/ConvLSTM-encoder/derived/samples_t_plus_15.csv" \
  --sky-mask-path "data/sky_mask.png" \
  --epochs 30 \
  --batch-size 32 \
  --img-channels 1 \
  --peak-power-w 66300 \
  --run-name convlstm_encoder
```

训练时标签按 `pv_norm = pv_W / 66300` 归一化，导出预测和指标时会还原回 W。

如果 `samples_csv` 或 `camera index` 里保存的是原始路径（如 `data/camera_data/raw/2026/...`），
但你想实际读取 `data/camera_data/resized_64/2026/...`，可加路径重映射参数：

```bash
--camera-path-prefix-from "/Users/huangchouyue/Projects/PVPF/data/camera_data/raw/2026" \
--camera-path-prefix-to "/Users/huangchouyue/Projects/PVPF/data/camera_data/resized_64/2026"
```

## 输出

结果会保存到：

`pv_forecasting/ConvLSTM-encoder/model_output/run_YYYYMMDD-HHMMSS_convlstm_encoder/`

包含：

- `train.log`
- `run_metadata.json`
- `history.csv`
- `best_model.pt`
- `last_model.pt`
- `metrics_val.csv`
- `metrics_all.csv`
- `predictions_val.csv`
- `predictions_all.csv`
