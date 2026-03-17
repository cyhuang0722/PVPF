# Sky + PV 多步预测

按 `sky_pv_forecasting_model_design.md` 实现：60 分钟天空图序列 + 过去 4 个 15min PV → 预测 5 个 horizon [t, t+15, t+30, t+45, t+60]。

## 用法

```bash
# 在 PVPF 项目根目录下
cd PVPF

# 1. 数据预处理（默认生成 derived/forecast_windows.csv），可选参数 --pack
python -m pv_forecasting.preprocess

# ablation 例子：30 张图、每 2 分钟采样一次，输出到新目录
python -m pv_forecasting.preprocess --img-len 30 --img-stride-min 2 --out-dir derived_ablation/30x2 --pack

# decoupled horizon 分层预测
python -m pv_forecasting.preprocess \
  --horizon 4 \
  --future-offsets-min 30,60,120,240 \
  --out-dir derived_ablation/h30_60_120_240 \
  --pack

# 2. 训练（结果写入 pv_forecasting/model_output/run_YYYYMMDD-HHMMSS/）
python -m pv_forecasting.train

# ablation 训练例子：指定数据目录和 run 名称
python -m pv_forecasting.train --pack-dir derived_ablation/30x2/packed_forecast --run-name 30x2

# 3. 测试数据推理（需 PV 覆盖测试时段）
python -m pv_forecasting.preprocess_test --cam-dir data/cam_test [--pv-csv data/power/xxx.csv] --pack
python -m pv_forecasting.infer --test-pack-dir derived/test/packed [run_dir]

# 分层预测
python -m pv_forecasting.preprocess_test \
  --horizon 4 \
  --future-offsets-min 30,60,120,240 \
  --out-dir derived/test_h30_60_120_240 \
  --pack
```

## 数据与路径

- 预处理默认：`data/cam_dir`（天空图）、`data/power/power-LSK_N.csv`（15min PV）。
- 输出：`pv_forecasting/derived/forecast_windows.csv`；训练输出：`model_output/run_*/`（history.csv, metrics.csv, predictions_all.csv, best_model.pt, run_metadata.json）。
- ablation 常用参数：`--img-len`、`--img-stride-min`、`--out-dir`、`--pack-dir`、`--run-name`。

## 模型结构

- Sky 分支：每帧 CNN(3→128) → GRU(128) → 128 维。
- PV 分支：Linear(1→32) → GRU(32→64) → 64 维。
- 融合：concat(192) → MLP → 5 维（MSE 损失，可选按 horizon 加权）。

依赖：PyTorch、pandas、numpy、PIL；预处理复用 `pv_output_prediction.preproces_data`（图像索引与 PV 加载）。
