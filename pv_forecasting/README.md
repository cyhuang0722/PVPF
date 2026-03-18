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

## Satellite-Only Baseline

新增一个独立的 satellite-only baseline，思路是：

- 输入过去 `T_in=5` 帧 FY-4 ROI patch。
- 原始输入默认用 `0.65 / 0.825 / 1.61 um` 三个通道，也就是 `[2, 3, 5]`，并额外为 `0.65 um` 通道构造 `cloud index map`。
- 模型 backbone 改成更接近论文的 `shared spatial encoder + 3D temporal encoder + recurrent future states`。
- 训练目标不再只有功率，而是同时预测未来 4 个 horizon 的 `power` 和 `cloud index map`。

默认配置文件在 [satellite_only_baseline.json](/Users/huangchouyue/Projects/PVPF/pv_forecasting/configs/satellite_only_baseline.json)。

```bash
# 1. 预处理 ROI 序列，生成 forecast_windows.csv、训练集通道 mean/std
python -m pv_forecasting.preprocess_satellite \
  --config pv_forecasting/configs/satellite_only_baseline.json

# 2. 可选：同时把标准化后的 patch 序列打包成 npz shards，加快训练
python -m pv_forecasting.preprocess_satellite \
  --config pv_forecasting/configs/satellite_only_baseline.json \
  --pack

# 3. 训练 satellite-only baseline
python -m pv_forecasting.train_satellite \
  --config pv_forecasting/configs/satellite_only_baseline.json \
  --run-name sat_only

# 4. 对 train/val/test split 做推理
python -m pv_forecasting.infer_satellite \
  --config pv_forecasting/configs/satellite_only_baseline.json \
  --run-dir pv_forecasting/model_output/run_xxx_sat_only \
  --split val
```

几个实现约定：

- ROI 数据默认读取 `data/satellite/roi`。
- patch 默认围绕 `(22.3364, 114.2633)` 裁 `96x96`。
- 预处理对 `[2,3,5]` 这几个 VIS/NIR 通道直接使用 `CALChannel` 查表后的值，也就是以反射率表征作为模型输入；同时按论文里的定义，用最近 `10` 天同一时刻的像素最小值和当前帧最大值构造 `cloud index map`。
- `targets` 统一按 `peak_power_w=66300` 做归一化训练，评估时再还原回 W，输出每个 horizon 的 `RMSE / MAE` 和 `mean_rmse_W`。
