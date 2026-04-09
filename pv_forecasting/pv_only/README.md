# PV-Only Forecasting (LSTM)

任务定义：

- 输入：`[t-120, t-105, ..., t-15]` 共 8 个 15min PV 点
- 输出：`t+15` 的 PV 功率
- 归一化：`power / 66300`（训练时），评估输出还原为 W

## 运行

在项目根目录执行：

```bash
# 1) 预处理，生成中间样本表
python -m pv_forecasting.pv_only.preprocess \
  --pv-csv data/power/power-LSK_N_1year_2025_to_2026.csv

# 2) 训练（自动区分 GPU/CPU）
python -m pv_forecasting.pv_only.train --run-name pv_only

# 3) 推理/重算结果（默认 latest run）
python -m pv_forecasting.pv_only.infer
```

## 输出

- 中间结果：`pv_forecasting/pv_only/derived/forecast_windows.csv`
- 训练 run：`pv_forecasting/pv_only/model_output/run_*/`
  - `train.log`
  - `run_metadata.json`
  - `history.csv`
  - `best_model.pt`
  - `last_model.pt`
  - `metrics_val.csv`
  - `metrics_all.csv`
  - `predictions_val.csv`
  - `predictions_all.csv`

