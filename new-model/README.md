# Minimal Sun-Conditioned PV Forecasting

这套代码按照 `/Users/huangchouyue/Desktop/minimal_sun_conditioned_pv_framework.md` 实现了一个最小但结构完整的超短时 PV 概率预测框架：

- 输入：8 张过去 14 分钟天空图（2 分钟间隔）+ 4 个历史 PV
- 结构：Frame Encoder + Explicit Motion Field + Sun-Conditioned Attention + PV Encoder + Quantile Head
- 输出：`t+15min` 的 `q10 / q50 / q90`
- 记录：数据清单、预处理摘要、训练历史、指标、预测结果、motion/attention 可视化

## 目录结构

```text
new-model/
  configs/base.json
  new_model/
    data/
    losses/
    models/
    train/
    utils/
    viz/
  scripts/
    prepare_dataset.py
    train.py
    infer.py
    make_case_study.py
```

## 设计说明

### 1. 太阳几何与旧标定逻辑

复用了仓库现有的 `data/calibration.json` 等距鱼眼标定思路：

- `r = f * theta`
- `theta = solar_zenith_rad`
- `cx, cy, f` 来自 `/Users/huangchouyue/Projects/PVPF/data/calibration.json`

因为旧代码里没有明确保存“方位角在图像里的绝对零方向”，新实现默认采用：

- `azimuth=0°` 对应图像正上方
- 方位角顺时针增加
- 额外提供 `azimuth_offset_deg` 配置，后续如果发现整体旋转偏了一个固定角度，只需要改配置

### 2. clear-sky 目标

文档推荐使用 clear-sky index。当前实现默认用 `pvlib` 计算站点 `Ineichen GHI`，再按峰值功率缩放成一个 **clear-sky power proxy**：

```text
clear_sky_power_w = peak_power_w * ghi_clear / ghi_clear_max
```

这是一个工程上可用的近似版本，优点是实现简单、可直接跑通；如果后续有组件倾角/朝向/逆变器信息，可以把这里替换成更真实的 clear-sky PV 模型。

### 3. 结果与中间结果记录

`prepare_dataset.py` 会输出：

- `artifacts/dataset/samples.csv`
- `artifacts/dataset/preprocess_summary.json`

`train.py` 每次运行会输出到：

- `artifacts/runs/run_YYYYMMDD-HHMMSS/`

其中包含：

- `run_config.json`
- `history.csv`
- `metrics_{train,val,test}.json`
- `predictions_{train,val,test}.csv`
- `best_model.pt`
- `train.log`
- `figures/forecast_band_*.png`
- `figures/motion_attention_*.png`

## 默认数据来源

默认配置使用：

- 相机目录：`/Users/huangchouyue/Projects/PVPF/data/camera_data/raw`
- PV：`/Users/huangchouyue/Projects/PVPF/data/power/power-LSK_N.csv`
- sky mask：`/Users/huangchouyue/Projects/PVPF/data/sky_mask.png`
- calibration：`/Users/huangchouyue/Projects/PVPF/data/calibration.json`

如果你更想直接走现成索引，也可以把 `camera_index_csv` 指到：

- `/Users/huangchouyue/Projects/PVPF/data/camera_data/index/raw_index.csv`

## 使用方法

### 1. 准备数据

```bash
python3 /Users/huangchouyue/Projects/PVPF/new-model/scripts/prepare_dataset.py \
  --config /Users/huangchouyue/Projects/PVPF/new-model/configs/base.json
```

### 2. 训练

```bash
python3 /Users/huangchouyue/Projects/PVPF/new-model/scripts/train.py \
  --config /Users/huangchouyue/Projects/PVPF/new-model/configs/base.json
```

### 3. 推理

```bash
python3 /Users/huangchouyue/Projects/PVPF/new-model/scripts/infer.py \
  --config /Users/huangchouyue/Projects/PVPF/new-model/configs/base.json \
  --run-dir /Users/huangchouyue/Projects/PVPF/new-model/artifacts/runs/<run_name> \
  --split test
```

### 4. 画 case study

```bash
python3 /Users/huangchouyue/Projects/PVPF/new-model/scripts/make_case_study.py \
  --run-dir /Users/huangchouyue/Projects/PVPF/new-model/artifacts/runs/<run_name> \
  --split test
```

## 依赖

当前实现依赖：

- Python 3.10+
- `torch`
- `numpy`
- `pandas`
- `matplotlib`
- `Pillow`
- `pvlib`

仓库根目录的 `requirements.txt` 里还没有写入 `torch` 和 `Pillow`，如果环境里没装，需要额外安装。

## 后续建议

优先建议你下一步做三件事：

1. 先跑一次 `prepare_dataset.py`，确认样本数、split、sun 坐标投影是否合理
2. 再跑小 epoch 的训练，检查 `figures/` 里的 motion 和 attention 是否聚焦在太阳附近或云边界
3. 如果太阳位置整体偏转，先只改 `configs/base.json` 里的 `azimuth_offset_deg`

