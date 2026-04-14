# Sun-Conditioned Stochastic Cloud State Network

这个目录实现了你在`/Users/huangchouyue/Desktop/seminar材料/scsn_model_design.md`里定义的新模型，并保持了和`/Users/huangchouyue/Projects/PVPF/new-model`相近的工程结构：

```text
scsn-model/
  configs/base.json
  scsn_model/
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
```

## 当前实现

- 输入：16帧天空图序列
- 通道：`RGB + RBR + sun_distance + mask = 6`
- 编码：`Image Encoder + ConvLSTM`
- 结构化潜变量：`motion / opacity / gap / sun`
- 动力学：`Variational GRU`
- 解码：未来15步的 motion / opacity / gap / sun occlusion / transmission
- 预测头：输出`q10 / q50 / q90`
- 训练损失：`Pinball + quantile crossing + KL + motion smoothness + RBR reconstruction`

## 使用方法

1. 准备样本

```bash
python3 /Users/huangchouyue/Projects/PVPF/scsn-model/scripts/prepare_dataset.py \
  --config /Users/huangchouyue/Projects/PVPF/scsn-model/configs/base.json
```

2. 训练

```bash
python3 /Users/huangchouyue/Projects/PVPF/scsn-model/scripts/train.py \
  --config /Users/huangchouyue/Projects/PVPF/scsn-model/configs/base.json
```

3. 推理

```bash
python3 /Users/huangchouyue/Projects/PVPF/scsn-model/scripts/infer.py \
  --config /Users/huangchouyue/Projects/PVPF/scsn-model/configs/base.json \
  --run-dir /Users/huangchouyue/Projects/PVPF/scsn-model/artifacts/runs/<run_name> \
  --split test
```

## 产物

训练输出目录位于：

- `/Users/huangchouyue/Projects/PVPF/scsn-model/artifacts/runs/run_YYYYMMDD-HHMMSS/`

其中包括：

- `run_config.json`
- `history.csv`
- `metrics_{train,val,test}.json`
- `predictions_{train,val,test}.csv`
- `best_model.pt`
- `train.log`
- `figures/forecast_band_*.png`
- `figures/cloud_state_*.png`

`cloud_state_*.png`会展示输入末帧、太阳注意力、transmission、opacity、gap、未来运动场和sun occlusion曲线，便于检查模型是否学到了你希望的可解释云状态。
