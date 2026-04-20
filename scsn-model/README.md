# Sun-Conditioned Stochastic Cloud State Network

这个目录实现了你在`/Users/huangchouyue/Desktop/seminar材料/scsn_model_design.md`里定义的 SCSN，并且目录结构直接对应设计文档里的模块。

```text
scsn-model/
  configs/base.json
  scsn_model/
    data/
    losses/
    models/
      image_encoder.py
      temporal_encoder.py
      latent_state.py
      stochastic_dynamics.py
      cloud_decoder.py
      power_head.py
      full_model.py
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
- 结构化潜变量：`z_dynamics`
- 动力学：`Variational GRU`
- 解码：未来15步 pixel-level `RBR variation mean / log-variance`
- 预测头：输出 PV 高斯分布的 `mu / sigma`，再确定性得到 `q10 / q25 / q50 / q75 / q90`
- 训练损失：`PV Gaussian NLL + KL + RBR reconstruction + future RBR variation Gaussian NLL`
- 弱监督：不再使用 cloud-mask/pseudo-mask supervision；`sky_mask_path` 只作为有效天空区域 mask

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

## 已移除的旧设计残留

以下 legacy 结构和监督逻辑已经从 `scsn-model` 中移除：

- 旧的多分支时序骨架
- 旧的伪标签运动监督与方向验证链路
- 旧版特征聚合拼装方式
- 任何依赖旧最小基线设计的模型分支

`cloud_state_*.png`会展示输入末帧、目标太阳区域权重、过去/未来 RBR variation、预测的 future RBR mean、预测的 future RBR variance、对应 overlay 和 PV 分布诊断。所有彩色 map 都带 colorbar。不可验证的 `transmission / opacity / gap / future cloud probability / cloud risk / sun occlusion` 空间图已经移除。

## RBR Variation 概率监督

`configs/base.json` 和 `configs/experiment_no_sun_attention.json` 默认启用了：

- `loss.rbr_distribution_weight`: `0.20`
- `loss.rbr_distribution_sun_weight`: `2.0`

未来 15 分钟的 target 来自真实未来 15 张图的相邻帧 RBR variation。模型预测每个 pixel 的 mean 和 variance，并用 Gaussian NLL 训练；PV head 使用 sun-region/global 的 RBR mean 和 variance 生成 PV 分布。模型不再预测显式运动方向，也不再使用 current-sun latent attention；太阳权重只在未来目标太阳位置附近做 RBR variation 读出与 NLL 加权。
