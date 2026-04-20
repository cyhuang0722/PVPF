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
- 结构化潜变量：`z_motion / z_opacity / z_gap / z_sun`
- 动力学：`Variational GRU`
- 解码：未来15步的 `motion / cloud probability / sun-region cloud probability`
- 预测头：输出`q10 / q50 / q90`
- 训练损失：`Pinball + quantile crossing + KL + motion smoothness + RBR reconstruction`
- 弱监督：可选读取 `cloud_seg/outputs_final/manifests/hourly_summary.csv`，按输入末帧匹配 cloudy/clear 图对，重算 cloud mask 后只监督当前 `cloud probability`

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

`cloud_state_*.png`会展示输入末帧、太阳注意力、当前云概率、15min 云风险、过去/未来 RBR 变化 hotspot、预测 15min hotspot、可用的 pseudo cloud mask 和太阳区域风险摘要，避免把不可验证的 `transmission / opacity / gap` 空间图当作强物理解释。

## Cloud Mask 弱监督

`configs/base.json` 和 `configs/experiment_no_sun_attention.json` 默认启用了：

- `data.cloud_mask_manifest_path`: `/Users/huangchouyue/Projects/PVPF/data/cloud_mask_ref/manifests/hourly_summary.csv`
- `data.cloud_mask_sky_mask_path`: `/Users/huangchouyue/Projects/PVPF/data/sky_mask.png`
- `loss.cloud_mask_weight`: `0.02`
- `loss.cloud_fraction_weight`: `0.10`
- `loss.future_hotspot_weight`: `0.05`

训练时只有能按输入序列末帧文件名命中 manifest 的样本会计算 cloud-mask loss；没有 cloud mask 的样本保持原来的训练逻辑。Cloud mask 只作为轻量 current-cloud prior；未来 15 分钟的运动/变化解释主要由相邻帧 RBR change hotspot 和 PV loss 约束。`opacity_proxy / gap_proxy / transmission_proxy` 默认权重已设为 `0.0`，避免 noisy mask 被强行解释成像素级光学状态。
