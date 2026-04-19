# Cloud Seg

轻量版 decision clear-sky library 思路：

- 从 `data/weather.csv` 里选 `2=多云` 的日期。
- 从 `data/clear_sky.csv` 里读取人工整理过的 clear-sky 时间窗。
- 对每个多云日的每个小时，找到它之前最近、且覆盖该小时的 clear-sky 时间窗。
- 在每天 `08:00` 到 `17:00`，每个整点各选一张最接近该小时的图。
- 先在 sky mask 内分别归一化 `R` 和 `B` 通道，再计算 `RBR`。
- 用 `median(RBR_cloudy) / median(RBR_clear)` 把 clear-sky RBR baseline 对齐到当前图。
- 先计算 `raw_diff = RBR_cloudy - RBR_clear_matched`。
- 使用 Gaussian blur (`sigma=100`) 估计 large-scale trend，并做 `local_residual = raw_diff - trend`。
- 默认使用上一版 final RBR local residual mask，尽量保留之前表现较好的 case。
- 根据当前图的 blue-sky fraction 和 gray/cloud-color fraction 做轻量天气判断。
- `clear_sky`: 蓝天比例高且 RBR/local residual 很弱时，直接输出全 0 cloud mask。
- `partly_cloudy`: 使用 RBR local residual，但排除明显蓝天区域。
- `broken_cloudy`: 蓝天洞适中且碎云很多时，使用 RBR residual + 非蓝天云区补充。
- `mixed`: 直接使用 RBR local residual。
- `overcast`: RBR residual 不可靠，切到低饱和度 gray/cloud-color mask。
- 使用 opening、closing 和 remove-small-components 做轻量清理。
- 默认输出到 `cloud_seg/outputs_final/review_pngs/`，所有 PNG 平铺在同一个文件夹，方便批量检查。
- 输出配对结果和统计表到 `cloud_seg/outputs_final/manifests/`，不生成 `.npz` 文件。
- 每次运行会清空旧的 `outputs_final/review_pngs` 和 `outputs_final/manifests`，避免旧结果残留。
- 默认只处理 20 个满足条件的 cloudy days；加 `--all` 会处理所有满足条件的 cloudy days。

运行：

```bash
python /home/chuangbn/projects/PVPF/cloud_seg/run_diff_experiment.py --all
```
