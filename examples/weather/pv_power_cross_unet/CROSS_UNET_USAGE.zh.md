<!-- SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Cross-Unet 光伏功率预测使用说明

本文档说明 `examples/weather/pv_power_cross_unet` 下的真实数据 Cross-Unet 工作流，包括数据准备、模型训练、模型推理、结果收集与论文指标检查。

## 1. 环境准备

在已安装 PhysicsNeMo 和 PyTorch 的环境中安装示例依赖。论文规模配置建议使用 GPU 训练。

```bash
cd /home/horde/tmp/physicsnemo
pip install -r examples/weather/pv_power_cross_unet/requirements.txt
```

如果使用 `physicsnemo_26.03` 容器，启动 sweep 前先确认 CUDA 可用：

```bash
docker exec physicsnemo_26.03 bash -lc 'nvidia-smi && python -c "import torch; print(torch.cuda.is_available())"'
```

如果容器内 `nvidia-smi` 出现 NVML 初始化错误，先 stop 再 start 该容器，然后重新验证 CUDA，再继续训练。

## 2. 数据准备

真实数据工作流需要如下目录结构：

```text
examples/weather/pv_power_cross_unet/dataset/
  All_dataset/
    S-1.csv
    S-2.csv
    S-3.csv
    S-4.csv
    KDASC.csv
  AIweatherdata/
    station_data/
      0.csv
      4.csv
      7.csv
      8.csv
    data/
      0/*.csv
      4/*.csv
      7/*.csv
      8/*.csv
```

支持三类前瞻天气输入：

- `nwp`：支持 S-1 到 S-4，使用 7 个历史目标/观测通道和 6 个 NWP 预报通道。
- `satellite`：支持 S-1 到 S-4 以及 KDASC，使用 `SWR` 作为前瞻辐照度通道。
- `ai`：支持 S-1 到 S-4，分别映射到 AI 站点 id `0`、`4`、`7`、`8`，使用每日发布 AI 预报文件中的 `ssrd_corrdiff`。

AI 数据读取会过滤 `9.96921e36` 这类缺测哨兵值，并且历史天气窗口只从预测时刻之前已经发布的 forecast 中拼接，避免未来信息泄漏。

## 3. 单个模型训练

使用 `train_real_cross_unet.py` 和 Hydra override 启动训练。默认配置适合快速 smoke test；论文规模配置使用 `paper_preset=True`。

示例：训练 S-1、NWP、4 小时预测窗口。

```bash
python examples/weather/pv_power_cross_unet/train_real_cross_unet.py \
  mode=train \
  dataset_root=examples/weather/pv_power_cross_unet/dataset \
  station_name=S-1 \
  weather_source=nwp \
  seq_len=96 \
  pred_len=16 \
  paper_preset=True \
  output_dir=examples/weather/pv_power_cross_unet/outputs/s1_nwp_4h
```

checkpoint 会写入：

```text
<output_dir>/<station_name>/checkpoints/real_cross_unet_<station_name>.pt
```

`paper_preset=True` 会应用以下论文规模配置：

```text
d_model=252, d_ff=492, n_heads=4, e_layers=3, seg_len=12,
max_epochs=100, early_stop_patience=10, start_lr=1e-4,
seed=2021, normalization=minmax
```

## 4. 模型推理

使用和训练相同的数据根目录与输出目录执行预测：

```bash
python examples/weather/pv_power_cross_unet/train_real_cross_unet.py \
  mode=predict \
  dataset_root=examples/weather/pv_power_cross_unet/dataset \
  station_name=S-1 \
  weather_source=nwp \
  paper_preset=True \
  output_dir=examples/weather/pv_power_cross_unet/outputs/s1_nwp_4h \
  prediction_path=examples/weather/pv_power_cross_unet/outputs/s1_nwp_4h/predictions.csv
```

预测 CSV 包含：

```text
station,timestamp,horizon_step,prediction,target
```

CSV 中的 `prediction` 和 `target` 是反归一化后的原始目标尺度，便于人工检查。和论文表对比时，runner 会按训练集 target range 归一化后计算 MAE/MSE。

## 5. 论文复现实验 sweep

使用 `run_paper_cross_unet.py` 只复现 Cross-Unet 相关行，覆盖 Table 4 的 NWP、Table 5 的 satellite、Table 6 的 AI weather。

代表性 smoke sweep：

```bash
python examples/weather/pv_power_cross_unet/run_paper_cross_unet.py \
  --phase representative \
  --paper-targets-csv examples/weather/pv_power_cross_unet/paper_targets.csv
```

完整 65 个 case sweep：

```bash
python examples/weather/pv_power_cross_unet/run_paper_cross_unet.py \
  --phase full \
  --resume \
  --paper-targets-csv examples/weather/pv_power_cross_unet/paper_targets.csv \
  --metric-scale normalized
```

只运行指定 case：准备包含 `station,source,horizon_label` 三列的 CSV，然后使用：

```bash
python examples/weather/pv_power_cross_unet/run_paper_cross_unet.py \
  --phase full \
  --cases-csv examples/weather/pv_power_cross_unet/paper_repro_failed_cases.csv \
  --resume \
  --paper-targets-csv examples/weather/pv_power_cross_unet/paper_targets.csv \
  --output-dir examples/weather/pv_power_cross_unet/outputs/paper_repro_retry \
  --metric-scale normalized
```

## 6. 结果收集与检查

runner 会输出：

```text
<output_dir>/metrics.csv
<output_dir>/report.md
<output_dir>/runs/<source>/<station>/<horizon>/...
```

`metrics.csv` 字段包括：

```text
station,source,horizon_label,seq_len,pred_len,seed,
mae,mse,r2,paper_mae,paper_mse,paper_r2,
pass_mae,pass_mse,pass_r2,param_count
```

pass 判定规则：

- MAE pass：`mae <= paper_mae * 1.05`
- MSE pass：`mse <= paper_mse * 1.05`
- R2 pass：`r2 >= paper_r2 * 0.95`

和论文表对比时应使用 `--metric-scale normalized`。该模式下 R2 不受线性尺度影响，MAE/MSE 与论文预处理使用的归一化尺度一致。

## 7. 常用验证命令

```bash
pytest test/examples/weather/test_pv_power_cross_unet_real_data.py   test/experimental/models/pv_power/cross_unet/test_cross_unet.py -q

ruff check examples/weather/pv_power_cross_unet/train_real_cross_unet.py   examples/weather/pv_power_cross_unet/run_paper_cross_unet.py   test/examples/weather/test_pv_power_cross_unet_real_data.py   test/experimental/models/pv_power/cross_unet/test_cross_unet.py
```
