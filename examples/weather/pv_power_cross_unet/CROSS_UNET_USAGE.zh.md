<!-- SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Cross-Unet 真实光伏数据使用说明

本文档说明使用Cross-Unet完成光伏功率预测的工作流。该工作流读取一个 15 分钟频率的 CSV 文件，用户通过
配置指定时间列、功率目标列和天气输入列。

## 1. 环境准备

在已安装 PhysicsNeMo 和 PyTorch 的环境中安装示例依赖：

```bash
cd /path/to/physicsnemo
pip install -r examples/weather/pv_power_cross_unet/requirements.txt
```

## 2. 数据准备

输入数据应为CSV格式，要求如下：

- 一列时间戳
- 一列光伏功率历史数据
- 至少一列天气数据
- 时间戳严格按 15 分钟间隔排列，不能有重复时间或缺失


可以在 `conf/real_data.yaml` 中修改列名，也可以用 Hydra override：

```yaml
data_file: ./dataset/your_data.csv
time_col: Time
target_col: pv_power
weather_cols:
  - irradiation
freq_minutes: 15
```

数据列名称应与CSV文件中的列名称一致。

## 3. 模型训练

从示例目录运行：

```bash
cd examples/weather/pv_power_cross_unet
python train_real_cross_unet.py mode=train
```

快速 smoke run：

```bash
python train_real_cross_unet.py \
  mode=train \
  data_file=./dataset/your_data.csv \
  time_col=Time \
  target_col=pv_power \
  weather_cols='[irradiation]' \
  horizon_label=4h \
  max_epochs=1 \
  max_train_samples=4 \
  max_valid_samples=2 \
  batch_size=2 \
  batch_size_valid=2 \
  d_model=32 \
  d_ff=64 \
  output_dir=./outputs/smoke
```

默认模型配置为Cross_Unet论文中的原始配置：

```text
d_model=256, d_ff=512, n_heads=4, e_layers=3,
start_lr=1e-4, max_epochs=100, seed=2021,
nonlinear_correlation_proj=True, use_bottleneck_in_decoder=True
```

`early_stop_patience` 默认是 `5`，可以在命令行或配置文件中修改。

## 4. 预测窗口

常用预测窗口可通过 `horizon_label` 设置：

```text
4h -> pred_len=16,  seq_len=96,  seg_len=12
1d -> pred_len=96,  seq_len=96,  seg_len=24
7d -> pred_len=672, seq_len=672, seg_len=48
```

也可以直接设置 `seq_len`、`pred_len` 和 `seg_len`。

## 5. 模型推理

训练完成后执行：

```bash
python train_real_cross_unet.py \
  mode=predict \
  data_file=./dataset/your_data.csv \
  horizon_label=4h \
  output_dir=./outputs/smoke
```

预测 CSV 字段为：

```text
data_file,timestamp,horizon_step,prediction,target
```

`prediction` 和 `target` 都是反归一化后的原始目标尺度，便于检查。

## 6. 结果评估

`mode=evaluate` 会先执行预测，然后向 metrics CSV 追加一行：

```bash
python train_real_cross_unet.py \
  mode=evaluate \
  data_file=./dataset/your_data.csv \
  horizon_label=4h \
  output_dir=./outputs/smoke \
  metrics_csv_path=./outputs/smoke/metrics.csv
```

metrics CSV 字段为：

```text
data_file,horizon_label,seq_len,pred_len,weather_cols,weather_channels,
mae,mse,rmse,r2,best_valid_loss,epoch,checkpoint_path,prediction_path
```

MAE、MSE、RMSE、R2 会在所有预测步展平后统一计算。

