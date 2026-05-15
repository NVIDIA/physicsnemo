<!-- SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Cross-Unet 真实光伏数据使用说明

本文档说明 `examples/weather/pv_power_cross_unet` 下的通用真实数据
Cross-Unet 工作流。该工作流读取一个 15 分钟频率的 CSV 文件，用户通过
配置指定时间列、功率目标列和天气输入列。

## 1. 环境准备

在已安装 PhysicsNeMo 和 PyTorch 的环境中安装示例依赖：

```bash
cd /home/horde/tmp/physicsnemo
pip install -r examples/weather/pv_power_cross_unet/requirements.txt
```

如果使用 `physicsnemo_26.03` 容器，训练前先确认 CUDA 可用：

```bash
docker exec physicsnemo_26.03 bash -lc 'nvidia-smi && python -c "import torch; print(torch.cuda.is_available())"'
```

如果容器内 `nvidia-smi` 出现 NVML 初始化错误，先 stop 再 start 该容器，
然后重新验证 CUDA，再继续训练。

## 2. 数据准备

准备一个 CSV，要求如下：

- 一列时间戳
- 一列待预测的光伏功率
- 至少一列天气输入
- 时间戳严格按 15 分钟间隔排列，不能有重复时间

其他列会被忽略。默认配置指向一个 `obs_data2` 文件：

```text
examples/weather/pv_power_cross_unet/dataset/obs_data2/653206.csv
```

默认使用这些列：

```text
Time,r_apower,r_tirra
```

可以在 `conf/real_data.yaml` 中修改列名，也可以用 Hydra override：

```yaml
data_file: ./dataset/obs_data2/653206.csv
time_col: Time
target_col: r_apower
weather_cols:
  - r_tirra
freq_minutes: 15
```

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
  data_file=./dataset/obs_data2/653206.csv \
  time_col=Time \
  target_col=r_apower \
  weather_cols='[r_tirra]' \
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

默认模型配置对齐上游 Cross-Unet 训练脚本：

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
  data_file=./dataset/obs_data2/653206.csv \
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
  data_file=./dataset/obs_data2/653206.csv \
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

## 7. 验证命令

运行真实数据工作流测试和模型测试：

```bash
pytest test/examples/weather/test_pv_power_cross_unet_real_data.py \
  test/experimental/models/pv_power/cross_unet/test_cross_unet.py -q
```
