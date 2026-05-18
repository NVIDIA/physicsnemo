<!-- SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Cross-Unet Real PV Data Usage

This guide describes how to use Cross-Unet to conduct photovoltaic power prediction. The workflow reads one CSV file at a
15-minute cadence and uses user-specified column names for time, power, and weather inputs.

## 1. Environment

Install the example requirements in a PhysicsNeMo environment with PyTorch
available:

```bash
cd /path/to/physicsnemo
pip install -r examples/weather/pv_power_cross_unet/requirements.txt
```

## 2. Data Preparation

The input data should be in a CSV file:

- one timestamp column
- one historical power column
- one or more weather columns
- regular 15-minute timestamps with no duplicate or missing times

Set the column names in `conf/real_data.yaml` or through Hydra overrides:

```yaml
data_file: ./dataset/your_data.csv
time_col: Time
target_col: pv_power
weather_cols:
  - irradiation
freq_minutes: 15
```

The name of the data columns should be in accordance with that in the CSV file.

## 3. Training

From the example directory:

```bash
cd examples/weather/pv_power_cross_unet
python train_real_cross_unet.py mode=train
```

A quick smoke run:

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

The default model settings follow the Cross-Unet paper:

```text
d_model=256, d_ff=512, n_heads=4, e_layers=3,
start_lr=1e-4, max_epochs=100, seed=2021,
nonlinear_correlation_proj=True, use_bottleneck_in_decoder=True
```

`early_stop_patience` defaults to `5` and can be overridden.

## 4. Horizons

Use `horizon_label` for common PV forecasting windows:

```text
4h -> pred_len=16,  seq_len=96,  seg_len=12
1d -> pred_len=96,  seq_len=96,  seg_len=24
7d -> pred_len=672, seq_len=672, seg_len=48
```

You can also set `seq_len`, `pred_len`, and `seg_len` directly.

## 5. Prediction

After training, run:

```bash
python train_real_cross_unet.py \
  mode=predict \
  data_file=./dataset/your_data.csv \
  horizon_label=4h \
  output_dir=./outputs/smoke
```

The prediction CSV contains:

```text
data_file,timestamp,horizon_step,prediction,target
```

Prediction and target values are saved on the original target scale.

## 6. Evaluation

`mode=evaluate` runs prediction and appends one row to the metrics CSV:

```bash
python train_real_cross_unet.py \
  mode=evaluate \
  data_file=./dataset/your_data.csv \
  horizon_label=4h \
  output_dir=./outputs/smoke \
  metrics_csv_path=./outputs/smoke/metrics.csv
```

The metrics CSV contains:

```text
data_file,horizon_label,seq_len,pred_len,weather_cols,weather_channels,
mae,mse,rmse,r2,best_valid_loss,epoch,checkpoint_path,prediction_path
```

MAE, MSE, RMSE, and R2 are computed over all forecast steps after flattening
the prediction and target arrays.

