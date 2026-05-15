<!-- SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Cross-Unet Real PV Data Usage

This guide describes the generic real-data Cross-Unet workflow under
`examples/weather/pv_power_cross_unet`. The workflow reads one CSV file at a
15-minute cadence and uses user-specified column names for time, power, and
weather inputs.

## 1. Environment

Install the example requirements in a PhysicsNeMo environment with PyTorch
available:

```bash
cd /home/horde/tmp/physicsnemo
pip install -r examples/weather/pv_power_cross_unet/requirements.txt
```

When using the `physicsnemo_26.03` container, verify CUDA before training:

```bash
docker exec physicsnemo_26.03 bash -lc 'nvidia-smi && python -c "import torch; print(torch.cuda.is_available())"'
```

If `nvidia-smi` reports an NVML initialization error inside the container, stop
and start the container, then verify CUDA again before resuming.

## 2. Data Preparation

Prepare a CSV with:

- one timestamp column
- one target power column
- one or more weather columns
- regular 15-minute timestamps with no duplicate times

Extra columns are ignored. The default config points at an `obs_data2` file:

```text
examples/weather/pv_power_cross_unet/dataset/obs_data2/653206.csv
```

with these columns:

```text
Time,r_apower,r_tirra
```

Set the column names in `conf/real_data.yaml` or through Hydra overrides:

```yaml
data_file: ./dataset/obs_data2/653206.csv
time_col: Time
target_col: r_apower
weather_cols:
  - r_tirra
freq_minutes: 15
```

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

The default model settings follow the upstream Cross-Unet training script:

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
  data_file=./dataset/obs_data2/653206.csv \
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
  data_file=./dataset/obs_data2/653206.csv \
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

## 7. Verification

Run the real-data workflow tests and the model tests:

```bash
pytest test/examples/weather/test_pv_power_cross_unet_real_data.py \
  test/experimental/models/pv_power/cross_unet/test_cross_unet.py -q
```
