# Cross_Unet PV-Power Forecasting

This example contains two CrossUnet workflows:

- `train_cross_unet.py`: a self-contained synthetic-data training example.
- `train_real_cross_unet.py`: a generic real CSV workflow for 15-minute PV
  power forecasting.

The model is
[`CrossUnet`](../../../physicsnemo/experimental/models/pv_power/cross_unet.py),
which consumes five tensors per sample:

```text
x_enc, w_enc, seq_w_nwp_hist, seq_x_hist, target
```

`x_enc` and `seq_x_hist` carry historical target power. `w_enc` and
`seq_w_nwp_hist` carry weather inputs. Training loss uses the primary power
target channel.

## Synthetic Example

```bash
pip install -r requirements.txt
python train_cross_unet.py
```

Hydra overrides can change any field:

```bash
python train_cross_unet.py max_epochs=5 batch_size=16 d_model=128
```

## Real CSV Workflow

The real-data script reads one CSV file. Configure the file and columns in
`conf/real_data.yaml` or on the command line:

```yaml
data_file: ./dataset/obs_data2/653206.csv
time_col: Time
target_col: r_apower
weather_cols:
  - r_tirra
freq_minutes: 15
```

CSV requirements:

- regular 15-minute timestamps
- no duplicate timestamps
- one target power column
- at least one weather column
- any extra columns are ignored

Train with the default config:

```bash
python train_real_cross_unet.py mode=train
```

Run prediction:

```bash
python train_real_cross_unet.py mode=predict
```

Run evaluation and append metrics:

```bash
python train_real_cross_unet.py mode=evaluate
```

Prediction CSV columns:

```text
data_file,timestamp,horizon_step,prediction,target
```

Metrics CSV columns:

```text
data_file,horizon_label,seq_len,pred_len,weather_cols,weather_channels,
mae,mse,rmse,r2,best_valid_loss,epoch,checkpoint_path,prediction_path
```

## Common Horizons

`horizon_label` can set the standard PV forecasting windows:

```text
4h -> pred_len=16,  seq_len=96,  seg_len=12
1d -> pred_len=96,  seq_len=96,  seg_len=24
7d -> pred_len=672, seq_len=672, seg_len=48
```

The same values can be set manually with `seq_len`, `pred_len`, and `seg_len`.

## Default Real-Data Model Settings

The real-data config follows the upstream Cross-Unet script defaults used for
actual PV runs:

```text
d_model=256, d_ff=512, n_heads=4, e_layers=3,
start_lr=1e-4, max_epochs=100, seed=2021,
nonlinear_correlation_proj=True, use_bottleneck_in_decoder=True
```

`early_stop_patience` defaults to `5`.

## More Documentation

- English: [`CROSS_UNET_USAGE.md`](CROSS_UNET_USAGE.md)
- Chinese: [`CROSS_UNET_USAGE.zh.md`](CROSS_UNET_USAGE.zh.md)
