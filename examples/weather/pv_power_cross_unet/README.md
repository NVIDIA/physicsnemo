# Cross_Unet PV-Power Forecasting (synthetic data)

Trains the experimental
[`CrossUnet`](../../../physicsnemo/experimental/models/pv_power/cross_unet.py)
PV-power forecaster on a deterministic synthetic time series. The example is
self-contained: no data download is required.

## What it demonstrates

- Multi-input forward signature (`x_enc`, `w_enc`, `seq_w_nwp_hist`, `seq_x_hist`).
- Standard PhysicsNeMo training-loop scaffolding: `DistributedManager`,
  `LaunchLogger`, `save_checkpoint` / `load_checkpoint`.
- Hydra-driven configuration; override any field on the command line.

## Run

```bash
pip install -r requirements.txt
python train_cross_unet.py
```

Configure via Hydra overrides, for example:

```bash
python train_cross_unet.py max_epochs=5 batch_size=16 d_model=128
```

Outputs land under `./outputs/` (Hydra working directory). Checkpoints are
written to `./outputs/<run>/checkpoints/`.

## Synthetic data

Each window is `seq_len + pred_len` consecutive steps drawn from a
deterministic series:

- A diurnal irradiance signal `clamp(sin(2π t / 96), 0, ∞)` (96 = 24 h at
  15-minute cadence).
- Power = `irradiance · (0.7 + 0.3·sin(t/97))` plus zero-mean Gaussian noise,
  clamped to [0, 1].
- Weather features = `[irradiance, cos-temperature-proxy, sin-humidity-proxy]`
  truncated/padded to `weather_channels`.
- Target features = `target_channels - 1` lagged copies of power followed by
  the current power channel (the last channel is the "primary" signal whose
  cross-channel correlations condition the attention).
- `seq_x_hist` contains the full target/history window with `target_channels`
  columns, matching `x_enc`; Cross_Unet appends the primary current target
  channel internally when constructing the correlation input.

## Real datasets

This recipe is structured so that swapping in a real dataset (e.g. the
[AI-PVOD](https://huggingface.co/datasets/yujiaA/AI-PVOD) collection or the
upstream [PV-power](https://github.com/Z-Yh1/PV-power) station data) only
requires replacing the `SyntheticPVDataset` in `train_cross_unet.py` with a
`Dataset` that emits the same five tensors per item:
`x_enc`, `w_enc`, `seq_w_nwp_hist`, `seq_x_hist`, `target`.

The real-data workflow is designed to be runnable from this directory without
editing Python code. It follows the upstream PV-power setup: KDASC uses one
power history channel plus `SWR`; S-1 through S-4 use six local measurement
history channels plus power, and either six NWP weather features or satellite
`SWR`.

## 1. Prepare the Environment

From the repository root, install PhysicsNeMo and the example dependencies in
the Python environment you want to use:

```bash
pip install -e .
cd examples/weather/pv_power_cross_unet
pip install -r requirements.txt
```

If PhysicsNeMo is already installed in your environment, only the final two
commands are needed.

## 2. Check the Real Data

The provided real data should already be under:

```text
examples/weather/pv_power_cross_unet/dataset/
├── All_dataset/
│   ├── KDASC.csv
│   ├── S-1.csv
│   ├── S-2.csv
│   ├── S-3.csv
│   └── S-4.csv
└── AIweatherdata/
```

The script below uses `dataset/All_dataset` by default. `AIweatherdata` mirrors
the upstream CorrDiff/AI-weather deployment data; the first runnable workflow
uses the simpler station CSV path so new users can train and predict directly.

## 3. Run a Quick Smoke Test

This command trains on a few KDASC windows and writes a small checkpoint:

```bash
python train_real_cross_unet.py \
  mode=train \
  station_name=KDASC \
  seq_len=24 pred_len=8 seg_len=4 \
  d_model=16 n_heads=4 d_ff=32 \
  max_epochs=1 \
  batch_size=2 batch_size_valid=2 \
  max_train_samples=4 max_valid_samples=2 \
  output_dir=./outputs/real_cross_unet_smoke
```

Run inference from that checkpoint:

```bash
python train_real_cross_unet.py \
  mode=predict \
  station_name=KDASC \
  seq_len=24 pred_len=8 \
  batch_size_valid=2 \
  max_predict_samples=2 \
  output_dir=./outputs/real_cross_unet_smoke
```

The prediction CSV will be written to:

```text
outputs/real_cross_unet_smoke/KDASC/predictions/KDASC_predictions.csv
```

It contains `station`, `timestamp`, `horizon_step`, `prediction`, and `target`.

## 4. Train on the Bundled Real Dataset

The default config trains KDASC for a 4-hour horizon (`pred_len=16` at
15-minute cadence):

```bash
python train_real_cross_unet.py mode=train station_name=KDASC
```

Outputs are saved under:

```text
outputs/real_cross_unet/KDASC/
├── checkpoints/real_cross_unet_KDASC.pt
└── predictions/                 # created during inference
```

For the China stations with NWP weather features:

```bash
python train_real_cross_unet.py mode=train station_name=S-1 weather_source=nwp
```

For the same station using satellite `SWR` instead of NWP features:

```bash
python train_real_cross_unet.py mode=train station_name=S-1 weather_source=satellite
```

Available station names in the bundled `All_dataset` folder are `KDASC`,
`S-1`, `S-2`, `S-3`, and `S-4`.

## 5. Run Inference

After training, run:

```bash
python train_real_cross_unet.py mode=predict station_name=KDASC
```

For another station, use the same `station_name` and `weather_source` used for
training:

```bash
python train_real_cross_unet.py \
  mode=predict \
  station_name=S-1 \
  weather_source=nwp
```

To load a specific checkpoint or write predictions somewhere else:

```bash
python train_real_cross_unet.py \
  mode=predict \
  checkpoint_path=./outputs/real_cross_unet/KDASC/checkpoints/real_cross_unet_KDASC.pt \
  prediction_path=./outputs/real_cross_unet/KDASC/predictions/custom_predictions.csv
```

The script reports MAE and RMSE in the original power units and writes all
forecast steps to CSV.

## 6. Common Overrides

Hydra command-line overrides can change any field in
[`conf/real_data.yaml`](conf/real_data.yaml):

```bash
python train_real_cross_unet.py \
  mode=train \
  station_name=S-2 \
  weather_source=nwp \
  pred_len=96 \
  seq_len=96 \
  seg_len=24 \
  max_epochs=20 \
  early_stop_patience=5 \
  batch_size=32 \
  d_model=128 \
  d_ff=256
```

Useful fields:

- `station_name`: `KDASC`, `S-1`, `S-2`, `S-3`, or `S-4`.
- `weather_source`: `nwp` or `satellite`; KDASC always uses `SWR`.
- `seq_len`: input history length. `96` means 24 hours at 15-minute cadence.
- `pred_len`: forecast horizon. `16` means 4 hours; `96` means 1 day.
- `early_stop_patience`: stop training after this many consecutive validation
  epochs without improvement. The default is `5`.
- `max_train_samples`, `max_valid_samples`, `max_predict_samples`: optional
  limits for quick tests.
- `output_dir`: directory for checkpoints and predictions.

## 7. Synthetic Data

The original synthetic example is still available:

```bash
python train_cross_unet.py
```

It fabricates a deterministic multivariate time series and is useful for
checking the training loop without reading the real CSV files.

## 8. Tensor Contract

Both scripts feed Cross_Unet with the same five tensors:

- `x_enc`: historical target/history channels, shape `(B, seq_len, target_channels)`.
- `w_enc`: forward-looking weather window, shape `(B, seq_len, weather_channels)`.
- `seq_w_nwp_hist`: historical weather window for correlation conditioning.
- `seq_x_hist`: full historical target/history window for correlation conditioning.
- `target`: future target/history channels; training loss uses the primary power
  channel, which is the last target channel.
