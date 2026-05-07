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

## Real datasets

This recipe is structured so that swapping in a real dataset (e.g. the
[AI-PVOD](https://huggingface.co/datasets/yujiaA/AI-PVOD) collection or the
upstream [PV-power](https://github.com/Z-Yh1/PV-power) station data) only
requires replacing the `SyntheticPVDataset` in `train_cross_unet.py` with a
`Dataset` that emits the same five tensors per item:
`x_enc`, `w_enc`, `seq_w_nwp_hist`, `seq_x_hist`, `target`.
