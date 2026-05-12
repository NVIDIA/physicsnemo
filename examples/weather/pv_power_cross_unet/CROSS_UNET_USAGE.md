<!-- SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Cross-Unet PV Power Forecasting Usage

This guide describes the real-data Cross-Unet workflow under `examples/weather/pv_power_cross_unet`. It covers data preparation, training, inference, and paper-reproduction result checks.

## 1. Environment

Install the example requirements in a PhysicsNeMo environment with PyTorch available. GPU training is recommended for the paper-size preset.

```bash
cd /home/horde/tmp/physicsnemo
pip install -r examples/weather/pv_power_cross_unet/requirements.txt
```

When using the `physicsnemo_26.03` container, verify CUDA before launching sweeps:

```bash
docker exec physicsnemo_26.03 bash -lc 'nvidia-smi && python -c "import torch; print(torch.cuda.is_available())"'
```

If `nvidia-smi` reports an NVML initialization error inside the container, stop and start the container, then verify CUDA again before resuming.

## 2. Data Preparation

The real-data workflow expects this directory layout:

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

Supported weather sources are:

- `nwp`: S-1 through S-4, with seven historical target channels and six NWP forecast channels.
- `satellite`: S-1 through S-4 and KDASC, using `SWR` as the forward-looking irradiance channel.
- `ai`: S-1 through S-4, mapped to AI station ids `0`, `4`, `7`, and `8`, using `ssrd_corrdiff` from issued AI forecast files.

The AI loader filters large missing-value sentinels such as `9.96921e36` and builds historical weather windows only from forecasts issued before the prediction period.

## 3. Training One Run

Use `train_real_cross_unet.py` with Hydra overrides. The default config is a short smoke configuration. Set `paper_preset=True` for the paper-scale Cross-Unet preset.

Example: train S-1 with NWP data for a 4-hour horizon.

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

The checkpoint is written under:

```text
<output_dir>/<station_name>/checkpoints/real_cross_unet_<station_name>.pt
```

For paper reproduction, `paper_preset=True` applies:

```text
d_model=252, d_ff=492, n_heads=4, e_layers=3, seg_len=12,
max_epochs=100, early_stop_patience=10, start_lr=1e-4,
seed=2021, normalization=minmax
```

## 4. Inference

Run prediction with the same data root and output directory used for training:

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

The prediction CSV contains:

```text
station,timestamp,horizon_step,prediction,target
```

Prediction and target values are saved on the original target scale for inspection. Paper-comparison MAE and MSE are computed on the normalized target range.

## 5. Paper-Reproduction Sweep

Use `run_paper_cross_unet.py` for Cross-Unet-only table reproduction. It supports Table 4 NWP, Table 5 satellite, and Table 6 AI weather cases.

Representative smoke sweep:

```bash
python examples/weather/pv_power_cross_unet/run_paper_cross_unet.py \
  --phase representative \
  --paper-targets-csv examples/weather/pv_power_cross_unet/paper_targets.csv
```

Full 65-case sweep:

```bash
python examples/weather/pv_power_cross_unet/run_paper_cross_unet.py \
  --phase full \
  --resume \
  --paper-targets-csv examples/weather/pv_power_cross_unet/paper_targets.csv \
  --metric-scale normalized
```

Run only selected cases from a CSV with `station,source,horizon_label` columns:

```bash
python examples/weather/pv_power_cross_unet/run_paper_cross_unet.py \
  --phase full \
  --cases-csv examples/weather/pv_power_cross_unet/paper_repro_failed_cases.csv \
  --resume \
  --paper-targets-csv examples/weather/pv_power_cross_unet/paper_targets.csv \
  --output-dir examples/weather/pv_power_cross_unet/outputs/paper_repro_retry \
  --metric-scale normalized
```

## 6. Results and Checks

The runner writes:

```text
<output_dir>/metrics.csv
<output_dir>/report.md
<output_dir>/runs/<source>/<station>/<horizon>/...
```

`metrics.csv` includes:

```text
station,source,horizon_label,seq_len,pred_len,seed,
mae,mse,r2,paper_mae,paper_mse,paper_r2,
pass_mae,pass_mse,pass_r2,param_count
```

Tolerance rules are:

- MAE pass: `mae <= paper_mae * 1.05`
- MSE pass: `mse <= paper_mse * 1.05`
- R2 pass: `r2 >= paper_r2 * 0.95`

For paper comparisons, use `--metric-scale normalized`. This keeps R2 unchanged under linear scaling and compares MAE/MSE in the same normalized scale used by the paper preprocessing.

## 7. Useful Verification Commands

```bash
pytest test/examples/weather/test_pv_power_cross_unet_real_data.py   test/experimental/models/pv_power/cross_unet/test_cross_unet.py -q

ruff check examples/weather/pv_power_cross_unet/train_real_cross_unet.py   examples/weather/pv_power_cross_unet/run_paper_cross_unet.py   test/examples/weather/test_pv_power_cross_unet_real_data.py   test/experimental/models/pv_power/cross_unet/test_cross_unet.py
```
