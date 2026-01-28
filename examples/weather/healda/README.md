<!-- markdownlint-disable -->
# HealDA: Highlighting the importance of initial errors in end-to-end AI weather forecasts

<p align="center">
<img src="../../../docs/img/healda.png" width="800"/>
</p>

[📄 arXiv](https://arxiv.org/abs/2601.17636) · 📦 Checkpoints (coming soon)

---

## Problem Overview

Machine-learning (ML) weather models now rival leading numerical weather prediction (NWP) systems in medium-range skill. However, almost all still rely on NWP data assimilation (DA) to provide initial conditions, tying them to expensive infrastructure and limiting the practical speed and accuracy gains of ML.

**HealDA** is a global ML-based data assimilation system that maps satellite and conventional observations (microwave sounders, aircraft, radiosondes, surface stations) to a 1° atmospheric state on the HEALPix grid. HealDA analyses can initialize off-the-shelf ML forecast models (e.g., FourCastNet3, Aurora, FengWu) without fine-tuning, enabling end-to-end ML weather forecasting with less than one day loss of skill compared to ERA5 initialization.

---

## Installation

### Using uv

```bash
# 1. From PhysicsNeMo root directory
cd /path/to/physicsnemo

# 2. Create .venv and install PNM
uv sync

# 3. Activate the virtual environment
source .venv/bin/activate

# 4. Install earth2grid
uv pip install setuptools hatchling
uv pip install --no-build-isolation \
    "earth2grid @ https://github.com/NVlabs/earth2grid/archive/main.tar.gz"

# 5. Install healda dependencies
uv pip install -r examples/weather/healda/requirements.txt
```

### Using pip

```bash
# 1. Install PhysicsNeMo
pip install nvidia-physicsnemo

# 2. Install earth2grid
pip install setuptools hatchling
pip install --no-build-isolation https://github.com/NVlabs/earth2grid/archive/main.tar.gz

# 3. Install healda dependencies
pip install -r requirements.txt
```

> **Warning:** Include `--no-build-isolation` when installing earth2grid to avoid building against the wrong PyTorch version.

---

## Configuration

Create a `.env` file in the `examples/weather/healda/` directory with the following:

```bash
# Project paths
PROJECT_ROOT=/path/to/project

# Raw observation data (NC4 files downloaded from NOAA S3)
UFS_RAW_OBS_DIR=/path/to/raw_obs

# Processed observation data (parquet from ETL)
UFS_OBS_PATH=/path/to/processed_obs
# UFS_OBS_PROFILE=

# ERA5 HEALPix zarr (training targets)
V6_ERA5_ZARR=/path/to/era5_hpx.zarr
# V6_ERA5_ZARR_PROFILE=

# Land fraction mask
UFS_LAND_DATA_ZARR=/path/to/land_frac.zarr
# UFS_LAND_DATA_PROFILE=  
```

> **Note:** The `*_PROFILE` variables configure [rclone](https://rclone.org/) S3 profiles for cloud storage access. Leave empty for local paths.

---

## Data Preparation

HealDA requires preprocessed observation data and ERA5 target fields. We source observational data from the [NOAA Unified Forecast System (UFS) GEFSv13 Replay dataset](https://psl.noaa.gov/data/ufs_replay/) (NOAA, 2024).

See [`datasets/etl/`](datasets/etl/) for ETL scripts to prepare observation data into a parquet data format.

---

## Training

```bash
python train.py --name era5-v2-dense-noInfill-10M-fusion512-lrObs1e-4
```

This uses the paper configuration defined in `train.py`. See `python train.py --help` for options.

> **Resource Requirements:** Training takes approximately **8.3 days on 1 H100 node** (8 GPUs total) with batch size 1 per GPU.

---

## Inference

### Step 1: Generate DA Analysis (Initial Conditions)

The following produces analyses for all of 2022. `See inference_helpers.py` to configure inference. Inference only requires ~20GB of memory and can produce an analysis in under 1 second on a single H100.
```bash
python inference.py \
    /path/to/checkpoint.pt \
    --output_path /path/to/da_output.zarr \
    --context_start -21 \
    --context_end 3 \
    --time_frequency 6h \
    --num_samples -1 \
    --batch_gpu 1
```

### Step 2: Forecast from HealDA initial conditions

#### Installing FCN3 dependencies

FCN3 requires `earth2studio`. Recommended to install torch-harmonics with CUDA extensions for best performance:

```bash
# Using uv
export FORCE_CUDA_EXTENSION=1
uv pip install torch-harmonics==0.8.0 --no-build-isolation
uv pip install earth2studio[fcn3]

# Or using pip
export FORCE_CUDA_EXTENSION=1
pip install torch-harmonics==0.8.0 --no-build-isolation
pip install earth2studio[fcn3]
```

> **Note:** See [Earth2Studio docs](https://nvidia.github.io/earth2studio/userguide/about/install.html) for more information or installing other forecast models beyond FCN3.

#### Running forecasts

Use the DA output to initialize the FCN3 forecast model and create a 10-day forecast (40 6-hour steps):

```bash
python scripts/forecast.py \
    --init_path /path/to/da_output.zarr \
    --out_dir /path/to/forecast_output \
    --model FCN3 \
    --num_steps 40 \
    --num_ensemble 1 \
    --num_times 1
```

> **Note:** The forecast script:
> - Regrids HealDA analysis (HPX64) → 0.25° lat-lon for FCN3 input
> - Regrids FCN3 output (0.25° lat-lon) → HPX64 NEST format for storage

> **ERA5-initialized forecasts:** To create forecasts from ERA5 instead of DA output, run `inference.py` with `--use_analysis` flag to create an ERA5 zarr in the same format, then use that as `--init_path`.


### Step 3: Score Forecasts

Score forecasts against a reference dataset (also on HPX64 grid):

```bash
python scripts/score_forecast.py \
    --forecast_path /path/to/forecast.zarr \
    --reference_path /path/to/era5.zarr \
    --output_path /path/to/scores.nc
```

To plot the metrics:

```bash
python scripts/plot_panel.py \
    --stats /path/to/scores.nc \
    --labels "HealDA-initialized FCN3" \
    --metric crps \
    --output_path /path/to/plots/crps_comparison.pdf
```

See `python inference.py --help` and `python scripts/forecast.py --help` for full options.

---

## Citation

```bibtex
@misc{gupta2026healdahighlightingimportanceinitial,
  title={HealDA: Highlighting the importance of initial errors in end-to-end AI weather forecasts},
  author={Aayush Gupta and Akshay Subramaniam and Michael S. Pritchard and Karthik Kashinath and Sergey Frolov and Kelsey Lieberman and Christopher Miller and Nicholas Silverman and Noah D. Brenowitz},
  year={2026},
  eprint={2601.17636},
  archivePrefix={arXiv},
  primaryClass={physics.ao-ph},
  url={https://arxiv.org/abs/2601.17636},
}
```
