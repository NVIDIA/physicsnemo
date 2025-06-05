# Pangu Weather for Global Weather Forecasting

A re-implementation of
[Pangu-Weather: A 3D High-Resolution Model for Fast and Accurate Global Weather Forecast](https://arxiv.org/abs/2211.02556)
in PhysicsNeMo.

## Problem Overview

Pangu-Weather is a transformer-based model that provides global weather forecasts at
0.25° resolution. The model uses a unique architecture that processes both surface-level
and upper-air variables, along with static geographical information
(land-sea mask, topography, and soil type). It generates predictions for multiple
atmospheric variables at both surface level and pressure levels.

## Dataset

The model requires a specific set of ERA5 variables organized into three components:

1. Surface variables (4 channels)
2. Upper-air variables (5 variables × 13 pressure levels = 65 channels)
3. Static geographical masks (3 channels)

### Download using ERA5 Downloader

1. First, ensure you have set up your CDS API key as described in the
`dataset_download` README.

2. Use a configuration file to specify the variables to download (user-defined):

```bash
python dataset_download/start_mirror.py --config-name="config_pangu.yaml"
```

The downloaded data will be organized as follows:

```bash
├── data_dir
    ├── train/
    │   ├── 1980.h5
    │   ├── 1981.h5
    │   └── ...
    ├── test/
    │   ├── 2017.h5
    │   └── ...
    ├── out_of_sample/
    │   └── 2018.h5
    └── stats/
        ├── global_means.npy
        └── global_stds.npy
```

Each HDF5 file contains:

- Data shape: (time_steps, channels, latitude, longitude)
- Latitude: 721 points (-90° to 90°)
- Longitude: 1440 points (-180° to 180°)
- Channels: Surface (4) + Upper-air (65) variables

### Required Variables

1. Surface Variables (4 channels):
   - 2m temperature
   - 10m u-component of wind
   - 10m v-component of wind
   - Mean sea level pressure

2. Upper-air Variables (5 variables × 13 pressure levels):
   - Temperature
   - U component of wind
   - V component of wind
   - Specific humidity
   - Geopotential

3. Static Masks (3 channels):
   - Land-sea mask
   - Soil type
   - Topography

## Installation

1. Install PhysicsNeMo with required extras:

    ```bash
    # If installing from the PhysicsNeMo repository
    pip install .[launch]
    ```

2. Install additional dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Install NVIDIA Apex (required for optimizer):

    ```bash
    git clone https://github.com/NVIDIA/apex
    cd apex
    pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation \
        --config-settings "--build-option=--cpp_ext" \
        --config-settings "--build-option=--cuda_ext" ./
    ```

## Training

Two training scripts are provided:

- `train_pangu_era5.py`: Full Pangu-Weather implementation
- `train_pangu_lite_era5.py`: Lightweight version for testing

To train the model on a single GPU:

```bash
# Full version
python train_pangu_era5.py

# Lite version
python train_pangu_lite_era5.py
```

### Multi-GPU Training

Data parallelism is supported with multi-GPU runs:

```bash
mpirun -np <num_GPUs> python train_pangu_era5.py
```

If running inside a docker container, add the `--allow-run-as-root` flag.

### Monitoring Training Progress

Training progress can be monitored using MLFlow:

```bash
mlflow ui -p 2458
```

View progress in a browser at [http://127.0.0.1:2458](http://127.0.0.1:2458)

## References

```text
@article{bi2023pangu,
  title={Pangu-Weather: A 3D High-Resolution Model for Fast and Accurate Global Weather Forecast},
  author={Bi, Kaifeng and Xie, Lingxi and Zhang, Hengheng and others},
  journal={arXiv preprint arXiv:2211.02556},
  year={2023}
}
```
