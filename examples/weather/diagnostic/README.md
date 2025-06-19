# Diagnostic models in PhysicsNeMo (precipitation)

This example contains code for training diagnostic models (models predicting an
additional variable from the atmospheric state) using PhysicsNeMo. It shows how to use
PhysicsNeMo to train a diagnostic model predicting precipitation from ERA-5 data.

## Installation

### Installing PhysicsNeMo

You need [PhysicsNeMo](https://github.com/NVIDIA/physicsnemo) installed on your Python
environment, installed with the `launch` extras. If installing from the PhysicsNeMo
repository, install PhysicsNeMo by running:

```bash
pip install .[launch]
```

in the PhysicsNeMo directory.

### Installing dependencies

You need to install the dependencies for the dataset download and the diagnostic model.

```bash
pip install -r requirements.txt
```

## Data Preparation

### Downloading ERA5 Data

This example requires two sets of ERA5 data:

1. Atmospheric state variables (input data)
2. Diagnostic variables (target data), i.e. precipitation

You can use the ERA5 downloader in the `dataset_download` example to obtain both datasets.
For each dataset, you'll need to:

1. Create a configuration file specifying the variables you want to download
2. Run the download script pointing to that configuration
3. Store the datasets in separate directories

For example:

```bash
# Download state variables
python dataset_download/start_mirror.py --config-name="config_34var.yaml"

# Download precipitation (create a new config with precipitation variable)
python dataset_download/start_mirror.py --config-name="config_precip.yaml"
```

### Data Format and Structure

The settings for the precipitation model training are in the
`config/diagnostic_precip.yaml` file. The ERA5 atmospheric state data is loaded from the
directory indicated in `sources.state_params.data_dir` and the target (precipitation)
data from `sources.diag_params.data_dir`. Both directories are assumed contain the
subdirectories `train/` (for training data) and `test/` (for validation data). These
should contain yearly data files:

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
- Channels: One per variable/pressure level combination

For more details on the data format, see the `ClimateDataSourceSpec` class in `physicsnemo.datapipes.climate.climate`.

Alphabetical order is used to determine the order of the files. The years you put in
`train/`, `test/` and `out_of_sample` respectively can differ from the example above,
but you should make sure that they are consistent between the state data and target
data. The training code does perform some sanity checks to ensure that the inputs are
consistent in time, but these should not be assumed to be foolproof.

Additionally, to use geopotential (effectively the terrain height) and the land-sea mask
(LSM) as predictors, you can set `datapipe.geopotential_filename` and
`datapipe.lsm_filename`, respectively. Alternatively you can delete these lines from the
configuration file, which will lead to the model being trained without these variables
as inputs.

## Input Channel Configuration

The `diagnostic_precip.yaml` configuration assumes an HDF5-format ERA5 training dataset
with variables specified in `sources.state_params.variables`.

Set `model.in_channels` to match your total input channels:

- Base: Length of `sources.state_params.variables`
- Additional channels:
  - Cosine zenith angle: +1 if `sources.state_params.use_cos_zenith == True`
  - Geopotential: +1 if `datapipe.geopotential_filename` is set
  - Land-sea mask: +1 if `datapipe.lsm_filename` is set
  - Lat/lon encoding: +4 if `datapipe.use_latlon == True`

## Training

### Start training from scratch

To start training of the model, go to the `scripts` directory and run

```bash
python train_diagnostic_precip.py
```

You can modify and add configuration settings from the command line using the
[Hydra](https://hydra.cc/) syntax.

### Continue training from checkpoint

This will continue training from the latest checkpoint:

```bash
python train_diagnostic_precip.py +training.load_epoch=latest
```

Alternatively, you can specify the epoch number instead of "latest". The checkpoint
directory is defined in `training.checkpoint_dir` in the configuration file.

### Multi-GPU training

Multiple GPUs will be detected automatically. You can start training using multiple GPUs
using:

```bash
mpirun -np <NUM_GPUS> python train_diagnostic_precip.py --config-name="diagnostic_precip.yaml"
```

where `NUM_GPUS` is the number of GPUs you're training on. Pass also the
`--allow-run-as-root` parameter to `mpirun` if running in a container as the root user.

## Testing

You can evaluate the model using out-of-sample data with the `eval_diagnostic_precip.py`
script that uses the same config file as the training:

```bash
python eval_diagnostic_precip.py +training.load_epoch=latest
```

This performs the testing with the data in the `out_of_sample` directory. It computes
the root-mean-square error for each point on the grid and saves the result in
`scripts/results/rmse.npy`. You can add more metrics by following the example of
`RMSECallback` in `eval_diagnostic_precip.py`.
