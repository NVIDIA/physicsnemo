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

## Preparing the data files

The settings for the precipitation model training are in the
`config/diagnostic_precip.yaml` file. The ERA5 atmospheric state data is loaded from the
directory indicated in `sources.state_params.data_dir` and the target (precipitation)
data from `sources.diag_params.data_dir`. Both directories are assumed contain the
subdirectories `train/` (for training data) and `test/` (for validation data). These
should contain yearly data files:

```text
├── data_dir
    ├── train
    │   ├── 1980.h5
    │   ├── 1981.h5
    │   ├── 1982.h5
    │   ├── ...
    │   └── 2016.h5
    ├── test
    │   ├── 2017.h5
    ├── out_of_sample
    │   ├── 2018.h5
```

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

## Determining the input channels

The `diagnostic_precip.yaml` configuration file assumes an HDF5-format ERA5 training
dataset constructed at NVIDIA, containing the variables specified in
`sources.state_params.variables`. You can modify this parameter to specify different
inputs.

You should also set the number of input channels in `model.in_channels`. This should be
equal to the length of `sources.state_params.variables` plus all the additional
channels:

* if `sources.state_params.use_cos_zenith == True`, add 1
* if `datapipe.geopotential_filename` is set, add 1
* if `datapipe.lsm_filename` is set, add 1
* if `datapipe.use_latlon == True`, add 4

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
