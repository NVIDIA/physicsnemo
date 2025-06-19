# Deep Learning Weather Prediction (DLWP) model for weather forecasting

This example is an implementation of the
[DLWP Cubed-sphere](https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/2021MS002502)
model. The DLWP model can be used to predict the state of the atmosphere given a previous
atmospheric state.  You can infer a 320-member ensemble set of six-week forecasts at 1.4°
resolution within a couple of minutes, demonstrating the potential of AI in developing
near real-time digital twins for weather prediction

## Problem overview

The goal is to train an AI model that can emulate the state of the atmosphere and
predict global weather over a certain time span. The Deep Learning Weather Prediction
(DLWP) model uses deep CNNs for globally gridded weather prediction. DLWP CNNs
directly map u(t) to its future state u(t+Δt) by learning from historical observations
of the weather, with Δt set to 6 hr

## Model overview and architecture

DLWP uses convolutional neural networks (CNNs) on a cubed sphere grid to produce global
forecasts. The latest DLWP model leverages a U-Net architecture with skip connections to
capture multi-scale processes. The model architecture is described in the following
papers

[Sub-Seasonal Forecasting With a Large Ensemble of Deep-Learning Weather Prediction Models](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2021MS002502)

[Improving Data-Driven Global Weather Prediction Using Deep Convolutional Neural Networks on a Cubed Sphere](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2020MS002109)

## Installation

### Prerequisites

1. Install PhysicsNeMo with required extras:

    ```bash
    pip install .[launch]
    ```

2. Install additional dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Install TempestRemap (required for coordinate transformation):

    ```bash
    git clone https://github.com/ClimateGlobalChange/tempestremap
    cd tempestremap
    mkdir build && cd build
    cmake ..
    make
    make install
    ```

## Dataset Preparation

There are two methods to prepare the training data for DLWP:

### Option 1: Using Dataset Download Example (Recommended)

This is the recommended approach for full model training. It provides more control over
variable selection and time periods.

1. First, ensure you have set up your CDS API key as described in the
`dataset_download` README.

2. Use the provided DLWP configuration:

    ```bash
    python dataset_download/start_mirror.py --config-name="config_dlwp.yaml"
    ```

    The configuration includes:

    - 7 ERA5 variables mapped to cubed-sphere grid
    - Resolution: 64x64 grid cells per face
    - Years: 1980-2015 (training), 2016-2017 (validation), 2018 (testing)
    - Temporal resolution: 6-hourly

3. Transform the downloaded data to cubed-sphere format:

    ```bash
    cd data_curation
    python post_processing.py --input-dir /path/to/downloaded/data --output-dir /path/to/output
    ```

### Option 2: Quick Start with Minimal Dataset

For testing or development, you can use the simplified data preparation scripts
in the `data_curation` directory:

1. Download a minimal set of ERA5 variables:

    ```bash
    cd data_curation
    python data_download_simple.py
    ```

2. Process the downloaded data:

    ```bash
    python post_processing.py
    ```

    See the `data_curation/README.md` for detailed instructions and parameters.

### Data Format

The final dataset should be organized as follows:

```bash
data_dir/
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

- Shape: (time_steps, channels, faces, height, width)
- Faces: 6 (cubed-sphere)
- Height/Width: 64 (resolution parameter)
- Channels: 7 (atmospheric variables)

## Training

To train the model, run:

```bash
python train_dlwp.py
```

### Multi-GPU Training

For distributed training:

```bash
mpirun -np <NUM_GPUS> python train_dlwp.py
```

Note: Add `--allow-run-as-root` if running in a container as root.

### Monitoring Training

Progress can be monitored using MLFlow:

```bash
mlflow ui -p 2458
```

## References

[Sub-Seasonal Forecasting With a Large Ensemble of Deep-Learning Weather Prediction Models](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2021MS002502)

[Arbitrary-Order Conservative and Consistent Remapping and a Theory of Linear Maps: Part 1](https://journals.ametsoc.org/view/journals/mwre/143/6/mwr-d-14-00343.1.xml)

[Arbitrary-Order Conservative and Consistent Remapping and a Theory of Linear Maps, Part 2](https://journals.ametsoc.org/view/journals/mwre/144/4/mwr-d-15-0301.1.xml)
