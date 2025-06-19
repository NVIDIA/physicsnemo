# Adaptive Fourier Neural Operator (AFNO) for weather forecasting

This repository contains the code used for [FourCastNet: A Global Data-driven
High-resolution Weather Model using Adaptive Fourier Neural
Operators](https://arxiv.org/abs/2202.11214)

The code was developed by the authors of the preprint:
Jaideep Pathak, Shashank Subramanian, Peter Harrington, Sanjeev Raja,
Ashesh Chattopadhyay, Morteza Mardani, Thorsten Kurth, David Hall, Zongyi Li,
Kamyar Azizzadenesheli, Pedram Hassanzadeh, Karthik Kashinath, Animashree Anandkumar

## Problem overview

FourCastNet, short for Fourier Forecasting Neural Network, is a global data-driven
weather forecasting model that provides accurate short to medium-range global
predictions at 0.25∘ resolution. FourCastNet accurately forecasts high-resolution,
fast-timescale variables such as the surface wind speed, precipitation, and atmospheric
water vapor. It has important implications for planning wind energy resources,
predicting extreme weather events such as tropical cyclones, extra-tropical cyclones,
and atmospheric rivers. FourCastNet matches the forecasting accuracy of the ECMWF
Integrated Forecasting System (IFS), a state-of-the-art Numerical Weather Prediction
(NWP) model, at short lead times for large-scale variables, while outperforming IFS
for variables with complex fine-scale structure, including precipitation. FourCastNet
generates a week-long forecast in less than 2 seconds, orders of magnitude faster than
IFS. The speed of FourCastNet enables the creation of rapid and inexpensive
large-ensemble forecasts with thousands of ensemble-members for improving probabilistic
forecasting. We discuss how data-driven deep learning models such as FourCastNet are a
valuable addition to the meteorology toolkit to aid and augment NWP models.

FourCastNet is based on the [vision transformer architecture with Adaptive Fourier
Neural Operator (AFNO) attention](https://openreview.net/pdf?id=EXHG-A3jlM)

![Comparison between the FourCastNet and the ground truth (ERA5) for $u-10$ for
different lead times.](../../../docs/img/FourCastNet.gif)

## Dataset

The model is trained on a 20-channel subset of the ERA5 reanalysis data.
You can obtain the data in two ways:

### Option 1: Download using ERA5 Downloader (Recommended)

You can download the ERA5 data directly using the ERA5 downloader provided in the
`dataset_download` example. This gives you more control over the variables and time
period you want to download.

1. First, ensure you have set up your CDS API key as described in the
`dataset_download` README.

2. Use the provided configuration file or create your own:

```bash
python dataset_download/start_mirror.py --config-name="config_34var.yaml"
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
- Channels: One per variable/pressure level combination

### Option 2: Download from NERSC via Globus

The subset of the ERA5 training data that FCN was trained on is hosted at the
National Energy Research Scientific Computing Center (NERSC). For convenience
[it is available to all via Globus](https://app.globus.org/file-manager?origin_id=945b3c9e-0f8c-11ed-8daf-9f359c660fbd&origin_path=%2F~%2Fdata%2F).
You will need a Globus account and will need to be logged in to your account in order
to access the data. You may also need the [Globus Connect](https://www.globus.org/globus-connect)
to transfer data. The full dataset that this version of FourCastNet was trained on is
approximately 5TB in size.

## Model overview and architecture

Please refer to the [reference paper](https://arxiv.org/abs/2202.11214) to learn about
the model architecture.

## Getting Started

### Prerequisites

Install the required dependencies by running below:

```bash
pip install -r requirements.txt
```

To train the model, run

```bash
python train_era5.py train_dir=path_to_train_dir validation_dir=path_to_val_dir stats_dir=path_to_stats_dir
```

Progress can be monitored using MLFlow. Open a new terminal and navigate to the training
directory, then run:

```bash
mlflow ui -p 2458
```

View progress in a browser at <http://127.0.0.1:2458>

Data parallelism is also supported with multi-GPU runs. To launch a multi-GPU training,
run

```bash
mpirun -np <num_GPUs> python train_era5.py provide_script_parameters_here
```

If running inside a docker container, you may need to include the `--allow-run-as-root`
in the multi-GPU run command.

## References

If you find this work useful, cite it using:

```text
@article{pathak2022fourcastnet,
  title={Fourcastnet: A global data-driven high-resolution weather model
         using adaptive fourier neural operators},
  author={Pathak, Jaideep and Subramanian, Shashank and Harrington, Peter
          and Raja, Sanjeev and Chattopadhyay, Ashesh and Mardani, Morteza
          and Kurth, Thorsten and Hall, David and Li, Zongyi and Azizzadenesheli, Kamyar
          and Hassanzadeh, Pedram and Kashinath, Karthik and Anandkumar, Animashree},
  journal={arXiv preprint arXiv:2202.11214},
  year={2022}
}
```

ERA5 data was downloaded from the Copernicus Climate Change Service (C3S)
Climate Data Store.

```text
Hersbach, H., Bell, B., Berrisford, P., Biavati, G., Horányi, A., Muñoz Sabater, J.,
Nicolas, J., Peubey, C., Radu, R., Rozum, I., Schepers, D., Simmons, A., Soci, C.,
Dee, D., Thépaut, J-N. (2018): ERA5 hourly data on pressure levels from 1959 to present.
Copernicus Climate Change Service (C3S) Climate Data Store (CDS). 10.24381/cds.bd0915c6

Hersbach, H., Bell, B., Berrisford, P., Biavati, G., Horányi, A., Muñoz Sabater, J.,
Nicolas, J., Peubey, C., Radu, R., Rozum, I., Schepers, D., Simmons, A., Soci, C.,
Dee, D., Thépaut, J-N. (2018): ERA5 hourly data on single levels from 1959 to present.
Copernicus Climate Change Service (C3S) Climate Data Store (CDS). 10.24381/cds.adbb2d47
```

Other references:

[Adaptive Fourier Neural Operators:
Efficient Token Mixers for Transformers](https://openreview.net/pdf?id=EXHG-A3jlM)
