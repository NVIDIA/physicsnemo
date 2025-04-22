# ERA5 Data Downloader and Converter

This repository provides tools for downloading ERA5 datasets via the Climate Data Store
(CDS) API and processing them into formats suitable for machine learning. Users can
flexibly select different meteorological variables for their training dataset.

## Files Overview

1. `start_mirror.py` starts by initializing the `ERA5Mirror` class, which will take care
    of downloading the ERA5 data and saving it in Zarr format as well as HDF5 conversion.
2. `era5_mirror.py` - Contains the ERA5Mirror class responsible for downloading ERA5
    datasets from the CDS API and storing them in Zarr format.
3. `conf/config_tas.yaml` - Configuration file for `start_mirror.py`, which dictates the
    parameters for downloading and processing. This config file will just download the
    surface temperature variable however if you would like a more complete dataset such
    as the one used to train [FourCastNet](https://arxiv.org/abs/2202.11214),
    please use `conf/config_34var.yaml`.

## How to Use

1. Make sure you have the CDS API key setup following
    [these instructions](https://cds.climate.copernicus.eu/how-to-api).
2. Run the main script, `python start_mirror.py`. This will perform all actions to
    generate HDF5 files needed for training. First it will download and save all
    variables as Zarr arrays. This may take a substantial amount of time. If the process
    gets interrupted, it saves the state of the download process, and you can restart.
    Restarting while changing date ranges for download may cause issues. Therefore,
    restarting should be done while keeping the same date configs. After the download is
    complete, the desired variables will be saved as HDF5 files in a standardized format
    that can be used by the datapipes seen in forecast training recipes such as
    `fcn_afno`.

## Configuration File

The config files contain several configurations you can modify,

- `zarr_store_path`: Path where Zarr datasets will be saved.
- `hdf5_store_path`: Path where HDF5 datasets will be saved.
- `dt`: Time resolution in hours.
- `start_train_year`: Start year for training data.
- `end_train_year`: End year for training data.
- `test_years`: List of years for testing data.
- `out_of_sample_years`: List of years for out-of-sample data.
- `compute_mean_std`: Whether or not to compute global mean and standard deviation.
- `variables`: ERA5 variables to be downloaded.

## Note

Make sure to handle your CDS API key with care. Always keep it confidential and avoid
pushing it to public repositories.
