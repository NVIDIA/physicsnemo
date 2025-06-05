# ERA5 Data Downloader and Converter

This repository provides tools for downloading ERA5 datasets via the Climate Data Store
(CDS) API and processing them into formats suitable for machine learning. Users can
flexibly select different meteorological variables for their training dataset.

## Files Overview

1. `start_mirror.py` - Main script that initializes the `ERA5Mirror` class to handle
    downloading ERA5 data and converting it to Zarr and HDF5 formats.
2. `era5_mirror.py` - Contains the ERA5Mirror class responsible for downloading ERA5
    datasets from the CDS API and storing them in Zarr format.
3. Configuration files in `conf/`:
   - `config_tas.yaml` - Basic config for downloading surface temperature only
   - `config_34var.yaml` - Complete config used to train
   [FourCastNet](https://arxiv.org/abs/2202.11214)

## Setup Instructions

### 1. CDS API Key Setup

1. Create a free account on the
[Copernicus Climate Data Store](https://cds.climate.copernicus.eu/user/register)
2. Once logged in, go to your [user profile](https://cds.climate.copernicus.eu/user)
3. Click on the "Show API key" button
4. Create the file `~/.cdsapirc` with the following content:

   ```bash
   url: https://cds.climate.copernicus.eu/api/v2
   key: <your-api-key-here>
   ```

5. Make sure the file has the correct permissions: `chmod 600 ~/.cdsapirc`

### 2. Running the Download Script

1. Install required dependencies (consider using a virtual environment):

   ```bash
   pip install cdsapi xarray dask netCDF4 h5netcdf hydra-core
   ```

2. Choose or modify a configuration file in the `conf/` directory. Make sure to
update L31 in `start_mirror.py` to the appropriate configuration file
(by default, this is `config_34var.py`).

3. Run the main script:

   ```bash
   python start_mirror.py
   ```

## Configuration Parameters

The config files (`*.yaml`) contain the following parameters:

- `zarr_store_path` (str): Directory where intermediate Zarr datasets will be saved.
 These files are used as a checkpoint system and can be safely deleted after the
 HDF5 files are created.

- `hdf5_store_path` (str): Directory where final HDF5 datasets will be saved.
  This is the format used for training machine learning models.

- `dt` (int): Time resolution in hours. Must be a number that evenly divides 24
 (i.e., 1, 2, 3, 4, 6, 8, 12, or 24). Common choices are:
  - 6: Standard for weather prediction tasks (4 samples per day)
  - 24: Daily data
  Note: Smaller dt values will result in larger datasets.

- `start_train_year` (int): First year of training data. ERA5 data is available
  from 1940 onwards, but pre-1979 data is considered preliminary.

- `end_train_year` (int): Last year of training data

- `test_years` (list): Years to use for testing. These should not overlap with
  training years and are typically chosen to be consecutive years after the training
  period.

- `out_of_sample_years` (list): Years to use for out-of-sample validation.
  These should not overlap with training or test years and are typically the most
  recent years in the dataset.

- `compute_mean_std` (bool): Whether to compute and save global mean/std statistics.
 These statistics are computed only using the training data to prevent data leakage.

- `variables` (list): ERA5 variables to download. Can be specified in two formats:
  - Single-level variables as strings (e.g., "2m_temperature", "total_precipitation")
  - Pressure-level variables as [variable, level] pairs
    (e.g., ["t", 850] for temperature at 850hPa)
  - View the links below for the full list of variables and their descriptions:
    - [ERA5 Single Level Variables](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=overview)
    - [ERA5 Pressure Level Variables](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels?tab=overview)
  
  Common variable types:
  - Surface variables: "2m_temperature", "10m_u_component_of_wind"
  - Pressure level variables: temperature ("t"), geopotential ("z"),
  wind components ("u", "v")
  - Moisture variables: "total_column_water_vapour", relative humidity ("r")

### Important Notes and Limitations

1. **Variable Selection**:
   - The mirror currently does not support invariant fields
    (e.g., land-sea mask, orography)
   - Some variables may require special handling or preprocessing
   - Ensure variable names match exactly with ERA5's naming convention

2. **Data Resolution**:
   - Spatial resolution is fixed at 0.25° x 0.25° (approximately 25km at the equator)
   - Temporal resolution must evenly divide 24 hours
   - Data is stored on a regular latitude-longitude grid

3. **Storage Requirements**:
   - Each variable requires approximately 15GB per year at 6-hourly resolution
   - Total storage needs scale with: number of variables × number of years × (24/dt)

4. **Memory Considerations**:
   - Processing is done in monthly chunks to manage memory usage
   - Zarr format allows for efficient parallel processing and chunked storage

## Output Structure

### 1. Zarr Storage (`zarr_store_path`)

- Intermediate storage format
- One Zarr array per variable
- Efficient for parallel reading/writing
- Structure:

  ```bash
  zarr_store_path/
  ├── variable1.zarr/
  ├── variable2.zarr/
  └── variable3_pressure_level_850.zarr/
  ```

### 2. HDF5 Files (`hdf5_store_path`)

- Final format used for training
- Data split into train/test/out-of-sample directories
- One file per year
- Structure:

  ```bash
  hdf5_store_path/
  ├── train/
  │   ├── 1980.h5
  │   └── ...
  ├── test/
  │   ├── 2016.h5
  │   └── 2017.h5
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

## Notes

- The download process can take several hours to days depending on the number of variables
  and years requested
- If interrupted, the process can be safely restarted and will continue from where it
   left off
- Ensure sufficient disk space is available (multiple TB for full dataset)
- Keep your CDS API key confidential and never commit it to public repositories
