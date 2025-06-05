# DLWP Data Curation

This directory contains scripts for downloading and processing ERA5 data for the DLWP
(Deep Learning Weather Prediction) model. Two options are provided for data
acquisition:

## Installation

1. Set up CDS API access:
   - Create an account at [Copernicus](https://cds.climate.copernicus.eu/)
   - Install the CDS API key following instructions at
   [CDS API How To](https://cds.climate.copernicus.eu/api-how-to)
   - The key should be stored in `$HOME/.cdsapirc`

2. Install required packages:

    ```bash
    pip install cdsapi xarray netCDF4 scipy h5py
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

## Option 1: Using Dataset Download Example (Complete Dataset)

For training the full model, we recommend using the ERA5 downloader in the
`dataset_download` example:

```bash
python dataset_download/start_mirror.py --config-name="config_dlwp.yaml"
```

This will download all required variables and organize them in the correct format.
The user will need to specify the variables to download in the `config_dlwp.yaml` file.
 The user will also need to transform the data to the cubed-sphere grid using a
 modification of the `post_processing.py` script,
or by using the linear transform provided by TempestRemap manually.

## Option 2: Quick Start with Minimal Dataset

For testing or getting started quickly, you can use the simplified download script that
 fetches a minimal set of variables:

```bash
python data_download_simple.py
```

### Data Download Script (`data_download_simple.py`)

This script downloads a minimal set of ERA5 variables:

Variables downloaded:

1. Pressure level variables:
   - Temperature at 850hPa
   - Geopotential at 1000, 700, 500, and 300hPa

2. Single level variables:
   - Total column water
   - 2m temperature

Parameters:

- Time range: 3 days in January
- Temporal resolution: 6-hourly (00:00, 06:00, 12:00, 18:00 UTC)
- Years: 1979 (train), 2017 (test), 2018 (out-of-sample)
- Spatial resolution: 0.25° x 0.25° (default ERA5 grid)

Usage:

```bash
python data_download_simple.py
```

Output structure:

```bash
data/
├── train_temp/
│   ├── 1979_0.nc  # Temperature at 850hPa
│   ├── 1979_1.nc  # Geopotential at 1000hPa
│   └── ...
├── test_temp/
│   ├── 2017_0.nc
│   └── ...
└── out_of_sample_temp/
    ├── 2018_0.nc
    └── ...
```

### Post-Processing Script (`post_processing.py`)

This script transforms the downloaded data from lat-lon grid to cubed-sphere grid
format.

Prerequisites:

1. Generate mapping files using TempestRemap:

```bash
# Generate lat-lon mesh (721x1440 grid)
GenerateRLLMesh \
    --lat 721 \
    --lon 1440 \
    --file out_latlon.g \
    --lat_begin 90 \
    --lat_end -90 \
    --out_format Netcdf4

# Generate cubed-sphere mesh
GenerateCSMesh \
    --res 64 \  # Resolution parameter (adjust as needed)
    --file out_cubedsphere.g \
    --out_format Netcdf4

# Generate overlap meshes and mapping files
GenerateOverlapMesh \
    --a out_latlon.g \
    --b out_cubedsphere.g \
    --out overlap_latlon_cubedsphere.g \
    --out_format Netcdf4

GenerateOfflineMap \
    --in_mesh out_latlon.g \
    --out_mesh out_cubedsphere.g \
    --ov_mesh overlap_latlon_cubedsphere.g \
    --in_np 1 \
    --in_type FV \
    --out_type FV \
    --out_map map_LL721x1440_CS64.nc \  # This filename is referenced in post_processing.py
    --out_format Netcdf4
```

Usage:

```bash
python post_processing.py
```

Output:

- Converts NetCDF files to HDF5 format
- Transforms data to cubed-sphere grid
- Organizes data into train/test/out-of-sample splits

## Notes

- The simplified download script is intended for testing and development. For full
model training, use Option 1.
- The cubed-sphere resolution (64 in the example) can be adjusted based on your needs.
- Ensure sufficient disk space for both downloaded data and transformed outputs.
- The post-processing script expects the mapping file to be named
`map_LL721x1440_CS64.nc`. Adjust the filename in the script if using different parameters.
