# ERA5-HRRR Dataset

The original StormCast is trained on the [HRRR
dataset](https://rapidrefresh.noaa.gov/hrrr/), conditioned on the [ERA5
dataset](https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5). The datapipe
in this example is tailored specifically for the domain and problem setting posed in the
[original StormCast preprint](https://arxiv.org/abs/2408.10958), namely a subset of HRRR
and ERA5 variables in a region over the Central US with spatial extent 1536km x 1920km.

A custom dataset object is defined in `datasets/data_loader_hrrr_era5.py`, which loads
temporally-aligned samples from HRRR and ERA5, interpolated to the same grid and
normalized appropriately. This data pipeline requires the HRRR and ERA5 data to abide by
a specific `zarr` format and for other datasets, you will need to [create a custom
datapipe](CUSTOM_DATASET.md). The following table lists the variables used to train
StormCast. In total, the model uses 26 ERA5 variables and 99 HRRR variables, along with
two static HRRR invariants (the land and water mask and orography).

## ERA5 Variables

| Parameter                           | Pressure Levels (hPa) | Height Levels (m) |
| ----------------------------------- | --------------------- | ----------------- |
| Zonal Wind (u)                      | 1000, 850, 500, 250   | 10                |
| Meridional Wind (v)                 | 1000, 850, 500, 250   | 10                |
| Geopotential Height (z)             | 1000, 850, 500, 250   | None              |
| Temperature (t)                     | 1000, 850, 500, 250   | 2                 |
| Humidity (q)                        | 1000, 850, 500, 250   | None              |
| Total Column of Water Vapour (tcwv) | Integrated            | -                 |
| Mean Sea Level Pressure (mslp)      | Surface               | -                 |
| Surface Pressure (sp)               | Surface               | -                 |

## HRRR Variables

| Parameter                         | Hybrid Model Levels (Index)                           | Height Levels (m) |
| --------------------------------- | ----------------------------------------------------- | ----------------- |
| Zonal Wind (u)                    | 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 20, 25, 30 | 10                |
| Meridional Wind (v)               | 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 20, 25, 30 | 10                |
| Geopotential Height (z)           | 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 20, 25, 30 | None              |
| Temperature (t)                   | 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 20, 25, 30 | 2                 |
| Humidity (q)                      | 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 20, 25, 30 | None              |
| Pressure (p)                      | 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 20         | None              |
| Max. Composite Radar Reflectivity | -                                                     | Integrated        |
| Mean Sea Level Pressure (mslp)    | -                                                     | Surface           |
| Orography                         | -                                                     | Surface           |
| Land/Water Mask                   | -                                                     | Surface           |
