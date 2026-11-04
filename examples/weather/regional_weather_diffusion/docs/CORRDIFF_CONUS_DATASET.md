# CorrDiff-ERA5-HRRR (CONUS) Dataset

[CorrDiff-ERA5-HRRR](https://huggingface.co/nvidia/corrdiff-era5-hrrr) downscales the
[ERA5 reanalysis](https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5) to
the 3 km [HRRR](https://rapidrefresh.noaa.gov/hrrr/) grid over the contiguous US. Its
training data is read by the custom dataset `corrdiff_conus.CorrDiffConusDataset`,
defined in `datasets/corrdiff_conus.py` and configured in
`config/dataset/corrdiff_conus.yaml`. It is used by the `replication_corrdiff_conus`
config; see [Flow Matching](../README.md#flow-matching) for the model and training
setup.

The dataset serves same-time pairs of ERA5 and HRRR, so it trains a pure downscaling
model (`model.diffusion_conditions: ["background"]`). For other downscaling datasets,
the [`XarrayDataset`](XARRAY_DATASET.md) is usually a better starting point; this
dataset is custom because it also randomly crops patches and randomly drops inputs.

## Data Layout

The dataset expects the following files under `dataset.data_path`:

```text
<data_path>
├── hrrr-conus/<year>.zarr         # "HRRR" (time, channel, y, x),
│                                  # "latitude" and "longitude" (y, x), and
│                                  # "valid_data" (time), a flag for complete samples
├── era5-conus/<year>.zarr         # "data" (time, channel, latitude, longitude)
│                                  # on a regular latitude-longitude grid
├── invariants/invariants_HRRR.nc  # static fields on the HRRR grid
└── stats/hrrr_era5_stats.json     # normalization statistics:
                                   # {"hrrr" | "era5" | "invariant":
                                   #     {variable: {"mean", "std"}}}
```

There is one HRRR and one ERA5 store for each year in `dataset.years`. The samples are
the times with valid HRRR data and matching ERA5 data, within
`dataset.time_range_train` or `dataset.time_range_valid`. The ERA5 grid must cover the
HRRR domain.

## Samples

`dataset.crop_window` selects the domain within the HRRR grid, 1024 × 1792 pixels by
default. Each sample contains the following normalized arrays:

- `"state"`: the HRRR variables.
- `"background"`: the ERA5 variables bilinearly interpolated to the HRRR grid,
  followed by the cosine of the solar zenith angle and the static fields in
  `dataset.invariant_variables` (mean and standard deviation of the elevation, and the
  land-sea mask). The static fields are part of the background, so the dataset does
  not provide separate invariants.
- `"scalar_conditions"`: the sine and cosine of the time of year, followed by one
  presence flag for each variable in `dataset.drop_variables`.

Training samples are random patches of shape `dataset.patch_shape` (256 × 256 by
default). The position of each validation patch depends only on the sample index, so
that validation is comparable between steps. With `dataset.patch_shape: null`, each
sample covers the full domain, as used for [inference](../README.md#inference).

During training, each ERA5 variable in `dataset.drop_variables` (`tcwv` and `sp` by
default) is independently replaced by its climatological mean, zero in normalized
units, with probability `dataset.drop_prob`, and its presence flag is set to 0. This
lets the model run without these inputs by setting their flags to 0. Validation and
inference never drop inputs.

## Variables

The ERA5 variables are the 26 variables of the
[ERA5 table](HRRR_ERA5_DATASET.md#era5-variables) used by StormCast. The HRRR variables
are the 99 variables of the [HRRR table](HRRR_ERA5_DATASET.md#hrrr-variables), without
the orography and land/water mask, which this dataset includes in the background
instead.
