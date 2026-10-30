# Xarray Dataset

`xarray.XarrayDataset` (in `datasets/xarray.py`) is a generic reader for regional
training data stored in any format readable by
[`xarray.open_dataset`](https://docs.xarray.dev/en/stable/generated/xarray.open_dataset.html),
such as Zarr or NetCDF. Which variables and time steps make up the `"state"` and
`"background"` of each sample is set in the config, so the same reader can be used to
train downscaling (CorrDiff-style), nowcasting (StormScope-style) and hybrid
(StormCast-style) models. It is also a good starting point for custom datasets, either
by subclassing or copying.

The template config is `config/dataset/xarray.yaml`, and the `reference_*` configs show
how to use it for each model type.

## Quick Start

To generate a small set of example files filled with random data, run the following in
the recipe directory:

```bash
python -c "from datasets.xarray import write_example_files; \
write_example_files('data', image_shape=(128, 128))"
```

This writes two data files, an invariants file and a normalization statistics file to
`data/`, matching the paths used in `config/dataset/xarray.yaml`. The grid size of
128×128 is large enough for the attention neighborhood of the reference DiT; the default
32×32 grid is only suitable for U-Nets or small DiT attention kernels. The example files
are useful as a template for formatting your own data and for checking that a config
works before pointing it to real data.

## File Format

### Data Files

The data can be split across any number of files, given as a list in `data_paths`. Paths
may be local or URLs (e.g. `s3://...`); extra arguments to `xarray.open_dataset`, such
as `engine` or `storage_options`, are given in `open_dataset_kwargs`.

Each file must contain:

* a `time` coordinate decodable to `numpy.datetime64`;
* one or more data arrays with dimensions `(time, channel, y, x)`, where the channel
  dimension has a string coordinate with the channel names.

All arrays must be on the same `(y, x)` grid. For downscaling, this means interpolating
the low-resolution input to the high-resolution grid when preparing the files. The grid
size must also suit the model (see [U-Net vs DiT](../README.md#u-net-vs-dit)). A grid
that doesn't fit can be trimmed with `crop_window`.

Dimensions are identified by their position, so only `time` needs to have a specific
name. For example, a file with an HRRR array and an ERA5 array regridded to the same
grid could look like this:

```text
<xarray.Dataset>
Dimensions:        (time: 24, hrrr_channel: 3, era5_channel: 2, y: 32, x: 32)
Coordinates:
  * time           (time) datetime64[ns] 2020-01-01 ... 2020-01-01T23:00:00
  * hrrr_channel   (hrrr_channel) <U3 't2m' 'u10' 'v10'
  * era5_channel   (era5_channel) <U4 'msl' 'tcwv'
Data variables:
    hrrr           (time, hrrr_channel, y, x) float32 ...
    era5           (time, era5_channel, y, x) float32 ...
    valid          (time) bool ...
```

The files may have different numbers of times and need not be in chronological order,
but each array must have the same channels, in the same order, and the same spatial
shape in every file. If the same time appears in more than one file, the last file in
`data_paths` is used and a warning is printed.

The optional boolean `valid` variable marks which times can be used; times where it is
`False` are ignored. This is a convenient way to exclude missing or corrupted data
without rewriting the files.

With chunked formats such as Zarr and NetCDF4, chunking each array with one chunk per
time step that includes all channels (as `write_example_files` does) lets each sample be
read with few requests.

### Invariants File

Time-independent data are read from a separate file given in `invariants_path`, which
may also be one of the data files. All variables in it must have dimensions `(y, x)`
matching the data arrays. It can contain:

* invariants given to the model as conditioning, e.g. orography or a land-sea mask,
  listed in `invariant_variables`; to use them, add `"invariant"` to
  `model.diffusion_conditions` and/or `model.regression_conditions`;
* latitude and longitude, named in `lat_lon_variables`, used by `inference.py`;
* a static mask, named in `mask_variable`, nonzero where the data are valid (e.g. on the
  satellite disk; see [Masking](#masking)).

### Normalization Statistics

Each channel is normalized as `(x - mean) / std` using statistics from the JSON file in
`stats_path`, organized by array and channel name. Invariant statistics go under the
`"invariant"` key:

```json
{
  "hrrr": {"t2m": {"mean": 285.1, "std": 12.3}, "u10": {"mean": 0.4, "std": 3.2}},
  "era5": {"msl": {"mean": 101300.0, "std": 950.0}},
  "invariant": {"orography": {"mean": 540.0, "std": 610.0}}
}
```

The statistics should be computed over the training period once and held fixed
afterwards for a given training project. Nonlinear transformations (see [Training Is
Unstable](../README.md#training-is-unstable--not-converging)) can be applied either when
preparing the files or by subclassing `XarrayDataset` and overriding
`in_place_normalize`, which normalizes each channel. For example, to log-transform
precipitation `tp`, save the following in e.g. `datasets/log_precip.py` and set
`dataset.name: log_precip.LogPrecipDataset`:

```python
import numpy as np
from datasets.xarray import XarrayDataset


class LogPrecipDataset(XarrayDataset):
    def in_place_normalize(self, x, array, var):
        if var == "tp":
            np.log1p(x, out=x)
        super().in_place_normalize(x, array, var)

    def denormalize_state(self, x):
        x = super().denormalize_state(x)
        i = self.state_channels().index("tp")
        x[..., i, :, :] = np.expm1(x[..., i, :, :])
        return x
```

The statistics of `tp` in the statistics file must then be computed from the
log-transformed values. `denormalize_state` is used by `inference.py` to save outputs in
physical units; `normalize_state`, `normalize_background` and `denormalize_background`
can be overridden in the same way if needed.

## Sample Composition

The contents of each sample are defined by _access patterns_: `state_access_patterns`
defines the `"state"` (the training target) and `background_access_patterns` the
`"background"` (the conditioning). Each maps an array name to the channels to read
(`null` for all) and the time steps to read, in units of `time_step_seconds` relative to
the sample time. For example, StormCast-style forecasting of three HRRR variables,
conditioned on the current HRRR state and on ERA5 at the target time, is configured as:

```yaml
state_access_patterns:
  hrrr: {channels: [t2m, u10, v10], time_steps: [1]}
background_access_patterns:
  hrrr: {channels: [t2m, u10, v10], time_steps: [0]}
  era5: {channels: null, time_steps: [1]}
```

Here time step 0 is the last past step and step 1 is the forecast target. To build a
sample, the selected channels of each array are read at each time step and concatenated
along the channel axis, in the order of the access patterns and then of the time steps.
With multiple time steps, all channels of the first step come first, followed by those
of the next step and so on.

Channel names (as returned by `state_channels` and `background_channels`) get a
`_t{step:+d}` suffix for steps other than 0. In the example above, the state channels
are `t2m_t+1`, `u10_t+1` and `v10_t+1`, and the background channels are `t2m`, `u10` and
`v10`, followed by `msl_t+1` and `tcwv_t+1`. These are the names to use e.g. in
`training.validation_plot_variables`.

The commented examples in `config/dataset/xarray.yaml` show the patterns for the other
model types:

| Model type              | `state_access_patterns`      | `background_access_patterns`                    |
| ----------------------- | ---------------------------- | ----------------------------------------------- |
| Downscaling (CorrDiff)  | high-res array at step `[0]` | low-res array at step `[0]`                     |
| Hybrid (StormCast)      | high-res array at step `[1]` | high-res array at `[0]`, low-res array at `[1]` |
| Nowcasting (StormScope) | observations at step `[1]`   | observations at steps `[-5, ..., 0]`            |
| Unconditional           | any                          | `{}` (no `"background"`)                        |

## Time Selection

A sample is used only if every time it accesses (the sample time plus each step in the
access patterns) exists in the data, is valid, and is within the time range.

* `time_step_seconds`: length of one time step; if `null`, inferred as the smallest
  interval between times in the data. Setting it explicitly is recommended, as the
  inferred value is wrong if the data have no consecutive times.
* `time_range_train` and `time_range_valid`: half-open `[start, end)` intervals of ISO
  8601 times for the training and validation sets, e.g.
  `["2020-01-01T00:00:00", "2021-01-01T00:00:00"]` for the year 2020; `null` for all
  times. If `time_range_valid` is `null`, it is the same as `time_range_train`.
* `exclude_times`: a list of individual times to skip, e.g. known bad samples. To
  exclude many times, the `valid` variable in the data files is usually more convenient,
  but `exclude_times` can be used without modifying the data files.

## Masking

When a mask is configured, the sample has a boolean `"mask"` of shape `(1, H, W)`,
`True` where valid, which the training loop uses to exclude invalid pixels from the loss
(see [Spatial Mask](CUSTOM_DATASET.md#spatial-mask)). There are two sources of masking,
which can be combined:

* `mask_variable`: a static mask from the invariants file, e.g. the coverage area of a
  satellite or radar.
* `nan_mask: true`: masks out pixels where any state or background channel is NaN or
  infinite in that sample, and replaces those values with 0. This is useful when the
  data have missing values that vary between samples.

## Other Options

* `crop_window`: `[[y_start, y_end], [x_start, x_end]]` crop applied to the data,
  invariants, latitude/longitude and mask, e.g. for training on a subdomain.
* `num_threads`: number of threads reading and normalizing each sample in parallel.
  Increasing it can help if the storage has high latency (e.g. bucket stores); see also
  [Training Is Slow](../README.md#training-is-slow).

### Caching

Setting `cache_location` enables an experimental cache of the Zarr chunks read during
training, useful when the storage is slow or the whole dataset fits in RAM. Other
formats than Zarr are not cached. The cache can be stored either:

* in RAM, with `cache_location: "memory"`. Each process, including each DataLoader
  worker, has a separate cache, so with this option it is best to set
  `training.num_data_workers=1` and parallelize reads with `num_threads` instead.
* on disk, by setting `cache_location` to a directory, preferably on node-local disk
  rather than a shared file system. The cache is shared by all processes using it and
  reused by later runs. It is not invalidated if the source data change, so clear it
  manually in that case.

The cache is never evicted, so it must be large enough for all data read on each node;
otherwise, the job will fail when the memory or disk runs out. Reads from sharded Zarr
v3 stores are only cached in memory, even when `cache_location` is a directory.
