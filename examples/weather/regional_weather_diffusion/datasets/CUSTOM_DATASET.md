# Custom Datasets

When `xarray.XarrayDataset` is not sufficient, a more flexible option is to define a
custom dataset class. These datasets must follow the
`StormCastDataSource`/`StormCastDataset` interface defined in `datasets/dataset.py`.
Refer to the docstrings in that file for a full specification of what the functions must
accept and return. You can use the `datasets/mock.py` implementation as a minimal
synthetic example and the `xarray.XarrayDataset` implementation as a real-world example;
often it is sufficient to modify or subclass the latter to fit your needs.

Most custom datasets should subclass `StormCastDataset`, the map-style (`__getitem__`)
tier that most existing datasets use and that gets the classic PyTorch `DataLoader` for
free. `StormCastDataSource` is the lower-level contract it builds on -- implement it
directly only if your data source isn't naturally indexable (e.g. a streaming source);
see [Loading Strategy: `torch` vs `datapipes`](#loading-strategy-torch-vs-datapipes)
below.

The `StormCastDataset` subclass must implement the following methods:

* `__len__`: Returns the number of items in the dataset.
* `__getitem__`: Returns the data for the item at index `idx`, a dict with the following
  keys:
  * `"background"`: all per-sample conditioning -- including any past state a
    forecasting model needs as input.
  * `"state"`: the training target (in past versions of this recipe, a two-element
    `[past_state, target]` form was used; this is no longer supported and a dataset that
    returns one will raise an error at training time).
  * `"mask"` (optional): mask to determine elements to include in attention and loss
    calculation (see [Spatial Mask](#spatial-mask)).
  * `"scalar_conditions"` (optional): a 1-D array with shape `(number_of_conditions,)`,
    conditioning used globally for all grid points, only used by DiT models.
* `background_channels`: Returns a list with the names of each background channel. It is
  important that the length of this list correspond to the exact number of channels in
  `background`, as this is used to set the number of inputs used by the model. May
  return an empty list for models that do not utilize a background input (that is, no
  `"background"` in `model.diffusion_conditions`/`model.regression_conditions`).
* `state_channels`: Returns a list with the names of each state channel. As above, the
  length of this list must match the number of channels in `state`. The channel names
  are used in creating inference files and in per-variable loss logging.
* `image_shape`: Returns a 2-tuple with the spatial dimensions of the data. To train DiT
  models, the dimensions must be divisible by the patch size, while with U-Net they must
  be divisible by `2**(len(channel_mult) - 1)` (see [U-Net vs
  DiT](../README.md#u-net-vs-dit)).

The following methods are optional for training:

* `get_invariants`: Returns the invariants (conditions that are the same for each
  sample) for the dataset. To use them, `"invariant"` needs to be included in
  `model.diffusion_conditions` and/or `model.regression_conditions`.
* `scalar_condition_channels`: Returns a list with the names of the scalar condition
  channels. The corresponding 1D array of scalar conditions should be returned in the
  `scalar_conditions` key of the dict returned by `__getitem__`.
* `make_loader`: Override to supply a loading strategy other than the default PyTorch
  `DataLoader` (e.g. a PhysicsNeMo datapipe); see [Loading Strategy: `torch` vs
  `datapipes`](#loading-strategy-torch-vs-datapipes).
* `index_segments` / `sample_group_size`: For datasets that concatenate several
  independently-shardable sub-sources (e.g. geographic domains), or that pack multiple
  training samples into one drawable item (e.g. several crops sharing one expensive
  read). See the docstrings in `datasets/dataset.py`.

The following methods are optional and are not used for training:

* `denormalize_state`: Performs the transformation from normalized values to physical
  values for the state array. Used by `inference.py` to save outputs in physical units.
* `normalize_state`: The inverse of `denormalize_state`.
* `denormalize_background` and `normalize_background`: As above, for the background
  array. If `background` concatenates channel blocks from different sources with
  different statistics (as `HrrrEra5Dataset` does for ERA5 + previous-HRRR-timestep),
  normalize each block with its own statistics.
* `latitude` and `longitude`: Return the latitude and longitude grids, respectively.
  Saved in the `inference.py` outputs if available.

The `StormCastDataset` interface also supports lead-time aware training. To enable this:

* Set the `self.lead_time_steps` attribute of the dataset to the number of possible lead
  times (an integer larger than 0).
* Include a key `"lead_time_label"` with a corresponding integer value (range
  `0 <= lead_time_label < self.lead_time_steps`) in the dict returned by `__getitem__`.
  Lead time labels can currently only be used with U-Nets, but `scalar_conditions` may
  achieve the same effect with DiT models.

After you have implemented the custom dataset, create a configuration file in
`config/dataset`. This configuration file must have one special attribute, `name`. This
indicates a module in the `datasets` directory and a class to be used for the dataset.
For instance, specifying `name: xarray.XarrayDataset` will use the reference
Xarray-compatible dataset, found in `datasets/xarray.py`. Aside from the reserved
`loader` block (see below), the other parameters in the dataset configuration file will
be passed to the `params` object used to initialize the dataset and can be used to
specify, for example, the file system path from which the dataset is loaded.

## Spatial Mask

`__getitem__` may optionally return a `"mask"` key containing a float32 array with
values in `{0, 1}` (or boolean `{False, True}`), where `1`/`True` marks valid pixels and
`0`/`False` marks invalid or excluded pixels (e.g. outside sensor coverage, LAM sponge
zones, land-sea boundaries). The shape of the mask can be:

* `(1, H, W)` for a mask shared by all state channels
* `(C_state, H, W)` for per-channel masks
* `(C_state, 1, 1)` to mask out entire channels

When present, the training loop uses the mask as a per-pixel loss weight. For the DiT
architecture with `model.hyperparameters.use_nan_mask_tokens: true` (the default in
`reference_dit`), the mask is additionally collapsed across channels (a pixel is invalid
if it is invalid in any channel) and pooled to token granularity; invalid tokens are
then replaced with learned mask tokens inside every NATTEN attention block, and all
pixels of an invalid token are excluded from the loss. Per-channel masks therefore only
have a per-channel effect with U-Nets or with `use_nan_mask_tokens: false`. The dataset
is responsible for producing this mask (e.g. loading a pre-computed coverage map or
computing it from quality flags); for static masks, caching the array internally and
returning it for every sample is recommended.

## Loading Strategy: `torch` vs `datapipes`

Every dataset serves batches through `make_loader(spec)`, where `spec` is a `LoaderSpec`
describing what the trainer needs (batch size, worker count, device, an already
rank-sharded sample stream, etc. -- see `datasets/dataset.py`).
`StormCastDataset.make_loader` defaults to building the classic PyTorch `DataLoader`
(fork worker processes), which is what every dataset in this recipe used before and
requires no changes to keep using.

A dataset can opt into a different loading strategy by overriding `make_loader`.
`datasets/mock.py` shows both strategies side by side for exactly this reason:
`make_loader` dispatches on `spec.backend` to either the default PyTorch path or a
PhysicsNeMo [`datapipes`](../../../../physicsnemo/datapipes/) loader (threads instead of
forked processes, and, with `use_streams: true`, host-to-device copies and device-side
transforms overlapped with training compute on a side CUDA stream). Which strategy
actually runs is controlled entirely by config, not by which dataset class you pick.

When to use which:

* **`torch`** (default): works with every dataset in this recipe, including
  `HrrrEra5Dataset`, with no extra work. Workers are forked processes. The `torch`
  backend is limited to data preprocessing on the CPU, which may introduce bottlenecks
  with complex preprocessing loads. On the other hand, CPU preprocessing leaves all GPU
  capacity available for training. Use this option unless you have data loading
  bottlenecks that cannot be resolved by optimizing the CPU-based processing.
* **`datapipes`**: workers are threads rather than forked processes, and with
  `use_streams: true` host-to-device copies and device-side transforms overlap with
  training compute, so this option can be more performant for more complex data loaders
  that do preprocessing/regridding on the GPU. A dataset must implement it by overriding
  `make_loader`. Of the provided datasets, only `mock.MockDataset` implements
  `make_loader`, so using it with your own data means writing a datapipe `Reader` for
  it. See the [datapipes documentation](../../../../physicsnemo/datapipes/) for the
  available readers, transforms and loader options.

```yaml
dataset:
  name: mock.MockDataset
  loader:
    backend: "datapipes"  # or "torch" (the default)
    num_workers: 4         # optional override of training.num_data_workers
    prefetch_factor: 2
    pin_memory: true
    use_streams: true      # datapipes only
    # give each rank a slice of every index segment, not just a contiguous block
    shard_by_domain: true
```

Fields a chosen backend can't honor are simply ignored, so the same `loader` block is
valid regardless of `backend`. If training seems data-bound (see [Training Is
Slow](../README.md#training-is-slow)) and your dataset implements a `torch` loader,
`backend: "datapipes"` is worth testing.

If you are implementing a custom dataset and want it to support both backends, look at
`datasets/mock.py`'s `make_loader`/`_make_datapipe_loader` for the pattern: dispatch on
`spec.backend`, keep the PhysicsNeMo datapipe import inside the `"datapipes"` branch so
the `torch` path has no hard dependency on it, and make sure both backends draw from
`spec.sampler` and produce identical batch shapes for the same index (this recipe's test
suite checks exactly that in `test_loader_backend_batches`).
