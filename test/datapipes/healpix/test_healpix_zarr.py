# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib.util
import random
import warnings

import pytest

from physicsnemo.distributed import DistributedManager
from test.conftest import requires_module
from test.datapipes.healpix.conftest import assert_shard_dataloaders

omegaconf = pytest.importorskip("omegaconf")
np = pytest.importorskip("numpy")
xr = pytest.importorskip("xarray")
zarr = pytest.importorskip("zarr")


def test_object_store_path_helpers(tmp_path):
    """Zarr datasets may live on object stores (e.g. s3://, or fsspec-chained paths
    like simplecache::s3://); such paths are recognized by path syntax alone and
    skip the local filesystem existence check performed for plain local paths.
    This behavior is unique to the Zarr-backed datapipes and doesn't require the
    NFS test dataset, so it can run unconditionally.
    """
    from physicsnemo.datapipes.healpix.base_timeseries_dataset_zarr import (
        _check_availability,
        _is_object_store_path,
    )

    assert _is_object_store_path("s3://bucket/data.zarr")
    assert _is_object_store_path("simplecache::s3://bucket/data.zarr")
    assert not _is_object_store_path(str(tmp_path / "local.zarr"))

    # an existing local path passes without needing fsspec
    existing_path = tmp_path / "exists.zarr"
    existing_path.mkdir()
    _check_availability(str(existing_path))

    # a missing local path raises FileNotFoundError
    with pytest.raises(FileNotFoundError, match=("Dataset not found at")):
        _check_availability(str(tmp_path / "missing.zarr"))

    # object store paths bypass the local existence check as long as fsspec
    # is installed, even when the remote resource doesn't actually exist
    if importlib.util.find_spec("fsspec"):
        _check_availability("s3://bucket-that-does-not-exist/missing.zarr")
    else:
        with pytest.raises(
            ImportError, match=("fsspec is required to access object store paths")
        ):
            _check_availability("s3://bucket-that-does-not-exist/missing.zarr")


@requires_module("omegaconf")
@requires_module("netCDF4")
def test_TimeSeriesDataset_initialization(
    dataset_path,
    scaling_dict,
    pytestconfig,
):
    from physicsnemo.datapipes.healpix.timeseries_dataset_zarr import (
        TimeSeriesDatasetZarr,
    )

    input_variables = ["t2m0", "t850", "z500"]

    bad_start_date = "1900-01-01"
    valid_start_date = "1979-01-01"
    bad_end_date = "2000-12-31"
    valid_end_date = "1979-01-02"

    time_da = xr.open_zarr(dataset_path).time

    # check for failure of invalid dataset path
    # Check if fsspec is available for object store paths, optional dependency
    if not importlib.util.find_spec("fsspec"):
        # If fsspec is not available, expect an ImportError
        with pytest.raises(
            ImportError, match=("fsspec is required to access object store paths")
        ):
            timeseries_ds = TimeSeriesDatasetZarr(
                dataset_path="s3://physicsnemo-data/datasets/healpix/healpix.zarr",
                scaling=scaling_dict,
                input_variables=input_variables,
            )

    # If path doesn't exist, expect a FileNotFoundError
    with pytest.raises(FileNotFoundError, match=("Dataset not found at")):
        timeseries_ds = TimeSeriesDatasetZarr(
            dataset_path="/boguspath.zarr",
            scaling=scaling_dict,
            input_variables=input_variables,
        )

    # check for failure of timestep not being a multiple of datatime step
    with pytest.raises(
        ValueError, match=("'time_step' must be a multiple of 'data_time_step' ")
    ):
        timeseries_ds = TimeSeriesDatasetZarr(
            dataset_path=dataset_path,
            data_time_step="2h",
            time_step="5h",
            scaling=scaling_dict,
            input_variables=input_variables,
            forecast_init_times=time_da[:2],
            batch_size=1,
        )

    # check for failure of gap not being a multiple of datatime step
    with pytest.raises(
        ValueError, match=("'gap' must be a multiple of 'data_time_step' ")
    ):
        timeseries_ds = TimeSeriesDatasetZarr(
            dataset_path=dataset_path,
            data_time_step="2h",
            time_step="6h",
            gap="3h",
            scaling=scaling_dict,
            start_date=valid_start_date,
            end_date=valid_end_date,
            input_variables=input_variables,
        )

    # check for failure of invalid scaling variable on input
    invalid_scaling = omegaconf.DictConfig(
        {
            "bogosity": {"mean": 0, "std": 42},
        }
    )
    with pytest.raises(KeyError, match=("Input channels ")):
        timeseries_ds = TimeSeriesDatasetZarr(
            dataset_path=dataset_path,
            data_time_step="3h",
            time_step="6h",
            scaling=invalid_scaling,
            forecast_init_times=time_da[:10],
            input_variables=input_variables,
            batch_size=1,
        )

    # check for failure when a requested variable isn't present in the dataset
    # this validation is specific to the Zarr-backed dataset, which selects
    # channels by name directly out of the store rather than relying on a
    # pre-sliced xarray Dataset
    with pytest.raises(
        KeyError,
        match=("Requested Input, coupled, or output variables not found in dataset"),
    ):
        timeseries_ds = TimeSeriesDatasetZarr(
            dataset_path=dataset_path,
            scaling=scaling_dict,
            start_date=valid_start_date,
            end_date=valid_end_date,
            input_variables=input_variables + ["DoesntExist"],
        )

    # check for warning when the configured data_time_step doesn't match the
    # dataset's native cadence; this cross-check against the stored time
    # coordinate only exists in the Zarr-backed dataset
    warnings.filterwarnings("error")
    with pytest.raises(UserWarning, match=("doesn't match configuration dt")):
        timeseries_ds = TimeSeriesDatasetZarr(
            dataset_path=dataset_path,
            scaling=scaling_dict,
            data_time_step="6h",
            time_step="6h",
            start_date=valid_start_date,
            end_date=valid_end_date,
            input_variables=input_variables,
        )
    warnings.resetwarnings()

    # check for warning on batch size > 1 and forecast mode
    warnings.filterwarnings("error")
    with pytest.raises(
        UserWarning,
        match=(
            "providing 'forecast_init_times' to TimeSeriesDataset requires `batch_size=1`"
        ),
    ):
        timeseries_ds = TimeSeriesDatasetZarr(
            dataset_path=dataset_path,
            scaling=scaling_dict,
            batch_size=2,
            forecast_init_times=time_da[:2],
            input_variables=input_variables,
        )
    warnings.resetwarnings()

    # check for no dates provided
    with pytest.raises(
        ValueError,
        match=("Either start and end date or forecast_init_times must be provided"),
    ):
        timeseries_ds = TimeSeriesDatasetZarr(
            dataset_path=dataset_path,
            scaling=scaling_dict,
            batch_size=2,
            input_variables=input_variables,
        )

    # check for out of range dates
    warnings.filterwarnings("error")
    with pytest.raises(
        UserWarning,
        match=(f"Start date {bad_start_date} is before first available date"),
    ):
        timeseries_ds = TimeSeriesDatasetZarr(
            dataset_path=dataset_path,
            scaling=scaling_dict,
            batch_size=2,
            start_date=bad_start_date,
            end_date=valid_end_date,
            input_variables=input_variables,
        )

    with pytest.raises(
        UserWarning, match=(f"End date {bad_end_date} is after last available date")
    ):
        timeseries_ds = TimeSeriesDatasetZarr(
            dataset_path=dataset_path,
            scaling=scaling_dict,
            batch_size=2,
            start_date=valid_start_date,
            end_date=bad_end_date,
            input_variables=input_variables,
        )
    warnings.resetwarnings()

    # test no scaling
    timeseries_ds = TimeSeriesDatasetZarr(
        dataset_path=dataset_path,
        start_date=valid_start_date,
        end_date=valid_end_date,
        input_variables=input_variables,
    )
    assert isinstance(timeseries_ds, TimeSeriesDatasetZarr)

    timeseries_ds = TimeSeriesDatasetZarr(
        dataset_path=dataset_path,
        scaling=scaling_dict,
        start_date=valid_start_date,
        end_date=valid_end_date,
        input_variables=input_variables,
    )
    assert isinstance(timeseries_ds, TimeSeriesDatasetZarr)

    timeseries_ds = TimeSeriesDatasetZarr(
        dataset_path=dataset_path,
        scaling=scaling_dict,
        batch_size=1,
        start_date=valid_start_date,
        end_date=valid_end_date,
        input_variables=input_variables,
    )
    assert isinstance(timeseries_ds, TimeSeriesDatasetZarr)

    timeseries_ds = TimeSeriesDatasetZarr(
        dataset_path=dataset_path,
        scaling=scaling_dict,
        batch_size=1,
        start_date=valid_start_date,
        end_date=valid_end_date,
        data_time_step="3h",
        time_step="6h",
        input_variables=input_variables,
    )
    assert isinstance(timeseries_ds, TimeSeriesDatasetZarr)

    # `start_date`/`end_date` can also be integer positional indices into the
    # time array rather than dates (use a nonzero start_date since 0 is falsy
    # and would take the same code path as "no start date provided")
    timeseries_ds = TimeSeriesDatasetZarr(
        dataset_path=dataset_path,
        scaling=scaling_dict,
        start_date=2,
        end_date=5,
        input_variables=input_variables,
    )
    assert timeseries_ds.start_index == 2
    assert timeseries_ds.total_samples == 3

    # check for failure when a requested output variable has no scaling
    # entry, even though it's present in the dataset (and in the input
    # scaling, if it happens to also be an input variable)
    scaling_missing_target = omegaconf.DictConfig(
        {k: v for k, v in scaling_dict.items() if k != "z1000"}
    )
    with pytest.raises(KeyError, match=("Target channels ")):
        timeseries_ds = TimeSeriesDatasetZarr(
            dataset_path=dataset_path,
            scaling=scaling_missing_target,
            start_date=valid_start_date,
            end_date=valid_end_date,
            input_variables=["t2m0"],
            output_variables=["t2m0", "z1000"],
        )

    # check for failure when a requested constant variable exists in the
    # dataset but has no corresponding scaling entry
    scaling_missing_constant = omegaconf.DictConfig(
        {k: v for k, v in scaling_dict.items() if k != "z"}
    )
    with pytest.raises(KeyError, match=("Constant variables ")):
        timeseries_ds = TimeSeriesDatasetZarr(
            dataset_path=dataset_path,
            scaling=scaling_missing_constant,
            start_date=valid_start_date,
            end_date=valid_end_date,
            input_variables=input_variables,
            constant_variables=["lsm", "z"],
        )


def test_TimeSeriesDataset_missing_time(tmp_path):
    """The Zarr-backed dataset validates that the store has a `time`
    variable before doing any time-based indexing; this check is unique to
    the Zarr path (the classic path always derives its dataset from a
    pre-opened xarray Dataset that already has a time dimension), so build a
    minimal Zarr store without a time coordinate to exercise it directly.
    """
    from physicsnemo.datapipes.healpix.timeseries_dataset_zarr import (
        TimeSeriesDatasetZarr,
    )

    no_time_ds = xr.Dataset(
        data_vars={
            "inputs": (
                ("t", "channel_in", "face", "height", "width"),
                np.zeros((3, 1, 1, 2, 2), dtype="float32"),
            ),
            "targets": (
                ("t", "channel_out", "face", "height", "width"),
                np.zeros((3, 1, 1, 2, 2), dtype="float32"),
            ),
        },
        coords={
            "channel_in": ["z500"],
            "channel_out": ["z500"],
        },
    )
    dataset_path = tmp_path / "no_time.zarr"
    no_time_ds.to_zarr(dataset_path)

    with pytest.raises(KeyError, match=("Dataset missing time")):
        TimeSeriesDatasetZarr(
            dataset_path=str(dataset_path),
            input_variables=["z500"],
            start_date="1979-01-01",
            end_date="1979-01-02",
        )


@requires_module("omegaconf")
@requires_module("netCDF4")
@requires_module("numpy")
@requires_module("zarr")
def test_TimeSeriesDataset_get_constants(dataset_path, scaling_dict, pytestconfig):
    from physicsnemo.datapipes.healpix.timeseries_dataset_zarr import (
        TimeSeriesDatasetZarr,
    )

    input_variables = ["z500", "z1000"]
    constant_variables = ["lsm"]

    # open our test dataset
    zarr_ds = xr.open_zarr(dataset_path)
    constant_indices = [
        int(np.where(zarr_ds.channel_c[:] == ch)[0][0]) for ch in constant_variables
    ]

    # Constant that isn't in dataset
    with pytest.raises(KeyError, match=("Requested constants not found in dataset")):
        timeseries_ds = TimeSeriesDatasetZarr(
            dataset_path=dataset_path,
            input_variables=input_variables,
            batch_size=1,
            scaling=scaling_dict,
            constant_variables=["DoesntExist"],
            forecast_init_times=zarr_ds.time[:2],
        )

    timeseries_ds = TimeSeriesDatasetZarr(
        dataset_path=dataset_path,
        input_variables=input_variables,
        scaling=scaling_dict,
        constant_variables=None,
        forecast_init_times=zarr_ds.time[:2],
        batch_size=1,
    )

    assert timeseries_ds.get_constants() is None

    timeseries_ds = TimeSeriesDatasetZarr(
        dataset_path=dataset_path,
        input_variables=input_variables,
        scaling=scaling_dict,
        constant_variables=constant_variables,
        forecast_init_times=zarr_ds.time[:2],
        batch_size=1,
    )

    # constants are reshaped
    expected = np.transpose(
        zarr_ds.constants.values[constant_indices], axes=(1, 0, 2, 3)
    )
    outvar = timeseries_ds.get_constants()
    assert np.array_equal(
        expected,
        outvar,
    )
    zarr_ds.close()


@requires_module("omegaconf")
@requires_module("netCDF4")
def test_TimeSeriesDataset_len(dataset_path, scaling_dict, pytestconfig):
    from physicsnemo.datapipes.healpix.timeseries_dataset_zarr import (
        TimeSeriesDatasetZarr,
    )

    input_variables = ["z500", "z1000"]
    batch_size = 2

    # open our test dataset
    zarr_ds = xr.open_zarr(dataset_path)

    # check forecast mode
    init_times = random.randint(1, zarr_ds.time.shape[0])
    timeseries_ds = TimeSeriesDatasetZarr(
        dataset_path=dataset_path,
        input_variables=input_variables,
        scaling=scaling_dict,
        batch_size=1,
        forecast_init_times=zarr_ds.time[:init_times],
    )
    assert len(timeseries_ds) == init_times

    # get the last index that's evenly divisible by 3 (9h / 3h)
    last_index = (zarr_ds.time.shape[0] // 3) * 3 - 1

    # check train mode
    timeseries_ds = TimeSeriesDatasetZarr(
        dataset_path=dataset_path,
        data_time_step="3h",
        time_step="9h",
        input_variables=input_variables,
        scaling=scaling_dict,
        batch_size=batch_size,
        start_date=zarr_ds.time[0].values,
        end_date=zarr_ds.time[last_index - 1].values,
    )
    # Window length of 3 for one sample size
    assert len(timeseries_ds) == (zarr_ds.time.shape[0] - 3) // batch_size

    # drop incomplete last window
    timeseries_ds = TimeSeriesDatasetZarr(
        dataset_path=dataset_path,
        data_time_step="3h",
        time_step="9h",
        input_variables=input_variables,
        scaling=scaling_dict,
        batch_size=batch_size,
        drop_last=True,
        start_date=zarr_ds.time[0].values,
        end_date=zarr_ds.time[last_index - 1].values,
    )
    assert len(timeseries_ds) == (zarr_ds.time.shape[0] - 4) // batch_size

    zarr_ds.close()
    DistributedManager.cleanup()


@requires_module("omegaconf")
@requires_module("netCDF4")
@requires_module("numpy")
def test_TimeSeriesDataset_get(dataset_path, scaling_double_dict, splits, pytestconfig):
    from physicsnemo.datapipes.healpix.timeseries_dataset_zarr import (
        TimeSeriesDatasetZarr,
    )

    input_variables = ["z500", "z1000"]
    constant_variables = ["lsm"]

    # open our test dataset
    zarr_ds = xr.open_zarr(dataset_path)
    time_da = zarr_ds.time.values

    batch_size = 2
    timeseries_ds = TimeSeriesDatasetZarr(
        dataset_path=dataset_path,
        input_variables=input_variables,
        scaling=scaling_double_dict,
        batch_size=batch_size,
        start_date=splits["train_date_start"],
        end_date=splits["test_date_start"],
    )

    # check for invalid index
    invalid_idx = len(zarr_ds.targets) + 1
    with pytest.raises(
        IndexError, match=(f"index {invalid_idx} out of range for dataset with length")
    ):
        inputs, targets = timeseries_ds[invalid_idx]

    inputs, targets = timeseries_ds[0]

    # make sure number of targets is correct
    assert len(targets) == batch_size

    # check target data
    # need to transpose
    targets_expected = (
        zarr_ds.targets[batch_size]
        .transpose("face", "channel_out", "height", "width")
        .sel(channel_out=input_variables)
    )

    # scale target data
    targets_expected = targets_expected.to_numpy() / 2
    assert np.array_equal(targets[0][:, 0, :, :], targets_expected)

    # check for negative index
    inputs, targets = timeseries_ds[-1]
    targets_expected = zarr_ds.targets[12].transpose(
        "face", "channel_out", "height", "width"
    )
    targets_expected = targets_expected.to_numpy() / 2

    # we're not dropping incomplete elements by default
    assert len(targets) == 0

    # this time dropping incomplete so that we get a full sample sample
    timeseries_ds = TimeSeriesDatasetZarr(
        dataset_path=dataset_path,
        input_variables=input_variables,
        scaling=scaling_double_dict,
        batch_size=batch_size,
        drop_last=True,
        start_date=time_da[0],
        end_date=time_da[-1],
    )

    inputs, targets = timeseries_ds[-1]
    targets_expected = (
        zarr_ds.targets[-1 - batch_size]
        .transpose("face", "channel_out", "height", "width")
        .sel(channel_out=input_variables)
    )
    targets_expected = targets_expected.to_numpy() / 2
    assert np.array_equal(targets[0][:, 0, :, :], targets_expected)

    # With insolation we get 1 extra tensor
    timeseries_ds = TimeSeriesDatasetZarr(
        dataset_path=dataset_path,
        input_variables=input_variables,
        scaling=scaling_double_dict,
        batch_size=batch_size,
        drop_last=True,
        add_insolation=True,
        start_date=time_da[0],
        end_date=time_da[-1],
    )

    # verify underlying data doesn't change
    targets_expected = (
        zarr_ds.targets[2]
        .transpose("face", "channel_out", "height", "width")
        .sel(channel_out=input_variables)
    )
    targets_expected = targets_expected.to_numpy() / 2
    result = timeseries_ds[0]
    assert (len(inputs)) + 1 == len(result[0])
    assert np.array_equal(result[1][0][:, 0, :, :], targets_expected)

    # With insolation and constants we get 2 extra tensors
    timeseries_ds = TimeSeriesDatasetZarr(
        dataset_path=dataset_path,
        input_variables=input_variables,
        constant_variables=constant_variables,
        scaling=scaling_double_dict,
        batch_size=batch_size,
        drop_last=True,
        add_insolation=True,
        start_date=time_da[0],
        end_date=time_da[-1],
    )
    assert (len(inputs)) + 2 == len(timeseries_ds[0][0])

    # train noise is applied directly to the inputs of the (non-coupled) Zarr
    # dataset; this option doesn't exist on the classic (non-Zarr) TimeSeriesDataset
    timeseries_ds = TimeSeriesDatasetZarr(
        dataset_path=dataset_path,
        input_variables=input_variables,
        scaling=scaling_double_dict,
        batch_size=batch_size,
        drop_last=True,
        start_date=time_da[0],
        end_date=time_da[-1],
    )
    non_perturbed = timeseries_ds[0]

    noise_params = {"inputs": scaling_double_dict}
    timeseries_ds = TimeSeriesDatasetZarr(
        dataset_path=dataset_path,
        input_variables=input_variables,
        scaling=scaling_double_dict,
        batch_size=batch_size,
        drop_last=True,
        add_train_noise=True,
        train_noise_params=noise_params,
        start_date=time_da[0],
        end_date=time_da[-1],
    )
    perturbed = timeseries_ds[0]

    # same shape, but the perturbed sample should differ from the un-perturbed one
    assert non_perturbed[0][0].shape == perturbed[0][0].shape
    assert not np.array_equal(non_perturbed[0][0], perturbed[0][0])

    # nothing should change with forecast mode other than getting just inputs
    init_times = random.randint(1, len(zarr_ds.time.values))
    timeseries_ds = TimeSeriesDatasetZarr(
        dataset_path=dataset_path,
        input_variables=input_variables,
        scaling=scaling_double_dict,
        batch_size=1,
        forecast_init_times=zarr_ds.time[:init_times].values,
    )
    inputs = timeseries_ds[0]

    assert len(inputs) == 1

    # insolation adds 1 extra tensor, same as above but using forecast mode
    init_times = random.randint(1, len(zarr_ds.time.values))
    timeseries_ds = TimeSeriesDatasetZarr(
        dataset_path=dataset_path,
        input_variables=input_variables,
        scaling=scaling_double_dict,
        batch_size=1,
        add_insolation=True,
        forecast_init_times=zarr_ds.time[:init_times].values,
    )
    assert (len(inputs) + 1) == len(timeseries_ds[0])


@requires_module("omegaconf")
@requires_module("netCDF4")
@requires_module("zarr")
def test_TimeSeriesDataModule_initialization(
    dataset_path, splits, scaling_double_dict, pytestconfig
):
    from physicsnemo.datapipes.healpix.data_modules_zarr import (
        TimeSeriesDataModuleZarr,
    )

    input_variables = ["z500", "z1000"]

    # open our test dataset
    zarr_ds = xr.open_zarr(dataset_path)

    # test for invalid path
    with pytest.raises(FileNotFoundError, match=("Dataset path not found")):
        timeseries_dm = TimeSeriesDataModuleZarr(
            dataset_path="DoesntExist",
            input_variables=input_variables,
            batch_size=1,
            scaling=scaling_double_dict,
        )

    # test for missing times
    with pytest.raises(
        ValueError, match=("Either splits or forecast_init_times must be provided")
    ):
        timeseries_dm = TimeSeriesDataModuleZarr(
            dataset_path=dataset_path,
            input_variables=input_variables,
            batch_size=1,
            scaling=scaling_double_dict,
        )

    # test for overlapping dates
    warnings.filterwarnings("error")
    with pytest.raises(
        UserWarning, match=("Training and validation date ranges overlap")
    ):
        bad_splits = splits.copy()
        bad_splits["val_date_start"] = splits["train_date_start"]
        timeseries_dm = TimeSeriesDataModuleZarr(
            dataset_path=dataset_path,
            input_variables=input_variables,
            batch_size=1,
            splits=bad_splits,
            scaling=scaling_double_dict,
        )
    warnings.resetwarnings()

    warnings.filterwarnings("error")
    with pytest.raises(UserWarning, match=("Training and test date ranges overlap")):
        bad_splits = splits.copy()
        bad_splits["test_date_start"] = splits["train_date_start"]
        timeseries_dm = TimeSeriesDataModuleZarr(
            dataset_path=dataset_path,
            input_variables=input_variables,
            batch_size=1,
            splits=bad_splits,
            scaling=scaling_double_dict,
        )
    warnings.resetwarnings()

    warnings.filterwarnings("error")
    with pytest.raises(UserWarning, match=("Test and validation date ranges overlap")):
        bad_splits = splits.copy()
        bad_splits["val_date_end"] = splits["test_date_start"]
        timeseries_dm = TimeSeriesDataModuleZarr(
            dataset_path=dataset_path,
            input_variables=input_variables,
            batch_size=1,
            splits=bad_splits,
            scaling=scaling_double_dict,
        )
    warnings.resetwarnings()

    # with init times
    timeseries_dm = TimeSeriesDataModuleZarr(
        dataset_path=dataset_path,
        input_variables=input_variables,
        batch_size=1,
        scaling=scaling_double_dict,
        forecast_init_times=zarr_ds.time[:2],
    )
    assert isinstance(timeseries_dm, TimeSeriesDataModuleZarr)

    # with splits
    timeseries_dm = TimeSeriesDataModuleZarr(
        dataset_path=dataset_path,
        input_variables=input_variables,
        batch_size=1,
        scaling=scaling_double_dict,
        splits=omegaconf.DictConfig(splits),
    )
    assert isinstance(timeseries_dm, TimeSeriesDataModuleZarr)
    zarr_ds.close()
    DistributedManager.cleanup()


@requires_module("omegaconf")
@requires_module("netCDF4")
@requires_module("numpy")
@requires_module("zarr")
def test_TimeSeriesDataModule_get_constants(
    dataset_path, scaling_double_dict, splits, pytestconfig
):
    from physicsnemo.datapipes.healpix.data_modules_zarr import (
        TimeSeriesDataModuleZarr,
    )

    input_variables = ["z500", "z1000"]
    constants = ["lsm"]

    # open our test dataset
    zarr_ds = xr.open_zarr(dataset_path)
    forecast_times = zarr_ds.time[:2]

    # No constants
    # Internally initializes DistributedManager
    timeseries_dm = TimeSeriesDataModuleZarr(
        dataset_path=dataset_path,
        input_variables=input_variables,
        batch_size=1,
        scaling=scaling_double_dict,
        constant_variables=None,
        forecast_init_times=forecast_times,
    )

    assert timeseries_dm.get_constants() is None

    # Constant that isn't in dataset
    with pytest.raises(KeyError, match=("Requested constants not found in dataset")):
        timeseries_dm = TimeSeriesDataModuleZarr(
            dataset_path=dataset_path,
            input_variables=input_variables,
            batch_size=1,
            scaling=scaling_double_dict,
            constant_variables={"missing": "missing"},
            forecast_init_times=forecast_times,
        )

    # just lsm as constant
    timeseries_dm = TimeSeriesDataModuleZarr(
        dataset_path=dataset_path,
        input_variables=input_variables,
        batch_size=1,
        scaling=scaling_double_dict,
        constant_variables=constants,
        forecast_init_times=forecast_times,
    )

    # open our test dataset
    zarr_ds = xr.open_zarr(dataset_path)

    # dividing by 2 due to scaling
    expected = (
        np.transpose(
            zarr_ds.constants.sel(channel_c=constants).values,
            axes=(1, 0, 2, 3),
        )
        / 2.0
    )

    assert np.array_equal(
        timeseries_dm.get_constants(),
        expected,
    )

    # with splits we're doing forecasting and get
    # constants from train instead of test dataset
    timeseries_dm = TimeSeriesDataModuleZarr(
        dataset_path=dataset_path,
        input_variables=input_variables,
        batch_size=1,
        scaling=scaling_double_dict,
        constant_variables=constants,
        splits=splits,
    )

    assert np.array_equal(
        timeseries_dm.get_constants(),
        expected,
    )
    zarr_ds.close()
    DistributedManager.cleanup()


@requires_module("omegaconf")
@requires_module("zarr")
def test_TimeSeriesDataModule_get_dataloaders(
    dataset_path, scaling_double_dict, splits, pytestconfig
):
    from physicsnemo.datapipes.healpix.data_modules_zarr import (
        TimeSeriesDataModuleZarr,
    )

    input_variables = ["z500", "z1000"]

    # use the prebuilt dataset
    # Internally initializes DistributedManager
    timeseries_dm = TimeSeriesDataModuleZarr(
        dataset_path=dataset_path,
        input_variables=input_variables,
        batch_size=1,
        scaling=scaling_double_dict,
        splits=splits,
        shuffle=False,
    )

    assert_shard_dataloaders(timeseries_dm)
    DistributedManager.cleanup()
