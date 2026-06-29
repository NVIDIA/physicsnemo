# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
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
from pathlib import Path

import pytest
from pytest_utils import import_or_fail, nfsdata_or_fail
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from physicsnemo.distributed import DistributedManager

omegaconf = pytest.importorskip("omegaconf")
np = pytest.importorskip("numpy")
xr = pytest.importorskip("xarray")
zarr = pytest.importorskip("zarr")


@pytest.fixture
def dataset_path():
    data_dir = "/data/nfs/modulus-data/datasets/healpix/"
    dataset_name = "healpix.zarr"
    path = Path(data_dir, dataset_name)
    return path


@pytest.fixture
def splits():
    split_dict = {
        "train_date_start": "1979-01-01",
        "train_date_end": "1979-01-01T21:00",
        "val_date_start": "1979-01-02",
        "val_date_end": "1979-01-02T09:00",
        "test_date_start": "1979-01-02T12:00",
        "test_date_end": "1979-01-02T18:00",
    }
    return split_dict


@pytest.fixture
def scaling_dict():
    scaling = {
        "t2m0": {"mean": 287.8665771484375, "std": 14.86227798461914},
        "t850": {"mean": 281.2710266113281, "std": 12.04991626739502},
        "tau300-700": {"mean": 61902.72265625, "std": 2559.8408203125},
        "tcwv0": {"mean": 24.034976959228516, "std": 16.411935806274414},
        "z1000": {"mean": 952.1435546875, "std": 895.7516479492188},
        "z250": {"mean": 101186.28125, "std": 5551.77978515625},
        "z500": {"mean": 55625.9609375, "std": 2681.712890625},
        "lsm": {"mean": 0, "std": 1},
        "z": {"mean": 0, "std": 1},
        "tp6": {"mean": 1, "std": 0, "log_epsilon": 1e-6},
        "extra": {"mean": 1, "std": 0},  # doesn't appear in test dataset
    }
    return omegaconf.DictConfig(scaling)


@pytest.fixture
def scaling_double_dict():
    scaling = {
        "t2m0": {"mean": 0, "std": 2},
        "t850": {"mean": 0, "std": 2},
        "tau300-700": {"mean": 0, "std": 2},
        "tcwv0": {"mean": 0, "std": 2},
        "z1000": {"mean": 0, "std": 2},
        "z250": {"mean": 0, "std": 2},
        "z500": {"mean": 0, "std": 2},
        "tp6": {"mean": 0, "std": 2, "log_epsilon": 1e-6},
        "lsm": {"mean": 0, "std": 2},
        "z": {"mean": 0, "std": 2},
        "extra": {"mean": 0, "std": 2},  # doesn't appear in test dataset
    }
    return omegaconf.DictConfig(scaling)


@import_or_fail("omegaconf")
@import_or_fail("netCDF4")
@nfsdata_or_fail
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

    zarr_ds = zarr.open(dataset_path)
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
            forecast_init_times=zarr_ds.time[:2],
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
            forecast_init_times=zarr_ds.time[:2],
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


@import_or_fail("omegaconf")
@import_or_fail("netCDF4")
@import_or_fail("numpy")
@import_or_fail("zarr")
@nfsdata_or_fail
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


@import_or_fail("omegaconf")
@import_or_fail("netCDF4")
@nfsdata_or_fail
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


@import_or_fail("omegaconf")
@import_or_fail("netCDF4")
@import_or_fail("numpy")
@nfsdata_or_fail
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


@import_or_fail("omegaconf")
@import_or_fail("netCDF4")
@import_or_fail("zarr")
@nfsdata_or_fail
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


@import_or_fail("omegaconf")
@import_or_fail("netCDF4")
@import_or_fail("numpy")
@import_or_fail("zarr")
@nfsdata_or_fail
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


@import_or_fail("omegaconf")
@import_or_fail("zarr")
@nfsdata_or_fail
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

    # with 1 shard should get no sampler
    train_dataloader, train_sampler = timeseries_dm.train_dataloader(num_shards=1)
    assert train_sampler is None
    assert isinstance(train_dataloader, DataLoader)

    val_dataloader, val_sampler = timeseries_dm.val_dataloader(num_shards=1)
    assert val_sampler is None
    assert isinstance(val_dataloader, DataLoader)

    test_dataloader, test_sampler = timeseries_dm.test_dataloader(num_shards=1)
    assert test_sampler is None
    assert isinstance(test_dataloader, DataLoader)

    # with >1 shard should be distributed sampler
    train_dataloader, train_sampler = timeseries_dm.train_dataloader(num_shards=2)
    assert isinstance(train_sampler, DistributedSampler)
    assert isinstance(train_dataloader, DataLoader)

    val_dataloader, val_sampler = timeseries_dm.val_dataloader(num_shards=2)
    assert isinstance(val_sampler, DistributedSampler)
    assert isinstance(val_dataloader, DataLoader)

    test_dataloader, test_sampler = timeseries_dm.test_dataloader(num_shards=2)
    assert isinstance(test_sampler, DistributedSampler)
    assert isinstance(test_dataloader, DataLoader)
    DistributedManager.cleanup()
