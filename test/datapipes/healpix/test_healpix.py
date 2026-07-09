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

import random
import warnings
from pathlib import Path

import pytest

from physicsnemo.distributed import DistributedManager
from test.conftest import requires_module
from test.datapipes.healpix.conftest import assert_shard_dataloaders

omegaconf = pytest.importorskip("omegaconf")
np = pytest.importorskip("numpy")
xr = pytest.importorskip("xarray")


@requires_module("omegaconf")
def test_open_time_series(data_dir, dataset_name, pytestconfig):
    # check for failure of non-existant dataset
    from physicsnemo.datapipes.healpix.data_modules import (
        open_time_series_dataset_classic_prebuilt,
    )

    with pytest.raises(FileNotFoundError, match=("Dataset doesn't exist at")):
        open_time_series_dataset_classic_prebuilt("/null_path", dataset_name)

    ds = open_time_series_dataset_classic_prebuilt(data_dir, dataset_name)
    assert isinstance(ds, xr.Dataset)
    ds.close()


@requires_module("omegaconf")
@requires_module("netCDF4")
def test_TimeSeriesDataset_initialization(
    data_dir, dataset_name, scaling_dict, pytestconfig
):
    from physicsnemo.datapipes.healpix.timeseries_dataset import TimeSeriesDataset

    # open our test dataset
    ds_path = Path(data_dir, dataset_name + ".zarr")
    zarr_ds = xr.open_zarr(ds_path)

    # check for failure of timestep not being a multiple of datatime step
    with pytest.raises(
        ValueError, match=("'time_step' must be a multiple of 'data_time_step' ")
    ):
        timeseries_ds = TimeSeriesDataset(
            dataset=zarr_ds,
            data_time_step="2h",
            time_step="5h",
            scaling=scaling_dict,
        )

    # check for failure of gap not being a multiple of datatime step
    with pytest.raises(
        ValueError, match=("'gap' must be a multiple of 'data_time_step' ")
    ):
        timeseries_ds = TimeSeriesDataset(
            dataset=zarr_ds,
            data_time_step="2h",
            time_step="6h",
            gap="3h",
            scaling=scaling_dict,
        )

    # check for failure of invalid scaling variable on input
    invalid_scaling = omegaconf.DictConfig(
        {
            "bogosity": {"mean": 0, "std": 42},
        }
    )
    with pytest.raises(KeyError, match=("Input channels ")):
        timeseries_ds = TimeSeriesDataset(
            dataset=zarr_ds,
            data_time_step="3h",
            time_step="6h",
            scaling=invalid_scaling,
        )

    # check for failure when a channel_out variable has no scaling entry.
    # dropping "z1000" from channel_in (but keeping it in channel_out) means
    # len(channel_out) != len(channel_in), so input scaling is selected by
    # channel_in (which succeeds), letting the target-scaling failure surface
    # on its own
    scaling_missing_target = omegaconf.DictConfig(
        {k: v for k, v in scaling_dict.items() if k != "z1000"}
    )
    zarr_ds_asymmetric = zarr_ds.sel(
        channel_in=[v for v in zarr_ds.channel_in.values if v != "z1000"],
        channel_out=zarr_ds.channel_out.values,
    )
    with pytest.raises(KeyError, match=("Target channels ")):
        timeseries_ds = TimeSeriesDataset(
            dataset=zarr_ds_asymmetric,
            scaling=scaling_missing_target,
        )

    # check for failure when a constant (channel_c) variable has no scaling
    # entry, even though every channel_in/channel_out variable does
    scaling_missing_constant = omegaconf.DictConfig(
        {k: v for k, v in scaling_dict.items() if k != "z"}
    )
    with pytest.raises(KeyError, match=("Constant channels ")):
        timeseries_ds = TimeSeriesDataset(
            dataset=zarr_ds,
            scaling=scaling_missing_constant,
        )

    # check for warning on batch size > 1 and forecast mode
    warnings.filterwarnings("error")
    with pytest.raises(
        UserWarning,
        match=(
            "providing 'forecast_init_times' to TimeSeriesDataset requires `batch_size=1`"
        ),
    ):
        timeseries_ds = TimeSeriesDataset(
            dataset=zarr_ds,
            scaling=scaling_dict,
            batch_size=2,
            forecast_init_times=zarr_ds.time[:2],
        )

    # test no scaling
    timeseries_ds = TimeSeriesDataset(
        dataset=zarr_ds,
    )
    assert isinstance(timeseries_ds, TimeSeriesDataset)

    timeseries_ds = TimeSeriesDataset(
        dataset=zarr_ds,
        scaling=scaling_dict,
    )
    assert isinstance(timeseries_ds, TimeSeriesDataset)

    timeseries_ds = TimeSeriesDataset(
        dataset=zarr_ds,
        scaling=scaling_dict,
        batch_size=1,
        forecast_init_times=zarr_ds.time[:2],
    )
    assert isinstance(timeseries_ds, TimeSeriesDataset)

    timeseries_ds = TimeSeriesDataset(
        dataset=zarr_ds,
        scaling=scaling_dict,
        batch_size=1,
        forecast_init_times=zarr_ds.time[:2],
        data_time_step="3h",
        time_step="6h",
    )
    assert isinstance(timeseries_ds, TimeSeriesDataset)
    zarr_ds.close()


@requires_module("omegaconf")
@requires_module("netCDF4")
@requires_module("numpy")
def test_TimeSeriesDataset_get_constants(
    data_dir, dataset_name, scaling_dict, pytestconfig
):
    from physicsnemo.datapipes.healpix.timeseries_dataset import TimeSeriesDataset

    # open our test dataset
    ds_path = Path(data_dir, dataset_name + ".zarr")
    zarr_ds = xr.open_zarr(ds_path)

    timeseries_ds = TimeSeriesDataset(
        dataset=zarr_ds,
        scaling=scaling_dict,
    )

    # constants are reshaped
    expected = np.transpose(zarr_ds.constants.values, axes=(1, 0, 2, 3))
    outvar = timeseries_ds.get_constants()
    assert np.array_equal(
        expected,
        outvar,
    )
    zarr_ds.close()


@requires_module("omegaconf")
@requires_module("netCDF4")
def test_TimeSeriesDataset_len(data_dir, dataset_name, scaling_dict, pytestconfig):
    from physicsnemo.datapipes.healpix.timeseries_dataset import TimeSeriesDataset

    # open our test dataset
    ds_path = Path(data_dir, dataset_name + ".zarr")
    zarr_ds = xr.open_zarr(ds_path)

    # check forecast mode
    init_times = random.randint(1, len(zarr_ds.time.values))
    timeseries_ds = TimeSeriesDataset(
        dataset=zarr_ds,
        scaling=scaling_dict,
        batch_size=1,
        forecast_init_times=zarr_ds.time[:init_times],
    )
    assert len(timeseries_ds) == init_times

    # check train mode
    timeseries_ds = TimeSeriesDataset(
        dataset=zarr_ds,
        data_time_step="3h",
        time_step="9h",
        scaling=scaling_dict,
        batch_size=2,
    )
    # Window length of 3 for one sample size
    assert len(timeseries_ds) == (len(zarr_ds.time.values) - 2) // 2

    # check train mode
    timeseries_ds = TimeSeriesDataset(
        dataset=zarr_ds,
        data_time_step="3h",
        time_step="9h",
        scaling=scaling_dict,
        batch_size=2,
        drop_last=True,
    )
    assert len(timeseries_ds) == (len(zarr_ds.time.values) - 2) // 2
    zarr_ds.close()


@requires_module("omegaconf")
@requires_module("netCDF4")
@requires_module("numpy")
def test_TimeSeriesDataset_get(
    data_dir, dataset_name, scaling_double_dict, pytestconfig
):
    from physicsnemo.datapipes.healpix.timeseries_dataset import TimeSeriesDataset

    # open our test dataset
    ds_path = Path(data_dir, dataset_name + ".zarr")
    zarr_ds = xr.open_zarr(ds_path)

    batch_size = 2
    timeseries_ds = TimeSeriesDataset(
        dataset=zarr_ds,
        scaling=scaling_double_dict,
        batch_size=batch_size,
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
    targets_expected = zarr_ds.targets[batch_size].transpose(
        "face", "channel_out", "height", "width"
    )
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
    timeseries_ds = TimeSeriesDataset(
        dataset=zarr_ds,
        scaling=scaling_double_dict,
        batch_size=batch_size,
        drop_last=True,
    )

    inputs, targets = timeseries_ds[-1]
    targets_expected = zarr_ds.targets[-1 - batch_size].transpose(
        "face", "channel_out", "height", "width"
    )
    targets_expected = targets_expected.to_numpy() / 2
    assert np.array_equal(targets[0][:, 0, :, :], targets_expected)

    # With insolation we get 1 extra channel
    timeseries_ds = TimeSeriesDataset(
        dataset=zarr_ds,
        scaling=scaling_double_dict,
        batch_size=batch_size,
        drop_last=True,
        add_insolation=True,
    )
    assert (len(inputs)) + 1 == len(timeseries_ds[0][0])

    # nothing should change with forecast mode other than getting just inputs
    init_times = random.randint(1, len(zarr_ds.time.values))
    timeseries_ds = TimeSeriesDataset(
        dataset=zarr_ds,
        scaling=scaling_double_dict,
        batch_size=1,
        forecast_init_times=zarr_ds.time[:init_times],
    )
    inputs = timeseries_ds[0]

    assert np.array_equal(targets[0][:, 0, :, :], targets_expected)

    # insolation adds 1 extra channel
    init_times = random.randint(1, len(zarr_ds.time.values))
    timeseries_ds = TimeSeriesDataset(
        dataset=zarr_ds,
        scaling=scaling_double_dict,
        batch_size=1,
        add_insolation=True,
        forecast_init_times=zarr_ds.time[:init_times],
    )
    assert (len(inputs)) + 1 == len(timeseries_ds[0])

    # No constants in input data
    init_times = random.randint(1, len(zarr_ds.time.values))
    zarr_ds_no_const = zarr_ds.drop_vars("constants")
    timeseries_ds = TimeSeriesDataset(
        dataset=zarr_ds_no_const,
        scaling=scaling_double_dict,
        batch_size=1,
        forecast_init_times=zarr_ds.time[:init_times],
    )
    assert len(inputs) == (len(timeseries_ds[0]) + 1)
    zarr_ds.close()


@requires_module("omegaconf")
@requires_module("netCDF4")
def test_TimeSeriesDataModule_initialization(
    data_dir, create_path, dataset_name, scaling_double_dict, splits, pytestconfig
):
    from physicsnemo.datapipes.healpix.data_modules import (
        TimeSeriesDataModule,
    )

    variables = ["z500", "z1000"]

    # open our test dataset
    ds_path = Path(data_dir, dataset_name + ".zarr")
    zarr_ds = xr.open_zarr(ds_path)

    # check for failure when a requested input variable isn't in the dataset
    with pytest.raises(ValueError, match=("Input variables not found in dataset")):
        TimeSeriesDataModule(
            src_directory=create_path,
            dst_directory=data_dir,
            dataset_name=dataset_name,
            input_variables=variables + ["DoesntExist"],
            batch_size=1,
            prebuilt_dataset=True,
            scaling=scaling_double_dict,
        )

    # check for failure when a requested output variable isn't in the dataset
    with pytest.raises(ValueError, match=("Output variables not found in dataset")):
        TimeSeriesDataModule(
            src_directory=create_path,
            dst_directory=data_dir,
            dataset_name=dataset_name,
            input_variables=variables,
            output_variables=variables + ["DoesntExist"],
            batch_size=1,
            prebuilt_dataset=True,
            scaling=scaling_double_dict,
        )

    # check for failure when a requested constant isn't in the dataset
    with pytest.raises(ValueError, match=("Constants not found in dataset")):
        TimeSeriesDataModule(
            src_directory=create_path,
            dst_directory=data_dir,
            dataset_name=dataset_name,
            input_variables=variables,
            batch_size=1,
            prebuilt_dataset=True,
            scaling=scaling_double_dict,
            constants={"lsm": "DoesntExist"},
        )

    # use the prebuilt dataset
    # Internally initializes DistributedManager
    timeseries_dm = TimeSeriesDataModule(
        src_directory=create_path,
        dst_directory=data_dir,
        dataset_name=dataset_name,
        input_variables=variables,
        batch_size=1,
        prebuilt_dataset=True,
        scaling=scaling_double_dict,
    )
    assert isinstance(timeseries_dm, TimeSeriesDataModule)

    # `prebuilt_dataset` is kept only for backwards compatibility; on-the-fly
    # dataset generation has been removed, so `setup()` always opens an
    # existing prebuilt dataset from `dst_directory` regardless of this flag
    timeseries_dm = TimeSeriesDataModule(
        src_directory=create_path,
        dst_directory=data_dir,
        dataset_name=dataset_name,
        input_variables=variables,
        batch_size=1,
        prebuilt_dataset=False,
        scaling=scaling_double_dict,
    )
    assert isinstance(timeseries_dm, TimeSeriesDataModule)

    # with init times
    timeseries_dm = TimeSeriesDataModule(
        src_directory=create_path,
        dst_directory=data_dir,
        dataset_name=dataset_name,
        input_variables=variables,
        batch_size=1,
        prebuilt_dataset=True,
        scaling=scaling_double_dict,
        forecast_init_times=zarr_ds.time[:2],
    )
    assert isinstance(timeseries_dm, TimeSeriesDataModule)

    # with splits
    timeseries_dm = TimeSeriesDataModule(
        src_directory=create_path,
        dst_directory=data_dir,
        dataset_name=dataset_name,
        input_variables=variables,
        batch_size=1,
        prebuilt_dataset=True,
        scaling=scaling_double_dict,
        splits=omegaconf.DictConfig(splits),
    )
    assert isinstance(timeseries_dm, TimeSeriesDataModule)
    zarr_ds.close()
    DistributedManager.cleanup()


@requires_module("omegaconf")
@requires_module("netCDF4")
@requires_module("numpy")
def test_TimeSeriesDataModule_get_constants(
    data_dir, create_path, dataset_name, scaling_double_dict, pytestconfig
):
    from physicsnemo.datapipes.healpix.data_modules import (
        TimeSeriesDataModule,
    )

    variables = ["z500", "z1000"]
    constants = {"lsm": "lsm"}

    # No constants
    # Internally initializes DistributedManager
    timeseries_dm = TimeSeriesDataModule(
        src_directory=create_path,
        dst_directory=data_dir,
        dataset_name=dataset_name,
        input_variables=variables,
        batch_size=1,
        prebuilt_dataset=True,
        scaling=scaling_double_dict,
        constants=None,
    )

    assert timeseries_dm.get_constants() is None

    # just lsm as constant
    timeseries_dm = TimeSeriesDataModule(
        src_directory=create_path,
        dst_directory=data_dir,
        dataset_name=dataset_name,
        input_variables=variables,
        batch_size=1,
        prebuilt_dataset=True,
        scaling=scaling_double_dict,
        constants=constants,
    )

    # open our test dataset
    ds_path = Path(data_dir, dataset_name + ".zarr")
    zarr_ds = xr.open_zarr(ds_path)

    # dividing by 2 due to scaling
    expected = (
        np.transpose(
            zarr_ds.constants.sel(channel_c=list(constants.keys())).values,
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
    timeseries_dm = TimeSeriesDataModule(
        src_directory=create_path,
        dst_directory=data_dir,
        dataset_name=dataset_name,
        input_variables=variables,
        batch_size=1,
        prebuilt_dataset=True,
        scaling=scaling_double_dict,
        constants=constants,
    )

    assert np.array_equal(
        timeseries_dm.get_constants(),
        expected,
    )
    zarr_ds.close()
    DistributedManager.cleanup()


@requires_module("omegaconf")
def test_TimeSeriesDataModule_get_dataloaders(
    data_dir, create_path, dataset_name, scaling_double_dict, splits, pytestconfig
):
    from physicsnemo.datapipes.healpix.data_modules import (
        TimeSeriesDataModule,
    )

    variables = ["z500", "z1000"]

    # use the prebuilt dataset
    # Internally initializes DistributedManager
    timeseries_dm = TimeSeriesDataModule(
        src_directory=create_path,
        dst_directory=data_dir,
        dataset_name=dataset_name,
        input_variables=variables,
        batch_size=1,
        prebuilt_dataset=True,
        scaling=scaling_double_dict,
        splits=splits,
        shuffle=False,
    )

    assert_shard_dataloaders(timeseries_dm)
    DistributedManager.cleanup()
