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
import shutil
import warnings
from pathlib import Path

import pytest
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from physicsnemo.distributed import DistributedManager
from test.conftest import requires_module

omegaconf = pytest.importorskip("omegaconf")
np = pytest.importorskip("numpy")
xr = pytest.importorskip("xarray")


@pytest.fixture
def data_dir(nfs_data_dir):
    return nfs_data_dir.joinpath("datasets/healpix")


@pytest.fixture
def dataset_name():
    name = "healpix"
    return name


@pytest.fixture
def create_path(nfs_data_dir):
    return nfs_data_dir.joinpath("datasets/healpix/merge")


def delete_dataset(create_path, dataset_name):
    """Helper that deletes a requested dataset at the specified location"""
    dataset_path = f"{create_path}/{dataset_name}.zarr"
    if Path(dataset_path).exists():
        shutil.rmtree(dataset_path)


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


@requires_module("omegaconf")
@requires_module("dask")
@requires_module("netCDF4")
def test_open_time_series_on_the_fly(create_path, pytestconfig):
    from physicsnemo.datapipes.healpix.data_modules import (
        open_time_series_dataset_classic_on_the_fly,
    )

    variables = ["z500", "z1000"]
    constants = {"lsm": "lsm"}

    ds = open_time_series_dataset_classic_on_the_fly(
        directory=create_path,
        input_variables=variables,
        output_variables=variables,
        constants=constants,
    )
    assert isinstance(ds, xr.Dataset)

    test_var = variables[0]
    base = xr.open_dataset(str(create_path.joinpath(f"{test_var}.nc")))
    ds_var = ds.inputs.sel(channel_in=test_var)

    assert ds_var.equals(base[test_var])
    ds.close()
    base.close()


@requires_module("omegaconf")
def test_open_time_series(data_dir, dataset_name, pytestconfig):
    # check for failure of non-existant dataset
    from physicsnemo.datapipes.healpix.data_modules import (
        open_time_series_dataset_classic_prebuilt,
    )

    with pytest.raises(FileNotFoundError, match=("Dataset doesn't appear to exist at")):
        open_time_series_dataset_classic_prebuilt("/null_path", dataset_name)

    ds = open_time_series_dataset_classic_prebuilt(data_dir, dataset_name)
    assert isinstance(ds, xr.Dataset)
    ds.close()


@requires_module("omegaconf")
@requires_module("dask")
@requires_module("netCDF4")
@requires_module("numpy")
def test_create_time_series(data_dir, dataset_name, create_path, pytestconfig):
    from physicsnemo.datapipes.healpix.data_modules import (
        create_time_series_dataset_classic,
    )

    variables = ["z500", "z1000"]
    constants = {"lsm": "lsm"}
    scaling = {"z500": {"log_epsilon": 2}}
    # check existing dataset
    ds = create_time_series_dataset_classic(
        src_directory="/null",
        dst_directory=data_dir,
        dataset_name=dataset_name,
        input_variables=["null", "null"],
    )
    assert isinstance(ds, xr.Dataset)
    ds.close()

    # create new dataset
    # open a base dataset to compare against
    test_var = list(scaling.keys())[0]
    base = xr.open_dataset(str(create_path.joinpath(f"{test_var}.nc")))
    # scale our test variable
    base[test_var] = np.log(base[test_var] + scaling[test_var]["log_epsilon"]) - np.log(
        scaling[test_var]["log_epsilon"]
    )
    ds = create_time_series_dataset_classic(
        src_directory=create_path,
        dst_directory=create_path,
        dataset_name=dataset_name,
        input_variables=variables,
        scaling=scaling,
    )
    ds_var = ds.inputs.sel(channel_in=test_var)

    assert ds_var.equals(base[test_var])
    ds.close()
    base.close()

    # delete the created file so we have a clean test for next time
    delete_dataset(create_path, dataset_name)

    # and with constants
    const = list(constants.keys())[0]
    const_ds = xr.open_dataset(str(create_path.joinpath(f"{const}.nc")))
    ds = create_time_series_dataset_classic(
        src_directory=create_path,
        dst_directory=create_path,
        dataset_name=dataset_name,
        input_variables=variables,
        scaling=scaling,
        constants=constants,
    )
    assert (const_ds[const] == ds.constants[0]).any()
    ds.close()
    const_ds.close()

    # delete the created file so we have a clean test for next time
    delete_dataset(create_path, dataset_name)


@requires_module("omegaconf")
@requires_module("dask")
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
@requires_module("dask")
@requires_module("netCDF4")
def test_TimeSeriesDataModule_initialization(
    data_dir, create_path, dataset_name, scaling_double_dict, pytestconfig
):
    from physicsnemo.datapipes.healpix.data_modules import (
        TimeSeriesDataModule,
    )

    variables = ["z500", "z1000"]
    splits = {
        "train_date_start": "1959-01-01",
        "train_date_end": "1998-12-31T18:00",
        "val_date_start": "1999-01-01",
        "val_date_end": "2000-12-31T18:00",
        "test_date_start": "2017-01-01",
        "test_date_end": "2018-12-31T18:00",
    }

    # open our test dataset
    ds_path = Path(data_dir, dataset_name + ".zarr")
    zarr_ds = xr.open_zarr(ds_path)

    # test with an invalid mode
    with pytest.raises(ValueError, match=("'data_format' must be one of")):
        timeseries_dm = TimeSeriesDataModule(
            src_directory=data_dir,
            dst_directory=create_path,
            dataset_name=dataset_name,
            batch_size=1,
            data_format="null",
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

    # without the prebuilt dataset
    timeseries_dm = TimeSeriesDataModule(
        src_directory=create_path,
        dst_directory=create_path,
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
    data_dir, create_path, dataset_name, scaling_double_dict, pytestconfig
):
    from physicsnemo.datapipes.healpix.data_modules import (
        TimeSeriesDataModule,
    )

    variables = ["z500", "z1000"]
    splits = {
        "train_date_start": "1979-01-01",
        "train_date_end": "1979-01-01T21:00",
        "val_date_start": "1979-01-02",
        "val_date_end": "1979-01-02T09:00",
        "test_date_start": "1979-01-02T12:00",
        "test_date_end": "1979-01-02T18:00",
    }

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
