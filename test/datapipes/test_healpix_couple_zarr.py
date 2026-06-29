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

import random
import warnings
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch as th
from pytest_utils import import_or_fail, nfsdata_or_fail
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from physicsnemo.distributed import DistributedManager

omegaconf = pytest.importorskip("omegaconf")
np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
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
def constant_coupler_config():
    constant_coupler = [
        {
            "coupler": "ConstantCoupler",
            "params": {
                "batch_size": 1,
                "variables": ["z250"],
                "input_times": ["0h"],
                "input_time_dim": 1,
                "output_time_dim": 1,
                "presteps": 0,
                "prepared_coupled_data": True,
            },
        }
    ]
    return constant_coupler


@pytest.fixture
def average_coupler_config():
    average_coupler = [
        {
            "coupler": "ConstantCoupler",
            "params": {
                "batch_size": 1,
                "variables": ["z250"],
                "input_times": ["0h"],
                "input_time_dim": 1,
                "output_time_dim": 1,
                "presteps": 0,
                "prepared_coupled_data": True,
            },
        }
    ]
    return average_coupler


@dataclass
class coupler_helper:
    """helper class for setting up the couplers"""

    output_variables: list
    time_step: str


@pytest.fixture
def scaling_dict():
    scaling = {
        "t2m0": {"mean": 287.8665771484375, "std": 14.86227798461914},
        "t850": {"mean": 281.2710266113281, "std": 12.04991626739502},
        "tau300-700": {"mean": 61902.72265625, "std": 2559.8408203125},
        "tcwv0": {"mean": 24.034976959228516, "std": 16.411935806274414},
        "z1000": {"mean": 952.1435546875, "std": 895.7516479492188},
        "z1000-12h": {"mean": 952.1435546875, "std": 895.7516479492188},
        "z250": {"mean": 101186.28125, "std": 5551.77978515625},
        "z500": {"mean": 55625.9609375, "std": 2681.712890625},
        "lsm": {"mean": 0, "std": 1},
        "z": {"mean": 0, "std": 1},
        "tp6": {"mean": 1, "std": 0, "log_epsilon": 1e-6},
        "extra": {"mean": 0, "std": 0},  # doesn't appear in test dataset
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
        "z1000-12h": {"mean": 0, "std": 2},
        "z250": {"mean": 0, "std": 2},
        "z500": {"mean": 0, "std": 2},
        "lsm": {"mean": 0, "std": 2},
        "z": {"mean": 0, "std": 2},
        "tp6": {"mean": 0, "std": 2, "log_epsilon": 1e-6},
        "extra": {"mean": 0, "std": 2},  # doesn't appear in test dataset
    }
    return omegaconf.DictConfig(scaling)


@import_or_fail("omegaconf")
@import_or_fail("netCDF4")
@import_or_fail("pandas")
@import_or_fail("xarray")
@nfsdata_or_fail
def test_ConstantCoupler(dataset_path, scaling_dict, pytestconfig):

    from physicsnemo.datapipes.healpix.couplers import (
        ConstantCoupler,
    )

    variables = ["z500", "z1000"]
    input_times = ["0h"]
    input_time_dim = 1
    output_time_dim = 1
    presteps = 0
    batch_size = 2
    batch = {"time": slice(0, 2)}

    # open our test dataset
    zarr_ds = zarr.open(dataset_path)
    input_indices = [
        int(np.where(zarr_ds.channel_in[:] == ch)[0][0]) for ch in variables
    ]

    # test fail initialization
    with pytest.raises(
        NotImplementedError, match=("Data preparation not yet implemented")
    ):
        coupler = ConstantCoupler(
            dataset=zarr_ds,
            batch_size=batch_size,
            variables=variables,
            presteps=presteps,
            input_times=input_times,
            input_time_dim=input_time_dim,
            output_time_dim=output_time_dim,
            prepared_coupled_data=False,
        )

    coupler = ConstantCoupler(
        dataset=zarr_ds,
        batch_size=batch_size,
        variables=variables,
        presteps=presteps,
        input_times=input_times,
        input_time_dim=input_time_dim,
        output_time_dim=output_time_dim,
    )
    assert isinstance(coupler, ConstantCoupler)

    # check setting coupled variable indices
    mock_coupled_module = coupler_helper(
        output_variables=["not_coupled", "z500"],
        time_step="0h",
    )
    with pytest.raises(ValueError, match=("Missing variables in coupled module")):
        coupler.setup_coupling(mock_coupled_module)

    mock_coupled_module.output_variables = ["z500", "z1000"]
    coupler.setup_coupling(mock_coupled_module)
    assert coupler.coupled_channel_indices == [0, 1]

    interval = 2
    data_time_step = "3h"
    coupler.compute_coupled_indices(interval, data_time_step)
    coupled_integration_dim = presteps + max(output_time_dim // input_time_dim, 1)
    expected = np.empty([batch_size, coupled_integration_dim, len(input_times)])
    for b in range(batch_size):
        for i in range(coupled_integration_dim):
            expected[b, i, :] = b + np.array(
                [pd.Timedelta(ts) / pd.Timedelta(data_time_step) for ts in input_times]
            )
    expected = expected.astype(int)
    assert np.array_equal(expected, coupler._coupled_offsets)

    scaling_df = pd.DataFrame.from_dict(omegaconf.OmegaConf.to_object(scaling_dict)).T
    scaling_df.loc["zeros"] = {"mean": 0.0, "std": 1.0}
    scaling_da = scaling_df.to_xarray().astype("float32")
    coupler.set_scaling(scaling_da)
    coupled_scaling = scaling_da.sel(index=variables).rename({"index": "channel_in"})
    expected = np.expand_dims(coupled_scaling["mean"].to_numpy(), (0, 2, 3, 4))
    assert np.array_equal(expected, coupler.coupled_scaling["mean"])
    expected = np.expand_dims(coupled_scaling["std"].to_numpy(), (0, 2, 3, 4))
    assert np.array_equal(expected, coupler.coupled_scaling["std"])

    coupled_fields_batch_size = batch_size
    coupled_fields_timedim = 4
    expected_shape = [
        coupler.coupled_integration_dim,
        coupled_fields_batch_size,
        coupler.timevar_dim,
    ] + list(coupler.spatial_dims)
    coupled_fields = th.rand(
        coupled_fields_batch_size,
        coupler.spatial_dims[0],
        coupled_fields_timedim,
        len(coupler.coupled_channel_indices),
        coupler.spatial_dims[1],
        coupler.spatial_dims[2],
    )
    coupler.set_coupled_fields(coupled_fields)
    assert coupler.coupled_mode
    assert list(coupler.construct_integrated_couplings().shape) == expected_shape

    # verify that the data is being properly transformed
    expected = coupled_fields[:, :, :, coupler.coupled_channel_indices, :, :].permute(
        2, 0, 3, 1, 4, 5
    )
    expected = expected[0, :, -1:, :, :, :]
    expected = expected.unsqueeze(0)
    expected = expected.repeat(
        coupler.coupled_integration_dim, 1, coupled_fields_batch_size, 1, 1, 1
    )
    result = coupler.construct_integrated_couplings()
    assert th.equal(expected, coupler.construct_integrated_couplings())

    # verify that dimensions aren't reordered when time_first is false
    coupler.time_first = False
    coupler.set_coupled_fields(coupled_fields)
    # [T, B, C, F, H, W]
    expected = expected.permute(1, 3, 0, 2, 4, 5)
    result = coupler.construct_integrated_couplings()
    assert th.equal(expected, result)
    coupler.time_first = True

    # test coupler reset
    coupler.reset_coupler()
    assert coupler.coupled_mode is False

    # test loading from the dataset
    coupled_scaling = {
        "mean": np.expand_dims(coupled_scaling["mean"].to_numpy(), (0, 2, 3, 4)),
        "std": np.expand_dims(coupled_scaling["std"].to_numpy(), (0, 2, 3, 4)),
    }
    expected = zarr_ds.inputs[:2][:, input_indices]
    expected = (expected - coupled_scaling["mean"]) / coupled_scaling["std"]
    coupled_field = coupler.construct_integrated_couplings(
        batch=batch, bsize=batch_size
    )
    assert np.array_equal(expected, coupled_field[0])

    DistributedManager.cleanup()


@import_or_fail("omegaconf")
@import_or_fail("netCDF4")
@import_or_fail("pandas")
@import_or_fail("xarray")
@nfsdata_or_fail
def test_TrailingAverageCoupler(dataset_path, scaling_dict, pytestconfig):

    from physicsnemo.datapipes.healpix.couplers import (
        TrailingAverageCoupler,
    )

    variables = ["z500", "z1000-12h"]
    input_times = ["6h", "12h"]
    input_time_dim = 2
    output_time_dim = 2
    presteps = 0
    batch_size = 2
    averaging_window = "6h"
    # open our test dataset
    zarr_ds = zarr.open(dataset_path)

    # test fail initialization when trying to prepare data
    with pytest.raises(
        NotImplementedError, match=("Data preparation not yet implemented")
    ):
        coupler = TrailingAverageCoupler(
            dataset=zarr_ds,
            batch_size=batch_size,
            variables=variables,
            presteps=presteps,
            averaging_window=averaging_window,
            input_times=input_times,
            input_time_dim=input_time_dim,
            output_time_dim=output_time_dim,
            prepared_coupled_data=False,
        )

    # test fail when input times aren't evenly divisible by dataset dt
    with pytest.raises(ValueError, match=("Coupled input times")):
        coupler = TrailingAverageCoupler(
            dataset=zarr_ds,
            batch_size=batch_size,
            variables=variables,
            presteps=presteps,
            averaging_window=averaging_window,
            input_times=["30m"],
            input_time_dim=input_time_dim,
            output_time_dim=output_time_dim,
        )

    coupler = TrailingAverageCoupler(
        dataset=zarr_ds,
        batch_size=batch_size,
        variables=variables,
        presteps=presteps,
        averaging_window=averaging_window,
        input_times=input_times,
        input_time_dim=input_time_dim,
        output_time_dim=output_time_dim,
    )
    assert isinstance(coupler, TrailingAverageCoupler)

    mock_coupled_module = coupler_helper(
        output_variables=["not_coupled", "z500"],
        time_step="3h",
    )
    with pytest.raises(ValueError, match=("Missing variables in coupled module")):
        coupler.setup_coupling(mock_coupled_module)

    # veryify averaging slices computed correctly
    mock_coupled_module = coupler_helper(
        output_variables=["z500", "z1000"],
        time_step="3h",
    )
    coupler.setup_coupling(mock_coupled_module)
    averaging_window_max_indices = [
        i // pd.Timedelta(mock_coupled_module.time_step) for i in input_times
    ]
    dt = averaging_window_max_indices[0]
    # assumes only 1 integration step, otherwise would be wrong
    expected_slices = [[]]
    for i, window_end in enumerate(averaging_window_max_indices):
        expected_slices[0].append(slice(i * dt, window_end))
    assert expected_slices == coupler.averaging_slices

    interval = 2
    data_time_step = "3h"
    coupler.compute_coupled_indices(interval, data_time_step)
    coupled_integration_dim = presteps + max(output_time_dim // input_time_dim, 1)
    expected = np.empty([batch_size, coupled_integration_dim, len(input_times)])
    for b in range(batch_size):
        for i in range(coupled_integration_dim):
            expected[b, i, :] = (
                b
                + (input_time_dim * i + 1) * interval
                + np.array(
                    [
                        pd.Timedelta(ts) / pd.Timedelta(data_time_step)
                        for ts in input_times
                    ]
                )
            )
    expected = expected.astype(int)
    assert np.array_equal(expected, coupler._coupled_offsets)

    scaling_df = pd.DataFrame.from_dict(omegaconf.OmegaConf.to_object(scaling_dict)).T
    scaling_df.loc["zeros"] = {"mean": 0.0, "std": 1.0}
    scaling_da = scaling_df.to_xarray().astype("float32")
    coupler.set_scaling(scaling_da)
    coupled_scaling = scaling_da.sel(index=variables).rename({"index": "channel_in"})
    expected = np.expand_dims(coupled_scaling["mean"].to_numpy(), (0, 2, 3, 4))
    assert np.array_equal(expected, coupler.coupled_scaling["mean"])
    expected = np.expand_dims(coupled_scaling["std"].to_numpy(), (0, 2, 3, 4))
    assert np.array_equal(expected, coupler.coupled_scaling["std"])

    averaging_window_max_indices = [
        i // pd.Timedelta(data_time_step) for i in coupler.input_times
    ]
    di = averaging_window_max_indices[0]
    averaging_slices = []
    for j in range(coupler.coupled_integration_dim):
        averaging_slices.append([])
        for i, r in enumerate(averaging_window_max_indices):
            averaging_slices[j].append(
                slice(
                    coupler.input_time_dim * j * di + i * di,
                    coupler.input_time_dim * j * di + r,
                )
            )
    coupler.averaging_slices = averaging_slices
    coupler.coupled_channel_indices = [0, 1]

    coupled_fields_batch_size = batch_size
    coupled_fields_timedim = 4
    expected_shape = [
        coupler.coupled_integration_dim,
        coupled_fields_batch_size,
        coupler.timevar_dim,
    ] + list(coupler.spatial_dims)
    coupled_fields = th.rand(
        coupled_fields_batch_size,
        coupler.spatial_dims[0],
        coupled_fields_timedim,
        len(coupler.coupled_channel_indices),
        coupler.spatial_dims[1],
        coupler.spatial_dims[2],
    )
    coupler.set_coupled_fields(coupled_fields)
    assert list(coupler.preset_coupled_fields.shape) == expected_shape

    # check reset
    assert coupler.coupled_mode
    coupler.reset_coupler()
    assert coupler.coupled_mode is False

    DistributedManager.cleanup()


@import_or_fail("omegaconf")
@import_or_fail("netCDF4")
@import_or_fail("xarray")
@nfsdata_or_fail
def test_CoupledTimeSeriesDatasetZarr_initialization(
    dataset_path, scaling_dict, pytestconfig
):

    from physicsnemo.datapipes.healpix.coupledtimeseries_dataset_zarr import (
        CoupledTimeSeriesDatasetZarr,
    )

    # open our test dataset
    time_da = xr.open_zarr(dataset_path).time.values
    input_variables = ["z500", "z1000"]
    valid_start_date = "1979-01-01"
    valid_end_date = "1979-01-02"

    # check for failure of timestep not being a multiple of datatime step
    with pytest.raises(
        ValueError, match=("'time_step' must be a multiple of 'data_time_step' ")
    ):
        timeseries_ds = CoupledTimeSeriesDatasetZarr(
            dataset_path=dataset_path,
            input_variables=input_variables,
            data_time_step="2h",
            time_step="5h",
            scaling=scaling_dict,
            forecast_init_times=time_da[:2],
        )

    # check for failure of gap not being a multiple of datatime step
    with pytest.raises(
        ValueError, match=("'gap' must be a multiple of 'data_time_step' ")
    ):
        timeseries_ds = CoupledTimeSeriesDatasetZarr(
            dataset_path=dataset_path,
            input_variables=input_variables,
            data_time_step="2h",
            time_step="6h",
            gap="3h",
            scaling=scaling_dict,
            batch_size=1,
            forecast_init_times=time_da[:2],
        )

    # check for failure of invalid scaling variable on input
    invalid_scaling = omegaconf.DictConfig(
        {
            "bogosity": {"mean": 0, "std": 42},
        }
    )
    with pytest.raises(KeyError, match=("Input channels ")):
        timeseries_ds = CoupledTimeSeriesDatasetZarr(
            dataset_path=dataset_path,
            input_variables=input_variables,
            data_time_step="3h",
            time_step="6h",
            scaling=invalid_scaling,
            batch_size=1,
            forecast_init_times=time_da[:2],
        )

    # check for warning on batch size > 1 and forecast mode
    warnings.filterwarnings("error")
    with pytest.raises(
        UserWarning,
        match=(
            "providing 'forecast_init_times' to TimeSeriesDataset requires `batch_size=1`"
        ),
    ):
        timeseries_ds = CoupledTimeSeriesDatasetZarr(
            dataset_path=dataset_path,
            input_variables=input_variables,
            scaling=scaling_dict,
            batch_size=2,
            forecast_init_times=time_da[:2],
        )
    warnings.resetwarnings()

    timeseries_ds = CoupledTimeSeriesDatasetZarr(
        dataset_path=dataset_path,
        input_variables=input_variables,
        scaling=scaling_dict,
        add_train_noise=True,
        batch_size=1,
        forecast_init_times=time_da[:2],
    )
    assert isinstance(timeseries_ds, CoupledTimeSeriesDatasetZarr)

    timeseries_ds = CoupledTimeSeriesDatasetZarr(
        dataset_path=dataset_path,
        input_variables=input_variables,
        scaling=scaling_dict,
        start_date=valid_start_date,
        end_date=valid_end_date,
    )
    assert isinstance(timeseries_ds, CoupledTimeSeriesDatasetZarr)

    timeseries_ds = CoupledTimeSeriesDatasetZarr(
        dataset_path=dataset_path,
        input_variables=input_variables,
        scaling=scaling_dict,
        batch_size=1,
        forecast_init_times=time_da[:2],
    )
    assert isinstance(timeseries_ds, CoupledTimeSeriesDatasetZarr)

    timeseries_ds = CoupledTimeSeriesDatasetZarr(
        dataset_path=dataset_path,
        input_variables=input_variables,
        scaling=scaling_dict,
        batch_size=1,
        forecast_init_times=time_da[:2],
        data_time_step="3h",
        time_step="6h",
    )
    assert isinstance(timeseries_ds, CoupledTimeSeriesDatasetZarr)

    DistributedManager.cleanup()


@import_or_fail("omegaconf")
@import_or_fail("netCDF4")
@import_or_fail("xarray")
@nfsdata_or_fail
def test_CoupledTimeSeriesDatasetZarr_get_constants(
    dataset_path, scaling_dict, constant_coupler_config, pytestconfig
):

    from physicsnemo.datapipes.healpix.coupledtimeseries_dataset_zarr import (
        CoupledTimeSeriesDatasetZarr,
    )

    input_variables = ["z500", "z1000"]
    constant_variables = ["lsm"]

    # open our test dataset
    zarr_ds = xr.open_zarr(dataset_path)
    constant_indices = [
        int(np.where(zarr_ds.channel_c[:] == ch)[0][0]) for ch in constant_variables
    ]

    timeseries_ds = CoupledTimeSeriesDatasetZarr(
        dataset_path=dataset_path,
        input_variables=input_variables,
        batch_size=1,
        scaling=scaling_dict,
        couplings=constant_coupler_config,
        forecast_init_times=zarr_ds.time[:2],
    )
    assert timeseries_ds.get_constants() is None

    timeseries_ds = CoupledTimeSeriesDatasetZarr(
        dataset_path=dataset_path,
        input_variables=input_variables,
        constant_variables=constant_variables,
        batch_size=1,
        scaling=scaling_dict,
        couplings=constant_coupler_config,
        forecast_init_times=zarr_ds.time[:2],
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
    DistributedManager.cleanup()


@import_or_fail("omegaconf")
@import_or_fail("netCDF4")
@import_or_fail("xarray")
@nfsdata_or_fail
def test_CoupledTimeSeriesDatasetZarr_len(
    dataset_path, scaling_dict, constant_coupler_config, pytestconfig
):
    from physicsnemo.datapipes.healpix.coupledtimeseries_dataset_zarr import (
        CoupledTimeSeriesDatasetZarr,
    )

    # open our test dataset
    zarr_ds = xr.open_zarr(dataset_path)

    variables = ["z500", "z1000"]
    batch_size = 2

    # check forecast mode
    init_times = random.randint(1, zarr_ds.time.shape[0])
    timeseries_ds = CoupledTimeSeriesDatasetZarr(
        dataset_path=dataset_path,
        input_variables=variables,
        scaling=scaling_dict,
        batch_size=1,
        forecast_init_times=zarr_ds.time[:init_times],
        couplings=constant_coupler_config,
    )
    assert len(timeseries_ds) == init_times

    batch2_coupler = constant_coupler_config.copy()
    batch2_coupler[0]["params"]["batch_size"] = 2

    # get the last index that's evenly divisible by 3 (9h / 3h)
    last_index = (zarr_ds.time.shape[0] // 3) * 3 - 1

    # check train mode
    timeseries_ds = CoupledTimeSeriesDatasetZarr(
        dataset_path=dataset_path,
        input_variables=variables,
        data_time_step="3h",
        time_step="9h",
        scaling=scaling_dict,
        batch_size=batch_size,
        couplings=batch2_coupler,
        start_date=zarr_ds.time[0].values,
        end_date=zarr_ds.time[last_index - 1].values,
    )
    # Window length of 3 for one sample size
    assert len(timeseries_ds) == (zarr_ds.time.shape[0] - 3) // batch_size

    # drop incomplete last window
    timeseries_ds = CoupledTimeSeriesDatasetZarr(
        dataset_path=dataset_path,
        input_variables=variables,
        data_time_step="3h",
        time_step="9h",
        scaling=scaling_dict,
        batch_size=batch_size,
        drop_last=True,
        couplings=batch2_coupler,
        start_date=zarr_ds.time[0].values,
        end_date=zarr_ds.time[last_index - 1].values,
    )
    assert len(timeseries_ds) == (zarr_ds.time.shape[0] - 4) // batch_size

    zarr_ds.close()
    DistributedManager.cleanup()


@import_or_fail("omegaconf")
@import_or_fail("netCDF4")
@import_or_fail("xarray")
@nfsdata_or_fail
def test_CoupledTimeSeriesDatasetZarr_get(
    dataset_path, scaling_double_dict, splits, constant_coupler_config, pytestconfig
):
    from physicsnemo.datapipes.healpix.coupledtimeseries_dataset_zarr import (
        CoupledTimeSeriesDatasetZarr,
    )

    # open our test dataset
    zarr_ds = xr.open_zarr(dataset_path)

    input_variables = list(zarr_ds.channel_out.values)
    constant_variables = ["lsm"]
    batch_size = 2
    batch2_constant_coupler = constant_coupler_config.copy()
    batch2_constant_coupler[0]["params"]["batch_size"] = 2

    timeseries_ds = CoupledTimeSeriesDatasetZarr(
        dataset_path=dataset_path,
        input_variables=input_variables,
        scaling=scaling_double_dict,
        batch_size=batch_size,
        start_date=zarr_ds.time[0].values,
        end_date=zarr_ds.time[-1].values,
        couplings=batch2_constant_coupler,
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

    # this time dropping incomplete so that we get a full sample
    timeseries_ds = CoupledTimeSeriesDatasetZarr(
        dataset_path=dataset_path,
        input_variables=input_variables,
        scaling=scaling_double_dict,
        batch_size=batch_size,
        drop_last=True,
        start_date=zarr_ds.time[0].values,
        end_date=zarr_ds.time[-1].values,
        couplings=batch2_constant_coupler,
    )

    inputs, targets = timeseries_ds[-1]
    targets_expected = zarr_ds.targets[-1 - batch_size].transpose(
        "face", "channel_out", "height", "width"
    )
    targets_expected = targets_expected.to_numpy() / 2
    assert np.array_equal(targets[0][:, 0, :, :], targets_expected)

    # without couplings
    timeseries_ds = CoupledTimeSeriesDatasetZarr(
        dataset_path=dataset_path,
        input_variables=input_variables,
        constant_variables=constant_variables,
        scaling=scaling_double_dict,
        batch_size=batch_size,
        drop_last=True,
        start_date=zarr_ds.time[0].values,
        end_date=zarr_ds.time[-1].values,
        couplings=[],
    )
    non_perturbed_inputs = timeseries_ds
    assert len(non_perturbed_inputs[0][0]) == 2  # just inputs and targets

    # without couplings but with noise
    noise_params = {
        "inputs": scaling_double_dict,
        "couplings": scaling_double_dict,
    }
    timeseries_ds = CoupledTimeSeriesDatasetZarr(
        dataset_path=dataset_path,
        input_variables=input_variables,
        constant_variables=constant_variables,
        scaling=scaling_double_dict,
        batch_size=batch_size,
        drop_last=True,
        add_train_noise=True,
        train_noise_params=noise_params,
        start_date=zarr_ds.time[0].values,
        end_date=zarr_ds.time[-1].values,
        couplings=[],
    )
    perturbed_inputs = timeseries_ds
    # The first input will be the same sample, with perturbation it should have
    # different values
    assert non_perturbed_inputs[0][0][0].shape == perturbed_inputs[0][0][0].shape
    assert not np.array_equal(non_perturbed_inputs[0][0][0], perturbed_inputs[0][0][0])

    # With insolation we get 1 extra channel
    timeseries_ds = CoupledTimeSeriesDatasetZarr(
        dataset_path=dataset_path,
        input_variables=input_variables,
        constant_variables=None,
        scaling=scaling_double_dict,
        batch_size=batch_size,
        drop_last=True,
        add_insolation=True,
        start_date=zarr_ds.time[0].values,
        end_date=zarr_ds.time[-1].values,
        couplings=batch2_constant_coupler,
    )
    assert (len(inputs)) + 1 == len(timeseries_ds[0][0])

    # nothing should change with forecast mode other than getting just inputs
    init_times = random.randint(1, len(zarr_ds.time.values))
    timeseries_ds = CoupledTimeSeriesDatasetZarr(
        dataset_path=dataset_path,
        input_variables=input_variables,
        constant_variables=None,
        scaling=scaling_double_dict,
        batch_size=1,
        start_date=zarr_ds.time[0].values,
        end_date=zarr_ds.time[-1].values,
        couplings=constant_coupler_config,
    )
    inputs = timeseries_ds[0]

    assert np.array_equal(targets[0][:, 0, :, :], targets_expected)

    # insolation adds 1 extra channel
    init_times = random.randint(1, len(zarr_ds.time.values))
    timeseries_ds = CoupledTimeSeriesDatasetZarr(
        dataset_path=dataset_path,
        input_variables=input_variables,
        constant_variables=None,
        scaling=scaling_double_dict,
        batch_size=1,
        add_insolation=True,
        forecast_init_times=zarr_ds.time[:init_times],
        couplings=constant_coupler_config,
    )
    assert (len(inputs)) + 1 == len(timeseries_ds[0])

    # Constants + insolation is 2 extra channels
    init_times = random.randint(1, len(zarr_ds.time.values))
    timeseries_ds = CoupledTimeSeriesDatasetZarr(
        dataset_path=dataset_path,
        input_variables=input_variables,
        constant_variables=constant_variables,
        scaling=scaling_double_dict,
        batch_size=1,
        add_insolation=True,
        forecast_init_times=zarr_ds.time[:init_times],
        couplings=constant_coupler_config,
    )
    assert len(inputs) + 2 == len(timeseries_ds[0])

    zarr_ds.close()
    DistributedManager.cleanup()


@import_or_fail("omegaconf")
@import_or_fail("netCDF4")
@import_or_fail("xarray")
@nfsdata_or_fail
def test_CoupledTimeSeriesDataModuleZarr_initialization(
    dataset_path, splits, scaling_double_dict, constant_coupler_config, pytestconfig
):

    from physicsnemo.datapipes.healpix.data_modules_zarr import (
        CoupledTimeSeriesDataModuleZarr,
    )

    input_variables = ["z500", "z1000"]

    # open our test dataset
    zarr_ds = xr.open_zarr(dataset_path)

    # test for invalid path
    with pytest.raises(FileNotFoundError, match=("Dataset path not found")):
        timeseries_dm = CoupledTimeSeriesDataModuleZarr(
            dataset_path="DoesntExist",
            input_variables=input_variables,
            batch_size=1,
            couplings=constant_coupler_config,
        )

    # use the prebuilt dataset
    # Internally initializes DistributedManager
    timeseries_dm = CoupledTimeSeriesDataModuleZarr(
        dataset_path=dataset_path,
        input_variables=input_variables,
        batch_size=1,
        scaling=scaling_double_dict,
        splits=omegaconf.DictConfig(splits),
        couplings=constant_coupler_config,
    )
    assert isinstance(timeseries_dm, CoupledTimeSeriesDataModuleZarr)

    # with init times
    timeseries_dm = CoupledTimeSeriesDataModuleZarr(
        dataset_path=dataset_path,
        input_variables=input_variables,
        batch_size=1,
        scaling=scaling_double_dict,
        forecast_init_times=zarr_ds.time[:2],
        couplings=constant_coupler_config,
    )
    assert isinstance(timeseries_dm, CoupledTimeSeriesDataModuleZarr)

    # with splits
    timeseries_dm = CoupledTimeSeriesDataModuleZarr(
        dataset_path=dataset_path,
        input_variables=input_variables,
        batch_size=1,
        scaling=scaling_double_dict,
        splits=omegaconf.DictConfig(splits),
        couplings=constant_coupler_config,
    )
    assert isinstance(timeseries_dm, CoupledTimeSeriesDataModuleZarr)

    zarr_ds.close()
    DistributedManager.cleanup()


@import_or_fail("omegaconf")
@import_or_fail("netCDF4")
@import_or_fail("xarray")
@nfsdata_or_fail
def test_CoupledTimeSeriesDataModuleZarr_get_constants(
    dataset_path, scaling_double_dict, splits, constant_coupler_config, pytestconfig
):

    from physicsnemo.datapipes.healpix.data_modules_zarr import (
        CoupledTimeSeriesDataModuleZarr,
    )

    variables = ["z500", "z1000"]
    constants = ["lsm"]

    # No constants
    # Internally initializes DistributedManager
    timeseries_dm = CoupledTimeSeriesDataModuleZarr(
        dataset_path=dataset_path,
        input_variables=variables,
        batch_size=1,
        scaling=scaling_double_dict,
        splits=splits,
        constant_variables=None,
        couplings=constant_coupler_config,
    )

    assert timeseries_dm.get_constants() is None

    # just lsm as constant
    timeseries_dm = CoupledTimeSeriesDataModuleZarr(
        dataset_path=dataset_path,
        input_variables=variables,
        batch_size=1,
        scaling=scaling_double_dict,
        splits=splits,
        constant_variables=constants,
        couplings=constant_coupler_config,
    )

    # open our test dataset
    zarr_ds = xr.open_zarr(dataset_path)

    # divide by 2 due to scaling
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
    timeseries_dm = CoupledTimeSeriesDataModuleZarr(
        dataset_path=dataset_path,
        input_variables=variables,
        batch_size=1,
        scaling=scaling_double_dict,
        splits=splits,
        constant_variables=constants,
        couplings=constant_coupler_config,
    )

    assert np.array_equal(
        timeseries_dm.get_constants(),
        expected,
    )
    zarr_ds.close()
    DistributedManager.cleanup()


@import_or_fail("omegaconf")
@nfsdata_or_fail
def test_CoupledTimeSeriesDataModuleZarr_get_dataloaders(
    dataset_path, scaling_double_dict, splits, constant_coupler_config, pytestconfig
):

    from physicsnemo.datapipes.healpix.data_modules_zarr import (
        CoupledTimeSeriesDataModuleZarr,
    )

    input_variables = ["z500", "z1000"]

    # use the prebuilt dataset
    # Internally initializes DistributedManager
    timeseries_dm = CoupledTimeSeriesDataModuleZarr(
        dataset_path=dataset_path,
        input_variables=input_variables,
        batch_size=1,
        scaling=scaling_double_dict,
        splits=splits,
        shuffle=False,
        couplings=constant_coupler_config,
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


@import_or_fail("omegaconf")
@nfsdata_or_fail
def test_CoupledTimeSeriesDataModuleZarr_get_coupled_vars(
    dataset_path,
    scaling_double_dict,
    splits,
    constant_coupler_config,
    average_coupler_config,
    pytestconfig,
):
    from physicsnemo.datapipes.healpix.data_modules_zarr import (
        CoupledTimeSeriesDataModuleZarr,
    )

    input_variables = ["z500", "z1000"]

    # Constant coupler
    # Internally initializes DistributedManager
    timeseries_dm = CoupledTimeSeriesDataModuleZarr(
        dataset_path=dataset_path,
        input_variables=input_variables,
        batch_size=1,
        scaling=scaling_double_dict,
        splits=splits,
        couplings=constant_coupler_config,
    )

    outvar = timeseries_dm._get_coupled_vars()
    outvar.sort()
    expected = ["z250"]
    expected.sort()

    assert expected == outvar

    # Average coupler
    # Internally initializes DistributedManager
    timeseries_dm = CoupledTimeSeriesDataModuleZarr(
        dataset_path=dataset_path,
        input_variables=input_variables,
        batch_size=1,
        scaling=scaling_double_dict,
        splits=splits,
        couplings=average_coupler_config,
    )
    outvar = timeseries_dm._get_coupled_vars()
    outvar.sort()

    assert expected == outvar

    DistributedManager.cleanup()


@import_or_fail("omegaconf")
@import_or_fail("netCDF4")
@import_or_fail("xarray")
@nfsdata_or_fail
def test_CoupledTimeSeriesDatasetZarr_next_integration(
    dataset_path, scaling_dict, pytestconfig
):
    from physicsnemo.datapipes.healpix.coupledtimeseries_dataset_zarr import (
        CoupledTimeSeriesDatasetZarr,
    )

    spatial_dims = [12, 32, 32]
    input_variables = ["z500", "z1000"]
    coupled_channel_indices = [0, 1]
    coupled_variables = ["z250"]
    num_variables = len(input_variables)
    input_time_dim = 1
    output_time_dim = 1
    batch_size = 1

    constant_coupler = [
        {
            "coupler": "ConstantCoupler",
            "params": {
                "batch_size": 1,
                "variables": coupled_variables,
                "input_times": ["0h"],
                "input_time_dim": input_time_dim,
                "output_time_dim": output_time_dim,
                "presteps": 0,
                "prepared_coupled_data": True,
            },
        }
    ]

    # open our test dataset
    ds = xr.open_zarr(dataset_path)
    init_times = random.randint(1, len(ds.time.values))
    # channels need to be subselected before being handed over
    test_ds = ds.sel(
        channel_in=input_variables + coupled_variables,
        channel_out=input_variables,
    )

    timeseries_ds = CoupledTimeSeriesDatasetZarr(
        dataset_path=dataset_path,
        input_variables=input_variables,
        scaling=scaling_dict,
        batch_size=batch_size,
        couplings=constant_coupler,
        data_time_step="6h",
        time_step="6h",
        drop_last=True,
        add_insolation=True,
        forecast_init_times=test_ds.time[:init_times],
    )

    test_model_outputs = th.rand(
        1,
        spatial_dims[0],
        output_time_dim,
        num_variables,
        spatial_dims[1],
        spatial_dims[2],
    )
    constants = np.transpose(ds.constants.values, axes=(1, 0, 2, 3))
    coupled_fields = th.rand(
        batch_size,
        spatial_dims[0],
        input_time_dim + output_time_dim,
        len(input_variables),
        spatial_dims[1],
        spatial_dims[2],
    )

    expected_coupling = coupled_fields[:, :, :, coupled_channel_indices, :, :].permute(
        2, 0, 3, 1, 4, 5
    )
    expected_coupling = expected_coupling[0, :, -1, :, :, :]
    expected_coupling = expected_coupling.unsqueeze(0).unsqueeze(0)
    expected_coupling = expected_coupling.repeat(1, batch_size, 1, 1, 1, 1)

    # need to grab at least 1 sample to properly intialize everything
    timeseries_ds[0]
    # hacky way to setup the indices since we don't actually have any coupled fields
    timeseries_ds.couplings[0].coupled_channel_indices = coupled_channel_indices

    # set the coupled fields
    timeseries_ds.couplings[0].set_coupled_fields(coupled_fields)
    test_integration = timeseries_ds.next_integration(test_model_outputs, constants)
    # test to make sure prognostics are used, constants stay the same, and couplings
    # are what we set
    assert np.array_equal(test_integration[0], test_model_outputs[:, :, -1:])
    assert np.array_equal(test_integration[2], constants)
    assert np.array_equal(test_integration[3], expected_coupling)

    # I have absolutely no idea why a coupled dataset has the option for 0 couplings
    timeseries_ds = CoupledTimeSeriesDatasetZarr(
        dataset_path=dataset_path,
        input_variables=input_variables,
        scaling=scaling_dict,
        batch_size=batch_size,
        couplings=[],
        data_time_step="6h",
        time_step="6h",
        drop_last=True,
        add_insolation=True,
        forecast_init_times=test_ds.time[:init_times],
    )
    # need to grab at least 1 sample to properly intialize everything
    timeseries_ds[0]
    test_integration = timeseries_ds.next_integration(test_model_outputs, constants)
    assert np.array_equal(test_integration[0], test_model_outputs[:, :, -1:])
    assert np.array_equal(test_integration[2], constants)
