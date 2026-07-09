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
import torch as th

from physicsnemo.distributed import DistributedManager
from test.conftest import requires_module
from test.datapipes.healpix.conftest import assert_shard_dataloaders, coupler_helper

omegaconf = pytest.importorskip("omegaconf")
np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
xr = pytest.importorskip("xarray")


@requires_module("omegaconf")
@requires_module("netCDF4")
@requires_module("pandas")
@requires_module("xarray")
def test_ConstantCoupler(data_dir, dataset_name, scaling_dict, pytestconfig):
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
    ds_path = Path(data_dir, dataset_name + ".zarr")
    zarr_ds = xr.open_zarr(ds_path)

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

    # `set_coupled_fields` derives its batch dimension from the shape of
    # `coupled_fields` itself rather than validating it against the
    # configured `batch_size` (an older physicsnemo implementation raised a
    # ValueError here, but the ported DLESyM coupler intentionally adapts to
    # whatever batch size is provided)
    coupler.coupled_channel_indices = [0, 1]
    coupled_fields_batch_size = batch_size * 2
    coupled_fields_timedim = 2
    coupled_fields = th.rand(
        coupled_fields_batch_size,
        coupler.spatial_dims[0],
        coupled_fields_timedim,
        len(coupler.coupled_channel_indices),
        coupler.spatial_dims[1],
        coupler.spatial_dims[2],
    )
    coupler.set_coupled_fields(coupled_fields)
    assert coupler.preset_coupled_fields.shape[1] == coupled_fields_batch_size

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

    # verify that the data is being properly transformed: set_coupled_fields
    # broadcasts the first time step of coupled_fields across the integration
    # dim and permutes to [integration_dim, batch, timevar, face, height, width]
    expected = coupled_fields[:, :, :, coupler.coupled_channel_indices, :, :]
    expected = expected[:, :, :1, :, :, :]
    expected = expected.permute(2, 0, 3, 1, 4, 5)
    assert th.equal(expected, coupler.construct_integrated_couplings())

    # test coupler reset
    coupler.reset_coupler()
    assert coupler.coupled_mode is False

    # once reset, batch/bsize are required to construct the couplings from
    # the dataset rather than from preset (cached) fields
    with pytest.raises(
        ValueError, match=("batch and bsize must be provided when not in coupled_mode")
    ):
        coupler.construct_integrated_couplings()

    # test loading from the dataset
    coupled_scaling = {
        "mean": np.expand_dims(coupled_scaling["mean"].to_numpy(), (0, 2, 3, 4)),
        "std": np.expand_dims(coupled_scaling["std"].to_numpy(), (0, 2, 3, 4)),
    }
    expected = zarr_ds.sel(channel_in=variables).inputs[:2].values
    expected = (expected - coupled_scaling["mean"]) / coupled_scaling["std"]
    coupled_field = coupler.construct_integrated_couplings(
        batch=batch, bsize=batch_size
    )
    assert np.array_equal(expected, coupled_field[0])

    # set_scaling raises if a coupled variable has no corresponding entry in
    # the provided scaling values
    scaling_missing_var = scaling_da.drop_sel(index="z1000")
    with pytest.raises(
        KeyError, match=("Coupled variable\\(s\\) not found in scaling values")
    ):
        coupler.set_scaling(scaling_missing_var)

    zarr_ds.close()
    DistributedManager.cleanup()


@requires_module("omegaconf")
@requires_module("netCDF4")
@requires_module("pandas")
@requires_module("xarray")
def test_TrailingAverageCoupler(data_dir, dataset_name, scaling_dict, pytestconfig):
    from physicsnemo.datapipes.healpix.couplers import (
        TrailingAverageCoupler,
    )

    variables = ["z500", "z1000"]
    input_times = ["6h", "12h"]
    input_time_dim = 2
    output_time_dim = 2
    presteps = 0
    batch_size = 2
    averaging_window = "6h"
    # open our test dataset
    ds_path = Path(data_dir, dataset_name + ".zarr")
    zarr_ds = xr.open_zarr(ds_path)

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

    # `set_coupled_fields` derives its batch dimension from the shape of
    # `coupled_fields` itself rather than validating it against the
    # configured `batch_size` (an older physicsnemo implementation raised a
    # ValueError here, but the ported DLESyM coupler intentionally adapts to
    # whatever batch size is provided)
    coupled_fields_batch_size = batch_size * 2
    coupled_fields_timedim = 4
    coupled_fields = th.rand(
        coupled_fields_batch_size,
        coupler.spatial_dims[0],
        coupled_fields_timedim,
        len(coupler.coupled_channel_indices),
        coupler.spatial_dims[1],
        coupler.spatial_dims[2],
    )
    coupler.set_coupled_fields(coupled_fields)
    assert coupler.preset_coupled_fields.shape[1] == coupled_fields_batch_size

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

    zarr_ds.close()
    DistributedManager.cleanup()


@requires_module("omegaconf")
@requires_module("netCDF4")
@requires_module("pandas")
@requires_module("xarray")
def test_TrailingAverageCoupler_multiple_variables(data_dir, dataset_name):
    """Regression test to ensure `set_coupled_fields` correctly averages and
    threads through every coupled variable, not just the first. Gives each
    coupled variable a distinct, constant value in the synthetic
    `coupled_fields` tensor so that averaging over any time window still
    yields that same constant, letting each variable's contribution to
    `preset_coupled_fields` be checked in isolation and by position.
    """
    from physicsnemo.datapipes.healpix.couplers import TrailingAverageCoupler

    variables = ["z500", "z1000", "z250"]
    input_times = ["6h", "12h"]
    input_time_dim = 2
    output_time_dim = 2
    batch_size = 2
    averaging_window = "6h"
    ds_path = Path(data_dir, dataset_name + ".zarr")
    zarr_ds = xr.open_zarr(ds_path)

    coupler = TrailingAverageCoupler(
        dataset=zarr_ds,
        batch_size=batch_size,
        variables=variables,
        presteps=0,
        averaging_window=averaging_window,
        input_times=input_times,
        input_time_dim=input_time_dim,
        output_time_dim=output_time_dim,
    )
    coupler.coupled_channel_indices = list(range(len(variables)))

    data_time_step = "3h"
    averaging_window_max_indices = [
        i // pd.Timedelta(data_time_step) for i in input_times
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

    # give each coupled variable a distinct, constant value across every
    # other dimension so the averaging window doesn't change its value, and
    # each variable's contribution can be checked independently of the
    # others
    channel_values = [100.0, 200.0, 300.0]
    coupled_fields = th.empty(
        batch_size,
        coupler.spatial_dims[0],
        4,
        len(variables),
        coupler.spatial_dims[1],
        coupler.spatial_dims[2],
    )
    for i, value in enumerate(channel_values):
        coupled_fields[:, :, :, i, :, :] = value

    coupler.set_coupled_fields(coupled_fields)
    result = coupler.construct_integrated_couplings()
    assert list(result.shape) == [
        coupler.coupled_integration_dim,
        batch_size,
        coupler.timevar_dim,
    ] + list(coupler.spatial_dims)

    # timevar ordering groups by averaging period first, then by variable
    # (matching the dataset-loading path in
    # `_construct_integrated_couplings_from_dataset`/`construct_integrated_couplings`)
    for period in range(len(input_times)):
        for var_idx, value in enumerate(channel_values):
            timevar_idx = period * len(variables) + var_idx
            slice_result = result[:, :, timevar_idx, :, :, :]
            assert th.allclose(slice_result, th.full_like(slice_result, value)), (
                f"coupled variable {variables[var_idx]!r} (value {value}) was not "
                f"preserved at averaging period {period}, timevar index {timevar_idx}"
            )

    zarr_ds.close()
    DistributedManager.cleanup()


@requires_module("omegaconf")
@requires_module("netCDF4")
@requires_module("pandas")
@requires_module("xarray")
def test_ConstantCoupler_multiple_variables_order(data_dir, dataset_name):
    """Sanity check that the xarray-backed coupling path preserves the order
    of `variables` as requested (rather than the dataset's native channel_in
    order) when 3+ coupled variables are requested out of native order. This
    mirrors the invariant that the Zarr-backed path also has to uphold (see
    the analogous, previously-failing test in test_healpix_couple_zarr.py).
    """
    from physicsnemo.datapipes.healpix.couplers import ConstantCoupler

    ds_path = Path(data_dir, dataset_name + ".zarr")
    zarr_ds = xr.open_zarr(ds_path)

    native_order = list(zarr_ds.channel_in.values)
    variables = ["z250", "z1000", "z500"]
    assert [native_order.index(v) for v in variables] == sorted(
        [native_order.index(v) for v in variables], reverse=True
    )

    batch_size = 2
    batch = {"time": slice(0, 2)}
    coupler = ConstantCoupler(
        dataset=zarr_ds,
        batch_size=batch_size,
        variables=variables,
        input_times=["0h"],
        input_time_dim=1,
        output_time_dim=1,
    )
    coupler.compute_coupled_indices(interval=2, data_time_step="3h")

    coupled_field = coupler.construct_integrated_couplings(
        batch=batch, bsize=batch_size
    )
    for i, var in enumerate(variables):
        expected_var = zarr_ds.sel(channel_in=var).inputs[:2].values
        assert np.array_equal(expected_var, coupled_field[0][:, i])

    zarr_ds.close()
    DistributedManager.cleanup()


@requires_module("omegaconf")
@requires_module("netCDF4")
@requires_module("xarray")
def test_CoupledTimeSeriesDataset_initialization(
    data_dir, dataset_name, scaling_dict, pytestconfig
):
    from physicsnemo.datapipes.healpix.coupledtimeseries_dataset import (
        CoupledTimeSeriesDataset,
    )

    # open our test dataset
    ds_path = Path(data_dir, dataset_name + ".zarr")
    zarr_ds = xr.open_zarr(ds_path)
    variables = ["z500", "z1000"]

    # check for failure of timestep not being a multiple of datatime step
    with pytest.raises(
        ValueError, match=("'time_step' must be a multiple of 'data_time_step' ")
    ):
        timeseries_ds = CoupledTimeSeriesDataset(
            dataset=zarr_ds,
            input_variables=variables,
            data_time_step="2h",
            time_step="5h",
            scaling=scaling_dict,
        )

    # check for failure of gap not being a multiple of datatime step
    with pytest.raises(
        ValueError, match=("'gap' must be a multiple of 'data_time_step' ")
    ):
        timeseries_ds = CoupledTimeSeriesDataset(
            dataset=zarr_ds,
            input_variables=variables,
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
        timeseries_ds = CoupledTimeSeriesDataset(
            dataset=zarr_ds,
            input_variables=variables,
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
        timeseries_ds = CoupledTimeSeriesDataset(
            dataset=zarr_ds,
            input_variables=variables,
            scaling=scaling_dict,
            batch_size=2,
            forecast_init_times=zarr_ds.time[:2],
        )

    timeseries_ds = CoupledTimeSeriesDataset(
        dataset=zarr_ds,
        input_variables=variables,
        scaling=scaling_dict,
        add_train_noise=True,
    )
    assert isinstance(timeseries_ds, CoupledTimeSeriesDataset)

    timeseries_ds = CoupledTimeSeriesDataset(
        dataset=zarr_ds,
        input_variables=variables,
        scaling=scaling_dict,
    )
    assert isinstance(timeseries_ds, CoupledTimeSeriesDataset)

    timeseries_ds = CoupledTimeSeriesDataset(
        dataset=zarr_ds,
        input_variables=variables,
        scaling=scaling_dict,
        batch_size=1,
        forecast_init_times=zarr_ds.time[:2],
    )
    assert isinstance(timeseries_ds, CoupledTimeSeriesDataset)

    timeseries_ds = CoupledTimeSeriesDataset(
        dataset=zarr_ds,
        input_variables=variables,
        scaling=scaling_dict,
        batch_size=1,
        forecast_init_times=zarr_ds.time[:2],
        data_time_step="3h",
        time_step="6h",
    )
    assert isinstance(timeseries_ds, CoupledTimeSeriesDataset)

    zarr_ds.close()
    DistributedManager.cleanup()


@requires_module("omegaconf")
@requires_module("netCDF4")
@requires_module("xarray")
def test_CoupledTimeSeriesDataset_get_constants(
    data_dir, dataset_name, scaling_dict, constant_coupler_config, pytestconfig
):
    from physicsnemo.datapipes.healpix.coupledtimeseries_dataset import (
        CoupledTimeSeriesDataset,
    )

    # open our test dataset
    ds_path = Path(data_dir, dataset_name + ".zarr")
    zarr_ds = xr.open_zarr(ds_path)

    variables = ["z500", "z1000"]

    timeseries_ds = CoupledTimeSeriesDataset(
        dataset=zarr_ds,
        input_variables=variables,
        scaling=scaling_dict,
        couplings=constant_coupler_config,
    )

    # constants are reshaped
    expected = np.transpose(zarr_ds.constants.values, axes=(1, 0, 2, 3))
    outvar = timeseries_ds.get_constants()
    assert np.array_equal(
        expected,
        outvar,
    )

    zarr_ds.close()
    DistributedManager.cleanup()


@requires_module("omegaconf")
@requires_module("netCDF4")
@requires_module("xarray")
def test_CoupledTimeSeriesDataset_len(
    data_dir, dataset_name, scaling_dict, constant_coupler_config, pytestconfig
):
    from physicsnemo.datapipes.healpix.coupledtimeseries_dataset import (
        CoupledTimeSeriesDataset,
    )

    # open our test dataset
    ds_path = Path(data_dir, dataset_name + ".zarr")
    zarr_ds = xr.open_zarr(ds_path)

    variables = ["z500", "z1000"]

    # check forecast mode
    init_times = random.randint(1, len(zarr_ds.time.values))
    timeseries_ds = CoupledTimeSeriesDataset(
        dataset=zarr_ds,
        input_variables=variables,
        scaling=scaling_dict,
        batch_size=1,
        forecast_init_times=zarr_ds.time[:init_times],
        couplings=constant_coupler_config,
    )
    assert len(timeseries_ds) == init_times

    constant_coupler = constant_coupler_config.copy()
    constant_coupler[0]["params"]["batch_size"] = 2

    # check train mode
    timeseries_ds = CoupledTimeSeriesDataset(
        dataset=zarr_ds,
        input_variables=variables,
        data_time_step="3h",
        time_step="9h",
        scaling=scaling_dict,
        batch_size=2,
        couplings=constant_coupler,
    )
    # Window length of 3 for one sample size
    assert len(timeseries_ds) == (len(zarr_ds.time.values) - 2) // 2

    # check train mode
    timeseries_ds = CoupledTimeSeriesDataset(
        dataset=zarr_ds,
        input_variables=variables,
        data_time_step="3h",
        time_step="9h",
        scaling=scaling_dict,
        batch_size=2,
        drop_last=True,
        couplings=constant_coupler,
    )
    assert len(timeseries_ds) == (len(zarr_ds.time.values) - 2) // 2

    zarr_ds.close()
    DistributedManager.cleanup()


@requires_module("omegaconf")
@requires_module("netCDF4")
@requires_module("xarray")
def test_CoupledTimeSeriesDataset_get(
    data_dir, dataset_name, scaling_double_dict, constant_coupler_config, pytestconfig
):
    from physicsnemo.datapipes.healpix.coupledtimeseries_dataset import (
        CoupledTimeSeriesDataset,
    )

    # open our test dataset
    ds_path = Path(data_dir, dataset_name + ".zarr")
    zarr_ds = xr.open_zarr(ds_path)

    variables = list(zarr_ds.channel_out.to_numpy())
    # "z250" is coupled in below rather than being a regular input variable:
    # channel_in scaling/slicing assumes channel_in == input_variables +
    # coupled variables (with the coupled ones appended last), so a coupled
    # variable can't also be one of input_variables
    coupled_input_variables = [v for v in variables if v != "z250"]

    batch_size = 2
    constant_coupler = constant_coupler_config.copy()
    constant_coupler[0]["params"]["batch_size"] = batch_size
    timeseries_ds = CoupledTimeSeriesDataset(
        dataset=zarr_ds,
        input_variables=coupled_input_variables,
        output_variables=variables,
        scaling=scaling_double_dict,
        batch_size=batch_size,
        couplings=constant_coupler,
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
    timeseries_ds = CoupledTimeSeriesDataset(
        dataset=zarr_ds,
        input_variables=coupled_input_variables,
        output_variables=variables,
        scaling=scaling_double_dict,
        batch_size=batch_size,
        drop_last=True,
        couplings=constant_coupler,
    )

    inputs, targets = timeseries_ds[-1]
    targets_expected = zarr_ds.targets[-1 - batch_size].transpose(
        "face", "channel_out", "height", "width"
    )
    targets_expected = targets_expected.to_numpy() / 2
    assert np.array_equal(targets[0][:, 0, :, :], targets_expected)

    # without couplings
    timeseries_ds = CoupledTimeSeriesDataset(
        dataset=zarr_ds,
        input_variables=variables,
        scaling=scaling_double_dict,
        batch_size=batch_size,
        drop_last=True,
        couplings=[],
    )
    non_perturbed_inputs = timeseries_ds
    assert len(non_perturbed_inputs[0][0]) == 2  # just inputs and targets

    # wihtout couplings but with noise
    noise_params = {
        "inputs": scaling_double_dict,
        "couplings": scaling_double_dict,
    }
    timeseries_ds = CoupledTimeSeriesDataset(
        dataset=zarr_ds,
        input_variables=variables,
        scaling=scaling_double_dict,
        batch_size=batch_size,
        drop_last=True,
        add_train_noise=True,
        train_noise_params=noise_params,
        couplings=[],
    )
    perturbed_inputs = timeseries_ds
    # The first input will be the same sample, with perturbation it should have
    # different values
    assert non_perturbed_inputs[0][0][0].shape == perturbed_inputs[0][0][0].shape
    assert not np.array_equal(non_perturbed_inputs[0][0][0], perturbed_inputs[0][0][0])

    # noise is also added to the integrated couplings themselves, not just
    # the regular inputs, when couplings are present. `inputs_result` here is
    # [inputs, constants, integrated_couplings] because the dataset carries a
    # "constants" data variable which the base class picks up unconditionally
    timeseries_ds = CoupledTimeSeriesDataset(
        dataset=zarr_ds,
        input_variables=coupled_input_variables,
        output_variables=variables,
        scaling=scaling_double_dict,
        batch_size=batch_size,
        drop_last=True,
        couplings=constant_coupler,
    )
    non_perturbed_sample = timeseries_ds[0][0]
    assert len(non_perturbed_sample) == 3
    non_perturbed_couplings = non_perturbed_sample[2]

    timeseries_ds = CoupledTimeSeriesDataset(
        dataset=zarr_ds,
        input_variables=coupled_input_variables,
        output_variables=variables,
        scaling=scaling_double_dict,
        batch_size=batch_size,
        drop_last=True,
        add_train_noise=True,
        train_noise_params=noise_params,
        couplings=constant_coupler,
    )
    perturbed_couplings = timeseries_ds[0][0][2]
    assert non_perturbed_couplings.shape == perturbed_couplings.shape
    assert not np.array_equal(non_perturbed_couplings, perturbed_couplings)

    # With insolation we get 1 extra channel
    timeseries_ds = CoupledTimeSeriesDataset(
        dataset=zarr_ds,
        input_variables=coupled_input_variables,
        output_variables=variables,
        scaling=scaling_double_dict,
        batch_size=batch_size,
        drop_last=True,
        add_insolation=True,
        couplings=constant_coupler,
    )
    assert (len(inputs)) + 1 == len(timeseries_ds[0][0])

    # nothing should change with forecast mode other than getting just inputs
    init_times = random.randint(1, len(zarr_ds.time.values))
    timeseries_ds = CoupledTimeSeriesDataset(
        dataset=zarr_ds,
        input_variables=coupled_input_variables,
        output_variables=variables,
        scaling=scaling_double_dict,
        batch_size=1,
        forecast_init_times=zarr_ds.time[:init_times],
        couplings=constant_coupler,
    )
    inputs = timeseries_ds[0]

    assert np.array_equal(targets[0][:, 0, :, :], targets_expected)

    # insolation adds 1 extra channel
    init_times = random.randint(1, len(zarr_ds.time.values))
    timeseries_ds = CoupledTimeSeriesDataset(
        dataset=zarr_ds,
        input_variables=coupled_input_variables,
        output_variables=variables,
        scaling=scaling_double_dict,
        batch_size=1,
        add_insolation=True,
        forecast_init_times=zarr_ds.time[:init_times],
        couplings=constant_coupler,
    )
    assert (len(inputs)) + 1 == len(timeseries_ds[0])

    # No constants in input data
    init_times = random.randint(1, len(zarr_ds.time.values))
    zarr_ds_no_const = zarr_ds.drop_vars("constants")
    timeseries_ds = CoupledTimeSeriesDataset(
        dataset=zarr_ds_no_const,
        input_variables=coupled_input_variables,
        output_variables=variables,
        scaling=scaling_double_dict,
        batch_size=1,
        forecast_init_times=zarr_ds.time[:init_times],
        couplings=constant_coupler,
    )
    assert len(inputs) == (len(timeseries_ds[0]) + 1)

    zarr_ds.close()
    DistributedManager.cleanup()


@requires_module("omegaconf")
@requires_module("netCDF4")
@requires_module("xarray")
def test_CoupledTimeSeriesDataModule_initialization(
    data_dir,
    create_path,
    dataset_name,
    scaling_double_dict,
    splits,
    constant_coupler_config,
    pytestconfig,
):
    from physicsnemo.datapipes.healpix.data_modules import (
        CoupledTimeSeriesDataModule,
    )

    variables = ["z500", "z1000"]
    constant_coupler = constant_coupler_config

    # open our test dataset
    ds_path = Path(data_dir, dataset_name + ".zarr")
    zarr_ds = xr.open_zarr(ds_path)

    # check for failure when a requested input variable isn't in the dataset
    with pytest.raises(ValueError, match=("Input variables not found in dataset")):
        CoupledTimeSeriesDataModule(
            src_directory=create_path,
            dst_directory=data_dir,
            dataset_name=dataset_name,
            input_variables=variables + ["DoesntExist"],
            batch_size=1,
            prebuilt_dataset=True,
            scaling=scaling_double_dict,
            couplings=constant_coupler,
        )

    # check for failure when a requested output variable isn't in the dataset
    with pytest.raises(ValueError, match=("Output variables not found in dataset")):
        CoupledTimeSeriesDataModule(
            src_directory=create_path,
            dst_directory=data_dir,
            dataset_name=dataset_name,
            input_variables=variables,
            output_variables=variables + ["DoesntExist"],
            batch_size=1,
            prebuilt_dataset=True,
            scaling=scaling_double_dict,
            couplings=constant_coupler,
        )

    # check for failure when a requested constant isn't in the dataset
    with pytest.raises(ValueError, match=("Constants not found in dataset")):
        CoupledTimeSeriesDataModule(
            src_directory=create_path,
            dst_directory=data_dir,
            dataset_name=dataset_name,
            input_variables=variables,
            batch_size=1,
            prebuilt_dataset=True,
            scaling=scaling_double_dict,
            constants={"lsm": "DoesntExist"},
            couplings=constant_coupler,
        )

    # use the prebuilt dataset
    # Internally initializes DistributedManager
    timeseries_dm = CoupledTimeSeriesDataModule(
        src_directory=create_path,
        dst_directory=data_dir,
        dataset_name=dataset_name,
        input_variables=variables,
        batch_size=1,
        prebuilt_dataset=True,
        scaling=scaling_double_dict,
        couplings=constant_coupler,
    )
    assert isinstance(timeseries_dm, CoupledTimeSeriesDataModule)

    # `prebuilt_dataset` is kept only for backwards compatibility; on-the-fly
    # dataset generation has been removed, so `setup()` always opens an
    # existing prebuilt dataset from `dst_directory` regardless of this flag
    timeseries_dm = CoupledTimeSeriesDataModule(
        src_directory=create_path,
        dst_directory=data_dir,
        dataset_name=dataset_name,
        input_variables=variables,
        batch_size=1,
        prebuilt_dataset=False,
        scaling=scaling_double_dict,
        couplings=constant_coupler,
    )
    assert isinstance(timeseries_dm, CoupledTimeSeriesDataModule)

    # with init times
    timeseries_dm = CoupledTimeSeriesDataModule(
        src_directory=create_path,
        dst_directory=data_dir,
        dataset_name=dataset_name,
        input_variables=variables,
        batch_size=1,
        prebuilt_dataset=True,
        scaling=scaling_double_dict,
        forecast_init_times=zarr_ds.time[:2],
        couplings=constant_coupler,
    )
    assert isinstance(timeseries_dm, CoupledTimeSeriesDataModule)

    # with splits
    timeseries_dm = CoupledTimeSeriesDataModule(
        src_directory=create_path,
        dst_directory=data_dir,
        dataset_name=dataset_name,
        input_variables=variables,
        batch_size=1,
        prebuilt_dataset=True,
        scaling=scaling_double_dict,
        splits=omegaconf.DictConfig(splits),
        couplings=constant_coupler,
    )
    assert isinstance(timeseries_dm, CoupledTimeSeriesDataModule)
    zarr_ds.close()
    DistributedManager.cleanup()


@requires_module("omegaconf")
@requires_module("netCDF4")
@requires_module("xarray")
def test_CoupledTimeSeriesDataModule_get_constants(
    data_dir,
    create_path,
    dataset_name,
    scaling_double_dict,
    constant_coupler_config,
    pytestconfig,
):
    from physicsnemo.datapipes.healpix.data_modules import (
        CoupledTimeSeriesDataModule,
    )

    variables = ["z500", "z1000"]
    constants = {"lsm": "lsm"}
    constant_coupler = constant_coupler_config

    # No constants
    # Internally initializes DistributedManager
    timeseries_dm = CoupledTimeSeriesDataModule(
        src_directory=create_path,
        dst_directory=data_dir,
        dataset_name=dataset_name,
        input_variables=variables,
        batch_size=1,
        prebuilt_dataset=True,
        scaling=scaling_double_dict,
        constants=None,
        couplings=constant_coupler,
    )

    assert timeseries_dm.get_constants() is None

    # just lsm as constant
    timeseries_dm = CoupledTimeSeriesDataModule(
        src_directory=create_path,
        dst_directory=data_dir,
        dataset_name=dataset_name,
        input_variables=variables,
        batch_size=1,
        prebuilt_dataset=True,
        scaling=scaling_double_dict,
        constants=constants,
        couplings=constant_coupler,
    )

    # open our test dataset
    ds_path = Path(data_dir, dataset_name + ".zarr")
    zarr_ds = xr.open_zarr(ds_path)

    # divide by 2 due to scaling
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
    timeseries_dm = CoupledTimeSeriesDataModule(
        src_directory=create_path,
        dst_directory=data_dir,
        dataset_name=dataset_name,
        input_variables=variables,
        batch_size=1,
        prebuilt_dataset=True,
        scaling=scaling_double_dict,
        constants=constants,
        couplings=constant_coupler,
    )

    assert np.array_equal(
        timeseries_dm.get_constants(),
        expected,
    )
    zarr_ds.close()
    DistributedManager.cleanup()


@requires_module("omegaconf")
def test_CoupledTimeSeriesDataModule_get_dataloaders(
    data_dir,
    create_path,
    dataset_name,
    scaling_double_dict,
    splits,
    constant_coupler_config,
    pytestconfig,
):
    from physicsnemo.datapipes.healpix.data_modules import (
        CoupledTimeSeriesDataModule,
    )

    variables = ["z500", "z1000"]

    # use the prebuilt dataset
    # Internally initializes DistributedManager
    timeseries_dm = CoupledTimeSeriesDataModule(
        src_directory=create_path,
        dst_directory=data_dir,
        dataset_name=dataset_name,
        input_variables=variables,
        batch_size=1,
        prebuilt_dataset=True,
        scaling=scaling_double_dict,
        splits=splits,
        shuffle=False,
        couplings=constant_coupler_config,
    )

    assert_shard_dataloaders(timeseries_dm)
    DistributedManager.cleanup()


@requires_module("omegaconf")
def test_CoupledTimeSeriesDataModule_get_coupled_vars(
    data_dir,
    create_path,
    dataset_name,
    scaling_double_dict,
    constant_coupler_config,
    average_coupler_config,
    pytestconfig,
):
    from physicsnemo.datapipes.healpix.data_modules import (
        CoupledTimeSeriesDataModule,
    )

    variables = ["z500", "z1000"]

    # Constant coupler
    # Internally initializes DistributedManager
    timeseries_dm = CoupledTimeSeriesDataModule(
        src_directory=create_path,
        dst_directory=data_dir,
        dataset_name=dataset_name,
        input_variables=variables,
        batch_size=1,
        prebuilt_dataset=True,
        scaling=scaling_double_dict,
        couplings=constant_coupler_config,
    )

    outvar = timeseries_dm._get_coupled_vars()
    outvar.sort()
    expected = ["z250"]
    expected.sort()

    assert expected == outvar

    # Average coupler
    # Internally initializes DistributedManager
    timeseries_dm = CoupledTimeSeriesDataModule(
        src_directory=create_path,
        dst_directory=data_dir,
        dataset_name=dataset_name,
        input_variables=variables,
        batch_size=1,
        prebuilt_dataset=True,
        scaling=scaling_double_dict,
        couplings=average_coupler_config,
    )
    outvar = timeseries_dm._get_coupled_vars()
    outvar.sort()

    assert expected == outvar

    DistributedManager.cleanup()


@requires_module("omegaconf")
@requires_module("netCDF4")
@requires_module("xarray")
def test_CoupledTimeSeriesDataset_next_integration(
    data_dir, dataset_name, scaling_dict, constant_coupler_config, pytestconfig
):
    from physicsnemo.datapipes.healpix.coupledtimeseries_dataset import (
        CoupledTimeSeriesDataset,
    )

    spatial_dims = [12, 32, 32]
    input_variables = ["z500", "z1000"]
    # length must match len(coupled_variables) (i.e. the coupler's timevar_dim);
    # values are arbitrary since this synthetic setup bypasses setup_coupling()
    coupled_channel_indices = [0]
    coupled_variables = ["z250"]
    num_variables = len(input_variables)
    input_time_dim = 1
    output_time_dim = 1
    batch_size = 1
    constant_coupler = constant_coupler_config

    # open our test dataset
    ds_path = Path(data_dir, dataset_name + ".zarr")
    ds = xr.open_zarr(ds_path)
    init_times = random.randint(1, len(ds.time.values))
    # channels need to be subselected before being handed over
    test_ds = ds.sel(
        channel_in=input_variables + coupled_variables,
        channel_out=input_variables,
    )

    timeseries_ds = CoupledTimeSeriesDataset(
        dataset=test_ds,
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

    # mirrors set_coupled_fields: broadcast the first time step across the
    # integration dim and permute to [integration_dim, batch, timevar, face, height, width]
    expected_coupling = coupled_fields[:, :, :, coupled_channel_indices, :, :]
    expected_coupling = expected_coupling[:, :, :1, :, :, :]
    expected_coupling = expected_coupling.permute(2, 0, 3, 1, 4, 5)

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
    # with no coupler, channel_in must exactly match input_variables (no extra
    # coupled-variable channel appended), so re-select the dataset without
    # the "z250" channel that `test_ds` carries for the coupled case above
    test_ds_no_couplings = ds.sel(
        channel_in=input_variables,
        channel_out=input_variables,
    )
    timeseries_ds = CoupledTimeSeriesDataset(
        dataset=test_ds_no_couplings,
        input_variables=input_variables,
        scaling=scaling_dict,
        batch_size=batch_size,
        couplings=[],
        data_time_step="6h",
        time_step="6h",
        drop_last=True,
        add_insolation=True,
        forecast_init_times=test_ds_no_couplings.time[:init_times],
    )
    # need to grab at least 1 sample to properly intialize everything
    timeseries_ds[0]
    test_integration = timeseries_ds.next_integration(test_model_outputs, constants)
    assert np.array_equal(test_integration[0], test_model_outputs[:, :, -1:])
    assert np.array_equal(test_integration[2], constants)
