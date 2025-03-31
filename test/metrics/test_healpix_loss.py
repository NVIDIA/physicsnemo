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

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pytest
import torch
from pytest_utils import import_or_fail, nfsdata_or_fail

from physicsnemo.metrics.climate.healpix_loss import (
    BaseMSE,
    OceanMSE,
    WeightedMSE,
    WeightedOceanMSE,
)

xr = pytest.importorskip("xarray")


@pytest.fixture
def test_data():
    # create dummy data

    # We'll pretend h,w are a lat lon grid instead of healpix
    # so the test works the same
    # Set lat/lon in terms of degrees (for use with _compute_lat_weights)
    def generate_test_data(channels=2, img_shape=(768, 768)):
        x = np.linspace(-180, 180, img_shape[1], dtype=np.float32)
        y = np.linspace(-90, 90, img_shape[0], dtype=np.float32)
        xv, yv = np.meshgrid(x, y)

        pred_tensor_np = np.cos(2 * np.pi * yv / (180))
        targ_tensor_np = np.cos(np.pi * yv / (180))

        return channels, pred_tensor_np, targ_tensor_np

    return generate_test_data


@dataclass
class trainer_helper:
    """helper class for setup with the MSE testers"""

    output_variables: Sequence
    device: str


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_BaseMSE(device, test_data, rtol: float = 1e-3, atol: float = 1e-3):
    mse_func = BaseMSE()
    mse_func.setup(None)  # for coverage
    channels, pred_tensor_np, targ_tensor_np = test_data()

    pred_tensor = torch.from_numpy(pred_tensor_np).to(device).expand(channels, -1, -1)
    targ_tensor = torch.from_numpy(targ_tensor_np).to(device).expand(channels, -1, -1)

    # expand out to 6 dimensions
    pred_tensor = pred_tensor[(None,) * 3]
    targ_tensor = targ_tensor[(None,) * 3]

    # test for insufficient dimensions
    with pytest.raises(
        AssertionError, match="Expected predictions to have 6 dimensions"
    ):
        mse_func(torch.zeros((10,), device=device), targ_tensor)

    with pytest.raises(
        AssertionError, match="Expected predictions to have 6 dimensions"
    ):
        mse_func(targ_tensor, torch.zeros((10,), device=device))

    with pytest.raises(
        AssertionError, match="Expected predictions to have 6 dimensions"
    ):
        mse_func(torch.zeros((10,), device=device), torch.zeros((10,), device=device))

    # test for 0 loss
    error = mse_func(targ_tensor, targ_tensor)
    assert torch.allclose(
        error,
        torch.zeros([1], dtype=torch.float32, device=device),
        rtol=rtol,
        atol=atol,
    )

    # int( cos(x)^2 - cos(2x)^2 )dx, x = 0...2*pi = pi/4
    # So MSE should be pi/4 / (pi) = 0.25
    error = mse_func(pred_tensor**2, targ_tensor**2)
    assert torch.allclose(
        error,
        0.25 * torch.ones([1], dtype=torch.float32, device=device),
        rtol=rtol,
        atol=atol,
    )

    # test for non averaged channesl
    # make the last channel of prediction and target the same
    tensor_size = pred_tensor.shape[-2:]
    ones = torch.ones(tensor_size, device=device)

    pred_tensor = pred_tensor.contiguous()
    targ_tensor = targ_tensor.contiguous()
    pred_tensor[0, 0, 0, -1, ...] = ones[...]
    targ_tensor[0, 0, 0, -1, ...] = ones[...]

    error = mse_func(pred_tensor**2, targ_tensor**2, average_channels=False)

    expected_err = 0.25 * torch.ones(error.shape, dtype=torch.float32, device=device)
    expected_err[-1] = 0

    assert torch.allclose(
        error,
        expected_err,
        rtol=rtol,
        atol=atol,
    )


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_WeightedMSE(device, test_data, rtol: float = 1e-3, atol: float = 1e-3):
    num_channels = 3
    channels, pred_tensor_np, targ_tensor_np = test_data(channels=num_channels)

    # first two channels will be as BaseMSE above, the last channel will be 0 loss
    # so per channel loss will be [0.25, 0.25, 0]
    # Giving the last channel half the weight results in a per loss of:
    # [0.25*0.5, 0.25*0.25, 0.0*0.25] == [0.125,0.0625,0]
    # and an average weighted loss of 0.0625
    channel_weights = [0.5, 0.25, 0.25]
    channel_weighted_mse = torch.Tensor([0.125, 0.0625, 0]).to(device)
    mean_weighted_mse = channel_weighted_mse.mean()
    weighted_mse_func = WeightedMSE(channel_weights)

    trainer = trainer_helper(
        output_variables=["a", "b", "ones"],
        device=device,
    )
    weighted_mse_func.setup(trainer)

    # test setup fail case if number of variables doesn't match number of weights
    trainer = trainer_helper(
        output_variables=["a", "b"],
        device=device,
    )
    with pytest.raises(
        ValueError, match="Length of outputs and loss_weights is not the same!"
    ):
        weighted_mse_func.setup(trainer)

    pred_tensor = torch.from_numpy(pred_tensor_np).to(device).expand(channels, -1, -1)
    targ_tensor = torch.from_numpy(targ_tensor_np).to(device).expand(channels, -1, -1)

    tensor_size = pred_tensor.shape[-2:]
    ones = torch.ones(tensor_size, device=device)

    # make the last channel of prediction and target the same
    pred_tensor = pred_tensor.contiguous()
    targ_tensor = targ_tensor.contiguous()
    pred_tensor[-1, ...] = ones[...]
    targ_tensor[-1, ...] = ones[...]

    # expand out to 6 dimensions
    pred_tensor = pred_tensor[(None,) * 3]
    targ_tensor = targ_tensor[(None,) * 3]

    # test for insufficient dimensions
    with pytest.raises(
        AssertionError, match="Expected predictions to have 6 dimensions"
    ):
        weighted_mse_func(torch.zeros((10,), device=device), targ_tensor)

    with pytest.raises(
        AssertionError, match="Expected predictions to have 6 dimensions"
    ):
        weighted_mse_func(targ_tensor, torch.zeros((10,), device=device))

    with pytest.raises(
        AssertionError, match="Expected predictions to have 6 dimensions"
    ):
        weighted_mse_func(
            torch.zeros((10,), device=device), torch.zeros((10,), device=device)
        )

    # test for 0 loss
    error = weighted_mse_func(pred_tensor**2, pred_tensor**2, average_channels=True)
    assert torch.allclose(
        error,
        torch.zeros(1).to(device),
        rtol=rtol,
        atol=atol,
    )

    # test for individual channel loss
    error = weighted_mse_func(
        pred_tensor**2, targ_tensor**2, average_channels=False
    )
    assert torch.allclose(
        error,
        channel_weighted_mse,
        rtol=rtol,
        atol=atol,
    )

    # test with mean across channels
    error = weighted_mse_func(pred_tensor**2, targ_tensor**2, average_channels=True)
    assert torch.allclose(
        error,
        mean_weighted_mse,
        rtol=rtol,
        atol=atol,
    )


@pytest.fixture
def data_dir():
    path = "/data/nfs/modulus-data/datasets/healpix/"
    return path


@pytest.fixture
def dataset_name():
    name = "healpix"
    return name


@nfsdata_or_fail
@import_or_fail("xarray")
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_OceanMSE(
    data_dir,
    dataset_name,
    device,
    test_data,
    pytestconfig,
    rtol: float = 1e-3,
    atol: float = 1e-3,
):
    num_channels = 3
    channels, pred_tensor_np, targ_tensor_np = test_data(
        channels=num_channels, img_shape=(32, 32)
    )
    ds_path = Path(data_dir, dataset_name + ".zarr")

    lsm_ds = xr.open_dataset(ds_path, engine="zarr").constants.sel({"channel_c": "lsm"})

    channel_ocean_mse = torch.Tensor([0.2706, 0.2706, 0]).to(device)
    mean_ocean_mse = channel_ocean_mse.mean()
    ocean_mse_func = OceanMSE(ds_path)

    trainer = trainer_helper(
        output_variables=["a", "b", "ones"],
        device=device,
    )
    lsm_tensor = 1 - torch.tensor(np.expand_dims(lsm_ds.values, (0, 2, 3))).to(
        trainer.device
    )
    ocean_mse_func.setup(trainer)

    pred_tensor = torch.from_numpy(pred_tensor_np).to(device).expand(channels, -1, -1)
    targ_tensor = torch.from_numpy(targ_tensor_np).to(device).expand(channels, -1, -1)

    tensor_size = pred_tensor.shape[-2:]
    ones = torch.ones(tensor_size, device=device)

    # make the last channel of prediction and target the same
    pred_tensor = pred_tensor.contiguous()
    targ_tensor = targ_tensor.contiguous()
    pred_tensor[-1, ...] = ones[...]
    targ_tensor[-1, ...] = ones[...]

    # expand out to 6 dimensions
    pred_tensor = pred_tensor[(None,) * 3]
    targ_tensor = targ_tensor[(None,) * 3]

    pred_tensor = pred_tensor.expand(1, lsm_tensor.shape[1], 1, -1, -1, -1)
    targ_tensor = targ_tensor.expand(1, lsm_tensor.shape[1], 1, -1, -1, -1)

    # test for 0 loss
    error = ocean_mse_func(pred_tensor**2, pred_tensor**2, average_channels=True)
    assert torch.allclose(
        error,
        torch.zeros(1).to(device),
        rtol=rtol,
        atol=atol,
    )

    # test for individual channels
    error = ocean_mse_func(pred_tensor**2, targ_tensor**2, average_channels=False)
    assert torch.allclose(
        error,
        channel_ocean_mse,
        rtol=rtol,
        atol=atol,
    )

    # test for mean across channels
    error = ocean_mse_func(pred_tensor**2, targ_tensor**2, average_channels=True)
    assert torch.allclose(
        error,
        mean_ocean_mse,
        rtol=rtol,
        atol=atol,
    )


@nfsdata_or_fail
@import_or_fail("xarray")
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_WeightedOceanMSE(
    data_dir,
    dataset_name,
    device,
    test_data,
    pytestconfig,
    rtol: float = 1e-3,
    atol: float = 1e-3,
):
    num_channels = 3
    identity_weights = [1, 1, 1]  # same as OceanMSE
    test_weights = [2.0, 0.5, 1]  # Check positive and negative weighing factors
    test_weights_tensor = torch.Tensor(test_weights).to(device)
    channels, pred_tensor_np, targ_tensor_np = test_data(
        channels=num_channels, img_shape=(32, 32)
    )
    ds_path = Path(data_dir, dataset_name + ".zarr")

    lsm_ds = xr.open_dataset(ds_path, engine="zarr").constants.sel({"channel_c": "lsm"})

    # setup target weights
    atmos_channel_mse = torch.Tensor([0.2706, 0.2706, 0]).to(device)
    mean_atmos_channel_mse = atmos_channel_mse.mean()
    weighted_atmos_channel_mse = atmos_channel_mse * test_weights_tensor
    mean_weighted_atmos_channel_mse = weighted_atmos_channel_mse.mean()

    trainer = trainer_helper(
        output_variables=["a", "b", "ones"],
        device=device,
    )
    lsm_tensor = 1 - torch.tensor(np.expand_dims(lsm_ds.values, (0, 2, 3))).to(
        trainer.device
    )

    # expand to 4 dims C, F, H, W
    pred_tensor = torch.from_numpy(pred_tensor_np).to(device).expand(channels, -1, -1)
    targ_tensor = torch.from_numpy(targ_tensor_np).to(device).expand(channels, -1, -1)

    tensor_size = pred_tensor.shape[-2:]
    ones = torch.ones(tensor_size, device=device)

    # make the last channel of prediction and target the same
    pred_tensor = pred_tensor.contiguous()
    targ_tensor = targ_tensor.contiguous()
    pred_tensor[-1, ...] = ones[...]
    targ_tensor[-1, ...] = ones[...]

    # expand out to 6 dimensions
    pred_tensor = pred_tensor[(None,) * 3]
    targ_tensor = targ_tensor[(None,) * 3]

    # fit to LSM field size
    pred_tensor = pred_tensor.expand(1, lsm_tensor.shape[1], 1, -1, -1, -1)
    targ_tensor = targ_tensor.expand(1, lsm_tensor.shape[1], 1, -1, -1, -1)

    # Test mismatch between weights and number of variables
    weighted_ocean_mse_func = WeightedOceanMSE(ds_path)
    with pytest.raises(
        ValueError, match="Length of outputs and loss_weights is not the same!"
    ):
        weighted_ocean_mse_func.setup(trainer)

    # Test with identity weights, same as OceanMSE
    weighted_ocean_mse_func = WeightedOceanMSE(ds_path, weights=identity_weights)
    weighted_ocean_mse_func.setup(trainer)

    # test for 0 loss
    error = weighted_ocean_mse_func(
        pred_tensor**2, pred_tensor**2, average_channels=True
    )
    assert torch.allclose(
        error,
        torch.zeros(1).to(device),
        rtol=rtol,
        atol=atol,
    )

    # test identity on individual channels
    error = weighted_ocean_mse_func(
        pred_tensor**2, targ_tensor**2, average_channels=False
    )
    assert torch.allclose(
        error,
        atmos_channel_mse,
        rtol=rtol,
        atol=atol,
    )

    # test identity on mean across channels
    error = weighted_ocean_mse_func(
        pred_tensor**2, targ_tensor**2, average_channels=True
    )
    assert torch.allclose(
        error,
        mean_atmos_channel_mse,
        rtol=rtol,
        atol=atol,
    )

    # Test with different weights
    weighted_ocean_mse_func = WeightedOceanMSE(ds_path, weights=test_weights)
    weighted_ocean_mse_func.setup(trainer)

    # test identity on individual channels
    error = weighted_ocean_mse_func(
        pred_tensor**2, targ_tensor**2, average_channels=False
    )
    assert torch.allclose(
        error,
        weighted_atmos_channel_mse,
        rtol=rtol,
        atol=atol,
    )

    # test identity on mean across channels
    error = weighted_ocean_mse_func(
        pred_tensor**2, targ_tensor**2, average_channels=True
    )
    assert torch.allclose(
        error,
        mean_weighted_atmos_channel_mse,
        rtol=rtol,
        atol=atol,
    )
