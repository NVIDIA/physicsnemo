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

import pytest
import torch
from torch.nn import functional as F

from physicsnemo.metrics.climate.loss import MSE_SSIM, SSIM


@dataclass
class Model(torch.nn.Module):
    """Minimal torch.nn.Module to test MSE_SSIM"""

    output_channels = 2
    output_time_dim = 1
    input_time_dim = 1
    output_variables = ["tcwv", "t2m"]


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_MSE_SSIM(device):
    model = Model()

    # test for invalid weights

    with pytest.raises(
        ValueError, match="Weights passed to MSE_SSIM loss must sum to 1"
    ):
        mse_ssim_loss = MSE_SSIM(weights=[1.0, 1.0])

    # test with no params
    mse_ssim_loss = MSE_SSIM()

    # test for fail case of invalid dims (h != w)
    shape = [1, 1, 1, 12, 64, 32]
    ones = torch.ones(shape).to(device)
    with pytest.raises(
        AssertionError, match="Spatial dims H and W must match: got 64 and 32"
    ):
        mse_ssim_loss(ones, ones, model)

    # test for fail case of F != 12
    shape = [1, 1, 2, 8, 32, 32]
    ones = torch.ones(shape).to(device)
    with pytest.raises(AssertionError, match="Spatial dim F must be 12: got 8"):
        mse_ssim_loss(ones, ones, model)

    # test for fail case of number of channels not matching number of weights
    shape = [1, 1, 4, 12, 32, 32]
    ones = torch.ones(shape).to(device)
    with pytest.raises(
        AssertionError,
        match="model output channels and prediction output channels don't match: got 2 for model and 4 for input",
    ):
        mse_ssim_loss(ones, ones, model)

    # test for fail case of wrong number of time dims
    shape = [1, 5, 2, 12, 32, 32]
    ones = torch.ones(shape).to(device)
    with pytest.raises(
        AssertionError,
        match="Number of time steps in prediction must equal to model output time dim, or model output time dime divided by model input time dim",
    ):
        mse_ssim_loss(ones, ones, model)

    shape = [1, 1, 2, 12, 32, 32]
    ones = torch.ones(shape).to(device)
    zeros = torch.zeros(shape).to(device)

    assert mse_ssim_loss(ones, ones, model) == 0
    assert mse_ssim_loss(ones, zeros, model) == 1

    invar = torch.ones(shape).to(device)
    invar[0, 0, 1, ...] = zeros[0, 0, 0, ...]
    assert mse_ssim_loss(invar, zeros, model) == 0.5

    # test with SSIM and 1 SSIM variable and 1 non SSIM variable
    ssim_variables = ["t2m", "u10m"]
    ssim_params = {
        "window_size": 11,
        "padding_mode": "constant",
        "time_series_forecasting": True,
    }
    mse_ssim_loss = MSE_SSIM(ssim_variables=ssim_variables, ssim_params=ssim_params)

    assert mse_ssim_loss(invar, zeros, model) == 0.5

    # test providing all params
    weights = [0.75, 0.25]
    mse_params = {
        "reduction": "sum",
    }
    ssim_variables = ["ttr1h", "tcwv0"]  # ['t2m', 'u10m']
    ssim_params = {
        "window_size": 11,
        "padding_mode": "constant",
        "time_series_forecasting": True,
    }
    mse_ssim_loss = MSE_SSIM(
        weights=weights,
        mse_params=mse_params,
        ssim_params=ssim_params,
        ssim_variables=ssim_variables,
    )

    # since we're using sum as the reduction instead of mean
    # difference is c * h * w
    expected = ones[0, 0, 0, ...].sum()
    assert mse_ssim_loss(ones, zeros, model) == expected


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_SSIM(device):
    """Test SSIM loss in loss"""
    ssim_loss = SSIM(time_series_forecasting=False)

    four_dim_var = torch.randn([2, 2, 2, 2]).to(device)
    six_dim_var = torch.randn([2, 2, 2, 2, 2, 2]).to(device)

    # Test fail cases
    # dimensions don't match
    with pytest.raises(
        AssertionError,
        match="Predicted and target tensor need to have the same number of dimensions",
    ):
        ssim_loss(four_dim_var, six_dim_var)

    # wrong dimensions without timeseries forecasting
    with pytest.raises(
        AssertionError,
        match=("Need 4 or 5 dimensions when not using time series forecasting"),
    ):
        ssim_loss(six_dim_var, six_dim_var)

    # wrong dimensions with timeseries forecasting
    ssim_loss = SSIM(time_series_forecasting=True)
    with pytest.raises(
        AssertionError,
        match=("Need 5 or 6 dimensions when using time series forecasting"),
    ):
        ssim_loss(four_dim_var, four_dim_var)

    ssim_loss = SSIM(time_series_forecasting=True)
    # 5 or 6 time series True
    # 4 or 5 time series False
    shape = [1, 1, 2, 2, 360, 360]

    # Test for exact match
    rand = torch.randn(shape).to(device)

    assert ssim_loss(rand, rand) == 1.0

    # Test for differences
    ones = torch.ones(shape).to(device)
    zeros = torch.zeros(shape).to(device)

    # test for opposite
    assert ssim_loss(ones, zeros) < 1.0e-4

    # test mask
    mask = torch.ones([1, 1, 2, 2, 180, 180], dtype=int)

    ssim_rand = ssim_loss(rand, ones)
    ssim_rand_mask = ssim_loss(rand, ones, mask)

    assert ssim_rand != ssim_rand_mask

    # Test window
    # Since SSIM looks over a window rolling will only cause a small dropoff
    ssim_loss = SSIM(mse=True)
    eye = torch.eye(360).to(device)
    eye = eye[None, None, ...]

    loss = ssim_loss(eye, torch.roll(eye, 1, -1))  # ~0.9516
    assert 0.95 < loss < 0.96

    # test the case of being below the mse_epoch threshold
    loss = ssim_loss(eye, torch.roll(eye, 1, -1), epoch=-1)  # ~0.9516
    assert loss == F.mse_loss(eye, torch.roll(eye, 1, -1))
