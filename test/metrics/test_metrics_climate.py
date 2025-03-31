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

import numpy as np
import pytest
import torch

import physicsnemo.metrics.climate.efi as efi
import physicsnemo.metrics.climate.reduction as clim_red
import physicsnemo.metrics.general.histogram as hist
import physicsnemo.metrics.general.reduction as gen_red
from physicsnemo.metrics.climate.acc import acc
from physicsnemo.metrics.general.mse import mse, rmse

Tensor = torch.Tensor


@pytest.fixture
def test_data(channels=2, img_shape=(721, 1440)):
    # create dummy data
    time_means = (
        np.pi / 2 * np.ones((channels, img_shape[0], img_shape[1]), dtype=np.float32)
    )

    # Set lat/lon in terms of degrees (for use with _compute_lat_weights)
    x = np.linspace(-180, 180, img_shape[1], dtype=np.float32)
    y = np.linspace(-90, 90, img_shape[0], dtype=np.float32)
    xv, yv = np.meshgrid(x, y)

    pred_tensor_np = np.cos(2 * np.pi * yv / (180))
    targ_tensor_np = np.cos(np.pi * yv / (180))

    return channels, x, y, pred_tensor_np, targ_tensor_np, time_means


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_climate_acc_mse(test_data, device, rtol: float = 1e-3, atol: float = 1e-3):
    channels, lon, lat, pred_tensor_np, targ_tensor_np, time_means = test_data
    lat = torch.from_numpy(lat).to(device)
    lon = torch.from_numpy(lon).to(device)

    pred_tensor = torch.from_numpy(pred_tensor_np).expand(channels, -1, -1).to(device)
    targ_tensor = torch.from_numpy(targ_tensor_np).expand(channels, -1, -1).to(device)
    means_tensor = torch.from_numpy(time_means).to(device)

    # Independent of the time means, the ACC score for cos(2*x) and cos(x) is 1/8 π sqrt(15/(32 - 3 π^2))
    # or about 0.98355. For derivation, note that the lat weight gives an extra factor of cos(x)/2 and
    # p1 = int[ (cos(x) -y - E[cos(x)-y]) * (cos(2x) - y - E[cos(2x)-y])] = pi/24
    # p2 = int[ (cos(2x) - y - E[cos(x) - y])^2 cos(x)/2 ] = 16/45
    # p3 = int[ (cos(x) - y - E[cos(x) - y])^2 cos(x)/2 ] = 2/3 - pi^2/16 (here E[.] denotes mean)
    # and acc = p / sqrt(p2 * p3) = 1/8 π sqrt(15/(32 - 3 π^2))

    acc_ = acc(pred_tensor, targ_tensor, means_tensor, lat)
    assert torch.allclose(
        acc_,
        0.9836 * torch.ones(channels).to(device),
        rtol=rtol,
        atol=atol,
    )

    # Test exceptions
    with pytest.raises(
        AssertionError, match="Expected predictions to have at least two dimensions"
    ):
        acc(torch.zeros((10,), device=device), targ_tensor, means_tensor, lat)

    with pytest.raises(
        AssertionError, match="Expected predictions to have at least two dimensions"
    ):
        acc(pred_tensor, torch.zeros((10,), device=device), means_tensor, lat)

    with pytest.raises(
        AssertionError, match="Expected predictions to have at least two dimensions"
    ):
        acc(pred_tensor, targ_tensor, torch.zeros((10,), device=device), lat)

    # int( cos(x)^2 - cos(2x)^2 )dx, x = 0...2*pi = pi/4
    # So MSE should be pi/4 / (pi) = 0.25
    error = mse(pred_tensor**2, targ_tensor**2, dim=(1, 2))
    rerror = rmse(pred_tensor**2, targ_tensor**2, dim=(1, 2))
    assert torch.allclose(
        error,
        0.25 * torch.ones([1], dtype=torch.float32, device=device),
        rtol=rtol,
        atol=atol,
    )
    assert torch.allclose(
        rerror,
        np.sqrt(0.25) * torch.ones([1], dtype=torch.float32, device=device),
        rtol=rtol,
        atol=atol,
    )


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_climate_reductions(test_data, device, rtol: float = 1e-3, atol: float = 1e-3):
    channels, lon, lat, pred_tensor_np, targ_tensor_np, time_means = test_data
    pred_tensor = torch.from_numpy(pred_tensor_np).expand(channels, -1, -1).to(device)
    lat = torch.from_numpy(lat).to(device)
    weights = clim_red._compute_lat_weights(lat)
    # Check main class
    ws = gen_red.WeightedStatistic(weights)
    # Check that it normalizes
    assert torch.allclose(
        torch.sum(ws.weights),
        torch.ones([1], dtype=torch.float32, device=device),
        rtol=rtol,
        atol=atol,
    )

    # Test assertion error
    with pytest.raises(ValueError):
        gen_red.WeightedStatistic(-1.0 * weights)

    with pytest.raises(ValueError):
        www = gen_red.WeightedStatistic(torch.rand((10,), device=device))
        www(torch.randn((13, 13), device=device), dim=1)

    with pytest.raises(ValueError):
        www = gen_red.WeightedStatistic(torch.rand((10, 10), device=device))
        www(torch.randn((13, 13), device=device), dim=1)

    # Check when weights are 1 dimensional
    weights = weights.flatten()
    wm = gen_red.WeightedMean(weights)
    our_weighted_mean = wm(pred_tensor, dim=1)
    np_weighted_mean = np.average(pred_tensor.cpu(), weights=weights.cpu(), axis=1)
    assert torch.allclose(
        our_weighted_mean,
        torch.from_numpy(np_weighted_mean).to(device),
        rtol=rtol,
        atol=atol,
    )

    # Check when weights are same shape as pred_tensor
    weights = weights.unsqueeze(0)
    weights = weights.unsqueeze(-1)
    wm = gen_red.WeightedMean(weights)
    our_weighted_mean = wm(pred_tensor, dim=1)

    np_weighted_mean = np.average(
        pred_tensor.cpu(), weights=weights.flatten().cpu(), axis=1
    )
    assert torch.allclose(
        our_weighted_mean,
        torch.from_numpy(np_weighted_mean).to(device),
        rtol=rtol,
        atol=atol,
    )

    # Check zonal mean == our_weighted_mean
    zonal_mean = clim_red.zonal_mean(pred_tensor, lat, dim=1)
    assert torch.allclose(
        our_weighted_mean,
        zonal_mean,
        rtol=rtol,
        atol=atol,
    )

    # Test variance
    # Check when weights are 1 dimensional
    weights = weights.flatten()
    wv = clim_red.WeightedVariance(weights)
    our_weighted_var = wv(pred_tensor, dim=1)

    np_weighted_mean = np.average(pred_tensor.cpu(), weights=weights.cpu(), axis=1)
    np_weighted_var = np.average(
        (pred_tensor.cpu() - np_weighted_mean[:, None, ...]) ** 2,
        weights=weights.cpu(),
        axis=1,
    )
    assert torch.allclose(
        our_weighted_var,
        torch.from_numpy(np_weighted_var).to(device),
        rtol=rtol,
        atol=atol,
    )

    # Check zonal var == our_weighted_var
    zonal_var = clim_red.zonal_var(pred_tensor, lat, dim=1)
    zonal_std = clim_red.zonal_var(pred_tensor, lat, dim=1, std=True)
    assert torch.allclose(
        our_weighted_var,
        zonal_var,
        rtol=rtol,
        atol=atol,
    )
    assert torch.allclose(
        torch.sqrt(our_weighted_var),
        zonal_std,
        rtol=rtol,
        atol=atol,
    )

    # Check global means and vars
    global_mean = clim_red.global_mean(pred_tensor, lat)
    assert torch.allclose(
        torch.mean(our_weighted_mean, dim=-1),
        global_mean,
        rtol=rtol,
        atol=atol,
    )

    # Test Raises Assertion
    with pytest.raises(
        AssertionError,
        match="Expected x to have at least two dimensions, with the last two dimensions representing lat and lon respectively",
    ):
        clim_red.global_mean(torch.zeros((10,), device=device), lat)

    # Global variance of cos(2x) should be
    # int[ (cos(2x) - E[cos(2x)])^2 * cos(2x)/2 ] dx
    # = int[ (cos(2x) - 1/3)^2 * cos(2x)/2 ] dx
    # = 16/45
    global_var = clim_red.global_var(pred_tensor, lat)
    global_std = clim_red.global_var(pred_tensor, lat, std=True)
    assert torch.allclose(
        16 / 45 * torch.ones([1], device=device),
        global_var,
        rtol=rtol,
        atol=atol,
    )
    assert torch.allclose(
        4 / 3 / np.sqrt(5) * torch.ones([1], device=device),
        global_std,
        rtol=rtol,
        atol=atol,
    )

    # Test Raises Assertion
    with pytest.raises(
        AssertionError,
        match="Expected x to have at least two dimensions, with the last two dimensions representing lat and lon respectively",
    ):
        clim_red.global_var(torch.zeros((10,), device=device), lat)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_climate_efi(test_data, device, rtol: float = 1e-1, atol: float = 1e-1):

    one = torch.ones((1, 1), dtype=torch.float32, device=device)
    bin_edges = hist.linspace(-10 * one, 10 * one, 30)
    bin_mids = 0.5 * bin_edges[1:] + 0.5 * bin_edges[:-1]

    clim_mean = torch.zeros((1, 1), dtype=torch.float32, device=device)
    clim_std = torch.ones((1, 1), dtype=torch.float32, device=device)

    # Test normal pdf and cdf
    _, test_counts = hist.histogram(
        torch.randn(1_000_000, 1, 1, dtype=torch.float32, device=device), bins=bin_edges
    )
    test_pdf = test_counts / torch.trapz(test_counts, bin_mids, dim=0)
    test_cdf = torch.cumsum(
        test_counts / torch.sum(test_counts, dim=0, keepdims=True), dim=0
    )

    clim_pdf = hist.normal_pdf(clim_mean, clim_std, bin_edges, grid="right")
    clim_cdf = hist.normal_cdf(clim_mean, clim_std, bin_edges, grid="right")
    assert torch.allclose(
        torch.trapz((clim_pdf - test_pdf) ** 2, bin_mids, dim=0),
        0.0 * one,
        rtol=rtol,
        atol=atol,
    )
    assert torch.allclose(
        torch.trapz((clim_cdf - test_cdf) ** 2, bin_mids, dim=0),
        0.0 * one,
        rtol=rtol,
        atol=atol,
    )

    x = torch.randn((1_000_000, 1, 1), dtype=torch.float32, device=device)
    _, cdf = hist.cdf(x, bins=bin_edges)
    e = efi.efi_gaussian(cdf, bin_edges, clim_mean, clim_std)
    _, pdf = hist.histogram(x, bins=bin_edges)
    e1 = efi.efi(bin_edges, pdf, clim_cdf)
    assert torch.allclose(e, 0.0 * one, rtol=rtol, atol=atol)
    assert torch.allclose(e1, 0.0 * one, rtol=rtol, atol=atol)
    assert torch.allclose(e, e1, rtol=rtol, atol=atol)

    x = 2.0 + 2.0 * torch.randn((1_000_000, 1, 1), dtype=torch.float32, device=device)
    _, cdf = hist.cdf(x, bins=bin_edges)
    e = efi.efi_gaussian(cdf, bin_edges, clim_mean, clim_std)
    _, pdf = hist.histogram(x, bins=bin_edges)
    e1 = efi.efi(bin_edges, pdf, clim_cdf)
    assert torch.all(torch.ge(e, 0.0 * one))
    assert torch.all(torch.ge(e1, 0.0 * one))
    assert torch.allclose(e, e1, rtol=rtol, atol=atol)

    x = 0.1 * torch.randn((1_000_000, 1, 1), dtype=torch.float32, device=device)
    _, cdf = hist.cdf(x, bins=bin_edges)
    e = efi.efi_gaussian(cdf, bin_edges, clim_mean, clim_std)
    _, pdf = hist.histogram(x, bins=bin_edges)
    e1 = efi.efi(bin_edges, pdf, clim_cdf)
    assert torch.allclose(e, 0.0 * one, rtol=rtol, atol=atol)
    assert torch.allclose(e1, 0.0 * one, rtol=rtol, atol=atol)

    ne = efi.normalized_entropy(test_pdf, bin_edges, clim_pdf)
    assert torch.allclose(ne, 0.0 * one, rtol=rtol, atol=atol)
