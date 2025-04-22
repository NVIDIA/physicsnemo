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

import os

import numpy as np
import pytest
import torch

import physicsnemo.metrics.general.calibration as cal
import physicsnemo.metrics.general.crps as crps
import physicsnemo.metrics.general.ensemble_metrics as em
import physicsnemo.metrics.general.entropy as ent
import physicsnemo.metrics.general.histogram as hist
import physicsnemo.metrics.general.power_spectrum as ps
import physicsnemo.metrics.general.wasserstein as w
from physicsnemo.distributed.manager import DistributedManager

Tensor = torch.Tensor


def get_disagreements(inputs, bins, counts, test):
    """
    Utility for testing disagreements in the bin counts.
    """
    sum_counts = torch.sum(counts, dim=0)
    disagreements = torch.nonzero(sum_counts != test, as_tuple=True)
    print("Disagreements: ", str(disagreements))

    number_of_disagree = len(disagreements[0])
    for i in range(number_of_disagree):
        ind = [disagreements[0][i], disagreements[1][i], disagreements[2][i]]
        print("Ind", ind)
        print(
            "Input ",
            inputs[:, disagreements[0][i], disagreements[1][i], disagreements[2][i]],
        )
        print(
            "Bins ",
            bins[:, disagreements[0][i], disagreements[1][i], disagreements[2][i]],
        )
        print(
            "Counts",
            counts[:, disagreements[0][i], disagreements[1][i], disagreements[2][i]],
        )

        trueh = torch.histogram(
            inputs[:, disagreements[0][i], disagreements[1][i], disagreements[2][i]],
            bins[:, disagreements[0][i], disagreements[1][i], disagreements[2][i]],
        )
        print("True counts", trueh)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("input_shape", [(1, 72, 144)])
def test_histogram(device, input_shape, rtol: float = 1e-3, atol: float = 1e-3):
    DistributedManager._shared_state = {}
    if (device == "cuda:0") and (not DistributedManager.is_initialized()):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12345"
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        DistributedManager.initialize()
    x = torch.randn([10, *input_shape], device=device)
    y = torch.randn([5, *input_shape], device=device)

    # Test linspace
    start = torch.zeros(input_shape, device=device)
    end = torch.ones(input_shape, device=device)
    lin = hist.linspace(start, end, 10)
    assert lin.shape[0] == 11
    l_np = np.linspace(start.cpu(), end.cpu(), 11)
    assert torch.allclose(
        lin,
        torch.from_numpy(l_np).to(device),
        rtol=rtol,
        atol=atol,
    )

    # Test histogram correctness
    xx = x[:, 0, 0, 0]
    xx_np = xx.cpu().numpy()
    bins, counts = hist.histogram(xx, bins=10)
    counts_np, bins_np = np.histogram(xx_np, bins=10)
    assert torch.allclose(
        bins,
        torch.from_numpy(bins_np).to(device),
        rtol=rtol,
        atol=atol,
    )
    assert torch.allclose(
        counts,
        torch.from_numpy(counts_np).to(device),
        rtol=rtol,
        atol=atol,
    )

    # Test low and high memory bin counts
    bins = lin
    counts = torch.zeros([10, *input_shape], device=device)
    counts_low_counts = hist._low_memory_bin_reduction_counts(x, bins, counts, 10)
    counts_high_counts = hist._high_memory_bin_reduction_counts(x, bins, counts, 10)
    counts_low_cdf = hist._low_memory_bin_reduction_cdf(x, bins, counts, 10)
    counts_high_cdf = hist._high_memory_bin_reduction_cdf(x, bins, counts, 10)
    assert torch.allclose(
        counts_low_counts,
        counts_high_counts,
        rtol=rtol,
        atol=atol,
    )
    assert torch.allclose(
        counts_low_cdf,
        counts_high_cdf,
        rtol=rtol,
        atol=atol,
    )

    # Test Raises Assertion
    with pytest.raises(ValueError):
        hist._count_bins(
            torch.zeros((1, 2), device=device),
            bins,
            counts,
        )
    # Test Raises Assertion
    with pytest.raises(ValueError):
        hist._count_bins(x, bins, torch.zeros((1,), device=device))

    with pytest.raises(ValueError):
        hist._get_mins_maxs()

    with pytest.raises(ValueError):
        hist._get_mins_maxs(
            torch.randn((10, 3), device=device), torch.randn((10, 5), device=device)
        )

    binsx, countsx = hist.histogram(x, bins=10, verbose=True)
    assert torch.allclose(
        torch.sum(countsx, dim=0),
        10 * torch.ones([1], dtype=torch.int64, device=device),
        rtol=rtol,
        atol=atol,
    ), get_disagreements(
        x, binsx, countsx, 10 * torch.ones([1], dtype=torch.int64, device=device)
    )

    binsxy, countsxy = hist.histogram(x, y, bins=5)
    assert torch.allclose(
        torch.sum(countsxy, dim=0),
        15 * torch.ones([1], dtype=torch.int64, device=device),
        rtol=rtol,
        atol=atol,
    ), get_disagreements(
        y,
        binsxy,
        countsxy - countsx,
        5 * torch.ones([1], dtype=torch.int64, device=device),
    )

    binsxy, countsxy = hist.histogram(x, y, bins=binsx)
    assert torch.allclose(
        torch.sum(countsxy, dim=0),
        15 * torch.ones([1], dtype=torch.int64, device=device),
        rtol=rtol,
        atol=atol,
    ), get_disagreements(
        y, binsxy, countsxy, 15 * torch.ones([1], dtype=torch.int64, device=device)
    )

    H = hist.Histogram(input_shape, bins=10, device=device)
    binsx, countsx = H(x)
    assert torch.allclose(
        torch.sum(countsx, dim=0),
        10 * torch.ones([1], dtype=torch.int64, device=device),
        rtol=rtol,
        atol=atol,
    ), get_disagreements(
        x, binsx, countsx, 10 * torch.ones([1], dtype=torch.int64, device=device)
    )

    binsxy, countsxy = H.update(y)
    if binsxy.shape[0] != binsx.shape[0]:
        dbins = binsx[1, 0, 0, 0] - binsx[0, 0, 0, 0]
        ind = torch.isclose(
            binsxy[:, 0, 0, 0], binsx[0, 0, 0, 0], rtol=0.1 * dbins, atol=1e-3
        ).nonzero(as_tuple=True)[0]
        new_counts = countsxy[ind : ind + 10] - countsx
    else:
        new_counts = countsxy - countsx
    assert torch.allclose(
        torch.sum(countsxy, dim=0),
        15 * torch.ones([1], dtype=torch.int64, device=device),
        rtol=rtol,
        atol=atol,
    ), get_disagreements(
        y, binsxy, new_counts, 5 * torch.ones([1], dtype=torch.int64, device=device)
    )

    _, pdf = H.finalize()
    _, cdf = H.finalize(cdf=True)
    assert torch.allclose(
        cdf[-1],
        torch.ones([1], dtype=torch.float32, device=device),
        rtol=rtol,
        atol=atol,
    )
    assert torch.allclose(
        torch.sum(pdf, dim=0),
        torch.ones([1], dtype=torch.float32, device=device),
        rtol=rtol,
        atol=atol,
    )
    if device == "cuda:0":
        DistributedManager.cleanup()
        del os.environ["RANK"]
        del os.environ["WORLD_SIZE"]
        del os.environ["MASTER_ADDR"]
        del os.environ["MASTER_PORT"]


def fair_crps(pred, obs, dim=-1):
    return crps.kcrps(pred, obs, dim=dim, biased=False)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_fair_crps_greater_than_zero(device):
    pred = torch.randn(5, 10, device=device)
    obs = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0], device=device)
    assert torch.all(fair_crps(pred, obs, dim=-1) > 0)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_fair_crps_is_fair(device):
    # fair means that a random prediction should outperform a non-random one on average
    # This is not always true of ``crps``...try replacing fair_crps function
    # below with ``crps``
    n = 256
    random_pred = torch.randn(n, 2, device=device)
    cheating_pred = torch.zeros((n, 2), device=device)
    obs = torch.randn(n, device=device)

    score_of_random = fair_crps(random_pred, obs, dim=-1).mean()
    score_of_cheating = fair_crps(cheating_pred, obs, dim=-1).mean()
    assert score_of_random.item() < score_of_cheating.item()


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_fair_crps_converges_to_crps(device):
    # for large ensemble fair cprs should be close to crps

    # generate deterministic sample with normal pdf
    # deterministic to ensure reproducibility for this test which
    # has a specific tolerance
    norm = torch.distributions.Normal(0, 1)
    u = torch.linspace(0, 1, 10_000)[1:-1].to(device)
    pred = norm.icdf(u)

    # expected value using exact gaussian crps formula
    zero = torch.tensor(0, device=device).float()
    one = torch.tensor(1, device=device).float()
    expected = crps._crps_gaussian(mean=zero, std=one, obs=zero).item()

    fair_value = fair_crps(pred, zero).item()

    assert pytest.approx(fair_value, rel=1e-3) == expected


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_fair_crps_dim_arg_works(device):
    pred = torch.randn((5, 10, 100), device=device)

    a, b, c = pred.shape

    value = fair_crps(pred, torch.zeros([a, c], device=device), dim=1)
    assert value.shape == (a, c)

    value = fair_crps(pred, torch.zeros([b, c], device=device), dim=0)
    assert value.shape == (b, c)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("num", [10, 23, 59])
@pytest.mark.parametrize("biased", [True, False])
def test_crps_finite(device, num, biased):

    # Test biased and unbiased on uniform data with analytic result
    pred = torch.linspace(0, 1, num + 1).unsqueeze(-1).to(device)
    obs = torch.zeros([1], device=device)

    analytic = (2 * num + 1) / (6 * num + 6) if biased else (num - 1) / (3 * num)
    analytic = torch.as_tensor([analytic], device=device)

    assert torch.all(torch.isclose(analytic, crps.kcrps(pred, obs, biased=biased)))

    # Test biased on random data
    pred = torch.as_tensor(
        [
            -1.9293,
            0.6454,
            -0.3902,
            -1.0438,
            -1.3573,
            -0.2942,
            2.6269,
            1.0405,
            -0.5659,
            -0.1438,
            -0.3993,
            0.7306,
            -0.8229,
            -0.8500,
            -0.5732,
            0.6746,
            0.7208,
            0.6172,
            -1.6648,
            0.5183,
            -0.4850,
            0.3033,
            2.4232,
            0.4714,
            -0.6040,
            -0.4617,
            1.3324,
            -0.7937,
            0.8862,
            -1.2291,
            2.7559,
            -1.0750,
            -0.3882,
            0.2331,
            0.4886,
            -0.3715,
            -0.3438,
            -0.4229,
            0.7913,
            1.0469,
        ],
        device=device,
    ).reshape([-1, 1])
    obs = torch.as_tensor([-1.1569], device=device)
    analytic = torch.as_tensor([0.7027], device=device)
    assert torch.all(torch.isclose(analytic, crps.kcrps(pred, obs, biased=True)))


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_crps(device, rtol: float = 1e-3, atol: float = 1e-3):
    # Uses eq (5) from Gneiting et al. https://doi.org/10.1175/MWR2904.1
    # crps(N(0, 1), 0.0) = 2 / sqrt(2*pi) - 1/sqrt(pi) ~= 0.23...
    x = torch.randn((1_000_000, 1), device=device, dtype=torch.float32)
    y = torch.zeros((1,), device=device, dtype=torch.float32)

    # Test pure crps
    c = crps.crps(x, y, method="histogram")
    true_crps = (np.sqrt(2) - 1.0) / np.sqrt(np.pi)
    assert torch.allclose(
        c,
        true_crps * torch.ones([1], dtype=torch.float32, device=device),
        rtol=rtol,
        atol=atol,
    )

    # Test when input is numpy array
    c = crps.crps(x, y.cpu().numpy(), method="histogram")
    assert torch.allclose(
        c,
        true_crps * torch.ones([1], dtype=torch.float32, device=device),
        rtol=rtol,
        atol=atol,
    )

    # Test pure crps
    c = crps.crps(x[:100], y, method="kernel")
    true_crps = (np.sqrt(2) - 1.0) / np.sqrt(np.pi)
    assert torch.allclose(
        c,
        true_crps * torch.ones([1], dtype=torch.float32, device=device),
        rtol=100 * rtol,
        atol=100 * atol,
    )

    # Test when input is numpy array
    c = crps.crps(x[:100], y.cpu().numpy(), method="kernel")
    assert torch.allclose(
        c,
        true_crps * torch.ones([1], dtype=torch.float32, device=device),
        rtol=100 * rtol,
        atol=100 * atol,
    )

    # Test kernel method, use fewer ensemble members
    c = crps.kcrps(x[:100], y)
    true_crps = (np.sqrt(2) - 1.0) / np.sqrt(np.pi)
    assert torch.allclose(
        c,
        true_crps * torch.ones([1], dtype=torch.float32, device=device),
        rtol=100 * rtol,
        atol=100 * atol,
    )

    # Test sorted crps
    c = crps.crps(x[:10_000], y, method="sort")
    true_crps = (np.sqrt(2) - 1.0) / np.sqrt(np.pi)
    assert torch.allclose(
        c,
        true_crps * torch.ones([1], dtype=torch.float32, device=device),
        rtol=10 * rtol,
        atol=10 * atol,
    )

    # Test when input is numpy array
    c = crps.crps(x[:10_000], y.cpu().numpy(), method="sort")
    assert torch.allclose(
        c,
        true_crps * torch.ones([1], dtype=torch.float32, device=device),
        rtol=10 * rtol,
        atol=10 * atol,
    )

    # Test Gaussian CRPS
    mm = torch.zeros([1], dtype=torch.float32, device=device)
    vv = torch.ones([1], dtype=torch.float32, device=device)
    gaussian_crps = crps._crps_gaussian(mm, vv, y)
    assert torch.allclose(
        gaussian_crps,
        true_crps * torch.ones([1], dtype=torch.float32, device=device),
        rtol=rtol,
        atol=atol,
    )
    gaussian_crps = crps._crps_gaussian(mm, vv, y.cpu().numpy())
    assert torch.allclose(
        gaussian_crps,
        true_crps * torch.ones([1], dtype=torch.float32, device=device),
        rtol=rtol,
        atol=atol,
    )

    # Test Assertions
    with pytest.raises(ValueError):
        crps._crps_gaussian(torch.tensor((10, 2), device=device), vv, y)

    with pytest.raises(ValueError):
        crps._crps_gaussian(
            mm,
            vv,
            torch.tensor((10, 2), device=device),
        )

    # Test from counts
    binsx, countsx = hist.histogram(x, bins=1_000)
    assert torch.allclose(
        torch.sum(countsx, dim=0),
        1_000_000 * torch.ones([1], dtype=torch.int64, device=device),
        rtol=rtol,
        atol=atol,
    )
    c = crps._crps_from_counts(binsx, countsx, y)
    assert torch.allclose(
        c,
        true_crps * torch.ones([1], dtype=torch.float32, device=device),
        rtol=rtol,
        atol=atol,
    )
    # Counts, numpy
    c = crps._crps_from_counts(binsx, countsx, y.cpu().numpy())
    assert torch.allclose(
        c,
        true_crps * torch.ones([1], dtype=torch.float32, device=device),
        rtol=rtol,
        atol=atol,
    )

    # Test raises Assertion
    with pytest.raises(ValueError):
        crps._crps_from_counts(torch.zeros((1, 2), device=device), countsx, y)
    with pytest.raises(ValueError):
        crps._crps_from_counts(binsx, torch.zeros((1, 2), device=device), y)
    with pytest.raises(ValueError):
        crps._crps_from_counts(binsx, countsx, torch.zeros((1, 2), device=device))

    # Test from cdf
    binsx, cdfx = hist.cdf(x, bins=1_000)
    assert torch.allclose(
        cdfx[-1],
        torch.ones([1], dtype=torch.float32, device=device),
        rtol=rtol,
        atol=atol,
    )
    c = crps._crps_from_cdf(binsx, cdfx, y)
    assert torch.allclose(
        c,
        true_crps * torch.ones([1], dtype=torch.float32, device=device),
        rtol=rtol,
        atol=atol,
    )

    # Test Raises Assertion
    with pytest.raises(ValueError):
        crps._crps_from_cdf(torch.zeros((1, 2), device=device), cdfx, y)
    with pytest.raises(ValueError):
        crps._crps_from_cdf(binsx, torch.zeros((1, 2), device=device), y)
    with pytest.raises(ValueError):
        crps._crps_from_cdf(binsx, cdfx, torch.zeros((1, 2), device=device))

    # Test different shape
    x = torch.randn((2, 3, 50, 100), device=device, dtype=torch.float32)
    y = torch.zeros((2, 3, 100), device=device, dtype=torch.float32)
    z = torch.zeros((2, 3, 50), device=device, dtype=torch.float32)

    # Test dim
    c = crps.crps(x, y, dim=2)
    assert c.shape == y.shape

    # Test when input is numpy array
    c = crps.crps(x, y.cpu().numpy(), dim=2)
    assert c.shape == y.shape

    # Test different dim
    c = crps.crps(x, z, dim=3)
    assert c.shape == z.shape

    # Test when input is numpy array
    c = crps.crps(x, z.cpu().numpy(), dim=3)
    assert c.shape == z.shape

    # Test kernel method
    c = crps.kcrps(x, z, dim=3)
    true_crps = (np.sqrt(2) - 1.0) / np.sqrt(np.pi)
    assert c.shape == z.shape


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("mean", [3.0])
@pytest.mark.parametrize("variance", [0.1])
def test_wasserstein(device, mean, variance, rtol: float = 1e-3, atol: float = 1e-3):
    mean = torch.as_tensor([mean], device=device, dtype=torch.float32)
    variance = torch.as_tensor([variance], device=device, dtype=torch.float32)

    x = mean + torch.sqrt(variance) * torch.randn(
        (10_000, 1), device=device, dtype=torch.float32
    )
    y = mean + torch.sqrt(variance) * torch.randn(
        (10_000, 1), device=device, dtype=torch.float32
    )

    binsx, cdfx = hist.cdf(x, bins=10)
    _, cdfy = hist.cdf(y, bins=binsx)
    w_cdf = w.wasserstein_from_cdf(binsx, cdfx, cdfy)
    w_samples = w.wasserstein_from_samples(x, y)

    assert torch.allclose(
        w.wasserstein_from_cdf(binsx, cdfx, cdfx),
        torch.zeros([1], dtype=torch.float32, device=device),
        rtol=rtol,
        atol=atol,
    )
    assert torch.allclose(w_cdf, w_samples, rtol=rtol, atol=atol)

    mu_x = x.mean(dim=0)
    sig_x = x.var(dim=0)
    mu_y = y.mean(dim=0)
    sig_y = y.var(dim=0)
    w_norm = w.wasserstein_from_normal(mu_x, sig_x, mu_y, sig_y)
    assert torch.allclose(
        w_norm,
        torch.zeros([1], dtype=torch.float32, device=device),
        rtol=rtol,
        atol=100 * atol,
    )

    x = mean + torch.sqrt(variance) * torch.randn(
        (100_000, 3), device=device, dtype=torch.float32
    )
    y = mean + torch.sqrt(variance) * torch.randn(
        (100_000, 3), device=device, dtype=torch.float32
    )
    mu_x = x.mean(dim=0)
    sig_x = torch.cov(x.T)
    mu_y = y.mean(dim=0)
    sig_y = torch.cov(y.T)
    w_norm = w.wasserstein_from_normal(mu_x, sig_x, mu_y, sig_y)
    assert not torch.any(torch.isnan(w_norm))

    x = mean + torch.sqrt(variance) * torch.randn(
        (1_000, 100_000, 3), device=device, dtype=torch.float32
    )
    y = mean + torch.sqrt(variance) * torch.randn(
        (1_000, 100_000, 3), device=device, dtype=torch.float32
    )
    mu_x = x.mean(dim=1)
    sig_x = torch.matmul((x - mu_x[:, None]).transpose(1, 2), (x - mu_x[:, None])) / (
        1_000 - 1
    )
    mu_y = y.mean(dim=1)
    sig_y = torch.matmul((y - mu_y[:, None]).transpose(1, 2), (y - mu_y[:, None])) / (
        1_000 - 1
    )
    w_mnorm = w.wasserstein_from_normal(mu_x, sig_x, mu_y, sig_y)
    assert not torch.any(torch.isnan(w_mnorm))


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_means_var(device, rtol: float = 1e-3, atol: float = 1e-3):
    DistributedManager._shared_state = {}
    if (device == "cuda:0") and (not DistributedManager.is_initialized()):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12345"
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        DistributedManager.initialize()

    ens_metric = em.EnsembleMetrics((1, 72, 144), device=device)
    with pytest.raises(NotImplementedError) as e_info:
        print(e_info)
        ens_metric.__call__()
    with pytest.raises(NotImplementedError) as e_info:
        print(e_info)
        ens_metric.update()
    with pytest.raises(NotImplementedError) as e_info:
        print(e_info)
        ens_metric.finalize()
    with pytest.raises(ValueError):
        ens_metric._check_shape(torch.zeros((1, 7, 14), device=device))

    x = torch.randn((10, 1, 72, 144), device=device)
    y = torch.randn((5, 1, 72, 144), device=device)

    M = em.Mean((1, 72, 144), device=device)
    meanx = M(x)
    assert torch.allclose(meanx, torch.mean(x, dim=0))
    meanxy = M.update(y)
    assert torch.allclose(
        meanxy, torch.mean(torch.cat((x, y), dim=0), dim=0), rtol=rtol, atol=atol
    )
    assert torch.allclose(meanxy, M.finalize(), rtol=rtol, atol=atol)

    # Test raises Assertion
    with pytest.raises(AssertionError):
        M(x.to("cuda:0" if device == "cpu" else "cpu"))
    with pytest.raises(AssertionError):
        M.update(y.to("cuda:0" if device == "cpu" else "cpu"))

    # Test _update_mean utility
    _sumxy, _n = em._update_mean(meanx * 10, 10, y, batch_dim=0)
    assert torch.allclose(meanxy, _sumxy / _n, rtol=rtol, atol=atol)
    # Test with flattened y
    _sumxy, _n = em._update_mean(meanx * 10, 10, y[0], batch_dim=None)
    _sumxy, _n = em._update_mean(_sumxy, _n, y[1:], batch_dim=0)
    assert torch.allclose(meanxy, _sumxy / _n, rtol=rtol, atol=atol)

    V = em.Variance((1, 72, 144), device=device)
    varx = V(x)
    assert torch.allclose(varx, torch.var(x, dim=0))
    varxy = V.update(y)
    assert torch.allclose(
        varxy, torch.var(torch.cat((x, y), dim=0), dim=0), rtol=rtol, atol=atol
    )
    varxy = V.finalize()
    assert torch.allclose(
        varxy, torch.var(torch.cat((x, y), dim=0), dim=0), rtol=rtol, atol=atol
    )
    stdxy = V.finalize(std=True)
    assert torch.allclose(
        stdxy, torch.std(torch.cat((x, y), dim=0), dim=0), rtol=rtol, atol=atol
    )
    # Test raises Assertion
    with pytest.raises(AssertionError):
        V(x.to("cuda:0" if device == "cpu" else "cpu"))
    with pytest.raises(AssertionError):
        V.update(y.to("cuda:0" if device == "cpu" else "cpu"))

    # Test _update_var utility function
    _sumxy, _sum2xy, _n = em._update_var(10 * meanx, 9 * varx, 10, y, batch_dim=0)
    assert _n == 15
    assert torch.allclose(varxy, _sum2xy / (_n - 1.0), rtol=rtol, atol=atol)

    # Test with flattened array
    # Test with flattened y
    _sumxy, _sum2xy, _n = em._update_var(10 * meanx, 9 * varx, 10, y[0], batch_dim=None)
    assert _n == 11
    assert torch.allclose(
        _sumxy / _n,
        torch.mean(torch.cat((x, y[0][None, ...]), dim=0), dim=0),
        rtol=rtol,
        atol=atol,
    )
    assert torch.allclose(
        _sum2xy / (_n - 1.0),
        torch.var(torch.cat((x, y[0][None, ...]), dim=0), dim=0),
        rtol=rtol,
        atol=atol,
    )
    _sumxy, _sum2xy, _n = em._update_var(_sumxy, _sum2xy, _n, y[1:], batch_dim=0)
    assert torch.allclose(varxy, _sum2xy / (_n - 1.0), rtol=rtol, atol=atol)

    if device == "cuda:0":
        DistributedManager.cleanup()
        del os.environ["RANK"]
        del os.environ["WORLD_SIZE"]
        del os.environ["MASTER_ADDR"]
        del os.environ["MASTER_PORT"]


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_calibration(device, rtol: float = 1e-2, atol: float = 1e-2):

    x = torch.randn((10_000, 30, 30), device=device, dtype=torch.float32)
    y = torch.randn((30, 30), device=device, dtype=torch.float32)

    bin_edges, bin_counts = hist.histogram(x, bins=30)

    # Test getting rank from histogram
    ranks = cal.find_rank(bin_edges, bin_counts, y)

    assert ranks.shape == y.shape
    assert torch.all(torch.le(ranks, 1.0))
    assert torch.all(torch.ge(ranks, 0.0))

    # Test getting rank from histogram (numpy)
    y = np.random.randn(30, 30)
    ranks_np = cal.find_rank(bin_edges, bin_counts, y)

    assert ranks_np.shape == y.shape
    assert torch.all(torch.le(ranks_np, 1.0))
    assert torch.all(torch.ge(ranks_np, 0.0))

    # Test Raises Assertions
    with pytest.raises(ValueError):
        cal.find_rank(torch.zeros((10,), device=device), bin_counts, y)

    with pytest.raises(ValueError):
        cal.find_rank(bin_edges, torch.zeros((10,), device=device), y)

    with pytest.raises(ValueError):
        cal.find_rank(
            bin_edges,
            bin_counts,
            torch.zeros((10,), device=device),
        )

    ranks = ranks.flatten()
    rank_bin_edges = torch.linspace(0, 1, 11).to(device)
    rank_bin_edges, rank_counts = hist.histogram(ranks, bins=rank_bin_edges)
    rps = cal._rank_probability_score_from_counts(rank_bin_edges, rank_counts)

    assert rps > 0.0
    assert rps < 1.0
    assert torch.allclose(
        rps, torch.zeros([1], device=device, dtype=torch.float32), rtol=rtol, atol=atol
    )

    rps = cal.rank_probability_score(ranks)
    assert rps > 0.0
    assert rps < 1.0
    assert torch.allclose(
        rps, torch.zeros([1], device=device, dtype=torch.float32), rtol=rtol, atol=atol
    )

    num_obs = 1000

    x = torch.randn((1_000, num_obs, 10, 10), device=device, dtype=torch.float32)
    bin_edges, bin_counts = hist.histogram(x, bins=20)

    obs = torch.randn((num_obs, 10, 10), device=device, dtype=torch.float32)
    ranks = cal.find_rank(bin_edges, bin_counts, obs)
    assert ranks.shape == (num_obs, 10, 10)

    rps = cal.rank_probability_score(ranks)
    assert rps.shape == (10, 10)
    assert torch.allclose(
        rps, torch.zeros([1], device=device, dtype=torch.float32), rtol=rtol, atol=atol
    )


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_entropy(device, rtol: float = 1e-2, atol: float = 1e-2):
    one = torch.ones([1], device=device, dtype=torch.float32)

    x = torch.randn((50_000, 10, 10), device=device, dtype=torch.float32)
    bin_edges, bin_counts = hist.histogram(x, bins=30)
    entropy = ent.entropy_from_counts(bin_counts, bin_edges, normalized=False)
    assert entropy.shape == (10, 10)
    assert torch.allclose(
        entropy, (0.5 + 0.5 * np.log(2 * np.pi)) * one, atol=atol, rtol=rtol
    )
    entropy = ent.entropy_from_counts(bin_counts, bin_edges, normalized=True)
    assert torch.all(torch.le(entropy, one))
    assert torch.all(torch.ge(entropy, 0.0 * one))

    # Test raises Assertion
    with pytest.raises(ValueError):
        ent.entropy_from_counts(
            torch.zeros((bin_counts.shape[0], 1, 1), device=device), bin_edges
        )
    with pytest.raises(ValueError):
        ent.entropy_from_counts(
            torch.zeros((1,) + bin_counts.shape[1:], device=device), bin_edges
        )

    # Test Maximum Entropy
    x = torch.rand((100_000, 10, 10), device=device, dtype=torch.float32)
    bin_edges, bin_counts = hist.histogram(x, bins=30)
    entropy = ent.entropy_from_counts(bin_counts, bin_edges, normalized=True)
    assert entropy.shape == (10, 10)
    assert torch.allclose(entropy, one, rtol=rtol, atol=atol)

    # Test Relative Entropy
    x = torch.randn((100_000, 10, 10), device=device, dtype=torch.float32)
    bin_edges, x_bin_counts = hist.histogram(x, bins=30)
    x1 = torch.randn((100_000, 10, 10), device=device, dtype=torch.float32)
    _, x1_bin_counts = hist.histogram(x1, bins=bin_edges)
    x2 = 0.1 * torch.randn((50_000, 10, 10), device=device, dtype=torch.float32)
    _, x2_bin_counts = hist.histogram(x2, bins=bin_edges)

    rel_ent_1 = ent.relative_entropy_from_counts(x_bin_counts, x1_bin_counts, bin_edges)
    rel_ent_2 = ent.relative_entropy_from_counts(x_bin_counts, x2_bin_counts, bin_edges)

    assert torch.all(torch.le(rel_ent_1, rel_ent_2))
    # assert torch.allclose(rel_ent_1, 0.0 * one, rtol=10.*rtol, atol = 10.*atol) # TODO
    assert torch.all(torch.ge(rel_ent_2, 0.0 * one))

    # Test raises Assertion
    with pytest.raises(ValueError):
        ent.relative_entropy_from_counts(
            torch.zeros((x_bin_counts.shape[0], 1, 1), device=device),
            x1_bin_counts,
            bin_edges,
        )
    with pytest.raises(ValueError):
        ent.relative_entropy_from_counts(
            torch.zeros((1,) + x_bin_counts.shape[1:], device=device),
            x1_bin_counts,
            bin_edges,
        )
    with pytest.raises(ValueError):
        ent.relative_entropy_from_counts(
            x_bin_counts,
            torch.zeros((1,) + x_bin_counts.shape[1:], device=device),
            bin_edges,
        )


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_power_spectrum(device):
    # Test the 2D power spectrum routine for correctness using a sine wave
    h, w = 32, 32
    kx, ky = 4, 4
    amplitude = 1.0

    # Create input sine wave
    x = torch.arange(w).view(1, -1).repeat(h, 1).float()
    y = torch.arange(h).view(-1, 1).repeat(1, w).float()
    signal = amplitude * torch.sin(2 * np.pi * kx * x / w + 2 * np.pi * ky * y / h)

    # Compute the power spectrum (added batch/channel dims)
    k, power = ps.power_spectrum(signal.unsqueeze(0).unsqueeze(0))

    # Assert that the power at expected wavenumber is dominant
    k_total = np.sqrt(kx**2 + ky**2)
    k_index = (torch.abs(k - k_total)).argmin()
    assert power[0, 0, k_index] > 0.9 * power[0, 0].max()  # Dominant peak
    assert (power[0, 0] < 1e-6).sum() > (
        power[0, 0].numel() * 0.9
    )  # Most bins are zero
