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

from typing import Union

import numpy as np
import torch

from .histogram import cdf as cdf_function

Tensor = torch.Tensor


@torch.jit.script
def _kernel_crps_implementation(pred: Tensor, obs: Tensor, biased: bool) -> Tensor:
    """An O(m log m) implementation of the kernel CRPS formulas"""
    skill = torch.abs(pred - obs[..., None]).mean(-1)
    pred, _ = torch.sort(pred)

    # derivation of fast implementation of spread-portion of CRPS formula when x is sorted
    # sum_(i,j=1)^m |x_i - x_j| = sum_(i<j) |x_i -x_j| + sum_(i > j) |x_i - x_j|
    #                           = 2 sum_(i <= j) |x_i -x_j|
    #                           = 2 sum_(i <= j) (x_j - x_i)
    #                           = 2 sum_(i <= j) x_j - 2 sum_(i <= j) x_i
    #                           = 2 sum_(j=1)^m j x_j - 2 sum (m - i + 1) x_i
    #                           = 2 sum_(i=1)^m (2i - m - 1) x_i
    m = pred.size(-1)
    i = torch.arange(1, m + 1, device=pred.device, dtype=pred.dtype)
    denom = m * m if biased else m * (m - 1)
    factor = (2 * i - m - 1) / denom
    spread = torch.sum(factor * pred, dim=-1)
    return skill - spread


def kcrps(pred: Tensor, obs: Tensor, dim: int = 0, biased: bool = True):
    """Estimate the CRPS from a finite ensemble

    Computes the local Continuous Ranked Probability Score (CRPS) by using
    the kernel version of CRPS. The cost is O(m log m).

    Creates a map of CRPS and does not accumulate over lat/lon regions.
    Approximates:
    .. math::
        CRPS(X, y) = E[X - y] - 0.5 E[X-X']

    with
    .. math::
        sum_i=1^m |X_i - y| / m - 1/(2m^2) sum_i,j=1^m |x_i - x_j|

    Parameters
    ----------
    pred : Tensor
        Tensor containing the ensemble predictions. The ensemble dimension
        is assumed to be the leading dimension unless 'dim' is specified.
    obs : Union[Tensor, np.ndarray]
        Tensor or array containing an observation over which the CRPS is computed
        with respect to.
    dim : int, optional
        The dimension over which to compute the CRPS, assumed to be 0.
    biased :
        When False, uses the unbiased estimators described in (Zamo and Naveau, 2018)::

            E|X-y|/m - 1/(2m(m-1)) sum_(i,j=1)|x_i - x_j|

        Unlike ``crps`` this is fair for finite ensembles. Non-fair ``crps`` favors less
        dispersive ensembles since it is biased high by E|X- X'|/ m where m is the
        ensemble size.

    Returns
    -------
    Tensor
        Map of CRPS
    """
    pred = torch.movedim(pred, dim, -1)
    return _kernel_crps_implementation(pred, obs, biased=biased)


def _crps_gaussian(mean: Tensor, std: Tensor, obs: Union[Tensor, np.ndarray]) -> Tensor:
    """
    Computes the local Continuous Ranked Probability Score (CRPS)
    using assuming that the forecast distribution is normal.

    Creates a map of CRPS and does not accumulate over lat/lon regions.

    Computes:

    .. math::

        CRPS(mean, std, y) = std * [ \\frac{1}{\\sqrt{\\pi}}} - 2 \\phi ( \\frac{x-mean}{std} ) -
                ( \\frac{x-mean}{std} ) * (2 \\Phi(\\frac{x-mean}{std}) - 1) ]

    where \\phi and \\Phi are the normal gaussian pdf/cdf respectively.

    Parameters
    ----------
    mean : Tensor
        Tensor of mean of forecast distribution.
    std : Tensor
        Tensor of standard deviation of forecast distribution.
    obs : Union[Tensor, np.ndarray]
        Tensor or array containing an observation over which the CRPS is computed
        with respect to. Broadcasting dimensions must be compatible with the non-zeroth
        dimensions of bins and cdf.

    Returns
    -------
    Tensor
        Map of CRPS
    """
    if isinstance(obs, np.ndarray):
        obs = torch.from_numpy(obs).to(mean.device)
    # Check shape compatibility
    if mean.shape != std.shape:
        raise ValueError(
            "Mean and standard deviation must have"
            + "compatible shapes but found"
            + str(mean.shape)
            + " and "
            + str(std.shape)
            + "."
        )
    if mean.shape != obs.shape:
        raise ValueError(
            "Mean and obs must have"
            + "compatible shapes but found"
            + str(mean.shape)
            + " and "
            + str(obs.shape)
            + "."
        )

    d = (obs - mean) / std
    phi = torch.exp(-0.5 * d**2) / torch.sqrt(torch.as_tensor(2 * torch.pi))

    # Note, simplified expression below is not exactly Gaussian CDF
    Phi = torch.erf(d / torch.sqrt(torch.as_tensor(2.0)))

    return std * (2 * phi + d * Phi - 1.0 / torch.sqrt(torch.as_tensor(torch.pi)))


def _crps_from_cdf(
    bin_edges: Tensor, cdf: Tensor, obs: Union[Tensor, np.ndarray]
) -> Tensor:
    """Computes the local Continuous Ranked Probability Score (CRPS)
    using a cumulative distribution function.

    Creates a map of CRPS and does not accumulate over lat/lon regions.

    Computes:

    .. math::

        CRPS(X, y) = \\int[ (F(x) - 1[x - y])^2 ] dx

    where F is the empirical cdf of X.

    Parameters
    ----------
    bins_edges : Tensor
        Tensor [N+1, ...] containing bin edges. The leading dimension must represent the
        N+1 bin edges.
    cdf : Tensor
        Tensor [N, ...] containing a cdf, defined over bins. The non-zeroth dimensions
        of bins and cdf must be compatible.
    obs : Union[Tensor, np.ndarray]
        Tensor or array containing an observation over which the CRPS is computed
        with respect to. Broadcasting dimensions must be compatible with the non-zeroth
        dimensions of bins and cdf.

    Returns
    -------
    Tensor
        Map of CRPS
    """
    if isinstance(obs, np.ndarray):
        obs = torch.from_numpy(obs).to(cdf.device)
    if bin_edges.shape[1:] != cdf.shape[1:]:
        raise ValueError(
            "Expected bins and cdf to have compatible non-zeroth dimensions but have shapes"
            + str(bin_edges.shape[1:])
            + " and "
            + str(cdf.shape[1:])
            + "."
        )
    if bin_edges.shape[1:] != obs.shape:
        raise ValueError(
            "Expected bins and observations to have compatible broadcasting dimensions but have shapes"
            + str(bin_edges.shape[1:])
            + " and "
            + str(obs.shape)
            + "."
        )
    if bin_edges.shape[0] != cdf.shape[0] + 1:
        raise ValueError(
            "Expected zeroth dimension of cdf to be equal to the zeroth dimension of bins + 1 but have shapes"
            + str(bin_edges.shape[0])
            + " and "
            + str(cdf.shape[0])
            + "+1."
        )
    dbins = bin_edges[1, ...] - bin_edges[0, ...]
    bin_mids = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    obs = torch.ge(bin_mids, obs).int()
    return torch.sum(torch.abs(cdf - obs) ** 2 * dbins, dim=0)


def _crps_from_counts(
    bin_edges: Tensor, counts: Tensor, obs: Union[Tensor, np.ndarray]
) -> Tensor:
    """Computes the local Continuous Ranked Probability Score (CRPS)
    using a histogram of counts.

    Creates a map of CRPS and does not accumulate over lat/lon regions.

    Computes:

    .. math::

        CRPS(X, y) = int[ (F(x) - 1[x - y])^2 ] dx

    where F is the empirical cdf of X.

    Parameters
    ----------
    bins_edges : Tensor
        Tensor [N+1, ...] containing bin edges. The leading dimension must represent the
        N+1 bin edges.
    counts : Tensor
        Tensor [N, ...] containing counts, defined over bins. The non-zeroth dimensions
        of bins and counts must be compatible.
    obs : Union[Tensor, np.ndarray]
        Tensor or array containing an observation over which the CRPS is computed
        with respect to. Broadcasting dimensions must be compatible with the non-zeroth
        dimensions of bins and counts.

    Returns
    -------
    Tensor
        Map of CRPS
    """
    if isinstance(obs, np.ndarray):
        obs = torch.from_numpy(obs).to(counts.device)
    if bin_edges.shape[1:] != counts.shape[1:]:
        raise ValueError(
            "Expected bins and cdf to have compatible non-zeroth dimensions but have shapes"
            + str(bin_edges.shape[1:])
            + " and "
            + str(counts.shape[1:])
            + "."
        )
    if bin_edges.shape[1:] != obs.shape:
        raise ValueError(
            "Expected bins and observations to have compatible broadcasting dimensions but have shapes"
            + str(bin_edges.shape[1:])
            + " and "
            + str(obs.shape)
            + "."
        )
    if bin_edges.shape[0] != counts.shape[0] + 1:
        raise ValueError(
            "Expected zeroth dimension of cdf to be equal to the zeroth dimension of bins + 1 but have shapes"
            + str(bin_edges.shape[0])
            + " and "
            + str(counts.shape[0])
            + "+1."
        )
    cdf_hat = torch.cumsum(counts / torch.sum(counts, dim=0), dim=0)
    return _crps_from_cdf(bin_edges, cdf_hat, obs)


def crps(
    pred: Tensor, obs: Union[Tensor, np.ndarray], dim: int = 0, method: str = "kernel"
) -> Tensor:
    """
    Computes the local Continuous Ranked Probability Score (CRPS).

    Creates a map of CRPS and does not accumulate over any other dimensions (e.g., lat/lon regions).

    Parameters
    ----------
    pred : Tensor
        Tensor containing the ensemble predictions.
    obs : Union[Tensor, np.ndarray]
        Tensor or array containing an observation over which the CRPS is computed
        with respect to.
    dim : int, Optional
        Dimension with which to calculate the CRPS over, the ensemble dimension.
        Assumed to be zero.
    method: str, Optional
        The method to calculate the crps. Can either be "kernel", "sort" or "histogram".

        The "kernel" method implements
        .. math::
            CRPS(x, y) = E[X-y] - 0.5*E[X-X']

        This method scales as O(n^2) where n is the number of ensemble members and
        can potentially induce large memory consumption as the algorithm attempts
        to vectorize over this O(n^2) operation.

        The "sort" method compute the exact CRPS using the CDF method
        .. math::
            CRPS(x, y) = int [F(x) - 1(x-y)]^2 dx

        where F is the empirical CDF and 1(x-y) = 1 if x > y.

        This method is more memory efficient than the kernel method, and uses O(n
        log n) compute instead of O(n^2), where n is the number of ensemble members.

        The "histogram" method computes an approximate CRPS using the CDF method
        .. math::
            CRPS(x, y) = int [F(x) - 1(x-y)]^2 dx

        where F is the empirical CDF, estimated via a histogram of the samples. The
        number of bins used is the lesser of the square root of the number of samples
        and 100. For more control over the implementation of this method consider using
        `cdf_function` to construct a cdf and `_crps_from_cdf` to compute CRPS.

    Returns
    -------
    Tensor
        Map of CRPS
    """
    if method not in ["kernel", "sort", "histogram"]:
        raise ValueError("Method must either be 'kernel', 'sort' or 'histogram'.")

    n = pred.shape[dim]
    obs = torch.as_tensor(obs, device=pred.device, dtype=pred.dtype)
    if method in ["kernel", "sort"]:
        return kcrps(pred, obs, dim=dim)
    else:
        pred = pred.unsqueeze(0).transpose(0, dim + 1).squeeze(dim + 1)
        number_of_bins = max(int(np.sqrt(n)), 100)
        bin_edges, cdf = cdf_function(pred, bins=number_of_bins)
        _crps = _crps_from_cdf(bin_edges, cdf, obs)
        return _crps
