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

import torch
from torch import Tensor

from physicsnemo.metrics.general.reduction import WeightedMean, WeightedVariance


def _compute_lat_weights(lat: Tensor) -> Tensor:
    """Computes weighting for latitude reduction

    Parameters
    ----------
    lat : Tensor
        A one-dimension tensor [H] representing the latitudes at which the function will
        return weights for

    Returns
    -------
    Tensor
        Latitude weight tensor [H]
    """

    lat_weight = torch.abs(torch.cos(torch.pi * (lat / 180.0)))

    lat_weight = lat_weight / lat_weight.sum()
    return lat_weight


def zonal_mean(x: Tensor, lat: Tensor, dim: int = -2, keepdims: bool = False) -> Tensor:
    """Computes zonal mean, weighting over the latitude direction that is specified by dim

    Parameters
    ----------
    x : Tensor
        The tensor [..., H, W] over which the mean will be computed
    lat : Tensor
        A one-dimension tensor representing the latitudes at which the function will
        return weights for
    dim : int, optional
        The int specifying which dimension of x the reduction will occur, by default -2
    keepdims : bool, optional
        Keep aggregated dimension, by default False

    Returns
    -------
    Tensor
        Zonal mean tensor of x over the latitude dimension
    """
    weights = _compute_lat_weights(lat)
    wm = WeightedMean(weights)
    return wm(x, dim=dim, keepdims=keepdims)


def zonal_var(
    x: Tensor,
    lat: Tensor,
    std: bool = False,
    dim: int = -2,
    keepdims: bool = False,
) -> Tensor:
    """Computes zonal variance, weighting over the latitude direction

    Parameters
    ----------
    x : Tensor
        The tensor [..., H, W] over which the variance will be computed
    lat : Tensor
        A one-dimension tensor [H] representing the latitudes at which the function will
        return weights for
    std : bool, optional
        Return zonal standard deviation, by default False
    dim : int, optional
        The int specifying which dimension of x the reduction will occur, by default -2
    keepdims : bool, optional
        Keep aggregated dimension, by default False

    Returns
    -------
    Tensor
        The variance (or standard deviation) of x over the latitude dimension
    """
    weights = _compute_lat_weights(lat)
    ws = WeightedVariance(weights)
    var = ws(x, dim=dim, keepdims=keepdims)
    if std:
        return torch.sqrt(var)
    else:
        return var


def global_mean(x: Tensor, lat: Tensor, keepdims: bool = False) -> Tensor:
    """Computes global mean

    This function computs the global mean of a lat/lon grid by weighting over the
    latitude direction and then averaging over longitude

    Parameters
    ----------
    x : Tensor
        The lat/lon tensor [..., H, W] over which the mean will be computed
    lat : Tensor
        A one-dimension tensor [H] representing the latitudes at which the function will
        return weights for
    keepdims : bool, optional
        Keep aggregated dimension, by default False

    Returns
    -------
    Tensor
        Global mean tensor
    """
    if not (x.ndim >= 2):
        raise AssertionError(
            "Expected x to have at least two dimensions, with the last two dimensions representing lat and lon respectively"
        )

    # Mean out the latitudes
    lat_reduced = zonal_mean(x, lat, dim=-2, keepdims=keepdims)

    # Return after reduction across longitudes
    return torch.mean(lat_reduced, dim=-1, keepdims=keepdims)


def global_var(
    x: Tensor,
    lat: Tensor,
    std: bool = False,
    keepdims: bool = False,
) -> Tensor:
    """Computes global variance

    This function computs the global variance of a lat/lon grid by weighting over the
    latitude direction and then averaging over longitude

    Parameters
    ----------
    x : Tensor
        The lat/lon tensor [..., H, W] over which the variance will be computed
    lat : Tensor
        A one-dimension tensor [H] representing the latitudes at which the function will
        return weights for
    std : bool, optional
        Return global standard deviation, by default False
    keepdims : bool, optional
        Keep aggregated dimension, by default False

    Returns
    -------
    Tensor
        Global variance tensor
    """
    if not (x.ndim >= 2):
        raise AssertionError(
            "Expected x to have at least two dimensions, with the last two dimensions representing lat and lon respectively"
        )

    # Take global mean, incorporated weights
    gm = global_mean(x, lat, keepdims=True)

    # Take var of lat
    lat_reduced = zonal_mean((x - gm) ** 2, lat, dim=-2, keepdims=keepdims)

    # Take var over longitude
    long_reduce = torch.mean(lat_reduced, dim=-1, keepdims=keepdims)

    if std:
        return torch.sqrt(long_reduce)
    else:
        return long_reduce
