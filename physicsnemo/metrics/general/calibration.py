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

from physicsnemo.metrics.general.histogram import histogram, linspace

Tensor = torch.Tensor


def find_rank(
    bin_edges: Tensor, counts: Tensor, obs: Union[Tensor, np.ndarray]
) -> Tensor:
    """Finds the rank of the observation with respect to the given counts and bins.


    Parameters
    ----------
    bins_edges : Tensor
        Tensor [N+1, ...] containing bin edges. The leading dimension must represent the
        N+1 bin edges.
    counts : Tensor
        Tensor [N, ...] containing counts, defined over bins. The non-zeroth dimensions
        of bins and counts must be compatible.
    obs : Union[Tensor, np.ndarray]
        Tensor or array containing an observation over which the ranks is computed
        with respect to.

    Returns
    -------
    Tensor
        Tensor of rank for eac of the batched dimensions [...]
    """
    if isinstance(obs, np.ndarray):
        obs = torch.from_numpy(obs).to(counts.device)
    if bin_edges.shape[1:] != counts.shape[1:]:
        raise ValueError(
            "Expected bins and counts to have compatible non-zeroth dimensions but have shapes"
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
            "Expected zeroth dimension of counts to be equal to the zeroth dimension of bins + 1 but have shapes"
            + str(bin_edges.shape[0])
            + " and "
            + str(counts.shape[0])
            + "+1."
        )
    n = torch.sum(counts, dim=0)[0]
    bin_mids = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    right = torch.sum(counts * (bin_mids <= obs[None, ...]), dim=0)

    return right / n


def _rank_probability_score_from_counts(
    rank_bin_edges: Tensor, rank_counts: Tensor
) -> Tensor:
    """Finds the rank of the observation with respect to the given counts and bins.

    Computes

    .. math::

        3 * \int_0^1 (F_X(x) - F_U(x))^2 dx

    where F represents a cumulative distribution function, X represents the rank distribution and
    U represents a Uniform distribution.

    Parameters
    ----------
    rank_bins_edges : Tensor
        Tensor [N+1, ...] containing rank bin edges. The leading dimension must represent the
        N+1 bin edges.
    rank_counts : Tensor
        Tensor [N, ...] containing rank counts, defined over bins. The non-zeroth dimensions
        of bin edges and counts must be compatible.

    Returns
    -------
    Tensor
        Tensor of the Ranked Probability Score for each batched dimension of the input.
    """
    cdf = torch.cumsum(rank_counts, dim=0)
    cdf = cdf / cdf[-1]
    normalization = torch.sum((1.0 - rank_bin_edges[1:]) ** 2, dim=0)
    return torch.sum((cdf - rank_bin_edges[1:]) ** 2, dim=0) / normalization


def rank_probability_score(ranks: Tensor) -> Tensor:
    """
    Computes the Rank Probability Score for the passed ranks.
    Internally, this creates a histogram for the ranks and computes the
    Rank Probability Score (RPS) using the histogram.

    With the histogram the RPS is computed as

    .. math::

        \int_0^1 (F_X(x) - F_U(x))^2 dx

    where F represents a cumulative distribution function,
    X represents the rank distribution and
    U represents a Uniform distribution.

    For computation of the ranks, use _find_rank.

    Parameters
    ----------
    ranks : Tensor
        Tensor [B, ...] containing ranks, where the leading dimension
        represents the batch, or ensemble, dimension.
        The non-zeroth dimensions are batched over.

    Returns
    -------
    Tensor
        Tensor of RPS for each of the batched dimensions [...]
    """
    start = 0.0 * ranks[0, ...]
    end = start + 1.0
    bins = linspace(start, end, 10)
    bin_edges, bin_counts = histogram(ranks, bins=bins)
    return _rank_probability_score_from_counts(bin_edges, bin_counts)
