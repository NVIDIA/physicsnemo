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

# TODO(Dallas) Introduce Distributed Class for computation.

import torch

Tensor = torch.Tensor


def entropy_from_counts(p: Tensor, bin_edges: Tensor, normalized=True) -> Tensor:
    """Computes the Statistical Entropy of a random variable using
    a histogram.

    Uses the formula:

    .. math::

        Entropy(X) = \\int p(x) * \\log( p(x) ) dx

    Parameters
    ----------
    p : Tensor
        Tensor [N, ...] containing counts/pdf, defined over bins. The non-zeroth dimensions
        of bin_edges and p must be compatible.
    bins_edges : Tensor
        Tensor [N+1, ...] containing bin edges. The leading dimension must represent the
        N+1 bin edges.
    normalized : Bool, Optional
        Boolean flag determining whether the returned statistical entropy is normalized.
        Normally the entropy for a compact bounded probability distribution is bounded
        between a pseudo-dirac distribution, ent_min, and a uniform distribution, ent_max.
        This normalization transforms the entropy from [ent_min, ent_max] to [0, 1]

    Returns
    -------
    Tensor
        Tensor containing the Information/Statistical Entropy
    """
    if bin_edges.shape[1:] != p.shape[1:]:
        raise ValueError(
            "Expected bins and pdf to have compatible non-zeroth dimensions but have shapes"
            + str(bin_edges.shape[1:])
            + " and "
            + str(p.shape[1:])
            + "."
        )
    if bin_edges.shape[0] != p.shape[0] + 1:
        raise ValueError(
            "Expected zeroth dimension of cdf to be equal to the zeroth dimension of bins + 1 but have shapes"
            + str(bin_edges.shape[0])
            + " and "
            + str(p.shape[0])
            + "+1."
        )
    dbins = bin_edges[1:] - bin_edges[:-1]
    bin_mids = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    p = p / torch.trapz(p, bin_mids, dim=0) + 1e-8

    ent = torch.trapz(-1.0 * p * torch.log(p), bin_mids, dim=0)
    if normalized:
        max_ent = torch.log(bin_edges[-1] - bin_edges[0])
        min_ent = 0.5 + 0.5 * torch.log(2 * torch.pi * dbins[0] ** 2)
        return (ent - min_ent) / (max_ent - min_ent)
    else:
        return ent


def relative_entropy_from_counts(
    p: Tensor,
    q: Tensor,
    bin_edges: Tensor,
) -> Tensor:
    """Computes the Relative Statistical Entropy, or KL Divergence of two
    random variables using their histograms.

    Uses the formula:

    .. math::

        Entropy(X) = \\int p(x) * \\log( p(x)/q(x) ) dx

    Parameters
    ----------
    p : Tensor
        Tensor [N, ...] containing counts/pdf, defined over bins. The non-zeroth dimensions
        of bin_edges and p must be compatible.
    q : Tensor
        Tensor [N, ...] containing counts/pdf, defined over bins. The non-zeroth dimensions
        of bin_edges and q must be compatible.
    bins_edges : Tensor
        Tensor [N+1, ...] containing bin edges. The leading dimension must represent the
        N+1 bin edges.


    Returns
    -------
    Tensor
        Map of Statistical Entropy
    """
    if bin_edges.shape[1:] != p.shape[1:]:
        raise ValueError(
            "Expected bins and pdf to have compatible non-zeroth dimensions but have shapes"
            + str(bin_edges.shape[1:])
            + " and "
            + str(p.shape[1:])
            + "."
        )
    if bin_edges.shape[0] != p.shape[0] + 1:
        raise ValueError(
            "Expected zeroth dimension of cdf to be equal to the zeroth dimension of bins + 1 but have shapes"
            + str(bin_edges.shape[0])
            + " and "
            + str(p.shape[0])
            + "+1."
        )

    if p.shape != q.shape:
        raise ValueError(
            "Expected p and q to have compatible shapes but have shapes"
            + str(p.shape)
            + " and "
            + str(q.shape)
            + "."
        )
    bin_mids = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    p = p / torch.trapz(p, bin_mids, dim=0) + 1e-8
    q = q / torch.trapz(q, bin_mids, dim=0) + 1e-8

    return torch.trapz(p * torch.log(p / q), bin_mids, dim=0)
