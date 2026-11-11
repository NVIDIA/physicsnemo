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

from warnings import warn

import torch
from jaxtyping import Float
from torch import Tensor


def wasserstein_from_normal(
    mu0: Float[Tensor, " *batch d"],
    sigma0: Float[Tensor, " *batch d"] | Float[Tensor, " *batch d d"],
    mu1: Float[Tensor, " *batch d"],
    sigma1: Float[Tensor, " *batch d"] | Float[Tensor, " *batch d d"],
) -> Float[Tensor, " *batch"] | Float[Tensor, " *batch d"]:
    """Compute the wasserstein distances between two (possibly multivariate) normal
    distributions.

    Parameters
    ----------
    mu0 : Tensor [B (optional), d1]
        The mean of distribution 0. Can optionally have a batched first dimension.
    sigma0 : Tensor [B (optional), d1, d2 (optional)]
        The variance or covariance of distribution 0. If mu0 has a batched dimension,
        then so must sigma0. If sigma0 is 2 dimension, it is assumed to be a covariance matrix
        and must be symmetric positive definite.
    mu1 : Tensor [B (optional), d1]
        The mean of distribution 1. Can optionally have a batched first dimension.
    sigma1 : Tensor [B (optional), d1, d2 (optional)]
        The variance or covariance of distribution 1. If mu1 has a batched dimension,
        then so must sigma1. If sigma1 is 2 dimension, it is assumed to be a covariance matrix
        and must be symmetric positive definite.

    Returns
    -------
    Tensor [B]
        The wasserstein distance between N(mu0, sigma0) and N(mu1, sigma1)
    """
    mu_ndim = mu0.ndim
    sigma_ndim = sigma0.ndim
    if sigma_ndim == mu_ndim:
        # Univariate normal distribution
        return (mu0 - mu1) ** 2 + (sigma0 + sigma1 - 2 * torch.sqrt(sigma0 * sigma1))

    else:
        # Multivariate normal distribution
        # Compute trace(sig0 + sig1 - 2*(sig0^1/2 * sig1 * sig0^1/2)^1/2) first

        # Compute sig0^1/2 first using eigen decomposition.
        vals0, vecs0 = torch.linalg.eigh(sigma0)
        if torch.any(vals0 < 0.0):
            warn(
                "Warning! Some eigenvalues are less than zero and matrix is not positive definite."
            )
            vals0 = torch.nn.functional.relu(vals0)
        sqrt_sig0 = torch.matmul(
            torch.matmul(vecs0, torch.diag_embed(torch.sqrt(vals0))),
            vecs0.transpose(-2, -1),
        )

        # Compute C = (sig0^1/2 * sig1 * sig0^1/2)
        C = torch.matmul(torch.matmul(sqrt_sig0, sigma1), sqrt_sig0)

        # Compute Csqrt = sqrt( C )
        vals0, vecs0 = torch.linalg.eigh(C)
        if torch.any(vals0 < 0.0):
            warn(
                "Warning! Some eigenvalues are less than zero and matrix is not positive definite."
            )
            vals0 = torch.nn.functional.relu(vals0)
        sqrtC = torch.matmul(
            torch.matmul(vecs0, torch.diag_embed(torch.sqrt(vals0))),
            vecs0.transpose(-2, -1),
        )

        # Compute T = tr(sig0 + sig1 - 2* sqrtC)
        if sigma_ndim > 2:
            T = torch.vmap(torch.trace)(sigma0 + sigma1 - 2 * sqrtC)
        else:
            T = torch.trace(sigma0 + sigma1 - 2 * sqrtC)

        return torch.norm((mu0 - mu1), p=2, dim=-1) ** 2 + T
