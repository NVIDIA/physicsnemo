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

"""Private numerical utilities shared by the diffusion solvers."""

from typing import Callable

import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor

_MAX_NUM_POINTS = 8


def _build_gauss_legendre_rules(
    max_num_points: int,
) -> dict[int, tuple[tuple[float, ...], tuple[float, ...]]]:
    """Precompute Gauss-Legendre nodes and weights on [-1, 1] as plain floats,
    so that quadrature calls inside compiled solver steps only see constants."""
    rules = {}
    for num_points in range(1, max_num_points + 1):
        nodes, weights = np.polynomial.legendre.leggauss(num_points)
        rules[num_points] = (tuple(nodes.tolist()), tuple(weights.tolist()))
    return rules


_GAUSS_LEGENDRE_RULES = _build_gauss_legendre_rules(_MAX_NUM_POINTS)


def gauss_legendre(
    fun: Callable[
        [Float[Tensor, " num_points *shape"]],
        Float[Tensor, " num_points *shape"],
    ],
    t0: Float[Tensor, " *shape"],
    t1: Float[Tensor, " *shape"],
    num_points: int,
) -> Float[Tensor, " *shape"]:
    r"""
    Approximate :math:`\int_{t_0}^{t_1} f(\tau) \, d\tau` with Gauss-Legendre
    quadrature.

    Parameters
    ----------
    fun : Callable[[Tensor], Tensor]
        Integrand, evaluated elementwise at the mapped quadrature points of
        shape ``(num_points, *t0.shape)`` and returning the same shape.
    t0 : Tensor
        Lower integration bound, of any shape.
    t1 : Tensor
        Upper integration bound, same shape as ``t0``.
    num_points : int
        Number of quadrature points, between 1 and 8. The rule is exact for
        polynomials up to degree ``2 * num_points - 1``.

    Returns
    -------
    Tensor
        Integral estimate, same shape as ``t0``.

    Raises
    ------
    ValueError
        If ``num_points`` is outside the supported range.
    """
    if not torch.compiler.is_compiling() and num_points not in _GAUSS_LEGENDRE_RULES:
        raise ValueError(
            f"num_points must be an integer in [1, {_MAX_NUM_POINTS}], got "
            f"{num_points}."
        )
    nodes, weights = _GAUSS_LEGENDRE_RULES[num_points]
    lead_shape = (-1,) + (1,) * t0.ndim
    nodes_bc = torch.tensor(nodes, dtype=t0.dtype, device=t0.device).reshape(lead_shape)
    weights_bc = torch.tensor(weights, dtype=t0.dtype, device=t0.device).reshape(
        lead_shape
    )

    midpoint = 0.5 * (t0 + t1)
    half_interval = 0.5 * (t1 - t0)
    points = midpoint + half_interval * nodes_bc

    return half_interval * torch.sum(weights_bc * fun(points), dim=0)
