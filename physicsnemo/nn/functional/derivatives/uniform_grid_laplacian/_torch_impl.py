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

from __future__ import annotations

from collections.abc import Sequence

import torch

from .utils import validate_scalar_field

_SUPPORTED_ORDERS = (2, 4)


def _normalize_spacing(
    spacing: float | Sequence[float], ndim: int
) -> tuple[float, ...]:
    if isinstance(spacing, (float, int)):
        return tuple(float(spacing) for _ in range(ndim))
    spacing_tuple = tuple(float(x) for x in spacing)
    if len(spacing_tuple) != ndim:
        raise ValueError(
            f"spacing must have {ndim} entries for a {ndim}D field, got {len(spacing_tuple)}"
        )
    return spacing_tuple


def _validate_order(order: int) -> int:
    if not isinstance(order, int):
        raise TypeError(f"order must be an integer, got {type(order)}")
    if order not in _SUPPORTED_ORDERS:
        raise ValueError(
            "uniform_grid_laplacian supports central orders "
            f"{list(_SUPPORTED_ORDERS)}, got order={order}"
        )
    return order


def _second_derivative_order2(
    field: torch.Tensor, axis: int, dx: float
) -> torch.Tensor:
    return (
        torch.roll(field, shifts=-1, dims=axis)
        - 2.0 * field
        + torch.roll(field, shifts=1, dims=axis)
    ) / (dx * dx)


def _second_derivative_order4(
    field: torch.Tensor, axis: int, dx: float
) -> torch.Tensor:
    return (
        -torch.roll(field, shifts=-2, dims=axis)
        + 16.0 * torch.roll(field, shifts=-1, dims=axis)
        - 30.0 * field
        + 16.0 * torch.roll(field, shifts=1, dims=axis)
        - torch.roll(field, shifts=2, dims=axis)
    ) / (12.0 * dx * dx)


_DERIVATIVE_DISPATCH = {
    2: _second_derivative_order2,
    4: _second_derivative_order4,
}


def uniform_grid_laplacian_torch(
    field: torch.Tensor,
    spacing: float | Sequence[float] = 1.0,
    order: int = 2,
) -> torch.Tensor:
    """Compute periodic uniform-grid Laplacian with PyTorch tensor ops."""
    validate_scalar_field(field)
    spacing_tuple = _normalize_spacing(spacing, field.ndim)
    for dx in spacing_tuple:
        if dx <= 0.0:
            raise ValueError("all spacing entries must be strictly positive")
    derivative_fn = _DERIVATIVE_DISPATCH[_validate_order(order)]

    laplacian = torch.zeros_like(field)
    for axis, dx in enumerate(spacing_tuple):
        laplacian = laplacian + derivative_fn(field, axis, dx)
    return laplacian
