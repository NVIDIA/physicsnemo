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

from .._request_utils import (
    compose_derivative_outputs,
    normalize_derivative_orders,
    normalize_include_mixed,
    validate_mixed_request,
)

_SUPPORTED_ORDERS = (2, 4)
_SUPPORTED_DERIVATIVE_ORDERS = (1, 2)


def _normalize_spacing(
    spacing: float | Sequence[float], ndim: int
) -> tuple[float, ...]:
    ### Normalize scalar/list spacing into one value per axis.
    if isinstance(spacing, (float, int)):
        return tuple(float(spacing) for _ in range(ndim))
    spacing_tuple = tuple(float(x) for x in spacing)
    if len(spacing_tuple) != ndim:
        raise ValueError(
            f"spacing must have {ndim} entries for a {ndim}D field, got {len(spacing_tuple)}"
        )
    return spacing_tuple


def _validate_order(order: int) -> int:
    ### Validate finite-difference order selection.
    if not isinstance(order, int):
        raise TypeError(f"order must be an integer, got {type(order)}")
    if order not in _SUPPORTED_ORDERS:
        raise ValueError(
            f"uniform_grid_gradient supports {list(_SUPPORTED_ORDERS)} central orders, got order={order}"
        )
    return order


def _validate_derivative_order(derivative_order: int) -> int:
    ### Validate derivative-order selection (first vs pure second derivative).
    if not isinstance(derivative_order, int):
        raise TypeError(
            f"derivative_order must be an integer, got {type(derivative_order)}"
        )
    if derivative_order not in _SUPPORTED_DERIVATIVE_ORDERS:
        raise ValueError(
            "uniform_grid_gradient supports derivative_order in [1, 2], "
            f"got derivative_order={derivative_order}"
        )
    return derivative_order


def _validate_include_mixed(
    *,
    derivative_order: int,
    include_mixed: bool,
) -> None:
    ### Phase-1 guard: mixed second derivatives are intentionally not yet exposed.
    if not isinstance(include_mixed, bool):
        raise TypeError(f"include_mixed must be a bool, got {type(include_mixed)}")
    if include_mixed and derivative_order != 2:
        raise ValueError("include_mixed is only valid when derivative_order=2")
    if include_mixed:
        raise NotImplementedError(
            "include_mixed=True is not yet supported; phase-1 supports pure axis-wise "
            "second derivatives only"
        )


def _central_derivative_order2(
    field: torch.Tensor, axis: int, dx: float
) -> torch.Tensor:
    ### Second-order periodic central difference.
    return (
        torch.roll(field, shifts=-1, dims=axis) - torch.roll(field, shifts=1, dims=axis)
    ) / (2.0 * dx)


def _central_derivative_order4(
    field: torch.Tensor, axis: int, dx: float
) -> torch.Tensor:
    ### Fourth-order periodic central difference.
    # d/dx f_i ≈ (-f_{i+2} + 8 f_{i+1} - 8 f_{i-1} + f_{i-2}) / (12 dx)
    return (
        -torch.roll(field, shifts=-2, dims=axis)
        + 8.0 * torch.roll(field, shifts=-1, dims=axis)
        - 8.0 * torch.roll(field, shifts=1, dims=axis)
        + torch.roll(field, shifts=2, dims=axis)
    ) / (12.0 * dx)


def _second_derivative_order2(
    field: torch.Tensor, axis: int, dx: float
) -> torch.Tensor:
    ### Second-order periodic second derivative.
    return (
        torch.roll(field, shifts=-1, dims=axis)
        - 2.0 * field
        + torch.roll(field, shifts=1, dims=axis)
    ) / (dx * dx)


def _second_derivative_order4(
    field: torch.Tensor, axis: int, dx: float
) -> torch.Tensor:
    ### Fourth-order periodic second derivative.
    # d2/dx2 f_i ≈ (-f_{i+2} + 16 f_{i+1} - 30 f_i + 16 f_{i-1} - f_{i-2}) / (12 dx^2)
    return (
        -torch.roll(field, shifts=-2, dims=axis)
        + 16.0 * torch.roll(field, shifts=-1, dims=axis)
        - 30.0 * field
        + 16.0 * torch.roll(field, shifts=1, dims=axis)
        - torch.roll(field, shifts=2, dims=axis)
    ) / (12.0 * dx * dx)


_DERIVATIVE_DISPATCH = {
    (1, 2): _central_derivative_order2,
    (1, 4): _central_derivative_order4,
    (2, 2): _second_derivative_order2,
    (2, 4): _second_derivative_order4,
}


def uniform_grid_gradient_torch(
    field: torch.Tensor,
    spacing: float | Sequence[float] = 1.0,
    order: int = 2,
    derivative_order: int = 1,
    include_mixed: bool = False,
) -> torch.Tensor:
    """Compute periodic first or pure second derivatives on a uniform grid."""
    ### Validate field shape and dtype.
    if field.ndim < 1 or field.ndim > 3:
        raise ValueError(
            f"uniform_grid_gradient supports 1D-3D fields, got {field.shape=}"
        )
    if not torch.is_floating_point(field):
        raise TypeError("field must be a floating-point tensor")
    order = _validate_order(order)
    derivative_order = _validate_derivative_order(derivative_order)
    _validate_include_mixed(
        derivative_order=derivative_order,
        include_mixed=include_mixed,
    )

    ### Expand spacing to one entry per field axis.
    spacing_tuple = _normalize_spacing(spacing, field.ndim)

    ### Compute periodic central differences independently per axis.
    gradients: list[torch.Tensor] = []
    derivative_fn = _DERIVATIVE_DISPATCH[(derivative_order, order)]
    for axis, dx in enumerate(spacing_tuple):
        if dx <= 0.0:
            raise ValueError("all spacing entries must be strictly positive")
        ### Periodic axis-wise derivative with configurable derivative/stencil order.
        grad_axis = derivative_fn(field, axis=axis, dx=dx)
        gradients.append(grad_axis)

    ### Stack per-axis derivative terms into (dims, *field.shape).
    return torch.stack(gradients, dim=0)


def uniform_grid_gradient_torch_multi(
    field: torch.Tensor,
    spacing: float | Sequence[float] = 1.0,
    order: int = 2,
    derivative_orders: int | Sequence[int] = 1,
    include_mixed: bool = False,
) -> torch.Tensor:
    """Compute first/second/mixed derivatives from a unified request."""
    requested_orders = normalize_derivative_orders(
        derivative_orders=derivative_orders,
        function_name="uniform_grid_gradient",
    )
    mixed_terms = normalize_include_mixed(
        include_mixed=include_mixed,
        function_name="uniform_grid_gradient",
    )
    validate_mixed_request(
        derivative_orders=requested_orders,
        include_mixed=mixed_terms,
        ndim=field.ndim,
        function_name="uniform_grid_gradient",
    )

    return compose_derivative_outputs(
        field=field,
        requested_orders=requested_orders,
        include_mixed=mixed_terms,
        single_order_fn=lambda input_field, derivative_order: (
            uniform_grid_gradient_torch(
                field=input_field,
                spacing=spacing,
                order=order,
                derivative_order=derivative_order,
                include_mixed=False,
            )
        ),
    )
