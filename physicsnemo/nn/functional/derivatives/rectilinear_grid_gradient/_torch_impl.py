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
from .utils import (
    axis_central_weights,
    axis_second_derivative_weights,
    validate_and_normalize_coordinates,
    validate_derivative_request,
    validate_field,
)


def rectilinear_grid_gradient_torch(
    field: torch.Tensor,
    coordinates: Sequence[torch.Tensor],
    periods: float | Sequence[float] | None = None,
    derivative_order: int = 1,
    include_mixed: bool = False,
) -> torch.Tensor:
    """Compute periodic first or pure second derivatives on rectilinear grids."""
    ### Validate field and coordinate inputs.
    validate_field(field)
    derivative_order = validate_derivative_request(
        derivative_order=derivative_order,
        include_mixed=include_mixed,
    )

    coords_tuple, period_tuple = validate_and_normalize_coordinates(
        field=field,
        coordinates=coordinates,
        periods=periods,
        coordinates_dtype=field.dtype,
        requires_grad_error="coordinate gradients are not supported; pass detached coordinates",
    )

    ### Compute per-axis nonuniform periodic central-difference derivatives.
    gradients: list[torch.Tensor] = []
    for axis in range(field.ndim):
        if derivative_order == 1:
            w_minus, w_center, w_plus = axis_central_weights(
                coords_tuple[axis],
                period_tuple[axis],
            )
        else:
            w_minus, w_center, w_plus = axis_second_derivative_weights(
                coords_tuple[axis],
                period_tuple[axis],
            )

        view_shape = [1] * field.ndim
        view_shape[axis] = field.shape[axis]
        w_minus = w_minus.view(view_shape)
        w_center = w_center.view(view_shape)
        w_plus = w_plus.view(view_shape)

        grad_axis = (
            w_minus * torch.roll(field, shifts=1, dims=axis)
            + w_center * field
            + w_plus * torch.roll(field, shifts=-1, dims=axis)
        )
        gradients.append(grad_axis)

    ### Stack per-axis derivative terms into (dims, *field.shape).
    return torch.stack(gradients, dim=0)


def rectilinear_grid_gradient_torch_multi(
    field: torch.Tensor,
    coordinates: Sequence[torch.Tensor],
    periods: float | Sequence[float] | None = None,
    derivative_orders: int | Sequence[int] = 1,
    include_mixed: bool = False,
) -> torch.Tensor:
    """Compute first/second/mixed derivatives from a unified request."""
    requested_orders = normalize_derivative_orders(
        derivative_orders=derivative_orders,
        function_name="rectilinear_grid_gradient",
    )
    mixed_terms = normalize_include_mixed(
        include_mixed=include_mixed,
        function_name="rectilinear_grid_gradient",
    )
    validate_mixed_request(
        derivative_orders=requested_orders,
        include_mixed=mixed_terms,
        ndim=field.ndim,
        function_name="rectilinear_grid_gradient",
    )

    return compose_derivative_outputs(
        field=field,
        requested_orders=requested_orders,
        include_mixed=mixed_terms,
        single_order_fn=lambda input_field, derivative_order: (
            rectilinear_grid_gradient_torch(
                field=input_field,
                coordinates=coordinates,
                periods=periods,
                derivative_order=derivative_order,
                include_mixed=False,
            )
        ),
    )
