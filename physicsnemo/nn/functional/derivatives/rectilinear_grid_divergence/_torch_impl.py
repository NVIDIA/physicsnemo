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

from .._rectilinear_grid_utils import (
    axis_central_weights,
    validate_and_normalize_coordinates,
)
from .utils import validate_vector_field


def _axis_derivative(
    field: torch.Tensor,
    axis: int,
    coords: torch.Tensor,
    period: float,
) -> torch.Tensor:
    w_minus, w_center, w_plus = axis_central_weights(coords, period)
    view_shape = [1] * field.ndim
    view_shape[axis] = field.shape[axis]
    return (
        w_minus.view(view_shape) * torch.roll(field, shifts=1, dims=axis)
        + w_center.view(view_shape) * field
        + w_plus.view(view_shape) * torch.roll(field, shifts=-1, dims=axis)
    )


def rectilinear_grid_divergence_torch(
    vector_field: torch.Tensor,
    coordinates: Sequence[torch.Tensor],
    periods: float | Sequence[float] | None = None,
) -> torch.Tensor:
    """Compute periodic rectilinear-grid divergence with PyTorch tensor ops."""
    grid_ndim = validate_vector_field(vector_field)
    coords_tuple, period_tuple = validate_and_normalize_coordinates(
        field=vector_field[0],
        coordinates=coordinates,
        periods=periods,
        coordinates_dtype=vector_field.dtype,
        requires_grad_error="coordinate gradients are not supported; pass detached coordinates",
    )

    divergence = torch.zeros_like(vector_field[0])
    for axis in range(grid_ndim):
        divergence = divergence + _axis_derivative(
            vector_field[axis],
            axis,
            coords_tuple[axis],
            period_tuple[axis],
        )
    return divergence
