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

import torch
import warp as wp

from ._kernels import (
    _rectilinear_derivatives_1d_fused_no_mixed_backward_kernel,
    _rectilinear_derivatives_2d_fused_no_mixed_backward_kernel,
    _rectilinear_derivatives_3d_fused_no_mixed_backward_kernel,
    _rectilinear_gradient_1d_backward_kernel,
    _rectilinear_gradient_2d_backward_kernel,
    _rectilinear_gradient_3d_backward_kernel,
    _rectilinear_second_derivative_1d_backward_kernel,
    _rectilinear_second_derivative_2d_backward_kernel,
    _rectilinear_second_derivative_3d_backward_kernel,
)

_BACKWARD_KERNELS = {
    (1, 1): _rectilinear_gradient_1d_backward_kernel,
    (1, 2): _rectilinear_second_derivative_1d_backward_kernel,
    (2, 1): _rectilinear_gradient_2d_backward_kernel,
    (2, 2): _rectilinear_second_derivative_2d_backward_kernel,
    (3, 1): _rectilinear_gradient_3d_backward_kernel,
    (3, 2): _rectilinear_second_derivative_3d_backward_kernel,
}

_FUSED_BACKWARD_NO_MIXED_KERNELS = {
    1: _rectilinear_derivatives_1d_fused_no_mixed_backward_kernel,
    2: _rectilinear_derivatives_2d_fused_no_mixed_backward_kernel,
    3: _rectilinear_derivatives_3d_fused_no_mixed_backward_kernel,
}


def _launch_dim(shape: torch.Size) -> int | tuple[int, ...]:
    """Return Warp launch dimensions for 1D vs ND kernels."""
    return shape[0] if len(shape) == 1 else tuple(shape)


def _to_wp_components(components: list[torch.Tensor], count: int) -> list[wp.array]:
    """Convert the leading tensor components to Warp arrays."""
    return [wp.from_torch(components[i], dtype=wp.float32) for i in range(count)]


def _to_wp_coords(coords_tuple: tuple[torch.Tensor, ...], ndim: int) -> list[wp.array]:
    """Convert coordinate axes to Warp arrays."""
    return [wp.from_torch(coords_tuple[i], dtype=wp.float32) for i in range(ndim)]


def _period_values(period_tuple: tuple[float, ...], ndim: int) -> list[float]:
    """Convert axis periods to float values."""
    return [float(period_tuple[i]) for i in range(ndim)]


def _launch_backward(
    *,
    grad_output_fp32: torch.Tensor,
    coords_tuple: tuple[torch.Tensor, ...],
    period_tuple: tuple[float, ...],
    derivative_order: int,
    grad_field: torch.Tensor,
    wp_device,
    wp_stream,
) -> None:
    ### Launch dimensionality-specific backward kernels.
    ndim = grad_field.ndim
    kernel = _BACKWARD_KERNELS[(ndim, derivative_order)]
    inputs = [
        *_to_wp_components(grad_output_fp32, ndim),
        *_to_wp_coords(coords_tuple, ndim),
        *_period_values(period_tuple, ndim),
        wp.from_torch(grad_field, dtype=wp.float32),
    ]

    with wp.ScopedStream(wp_stream):
        wp.launch(
            kernel=kernel,
            dim=_launch_dim(grad_field.shape),
            inputs=inputs,
            device=wp_device,
            stream=wp_stream,
        )


def _launch_backward_fused_no_mixed(
    *,
    grad_first_components: list[torch.Tensor],
    grad_second_components: list[torch.Tensor],
    coords_tuple: tuple[torch.Tensor, ...],
    period_tuple: tuple[float, ...],
    grad_field: torch.Tensor,
    wp_device,
    wp_stream,
) -> None:
    """Launch dimensionality-specific fused first+second backward kernels."""
    ndim = grad_field.ndim
    kernel = _FUSED_BACKWARD_NO_MIXED_KERNELS[ndim]
    inputs = [
        *_to_wp_components(grad_first_components, ndim),
        *_to_wp_components(grad_second_components, ndim),
        *_to_wp_coords(coords_tuple, ndim),
        *_period_values(period_tuple, ndim),
        wp.from_torch(grad_field, dtype=wp.float32),
    ]

    with wp.ScopedStream(wp_stream):
        wp.launch(
            kernel=kernel,
            dim=_launch_dim(grad_field.shape),
            inputs=inputs,
            device=wp_device,
            stream=wp_stream,
        )
