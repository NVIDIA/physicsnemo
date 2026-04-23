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
    _rectilinear_derivatives_1d_fused_no_mixed_kernel,
    _rectilinear_derivatives_2d_fused_no_mixed_kernel,
    _rectilinear_derivatives_3d_fused_no_mixed_kernel,
    _rectilinear_gradient_1d_kernel,
    _rectilinear_gradient_2d_kernel,
    _rectilinear_gradient_3d_kernel,
    _rectilinear_second_derivative_1d_kernel,
    _rectilinear_second_derivative_2d_kernel,
    _rectilinear_second_derivative_3d_kernel,
)

_FORWARD_KERNELS = {
    (1, 1): _rectilinear_gradient_1d_kernel,
    (1, 2): _rectilinear_second_derivative_1d_kernel,
    (2, 1): _rectilinear_gradient_2d_kernel,
    (2, 2): _rectilinear_second_derivative_2d_kernel,
    (3, 1): _rectilinear_gradient_3d_kernel,
    (3, 2): _rectilinear_second_derivative_3d_kernel,
}

_FUSED_FORWARD_NO_MIXED_KERNELS = {
    1: _rectilinear_derivatives_1d_fused_no_mixed_kernel,
    2: _rectilinear_derivatives_2d_fused_no_mixed_kernel,
    3: _rectilinear_derivatives_3d_fused_no_mixed_kernel,
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


def _launch_forward(
    *,
    field_fp32: torch.Tensor,
    coords_tuple: tuple[torch.Tensor, ...],
    period_tuple: tuple[float, ...],
    derivative_order: int,
    grad_components: list[torch.Tensor],
    wp_device,
    wp_stream,
) -> None:
    ### Launch dimensionality-specific forward kernels.
    ndim = field_fp32.ndim
    kernel = _FORWARD_KERNELS[(ndim, derivative_order)]
    inputs = [
        wp.from_torch(field_fp32, dtype=wp.float32),
        *_to_wp_coords(coords_tuple, ndim),
        *_period_values(period_tuple, ndim),
        *_to_wp_components(grad_components, ndim),
    ]

    with wp.ScopedStream(wp_stream):
        wp.launch(
            kernel=kernel,
            dim=_launch_dim(field_fp32.shape),
            inputs=inputs,
            device=wp_device,
            stream=wp_stream,
        )


def _launch_forward_fused_no_mixed(
    *,
    field_fp32: torch.Tensor,
    coords_tuple: tuple[torch.Tensor, ...],
    period_tuple: tuple[float, ...],
    first_components: list[torch.Tensor],
    second_components: list[torch.Tensor],
    wp_device,
    wp_stream,
) -> None:
    """Launch dimensionality-specific fused first+second derivative kernels."""
    ndim = field_fp32.ndim
    kernel = _FUSED_FORWARD_NO_MIXED_KERNELS[ndim]
    inputs = [
        wp.from_torch(field_fp32, dtype=wp.float32),
        *_to_wp_coords(coords_tuple, ndim),
        *_period_values(period_tuple, ndim),
        *_to_wp_components(first_components, ndim),
        *_to_wp_components(second_components, ndim),
    ]

    with wp.ScopedStream(wp_stream):
        wp.launch(
            kernel=kernel,
            dim=_launch_dim(field_fp32.shape),
            inputs=inputs,
            device=wp_device,
            stream=wp_stream,
        )
