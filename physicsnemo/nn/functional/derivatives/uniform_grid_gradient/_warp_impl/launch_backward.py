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

from ._kernels import (
    _uniform_grid_derivatives_1d_order2_fused_backward_kernel,
    _uniform_grid_derivatives_2d_order2_fused_backward_kernel,
    _uniform_grid_derivatives_2d_order2_fused_no_mixed_backward_kernel,
    _uniform_grid_derivatives_3d_order2_fused_backward_kernel,
    _uniform_grid_derivatives_3d_order2_fused_no_mixed_backward_kernel,
    _uniform_grid_gradient_1d_backward_kernel,
    _uniform_grid_gradient_1d_order4_backward_kernel,
    _uniform_grid_gradient_2d_backward_kernel,
    _uniform_grid_gradient_2d_order4_backward_kernel,
    _uniform_grid_gradient_3d_backward_kernel,
    _uniform_grid_gradient_3d_order4_backward_kernel,
    _uniform_grid_second_derivative_1d_backward_kernel,
    _uniform_grid_second_derivative_1d_order4_backward_kernel,
    _uniform_grid_second_derivative_2d_backward_kernel,
    _uniform_grid_second_derivative_2d_order4_backward_kernel,
    _uniform_grid_second_derivative_3d_backward_kernel,
    _uniform_grid_second_derivative_3d_order4_backward_kernel,
)
from .utils import (
    _inverse_spacings,
    _launch_dim,
    _mixed_inverse_spacings,
    _to_wp_components,
    _to_wp_tensor,
    _wp_launch,
)

_BACKWARD_KERNELS = {
    (1, 1, 2): _uniform_grid_gradient_1d_backward_kernel,
    (1, 1, 4): _uniform_grid_gradient_1d_order4_backward_kernel,
    (1, 2, 2): _uniform_grid_second_derivative_1d_backward_kernel,
    (1, 2, 4): _uniform_grid_second_derivative_1d_order4_backward_kernel,
    (2, 1, 2): _uniform_grid_gradient_2d_backward_kernel,
    (2, 1, 4): _uniform_grid_gradient_2d_order4_backward_kernel,
    (2, 2, 2): _uniform_grid_second_derivative_2d_backward_kernel,
    (2, 2, 4): _uniform_grid_second_derivative_2d_order4_backward_kernel,
    (3, 1, 2): _uniform_grid_gradient_3d_backward_kernel,
    (3, 1, 4): _uniform_grid_gradient_3d_order4_backward_kernel,
    (3, 2, 2): _uniform_grid_second_derivative_3d_backward_kernel,
    (3, 2, 4): _uniform_grid_second_derivative_3d_order4_backward_kernel,
}

_FUSED_BACKWARD_KERNELS = {
    (1, False): _uniform_grid_derivatives_1d_order2_fused_backward_kernel,
    (1, True): _uniform_grid_derivatives_1d_order2_fused_backward_kernel,
    (2, False): _uniform_grid_derivatives_2d_order2_fused_no_mixed_backward_kernel,
    (2, True): _uniform_grid_derivatives_2d_order2_fused_backward_kernel,
    (3, False): _uniform_grid_derivatives_3d_order2_fused_no_mixed_backward_kernel,
    (3, True): _uniform_grid_derivatives_3d_order2_fused_backward_kernel,
}


def _launch_backward(
    *,
    grad_output_fp32: torch.Tensor,
    spacing_tuple: tuple[float, ...],
    order: int,
    derivative_order: int,
    grad_field: torch.Tensor,
    wp_device,
    wp_stream,
) -> None:
    ### Launch dimensionality/order-specific backward kernels.
    ndim = grad_field.ndim
    kernel = _BACKWARD_KERNELS[(ndim, derivative_order, order)]
    local_spacing = spacing_tuple[:ndim]
    inv_terms = _inverse_spacings(
        local_spacing,
        power=1 if derivative_order == 1 else 2,
    )

    _wp_launch(
        kernel=kernel,
        dim=_launch_dim(grad_field.shape),
        inputs=[
            *_to_wp_components(grad_output_fp32, ndim),
            *inv_terms,
            _to_wp_tensor(grad_field),
        ],
        device=wp_device,
        stream=wp_stream,
    )


def _launch_backward_fused_order2_no_mixed(
    *,
    grad_first_components: list[torch.Tensor],
    grad_second_components: list[torch.Tensor],
    grad_mixed_components: list[torch.Tensor],
    spacing_tuple: tuple[float, ...],
    include_mixed: bool,
    grad_field: torch.Tensor,
    wp_device,
    wp_stream,
) -> None:
    """Launch fused order-2 backward kernels for first/second/(optional mixed)."""
    ndim = grad_field.ndim
    kernel = _FUSED_BACKWARD_KERNELS[(ndim, include_mixed)]
    local_spacing = spacing_tuple[:ndim]
    inv_first = _inverse_spacings(local_spacing, power=1)
    inv_second = _inverse_spacings(local_spacing, power=2)

    inputs: list = [
        *_to_wp_components(grad_first_components, ndim),
        *_to_wp_components(grad_second_components, ndim),
    ]

    if include_mixed and ndim > 1:
        mixed_count = ndim * (ndim - 1) // 2
        inputs.extend(_to_wp_components(grad_mixed_components, mixed_count))

    inputs.extend(inv_first)
    inputs.extend(inv_second)

    if include_mixed and ndim > 1:
        inputs.extend(_mixed_inverse_spacings(local_spacing))

    inputs.append(_to_wp_tensor(grad_field))

    _wp_launch(
        kernel=kernel,
        dim=_launch_dim(grad_field.shape),
        inputs=inputs,
        device=wp_device,
        stream=wp_stream,
    )
