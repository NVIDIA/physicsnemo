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

from ...uniform_grid_gradient._warp_impl.utils import (
    _inverse_spacings,
    _launch_dim,
    _to_wp_tensor,
    _wp_launch,
)
from ._kernels import (
    _laplacian_1d_order2_backward_kernel,
    _laplacian_1d_order4_backward_kernel,
    _laplacian_2d_order2_backward_kernel,
    _laplacian_2d_order4_backward_kernel,
    _laplacian_3d_order2_backward_kernel,
    _laplacian_3d_order4_backward_kernel,
)

_BACKWARD_KERNELS = {
    (1, 2): _laplacian_1d_order2_backward_kernel,
    (1, 4): _laplacian_1d_order4_backward_kernel,
    (2, 2): _laplacian_2d_order2_backward_kernel,
    (2, 4): _laplacian_2d_order4_backward_kernel,
    (3, 2): _laplacian_3d_order2_backward_kernel,
    (3, 4): _laplacian_3d_order4_backward_kernel,
}


def _launch_backward(
    *,
    grad_output_fp32: torch.Tensor,
    spacing_tuple: tuple[float, ...],
    order: int,
    grad_field: torch.Tensor,
    wp_device,
    wp_stream,
) -> None:
    """Launch the Laplacian adjoint kernel."""
    inv_sq = _inverse_spacings(spacing_tuple, power=2)
    _wp_launch(
        kernel=_BACKWARD_KERNELS[(grad_field.ndim, order)],
        dim=_launch_dim(grad_field.shape),
        inputs=[
            _to_wp_tensor(grad_output_fp32),
            *inv_sq,
            _to_wp_tensor(grad_field),
        ],
        device=wp_device,
        stream=wp_stream,
    )
