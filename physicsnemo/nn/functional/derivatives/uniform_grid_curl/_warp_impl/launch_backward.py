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
    _launch_dim,
    _to_wp_tensor,
    _wp_launch,
)
from ._kernels import (
    _curl_backward_2d_order2_kernel,
    _curl_backward_2d_order4_kernel,
)

_BACKWARD_KERNELS = {
    2: _curl_backward_2d_order2_kernel,
    4: _curl_backward_2d_order4_kernel,
}


def _launch_backward(
    *,
    grad_output_fp32: torch.Tensor,
    spacing_tuple: tuple[float, ...],
    order: int,
    grad_vector_fp32: torch.Tensor,
    wp_device,
    wp_stream,
) -> None:
    _wp_launch(
        kernel=_BACKWARD_KERNELS[order],
        dim=_launch_dim(grad_output_fp32.shape),
        inputs=[
            _to_wp_tensor(grad_output_fp32),
            *[1.0 / float(dx) for dx in spacing_tuple],
            _to_wp_tensor(grad_vector_fp32),
        ],
        device=wp_device,
        stream=wp_stream,
    )
