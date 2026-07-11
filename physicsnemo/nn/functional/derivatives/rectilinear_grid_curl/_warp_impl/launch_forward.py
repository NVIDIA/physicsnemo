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

from physicsnemo.core.function_spec import FunctionSpec

from ..._rectilinear_grid_warp_utils import _launch_dim
from ._kernels import _curl_2d_kernel, _curl_3d_kernel

_FORWARD_KERNELS = {
    2: _curl_2d_kernel,
    3: _curl_3d_kernel,
}


def _launch_forward(
    *,
    vector_field_fp32: torch.Tensor,
    coords_tuple: tuple[torch.Tensor, ...],
    period_tuple: tuple[float, ...],
    output_fp32: torch.Tensor,
    wp_device,
    wp_stream,
) -> None:
    grid_ndim = vector_field_fp32.ndim - 1
    inputs = [
        wp.from_torch(vector_field_fp32, dtype=wp.float32),
        *[wp.from_torch(coords_tuple[i], dtype=wp.float32) for i in range(grid_ndim)],
        *[float(period_tuple[i]) for i in range(grid_ndim)],
        wp.from_torch(output_fp32, dtype=wp.float32),
    ]
    with FunctionSpec.warp_stream_scope(wp_stream):
        wp.launch(
            kernel=_FORWARD_KERNELS[grid_ndim],
            dim=_launch_dim(output_fp32.shape[-grid_ndim:]),
            inputs=inputs,
            device=wp_device,
            stream=wp_stream,
        )
