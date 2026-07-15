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

from ..utils import (
    _jet_colormap,
    _validate_opacity,
    _validate_transfer_range,
    _validate_vector_field,
)


@wp.kernel
def _vector_field_to_rgba_kernel(
    vector_field: wp.array4d(dtype=wp.float32),
    lic_field: wp.array3d(dtype=wp.float32),
    vmin: wp.float32,
    vmax: wp.float32,
    max_opacity: wp.float32,
    lic_threshold: wp.float32,
    nx: int,
    ny: int,
    nz: int,
    rgba_volume: wp.array4d(dtype=wp.uint8),
):
    i, j, k = wp.tid()
    vx = vector_field[i, j, k, 0]
    vy = vector_field[i, j, k, 1]
    vz = vector_field[i, j, k, 2]
    magnitude = wp.sqrt(vx * vx + vy * vy + vz * vz)
    normalized = wp.min(wp.max((magnitude - vmin) / (vmax - vmin), 0.0), 1.0)
    color = _jet_colormap(normalized)

    lic_value = wp.min(wp.max(lic_field[i, j, k], 0.0), 1.0)
    if lic_value < lic_threshold:
        lic_value = 0.0
    alpha = wp.min(wp.max(lic_value * normalized * max_opacity, 0.0), 1.0)

    rgba_volume[i, j, k, 0] = wp.uint8(color[0] * 255.0)
    rgba_volume[i, j, k, 1] = wp.uint8(color[1] * 255.0)
    rgba_volume[i, j, k, 2] = wp.uint8(color[2] * 255.0)
    rgba_volume[i, j, k, 3] = wp.uint8(alpha * 255.0)


@torch.library.custom_op("physicsnemo::vector_field_to_rgba_warp", mutates_args=())
def vector_field_to_rgba_impl(
    vector_field: torch.Tensor,
    lic_field: torch.Tensor,
    vmin: float,
    vmax: float,
    max_opacity: float = 0.8,
    lic_threshold: float = 0.5,
) -> torch.Tensor:
    """Launch the Warp vector LIC-to-RGBA transfer custom op."""
    _validate_vector_field(vector_field)
    if lic_field.shape != vector_field.shape[:3]:
        raise ValueError(
            "lic_field must have shape matching vector_field spatial dimensions"
        )
    _validate_transfer_range(vmin, vmax)
    _validate_opacity(max_opacity, name="max_opacity")
    _validate_opacity(lic_threshold, name="lic_threshold")

    vector_fp32 = vector_field.to(dtype=torch.float32).contiguous()
    lic_fp32 = lic_field.to(
        device=vector_field.device, dtype=torch.float32
    ).contiguous()
    rgba_volume = torch.empty(
        *vector_field.shape[:3], 4, device=vector_field.device, dtype=torch.uint8
    )
    wp_device, wp_stream = FunctionSpec.warp_launch_context(vector_fp32)
    with FunctionSpec.warp_stream_scope(wp_stream):
        wp.launch(
            _vector_field_to_rgba_kernel,
            dim=tuple(int(size) for size in vector_field.shape[:3]),
            inputs=[
                wp.from_torch(vector_fp32, dtype=wp.float32),
                wp.from_torch(lic_fp32, dtype=wp.float32),
                float(vmin),
                float(vmax),
                float(max_opacity),
                float(lic_threshold),
                int(vector_field.shape[0]),
                int(vector_field.shape[1]),
                int(vector_field.shape[2]),
            ],
            outputs=[wp.from_torch(rgba_volume, dtype=wp.uint8)],
            device=wp_device,
            stream=wp_stream,
        )
    return rgba_volume


@vector_field_to_rgba_impl.register_fake
def _(
    vector_field: torch.Tensor,
    lic_field: torch.Tensor,
    vmin: float,
    vmax: float,
    max_opacity: float = 0.8,
    lic_threshold: float = 0.5,
) -> torch.Tensor:
    return torch.empty(
        *vector_field.shape[:3], 4, device=vector_field.device, dtype=torch.uint8
    )


def vector_field_to_rgba_warp(
    vector_field: torch.Tensor,
    lic_field: torch.Tensor,
    vmin: float,
    vmax: float,
    max_opacity: float = 0.8,
    lic_threshold: float = 0.5,
) -> torch.Tensor:
    """Map vector magnitude and LIC values to RGBA with Warp."""
    return vector_field_to_rgba_impl(
        vector_field,
        lic_field,
        vmin,
        vmax,
        max_opacity,
        lic_threshold,
    )


__all__ = ["vector_field_to_rgba_warp"]
