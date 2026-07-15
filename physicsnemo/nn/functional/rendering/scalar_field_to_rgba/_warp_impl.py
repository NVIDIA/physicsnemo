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

from ..utils import _jet_colormap, _validate_opacity, _validate_transfer_range


@wp.kernel
def _scalar_field_to_rgba_kernel(
    field: wp.array3d(dtype=wp.float32),
    vmin: wp.float32,
    vmax: wp.float32,
    max_opacity: wp.float32,
    opacity_threshold: wp.float32,
    nx: int,
    ny: int,
    nz: int,
    rgba_volume: wp.array4d(dtype=wp.uint8),
):
    i, j, k = wp.tid()
    value = (field[i, j, k] - vmin) / (vmax - vmin)
    value = wp.min(wp.max(value, 0.0), 1.0)
    color = _jet_colormap(value)

    alpha = value
    if alpha < opacity_threshold:
        alpha = 0.0
    alpha = wp.min(wp.max(alpha * max_opacity, 0.0), 1.0)

    rgba_volume[i, j, k, 0] = wp.uint8(color[0] * 255.0)
    rgba_volume[i, j, k, 1] = wp.uint8(color[1] * 255.0)
    rgba_volume[i, j, k, 2] = wp.uint8(color[2] * 255.0)
    rgba_volume[i, j, k, 3] = wp.uint8(alpha * 255.0)


@torch.library.custom_op("physicsnemo::scalar_field_to_rgba_warp", mutates_args=())
def scalar_field_to_rgba_impl(
    field: torch.Tensor,
    vmin: float,
    vmax: float,
    max_opacity: float = 0.8,
    opacity_threshold: float = 0.1,
) -> torch.Tensor:
    """Launch the Warp scalar-to-RGBA transfer custom op."""
    if field.ndim != 3:
        raise ValueError(
            f"field must have shape (nx, ny, nz), got {tuple(field.shape)}"
        )
    _validate_transfer_range(vmin, vmax)
    _validate_opacity(max_opacity, name="max_opacity")
    _validate_opacity(opacity_threshold, name="opacity_threshold")

    field_fp32 = field.to(dtype=torch.float32).contiguous()
    rgba_volume = torch.empty(*field.shape, 4, device=field.device, dtype=torch.uint8)
    wp_device, wp_stream = FunctionSpec.warp_launch_context(field_fp32)
    with FunctionSpec.warp_stream_scope(wp_stream):
        wp.launch(
            _scalar_field_to_rgba_kernel,
            dim=tuple(int(size) for size in field.shape),
            inputs=[
                wp.from_torch(field_fp32, dtype=wp.float32),
                float(vmin),
                float(vmax),
                float(max_opacity),
                float(opacity_threshold),
                int(field.shape[0]),
                int(field.shape[1]),
                int(field.shape[2]),
            ],
            outputs=[wp.from_torch(rgba_volume, dtype=wp.uint8)],
            device=wp_device,
            stream=wp_stream,
        )
    return rgba_volume


@scalar_field_to_rgba_impl.register_fake
def _(
    field: torch.Tensor,
    vmin: float,
    vmax: float,
    max_opacity: float = 0.8,
    opacity_threshold: float = 0.1,
) -> torch.Tensor:
    return torch.empty(*field.shape, 4, device=field.device, dtype=torch.uint8)


def scalar_field_to_rgba_warp(
    field: torch.Tensor,
    vmin: float,
    vmax: float,
    max_opacity: float = 0.8,
    opacity_threshold: float = 0.1,
) -> torch.Tensor:
    """Map a scalar field to an RGBA volume with Warp."""
    return scalar_field_to_rgba_impl(
        field,
        vmin,
        vmax,
        max_opacity,
        opacity_threshold,
    )


__all__ = ["scalar_field_to_rgba_warp"]
