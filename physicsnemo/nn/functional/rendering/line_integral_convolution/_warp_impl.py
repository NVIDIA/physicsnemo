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
    _sample_seed_trilinear,
    _sample_vector_trilinear,
    _validate_vector_field,
)


@wp.kernel
def _line_integral_convolution_kernel(
    vector_field: wp.array4d(dtype=wp.float32),
    seed: wp.array3d(dtype=wp.float32),
    step_size: wp.float32,
    num_steps: int,
    contrast: wp.float32,
    nx: int,
    ny: int,
    nz: int,
    line_integral: wp.array3d(dtype=wp.float32),
):
    i, j, k = wp.tid()
    pos = wp.vec3(wp.float32(i), wp.float32(j), wp.float32(k))

    total = seed[i, j, k]
    total_weight = wp.float32(1.0)

    for direction_sign in range(2):
        direction_scale = wp.float32(1.0)
        if direction_sign == 1:
            direction_scale = -1.0

        current = pos
        for step in range(num_steps):
            vector = _sample_vector_trilinear(vector_field, current, nx, ny, nz)
            vector_length = wp.length(vector)
            if vector_length <= 1.0e-6:
                break
            vector = direction_scale * vector / vector_length

            mid = current + 0.5 * step_size * vector
            mid_vector = _sample_vector_trilinear(vector_field, mid, nx, ny, nz)
            mid_length = wp.length(mid_vector)
            if mid_length <= 1.0e-6:
                break
            mid_vector = direction_scale * mid_vector / mid_length
            current = current + step_size * mid_vector

            if (
                current[0] < 0.0
                or current[0] > wp.float32(nx - 1)
                or current[1] < 0.0
                or current[1] > wp.float32(ny - 1)
                or current[2] < 0.0
                or current[2] > wp.float32(nz - 1)
            ):
                break

            normalized_step = wp.float32(step + 1) / wp.float32(num_steps + 1)
            weight = 1.0 - normalized_step
            total += _sample_seed_trilinear(seed, current, nx, ny, nz) * weight
            total_weight += weight

    value = total / wp.max(total_weight, 1.0e-6)
    value = wp.min(wp.max(value, 0.0), 1.0)
    value = (value - 0.5) * contrast + 0.5
    line_integral[i, j, k] = wp.min(wp.max(value, 0.0), 1.0)


@torch.library.custom_op("physicsnemo::line_integral_convolution_warp", mutates_args=())
def line_integral_convolution_impl(
    vector_field: torch.Tensor,
    seed: torch.Tensor,
    step_size: float = 0.5,
    num_steps: int = 20,
    contrast: float = 1.4,
) -> torch.Tensor:
    """Launch the Warp line integral convolution custom op."""
    _validate_vector_field(vector_field)
    if seed.shape != vector_field.shape[:3]:
        raise ValueError(
            "seed must have shape matching vector_field spatial dimensions, got "
            f"{tuple(seed.shape)} and {tuple(vector_field.shape[:3])}"
        )
    if step_size <= 0.0:
        raise ValueError("step_size must be strictly positive")
    if num_steps <= 0:
        raise ValueError("num_steps must be strictly positive")
    if contrast <= 0.0:
        raise ValueError("contrast must be strictly positive")

    vector_fp32 = vector_field.to(dtype=torch.float32).contiguous()
    seed_fp32 = seed.to(device=vector_field.device, dtype=torch.float32).contiguous()
    line_integral = torch.empty_like(seed_fp32)
    wp_device, wp_stream = FunctionSpec.warp_launch_context(vector_fp32)
    with FunctionSpec.warp_stream_scope(wp_stream):
        wp.launch(
            _line_integral_convolution_kernel,
            dim=tuple(int(size) for size in seed.shape),
            inputs=[
                wp.from_torch(vector_fp32, dtype=wp.float32),
                wp.from_torch(seed_fp32, dtype=wp.float32),
                float(step_size),
                int(num_steps),
                float(contrast),
                int(seed.shape[0]),
                int(seed.shape[1]),
                int(seed.shape[2]),
            ],
            outputs=[wp.from_torch(line_integral, dtype=wp.float32)],
            device=wp_device,
            stream=wp_stream,
        )
    return line_integral


@line_integral_convolution_impl.register_fake
def _(
    vector_field: torch.Tensor,
    seed: torch.Tensor,
    step_size: float = 0.5,
    num_steps: int = 20,
    contrast: float = 1.4,
) -> torch.Tensor:
    return torch.empty_like(seed, dtype=torch.float32)


def line_integral_convolution_warp(
    vector_field: torch.Tensor,
    seed: torch.Tensor,
    step_size: float = 0.5,
    num_steps: int = 20,
    contrast: float = 1.4,
) -> torch.Tensor:
    """Compute line integral convolution with Warp."""
    return line_integral_convolution_impl(
        vector_field,
        seed,
        step_size,
        num_steps,
        contrast,
    )


__all__ = ["line_integral_convolution_warp"]
