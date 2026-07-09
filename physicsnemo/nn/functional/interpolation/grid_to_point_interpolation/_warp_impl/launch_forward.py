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

"""Dimension-specific Warp launch helpers for interpolation forward passes.

The enclosing torch custom op owns autograd, so input views explicitly disable
Warp-side gradient allocation.
"""

import torch
import warp as wp

from .kernels import FORWARD_KERNELS


def _kernel_param(center_offset: float, interp_id: int, stride: int) -> float | int:
    return int(interp_id) if stride == 2 else float(center_offset)


# Launch the 1D forward interpolation kernel based on the current stride.
def _launch_forward_1d(
    query_points: torch.Tensor,
    context_grid: torch.Tensor,
    output: torch.Tensor,
    start_vals: list[float],
    dx_vals: list[float],
    padded_sizes: list[int],
    lower_vals: list[float],
    upper_vals: list[float],
    center_offset: float,
    interp_id: int,
    stride: int,
    num_points: int,
    wp_device,
    wp_stream,
) -> None:
    # Convert torch tensors to warp views with dtypes expected by 1D kernels.
    points = query_points[:, 0].contiguous()
    wp_points = wp.from_torch(points, dtype=wp.float32, requires_grad=False)
    wp_grid = wp.from_torch(context_grid.contiguous(), requires_grad=False)
    wp_out = wp.from_torch(output, return_ctype=True)

    inputs = [
        wp_points,
        wp_grid,
        wp_out,
        float(start_vals[0]),
        float(dx_vals[0]),
        int(padded_sizes[0]),
    ]
    if stride == 2:
        inputs.extend(
            [
                float(lower_vals[0]),
                float(upper_vals[0]),
                int(interp_id),
            ]
        )
    else:
        inputs.append(_kernel_param(center_offset, interp_id, stride))

    wp.launch(
        FORWARD_KERNELS[1][stride],
        dim=num_points,
        inputs=inputs,
        device=wp_device,
        stream=wp_stream,
    )


# Launch the 2D forward interpolation kernel based on the current stride.
def _launch_forward_2d(
    query_points: torch.Tensor,
    context_grid: torch.Tensor,
    output: torch.Tensor,
    start_vals: list[float],
    dx_vals: list[float],
    padded_sizes: list[int],
    lower_vals: list[float],
    upper_vals: list[float],
    center_offset: float,
    interp_id: int,
    stride: int,
    num_points: int,
    wp_device,
    wp_stream,
) -> None:
    # Convert torch tensors to warp views with dtypes expected by 2D kernels.
    wp_points = wp.from_torch(
        query_points.contiguous(), dtype=wp.vec2f, requires_grad=False
    )
    wp_grid = wp.from_torch(context_grid.contiguous(), requires_grad=False)
    wp_out = wp.from_torch(output, return_ctype=True)
    origin = wp.vec2f(float(start_vals[0]), float(start_vals[1]))
    spacing = wp.vec2f(float(dx_vals[0]), float(dx_vals[1]))
    size = wp.vec2i(int(padded_sizes[0]), int(padded_sizes[1]))
    logical_lower = wp.vec2f(float(lower_vals[0]), float(lower_vals[1]))
    logical_upper = wp.vec2f(float(upper_vals[0]), float(upper_vals[1]))

    inputs = [wp_points, wp_grid, wp_out, origin, spacing, size]
    if stride == 2:
        inputs.extend([logical_lower, logical_upper, int(interp_id)])
    else:
        inputs.append(_kernel_param(center_offset, interp_id, stride))

    wp.launch(
        FORWARD_KERNELS[2][stride],
        dim=num_points,
        inputs=inputs,
        device=wp_device,
        stream=wp_stream,
    )


# Launch the 3D forward interpolation kernel based on the current stride.
def _launch_forward_3d(
    query_points: torch.Tensor,
    context_grid: torch.Tensor,
    output: torch.Tensor,
    start_vals: list[float],
    dx_vals: list[float],
    padded_sizes: list[int],
    lower_vals: list[float],
    upper_vals: list[float],
    center_offset: float,
    interp_id: int,
    stride: int,
    num_points: int,
    wp_device,
    wp_stream,
) -> None:
    # Convert torch tensors to warp views with dtypes expected by 3D kernels.
    wp_points = wp.from_torch(
        query_points.contiguous(), dtype=wp.vec3f, requires_grad=False
    )
    wp_grid = wp.from_torch(context_grid.contiguous(), requires_grad=False)
    wp_out = wp.from_torch(output, return_ctype=True)
    origin = wp.vec3f(
        float(start_vals[0]),
        float(start_vals[1]),
        float(start_vals[2]),
    )
    spacing = wp.vec3f(float(dx_vals[0]), float(dx_vals[1]), float(dx_vals[2]))
    size = wp.vec3i(
        int(padded_sizes[0]),
        int(padded_sizes[1]),
        int(padded_sizes[2]),
    )
    logical_lower = wp.vec3f(
        float(lower_vals[0]),
        float(lower_vals[1]),
        float(lower_vals[2]),
    )
    logical_upper = wp.vec3f(
        float(upper_vals[0]),
        float(upper_vals[1]),
        float(upper_vals[2]),
    )

    inputs = [wp_points, wp_grid, wp_out, origin, spacing, size]
    if stride == 2:
        inputs.extend([logical_lower, logical_upper, int(interp_id)])
    else:
        inputs.append(_kernel_param(center_offset, interp_id, stride))

    wp.launch(
        FORWARD_KERNELS[3][stride],
        dim=num_points,
        inputs=inputs,
        device=wp_device,
        stream=wp_stream,
    )


# Dispatch to the forward kernel launcher matching input dimensionality.
def launch_forward(
    query_points: torch.Tensor,
    context_grid: torch.Tensor,
    output: torch.Tensor,
    dims: int,
    start_vals: list[float],
    dx_vals: list[float],
    padded_sizes: list[int],
    lower_vals: list[float],
    upper_vals: list[float],
    center_offset: float,
    interp_id: int,
    stride: int,
    num_points: int,
    wp_device,
    wp_stream,
) -> None:
    """Launch the dimension-specific Warp interpolation forward kernel."""
    if dims == 1:
        _launch_forward_1d(
            query_points=query_points,
            context_grid=context_grid,
            output=output,
            start_vals=start_vals,
            dx_vals=dx_vals,
            padded_sizes=padded_sizes,
            lower_vals=lower_vals,
            upper_vals=upper_vals,
            center_offset=center_offset,
            interp_id=interp_id,
            stride=stride,
            num_points=num_points,
            wp_device=wp_device,
            wp_stream=wp_stream,
        )
        return

    if dims == 2:
        _launch_forward_2d(
            query_points=query_points,
            context_grid=context_grid,
            output=output,
            start_vals=start_vals,
            dx_vals=dx_vals,
            padded_sizes=padded_sizes,
            lower_vals=lower_vals,
            upper_vals=upper_vals,
            center_offset=center_offset,
            interp_id=interp_id,
            stride=stride,
            num_points=num_points,
            wp_device=wp_device,
            wp_stream=wp_stream,
        )
        return

    if dims == 3:
        _launch_forward_3d(
            query_points=query_points,
            context_grid=context_grid,
            output=output,
            start_vals=start_vals,
            dx_vals=dx_vals,
            padded_sizes=padded_sizes,
            lower_vals=lower_vals,
            upper_vals=upper_vals,
            center_offset=center_offset,
            interp_id=interp_id,
            stride=stride,
            num_points=num_points,
            wp_device=wp_device,
            wp_stream=wp_stream,
        )
        return

    raise ValueError(f"Unsupported interpolation dimensionality {dims}")


__all__ = ["launch_forward"]
