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

"""Dimension-specific Warp launch helpers for point-to-grid forward passes."""

import torch
import warp as wp

from .kernels import FORWARD_KERNELS


def _kernel_param(center_offset: float, interp_id: int, stride: int) -> float | int:
    return int(interp_id) if stride == 2 else float(center_offset)


# Launch 1D point-to-grid forward kernel for the selected stride.
def _launch_forward_1d(
    query_points: torch.Tensor,
    point_values: torch.Tensor,
    out_grid: torch.Tensor,
    start_vals: list[float],
    dx_vals: list[float],
    sizes: list[int],
    center_offset: float,
    interp_id: int,
    stride: int,
    num_points: int,
    wp_device,
    wp_stream,
) -> None:
    points = query_points[:, 0].contiguous()
    wp_points = wp.from_torch(points, dtype=wp.float32)
    wp_values = wp.from_torch(point_values.contiguous())
    wp_out = wp.from_torch(out_grid.contiguous(), return_ctype=True)

    wp.launch(
        FORWARD_KERNELS[1][stride],
        dim=num_points,
        inputs=[
            wp_points,
            wp_values,
            wp_out,
            float(start_vals[0]),
            float(dx_vals[0]),
            int(sizes[0]),
            _kernel_param(center_offset, interp_id, stride),
        ],
        device=wp_device,
        stream=wp_stream,
    )


# Launch 2D point-to-grid forward kernel for the selected stride.
def _launch_forward_2d(
    query_points: torch.Tensor,
    point_values: torch.Tensor,
    out_grid: torch.Tensor,
    start_vals: list[float],
    dx_vals: list[float],
    sizes: list[int],
    center_offset: float,
    interp_id: int,
    stride: int,
    num_points: int,
    wp_device,
    wp_stream,
) -> None:
    wp_points = wp.from_torch(query_points.contiguous(), dtype=wp.vec2f)
    wp_values = wp.from_torch(point_values.contiguous())
    wp_out = wp.from_torch(out_grid.contiguous(), return_ctype=True)
    origin = wp.vec2f(float(start_vals[0]), float(start_vals[1]))
    spacing = wp.vec2f(float(dx_vals[0]), float(dx_vals[1]))
    size = wp.vec2i(int(sizes[0]), int(sizes[1]))

    wp.launch(
        FORWARD_KERNELS[2][stride],
        dim=num_points,
        inputs=[
            wp_points,
            wp_values,
            wp_out,
            origin,
            spacing,
            size,
            _kernel_param(center_offset, interp_id, stride),
        ],
        device=wp_device,
        stream=wp_stream,
    )


# Launch 3D point-to-grid forward kernel for the selected stride.
def _launch_forward_3d(
    query_points: torch.Tensor,
    point_values: torch.Tensor,
    out_grid: torch.Tensor,
    start_vals: list[float],
    dx_vals: list[float],
    sizes: list[int],
    center_offset: float,
    interp_id: int,
    stride: int,
    num_points: int,
    wp_device,
    wp_stream,
) -> None:
    wp_points = wp.from_torch(query_points.contiguous(), dtype=wp.vec3f)
    wp_values = wp.from_torch(point_values.contiguous())
    wp_out = wp.from_torch(out_grid.contiguous(), return_ctype=True)
    origin = wp.vec3f(float(start_vals[0]), float(start_vals[1]), float(start_vals[2]))
    spacing = wp.vec3f(float(dx_vals[0]), float(dx_vals[1]), float(dx_vals[2]))
    size = wp.vec3i(int(sizes[0]), int(sizes[1]), int(sizes[2]))

    wp.launch(
        FORWARD_KERNELS[3][stride],
        dim=num_points,
        inputs=[
            wp_points,
            wp_values,
            wp_out,
            origin,
            spacing,
            size,
            _kernel_param(center_offset, interp_id, stride),
        ],
        device=wp_device,
        stream=wp_stream,
    )


# Dispatch to dimension-specific point-to-grid forward launchers.
def launch_forward(
    query_points: torch.Tensor,
    point_values: torch.Tensor,
    out_grid: torch.Tensor,
    dims: int,
    start_vals: list[float],
    dx_vals: list[float],
    sizes: list[int],
    center_offset: float,
    interp_id: int,
    stride: int,
    num_points: int,
    wp_device,
    wp_stream,
) -> None:
    if dims == 1:
        _launch_forward_1d(
            query_points=query_points,
            point_values=point_values,
            out_grid=out_grid,
            start_vals=start_vals,
            dx_vals=dx_vals,
            sizes=sizes,
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
            point_values=point_values,
            out_grid=out_grid,
            start_vals=start_vals,
            dx_vals=dx_vals,
            sizes=sizes,
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
            point_values=point_values,
            out_grid=out_grid,
            start_vals=start_vals,
            dx_vals=dx_vals,
            sizes=sizes,
            center_offset=center_offset,
            interp_id=interp_id,
            stride=stride,
            num_points=num_points,
            wp_device=wp_device,
            wp_stream=wp_stream,
        )
        return
    raise ValueError(f"Unsupported point-to-grid dimensionality {dims}")


__all__ = ["launch_forward"]
