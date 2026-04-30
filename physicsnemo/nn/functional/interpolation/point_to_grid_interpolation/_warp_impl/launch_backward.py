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

"""Warp backward launch surface for point-to-grid interpolation."""

import torch
import warp as wp

from physicsnemo.core.function_spec import FunctionSpec
from physicsnemo.nn.functional.interpolation._warp_common import (
    _INTERP_ID_TO_STRIDE,
    _INTERP_NEAREST,
    interpolation_geometry,
)

from .kernels import BACKWARD_KERNELS


# Restore gradient tensors to caller dtype/shape contract.
def _restore_grad_layout(
    grad_query: torch.Tensor | None,
    grad_point_values: torch.Tensor | None,
    query_points: torch.Tensor,
    point_values: torch.Tensor,
    squeeze_query: bool,
    squeeze_values: bool,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if grad_query is not None and query_points.dtype != torch.float32:
        grad_query = grad_query.to(query_points.dtype)
    if grad_point_values is not None and point_values.dtype != torch.float32:
        grad_point_values = grad_point_values.to(point_values.dtype)
    if grad_query is not None and squeeze_query:
        grad_query = grad_query.squeeze(-1)
    if grad_point_values is not None and squeeze_values:
        grad_point_values = grad_point_values.squeeze(-1)
    return grad_query, grad_point_values


# Launch 1D point-to-grid backward kernel for the selected stride.
def _launch_backward_1d(
    query_points: torch.Tensor,
    point_values: torch.Tensor,
    grad_grid_output: torch.Tensor,
    grad_query: torch.Tensor,
    grad_point_values: torch.Tensor,
    start_vals: list[float],
    dx_vals: list[float],
    sizes: list[int],
    center_offset: float,
    interp_id: int,
    stride: int,
    num_points: int,
    compute_query_grad: int,
    compute_values_grad: int,
    wp_device,
    wp_stream,
) -> None:
    points = query_points[:, 0].contiguous()
    wp_points = wp.from_torch(points, dtype=wp.float32)
    wp_values = wp.from_torch(point_values.contiguous())
    wp_grad_grid = wp.from_torch(grad_grid_output.contiguous())
    wp_grad_query = wp.from_torch(grad_query.contiguous(), return_ctype=True)
    wp_grad_values = wp.from_torch(grad_point_values.contiguous(), return_ctype=True)

    wp.launch(
        BACKWARD_KERNELS[1][stride],
        dim=num_points,
        inputs=[
            wp_points,
            wp_values,
            wp_grad_grid,
            wp_grad_query,
            wp_grad_values,
            float(start_vals[0]),
            float(dx_vals[0]),
            int(sizes[0]),
            int(interp_id) if stride == 2 else float(center_offset),
            int(compute_query_grad),
            int(compute_values_grad),
        ],
        device=wp_device,
        stream=wp_stream,
    )


# Launch 2D point-to-grid backward kernel for the selected stride.
def _launch_backward_2d(
    query_points: torch.Tensor,
    point_values: torch.Tensor,
    grad_grid_output: torch.Tensor,
    grad_query: torch.Tensor,
    grad_point_values: torch.Tensor,
    start_vals: list[float],
    dx_vals: list[float],
    sizes: list[int],
    center_offset: float,
    interp_id: int,
    stride: int,
    num_points: int,
    compute_query_grad: int,
    compute_values_grad: int,
    wp_device,
    wp_stream,
) -> None:
    wp_points = wp.from_torch(query_points.contiguous(), dtype=wp.vec2f)
    wp_values = wp.from_torch(point_values.contiguous())
    wp_grad_grid = wp.from_torch(grad_grid_output.contiguous())
    wp_grad_query = wp.from_torch(grad_query.contiguous(), return_ctype=True)
    wp_grad_values = wp.from_torch(grad_point_values.contiguous(), return_ctype=True)
    origin = wp.vec2f(float(start_vals[0]), float(start_vals[1]))
    spacing = wp.vec2f(float(dx_vals[0]), float(dx_vals[1]))
    size = wp.vec2i(int(sizes[0]), int(sizes[1]))

    wp.launch(
        BACKWARD_KERNELS[2][stride],
        dim=num_points,
        inputs=[
            wp_points,
            wp_values,
            wp_grad_grid,
            wp_grad_query,
            wp_grad_values,
            origin,
            spacing,
            size,
            int(interp_id) if stride == 2 else float(center_offset),
            int(compute_query_grad),
            int(compute_values_grad),
        ],
        device=wp_device,
        stream=wp_stream,
    )


# Launch 3D point-to-grid backward kernel for the selected stride.
def _launch_backward_3d(
    query_points: torch.Tensor,
    point_values: torch.Tensor,
    grad_grid_output: torch.Tensor,
    grad_query: torch.Tensor,
    grad_point_values: torch.Tensor,
    start_vals: list[float],
    dx_vals: list[float],
    sizes: list[int],
    center_offset: float,
    interp_id: int,
    stride: int,
    num_points: int,
    compute_query_grad: int,
    compute_values_grad: int,
    wp_device,
    wp_stream,
) -> None:
    wp_points = wp.from_torch(query_points.contiguous(), dtype=wp.vec3f)
    wp_values = wp.from_torch(point_values.contiguous())
    wp_grad_grid = wp.from_torch(grad_grid_output.contiguous())
    wp_grad_query = wp.from_torch(grad_query.contiguous(), return_ctype=True)
    wp_grad_values = wp.from_torch(grad_point_values.contiguous(), return_ctype=True)
    origin = wp.vec3f(float(start_vals[0]), float(start_vals[1]), float(start_vals[2]))
    spacing = wp.vec3f(float(dx_vals[0]), float(dx_vals[1]), float(dx_vals[2]))
    size = wp.vec3i(int(sizes[0]), int(sizes[1]), int(sizes[2]))

    wp.launch(
        BACKWARD_KERNELS[3][stride],
        dim=num_points,
        inputs=[
            wp_points,
            wp_values,
            wp_grad_grid,
            wp_grad_query,
            wp_grad_values,
            origin,
            spacing,
            size,
            int(interp_id) if stride == 2 else float(center_offset),
            int(compute_query_grad),
            int(compute_values_grad),
        ],
        device=wp_device,
        stream=wp_stream,
    )


# Compute point-to-grid backward via specialized Warp kernels.
def launch_backward(
    query_points: torch.Tensor,
    point_values: torch.Tensor,
    grid_meta: torch.Tensor,
    interp_id: int,
    grad_grid_output: torch.Tensor,
    needs_input_grad: tuple[bool, ...],
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    grid = [(float(g[0]), float(g[1]), int(g[2])) for g in grid_meta.to("cpu").tolist()]
    dims = len(grid)
    if dims < 1 or dims > 3:
        raise ValueError(f"Unsupported dimensionality {dims} for backward")

    squeeze_query = dims == 1 and query_points.ndim == 1
    if squeeze_query:
        query_points = query_points.unsqueeze(-1)
    squeeze_values = point_values.ndim == 1
    if squeeze_values:
        point_values = point_values.unsqueeze(-1)

    query_fp32 = query_points.to(torch.float32)
    values_fp32 = point_values.to(torch.float32)
    grad_grid_fp32 = grad_grid_output.to(torch.float32)

    stride = _INTERP_ID_TO_STRIDE[interp_id]
    start_vals, dx_vals, sizes, center_offset = interpolation_geometry(
        grid, stride, pad_grid=False
    )

    compute_query_grad = int(bool(needs_input_grad[0]) and interp_id != _INTERP_NEAREST)
    compute_values_grad = int(bool(needs_input_grad[1]))
    if compute_query_grad == 0 and compute_values_grad == 0:
        return None, None

    num_points = query_fp32.shape[0]
    channels = values_fp32.shape[1]

    grad_query_work = (
        torch.zeros((num_points, dims), device=query_fp32.device, dtype=torch.float32)
        if compute_query_grad
        else torch.zeros((1, dims), device=query_fp32.device, dtype=torch.float32)
    )
    grad_values_work = (
        torch.zeros(
            (num_points, channels), device=query_fp32.device, dtype=torch.float32
        )
        if compute_values_grad
        else torch.zeros((1, channels), device=query_fp32.device, dtype=torch.float32)
    )

    wp_device, wp_stream = FunctionSpec.warp_launch_context(query_fp32)
    with wp.ScopedStream(wp_stream):
        if dims == 1:
            _launch_backward_1d(
                query_points=query_fp32,
                point_values=values_fp32,
                grad_grid_output=grad_grid_fp32,
                grad_query=grad_query_work,
                grad_point_values=grad_values_work,
                start_vals=start_vals,
                dx_vals=dx_vals,
                sizes=sizes,
                center_offset=center_offset,
                interp_id=interp_id,
                stride=stride,
                num_points=num_points,
                compute_query_grad=compute_query_grad,
                compute_values_grad=compute_values_grad,
                wp_device=wp_device,
                wp_stream=wp_stream,
            )
        elif dims == 2:
            _launch_backward_2d(
                query_points=query_fp32,
                point_values=values_fp32,
                grad_grid_output=grad_grid_fp32,
                grad_query=grad_query_work,
                grad_point_values=grad_values_work,
                start_vals=start_vals,
                dx_vals=dx_vals,
                sizes=sizes,
                center_offset=center_offset,
                interp_id=interp_id,
                stride=stride,
                num_points=num_points,
                compute_query_grad=compute_query_grad,
                compute_values_grad=compute_values_grad,
                wp_device=wp_device,
                wp_stream=wp_stream,
            )
        else:
            _launch_backward_3d(
                query_points=query_fp32,
                point_values=values_fp32,
                grad_grid_output=grad_grid_fp32,
                grad_query=grad_query_work,
                grad_point_values=grad_values_work,
                start_vals=start_vals,
                dx_vals=dx_vals,
                sizes=sizes,
                center_offset=center_offset,
                interp_id=interp_id,
                stride=stride,
                num_points=num_points,
                compute_query_grad=compute_query_grad,
                compute_values_grad=compute_values_grad,
                wp_device=wp_device,
                wp_stream=wp_stream,
            )

    grad_query = grad_query_work if compute_query_grad else None
    grad_values = grad_values_work if compute_values_grad else None
    return _restore_grad_layout(
        grad_query=grad_query,
        grad_point_values=grad_values,
        query_points=query_points,
        point_values=point_values,
        squeeze_query=squeeze_query,
        squeeze_values=squeeze_values,
    )


__all__ = ["launch_backward"]
