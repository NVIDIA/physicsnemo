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

from typing import List, Tuple

import torch
import torch.nn.functional as F
import warp as wp

from physicsnemo.core.function_spec import FunctionSpec

# Define interpolation identifiers used by both Python and Warp kernels.
_INTERP_NEAREST = 0
_INTERP_LINEAR = 1
_INTERP_SMOOTH_1 = 2
_INTERP_SMOOTH_2 = 3
_INTERP_GAUSSIAN = 4

# Map interpolation names to internal ids.
_INTERP_NAME_TO_ID = {
    "nearest_neighbor": _INTERP_NEAREST,
    "linear": _INTERP_LINEAR,
    "smooth_step_1": _INTERP_SMOOTH_1,
    "smooth_step_2": _INTERP_SMOOTH_2,
    "gaussian": _INTERP_GAUSSIAN,
}

# Map interpolation ids back to their string names for autograd parity.
_INTERP_ID_TO_NAME = {v: k for k, v in _INTERP_NAME_TO_ID.items()}

# Map interpolation ids to their neighborhood stride.
_INTERP_ID_TO_STRIDE = {
    _INTERP_NEAREST: 1,
    _INTERP_LINEAR: 2,
    _INTERP_SMOOTH_1: 2,
    _INTERP_SMOOTH_2: 2,
    _INTERP_GAUSSIAN: 5,
}

# Initialize Warp once for kernel launch.
wp.config.quiet = True
wp.init()


# Define scalar basis functions used by linear and smooth interpolation modes.
@wp.func
def _smooth_step_1(x: wp.float32) -> wp.float32:
    return wp.clamp(3.0 * x * x - 2.0 * x * x * x, 0.0, 1.0)


@wp.func
def _smooth_step_2(x: wp.float32) -> wp.float32:
    return wp.clamp(x * x * x * (6.0 * x * x - 15.0 * x + 10.0), 0.0, 1.0)


@wp.func
def _basis_value(interp_id: int, x: wp.float32) -> wp.float32:
    if interp_id == _INTERP_SMOOTH_1:
        return _smooth_step_1(x)
    if interp_id == _INTERP_SMOOTH_2:
        return _smooth_step_2(x)
    return x


# 1D interpolation kernels.
@wp.kernel
def _interp_1d_stride1(
    points: wp.array(dtype=wp.float32),
    grid: wp.array2d(dtype=wp.float32),
    out: wp.array2d(dtype=wp.float32),
    origin: wp.float32,
    dx: wp.float32,
    size_x: int,
    center_offset: wp.float32,
):
    tid = wp.tid()
    x = points[tid]
    center = wp.int32((x - origin) / dx + center_offset)
    if center < 0:
        center = 0
    if center >= size_x:
        center = size_x - 1
    for c in range(grid.shape[0]):
        out[tid, c] = grid[c, center]


@wp.kernel
def _interp_1d_stride2(
    points: wp.array(dtype=wp.float32),
    grid: wp.array2d(dtype=wp.float32),
    out: wp.array2d(dtype=wp.float32),
    origin: wp.float32,
    dx: wp.float32,
    size_x: int,
    interp_id: int,
):
    tid = wp.tid()
    x = points[tid]
    pos = (x - origin) / dx
    center = wp.int32(pos)
    frac = pos - wp.float32(center)
    lower = _basis_value(interp_id, frac)
    upper = _basis_value(interp_id, 1.0 - frac)
    idx0 = center
    idx1 = center + 1
    if idx0 < 0:
        idx0 = 0
    if idx0 >= size_x:
        idx0 = size_x - 1
    if idx1 < 0:
        idx1 = 0
    if idx1 >= size_x:
        idx1 = size_x - 1
    for c in range(grid.shape[0]):
        out[tid, c] = upper * grid[c, idx0] + lower * grid[c, idx1]


@wp.kernel
def _interp_1d_stride5(
    points: wp.array(dtype=wp.float32),
    grid: wp.array2d(dtype=wp.float32),
    out: wp.array2d(dtype=wp.float32),
    origin: wp.float32,
    dx: wp.float32,
    size_x: int,
    center_offset: wp.float32,
):
    tid = wp.tid()
    x = points[tid]
    pos = (x - origin) / dx
    center = wp.int32(pos + center_offset)
    sigma = dx / 2.0
    sum_w = 0.0
    for c in range(grid.shape[0]):
        out[tid, c] = 0.0
    for ox in range(-2, 3):
        idx = center + ox
        if idx < 0:
            idx = 0
        if idx >= size_x:
            idx = size_x - 1
        coord = origin + wp.float32(idx) * dx
        dist = (x - coord) / sigma
        weight = wp.exp(-0.5 * dist * dist)
        sum_w += weight
        for c in range(grid.shape[0]):
            out[tid, c] += weight * grid[c, idx]
    if sum_w > 0.0:
        inv = 1.0 / sum_w
        for c in range(grid.shape[0]):
            out[tid, c] = out[tid, c] * inv


# 2D interpolation kernels.
@wp.kernel
def _interp_2d_stride1(
    points: wp.array(dtype=wp.vec2f),
    grid: wp.array3d(dtype=wp.float32),
    out: wp.array2d(dtype=wp.float32),
    origin: wp.vec2f,
    dx: wp.vec2f,
    size: wp.vec2i,
    center_offset: wp.float32,
):
    tid = wp.tid()
    p = points[tid]
    pos = wp.vec2f((p[0] - origin[0]) / dx[0], (p[1] - origin[1]) / dx[1])
    center_x = wp.int32(pos[0] + center_offset)
    center_y = wp.int32(pos[1] + center_offset)
    if center_x < 0:
        center_x = 0
    if center_x >= size[0]:
        center_x = size[0] - 1
    if center_y < 0:
        center_y = 0
    if center_y >= size[1]:
        center_y = size[1] - 1
    for c in range(grid.shape[0]):
        out[tid, c] = grid[c, center_x, center_y]


@wp.kernel
def _interp_2d_stride2(
    points: wp.array(dtype=wp.vec2f),
    grid: wp.array3d(dtype=wp.float32),
    out: wp.array2d(dtype=wp.float32),
    origin: wp.vec2f,
    dx: wp.vec2f,
    size: wp.vec2i,
    interp_id: int,
):
    tid = wp.tid()
    p = points[tid]
    pos = wp.vec2f((p[0] - origin[0]) / dx[0], (p[1] - origin[1]) / dx[1])
    center_x = wp.int32(pos[0])
    center_y = wp.int32(pos[1])
    frac_x = pos[0] - wp.float32(center_x)
    frac_y = pos[1] - wp.float32(center_y)
    lower_x = _basis_value(interp_id, frac_x)
    upper_x = _basis_value(interp_id, 1.0 - frac_x)
    lower_y = _basis_value(interp_id, frac_y)
    upper_y = _basis_value(interp_id, 1.0 - frac_y)
    idx_x0 = center_x
    idx_x1 = center_x + 1
    idx_y0 = center_y
    idx_y1 = center_y + 1
    if idx_x0 < 0:
        idx_x0 = 0
    if idx_x0 >= size[0]:
        idx_x0 = size[0] - 1
    if idx_x1 < 0:
        idx_x1 = 0
    if idx_x1 >= size[0]:
        idx_x1 = size[0] - 1
    if idx_y0 < 0:
        idx_y0 = 0
    if idx_y0 >= size[1]:
        idx_y0 = size[1] - 1
    if idx_y1 < 0:
        idx_y1 = 0
    if idx_y1 >= size[1]:
        idx_y1 = size[1] - 1
    for c in range(grid.shape[0]):
        out[tid, c] = (
            upper_x * upper_y * grid[c, idx_x0, idx_y0]
            + upper_x * lower_y * grid[c, idx_x0, idx_y1]
            + lower_x * upper_y * grid[c, idx_x1, idx_y0]
            + lower_x * lower_y * grid[c, idx_x1, idx_y1]
        )


@wp.kernel
def _interp_2d_stride5(
    points: wp.array(dtype=wp.vec2f),
    grid: wp.array3d(dtype=wp.float32),
    out: wp.array2d(dtype=wp.float32),
    origin: wp.vec2f,
    dx: wp.vec2f,
    size: wp.vec2i,
    center_offset: wp.float32,
):
    tid = wp.tid()
    p = points[tid]
    pos = wp.vec2f((p[0] - origin[0]) / dx[0], (p[1] - origin[1]) / dx[1])
    center_x = wp.int32(pos[0] + center_offset)
    center_y = wp.int32(pos[1] + center_offset)
    sigma_x = dx[0] / 2.0
    sigma_y = dx[1] / 2.0
    sum_w = 0.0
    for c in range(grid.shape[0]):
        out[tid, c] = 0.0
    for ox in range(-2, 3):
        idx_x = center_x + ox
        if idx_x < 0:
            idx_x = 0
        if idx_x >= size[0]:
            idx_x = size[0] - 1
        coord_x = origin[0] + wp.float32(idx_x) * dx[0]
        dist_x = (p[0] - coord_x) / sigma_x
        gx = wp.exp(-0.5 * dist_x * dist_x)
        for oy in range(-2, 3):
            idx_y = center_y + oy
            if idx_y < 0:
                idx_y = 0
            if idx_y >= size[1]:
                idx_y = size[1] - 1
            coord_y = origin[1] + wp.float32(idx_y) * dx[1]
            dist_y = (p[1] - coord_y) / sigma_y
            weight = gx * wp.exp(-0.5 * dist_y * dist_y)
            sum_w += weight
            for c in range(grid.shape[0]):
                out[tid, c] += weight * grid[c, idx_x, idx_y]
    if sum_w > 0.0:
        inv = 1.0 / sum_w
        for c in range(grid.shape[0]):
            out[tid, c] = out[tid, c] * inv


# 3D interpolation kernels.
@wp.kernel
def _interp_3d_stride1(
    points: wp.array(dtype=wp.vec3f),
    grid: wp.array4d(dtype=wp.float32),
    out: wp.array2d(dtype=wp.float32),
    origin: wp.vec3f,
    dx: wp.vec3f,
    size: wp.vec3i,
    center_offset: wp.float32,
):
    tid = wp.tid()
    p = points[tid]
    pos = wp.vec3f(
        (p[0] - origin[0]) / dx[0],
        (p[1] - origin[1]) / dx[1],
        (p[2] - origin[2]) / dx[2],
    )
    center_x = wp.int32(pos[0] + center_offset)
    center_y = wp.int32(pos[1] + center_offset)
    center_z = wp.int32(pos[2] + center_offset)
    if center_x < 0:
        center_x = 0
    if center_x >= size[0]:
        center_x = size[0] - 1
    if center_y < 0:
        center_y = 0
    if center_y >= size[1]:
        center_y = size[1] - 1
    if center_z < 0:
        center_z = 0
    if center_z >= size[2]:
        center_z = size[2] - 1
    for c in range(grid.shape[0]):
        out[tid, c] = grid[c, center_x, center_y, center_z]


@wp.kernel
def _interp_3d_stride2(
    points: wp.array(dtype=wp.vec3f),
    grid: wp.array4d(dtype=wp.float32),
    out: wp.array2d(dtype=wp.float32),
    origin: wp.vec3f,
    dx: wp.vec3f,
    size: wp.vec3i,
    interp_id: int,
):
    tid = wp.tid()
    p = points[tid]
    pos = wp.vec3f(
        (p[0] - origin[0]) / dx[0],
        (p[1] - origin[1]) / dx[1],
        (p[2] - origin[2]) / dx[2],
    )
    center_x = wp.int32(pos[0])
    center_y = wp.int32(pos[1])
    center_z = wp.int32(pos[2])
    frac_x = pos[0] - wp.float32(center_x)
    frac_y = pos[1] - wp.float32(center_y)
    frac_z = pos[2] - wp.float32(center_z)
    lower_x = _basis_value(interp_id, frac_x)
    upper_x = _basis_value(interp_id, 1.0 - frac_x)
    lower_y = _basis_value(interp_id, frac_y)
    upper_y = _basis_value(interp_id, 1.0 - frac_y)
    lower_z = _basis_value(interp_id, frac_z)
    upper_z = _basis_value(interp_id, 1.0 - frac_z)
    idx_x0 = center_x
    idx_x1 = center_x + 1
    idx_y0 = center_y
    idx_y1 = center_y + 1
    idx_z0 = center_z
    idx_z1 = center_z + 1
    if idx_x0 < 0:
        idx_x0 = 0
    if idx_x0 >= size[0]:
        idx_x0 = size[0] - 1
    if idx_x1 < 0:
        idx_x1 = 0
    if idx_x1 >= size[0]:
        idx_x1 = size[0] - 1
    if idx_y0 < 0:
        idx_y0 = 0
    if idx_y0 >= size[1]:
        idx_y0 = size[1] - 1
    if idx_y1 < 0:
        idx_y1 = 0
    if idx_y1 >= size[1]:
        idx_y1 = size[1] - 1
    if idx_z0 < 0:
        idx_z0 = 0
    if idx_z0 >= size[2]:
        idx_z0 = size[2] - 1
    if idx_z1 < 0:
        idx_z1 = 0
    if idx_z1 >= size[2]:
        idx_z1 = size[2] - 1
    for c in range(grid.shape[0]):
        out[tid, c] = (
            upper_x * upper_y * upper_z * grid[c, idx_x0, idx_y0, idx_z0]
            + upper_x * upper_y * lower_z * grid[c, idx_x0, idx_y0, idx_z1]
            + upper_x * lower_y * upper_z * grid[c, idx_x0, idx_y1, idx_z0]
            + upper_x * lower_y * lower_z * grid[c, idx_x0, idx_y1, idx_z1]
            + lower_x * upper_y * upper_z * grid[c, idx_x1, idx_y0, idx_z0]
            + lower_x * upper_y * lower_z * grid[c, idx_x1, idx_y0, idx_z1]
            + lower_x * lower_y * upper_z * grid[c, idx_x1, idx_y1, idx_z0]
            + lower_x * lower_y * lower_z * grid[c, idx_x1, idx_y1, idx_z1]
        )


@wp.kernel
def _interp_3d_stride5(
    points: wp.array(dtype=wp.vec3f),
    grid: wp.array4d(dtype=wp.float32),
    out: wp.array2d(dtype=wp.float32),
    origin: wp.vec3f,
    dx: wp.vec3f,
    size: wp.vec3i,
    center_offset: wp.float32,
):
    tid = wp.tid()
    p = points[tid]
    pos = wp.vec3f(
        (p[0] - origin[0]) / dx[0],
        (p[1] - origin[1]) / dx[1],
        (p[2] - origin[2]) / dx[2],
    )
    center_x = wp.int32(pos[0] + center_offset)
    center_y = wp.int32(pos[1] + center_offset)
    center_z = wp.int32(pos[2] + center_offset)
    sigma_x = dx[0] / 2.0
    sigma_y = dx[1] / 2.0
    sigma_z = dx[2] / 2.0
    sum_w = 0.0
    for c in range(grid.shape[0]):
        out[tid, c] = 0.0
    for ox in range(-2, 3):
        idx_x = center_x + ox
        if idx_x < 0:
            idx_x = 0
        if idx_x >= size[0]:
            idx_x = size[0] - 1
        coord_x = origin[0] + wp.float32(idx_x) * dx[0]
        dist_x = (p[0] - coord_x) / sigma_x
        gx = wp.exp(-0.5 * dist_x * dist_x)
        for oy in range(-2, 3):
            idx_y = center_y + oy
            if idx_y < 0:
                idx_y = 0
            if idx_y >= size[1]:
                idx_y = size[1] - 1
            coord_y = origin[1] + wp.float32(idx_y) * dx[1]
            dist_y = (p[1] - coord_y) / sigma_y
            gy = wp.exp(-0.5 * dist_y * dist_y)
            for oz in range(-2, 3):
                idx_z = center_z + oz
                if idx_z < 0:
                    idx_z = 0
                if idx_z >= size[2]:
                    idx_z = size[2] - 1
                coord_z = origin[2] + wp.float32(idx_z) * dx[2]
                dist_z = (p[2] - coord_z) / sigma_z
                weight = gx * gy * wp.exp(-0.5 * dist_z * dist_z)
                sum_w += weight
                for c in range(grid.shape[0]):
                    out[tid, c] += weight * grid[c, idx_x, idx_y, idx_z]
    if sum_w > 0.0:
        inv = 1.0 / sum_w
        for c in range(grid.shape[0]):
            out[tid, c] = out[tid, c] * inv


# Launch helpers keep per-dimension kernel invocation logic in one place.
def _launch_1d(
    query_points: torch.Tensor,
    context_grid: torch.Tensor,
    output: torch.Tensor,
    start_vals: list[float],
    dx_vals: list[float],
    padded_sizes: list[int],
    center_offset: float,
    interp_id: int,
    stride: int,
    num_points: int,
    wp_device,
    wp_stream,
) -> None:
    points = query_points[:, 0].contiguous()
    wp_points = wp.from_torch(points, dtype=wp.float32)
    wp_grid = wp.from_torch(context_grid.contiguous())
    wp_out = wp.from_torch(output, return_ctype=True)
    if stride == 1:
        wp.launch(
            _interp_1d_stride1,
            dim=num_points,
            inputs=[
                wp_points,
                wp_grid,
                wp_out,
                float(start_vals[0]),
                float(dx_vals[0]),
                int(padded_sizes[0]),
                float(center_offset),
            ],
            device=wp_device,
            stream=wp_stream,
        )
    elif stride == 2:
        wp.launch(
            _interp_1d_stride2,
            dim=num_points,
            inputs=[
                wp_points,
                wp_grid,
                wp_out,
                float(start_vals[0]),
                float(dx_vals[0]),
                int(padded_sizes[0]),
                int(interp_id),
            ],
            device=wp_device,
            stream=wp_stream,
        )
    else:
        wp.launch(
            _interp_1d_stride5,
            dim=num_points,
            inputs=[
                wp_points,
                wp_grid,
                wp_out,
                float(start_vals[0]),
                float(dx_vals[0]),
                int(padded_sizes[0]),
                float(center_offset),
            ],
            device=wp_device,
            stream=wp_stream,
        )


def _launch_2d(
    query_points: torch.Tensor,
    context_grid: torch.Tensor,
    output: torch.Tensor,
    start_vals: list[float],
    dx_vals: list[float],
    padded_sizes: list[int],
    center_offset: float,
    interp_id: int,
    stride: int,
    num_points: int,
    wp_device,
    wp_stream,
) -> None:
    wp_points = wp.from_torch(query_points.contiguous(), dtype=wp.vec2f)
    wp_grid = wp.from_torch(context_grid.contiguous())
    wp_out = wp.from_torch(output, return_ctype=True)
    origin = wp.vec2f(float(start_vals[0]), float(start_vals[1]))
    spacing = wp.vec2f(float(dx_vals[0]), float(dx_vals[1]))
    size = wp.vec2i(int(padded_sizes[0]), int(padded_sizes[1]))
    if stride == 1:
        wp.launch(
            _interp_2d_stride1,
            dim=num_points,
            inputs=[
                wp_points,
                wp_grid,
                wp_out,
                origin,
                spacing,
                size,
                float(center_offset),
            ],
            device=wp_device,
            stream=wp_stream,
        )
    elif stride == 2:
        wp.launch(
            _interp_2d_stride2,
            dim=num_points,
            inputs=[
                wp_points,
                wp_grid,
                wp_out,
                origin,
                spacing,
                size,
                int(interp_id),
            ],
            device=wp_device,
            stream=wp_stream,
        )
    else:
        wp.launch(
            _interp_2d_stride5,
            dim=num_points,
            inputs=[
                wp_points,
                wp_grid,
                wp_out,
                origin,
                spacing,
                size,
                float(center_offset),
            ],
            device=wp_device,
            stream=wp_stream,
        )


def _launch_3d(
    query_points: torch.Tensor,
    context_grid: torch.Tensor,
    output: torch.Tensor,
    start_vals: list[float],
    dx_vals: list[float],
    padded_sizes: list[int],
    center_offset: float,
    interp_id: int,
    stride: int,
    num_points: int,
    wp_device,
    wp_stream,
) -> None:
    wp_points = wp.from_torch(query_points.contiguous(), dtype=wp.vec3f)
    wp_grid = wp.from_torch(context_grid.contiguous())
    wp_out = wp.from_torch(output, return_ctype=True)
    origin = wp.vec3f(float(start_vals[0]), float(start_vals[1]), float(start_vals[2]))
    spacing = wp.vec3f(float(dx_vals[0]), float(dx_vals[1]), float(dx_vals[2]))
    size = wp.vec3i(
        int(padded_sizes[0]),
        int(padded_sizes[1]),
        int(padded_sizes[2]),
    )
    if stride == 1:
        wp.launch(
            _interp_3d_stride1,
            dim=num_points,
            inputs=[
                wp_points,
                wp_grid,
                wp_out,
                origin,
                spacing,
                size,
                float(center_offset),
            ],
            device=wp_device,
            stream=wp_stream,
        )
    elif stride == 2:
        wp.launch(
            _interp_3d_stride2,
            dim=num_points,
            inputs=[
                wp_points,
                wp_grid,
                wp_out,
                origin,
                spacing,
                size,
                int(interp_id),
            ],
            device=wp_device,
            stream=wp_stream,
        )
    else:
        wp.launch(
            _interp_3d_stride5,
            dim=num_points,
            inputs=[
                wp_points,
                wp_grid,
                wp_out,
                origin,
                spacing,
                size,
                float(center_offset),
            ],
            device=wp_device,
            stream=wp_stream,
        )


# Register the warp-backed interpolation op with torch custom ops.
@torch.library.custom_op("physicsnemo::interpolation_warp", mutates_args=())
def interpolation_impl(
    query_points: torch.Tensor,
    context_grid: torch.Tensor,
    grid_meta: torch.Tensor,
    interp_id: int,
    mem_speed_trade: bool = True,
) -> torch.Tensor:
    # Keep signature parity with the torch implementation API.
    _ = mem_speed_trade

    # Validate the tensor/device contract before launching any kernels.
    if query_points.device != context_grid.device:
        raise ValueError("query_points and context_grid must be on the same device")

    if grid_meta.ndim != 2 or grid_meta.shape[1] != 3:
        raise ValueError(
            "grid metadata must have shape (dims, 3) with (min, max, size)"
        )

    # Normalize grid metadata to Python tuples for launch preparation.
    grid = grid_meta.to("cpu").tolist()
    grid = [(float(g[0]), float(g[1]), int(g[2])) for g in grid]
    dims = len(grid)
    if dims < 1 or dims > 3:
        raise ValueError("warp interpolation supports 1-3D grids")

    if query_points.ndim == 1 and dims == 1:
        query_points = query_points.unsqueeze(-1)

    if query_points.shape[-1] != dims:
        raise ValueError(
            f"query_points must have last dimension {dims}, got {query_points.shape}"
        )

    grid_sizes = [g[2] for g in grid]
    if list(context_grid.shape[1:]) != grid_sizes:
        raise ValueError(
            "context_grid shape must match grid sizes: "
            f"expected {grid_sizes}, got {list(context_grid.shape[1:])}"
        )

    stride = _INTERP_ID_TO_STRIDE.get(interp_id)
    if stride is None:
        raise ValueError(f"Unsupported interpolation id {interp_id}")

    # Pad the source grid for non-nearest kernels to match neighborhood access.
    k = stride // 2
    padding = dims * (k, k)
    if k > 0:
        context_grid = F.pad(context_grid, padding)

    # Normalize inputs to float32 for warp kernels and restore output dtype later.
    input_dtype = context_grid.dtype
    if input_dtype != torch.float32:
        context_grid = context_grid.to(torch.float32)
    if query_points.dtype != torch.float32:
        query_points = query_points.to(torch.float32)

    device = query_points.device
    num_points = query_points.shape[0]
    num_channels = context_grid.shape[0]
    output = torch.empty(
        (num_points, num_channels),
        device=device,
        dtype=torch.float32,
    )

    # Precompute grid geometry used by all dimension-specific launches.
    dx_vals = [(g[1] - g[0]) / (g[2] - 1) for g in grid]
    start_vals = [g[0] - k * dx for g, dx in zip(grid, dx_vals)]
    padded_sizes = [size + 2 * k for size in grid_sizes]
    center_offset = 0.5 if stride % 2 == 1 else 0.0

    # Resolve warp device/stream from torch inputs.
    wp_device, wp_stream = FunctionSpec.warp_launch_context(query_points)

    # Launch the specialized interpolation kernel for the input dimensionality.
    with wp.ScopedStream(wp_stream):
        if dims == 1:
            _launch_1d(
                query_points,
                context_grid,
                output,
                start_vals,
                dx_vals,
                padded_sizes,
                center_offset,
                interp_id,
                stride,
                num_points,
                wp_device,
                wp_stream,
            )
        elif dims == 2:
            _launch_2d(
                query_points,
                context_grid,
                output,
                start_vals,
                dx_vals,
                padded_sizes,
                center_offset,
                interp_id,
                stride,
                num_points,
                wp_device,
                wp_stream,
            )
        else:
            _launch_3d(
                query_points,
                context_grid,
                output,
                start_vals,
                dx_vals,
                padded_sizes,
                center_offset,
                interp_id,
                stride,
                num_points,
                wp_device,
                wp_stream,
            )

    # Cast outputs back to the input grid dtype for API consistency.
    if input_dtype != torch.float32:
        output = output.to(input_dtype)
    return output


# Register fake tensor propagation for torch compile/fake mode.
@interpolation_impl.register_fake
def _(
    query_points: torch.Tensor,
    context_grid: torch.Tensor,
    grid_meta: torch.Tensor,
    interp_id: int,
    mem_speed_trade: bool = True,
) -> torch.Tensor:
    return torch.empty(
        query_points.shape[0],
        context_grid.shape[0],
        device=query_points.device,
        dtype=context_grid.dtype,
    )


# Setup tensors and metadata required for custom-op backward.
def setup_interpolation_context(
    ctx: torch.autograd.function.FunctionCtx, inputs: tuple, output: torch.Tensor
) -> None:
    query_points, context_grid, grid_meta, interp_id, mem_speed_trade = inputs
    ctx.save_for_backward(query_points, context_grid, grid_meta)
    ctx.interp_id = int(interp_id)
    ctx.mem_speed_trade = bool(mem_speed_trade)


# Backward is computed with the torch implementation to keep gradient parity.
def backward_interpolation(
    ctx: torch.autograd.function.FunctionCtx,
    grad_output: torch.Tensor,
) -> tuple[torch.Tensor | None, torch.Tensor | None, None, None, None]:
    from ._torch_impl import interpolation_torch

    query_points, context_grid, grid_meta = ctx.saved_tensors
    if grad_output is None:
        return None, None, None, None, None

    # Rebuild grid metadata and interpolation mode for the torch reference call.
    grid = [(float(g[0]), float(g[1]), int(g[2])) for g in grid_meta.to("cpu").tolist()]
    interpolation_type = _INTERP_ID_TO_NAME[ctx.interp_id]

    # Re-run forward with autograd-enabled tensors, then differentiate.
    query_points_ref = query_points.detach().requires_grad_(ctx.needs_input_grad[0])
    context_grid_ref = context_grid.detach().requires_grad_(ctx.needs_input_grad[1])
    with torch.enable_grad():
        output_ref = interpolation_torch(
            query_points_ref,
            context_grid_ref,
            grid,
            interpolation_type=interpolation_type,
            mem_speed_trade=ctx.mem_speed_trade,
        )
    grad_query, grad_grid = torch.autograd.grad(
        output_ref,
        (query_points_ref, context_grid_ref),
        grad_outputs=grad_output,
        allow_unused=True,
    )
    return grad_query, grad_grid, None, None, None


# Register custom-op backward.
interpolation_impl.register_autograd(
    backward_interpolation, setup_context=setup_interpolation_context
)


# Public warp entry point used by the interpolation FunctionSpec.
def interpolation_warp(
    query_points: torch.Tensor,
    context_grid: torch.Tensor,
    grid: List[Tuple[float, float, int]],
    interpolation_type: str = "smooth_step_2",
    mem_speed_trade: bool = True,
) -> torch.Tensor:
    interp_id = _INTERP_NAME_TO_ID.get(interpolation_type)
    if interp_id is None:
        raise ValueError(
            "interpolation_type must be one of "
            f"{list(_INTERP_NAME_TO_ID)}, got {interpolation_type}"
        )

    grid_meta = torch.tensor(grid, dtype=torch.float32, device="cpu")

    return interpolation_impl(
        query_points,
        context_grid,
        grid_meta,
        int(interp_id),
        mem_speed_trade,
    )
