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

"""Consolidated Warp kernels for point-to-grid interpolation."""

import warp as wp

from physicsnemo.nn.functional.interpolation._warp_common import (
    basis_derivative,
    basis_value,
    clamp_index,
    clamp_stencil_pair,
)


@wp.kernel
def point_to_grid_backward_1d_stride1(
    points: wp.array(dtype=wp.float32),
    point_values: wp.array2d(dtype=wp.float32),
    grad_grid_output: wp.array2d(dtype=wp.float32),
    grad_query: wp.array2d(dtype=wp.float32),
    grad_point_values: wp.array2d(dtype=wp.float32),
    origin: wp.float32,
    dx: wp.float32,
    size_x: int,
    center_offset: wp.float32,
    compute_query_grad: int,
    compute_values_grad: int,
):
    """Backpropagate 1D nearest point-to-grid interpolation."""
    tid = wp.tid()

    # Map one Warp thread to one query/scatter sample.
    x = points[tid]

    # Convert world-space coordinates into grid-space coordinates.
    center = wp.int32((x - origin) / dx + center_offset)

    # Clamp stencil indices so boundary samples stay in bounds.
    center = clamp_index(center, size_x)

    # Accumulate gradient contributions for per-point input values.
    if compute_values_grad != 0:
        # Accumulate channel contributions for this sample.
        for c in range(point_values.shape[1]):
            grad_point_values[tid, c] = grad_grid_output[c, center]

    # Accumulate gradient contributions for query-point coordinates.
    if compute_query_grad != 0:
        grad_query[tid, 0] = 0.0


@wp.kernel
def point_to_grid_backward_1d_stride2(
    points: wp.array(dtype=wp.float32),
    point_values: wp.array2d(dtype=wp.float32),
    grad_grid_output: wp.array2d(dtype=wp.float32),
    grad_query: wp.array2d(dtype=wp.float32),
    grad_point_values: wp.array2d(dtype=wp.float32),
    origin: wp.float32,
    dx: wp.float32,
    size_x: int,
    interp_id: int,
    compute_query_grad: int,
    compute_values_grad: int,
):
    """Backpropagate 1D linear/smooth point-to-grid interpolation."""
    tid = wp.tid()

    # Map one Warp thread to one query/scatter sample.

    # Convert world-space coordinates into grid-space coordinates.
    pos = (points[tid] - origin) / dx
    center = wp.int32(pos)
    frac = pos - wp.float32(center)
    lower = basis_value(interp_id, frac)
    upper = basis_value(interp_id, 1.0 - frac)
    d_lower = basis_derivative(interp_id, frac) / dx
    d_upper = -basis_derivative(interp_id, 1.0 - frac) / dx

    # Clamp stencil indices so boundary samples stay in bounds.
    idx = clamp_stencil_pair(center, size_x)
    idx0 = idx[0]
    idx1 = idx[1]

    grad_x = wp.float32(0.0)

    # Accumulate channel contributions for this sample.
    for c in range(point_values.shape[1]):
        g0 = grad_grid_output[c, idx0]
        g1 = grad_grid_output[c, idx1]

        # Accumulate gradient contributions for per-point input values.
        if compute_values_grad != 0:
            grad_point_values[tid, c] = upper * g0 + lower * g1

        # Accumulate gradient contributions for query-point coordinates.
        if compute_query_grad != 0:
            value = point_values[tid, c]
            grad_x += value * (g0 * d_upper + g1 * d_lower)

    if compute_query_grad != 0:
        grad_query[tid, 0] = grad_x


@wp.kernel
def point_to_grid_backward_1d_stride5(
    points: wp.array(dtype=wp.float32),
    point_values: wp.array2d(dtype=wp.float32),
    grad_grid_output: wp.array2d(dtype=wp.float32),
    grad_query: wp.array2d(dtype=wp.float32),
    grad_point_values: wp.array2d(dtype=wp.float32),
    origin: wp.float32,
    dx: wp.float32,
    size_x: int,
    center_offset: wp.float32,
    compute_query_grad: int,
    compute_values_grad: int,
):
    """Backpropagate 1D Gaussian point-to-grid interpolation."""
    tid = wp.tid()

    # Map one Warp thread to one query/scatter sample.
    x = points[tid]

    # Convert world-space coordinates into grid-space coordinates.
    pos = (x - origin) / dx
    center = wp.int32(pos + center_offset)
    sigma = dx / 2.0

    sum_w = wp.float32(0.0)
    for ox in range(-2, 3):
        idx = clamp_index(center + ox, size_x)
        coord = origin + wp.float32(idx) * dx
        dist = (x - coord) / sigma
        sum_w += wp.exp(-0.5 * dist * dist)

    if sum_w <= 0.0:
        # Accumulate gradient contributions for query-point coordinates.
        if compute_query_grad != 0:
            grad_query[tid, 0] = 0.0

        # Accumulate gradient contributions for per-point input values.
        if compute_values_grad != 0:
            # Accumulate channel contributions for this sample.
            for c in range(point_values.shape[1]):
                grad_point_values[tid, c] = 0.0
        return
    inv_sum_w = 1.0 / sum_w

    grad_x = wp.float32(0.0)
    for c in range(point_values.shape[1]):
        y = wp.float32(0.0)
        for ox in range(-2, 3):
            idx = clamp_index(center + ox, size_x)
            coord = origin + wp.float32(idx) * dx
            dist = (x - coord) / sigma
            w = wp.exp(-0.5 * dist * dist)
            y += w * grad_grid_output[c, idx]
        y = y * inv_sum_w

        if compute_values_grad != 0:
            grad_point_values[tid, c] = y

        if compute_query_grad != 0:
            value = point_values[tid, c]
            for ox in range(-2, 3):
                idx = clamp_index(center + ox, size_x)
                coord = origin + wp.float32(idx) * dx
                dist = (x - coord) / sigma
                w = wp.exp(-0.5 * dist * dist)
                dwdx = -w * dist / sigma
                grad_x += value * ((dwdx * inv_sum_w) * (grad_grid_output[c, idx] - y))

    if compute_query_grad != 0:
        grad_query[tid, 0] = grad_x


@wp.kernel
def point_to_grid_backward_2d_stride1(
    points: wp.array(dtype=wp.vec2f),
    point_values: wp.array2d(dtype=wp.float32),
    grad_grid_output: wp.array3d(dtype=wp.float32),
    grad_query: wp.array2d(dtype=wp.float32),
    grad_point_values: wp.array2d(dtype=wp.float32),
    origin: wp.vec2f,
    dx: wp.vec2f,
    size: wp.vec2i,
    center_offset: wp.float32,
    compute_query_grad: int,
    compute_values_grad: int,
):
    """Backpropagate 2D nearest point-to-grid interpolation."""
    tid = wp.tid()

    # Map one Warp thread to one query/scatter sample.
    p = points[tid]
    center_x = wp.int32((p[0] - origin[0]) / dx[0] + center_offset)
    center_y = wp.int32((p[1] - origin[1]) / dx[1] + center_offset)
    center_x = clamp_index(center_x, size[0])
    center_y = clamp_index(center_y, size[1])

    # Accumulate gradient contributions for per-point input values.
    if compute_values_grad != 0:
        # Accumulate channel contributions for this sample.
        for c in range(point_values.shape[1]):
            grad_point_values[tid, c] = grad_grid_output[c, center_x, center_y]

    # Accumulate gradient contributions for query-point coordinates.
    if compute_query_grad != 0:
        grad_query[tid, 0] = 0.0
        grad_query[tid, 1] = 0.0


@wp.kernel
def point_to_grid_backward_2d_stride2(
    points: wp.array(dtype=wp.vec2f),
    point_values: wp.array2d(dtype=wp.float32),
    grad_grid_output: wp.array3d(dtype=wp.float32),
    grad_query: wp.array2d(dtype=wp.float32),
    grad_point_values: wp.array2d(dtype=wp.float32),
    origin: wp.vec2f,
    dx: wp.vec2f,
    size: wp.vec2i,
    interp_id: int,
    compute_query_grad: int,
    compute_values_grad: int,
):
    """Backpropagate 2D bilinear/smooth point-to-grid interpolation."""
    tid = wp.tid()

    # Map one Warp thread to one query/scatter sample.
    p = points[tid]

    # Convert world-space coordinates into grid-space coordinates.
    pos_x = (p[0] - origin[0]) / dx[0]
    pos_y = (p[1] - origin[1]) / dx[1]
    center_x = wp.int32(pos_x)
    center_y = wp.int32(pos_y)
    frac_x = pos_x - wp.float32(center_x)
    frac_y = pos_y - wp.float32(center_y)

    lower_x = basis_value(interp_id, frac_x)
    upper_x = basis_value(interp_id, 1.0 - frac_x)
    lower_y = basis_value(interp_id, frac_y)
    upper_y = basis_value(interp_id, 1.0 - frac_y)
    d_lower_x = basis_derivative(interp_id, frac_x) / dx[0]
    d_upper_x = -basis_derivative(interp_id, 1.0 - frac_x) / dx[0]
    d_lower_y = basis_derivative(interp_id, frac_y) / dx[1]
    d_upper_y = -basis_derivative(interp_id, 1.0 - frac_y) / dx[1]

    # Clamp stencil indices so boundary samples stay in bounds.
    idx_x = clamp_stencil_pair(center_x, size[0])
    idx_y = clamp_stencil_pair(center_y, size[1])
    idx_x0 = idx_x[0]
    idx_x1 = idx_x[1]
    idx_y0 = idx_y[0]
    idx_y1 = idx_y[1]

    w00 = upper_x * upper_y
    w01 = upper_x * lower_y
    w10 = lower_x * upper_y
    w11 = lower_x * lower_y
    dw00_dx = d_upper_x * upper_y
    dw01_dx = d_upper_x * lower_y
    dw10_dx = d_lower_x * upper_y
    dw11_dx = d_lower_x * lower_y
    dw00_dy = upper_x * d_upper_y
    dw01_dy = upper_x * d_lower_y
    dw10_dy = lower_x * d_upper_y
    dw11_dy = lower_x * d_lower_y

    grad_x = wp.float32(0.0)
    grad_y = wp.float32(0.0)

    # Accumulate channel contributions for this sample.
    for c in range(point_values.shape[1]):
        g00 = grad_grid_output[c, idx_x0, idx_y0]
        g01 = grad_grid_output[c, idx_x0, idx_y1]
        g10 = grad_grid_output[c, idx_x1, idx_y0]
        g11 = grad_grid_output[c, idx_x1, idx_y1]

        # Accumulate gradient contributions for per-point input values.
        if compute_values_grad != 0:
            grad_point_values[tid, c] = w00 * g00 + w01 * g01 + w10 * g10 + w11 * g11

        # Accumulate gradient contributions for query-point coordinates.
        if compute_query_grad != 0:
            value = point_values[tid, c]
            grad_x += value * (
                g00 * dw00_dx + g01 * dw01_dx + g10 * dw10_dx + g11 * dw11_dx
            )
            grad_y += value * (
                g00 * dw00_dy + g01 * dw01_dy + g10 * dw10_dy + g11 * dw11_dy
            )

    if compute_query_grad != 0:
        grad_query[tid, 0] = grad_x
        grad_query[tid, 1] = grad_y


@wp.kernel
def point_to_grid_backward_2d_stride5(
    points: wp.array(dtype=wp.vec2f),
    point_values: wp.array2d(dtype=wp.float32),
    grad_grid_output: wp.array3d(dtype=wp.float32),
    grad_query: wp.array2d(dtype=wp.float32),
    grad_point_values: wp.array2d(dtype=wp.float32),
    origin: wp.vec2f,
    dx: wp.vec2f,
    size: wp.vec2i,
    center_offset: wp.float32,
    compute_query_grad: int,
    compute_values_grad: int,
):
    """Backpropagate 2D Gaussian point-to-grid interpolation."""
    tid = wp.tid()

    # Map one Warp thread to one query/scatter sample.
    p = points[tid]

    # Convert world-space coordinates into grid-space coordinates.
    pos_x = (p[0] - origin[0]) / dx[0]
    pos_y = (p[1] - origin[1]) / dx[1]
    center_x = wp.int32(pos_x + center_offset)
    center_y = wp.int32(pos_y + center_offset)
    sigma_x = dx[0] / 2.0
    sigma_y = dx[1] / 2.0

    sum_w = wp.float32(0.0)
    for ox in range(-2, 3):
        idx_x = clamp_index(center_x + ox, size[0])
        coord_x = origin[0] + wp.float32(idx_x) * dx[0]
        dist_x = (p[0] - coord_x) / sigma_x
        gx = wp.exp(-0.5 * dist_x * dist_x)
        for oy in range(-2, 3):
            idx_y = clamp_index(center_y + oy, size[1])
            coord_y = origin[1] + wp.float32(idx_y) * dx[1]
            dist_y = (p[1] - coord_y) / sigma_y
            sum_w += gx * wp.exp(-0.5 * dist_y * dist_y)

    if sum_w <= 0.0:
        # Accumulate gradient contributions for query-point coordinates.
        if compute_query_grad != 0:
            grad_query[tid, 0] = 0.0
            grad_query[tid, 1] = 0.0

        # Accumulate gradient contributions for per-point input values.
        if compute_values_grad != 0:
            # Accumulate channel contributions for this sample.
            for c in range(point_values.shape[1]):
                grad_point_values[tid, c] = 0.0
        return
    inv_sum_w = 1.0 / sum_w

    grad_x = wp.float32(0.0)
    grad_y = wp.float32(0.0)
    for c in range(point_values.shape[1]):
        y = wp.float32(0.0)
        for ox in range(-2, 3):
            idx_x = clamp_index(center_x + ox, size[0])
            coord_x = origin[0] + wp.float32(idx_x) * dx[0]
            dist_x = (p[0] - coord_x) / sigma_x
            gx = wp.exp(-0.5 * dist_x * dist_x)
            for oy in range(-2, 3):
                idx_y = clamp_index(center_y + oy, size[1])
                coord_y = origin[1] + wp.float32(idx_y) * dx[1]
                dist_y = (p[1] - coord_y) / sigma_y
                w = gx * wp.exp(-0.5 * dist_y * dist_y)
                y += w * grad_grid_output[c, idx_x, idx_y]
        y = y * inv_sum_w

        if compute_values_grad != 0:
            grad_point_values[tid, c] = y

        if compute_query_grad != 0:
            value = point_values[tid, c]
            for ox in range(-2, 3):
                idx_x = clamp_index(center_x + ox, size[0])
                coord_x = origin[0] + wp.float32(idx_x) * dx[0]
                dist_x = (p[0] - coord_x) / sigma_x
                gx = wp.exp(-0.5 * dist_x * dist_x)
                for oy in range(-2, 3):
                    idx_y = clamp_index(center_y + oy, size[1])
                    coord_y = origin[1] + wp.float32(idx_y) * dx[1]
                    dist_y = (p[1] - coord_y) / sigma_y
                    gy = wp.exp(-0.5 * dist_y * dist_y)
                    w = gx * gy
                    dwdx = -w * dist_x / sigma_x
                    dwdy = -w * dist_y / sigma_y
                    grid_value = grad_grid_output[c, idx_x, idx_y]
                    grad_x += value * ((dwdx * inv_sum_w) * (grid_value - y))
                    grad_y += value * ((dwdy * inv_sum_w) * (grid_value - y))

    if compute_query_grad != 0:
        grad_query[tid, 0] = grad_x
        grad_query[tid, 1] = grad_y


@wp.kernel
def point_to_grid_backward_3d_stride1(
    points: wp.array(dtype=wp.vec3f),
    point_values: wp.array2d(dtype=wp.float32),
    grad_grid_output: wp.array4d(dtype=wp.float32),
    grad_query: wp.array2d(dtype=wp.float32),
    grad_point_values: wp.array2d(dtype=wp.float32),
    origin: wp.vec3f,
    dx: wp.vec3f,
    size: wp.vec3i,
    center_offset: wp.float32,
    compute_query_grad: int,
    compute_values_grad: int,
):
    """Backpropagate 3D nearest point-to-grid interpolation."""
    tid = wp.tid()

    # Map one Warp thread to one query/scatter sample.
    p = points[tid]
    center_x = wp.int32((p[0] - origin[0]) / dx[0] + center_offset)
    center_y = wp.int32((p[1] - origin[1]) / dx[1] + center_offset)
    center_z = wp.int32((p[2] - origin[2]) / dx[2] + center_offset)
    center_x = clamp_index(center_x, size[0])
    center_y = clamp_index(center_y, size[1])
    center_z = clamp_index(center_z, size[2])

    # Accumulate gradient contributions for per-point input values.
    if compute_values_grad != 0:
        # Accumulate channel contributions for this sample.
        for c in range(point_values.shape[1]):
            grad_point_values[tid, c] = grad_grid_output[
                c, center_x, center_y, center_z
            ]

    # Accumulate gradient contributions for query-point coordinates.
    if compute_query_grad != 0:
        grad_query[tid, 0] = 0.0
        grad_query[tid, 1] = 0.0
        grad_query[tid, 2] = 0.0


@wp.kernel
def point_to_grid_backward_3d_stride2(
    points: wp.array(dtype=wp.vec3f),
    point_values: wp.array2d(dtype=wp.float32),
    grad_grid_output: wp.array4d(dtype=wp.float32),
    grad_query: wp.array2d(dtype=wp.float32),
    grad_point_values: wp.array2d(dtype=wp.float32),
    origin: wp.vec3f,
    dx: wp.vec3f,
    size: wp.vec3i,
    interp_id: int,
    compute_query_grad: int,
    compute_values_grad: int,
):
    """Backpropagate 3D trilinear/smooth point-to-grid interpolation."""
    tid = wp.tid()

    # Map one Warp thread to one query/scatter sample.
    p = points[tid]

    # Convert world-space coordinates into grid-space coordinates.
    pos_x = (p[0] - origin[0]) / dx[0]
    pos_y = (p[1] - origin[1]) / dx[1]
    pos_z = (p[2] - origin[2]) / dx[2]
    center_x = wp.int32(pos_x)
    center_y = wp.int32(pos_y)
    center_z = wp.int32(pos_z)
    frac_x = pos_x - wp.float32(center_x)
    frac_y = pos_y - wp.float32(center_y)
    frac_z = pos_z - wp.float32(center_z)

    lower_x = basis_value(interp_id, frac_x)
    upper_x = basis_value(interp_id, 1.0 - frac_x)
    lower_y = basis_value(interp_id, frac_y)
    upper_y = basis_value(interp_id, 1.0 - frac_y)
    lower_z = basis_value(interp_id, frac_z)
    upper_z = basis_value(interp_id, 1.0 - frac_z)
    d_lower_x = basis_derivative(interp_id, frac_x) / dx[0]
    d_upper_x = -basis_derivative(interp_id, 1.0 - frac_x) / dx[0]
    d_lower_y = basis_derivative(interp_id, frac_y) / dx[1]
    d_upper_y = -basis_derivative(interp_id, 1.0 - frac_y) / dx[1]
    d_lower_z = basis_derivative(interp_id, frac_z) / dx[2]
    d_upper_z = -basis_derivative(interp_id, 1.0 - frac_z) / dx[2]

    # Clamp stencil indices so boundary samples stay in bounds.
    idx_x = clamp_stencil_pair(center_x, size[0])
    idx_y = clamp_stencil_pair(center_y, size[1])
    idx_z = clamp_stencil_pair(center_z, size[2])
    idx_x0 = idx_x[0]
    idx_x1 = idx_x[1]
    idx_y0 = idx_y[0]
    idx_y1 = idx_y[1]
    idx_z0 = idx_z[0]
    idx_z1 = idx_z[1]

    w000 = upper_x * upper_y * upper_z
    w001 = upper_x * upper_y * lower_z
    w010 = upper_x * lower_y * upper_z
    w011 = upper_x * lower_y * lower_z
    w100 = lower_x * upper_y * upper_z
    w101 = lower_x * upper_y * lower_z
    w110 = lower_x * lower_y * upper_z
    w111 = lower_x * lower_y * lower_z
    dw000_dx = d_upper_x * upper_y * upper_z
    dw001_dx = d_upper_x * upper_y * lower_z
    dw010_dx = d_upper_x * lower_y * upper_z
    dw011_dx = d_upper_x * lower_y * lower_z
    dw100_dx = d_lower_x * upper_y * upper_z
    dw101_dx = d_lower_x * upper_y * lower_z
    dw110_dx = d_lower_x * lower_y * upper_z
    dw111_dx = d_lower_x * lower_y * lower_z
    dw000_dy = upper_x * d_upper_y * upper_z
    dw001_dy = upper_x * d_upper_y * lower_z
    dw010_dy = upper_x * d_lower_y * upper_z
    dw011_dy = upper_x * d_lower_y * lower_z
    dw100_dy = lower_x * d_upper_y * upper_z
    dw101_dy = lower_x * d_upper_y * lower_z
    dw110_dy = lower_x * d_lower_y * upper_z
    dw111_dy = lower_x * d_lower_y * lower_z
    dw000_dz = upper_x * upper_y * d_upper_z
    dw001_dz = upper_x * upper_y * d_lower_z
    dw010_dz = upper_x * lower_y * d_upper_z
    dw011_dz = upper_x * lower_y * d_lower_z
    dw100_dz = lower_x * upper_y * d_upper_z
    dw101_dz = lower_x * upper_y * d_lower_z
    dw110_dz = lower_x * lower_y * d_upper_z
    dw111_dz = lower_x * lower_y * d_lower_z

    grad_x = wp.float32(0.0)
    grad_y = wp.float32(0.0)
    grad_z = wp.float32(0.0)

    # Accumulate channel contributions for this sample.
    for c in range(point_values.shape[1]):
        g000 = grad_grid_output[c, idx_x0, idx_y0, idx_z0]
        g001 = grad_grid_output[c, idx_x0, idx_y0, idx_z1]
        g010 = grad_grid_output[c, idx_x0, idx_y1, idx_z0]
        g011 = grad_grid_output[c, idx_x0, idx_y1, idx_z1]
        g100 = grad_grid_output[c, idx_x1, idx_y0, idx_z0]
        g101 = grad_grid_output[c, idx_x1, idx_y0, idx_z1]
        g110 = grad_grid_output[c, idx_x1, idx_y1, idx_z0]
        g111 = grad_grid_output[c, idx_x1, idx_y1, idx_z1]

        # Accumulate gradient contributions for per-point input values.
        if compute_values_grad != 0:
            grad_point_values[tid, c] = (
                w000 * g000
                + w001 * g001
                + w010 * g010
                + w011 * g011
                + w100 * g100
                + w101 * g101
                + w110 * g110
                + w111 * g111
            )

        # Accumulate gradient contributions for query-point coordinates.
        if compute_query_grad != 0:
            value = point_values[tid, c]
            grad_x += value * (
                g000 * dw000_dx
                + g001 * dw001_dx
                + g010 * dw010_dx
                + g011 * dw011_dx
                + g100 * dw100_dx
                + g101 * dw101_dx
                + g110 * dw110_dx
                + g111 * dw111_dx
            )
            grad_y += value * (
                g000 * dw000_dy
                + g001 * dw001_dy
                + g010 * dw010_dy
                + g011 * dw011_dy
                + g100 * dw100_dy
                + g101 * dw101_dy
                + g110 * dw110_dy
                + g111 * dw111_dy
            )
            grad_z += value * (
                g000 * dw000_dz
                + g001 * dw001_dz
                + g010 * dw010_dz
                + g011 * dw011_dz
                + g100 * dw100_dz
                + g101 * dw101_dz
                + g110 * dw110_dz
                + g111 * dw111_dz
            )

    if compute_query_grad != 0:
        grad_query[tid, 0] = grad_x
        grad_query[tid, 1] = grad_y
        grad_query[tid, 2] = grad_z


@wp.kernel
def point_to_grid_backward_3d_stride5(
    points: wp.array(dtype=wp.vec3f),
    point_values: wp.array2d(dtype=wp.float32),
    grad_grid_output: wp.array4d(dtype=wp.float32),
    grad_query: wp.array2d(dtype=wp.float32),
    grad_point_values: wp.array2d(dtype=wp.float32),
    origin: wp.vec3f,
    dx: wp.vec3f,
    size: wp.vec3i,
    center_offset: wp.float32,
    compute_query_grad: int,
    compute_values_grad: int,
):
    """Backpropagate 3D Gaussian point-to-grid interpolation."""
    tid = wp.tid()

    # Map one Warp thread to one query/scatter sample.
    p = points[tid]

    # Convert world-space coordinates into grid-space coordinates.
    pos_x = (p[0] - origin[0]) / dx[0]
    pos_y = (p[1] - origin[1]) / dx[1]
    pos_z = (p[2] - origin[2]) / dx[2]
    center_x = wp.int32(pos_x + center_offset)
    center_y = wp.int32(pos_y + center_offset)
    center_z = wp.int32(pos_z + center_offset)
    sigma_x = dx[0] / 2.0
    sigma_y = dx[1] / 2.0
    sigma_z = dx[2] / 2.0

    sum_w = wp.float32(0.0)
    for ox in range(-2, 3):
        idx_x = clamp_index(center_x + ox, size[0])
        coord_x = origin[0] + wp.float32(idx_x) * dx[0]
        dist_x = (p[0] - coord_x) / sigma_x
        gx = wp.exp(-0.5 * dist_x * dist_x)
        for oy in range(-2, 3):
            idx_y = clamp_index(center_y + oy, size[1])
            coord_y = origin[1] + wp.float32(idx_y) * dx[1]
            dist_y = (p[1] - coord_y) / sigma_y
            gy = wp.exp(-0.5 * dist_y * dist_y)
            for oz in range(-2, 3):
                idx_z = clamp_index(center_z + oz, size[2])
                coord_z = origin[2] + wp.float32(idx_z) * dx[2]
                dist_z = (p[2] - coord_z) / sigma_z
                sum_w += gx * gy * wp.exp(-0.5 * dist_z * dist_z)

    if sum_w <= 0.0:
        # Accumulate gradient contributions for query-point coordinates.
        if compute_query_grad != 0:
            grad_query[tid, 0] = 0.0
            grad_query[tid, 1] = 0.0
            grad_query[tid, 2] = 0.0

        # Accumulate gradient contributions for per-point input values.
        if compute_values_grad != 0:
            # Accumulate channel contributions for this sample.
            for c in range(point_values.shape[1]):
                grad_point_values[tid, c] = 0.0
        return
    inv_sum_w = 1.0 / sum_w

    grad_x = wp.float32(0.0)
    grad_y = wp.float32(0.0)
    grad_z = wp.float32(0.0)
    for c in range(point_values.shape[1]):
        y = wp.float32(0.0)
        for ox in range(-2, 3):
            idx_x = clamp_index(center_x + ox, size[0])
            coord_x = origin[0] + wp.float32(idx_x) * dx[0]
            dist_x = (p[0] - coord_x) / sigma_x
            gx = wp.exp(-0.5 * dist_x * dist_x)
            for oy in range(-2, 3):
                idx_y = clamp_index(center_y + oy, size[1])
                coord_y = origin[1] + wp.float32(idx_y) * dx[1]
                dist_y = (p[1] - coord_y) / sigma_y
                gy = wp.exp(-0.5 * dist_y * dist_y)
                for oz in range(-2, 3):
                    idx_z = clamp_index(center_z + oz, size[2])
                    coord_z = origin[2] + wp.float32(idx_z) * dx[2]
                    dist_z = (p[2] - coord_z) / sigma_z
                    w = gx * gy * wp.exp(-0.5 * dist_z * dist_z)
                    y += w * grad_grid_output[c, idx_x, idx_y, idx_z]
        y = y * inv_sum_w

        if compute_values_grad != 0:
            grad_point_values[tid, c] = y

        if compute_query_grad != 0:
            value = point_values[tid, c]
            for ox in range(-2, 3):
                idx_x = clamp_index(center_x + ox, size[0])
                coord_x = origin[0] + wp.float32(idx_x) * dx[0]
                dist_x = (p[0] - coord_x) / sigma_x
                gx = wp.exp(-0.5 * dist_x * dist_x)
                for oy in range(-2, 3):
                    idx_y = clamp_index(center_y + oy, size[1])
                    coord_y = origin[1] + wp.float32(idx_y) * dx[1]
                    dist_y = (p[1] - coord_y) / sigma_y
                    gy = wp.exp(-0.5 * dist_y * dist_y)
                    for oz in range(-2, 3):
                        idx_z = clamp_index(center_z + oz, size[2])
                        coord_z = origin[2] + wp.float32(idx_z) * dx[2]
                        dist_z = (p[2] - coord_z) / sigma_z
                        gz = wp.exp(-0.5 * dist_z * dist_z)
                        w = gx * gy * gz
                        dwdx = -w * dist_x / sigma_x
                        dwdy = -w * dist_y / sigma_y
                        dwdz = -w * dist_z / sigma_z
                        grid_value = grad_grid_output[c, idx_x, idx_y, idx_z]
                        grad_x += value * ((dwdx * inv_sum_w) * (grid_value - y))
                        grad_y += value * ((dwdy * inv_sum_w) * (grid_value - y))
                        grad_z += value * ((dwdz * inv_sum_w) * (grid_value - y))

    if compute_query_grad != 0:
        grad_query[tid, 0] = grad_x
        grad_query[tid, 1] = grad_y
        grad_query[tid, 2] = grad_z


@wp.kernel
def point_to_grid_forward_1d_stride1(
    points: wp.array(dtype=wp.float32),
    point_values: wp.array2d(dtype=wp.float32),
    out_grid: wp.array2d(dtype=wp.float32),
    origin: wp.float32,
    dx: wp.float32,
    size_x: int,
    center_offset: wp.float32,
):
    """Scatter points to a 1D grid with nearest-neighbor weights."""
    tid = wp.tid()

    # Map one Warp thread to one query/scatter sample.
    x = points[tid]

    # Convert world-space coordinates into grid-space coordinates.
    center = wp.int32((x - origin) / dx + center_offset)

    # Clamp stencil indices so boundary samples stay in bounds.
    center = clamp_index(center, size_x)

    # Accumulate channel contributions for this sample.
    for c in range(point_values.shape[1]):
        wp.atomic_add(out_grid, c, center, point_values[tid, c])


@wp.kernel
def point_to_grid_forward_1d_stride2(
    points: wp.array(dtype=wp.float32),
    point_values: wp.array2d(dtype=wp.float32),
    out_grid: wp.array2d(dtype=wp.float32),
    origin: wp.float32,
    dx: wp.float32,
    size_x: int,
    interp_id: int,
):
    """Scatter points to a 1D grid with linear/smooth weights."""
    tid = wp.tid()

    # Map one Warp thread to one query/scatter sample.
    x = points[tid]

    # Convert world-space coordinates into grid-space coordinates.
    pos = (x - origin) / dx
    center = wp.int32(pos)
    frac = pos - wp.float32(center)
    lower = basis_value(interp_id, frac)
    upper = basis_value(interp_id, 1.0 - frac)

    # Clamp stencil indices so boundary samples stay in bounds.
    idx = clamp_stencil_pair(center, size_x)
    idx0 = idx[0]
    idx1 = idx[1]

    # Accumulate channel contributions for this sample.
    for c in range(point_values.shape[1]):
        value = point_values[tid, c]
        wp.atomic_add(out_grid, c, idx0, upper * value)
        wp.atomic_add(out_grid, c, idx1, lower * value)


@wp.kernel
def point_to_grid_forward_1d_stride5(
    points: wp.array(dtype=wp.float32),
    point_values: wp.array2d(dtype=wp.float32),
    out_grid: wp.array2d(dtype=wp.float32),
    origin: wp.float32,
    dx: wp.float32,
    size_x: int,
    center_offset: wp.float32,
):
    """Scatter points to a 1D grid with Gaussian stencil weights."""
    tid = wp.tid()

    # Map one Warp thread to one query/scatter sample.
    x = points[tid]

    # Convert world-space coordinates into grid-space coordinates.
    pos = (x - origin) / dx
    center = wp.int32(pos + center_offset)
    sigma = dx / 2.0

    sum_w = wp.float32(0.0)
    for ox in range(-2, 3):
        idx = clamp_index(center + ox, size_x)
        coord = origin + wp.float32(idx) * dx
        dist = (x - coord) / sigma
        sum_w += wp.exp(-0.5 * dist * dist)

    if sum_w <= 0.0:
        return
    inv_sum_w = 1.0 / sum_w

    for ox in range(-2, 3):
        idx = clamp_index(center + ox, size_x)
        coord = origin + wp.float32(idx) * dx
        dist = (x - coord) / sigma
        weight = wp.exp(-0.5 * dist * dist) * inv_sum_w

        # Accumulate channel contributions for this sample.
        for c in range(point_values.shape[1]):
            wp.atomic_add(out_grid, c, idx, weight * point_values[tid, c])


@wp.kernel
def point_to_grid_forward_2d_stride1(
    points: wp.array(dtype=wp.vec2f),
    point_values: wp.array2d(dtype=wp.float32),
    out_grid: wp.array3d(dtype=wp.float32),
    origin: wp.vec2f,
    dx: wp.vec2f,
    size: wp.vec2i,
    center_offset: wp.float32,
):
    """Scatter points to a 2D grid with nearest-neighbor weights."""
    tid = wp.tid()

    # Map one Warp thread to one query/scatter sample.
    p = points[tid]
    center_x = wp.int32((p[0] - origin[0]) / dx[0] + center_offset)
    center_y = wp.int32((p[1] - origin[1]) / dx[1] + center_offset)
    center_x = clamp_index(center_x, size[0])
    center_y = clamp_index(center_y, size[1])

    # Accumulate channel contributions for this sample.
    for c in range(point_values.shape[1]):
        wp.atomic_add(out_grid, c, center_x, center_y, point_values[tid, c])


@wp.kernel
def point_to_grid_forward_2d_stride2(
    points: wp.array(dtype=wp.vec2f),
    point_values: wp.array2d(dtype=wp.float32),
    out_grid: wp.array3d(dtype=wp.float32),
    origin: wp.vec2f,
    dx: wp.vec2f,
    size: wp.vec2i,
    interp_id: int,
):
    """Scatter points to a 2D grid with bilinear/smooth weights."""
    tid = wp.tid()

    # Map one Warp thread to one query/scatter sample.
    p = points[tid]

    # Convert world-space coordinates into grid-space coordinates.
    pos_x = (p[0] - origin[0]) / dx[0]
    pos_y = (p[1] - origin[1]) / dx[1]
    center_x = wp.int32(pos_x)
    center_y = wp.int32(pos_y)
    frac_x = pos_x - wp.float32(center_x)
    frac_y = pos_y - wp.float32(center_y)

    lower_x = basis_value(interp_id, frac_x)
    upper_x = basis_value(interp_id, 1.0 - frac_x)
    lower_y = basis_value(interp_id, frac_y)
    upper_y = basis_value(interp_id, 1.0 - frac_y)

    # Clamp stencil indices so boundary samples stay in bounds.
    idx_x = clamp_stencil_pair(center_x, size[0])
    idx_y = clamp_stencil_pair(center_y, size[1])
    idx_x0 = idx_x[0]
    idx_x1 = idx_x[1]
    idx_y0 = idx_y[0]
    idx_y1 = idx_y[1]

    w00 = upper_x * upper_y
    w01 = upper_x * lower_y
    w10 = lower_x * upper_y
    w11 = lower_x * lower_y

    # Accumulate channel contributions for this sample.
    for c in range(point_values.shape[1]):
        value = point_values[tid, c]
        wp.atomic_add(out_grid, c, idx_x0, idx_y0, w00 * value)
        wp.atomic_add(out_grid, c, idx_x0, idx_y1, w01 * value)
        wp.atomic_add(out_grid, c, idx_x1, idx_y0, w10 * value)
        wp.atomic_add(out_grid, c, idx_x1, idx_y1, w11 * value)


@wp.kernel
def point_to_grid_forward_2d_stride5(
    points: wp.array(dtype=wp.vec2f),
    point_values: wp.array2d(dtype=wp.float32),
    out_grid: wp.array3d(dtype=wp.float32),
    origin: wp.vec2f,
    dx: wp.vec2f,
    size: wp.vec2i,
    center_offset: wp.float32,
):
    """Scatter points to a 2D grid with Gaussian stencil weights."""
    tid = wp.tid()

    # Map one Warp thread to one query/scatter sample.
    p = points[tid]

    # Convert world-space coordinates into grid-space coordinates.
    pos_x = (p[0] - origin[0]) / dx[0]
    pos_y = (p[1] - origin[1]) / dx[1]
    center_x = wp.int32(pos_x + center_offset)
    center_y = wp.int32(pos_y + center_offset)
    sigma_x = dx[0] / 2.0
    sigma_y = dx[1] / 2.0

    sum_w = wp.float32(0.0)
    for ox in range(-2, 3):
        idx_x = clamp_index(center_x + ox, size[0])
        coord_x = origin[0] + wp.float32(idx_x) * dx[0]
        dist_x = (p[0] - coord_x) / sigma_x
        gx = wp.exp(-0.5 * dist_x * dist_x)
        for oy in range(-2, 3):
            idx_y = clamp_index(center_y + oy, size[1])
            coord_y = origin[1] + wp.float32(idx_y) * dx[1]
            dist_y = (p[1] - coord_y) / sigma_y
            sum_w += gx * wp.exp(-0.5 * dist_y * dist_y)

    if sum_w <= 0.0:
        return
    inv_sum_w = 1.0 / sum_w

    for ox in range(-2, 3):
        idx_x = clamp_index(center_x + ox, size[0])
        coord_x = origin[0] + wp.float32(idx_x) * dx[0]
        dist_x = (p[0] - coord_x) / sigma_x
        gx = wp.exp(-0.5 * dist_x * dist_x)
        for oy in range(-2, 3):
            idx_y = clamp_index(center_y + oy, size[1])
            coord_y = origin[1] + wp.float32(idx_y) * dx[1]
            dist_y = (p[1] - coord_y) / sigma_y
            weight = gx * wp.exp(-0.5 * dist_y * dist_y) * inv_sum_w

            # Accumulate channel contributions for this sample.
            for c in range(point_values.shape[1]):
                wp.atomic_add(out_grid, c, idx_x, idx_y, weight * point_values[tid, c])


@wp.kernel
def point_to_grid_forward_3d_stride1(
    points: wp.array(dtype=wp.vec3f),
    point_values: wp.array2d(dtype=wp.float32),
    out_grid: wp.array4d(dtype=wp.float32),
    origin: wp.vec3f,
    dx: wp.vec3f,
    size: wp.vec3i,
    center_offset: wp.float32,
):
    """Scatter points to a 3D grid with nearest-neighbor weights."""
    tid = wp.tid()

    # Map one Warp thread to one query/scatter sample.
    p = points[tid]
    center_x = wp.int32((p[0] - origin[0]) / dx[0] + center_offset)
    center_y = wp.int32((p[1] - origin[1]) / dx[1] + center_offset)
    center_z = wp.int32((p[2] - origin[2]) / dx[2] + center_offset)
    center_x = clamp_index(center_x, size[0])
    center_y = clamp_index(center_y, size[1])
    center_z = clamp_index(center_z, size[2])

    # Accumulate channel contributions for this sample.
    for c in range(point_values.shape[1]):
        wp.atomic_add(out_grid, c, center_x, center_y, center_z, point_values[tid, c])


@wp.kernel
def point_to_grid_forward_3d_stride2(
    points: wp.array(dtype=wp.vec3f),
    point_values: wp.array2d(dtype=wp.float32),
    out_grid: wp.array4d(dtype=wp.float32),
    origin: wp.vec3f,
    dx: wp.vec3f,
    size: wp.vec3i,
    interp_id: int,
):
    """Scatter points to a 3D grid with trilinear/smooth weights."""
    tid = wp.tid()

    # Map one Warp thread to one query/scatter sample.
    p = points[tid]

    # Convert world-space coordinates into grid-space coordinates.
    pos_x = (p[0] - origin[0]) / dx[0]
    pos_y = (p[1] - origin[1]) / dx[1]
    pos_z = (p[2] - origin[2]) / dx[2]
    center_x = wp.int32(pos_x)
    center_y = wp.int32(pos_y)
    center_z = wp.int32(pos_z)
    frac_x = pos_x - wp.float32(center_x)
    frac_y = pos_y - wp.float32(center_y)
    frac_z = pos_z - wp.float32(center_z)

    lower_x = basis_value(interp_id, frac_x)
    upper_x = basis_value(interp_id, 1.0 - frac_x)
    lower_y = basis_value(interp_id, frac_y)
    upper_y = basis_value(interp_id, 1.0 - frac_y)
    lower_z = basis_value(interp_id, frac_z)
    upper_z = basis_value(interp_id, 1.0 - frac_z)

    # Clamp stencil indices so boundary samples stay in bounds.
    idx_x = clamp_stencil_pair(center_x, size[0])
    idx_y = clamp_stencil_pair(center_y, size[1])
    idx_z = clamp_stencil_pair(center_z, size[2])
    idx_x0 = idx_x[0]
    idx_x1 = idx_x[1]
    idx_y0 = idx_y[0]
    idx_y1 = idx_y[1]
    idx_z0 = idx_z[0]
    idx_z1 = idx_z[1]

    w000 = upper_x * upper_y * upper_z
    w001 = upper_x * upper_y * lower_z
    w010 = upper_x * lower_y * upper_z
    w011 = upper_x * lower_y * lower_z
    w100 = lower_x * upper_y * upper_z
    w101 = lower_x * upper_y * lower_z
    w110 = lower_x * lower_y * upper_z
    w111 = lower_x * lower_y * lower_z

    # Accumulate channel contributions for this sample.
    for c in range(point_values.shape[1]):
        value = point_values[tid, c]
        wp.atomic_add(out_grid, c, idx_x0, idx_y0, idx_z0, w000 * value)
        wp.atomic_add(out_grid, c, idx_x0, idx_y0, idx_z1, w001 * value)
        wp.atomic_add(out_grid, c, idx_x0, idx_y1, idx_z0, w010 * value)
        wp.atomic_add(out_grid, c, idx_x0, idx_y1, idx_z1, w011 * value)
        wp.atomic_add(out_grid, c, idx_x1, idx_y0, idx_z0, w100 * value)
        wp.atomic_add(out_grid, c, idx_x1, idx_y0, idx_z1, w101 * value)
        wp.atomic_add(out_grid, c, idx_x1, idx_y1, idx_z0, w110 * value)
        wp.atomic_add(out_grid, c, idx_x1, idx_y1, idx_z1, w111 * value)


@wp.kernel
def point_to_grid_forward_3d_stride5(
    points: wp.array(dtype=wp.vec3f),
    point_values: wp.array2d(dtype=wp.float32),
    out_grid: wp.array4d(dtype=wp.float32),
    origin: wp.vec3f,
    dx: wp.vec3f,
    size: wp.vec3i,
    center_offset: wp.float32,
):
    """Scatter points to a 3D grid with Gaussian stencil weights."""
    tid = wp.tid()

    # Map one Warp thread to one query/scatter sample.
    p = points[tid]

    # Convert world-space coordinates into grid-space coordinates.
    pos_x = (p[0] - origin[0]) / dx[0]
    pos_y = (p[1] - origin[1]) / dx[1]
    pos_z = (p[2] - origin[2]) / dx[2]
    center_x = wp.int32(pos_x + center_offset)
    center_y = wp.int32(pos_y + center_offset)
    center_z = wp.int32(pos_z + center_offset)
    sigma_x = dx[0] / 2.0
    sigma_y = dx[1] / 2.0
    sigma_z = dx[2] / 2.0

    sum_w = wp.float32(0.0)
    for ox in range(-2, 3):
        idx_x = clamp_index(center_x + ox, size[0])
        coord_x = origin[0] + wp.float32(idx_x) * dx[0]
        dist_x = (p[0] - coord_x) / sigma_x
        gx = wp.exp(-0.5 * dist_x * dist_x)
        for oy in range(-2, 3):
            idx_y = clamp_index(center_y + oy, size[1])
            coord_y = origin[1] + wp.float32(idx_y) * dx[1]
            dist_y = (p[1] - coord_y) / sigma_y
            gy = wp.exp(-0.5 * dist_y * dist_y)
            for oz in range(-2, 3):
                idx_z = clamp_index(center_z + oz, size[2])
                coord_z = origin[2] + wp.float32(idx_z) * dx[2]
                dist_z = (p[2] - coord_z) / sigma_z
                sum_w += gx * gy * wp.exp(-0.5 * dist_z * dist_z)

    if sum_w <= 0.0:
        return
    inv_sum_w = 1.0 / sum_w

    for ox in range(-2, 3):
        idx_x = clamp_index(center_x + ox, size[0])
        coord_x = origin[0] + wp.float32(idx_x) * dx[0]
        dist_x = (p[0] - coord_x) / sigma_x
        gx = wp.exp(-0.5 * dist_x * dist_x)
        for oy in range(-2, 3):
            idx_y = clamp_index(center_y + oy, size[1])
            coord_y = origin[1] + wp.float32(idx_y) * dx[1]
            dist_y = (p[1] - coord_y) / sigma_y
            gy = wp.exp(-0.5 * dist_y * dist_y)
            for oz in range(-2, 3):
                idx_z = clamp_index(center_z + oz, size[2])
                coord_z = origin[2] + wp.float32(idx_z) * dx[2]
                dist_z = (p[2] - coord_z) / sigma_z
                weight = gx * gy * wp.exp(-0.5 * dist_z * dist_z) * inv_sum_w

                # Accumulate channel contributions for this sample.
                for c in range(point_values.shape[1]):
                    wp.atomic_add(
                        out_grid,
                        c,
                        idx_x,
                        idx_y,
                        idx_z,
                        weight * point_values[tid, c],
                    )


FORWARD_KERNELS = {
    1: {
        1: point_to_grid_forward_1d_stride1,
        2: point_to_grid_forward_1d_stride2,
        5: point_to_grid_forward_1d_stride5,
    },
    2: {
        1: point_to_grid_forward_2d_stride1,
        2: point_to_grid_forward_2d_stride2,
        5: point_to_grid_forward_2d_stride5,
    },
    3: {
        1: point_to_grid_forward_3d_stride1,
        2: point_to_grid_forward_3d_stride2,
        5: point_to_grid_forward_3d_stride5,
    },
}

BACKWARD_KERNELS = {
    1: {
        1: point_to_grid_backward_1d_stride1,
        2: point_to_grid_backward_1d_stride2,
        5: point_to_grid_backward_1d_stride5,
    },
    2: {
        1: point_to_grid_backward_2d_stride1,
        2: point_to_grid_backward_2d_stride2,
        5: point_to_grid_backward_2d_stride5,
    },
    3: {
        1: point_to_grid_backward_3d_stride1,
        2: point_to_grid_backward_3d_stride2,
        5: point_to_grid_backward_3d_stride5,
    },
}

__all__ = [
    "point_to_grid_backward_1d_stride1",
    "point_to_grid_backward_1d_stride2",
    "point_to_grid_backward_1d_stride5",
    "point_to_grid_backward_2d_stride1",
    "point_to_grid_backward_2d_stride2",
    "point_to_grid_backward_2d_stride5",
    "point_to_grid_backward_3d_stride1",
    "point_to_grid_backward_3d_stride2",
    "point_to_grid_backward_3d_stride5",
    "point_to_grid_forward_1d_stride1",
    "point_to_grid_forward_1d_stride2",
    "point_to_grid_forward_1d_stride5",
    "point_to_grid_forward_2d_stride1",
    "point_to_grid_forward_2d_stride2",
    "point_to_grid_forward_2d_stride5",
    "point_to_grid_forward_3d_stride1",
    "point_to_grid_forward_3d_stride2",
    "point_to_grid_forward_3d_stride5",
    "FORWARD_KERNELS",
    "BACKWARD_KERNELS",
]
