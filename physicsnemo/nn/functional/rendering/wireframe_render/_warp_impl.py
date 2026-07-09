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

import math

import torch
import warp as wp

from physicsnemo.core.function_spec import FunctionSpec

from ..utils import (
    _as_vec3,
    _camera_basis,
    _empty_image_outputs,
    _optional_tensor_arg,
    _uniform_color_tensor,
    _validate_clip_range,
    _validate_fov,
    _validate_image_shape,
)


@wp.func
def _write_wireframe_winner(
    x: int,
    y: int,
    z: wp.float32,
    width: int,
    height: int,
    edge_id: int,
    num_edges: int,
    depth_scale: wp.float32,
    winners: wp.array(dtype=wp.int64),
):
    if x >= 0 and x < width and y >= 0 and y < height:
        depth_key = wp.int64(z * depth_scale)
        key = depth_key * wp.int64(num_edges) + wp.int64(edge_id)
        wp.atomic_min(winners, y * width + x, key)


@wp.func
def _draw_line_depth_tested(
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    z0: wp.float32,
    z1: wp.float32,
    width: int,
    height: int,
    thickness: int,
    edge_id: int,
    num_edges: int,
    depth_scale: wp.float32,
    winners: wp.array(dtype=wp.int64),
):
    dx = wp.abs(x1 - x0)
    dy = wp.abs(y1 - y0)
    sx = wp.int32(1)
    if x0 > x1:
        sx = -1
    sy = wp.int32(1)
    if y0 > y1:
        sy = -1
    err = dx - dy
    steps = wp.max(dx, dy)
    radius = thickness / 2
    x = x0
    y = y0
    step = int(0)

    while step <= steps:
        alpha = wp.float32(0.0)
        if steps > 0:
            alpha = wp.float32(step) / wp.float32(steps)
        z = z0 * (1.0 - alpha) + z1 * alpha
        for oy in range(-radius, radius + 1):
            for ox in range(-radius, radius + 1):
                _write_wireframe_winner(
                    x + ox,
                    y + oy,
                    z,
                    width,
                    height,
                    edge_id,
                    num_edges,
                    depth_scale,
                    winners,
                )

        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
        step += 1


@wp.kernel
def _wireframe_depth_kernel(
    edges: wp.array2d(dtype=wp.float32),
    camera: wp.array(dtype=wp.vec3),
    width: int,
    height: int,
    tan_half_fov: wp.float32,
    aspect: wp.float32,
    near: wp.float32,
    far: wp.float32,
    line_thickness: int,
    num_edges: int,
    depth_scale: wp.float32,
    winners: wp.array(dtype=wp.int64),
):
    tid = wp.tid()
    p0 = wp.vec3(edges[tid, 0], edges[tid, 1], edges[tid, 2])
    p1 = wp.vec3(edges[tid, 3], edges[tid, 4], edges[tid, 5])

    rel0 = p0 - camera[0]
    rel1 = p1 - camera[0]
    z0 = wp.dot(rel0, camera[1])
    z1 = wp.dot(rel1, camera[1])

    if (z0 <= near and z1 <= near) or (z0 >= far and z1 >= far):
        return

    if z0 < near:
        alpha = (near - z0) / (z1 - z0)
        p0 = p0 + alpha * (p1 - p0)
        z0 = near
    elif z1 < near:
        alpha = (near - z1) / (z0 - z1)
        p1 = p1 + alpha * (p0 - p1)
        z1 = near

    if z0 > far:
        alpha = (far - z0) / (z1 - z0)
        p0 = p0 + alpha * (p1 - p0)
        z0 = far
    elif z1 > far:
        alpha = (far - z1) / (z0 - z1)
        p1 = p1 + alpha * (p0 - p1)
        z1 = far

    clipped_rel0 = p0 - camera[0]
    clipped_rel1 = p1 - camera[0]
    x0_camera = wp.dot(clipped_rel0, camera[2])
    y0_camera = wp.dot(clipped_rel0, camera[3])
    x1_camera = wp.dot(clipped_rel1, camera[2])
    y1_camera = wp.dot(clipped_rel1, camera[3])
    screen_x0 = (
        (x0_camera / (z0 * tan_half_fov * aspect) + 1.0) * 0.5 * wp.float32(width)
    )
    screen_y0 = (1.0 - (y0_camera / (z0 * tan_half_fov) + 1.0) * 0.5) * wp.float32(
        height
    )
    screen_x1 = (
        (x1_camera / (z1 * tan_half_fov * aspect) + 1.0) * 0.5 * wp.float32(width)
    )
    screen_y1 = (1.0 - (y1_camera / (z1 * tan_half_fov) + 1.0) * 0.5) * wp.float32(
        height
    )

    _draw_line_depth_tested(
        int(screen_x0),
        int(screen_y0),
        int(screen_x1),
        int(screen_y1),
        z0,
        z1,
        width,
        height,
        line_thickness,
        tid,
        num_edges,
        depth_scale,
        winners,
    )


@wp.kernel
def _wireframe_resolve_kernel(
    uniform_color: wp.array(dtype=wp.vec4),
    num_edges: int,
    empty_key: wp.int64,
    depth_scale: wp.float32,
    winners: wp.array(dtype=wp.int64),
    rgba: wp.array(dtype=wp.vec4),
    depth: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    key = winners[tid]
    if key == empty_key:
        return

    depth_key = key / wp.int64(num_edges)
    rgba[tid] = uniform_color[0]
    depth[tid] = wp.float32(depth_key) / depth_scale


@torch.library.custom_op("physicsnemo::wireframe_render_warp", mutates_args=())
def wireframe_render_impl(
    edges: torch.Tensor,
    image_height: int,
    image_width: int,
    eye: torch.Tensor,
    center: torch.Tensor,
    up: torch.Tensor,
    fov_y_degrees: float,
    line_color: torch.Tensor | None = None,
    line_thickness: int = 1,
    near: float = 0.01,
    far: float = 1.0e8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch the Warp wireframe rendering custom op."""
    if edges.ndim == 3:
        if edges.shape[1:] != (2, 3):
            raise ValueError(
                "edges must have shape (num_edges, 2, 3) or (num_edges, 6)"
            )
        edges = edges.reshape(edges.shape[0], 6)
    elif edges.ndim != 2 or edges.shape[-1] != 6:
        raise ValueError("edges must have shape (num_edges, 2, 3) or (num_edges, 6)")
    if edges.shape[0] == 0:
        raise ValueError("edges must contain at least one edge")
    _validate_image_shape(image_height, image_width)
    _validate_fov(fov_y_degrees)
    if line_thickness <= 0:
        raise ValueError("line_thickness must be strictly positive")
    _validate_clip_range(near, far)

    device = edges.device
    edges_fp32 = edges.to(dtype=torch.float32).contiguous()
    uniform_color = _uniform_color_tensor(line_color, device=device)
    camera = _camera_basis(eye, center, up, device=device)
    rgba, depth = _empty_image_outputs(image_height, image_width, device=device)
    empty_key = torch.iinfo(torch.int64).max
    max_depth_key = float(empty_key // max(int(edges.shape[0]), 1) - 1)
    depth_scale = min(1.0e6, max_depth_key / float(far))
    winners = torch.full(
        (image_height, image_width), empty_key, device=device, dtype=torch.int64
    )
    wp_device, wp_stream = FunctionSpec.warp_launch_context(edges_fp32)
    with FunctionSpec.warp_stream_scope(wp_stream):
        wp.launch(
            _wireframe_depth_kernel,
            dim=int(edges.shape[0]),
            inputs=[
                wp.from_torch(edges_fp32, dtype=wp.float32),
                wp.from_torch(camera, dtype=wp.vec3),
                int(image_width),
                int(image_height),
                float(math.tan(math.radians(float(fov_y_degrees)) * 0.5)),
                float(image_width) / float(image_height),
                float(near),
                float(far),
                int(line_thickness),
                int(edges.shape[0]),
                float(depth_scale),
            ],
            outputs=[wp.from_torch(winners.reshape(-1), dtype=wp.int64)],
            device=wp_device,
            stream=wp_stream,
        )
        wp.launch(
            _wireframe_resolve_kernel,
            dim=image_height * image_width,
            inputs=[
                wp.from_torch(uniform_color, dtype=wp.vec4),
                int(edges.shape[0]),
                int(empty_key),
                float(depth_scale),
                wp.from_torch(winners.reshape(-1), dtype=wp.int64),
            ],
            outputs=[
                wp.from_torch(rgba.reshape(-1, 4), dtype=wp.vec4),
                wp.from_torch(depth.reshape(-1), dtype=wp.float32),
            ],
            device=wp_device,
            stream=wp_stream,
        )
    depth = torch.where(depth >= 3.0e38, torch.full_like(depth, torch.inf), depth)
    return rgba, depth


@wireframe_render_impl.register_fake
def _(
    edges: torch.Tensor,
    image_height: int,
    image_width: int,
    eye: torch.Tensor,
    center: torch.Tensor,
    up: torch.Tensor,
    fov_y_degrees: float,
    line_color: torch.Tensor | None = None,
    line_thickness: int = 1,
    near: float = 0.01,
    far: float = 1.0e8,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _empty_image_outputs(image_height, image_width, device=edges.device)


def wireframe_render_warp(
    edges: torch.Tensor,
    image_height: int,
    image_width: int,
    eye: torch.Tensor,
    center: torch.Tensor,
    up: torch.Tensor,
    fov_y_degrees: float,
    line_color: torch.Tensor | None = None,
    line_thickness: int = 1,
    near: float = 0.01,
    far: float = 1.0e8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare tensor arguments and rasterize wireframe segments with Warp."""
    device = edges.device
    return wireframe_render_impl(
        edges,
        image_height,
        image_width,
        _as_vec3(eye, name="eye", device=device),
        _as_vec3(center, name="center", device=device),
        _as_vec3(up, name="up", device=device),
        fov_y_degrees,
        _optional_tensor_arg(line_color, device=device),
        line_thickness,
        near,
        far,
    )


__all__ = ["wireframe_render_warp"]
