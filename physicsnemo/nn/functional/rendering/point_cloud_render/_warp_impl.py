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
    _color_tensor,
    _empty_image_outputs,
    _optional_tensor_arg,
    _project_point,
    _uniform_color_tensor,
    _validate_clip_range,
    _validate_fov,
    _validate_image_shape,
)


@wp.kernel
def _point_cloud_depth_kernel(
    points: wp.array2d(dtype=wp.float32),
    camera: wp.array(dtype=wp.vec3),
    width: int,
    height: int,
    tan_half_fov: wp.float32,
    aspect: wp.float32,
    near: wp.float32,
    far: wp.float32,
    point_size: int,
    num_points: int,
    depth_scale: wp.float32,
    winners: wp.array(dtype=wp.int64),
):
    tid = wp.tid()
    point = wp.vec3(points[tid, 0], points[tid, 1], points[tid, 2])
    projected = _project_point(point, camera, width, height, tan_half_fov, aspect)
    z = projected[2]
    if z <= near or z >= far:
        return

    radius = point_size / 2
    center_x = int(projected[0])
    center_y = int(projected[1])
    key = wp.int64(z * depth_scale) * wp.int64(num_points) + wp.int64(tid)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            x = center_x + dx
            y = center_y + dy
            if x >= 0 and x < width and y >= 0 and y < height:
                wp.atomic_min(winners, y * width + x, key)


@wp.kernel
def _point_cloud_resolve_kernel(
    points: wp.array2d(dtype=wp.float32),
    colors: wp.array2d(dtype=wp.float32),
    camera: wp.array(dtype=wp.vec3),
    uniform_color: wp.array(dtype=wp.vec4),
    width: int,
    height: int,
    tan_half_fov: wp.float32,
    aspect: wp.float32,
    has_point_colors: bool,
    num_points: int,
    empty_key: wp.int64,
    winners: wp.array(dtype=wp.int64),
    rgba: wp.array(dtype=wp.vec4),
    depth: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    key = winners[tid]
    if key == empty_key:
        return

    point_id = int(key % wp.int64(num_points))
    point = wp.vec3(points[point_id, 0], points[point_id, 1], points[point_id, 2])
    projected = _project_point(point, camera, width, height, tan_half_fov, aspect)

    color = uniform_color[0]
    if has_point_colors:
        color = wp.vec4(
            colors[point_id, 0],
            colors[point_id, 1],
            colors[point_id, 2],
            colors[point_id, 3],
        )
    rgba[tid] = color
    depth[tid] = projected[2]


@torch.library.custom_op("physicsnemo::point_cloud_render_warp", mutates_args=())
def point_cloud_render_impl(
    points: torch.Tensor,
    image_height: int,
    image_width: int,
    eye: torch.Tensor,
    center: torch.Tensor,
    up: torch.Tensor,
    fov_y_degrees: float,
    point_colors: torch.Tensor | None = None,
    point_color: torch.Tensor | None = None,
    point_size: int = 1,
    near: float = 0.01,
    far: float = 1.0e8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch the Warp point cloud rendering custom op."""
    if points.ndim != 2 or points.shape[-1] != 3:
        raise ValueError(f"points must have shape (num_points, 3), got {points.shape}")
    if points.shape[0] == 0:
        raise ValueError("points must contain at least one point")
    _validate_image_shape(image_height, image_width)
    _validate_fov(fov_y_degrees)
    if point_size <= 0:
        raise ValueError("point_size must be strictly positive")
    _validate_clip_range(near, far)

    device = points.device
    points_fp32 = points.to(dtype=torch.float32).contiguous()
    colors = torch.zeros((1, 4), device=device, dtype=torch.float32)
    has_point_colors = point_colors is not None
    if point_colors is not None:
        if point_color is not None:
            raise ValueError("Pass either point_colors or point_color, not both")
        if point_colors.shape[0] != points.shape[0]:
            raise ValueError("point_colors must have one color per point")
        colors = _color_tensor(
            point_colors, device=device, shape_name="point_colors", expected_rank=2
        )
    uniform_color = _uniform_color_tensor(point_color, device=device)
    camera = _camera_basis(eye, center, up, device=device)
    rgba, depth = _empty_image_outputs(image_height, image_width, device=device)
    empty_key = torch.iinfo(torch.int64).max
    max_depth_key = float(empty_key // max(int(points.shape[0]), 1) - 1)
    depth_scale = min(1.0e6, max_depth_key / float(far))
    winners = torch.full(
        (image_height, image_width), empty_key, device=device, dtype=torch.int64
    )
    wp_device, wp_stream = FunctionSpec.warp_launch_context(points_fp32)
    with FunctionSpec.warp_stream_scope(wp_stream):
        wp.launch(
            _point_cloud_depth_kernel,
            dim=int(points.shape[0]),
            inputs=[
                wp.from_torch(points_fp32, dtype=wp.float32),
                wp.from_torch(camera, dtype=wp.vec3),
                int(image_width),
                int(image_height),
                float(math.tan(math.radians(float(fov_y_degrees)) * 0.5)),
                float(image_width) / float(image_height),
                float(near),
                float(far),
                int(point_size),
                int(points.shape[0]),
                float(depth_scale),
            ],
            outputs=[wp.from_torch(winners.reshape(-1), dtype=wp.int64)],
            device=wp_device,
            stream=wp_stream,
        )
        wp.launch(
            _point_cloud_resolve_kernel,
            dim=image_height * image_width,
            inputs=[
                wp.from_torch(points_fp32, dtype=wp.float32),
                wp.from_torch(colors, dtype=wp.float32),
                wp.from_torch(camera, dtype=wp.vec3),
                wp.from_torch(uniform_color, dtype=wp.vec4),
                int(image_width),
                int(image_height),
                float(math.tan(math.radians(float(fov_y_degrees)) * 0.5)),
                float(image_width) / float(image_height),
                bool(has_point_colors),
                int(points.shape[0]),
                int(empty_key),
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


@point_cloud_render_impl.register_fake
def _(
    points: torch.Tensor,
    image_height: int,
    image_width: int,
    eye: torch.Tensor,
    center: torch.Tensor,
    up: torch.Tensor,
    fov_y_degrees: float,
    point_colors: torch.Tensor | None = None,
    point_color: torch.Tensor | None = None,
    point_size: int = 1,
    near: float = 0.01,
    far: float = 1.0e8,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _empty_image_outputs(image_height, image_width, device=points.device)


def point_cloud_render_warp(
    points: torch.Tensor,
    image_height: int,
    image_width: int,
    eye: torch.Tensor,
    center: torch.Tensor,
    up: torch.Tensor,
    fov_y_degrees: float,
    point_colors: torch.Tensor | None = None,
    point_color: torch.Tensor | None = None,
    point_size: int = 1,
    near: float = 0.01,
    far: float = 1.0e8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare tensor arguments and rasterize a point cloud with Warp."""
    device = points.device
    return point_cloud_render_impl(
        points,
        image_height,
        image_width,
        _as_vec3(eye, name="eye", device=device),
        _as_vec3(center, name="center", device=device),
        _as_vec3(up, name="up", device=device),
        fov_y_degrees,
        point_colors,
        _optional_tensor_arg(point_color, device=device),
        point_size,
        near,
        far,
    )


__all__ = ["point_cloud_render_warp"]
