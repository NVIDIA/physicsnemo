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
    _bounds_tensor,
    _camera_basis,
    _color_tensor,
    _empty_render_outputs,
    _field_gradient,
    _light_tensor,
    _make_ray_direction,
    _normalize_vec3,
    _optional_tensor_arg,
    _ray_box_intersection,
    _sample_color_trilinear,
    _sample_field_trilinear,
    _shade,
    _uniform_color_tensor,
    _validate_ambient,
    _validate_fov,
    _validate_image_shape,
)


@wp.kernel
def _isosurface_render_kernel(
    field: wp.array3d(dtype=wp.float32),
    color_field: wp.array4d(dtype=wp.float32),
    camera: wp.array(dtype=wp.vec3),
    bounds: wp.array(dtype=wp.vec3),
    uniform_color: wp.array(dtype=wp.vec4),
    light: wp.array(dtype=wp.vec3),
    width: int,
    height: int,
    threshold: wp.float32,
    step_size: wp.float32,
    max_steps: int,
    tan_half_fov: wp.float32,
    aspect: wp.float32,
    ambient: wp.float32,
    has_color_field: bool,
    nx: int,
    ny: int,
    nz: int,
    rgba: wp.array(dtype=wp.vec4),
    depth: wp.array(dtype=wp.float32),
    normal_out: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    ray_origin = camera[0]
    ray_direction = _make_ray_direction(
        tid, width, height, camera, tan_half_fov, aspect
    )
    bounds_min = bounds[0]
    bounds_max = bounds[1]
    intersection = _ray_box_intersection(
        ray_origin, ray_direction, bounds_min, bounds_max
    )

    if intersection[0] <= 0.0:
        rgba[tid] = wp.vec4(0.0, 0.0, 0.0, 0.0)
        depth[tid] = 3.402823e38
        normal_out[tid] = wp.vec3(0.0, 0.0, 0.0)
        return

    t_far = intersection[2]
    prev_t = intersection[1]
    prev_point = ray_origin + prev_t * ray_direction
    prev_value = _sample_field_trilinear(
        field, prev_point, bounds_min, bounds_max, nx, ny, nz
    )

    found = bool(False)
    hit_t = wp.float32(3.402823e38)

    for _ in range(max_steps):
        if found:
            break
        next_t = prev_t + step_size
        if next_t > t_far:
            break

        next_point = ray_origin + next_t * ray_direction
        next_value = _sample_field_trilinear(
            field, next_point, bounds_min, bounds_max, nx, ny, nz
        )
        if (prev_value - threshold) * (next_value - threshold) <= 0.0:
            denom = next_value - prev_value
            if wp.abs(denom) < 1.0e-7:
                if denom < 0.0:
                    denom = -1.0e-7
                else:
                    denom = 1.0e-7
            alpha = wp.min(wp.max((threshold - prev_value) / denom, 0.0), 1.0)
            hit_t = prev_t + alpha * step_size
            found = True

        prev_t = next_t
        prev_value = next_value

    if not found:
        rgba[tid] = wp.vec4(0.0, 0.0, 0.0, 0.0)
        depth[tid] = 3.402823e38
        normal_out[tid] = wp.vec3(0.0, 0.0, 0.0)
        return

    hit_point = ray_origin + hit_t * ray_direction
    normal = _normalize_vec3(
        _field_gradient(field, hit_point, bounds_min, bounds_max, nx, ny, nz)
    )
    if wp.dot(normal, ray_direction) > 0.0:
        normal = -normal

    color = uniform_color[0]
    if has_color_field:
        color = _sample_color_trilinear(
            color_field, hit_point, bounds_min, bounds_max, nx, ny, nz
        )

    rgba[tid] = _shade(color, normal, light[0], ambient)
    depth[tid] = hit_t
    normal_out[tid] = normal


@torch.library.custom_op("physicsnemo::isosurface_render_warp", mutates_args=())
def isosurface_render_impl(
    field: torch.Tensor,
    image_height: int,
    image_width: int,
    eye: torch.Tensor,
    center: torch.Tensor,
    up: torch.Tensor,
    fov_y_degrees: float,
    bounds_min: torch.Tensor,
    bounds_max: torch.Tensor,
    threshold: float = 0.0,
    step_size: float = 0.01,
    max_steps: int = 512,
    color_field: torch.Tensor | None = None,
    surface_color: torch.Tensor | None = None,
    light_direction: torch.Tensor | None = None,
    ambient: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Launch the Warp isosurface rendering custom op."""
    if field.ndim != 3:
        raise ValueError(
            f"field must have shape (nx, ny, nz), got {tuple(field.shape)}"
        )
    if any(size < 2 for size in field.shape):
        raise ValueError("field must have at least two samples in each dimension")
    _validate_image_shape(image_height, image_width)
    _validate_fov(fov_y_degrees)
    _validate_ambient(ambient)
    if step_size <= 0.0:
        raise ValueError("step_size must be strictly positive")
    if max_steps <= 0:
        raise ValueError("max_steps must be strictly positive")

    device = field.device
    field_fp32 = field.to(device=device, dtype=torch.float32).contiguous()
    camera = _camera_basis(eye, center, up, device=device)
    bounds = _bounds_tensor(bounds_min, bounds_max, device=device)
    color_field_fp32 = _color_tensor(
        color_field, device=device, shape_name="color_field", expected_rank=4
    )
    if color_field is not None and color_field.shape[:3] != field.shape:
        raise ValueError(
            f"color_field spatial shape must match field, got {tuple(color_field.shape[:3])}"
            f" and {tuple(field.shape)}"
        )
    uniform_color = _uniform_color_tensor(surface_color, device=device)
    light = _light_tensor(light_direction, device=device)

    rgba, depth, normal = _empty_render_outputs(
        image_height, image_width, device=device
    )
    wp_device, wp_stream = FunctionSpec.warp_launch_context(field_fp32)
    with FunctionSpec.warp_stream_scope(wp_stream):
        wp.launch(
            _isosurface_render_kernel,
            dim=image_height * image_width,
            inputs=[
                wp.from_torch(field_fp32, dtype=wp.float32),
                wp.from_torch(color_field_fp32, dtype=wp.float32),
                wp.from_torch(camera, dtype=wp.vec3),
                wp.from_torch(bounds, dtype=wp.vec3),
                wp.from_torch(uniform_color, dtype=wp.vec4),
                wp.from_torch(light, dtype=wp.vec3),
                int(image_width),
                int(image_height),
                float(threshold),
                float(step_size),
                int(max_steps),
                float(math.tan(math.radians(float(fov_y_degrees)) * 0.5)),
                float(image_width) / float(image_height),
                float(ambient),
                color_field is not None,
                int(field_fp32.shape[0]),
                int(field_fp32.shape[1]),
                int(field_fp32.shape[2]),
            ],
            outputs=[
                wp.from_torch(rgba.reshape(-1, 4), dtype=wp.vec4),
                wp.from_torch(depth.reshape(-1), dtype=wp.float32),
                wp.from_torch(normal.reshape(-1, 3), dtype=wp.vec3),
            ],
            device=wp_device,
            stream=wp_stream,
        )
    depth = torch.where(depth >= 3.0e38, torch.full_like(depth, torch.inf), depth)
    return rgba, depth, normal


@isosurface_render_impl.register_fake
def _(
    field: torch.Tensor,
    image_height: int,
    image_width: int,
    eye: torch.Tensor,
    center: torch.Tensor,
    up: torch.Tensor,
    fov_y_degrees: float,
    bounds_min: torch.Tensor,
    bounds_max: torch.Tensor,
    threshold: float = 0.0,
    step_size: float = 0.01,
    max_steps: int = 512,
    color_field: torch.Tensor | None = None,
    surface_color: torch.Tensor | None = None,
    light_direction: torch.Tensor | None = None,
    ambient: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _empty_render_outputs(image_height, image_width, device=field.device)


def isosurface_render_warp(
    field: torch.Tensor,
    image_height: int,
    image_width: int,
    eye: torch.Tensor,
    center: torch.Tensor,
    up: torch.Tensor,
    fov_y_degrees: float,
    bounds_min: torch.Tensor,
    bounds_max: torch.Tensor,
    threshold: float = 0.0,
    step_size: float = 0.01,
    max_steps: int = 512,
    color_field: torch.Tensor | None = None,
    surface_color: torch.Tensor | None = None,
    light_direction: torch.Tensor | None = None,
    ambient: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare tensor arguments and render an isosurface with Warp."""
    device = field.device
    return isosurface_render_impl(
        field,
        image_height,
        image_width,
        _as_vec3(eye, name="eye", device=device),
        _as_vec3(center, name="center", device=device),
        _as_vec3(up, name="up", device=device),
        fov_y_degrees,
        _as_vec3(bounds_min, name="bounds_min", device=device),
        _as_vec3(bounds_max, name="bounds_max", device=device),
        threshold,
        step_size,
        max_steps,
        color_field,
        _optional_tensor_arg(surface_color, device=device),
        _optional_tensor_arg(light_direction, device=device),
        ambient,
    )


__all__ = ["isosurface_render_warp"]
