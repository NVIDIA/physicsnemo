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
    _empty_image_outputs,
    _make_ray_direction,
    _normalize_rgba_volume,
    _ray_box_intersection,
    _sample_color_trilinear,
    _validate_fov,
    _validate_image_shape,
    _validate_opacity,
)


@wp.kernel
def _volume_render_kernel(
    rgba_volume: wp.array4d(dtype=wp.float32),
    camera: wp.array(dtype=wp.vec3),
    bounds: wp.array(dtype=wp.vec3),
    width: int,
    height: int,
    step_size: wp.float32,
    max_steps: int,
    tan_half_fov: wp.float32,
    aspect: wp.float32,
    opacity_threshold: wp.float32,
    depth_threshold: wp.float32,
    nx: int,
    ny: int,
    nz: int,
    rgba: wp.array(dtype=wp.vec4),
    depth: wp.array(dtype=wp.float32),
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
        return

    accum = wp.vec4(0.0, 0.0, 0.0, 0.0)
    first_depth = wp.float32(3.402823e38)
    t = intersection[1]
    for _ in range(max_steps):
        if t > intersection[2] or accum[3] >= opacity_threshold:
            break
        sample = _sample_color_trilinear(
            rgba_volume,
            ray_origin + t * ray_direction,
            bounds_min,
            bounds_max,
            nx,
            ny,
            nz,
        )
        sample_alpha = wp.min(wp.max(sample[3], 0.0), 1.0)
        if sample_alpha > 0.0:
            opacity = (1.0 - accum[3]) * sample_alpha
            accum[0] += sample[0] * opacity
            accum[1] += sample[1] * opacity
            accum[2] += sample[2] * opacity
            accum[3] += opacity
            if first_depth >= 3.0e38 and accum[3] >= depth_threshold:
                first_depth = t
        t += step_size

    if accum[3] <= 0.0:
        rgba[tid] = wp.vec4(0.0, 0.0, 0.0, 0.0)
        depth[tid] = 3.402823e38
        return

    rgba[tid] = wp.vec4(
        accum[0] / accum[3],
        accum[1] / accum[3],
        accum[2] / accum[3],
        accum[3],
    )
    depth[tid] = first_depth


@torch.library.custom_op("physicsnemo::volume_render_warp", mutates_args=())
def volume_render_impl(
    rgba_volume: torch.Tensor,
    image_height: int,
    image_width: int,
    eye: torch.Tensor,
    center: torch.Tensor,
    up: torch.Tensor,
    fov_y_degrees: float,
    bounds_min: torch.Tensor,
    bounds_max: torch.Tensor,
    step_size: float = 0.01,
    max_steps: int = 512,
    opacity_threshold: float = 0.95,
    depth_threshold: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch the Warp RGBA volume rendering custom op."""
    _validate_image_shape(image_height, image_width)
    _validate_fov(fov_y_degrees)
    if step_size <= 0.0:
        raise ValueError("step_size must be strictly positive")
    if max_steps <= 0:
        raise ValueError("max_steps must be strictly positive")
    _validate_opacity(opacity_threshold, name="opacity_threshold")
    _validate_opacity(depth_threshold, name="depth_threshold")

    device = rgba_volume.device
    rgba_volume_fp32 = _normalize_rgba_volume(rgba_volume)
    camera = _camera_basis(eye, center, up, device=device)
    bounds = _bounds_tensor(bounds_min, bounds_max, device=device)
    rgba, depth = _empty_image_outputs(image_height, image_width, device=device)
    wp_device, wp_stream = FunctionSpec.warp_launch_context(rgba_volume_fp32)
    with FunctionSpec.warp_stream_scope(wp_stream):
        wp.launch(
            _volume_render_kernel,
            dim=image_height * image_width,
            inputs=[
                wp.from_torch(rgba_volume_fp32, dtype=wp.float32),
                wp.from_torch(camera, dtype=wp.vec3),
                wp.from_torch(bounds, dtype=wp.vec3),
                int(image_width),
                int(image_height),
                float(step_size),
                int(max_steps),
                float(math.tan(math.radians(float(fov_y_degrees)) * 0.5)),
                float(image_width) / float(image_height),
                float(opacity_threshold),
                float(depth_threshold),
                int(rgba_volume.shape[0]),
                int(rgba_volume.shape[1]),
                int(rgba_volume.shape[2]),
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


@volume_render_impl.register_fake
def _(
    rgba_volume: torch.Tensor,
    image_height: int,
    image_width: int,
    eye: torch.Tensor,
    center: torch.Tensor,
    up: torch.Tensor,
    fov_y_degrees: float,
    bounds_min: torch.Tensor,
    bounds_max: torch.Tensor,
    step_size: float = 0.01,
    max_steps: int = 512,
    opacity_threshold: float = 0.95,
    depth_threshold: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _empty_image_outputs(image_height, image_width, device=rgba_volume.device)


def volume_render_warp(
    rgba_volume: torch.Tensor,
    image_height: int,
    image_width: int,
    eye: torch.Tensor,
    center: torch.Tensor,
    up: torch.Tensor,
    fov_y_degrees: float,
    bounds_min: torch.Tensor,
    bounds_max: torch.Tensor,
    step_size: float = 0.01,
    max_steps: int = 512,
    opacity_threshold: float = 0.95,
    depth_threshold: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare tensor arguments and render an RGBA volume with Warp."""
    device = rgba_volume.device
    return volume_render_impl(
        rgba_volume,
        image_height,
        image_width,
        _as_vec3(eye, name="eye", device=device),
        _as_vec3(center, name="center", device=device),
        _as_vec3(up, name="up", device=device),
        fov_y_degrees,
        _as_vec3(bounds_min, name="bounds_min", device=device),
        _as_vec3(bounds_max, name="bounds_max", device=device),
        step_size,
        max_steps,
        opacity_threshold,
        depth_threshold,
    )


__all__ = ["volume_render_warp"]
