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
    _empty_render_outputs,
    _light_tensor,
    _make_ray_direction,
    _normalize_vec3,
    _optional_tensor_arg,
    _shade,
    _uniform_color_tensor,
    _validate_ambient,
    _validate_fov,
    _validate_image_shape,
)


@wp.kernel
def _mesh_raycast_kernel(
    mesh_id: wp.uint64,
    color_values: wp.array2d(dtype=wp.float32),
    camera: wp.array(dtype=wp.vec3),
    uniform_color: wp.array(dtype=wp.vec4),
    light: wp.array(dtype=wp.vec3),
    width: int,
    height: int,
    tan_half_fov: wp.float32,
    aspect: wp.float32,
    max_distance: wp.float32,
    ambient: wp.float32,
    color_mode: int,
    rgba: wp.array(dtype=wp.vec4),
    depth: wp.array(dtype=wp.float32),
    normal_out: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    ray_origin = camera[0]
    ray_direction = _make_ray_direction(
        tid, width, height, camera, tan_half_fov, aspect
    )
    query = wp.mesh_query_ray(mesh_id, ray_origin, ray_direction, max_distance)

    if not query.result:
        rgba[tid] = wp.vec4(0.0, 0.0, 0.0, 0.0)
        depth[tid] = 3.402823e38
        normal_out[tid] = wp.vec3(0.0, 0.0, 0.0)
        return

    normal = _normalize_vec3(query.normal)
    if wp.dot(normal, ray_direction) > 0.0:
        normal = -normal

    color = uniform_color[0]
    if color_mode == 1:
        mesh = wp.mesh_get(mesh_id)
        i0 = mesh.indices[3 * query.face + 0]
        i1 = mesh.indices[3 * query.face + 1]
        i2 = mesh.indices[3 * query.face + 2]
        w0 = query.u
        w1 = query.v
        w2 = 1.0 - query.u - query.v
        color = wp.vec4(
            w0 * color_values[i0, 0]
            + w1 * color_values[i1, 0]
            + w2 * color_values[i2, 0],
            w0 * color_values[i0, 1]
            + w1 * color_values[i1, 1]
            + w2 * color_values[i2, 1],
            w0 * color_values[i0, 2]
            + w1 * color_values[i1, 2]
            + w2 * color_values[i2, 2],
            w0 * color_values[i0, 3]
            + w1 * color_values[i1, 3]
            + w2 * color_values[i2, 3],
        )
    elif color_mode == 2:
        color = wp.vec4(
            color_values[query.face, 0],
            color_values[query.face, 1],
            color_values[query.face, 2],
            color_values[query.face, 3],
        )

    rgba[tid] = _shade(color, normal, light[0], ambient)
    depth[tid] = query.t
    normal_out[tid] = normal


@torch.library.custom_op("physicsnemo::mesh_raycast_warp", mutates_args=())
def mesh_raycast_impl(
    mesh_vertices: torch.Tensor,
    mesh_indices: torch.Tensor,
    image_height: int,
    image_width: int,
    eye: torch.Tensor,
    center: torch.Tensor,
    up: torch.Tensor,
    fov_y_degrees: float,
    vertex_colors: torch.Tensor | None = None,
    face_colors: torch.Tensor | None = None,
    surface_color: torch.Tensor | None = None,
    light_direction: torch.Tensor | None = None,
    ambient: float = 0.2,
    max_distance: float = 1.0e8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Launch the Warp mesh raycast rendering custom op."""
    if mesh_vertices.ndim != 2 or mesh_vertices.shape[-1] != 3:
        raise ValueError(
            "mesh_vertices must have shape (num_vertices, 3), got "
            f"{tuple(mesh_vertices.shape)}"
        )
    if mesh_vertices.shape[0] == 0:
        raise ValueError("mesh_vertices must contain at least one vertex")
    if mesh_indices.ndim == 2:
        if mesh_indices.shape[-1] != 3:
            raise ValueError("mesh_indices must have shape (num_faces, 3)")
        mesh_indices = mesh_indices.reshape(-1)
    elif mesh_indices.ndim != 1:
        raise ValueError("mesh_indices must be 1D or have shape (num_faces, 3)")
    if mesh_indices.numel() == 0 or mesh_indices.numel() % 3 != 0:
        raise ValueError("mesh_indices must contain complete triangle faces")
    if vertex_colors is not None and face_colors is not None:
        raise ValueError("Pass either vertex_colors or face_colors, not both")
    _validate_image_shape(image_height, image_width)
    _validate_fov(fov_y_degrees)
    _validate_ambient(ambient)
    if max_distance <= 0.0:
        raise ValueError("max_distance must be strictly positive")

    device = mesh_vertices.device
    mesh_vertices_fp32 = mesh_vertices.to(dtype=torch.float32).contiguous()
    mesh_indices_i32 = mesh_indices.to(device=device, dtype=torch.int32).contiguous()
    camera = _camera_basis(eye, center, up, device=device)
    uniform_color = _uniform_color_tensor(surface_color, device=device)
    light = _light_tensor(light_direction, device=device)

    color_mode = 0
    color_values = torch.zeros((1, 4), device=device, dtype=torch.float32)
    if vertex_colors is not None:
        if vertex_colors.shape[0] != mesh_vertices.shape[0]:
            raise ValueError("vertex_colors must have one color per mesh vertex")
        color_values = _color_tensor(
            vertex_colors, device=device, shape_name="vertex_colors", expected_rank=2
        )
        color_mode = 1
    elif face_colors is not None:
        num_faces = mesh_indices_i32.numel() // 3
        if face_colors.shape[0] != num_faces:
            raise ValueError("face_colors must have one color per mesh face")
        color_values = _color_tensor(
            face_colors, device=device, shape_name="face_colors", expected_rank=2
        )
        color_mode = 2

    rgba, depth, normal = _empty_render_outputs(
        image_height, image_width, device=device
    )
    wp_device, wp_stream = FunctionSpec.warp_launch_context(mesh_vertices_fp32)
    with FunctionSpec.warp_stream_scope(wp_stream):
        wp_vertices = wp.from_torch(mesh_vertices_fp32, dtype=wp.vec3)
        wp_indices = wp.from_torch(mesh_indices_i32, dtype=wp.int32)
        mesh = wp.Mesh(points=wp_vertices, indices=wp_indices)
        wp.launch(
            _mesh_raycast_kernel,
            dim=image_height * image_width,
            inputs=[
                mesh.id,
                wp.from_torch(color_values, dtype=wp.float32),
                wp.from_torch(camera, dtype=wp.vec3),
                wp.from_torch(uniform_color, dtype=wp.vec4),
                wp.from_torch(light, dtype=wp.vec3),
                int(image_width),
                int(image_height),
                float(math.tan(math.radians(float(fov_y_degrees)) * 0.5)),
                float(image_width) / float(image_height),
                float(max_distance),
                float(ambient),
                int(color_mode),
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


@mesh_raycast_impl.register_fake
def _(
    mesh_vertices: torch.Tensor,
    mesh_indices: torch.Tensor,
    image_height: int,
    image_width: int,
    eye: torch.Tensor,
    center: torch.Tensor,
    up: torch.Tensor,
    fov_y_degrees: float,
    vertex_colors: torch.Tensor | None = None,
    face_colors: torch.Tensor | None = None,
    surface_color: torch.Tensor | None = None,
    light_direction: torch.Tensor | None = None,
    ambient: float = 0.2,
    max_distance: float = 1.0e8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _empty_render_outputs(image_height, image_width, device=mesh_vertices.device)


def mesh_raycast_warp(
    mesh_vertices: torch.Tensor,
    mesh_indices: torch.Tensor,
    image_height: int,
    image_width: int,
    eye: torch.Tensor,
    center: torch.Tensor,
    up: torch.Tensor,
    fov_y_degrees: float,
    vertex_colors: torch.Tensor | None = None,
    face_colors: torch.Tensor | None = None,
    surface_color: torch.Tensor | None = None,
    light_direction: torch.Tensor | None = None,
    ambient: float = 0.2,
    max_distance: float = 1.0e8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare tensor arguments and raycast a mesh with Warp."""
    device = mesh_vertices.device
    return mesh_raycast_impl(
        mesh_vertices,
        mesh_indices,
        image_height,
        image_width,
        _as_vec3(eye, name="eye", device=device),
        _as_vec3(center, name="center", device=device),
        _as_vec3(up, name="up", device=device),
        fov_y_degrees,
        vertex_colors,
        face_colors,
        _optional_tensor_arg(surface_color, device=device),
        _optional_tensor_arg(light_direction, device=device),
        ambient,
        max_distance,
    )


__all__ = ["mesh_raycast_warp"]
