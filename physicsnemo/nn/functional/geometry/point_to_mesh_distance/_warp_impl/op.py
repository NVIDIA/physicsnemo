# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Torch custom-op integration for Warp nearest-face queries."""

from __future__ import annotations

import torch
import warp as wp

from physicsnemo.core.function_spec import FunctionSpec

from .._torch_impl import _point_to_mesh_distance_from_face_indices
from .._utils import normalize_point_to_mesh_inputs
from .kernels import nearest_face_indices_f32

wp.init()
wp.config.log_level = wp.LOG_WARNING


def _prepare_warp_search_inputs(
    mesh_vertices: torch.Tensor,
    mesh_indices: torch.Tensor,
    input_points: torch.Tensor,
    inputs_normalized: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Validate normalized tensors and make the Warp launch layouts."""

    if input_points.ndim != 2 or input_points.shape[-1] != 3:
        raise ValueError("input_points must have normalized shape (num_queries, 3)")
    if inputs_normalized:
        # The public wrapper has already enforced the full device-side tensor
        # contract. Retain cheap structural checks because this boundary is an
        # opaque custom op under torch.compile.
        if mesh_vertices.ndim != 2 or mesh_vertices.shape[-1] != 3:
            raise ValueError("mesh_vertices must have shape (num_vertices, 3)")
        if mesh_indices.ndim != 2 or mesh_indices.shape[-1] != 3:
            raise ValueError("mesh_indices must have shape (num_faces, 3)")
        if mesh_vertices.shape[0] == 0 or mesh_indices.shape[0] == 0:
            raise ValueError("the target mesh must contain vertices and triangles")
        if mesh_indices.dtype not in (torch.int32, torch.int64):
            raise TypeError("mesh_indices must have dtype torch.int32 or torch.int64")
        if (
            mesh_vertices.device != input_points.device
            or mesh_vertices.device != mesh_indices.device
        ):
            raise ValueError(
                "mesh_vertices, mesh_indices, and input_points must share a device"
            )
        vertices, faces, queries = mesh_vertices, mesh_indices, input_points
    else:
        vertices, faces, queries, _ = normalize_point_to_mesh_inputs(
            mesh_vertices, mesh_indices, input_points, False
        )
    if vertices.dtype != torch.float32:
        raise TypeError(
            "the Warp point-to-mesh backend supports only torch.float32 coordinates"
        )
    if vertices.device.type not in ("cpu", "cuda"):
        raise ValueError("the Warp point-to-mesh backend requires a CPU or CUDA device")

    # Warp Mesh requires contiguous vec3f coordinates and flattened int32
    # connectivity. Shared device-side validation has established bounds,
    # finiteness, and nondegeneracy without a host readback.
    return (
        vertices.contiguous(),
        faces.to(torch.int32).reshape(-1).contiguous(),
        queries.contiguous(),
    )


@torch.library.custom_op(
    "physicsnemo::point_to_mesh_nearest_face_warp_impl", mutates_args=()
)
def nearest_face_indices_warp_impl(
    mesh_vertices: torch.Tensor,
    mesh_indices: torch.Tensor,
    input_points: torch.Tensor,
    inputs_normalized: bool = False,
) -> torch.Tensor:
    """Find nearest triangle IDs with a stateless Warp BVH query."""

    vertices, flattened_indices, queries = _prepare_warp_search_inputs(
        mesh_vertices, mesh_indices, input_points, inputs_normalized
    )
    nearest_faces = torch.empty(
        queries.shape[0], dtype=torch.long, device=queries.device
    )
    if queries.shape[0] == 0:
        return nearest_faces

    wp_device, wp_stream = FunctionSpec.warp_launch_context(queries)
    # This guard orders Warp's temporary Mesh/BVH cleanup after work enqueued on
    # Torch's borrowed stream. The stateless API intentionally rebuilds its BVH
    # for every call.
    with FunctionSpec.warp_stream_scope(wp_stream, sync_enter=False):
        wp_vertices = wp.from_torch(
            vertices.detach(), dtype=wp.vec3f, requires_grad=False
        )
        wp_indices = wp.from_torch(
            flattened_indices.detach(), dtype=wp.int32, requires_grad=False
        )
        wp_queries = wp.from_torch(
            queries.detach(), dtype=wp.vec3f, requires_grad=False
        )
        wp_nearest_faces = wp.from_torch(
            nearest_faces, dtype=wp.int64, requires_grad=False
        )
        mesh = wp.Mesh(points=wp_vertices, indices=wp_indices)
        wp.launch(
            nearest_face_indices_f32,
            dim=queries.shape[0],
            inputs=[mesh.id, wp_queries, float("inf"), wp_nearest_faces],
            device=wp_device,
            stream=wp_stream,
        )

    return nearest_faces


@nearest_face_indices_warp_impl.register_fake
def _nearest_face_indices_warp_fake(
    mesh_vertices: torch.Tensor,
    mesh_indices: torch.Tensor,
    input_points: torch.Tensor,
    inputs_normalized: bool = False,
) -> torch.Tensor:
    """Describe the integer correspondence output for tracing and export."""

    _ = mesh_vertices, mesh_indices, inputs_normalized
    return torch.empty(
        input_points.shape[0], dtype=torch.long, device=input_points.device
    )


def point_to_mesh_distance_warp(
    mesh_vertices: torch.Tensor,
    mesh_indices: torch.Tensor,
    input_points: torch.Tensor,
    *,
    squared: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Use Warp for hard face selection and Torch for continuous projection."""

    vertices, faces, queries, input_shape = normalize_point_to_mesh_inputs(
        mesh_vertices, mesh_indices, input_points, squared
    )
    if vertices.dtype != torch.float32 or queries.dtype != torch.float32:
        raise TypeError(
            "the Warp point-to-mesh backend supports only torch.float32 coordinates"
        )
    if vertices.device.type not in ("cpu", "cuda"):
        raise ValueError("the Warp point-to-mesh backend requires a CPU or CUDA device")

    if queries.shape[0] == 0:
        nearest_faces = torch.empty(0, dtype=torch.long, device=queries.device)
    else:
        nearest_faces = nearest_face_indices_warp_impl(
            vertices, faces, queries, inputs_normalized=True
        )
    return _point_to_mesh_distance_from_face_indices(
        vertices,
        faces,
        queries,
        input_shape,
        nearest_faces,
        squared=squared,
    )


__all__ = ["nearest_face_indices_warp_impl", "point_to_mesh_distance_warp"]
