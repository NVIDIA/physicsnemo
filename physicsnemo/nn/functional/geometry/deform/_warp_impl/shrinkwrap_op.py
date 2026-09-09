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

"""Torch custom-op integration for Warp shrinkwrap face searches."""

from __future__ import annotations

import torch
import warp as wp

from physicsnemo.core.function_spec import FunctionSpec

from .._shrinkwrap_torch_impl import (
    _valid_triangles,
    nearest_surface_faces_torch,
)
from .shrinkwrap_kernels import nearest_surface_faces_kernel

wp.init()
wp.config.log_level = wp.LOG_WARNING

_FLOAT_DTYPES = (torch.float32, torch.float64)
_INDEX_DTYPES = (torch.int32, torch.int64)
_WARP_MAX_DISTANCE = float(torch.finfo(torch.float32).max)
_WARP_MAX_COORDINATE_MAGNITUDE = float(2**20)
_WARP_MIN_EDGE_SCALE = float(2**-20)
_WARP_MAX_EDGE_SCALE = float(2**20)
_WARP_MIN_RELATIVE_TARGET_AREA = 256.0 * torch.finfo(torch.float32).eps


def _validate_inputs(
    target_points: torch.Tensor,
    target_faces: torch.Tensor,
    query_points: torch.Tensor,
    max_distance: torch.Tensor,
) -> None:
    """Validate the normalized nearest-face search contract."""

    if target_points.ndim != 2 or tuple(target_points.shape[1:]) != (3,):
        raise ValueError("target_points must have shape (num_vertices, 3)")
    if target_faces.ndim != 2 or tuple(target_faces.shape[1:]) != (3,):
        raise ValueError("target_faces must have shape (num_faces, 3)")
    if query_points.ndim != 2 or tuple(query_points.shape[1:]) != (3,):
        raise ValueError("query_points must have shape (num_queries, 3)")
    if target_points.dtype not in _FLOAT_DTYPES:
        raise TypeError("target_points must have float32 or float64 dtype")
    if query_points.dtype not in _FLOAT_DTYPES:
        raise TypeError("query_points must have float32 or float64 dtype")
    if max_distance.dtype not in _FLOAT_DTYPES:
        raise TypeError("max_distance must have float32 or float64 dtype")
    if target_faces.dtype not in _INDEX_DTYPES:
        raise TypeError("target_faces must have int32 or int64 dtype")
    if (
        target_points.device != target_faces.device
        or target_points.device != query_points.device
        or target_points.device != max_distance.device
    ):
        raise ValueError("all nearest-face search tensors must be on the same device")
    if target_points.device.type not in ("cpu", "cuda"):
        raise ValueError("the Warp nearest-face search supports CPU and CUDA tensors")
    if max_distance.ndim != 0:
        raise ValueError("max_distance must be a scalar tensor")
    if target_faces.shape[0] > 0 and target_points.shape[0] == 0:
        raise ValueError("target_points cannot be empty when target_faces is nonempty")


def _prepare_float32_search_coordinates(
    target_points: torch.Tensor,
    query_points: torch.Tensor,
    max_distance: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Prepare unchanged float32 coordinates for a safe Warp search."""

    target_f32 = target_points.detach().contiguous()
    query_f32 = query_points.detach()
    # Warp's closest-point intermediates include products of dot products.
    # Queries outside the verified float32 range use Torch below.
    warp_query_mask = torch.isfinite(query_f32).all(dim=-1) & (
        query_f32.abs().amax(dim=-1) <= _WARP_MAX_COORDINATE_MAGNITUDE
    )
    query_f32 = torch.where(
        warp_query_mask.unsqueeze(-1),
        query_f32,
        torch.zeros_like(query_f32),
    )
    max_distance_f32 = max_distance.detach().reshape(1)
    search_padding = (
        64.0 * torch.finfo(torch.float32).eps * (1.0 + max_distance_f32.abs())
    )
    max_distance_f32 = (max_distance_f32 + search_padding).clamp(max=_WARP_MAX_DISTANCE)
    return (
        target_f32,
        query_f32.contiguous(),
        max_distance_f32.contiguous(),
        warp_query_mask,
    )


def _float32_target_search_is_safe(
    target_points: torch.Tensor,
    target_faces: torch.Tensor,
) -> bool:
    """Return whether Warp can search the original target coordinates."""

    # Float64 face separations can be smaller than one float32 coordinate step
    # and therefore cannot preserve nearest-face ordering in a float32 search.
    if target_points.dtype != torch.float32:
        return False

    target_f32 = target_points.detach()
    target_magnitude_is_safe = torch.isfinite(target_f32).all() & (
        target_f32.abs().amax() <= _WARP_MAX_COORDINATE_MAGNITUDE
    )
    if not bool(target_magnitude_is_safe):
        return False

    faces = target_faces.reshape(-1, 3).to(torch.int64)
    triangles = target_f32[faces]
    ab = triangles[:, 1] - triangles[:, 0]
    ac = triangles[:, 2] - triangles[:, 0]
    bc = triangles[:, 2] - triangles[:, 1]
    edge_scale = torch.stack(
        (
            ab.abs().amax(dim=-1),
            ac.abs().amax(dim=-1),
            bc.abs().amax(dim=-1),
        ),
        dim=-1,
    ).amax(dim=-1)
    safe_edge_scale = torch.where(
        edge_scale > 0.0,
        edge_scale,
        torch.ones_like(edge_scale),
    )
    scale_column = safe_edge_scale.unsqueeze(-1)
    relative_area = (
        torch.linalg.cross(
            ab / scale_column,
            ac / scale_column,
        )
        .abs()
        .amax(dim=-1)
    )
    # Warp's float32 mesh query can miss very skinny faces that are otherwise
    # valid on CUDA. Keep those faces on the numerically stable Torch search.
    aspect_is_safe = relative_area > _WARP_MIN_RELATIVE_TARGET_AREA
    edge_scale_is_safe = (edge_scale >= _WARP_MIN_EDGE_SCALE) & (
        edge_scale <= _WARP_MAX_EDGE_SCALE
    )
    faces_are_valid = _valid_triangles(triangles)
    # The lower edge and relative-area bounds keep Warp's normal-length square
    # above 2^-110. The coordinate and upper-edge bounds keep its closest-point
    # products below 72 * 2^80. Both have ample room inside float32's normal
    # exponent range on CPU and CUDA.
    search_is_safe = (faces_are_valid & aspect_is_safe & edge_scale_is_safe).all()
    return bool(search_is_safe)


@torch.library.custom_op(
    "physicsnemo::nearest_surface_faces_warp_impl",
    mutates_args=(),
    tags=(torch.Tag.cudagraph_unsafe,),
)
def nearest_surface_faces_warp_impl(
    target_points: torch.Tensor,
    target_faces: torch.Tensor,
    query_points: torch.Tensor,
    max_distance: torch.Tensor,
) -> torch.Tensor:
    """Find the nearest target triangle for each query point with Warp."""

    _validate_inputs(target_points, target_faces, query_points, max_distance)
    face_ids = torch.full(
        (query_points.shape[0],),
        -1,
        dtype=torch.int64,
        device=query_points.device,
    )
    if query_points.shape[0] == 0 or target_faces.shape[0] == 0:
        return face_ids

    target_faces_long = target_faces.detach().to(torch.int64)
    triangles = target_points.detach()[target_faces_long]
    valid_face_ids = torch.nonzero(
        _valid_triangles(triangles),
        as_tuple=False,
    ).flatten()
    if valid_face_ids.numel() == 0:
        return face_ids

    valid_faces = target_faces_long[valid_face_ids]
    valid_vertex_ids, compact_faces = torch.unique(
        valid_faces.reshape(-1),
        sorted=True,
        return_inverse=True,
    )
    compact_target_points = target_points[valid_vertex_ids]
    if compact_target_points.shape[0] > torch.iinfo(torch.int32).max:
        raise ValueError("Warp meshes support at most int32-addressable vertices")

    search_tensors_are_float32 = (
        query_points.dtype == torch.float32 and max_distance.dtype == torch.float32
    )
    target_search_is_safe = (
        search_tensors_are_float32
        and _float32_target_search_is_safe(
            compact_target_points,
            compact_faces,
        )
    )
    if not target_search_is_safe:
        fallback_max_distance = float(max_distance.detach())
        return nearest_surface_faces_torch(
            query_points,
            target_points,
            target_faces,
            fallback_max_distance,
        )

    # The public replay uses the original tensors for values and gradients
    # after this discrete float32 search.
    (
        target_points_f32,
        query_points_f32,
        max_distance_f32,
        warp_query_mask,
    ) = _prepare_float32_search_coordinates(
        compact_target_points,
        query_points,
        max_distance,
    )
    has_far_queries = not bool(warp_query_mask.all())
    target_faces_i32 = compact_faces.to(torch.int32).contiguous()

    wp_device, wp_stream = FunctionSpec.warp_launch_context(target_points_f32)
    with FunctionSpec.warp_stream_scope(wp_stream):
        wp_target_points = wp.from_torch(target_points_f32, dtype=wp.vec3f)
        wp_target_faces = wp.from_torch(target_faces_i32, dtype=wp.int32)
        mesh = wp.Mesh(points=wp_target_points, indices=wp_target_faces)
        wp.launch(
            nearest_surface_faces_kernel,
            dim=query_points.shape[0],
            inputs=[
                mesh.id,
                wp.from_torch(query_points_f32, dtype=wp.vec3f),
                wp.from_torch(max_distance_f32, dtype=wp.float32),
                wp.from_torch(face_ids, dtype=wp.int64),
            ],
            device=wp_device,
            stream=wp_stream,
        )
    hit = face_ids >= 0
    mapped_face_ids = torch.where(
        hit,
        valid_face_ids[face_ids.clamp_min(0)],
        face_ids,
    )
    if has_far_queries:
        fallback_max_distance = float(max_distance.detach())
        far_query_mask = ~warp_query_mask
        far_face_ids = nearest_surface_faces_torch(
            query_points[far_query_mask],
            target_points,
            target_faces,
            fallback_max_distance,
        )
        mapped_face_ids[far_query_mask] = far_face_ids
    return mapped_face_ids


@nearest_surface_faces_warp_impl.register_fake
def _nearest_surface_faces_warp_fake(
    target_points: torch.Tensor,
    target_faces: torch.Tensor,
    query_points: torch.Tensor,
    max_distance: torch.Tensor,
) -> torch.Tensor:
    _ = target_points, target_faces, max_distance
    return torch.empty(
        (query_points.shape[0],),
        dtype=torch.int64,
        device=query_points.device,
    )


def nearest_surface_faces_warp(
    target_points: torch.Tensor,
    target_faces: torch.Tensor,
    query_points: torch.Tensor,
    max_distance: float,
) -> torch.Tensor:
    """Return closest target-face indices, with ``-1`` marking misses."""

    max_distance_tensor = torch.tensor(
        max_distance,
        dtype=target_points.dtype,
        device=target_points.device,
    )
    return nearest_surface_faces_warp_impl(
        target_points,
        target_faces,
        query_points,
        max_distance_tensor,
    )


__all__ = ["nearest_surface_faces_warp", "nearest_surface_faces_warp_impl"]
