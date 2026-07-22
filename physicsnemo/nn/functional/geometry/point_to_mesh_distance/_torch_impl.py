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

"""Pure-Torch point-to-triangle-mesh distance implementation."""

from __future__ import annotations

from math import isqrt

import torch

from ._utils import normalize_point_to_mesh_inputs

# Limit the live pairwise point/triangle intermediates produced by the Ericson
# region-classification table. The factor includes vector-valued differences,
# scalar dot products, projected points, barycentrics, and boolean masks.
_PAIRWISE_TEMPORARY_BYTE_BUDGET = 256 * 1024 * 1024
_PAIRWISE_LIVE_VALUE_FACTOR = 48


def _closest_point_on_triangles(
    query: torch.Tensor,
    triangles: torch.Tensor,
) -> torch.Tensor:
    """Project paired query points onto closed triangles.

    ``query`` and ``triangles`` may have any broadcast-compatible leading
    dimensions and must end in ``(3,)`` and ``(3, 3)`` respectively.

    The implementation follows the Voronoi-region classification from
    Ericson, *Real-Time Collision Detection*. Region predicates are discrete;
    inside a selected face, edge, or vertex region, the returned values retain
    ordinary Torch gradients through both inputs.
    """

    a = triangles[..., 0, :]
    b = triangles[..., 1, :]
    c = triangles[..., 2, :]

    ab = b - a
    ac = c - a
    ap = query - a

    d1 = (ab * ap).sum(dim=-1)
    d2 = (ac * ap).sum(dim=-1)

    bp = query - b
    d3 = (ab * bp).sum(dim=-1)
    d4 = (ac * bp).sum(dim=-1)

    cp = query - c
    d5 = (ab * cp).sum(dim=-1)
    d6 = (ac * cp).sum(dim=-1)

    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4

    tiny = torch.finfo(query.dtype).tiny
    face_denom = (va + vb + vc).clamp_min(tiny)
    face_v = vb / face_denom
    face_w = vc / face_denom
    closest = a + ab * face_v.unsqueeze(-1) + ac * face_w.unsqueeze(-1)

    # Vertex A.
    mask_a = (d1 <= 0.0) & (d2 <= 0.0)
    closest = torch.where(mask_a.unsqueeze(-1), a, closest)

    # Vertex B.
    mask_b = (d3 >= 0.0) & (d4 <= d3)
    closest = torch.where(mask_b.unsqueeze(-1), b, closest)

    # Vertex C.
    mask_c = (d6 >= 0.0) & (d5 <= d6)
    closest = torch.where(mask_c.unsqueeze(-1), c, closest)

    # Edge AB.
    mask_ab = (vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0) & ~mask_a & ~mask_b
    edge_ab_t = (d1 / (d1 - d3).clamp_min(tiny)).clamp(0.0, 1.0)
    projected_ab = a + ab * edge_ab_t.unsqueeze(-1)
    closest = torch.where(mask_ab.unsqueeze(-1), projected_ab, closest)

    # Edge AC.
    mask_ac = (vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0) & ~mask_a & ~mask_c
    edge_ac_t = (d2 / (d2 - d6).clamp_min(tiny)).clamp(0.0, 1.0)
    projected_ac = a + ac * edge_ac_t.unsqueeze(-1)
    closest = torch.where(mask_ac.unsqueeze(-1), projected_ac, closest)

    # Edge BC.
    mask_bc = (va <= 0.0) & ((d4 - d3) >= 0.0) & ((d5 - d6) >= 0.0) & ~mask_b & ~mask_c
    edge_bc_denom = ((d4 - d3) + (d5 - d6)).clamp_min(tiny)
    edge_bc_t = ((d4 - d3) / edge_bc_denom).clamp(0.0, 1.0)
    projected_bc = b + (c - b) * edge_bc_t.unsqueeze(-1)
    closest = torch.where(mask_bc.unsqueeze(-1), projected_bc, closest)

    return closest


def _chunk_sizes(
    num_queries: int,
    num_faces: int,
    element_size: int,
) -> tuple[int, int]:
    """Choose query and face chunks within the pairwise memory budget."""

    pair_budget = max(
        1,
        _PAIRWISE_TEMPORARY_BYTE_BUDGET
        // _PAIRWISE_LIVE_VALUE_FACTOR
        // max(element_size, 1),
    )
    query_chunk = min(num_queries, max(1, isqrt(pair_budget)))
    face_chunk = min(num_faces, max(1, pair_budget // query_chunk))
    if face_chunk == num_faces:
        query_chunk = min(num_queries, max(1, pair_budget // num_faces))
    return query_chunk, face_chunk


@torch.no_grad()
def _nearest_face_indices_torch(
    mesh_vertices: torch.Tensor,
    mesh_indices: torch.Tensor,
    input_points: torch.Tensor,
) -> torch.Tensor:
    """Find each query's nearest triangle with a chunked exhaustive search.

    Correspondence selection intentionally runs without autograd. Hard nearest
    face identity is discrete; after selection, the public implementation
    recomputes the winning point-to-triangle projection from live tensors.
    """

    num_queries = input_points.shape[0]
    num_faces = mesh_indices.shape[0]
    if num_queries == 0:
        return torch.empty(0, dtype=torch.long, device=input_points.device)

    detached_queries = input_points.detach()
    detached_triangles = mesh_vertices.detach()[mesh_indices]
    query_chunk, face_chunk = _chunk_sizes(
        num_queries,
        num_faces,
        detached_queries.element_size(),
    )
    nearest_faces = torch.empty(
        num_queries, dtype=torch.long, device=input_points.device
    )

    for query_start in range(0, num_queries, query_chunk):
        query_end = min(query_start + query_chunk, num_queries)
        query = detached_queries[query_start:query_end]
        best_distance = torch.full(
            (query.shape[0],),
            float("inf"),
            dtype=query.dtype,
            device=query.device,
        )
        best_face = torch.zeros(query.shape[0], dtype=torch.long, device=query.device)

        for face_start in range(0, num_faces, face_chunk):
            face_end = min(face_start + face_chunk, num_faces)
            triangles = detached_triangles[face_start:face_end]
            paired_queries = query.unsqueeze(1).expand(-1, triangles.shape[0], -1)
            paired_triangles = triangles.unsqueeze(0).expand(query.shape[0], -1, -1, -1)
            closest = _closest_point_on_triangles(paired_queries, paired_triangles)
            residual = paired_queries - closest
            distance_squared = (residual * residual).sum(dim=-1)
            chunk_distance, chunk_face = distance_squared.min(dim=1)

            improved = chunk_distance < best_distance
            best_distance = torch.where(improved, chunk_distance, best_distance)
            best_face = torch.where(improved, chunk_face + face_start, best_face)

        nearest_faces[query_start:query_end] = best_face

    return nearest_faces


def point_to_mesh_distance_torch(
    mesh_vertices: torch.Tensor,
    mesh_indices: torch.Tensor,
    input_points: torch.Tensor,
    *,
    squared: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute unsigned distance and closest surface points with Torch.

    The exhaustive correspondence search is detached, then the selected
    point-to-triangle projection is recomputed from the original tensors. This
    gives the standard almost-everywhere derivative of hard closest-surface
    distance without retaining a pairwise autograd graph.
    """

    vertices, faces, queries, input_shape = normalize_point_to_mesh_inputs(
        mesh_vertices, mesh_indices, input_points, squared
    )
    nearest_faces = _nearest_face_indices_torch(vertices, faces, queries)
    return _point_to_mesh_distance_from_face_indices(
        vertices,
        faces,
        queries,
        input_shape,
        nearest_faces,
        squared=squared,
    )


def _point_to_mesh_distance_from_face_indices(
    mesh_vertices: torch.Tensor,
    mesh_indices: torch.Tensor,
    input_points: torch.Tensor,
    input_shape: torch.Size,
    nearest_faces: torch.Tensor,
    *,
    squared: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recompute selected projections from live tensors and restore query rank.

    ``nearest_faces`` is a discrete correspondence result from any search
    backend. Keeping this common tail in native Torch gives every backend the
    same continuous gradient contract through queries and target vertices.
    """

    selected_triangles = mesh_vertices[mesh_indices[nearest_faces]]
    closest_points = _closest_point_on_triangles(input_points, selected_triangles)
    residual = input_points - closest_points
    if squared:
        distance = (residual * residual).sum(dim=-1)
    else:
        distance = torch.linalg.vector_norm(residual, dim=-1)

    return (
        distance.reshape(input_shape[:-1]),
        closest_points.reshape(input_shape),
    )


__all__ = ["point_to_mesh_distance_torch"]
