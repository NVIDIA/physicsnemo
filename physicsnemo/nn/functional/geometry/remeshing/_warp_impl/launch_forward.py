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

"""Forward launch orchestration for Warp surface remeshing."""

from __future__ import annotations

import math

import torch
import warp as wp

from physicsnemo.core.function_spec import FunctionSpec
from physicsnemo.nn.functional.geometry.farthest_point_sampling import (
    farthest_point_sampling,
)
from physicsnemo.utils._index_tuple_ops import unique_index_tuples

from ._kernels import (
    accumulate_vertex_areas,
    assign_vertices,
    project_centroids_to_surface,
    update_centroids,
)

wp.init()

_WEIGHTED_SAMPLE_CHUNK_SIZE = 1 << 22


def _wp_view(tensor: torch.Tensor, dtype):  # noqa: ANN001, ANN202
    """Return a zero-copy Warp launch descriptor for a detached tensor."""
    return wp.from_torch(
        tensor.detach(),
        dtype=dtype,
        return_ctype=True,
        requires_grad=False,
    )


def _weighted_sample_without_replacement(
    weights: torch.Tensor,
    count: int,
) -> torch.Tensor:
    """Sample indices by an uncapped, chunked exponential race.

    Each chunk contributes its local ``count`` smallest keys. The global
    ``count`` smallest keys must be in that union, so the chunking reduces
    temporary memory without changing the weighted-without-replacement draw.
    """
    if count == weights.shape[0]:
        return torch.arange(weights.shape[0], device=weights.device)

    generator = torch.Generator(device=weights.device).manual_seed(0)
    candidate_keys = []
    candidate_indices = []
    tiny = torch.finfo(weights.dtype).tiny
    for start in range(0, weights.shape[0], _WEIGHTED_SAMPLE_CHUNK_SIZE):
        stop = min(start + _WEIGHTED_SAMPLE_CHUNK_SIZE, weights.shape[0])
        chunk_weights = weights[start:stop].clamp_min(tiny)
        keys = torch.empty_like(chunk_weights).exponential_(generator=generator)
        keys.div_(chunk_weights)
        local_count = min(count, stop - start)
        local_keys, local_indices = torch.topk(
            keys,
            local_count,
            largest=False,
            sorted=False,
        )
        candidate_keys.append(local_keys)
        candidate_indices.append(local_indices + start)

    if len(candidate_keys) == 1:
        return candidate_indices[0]

    keys = torch.cat(candidate_keys)
    indices = torch.cat(candidate_indices)
    selected = torch.topk(keys, count, largest=False, sorted=False).indices
    return indices[selected]


def _select_fps_centroids(
    points: torch.Tensor,
    vertex_areas: torch.Tensor,
    n_clusters: int,
    farthest_point_oversampling: int,
) -> torch.Tensor:
    """Select high-quality seeds with FPS over an area-weighted candidate set."""
    candidate_count = min(points.shape[0], farthest_point_oversampling * n_clusters)
    candidate_indices = _weighted_sample_without_replacement(
        vertex_areas,
        candidate_count,
    )
    candidates = points[candidate_indices]
    selected = farthest_point_sampling(
        candidates,
        n_clusters,
        random_start=False,
    )
    return candidates[selected].clone()


def _voxel_representatives(
    points: torch.Tensor,
    lower_bound: torch.Tensor,
    upper_bound: torch.Tensor,
    cell_width: float,
) -> torch.Tensor:
    """Return one source vertex index from each occupied spatial voxel."""
    coordinates = torch.floor((points - lower_bound) / cell_width)

    # Packing the three coordinates into one int64 key is substantially faster
    # than a row-wise unique for ordinary voxel widths. Very small user-provided
    # widths can exceed that key space, so retain the exact float coordinates
    # for a safe fallback instead of allowing integer overflow and collisions.
    # Point normalization bounds every coordinate to [-1, 1].
    max_dimension = math.floor(2.0 / cell_width) + 1
    if max_dimension**3 <= torch.iinfo(torch.int64).max:
        dimensions = (
            torch.floor((upper_bound - lower_bound) / cell_width).to(torch.int64) + 1
        )
        coordinates_i64 = coordinates.to(torch.int64)
        keys = (
            coordinates_i64[:, 0] * dimensions[1] + coordinates_i64[:, 1]
        ) * dimensions[2] + coordinates_i64[:, 2]
        sorted_keys, order = torch.sort(keys)
        first = torch.ones(points.shape[0], dtype=torch.bool, device=points.device)
        first[1:] = sorted_keys[1:] != sorted_keys[:-1]
        return order[first]

    unique_coordinates, inverse = torch.unique(
        coordinates,
        dim=0,
        return_inverse=True,
    )
    source_indices = torch.arange(
        points.shape[0], device=points.device, dtype=torch.int64
    )
    first = torch.full(
        (unique_coordinates.shape[0],),
        points.shape[0],
        device=points.device,
        dtype=torch.int64,
    )
    first.scatter_reduce_(0, inverse, source_indices, reduce="amin")
    return first


def _select_stratified_centroids(
    points: torch.Tensor,
    vertex_areas: torch.Tensor,
    n_clusters: int,
    total_area: float,
    voxel_width_scale: float,
    lower_bound: torch.Tensor,
    upper_bound: torch.Tensor,
) -> torch.Tensor:
    """Select large seed sets with an O(N log N) spatial stratification.

    One source point per occupied, surface-area-scaled voxel avoids the
    quadratic cost of FPS for large target meshes. A small deterministic fill
    or reduction produces the exact requested seed count.
    """
    cell_width = max(
        voxel_width_scale * math.sqrt(total_area / n_clusters),
        torch.finfo(torch.float32).tiny,
    )
    representatives = _voxel_representatives(
        points,
        lower_bound,
        upper_bound,
        cell_width,
    )
    if representatives.numel() > n_clusters:
        generator = torch.Generator(device=points.device).manual_seed(0)
        selection = torch.randperm(
            representatives.numel(), device=points.device, generator=generator
        )[:n_clusters]
        representatives = representatives[selection]
    elif representatives.numel() < n_clusters:
        # Fill sparse voxelizations from vertices not already selected. The
        # explicit remaining set keeps representatives excluded even when
        # isolated vertices have zero area.
        available = torch.ones(points.shape[0], dtype=torch.bool, device=points.device)
        available[representatives] = False
        remaining = torch.nonzero(available, as_tuple=False).flatten()
        fill = remaining[
            _weighted_sample_without_replacement(
                vertex_areas[remaining],
                n_clusters - representatives.numel(),
            )
        ]
        representatives = torch.cat([representatives, fill])
    return points[representatives].clone()


def _deduplicate_faces(
    mapped_cells: torch.Tensor,
    n_clusters: int,
) -> torch.Tensor:
    """Drop collapsed faces and orientation-independent duplicates."""
    distinct = (
        (mapped_cells[:, 0] != mapped_cells[:, 1])
        & (mapped_cells[:, 1] != mapped_cells[:, 2])
        & (mapped_cells[:, 0] != mapped_cells[:, 2])
    )
    mapped_cells = mapped_cells[distinct]
    if mapped_cells.numel() == 0:
        return mapped_cells

    canonical = torch.sort(mapped_cells, dim=1).values
    unique_faces, inverse = unique_index_tuples(
        canonical,
        n_clusters,
        return_inverse=True,
    )
    n_unique = unique_faces.shape[0]
    source_indices = torch.arange(
        mapped_cells.shape[0], device=mapped_cells.device, dtype=torch.int64
    )
    first = torch.full(
        (n_unique,),
        mapped_cells.shape[0],
        device=mapped_cells.device,
        dtype=torch.int64,
    )
    first.scatter_reduce_(0, inverse, source_indices, reduce="amin")
    return mapped_cells[first]


def _remove_nonmanifold_faces(
    points: torch.Tensor,
    cells: torch.Tensor,
    n_points: int,
) -> torch.Tensor:
    """Remove redundant faces from edges incident to more than two faces.

    Spatial clustering can very occasionally reconstruct an overlapping local
    patch. Such a patch appears as a small set of edges with three or more
    incident faces. For each over-subscribed edge, this cleanup prefers a face
    shared by the largest number of problematic edges, then removes the
    smallest-area candidate. Ordinary manifold outputs exit after one edge
    count pass without changing connectivity.
    """
    while True:
        n_faces = cells.shape[0]
        edges = torch.cat([cells[:, [0, 1]], cells[:, [1, 2]], cells[:, [2, 0]]], dim=0)
        face_indices = torch.arange(
            n_faces, device=cells.device, dtype=torch.int64
        ).repeat(3)
        canonical_edges = torch.sort(edges, dim=1).values
        unique_edges, inverse, counts = unique_index_tuples(
            canonical_edges,
            n_points,
            return_inverse=True,
            return_counts=True,
        )
        bad_incidence = counts[inverse] > 2
        if not bool(bad_incidence.any()):
            return cells

        # Favor one face that resolves several problematic edges at once.
        face_bad_edge_counts = torch.zeros(
            n_faces, dtype=torch.int32, device=cells.device
        )
        face_bad_edge_counts.scatter_add_(
            0, face_indices, bad_incidence.to(torch.int32)
        )
        best_score = torch.zeros(
            unique_edges.shape[0], dtype=torch.int32, device=cells.device
        )
        best_score.scatter_reduce_(
            0,
            inverse,
            face_bad_edge_counts[face_indices],
            reduce="amax",
        )
        candidates = bad_incidence & (
            face_bad_edge_counts[face_indices] == best_score[inverse]
        )

        vertices = points[cells]
        face_areas = torch.linalg.vector_norm(
            torch.linalg.cross(
                vertices[:, 1] - vertices[:, 0],
                vertices[:, 2] - vertices[:, 0],
                dim=1,
            ),
            dim=1,
        )
        candidate_areas = torch.where(
            candidates,
            face_areas[face_indices],
            torch.full_like(face_areas[face_indices], torch.inf),
        )
        smallest_area = torch.full(
            (unique_edges.shape[0],),
            torch.inf,
            dtype=face_areas.dtype,
            device=cells.device,
        )
        smallest_area.scatter_reduce_(0, inverse, candidate_areas, reduce="amin")
        candidates &= candidate_areas == smallest_area[inverse]

        # Break exact-area ties by face index so each problematic edge chooses
        # one face deterministically.
        candidate_faces = torch.where(
            candidates,
            face_indices,
            torch.full_like(face_indices, n_faces),
        )
        selected_faces = torch.full(
            (unique_edges.shape[0],),
            n_faces,
            dtype=torch.int64,
            device=cells.device,
        )
        selected_faces.scatter_reduce_(0, inverse, candidate_faces, reduce="amin")
        remove = torch.zeros(n_faces, dtype=torch.bool, device=cells.device)
        remove[selected_faces[counts > 2]] = True
        cells = cells[~remove]
        if cells.numel() == 0:
            raise RuntimeError("manifold cleanup removed every remeshed face")
        # Removing one incident face resolves every edge with count three.
        # Only higher-order overlaps require another counting pass.
        if not bool((counts > 3).any()):
            return cells


def _build_output_tensors(
    source_points: torch.Tensor,
    source_cells: torch.Tensor,
    centroids: torch.Tensor,
    labels: torch.Tensor,
    n_clusters: int,
    normalization_center: torch.Tensor,
    normalization_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reconstruct, clean, and compact triangle connectivity."""
    mapped_cells = labels.to(torch.int64)[source_cells.to(torch.int64)]
    output_cells = _deduplicate_faces(mapped_cells, n_clusters)
    if output_cells.numel() == 0:
        raise RuntimeError(
            "Warp remeshing collapsed every input face. Request more clusters "
            "or provide a nondegenerate triangle surface."
        )

    # Reject geometric degeneracies after centroid movement and projection.
    vertices = centroids[output_cells]
    doubled_areas = torch.linalg.vector_norm(
        torch.linalg.cross(
            vertices[:, 1] - vertices[:, 0],
            vertices[:, 2] - vertices[:, 0],
            dim=1,
        ),
        dim=1,
    )
    output_cells = output_cells[torch.isfinite(doubled_areas) & (doubled_areas > 0.0)]
    if output_cells.numel() == 0:
        raise RuntimeError(
            "Warp remeshing produced only zero-area faces. Request more clusters."
        )

    output_cells = _remove_nonmanifold_faces(
        centroids,
        output_cells,
        n_clusters,
    )

    used_centroids, compact_cells = torch.unique(
        output_cells.reshape(-1),
        sorted=True,
        return_inverse=True,
    )
    output_points = centroids[used_centroids].to(dtype=normalization_center.dtype)
    output_points = output_points * normalization_scale + normalization_center
    output_points = output_points.to(dtype=source_points.dtype)
    output_cells = compact_cells.reshape(-1, 3).to(dtype=torch.int64)
    return output_points, output_cells


def _normalize_points_for_warp(
    points: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Center and scale coordinates before float32 Warp computation."""
    working_dtype = torch.float64 if points.dtype == torch.float64 else torch.float32
    working_points = points.detach().to(dtype=working_dtype)
    lower_bound = working_points.amin(dim=0)
    upper_bound = working_points.amax(dim=0)

    # Forming the midpoint as two half-sized terms avoids overflow for finite
    # coordinates near the dtype limits. Subtract in the input precision so a
    # large world-space translation does not erase local geometry in float32.
    center = 0.5 * lower_bound + 0.5 * upper_bound
    scale = torch.maximum(
        (lower_bound - center).abs(),
        (upper_bound - center).abs(),
    ).amax()
    safe_scale = torch.where(scale > 0.0, scale, torch.ones_like(scale))
    normalized = ((working_points - center) / safe_scale).to(torch.float32)
    normalized_lower = ((lower_bound - center) / safe_scale).to(torch.float32)
    normalized_upper = ((upper_bound - center) / safe_scale).to(torch.float32)
    return (
        normalized.contiguous(),
        center,
        safe_scale,
        normalized_lower,
        normalized_upper,
    )


def launch_remeshing(
    mesh_vertices: torch.Tensor,
    mesh_indices: torch.Tensor,
    n_clusters: int,
    *,
    max_iterations: int,
    search_radius_scale: float,
    voxel_width_scale: float,
    hash_grid_resolution: int,
    farthest_point_threshold: int,
    farthest_point_oversampling: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Remesh a triangle surface with Warp CVT clustering."""
    (
        points,
        normalization_center,
        normalization_scale,
        lower_bound,
        upper_bound,
    ) = _normalize_points_for_warp(mesh_vertices)
    cells = mesh_indices.detach().to(dtype=torch.int32).contiguous()
    flat_cells = cells.reshape(-1).contiguous()
    n_points = points.shape[0]

    vertex_areas = torch.zeros(n_points, dtype=torch.float32, device=points.device)
    wp_launch_device, wp_launch_stream = FunctionSpec.warp_launch_context(points)
    with FunctionSpec.warp_stream_scope(wp_launch_stream):
        wp.launch(
            accumulate_vertex_areas,
            dim=cells.shape[0],
            inputs=[
                _wp_view(points, wp.vec3f),
                _wp_view(cells, wp.int32),
                _wp_view(vertex_areas, wp.float32),
            ],
            device=wp_launch_device,
            stream=wp_launch_stream,
        )

    total_area = float(vertex_areas.sum().item())
    if not math.isfinite(total_area) or total_area <= 0.0:
        raise ValueError("mesh must have positive finite surface area")

    if n_clusters <= farthest_point_threshold:
        centroids = _select_fps_centroids(
            points,
            vertex_areas,
            n_clusters,
            farthest_point_oversampling,
        )
    else:
        centroids = _select_stratified_centroids(
            points,
            vertex_areas,
            n_clusters,
            total_area,
            voxel_width_scale,
            lower_bound,
            upper_bound,
        )

    labels = torch.empty(n_points, dtype=torch.int32, device=points.device)
    centroid_sums = torch.zeros(
        n_clusters, 3, dtype=torch.float32, device=points.device
    )
    centroid_areas = torch.zeros(n_clusters, dtype=torch.float32, device=points.device)
    search_radius = max(
        search_radius_scale * math.sqrt(total_area / n_clusters),
        torch.finfo(torch.float32).tiny,
    )

    with FunctionSpec.warp_stream_scope(wp_launch_stream):
        wp_points = wp.from_torch(points, dtype=wp.vec3f)
        wp_centroids = wp.from_torch(centroids, dtype=wp.vec3f)
        centroid_grid = wp.HashGrid(
            dim_x=hash_grid_resolution,
            dim_y=hash_grid_resolution,
            dim_z=hash_grid_resolution,
            device=wp_centroids.device,
        )
        centroid_grid.reserve(n_clusters)

        assignment_inputs = [
            centroid_grid.id,
            _wp_view(points, wp.vec3f),
            _wp_view(centroids, wp.vec3f),
            _wp_view(vertex_areas, wp.float32),
            _wp_view(labels, wp.int32),
            _wp_view(centroid_sums, wp.float32),
            _wp_view(centroid_areas, wp.float32),
            search_radius,
        ]

        for _ in range(max_iterations):
            centroid_grid.build(wp_centroids, radius=search_radius)
            wp.launch(
                assign_vertices,
                dim=n_points,
                inputs=[
                    *assignment_inputs,
                    1,
                ],
                device=wp_launch_device,
                stream=wp_launch_stream,
            )
            wp.launch(
                update_centroids,
                dim=n_clusters,
                inputs=[
                    _wp_view(centroids, wp.vec3f),
                    _wp_view(centroid_sums, wp.float32),
                    _wp_view(centroid_areas, wp.float32),
                ],
                device=wp_launch_device,
                stream=wp_launch_stream,
            )

        # A final assignment reflects the last centroid update and supplies the
        # labels used to reconstruct topology.
        centroid_grid.build(wp_centroids, radius=search_radius)
        wp.launch(
            assign_vertices,
            dim=n_points,
            inputs=[
                *assignment_inputs,
                0,
            ],
            device=wp_launch_device,
            stream=wp_launch_stream,
        )

        source_surface = wp.Mesh(
            points=wp_points,
            indices=wp.from_torch(flat_cells, dtype=wp.int32),
        )
        wp.launch(
            project_centroids_to_surface,
            dim=n_clusters,
            inputs=[
                source_surface.id,
                _wp_view(centroids, wp.vec3f),
                float(1.0e30),
            ],
            device=wp_launch_device,
            stream=wp_launch_stream,
        )

    return _build_output_tensors(
        mesh_vertices,
        mesh_indices,
        centroids,
        labels,
        n_clusters,
        normalization_center,
        normalization_scale,
    )


__all__ = ["launch_remeshing"]
