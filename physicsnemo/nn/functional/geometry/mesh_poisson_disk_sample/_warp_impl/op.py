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
import warnings
from collections.abc import Sequence

import numpy as np
import torch
import warp as wp

from physicsnemo.core.function_spec import FunctionSpec

from ._kernels import (
    _commit_accepted_candidates,
    _count_wse_neighbors,
    _generate_surface_candidates,
    _initialize_wse_weights_from_csr,
    _mark_wse_deleted_batch,
    _reject_candidates_vs_accepted,
    _resolve_candidate_conflicts,
    _subtract_deleted_wse_contribution_batch_csr,
    _write_wse_csr,
)

wp.init()
wp.config.quiet = True

_DART_THROWING_MODE = "dart_throwing"
_WEIGHTED_SAMPLE_ELIMINATION_MODE = "weighted_sample_elimination"
_VALID_MODES = {
    _DART_THROWING_MODE,
    _WEIGHTED_SAMPLE_ELIMINATION_MODE,
}
_WSE_DELETE_BATCH_SIZE_MIN = 8
_WSE_DELETE_BATCH_SIZE_MAX = 128
_WSE_DELETE_CANDIDATE_POOL_MIN = 1024
_WSE_DELETE_CANDIDATE_POOL_PER_DELETE = 64
_WSE_OPEN3D_INIT_FACTOR = 5


# ----------------------------------------------------------------------------
# Input Normalization Helpers
# ----------------------------------------------------------------------------
def _normalize_mesh_indices(
    mesh_indices: torch.Tensor,
    *,
    n_vertices: int | None = None,
) -> torch.Tensor:
    # Mesh connectivity must use integer dtype.
    if mesh_indices.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        raise TypeError("mesh_indices must use an integer dtype")

    # Accept either flattened indices or (n_faces, 3) connectivity.
    if mesh_indices.ndim == 2:
        if mesh_indices.shape[-1] != 3:
            raise ValueError("mesh_indices with rank 2 must have shape (n_faces, 3)")
        mesh_indices = mesh_indices.reshape(-1)
    elif mesh_indices.ndim != 1:
        raise ValueError(
            "mesh_indices must be either rank-1 flattened indices or rank-2 (n_faces, 3)"
        )

    # Flattened connectivity must contain complete triangle triplets.
    if mesh_indices.numel() == 0 or mesh_indices.numel() % 3 != 0:
        raise ValueError(
            "mesh_indices must contain a positive number of triangle-triplet indices"
        )

    # Validate index bounds when vertex count is provided.
    if n_vertices is not None:
        min_index = int(mesh_indices.min().item())
        max_index = int(mesh_indices.max().item())
        if min_index < 0 or max_index >= n_vertices:
            raise ValueError("mesh_indices values must satisfy 0 <= index < n_vertices")
    return mesh_indices


def _normalize_hash_grid_resolution(
    hash_grid_resolution: int | Sequence[int] | torch.Tensor,
) -> tuple[int, int, int]:
    # Accept scalar or explicit 3D grid resolution.
    if isinstance(hash_grid_resolution, int):
        resolution = (
            int(hash_grid_resolution),
            int(hash_grid_resolution),
            int(hash_grid_resolution),
        )
    elif torch.is_tensor(hash_grid_resolution):
        if hash_grid_resolution.ndim != 1 or hash_grid_resolution.numel() != 3:
            raise ValueError("hash_grid_resolution tensor must have exactly 3 elements")
        resolution = (
            int(hash_grid_resolution[0].item()),
            int(hash_grid_resolution[1].item()),
            int(hash_grid_resolution[2].item()),
        )
    else:
        if len(hash_grid_resolution) != 3:
            raise ValueError("hash_grid_resolution must contain exactly 3 values")
        resolution = (
            int(hash_grid_resolution[0]),
            int(hash_grid_resolution[1]),
            int(hash_grid_resolution[2]),
        )

    # Resolution values must be positive.
    if resolution[0] <= 0 or resolution[1] <= 0 or resolution[2] <= 0:
        raise ValueError("hash_grid_resolution values must be strictly positive")
    return resolution


# Compute an adaptive batch target for weighted sample elimination.
def _wse_target_batch_size(*, delete_count: int, steps_done: int) -> int:
    remaining = max(delete_count - steps_done, 0)
    if remaining <= 0:
        return 0

    # Start with larger batches and taper down as the pool shrinks.
    remaining_fraction = float(remaining) / float(delete_count)
    scheduled = _WSE_DELETE_BATCH_SIZE_MIN + int(
        math.ceil(
            (_WSE_DELETE_BATCH_SIZE_MAX - _WSE_DELETE_BATCH_SIZE_MIN)
            * remaining_fraction
        )
    )
    return max(
        1,
        min(
            remaining,
            max(_WSE_DELETE_BATCH_SIZE_MIN, scheduled),
        ),
    )


def _normalize_per_vertex_radius(
    per_vertex_radius: torch.Tensor | None,
    *,
    n_vertices: int,
    device: torch.device,
) -> torch.Tensor:
    # Missing adaptive radius input means constant-radius mode.
    if per_vertex_radius is None:
        return torch.empty(0, device=device, dtype=torch.float32)

    # Validate shape and dtype for adaptive radii.
    if per_vertex_radius.ndim != 1:
        raise ValueError("per_vertex_radius must be rank-1 with shape (n_vertices,)")
    if per_vertex_radius.shape[0] != n_vertices:
        raise ValueError("per_vertex_radius must have shape (n_vertices,)")
    if per_vertex_radius.dtype not in {
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    }:
        raise TypeError("per_vertex_radius must use a floating dtype")

    per_vertex_radius = per_vertex_radius.to(device=device, dtype=torch.float32)
    if float(per_vertex_radius.min().item()) <= 0.0:
        raise ValueError("per_vertex_radius values must be strictly positive")
    return per_vertex_radius.contiguous()


def _create_hash_grid(
    *,
    points: torch.Tensor,
    search_radius: float,
    resolution: tuple[int, int, int],
    device: str,
) -> wp.HashGrid:
    # Build a hash grid for neighbor queries on the provided points.
    hash_grid = wp.HashGrid(
        dim_x=resolution[0],
        dim_y=resolution[1],
        dim_z=resolution[2],
        device=device,
    )
    if points.shape[0] > 0:
        hash_grid.reserve(points.shape[0])
        # Match Warp guidance: use a build cell size close to query radius.
        hash_grid.build(
            points=wp.from_torch(points, dtype=wp.vec3f), radius=search_radius
        )
    return hash_grid


# ----------------------------------------------------------------------------
# Weighted Elimination Implementation
# ----------------------------------------------------------------------------
def _generate_uniform_surface_samples_warp(
    *,
    tri_vertices: torch.Tensor,
    tri_edge1: torch.Tensor,
    tri_edge2: torch.Tensor,
    mesh_indices: torch.Tensor,
    area_cdf: torch.Tensor,
    num_samples: int,
    random_seed: int,
) -> torch.Tensor:
    # Generate one oversampled uniform point set on the mesh surface.
    sample_positions = torch.empty(
        (num_samples, 3),
        device=tri_vertices.device,
        dtype=torch.float32,
    )
    sample_radii = torch.empty(
        (num_samples,),
        device=tri_vertices.device,
        dtype=torch.float32,
    )
    sample_priorities = torch.empty(
        (num_samples,),
        device=tri_vertices.device,
        dtype=torch.float32,
    )
    empty_radius = torch.empty(
        (0,),
        device=tri_vertices.device,
        dtype=torch.float32,
    )

    wp_launch_device, wp_launch_stream = FunctionSpec.warp_launch_context(tri_vertices)
    with wp.ScopedStream(wp_launch_stream):
        wp.launch(
            kernel=_generate_surface_candidates,
            dim=num_samples,
            inputs=[
                wp.from_torch(tri_vertices, dtype=wp.vec3f, return_ctype=True),
                wp.from_torch(tri_edge1, dtype=wp.vec3f, return_ctype=True),
                wp.from_torch(tri_edge2, dtype=wp.vec3f, return_ctype=True),
                wp.from_torch(mesh_indices, dtype=wp.int32, return_ctype=True),
                wp.from_torch(area_cdf, dtype=wp.float32, return_ctype=True),
                wp.from_torch(empty_radius, dtype=wp.float32, return_ctype=True),
                1.0,
                int(random_seed),
                wp.from_torch(sample_positions, dtype=wp.vec3f, return_ctype=True),
                wp.from_torch(sample_radii, dtype=wp.float32, return_ctype=True),
                wp.from_torch(sample_priorities, dtype=wp.float32, return_ctype=True),
            ],
            device=wp_launch_device,
            stream=wp_launch_stream,
        )
    return sample_positions.contiguous()


def _weighted_sample_elimination_warp(
    *,
    sample_positions: torch.Tensor,
    target_num_points: int,
    surface_area: float,
    hash_grid_resolution: tuple[int, int, int],
) -> torch.Tensor:
    # Early return when no elimination is required.
    num_samples = sample_positions.shape[0]
    if target_num_points >= num_samples:
        return sample_positions.contiguous()

    # Match Open3D's Yuksel elimination constants.
    alpha = 8.0
    beta = 0.65
    gamma = 1.5
    ratio = float(target_num_points) / float(num_samples)
    r_max = 2.0 * math.sqrt(
        (surface_area / float(target_num_points)) / (2.0 * math.sqrt(3.0))
    )
    r_min = max(r_max * beta * (1.0 - math.pow(ratio, gamma)), 0.0)

    deleted = torch.zeros(
        (num_samples,),
        device=sample_positions.device,
        dtype=torch.int32,
    )
    weights = torch.empty(
        (num_samples,),
        device=sample_positions.device,
        dtype=torch.float32,
    )

    # Build one static neighborhood structure for elimination updates.
    sample_grid = _create_hash_grid(
        points=sample_positions,
        search_radius=r_max,
        resolution=hash_grid_resolution,
        device=str(sample_positions.device),
    )

    wp_launch_device, wp_launch_stream = FunctionSpec.warp_launch_context(
        sample_positions
    )
    with wp.ScopedStream(wp_launch_stream):
        wp_sample_positions = wp.from_torch(
            sample_positions,
            dtype=wp.vec3f,
            return_ctype=True,
        )
        wp_deleted = wp.from_torch(
            deleted,
            dtype=wp.int32,
            return_ctype=True,
        )
        wp_weights = wp.from_torch(
            weights,
            dtype=wp.float32,
            return_ctype=True,
        )

        # Build weighted-elimination CSR neighbors once.
        neighbor_counts = torch.empty(
            (num_samples,),
            device=sample_positions.device,
            dtype=torch.int32,
        )
        row_ptr = torch.empty(
            (num_samples + 1,),
            device=sample_positions.device,
            dtype=torch.int32,
        )
        wp_neighbor_counts = wp.from_torch(
            neighbor_counts,
            dtype=wp.int32,
            return_ctype=True,
        )
        wp.launch(
            kernel=_count_wse_neighbors,
            dim=num_samples,
            inputs=[
                sample_grid.id,
                wp_sample_positions,
                float(r_max),
                wp_neighbor_counts,
            ],
            device=wp_launch_device,
            stream=wp_launch_stream,
        )
        row_ptr[0] = 0
        torch.cumsum(neighbor_counts, dim=0, out=row_ptr[1:])
        total_edges = int(row_ptr[-1].item())
        max_row_size = int(neighbor_counts.max().item()) if num_samples > 0 else 0
        row_ptr_cpu = row_ptr.detach().cpu().numpy()

        col_idx = torch.empty(
            (total_edges,),
            device=sample_positions.device,
            dtype=torch.int32,
        )
        pair_weights = torch.empty(
            (total_edges,),
            device=sample_positions.device,
            dtype=torch.float32,
        )
        wp_row_ptr = wp.from_torch(
            row_ptr,
            dtype=wp.int32,
            return_ctype=True,
        )
        wp_col_idx = wp.from_torch(
            col_idx,
            dtype=wp.int32,
            return_ctype=True,
        )
        wp_pair_weights = wp.from_torch(
            pair_weights,
            dtype=wp.float32,
            return_ctype=True,
        )
        if total_edges > 0:
            wp.launch(
                kernel=_write_wse_csr,
                dim=num_samples,
                inputs=[
                    sample_grid.id,
                    wp_sample_positions,
                    wp_row_ptr,
                    float(r_min),
                    float(r_max),
                    float(alpha),
                    wp_col_idx,
                    wp_pair_weights,
                ],
                device=wp_launch_device,
                stream=wp_launch_stream,
            )
        col_idx_cpu = col_idx.detach().cpu().numpy()

        # Initialize all sample weights from CSR pair sums.
        wp.launch(
            kernel=_initialize_wse_weights_from_csr,
            dim=num_samples,
            inputs=[
                wp_row_ptr,
                wp_pair_weights,
                wp_deleted,
                wp_weights,
            ],
            device=wp_launch_device,
            stream=wp_launch_stream,
        )

        # Remove highest-weight samples until target size is reached.
        delete_count = num_samples - target_num_points
        deleted_batch = torch.empty(
            (_WSE_DELETE_BATCH_SIZE_MAX,),
            device=sample_positions.device,
            dtype=torch.int32,
        )
        wp_deleted_batch = wp.from_torch(
            deleted_batch,
            dtype=wp.int32,
            return_ctype=True,
        )
        selected_batch_host = torch.empty(
            (_WSE_DELETE_BATCH_SIZE_MAX,),
            dtype=torch.int32,
            pin_memory=sample_positions.is_cuda,
        )
        selected_batch_host_np = selected_batch_host.numpy()
        deleted_cpu = np.zeros((num_samples,), dtype=np.uint8)
        blocked_epoch = np.zeros((num_samples,), dtype=np.int32)
        current_epoch = 1
        neg_inf = -1.0e30
        steps_done = 0
        while steps_done < delete_count:
            # Select a high-weight candidate pool, then greedily form an independent batch.
            target_batch = _wse_target_batch_size(
                delete_count=delete_count,
                steps_done=steps_done,
            )
            pool_k = min(
                num_samples,
                max(
                    _WSE_DELETE_CANDIDATE_POOL_MIN,
                    target_batch * _WSE_DELETE_CANDIDATE_POOL_PER_DELETE,
                ),
            )
            candidate_indices = (
                torch.topk(weights, k=pool_k, largest=True, sorted=True)
                .indices.detach()
                .cpu()
                .numpy()
            )

            current_epoch += 1
            if current_epoch >= np.iinfo(np.int32).max:
                blocked_epoch.fill(0)
                current_epoch = 1

            batch_count = 0
            for candidate_idx in candidate_indices:
                candidate_idx = int(candidate_idx)
                if deleted_cpu[candidate_idx] != 0:
                    continue
                if blocked_epoch[candidate_idx] == current_epoch:
                    continue

                selected_batch_host_np[batch_count] = candidate_idx
                batch_count += 1
                if batch_count >= target_batch:
                    break

                blocked_epoch[candidate_idx] = current_epoch
                start = int(row_ptr_cpu[candidate_idx])
                end = int(row_ptr_cpu[candidate_idx + 1])
                blocked_epoch[col_idx_cpu[start:end]] = current_epoch

            # Fallback guard: always delete at least one point.
            if batch_count == 0:
                selected_batch_host_np[0] = int(candidate_indices[0])
                batch_count = 1

            deleted_cpu[selected_batch_host_np[:batch_count]] = 1
            deleted_batch[:batch_count].copy_(
                selected_batch_host[:batch_count],
                non_blocking=sample_positions.is_cuda,
            )
            wp.launch(
                kernel=_mark_wse_deleted_batch,
                dim=batch_count,
                inputs=[
                    wp_deleted_batch,
                    int(batch_count),
                    wp_deleted,
                    wp_weights,
                    float(neg_inf),
                ],
                device=wp_launch_device,
                stream=wp_launch_stream,
            )

            if max_row_size > 0:
                wp.launch(
                    kernel=_subtract_deleted_wse_contribution_batch_csr,
                    dim=batch_count * max_row_size,
                    inputs=[
                        wp_row_ptr,
                        wp_col_idx,
                        wp_pair_weights,
                        wp_deleted_batch,
                        int(batch_count),
                        int(max_row_size),
                        wp_deleted,
                        wp_weights,
                    ],
                    device=wp_launch_device,
                    stream=wp_launch_stream,
                )

            steps_done += batch_count

    kept_indices = torch.nonzero(deleted == 0, as_tuple=False).squeeze(1)
    if kept_indices.numel() > target_num_points:
        kept_indices = kept_indices[:target_num_points]
    return sample_positions.index_select(0, kept_indices).contiguous()


def _mesh_poisson_disk_sample_warp(
    mesh_vertices: torch.Tensor,
    mesh_indices: torch.Tensor,
    min_distance: float,
    per_vertex_radius: torch.Tensor | None = None,
    batch_size: int = 131072,
    max_points: int = 2_000_000,
    max_iterations: int = 64,
    random_seed: int = 42,
    hash_grid_resolution: int | Sequence[int] | torch.Tensor = 128,
    mode: str = _DART_THROWING_MODE,
    target_num_points: int | None = None,
) -> torch.Tensor:
    # Validate mesh and scalar arguments.
    if mesh_vertices.ndim != 2 or mesh_vertices.shape[-1] != 3:
        raise ValueError("mesh_vertices must have shape (n_vertices, 3)")
    if min_distance <= 0.0:
        raise ValueError("min_distance must be strictly positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be strictly positive")
    if max_points <= 0:
        raise ValueError("max_points must be strictly positive")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be strictly positive")

    # Normalize the Poisson sampling mode.
    if mode not in _VALID_MODES:
        raise ValueError(
            "mode must be one of {'dart_throwing', 'weighted_sample_elimination'}"
        )

    mesh_vertices = mesh_vertices.to(dtype=torch.float32).contiguous()
    mesh_indices = _normalize_mesh_indices(
        mesh_indices,
        n_vertices=mesh_vertices.shape[0],
    ).to(device=mesh_vertices.device, dtype=torch.int32, copy=False)
    per_vertex_radius = _normalize_per_vertex_radius(
        per_vertex_radius,
        n_vertices=mesh_vertices.shape[0],
        device=mesh_vertices.device,
    )
    grid_resolution = _normalize_hash_grid_resolution(hash_grid_resolution)

    # Build area-weighted triangle CDF and triangle geometry tensors.
    tri = mesh_indices.reshape(-1, 3).to(torch.long)
    p0 = mesh_vertices[tri[:, 0]]
    p1 = mesh_vertices[tri[:, 1]]
    p2 = mesh_vertices[tri[:, 2]]
    tri_edge1 = (p1 - p0).contiguous()
    tri_edge2 = (p2 - p0).contiguous()
    tri_vertices = p0.contiguous()

    areas = 0.5 * torch.linalg.norm(torch.cross(tri_edge1, tri_edge2, dim=1), dim=1)
    total_area = float(areas.sum().item())
    if total_area <= 0.0:
        raise ValueError("mesh triangle areas must sum to a positive value")
    area_cdf = (torch.cumsum(areas, dim=0) / total_area).to(torch.float32).contiguous()

    # Weighted elimination mode: oversample uniformly then run Warp elimination.
    if mode == _WEIGHTED_SAMPLE_ELIMINATION_MODE:
        if target_num_points is None:
            target_num_points = max_points
        if target_num_points <= 0:
            raise ValueError("target_num_points must be strictly positive")

        # Match Open3D's initialization behavior: init_factor * target samples.
        pool_target = max(
            target_num_points + 1,
            int(round(target_num_points * _WSE_OPEN3D_INIT_FACTOR)),
        )

        if per_vertex_radius.numel() > 0:
            warnings.warn(
                "per_vertex_radius is ignored in weighted_sample_elimination mode",
                stacklevel=2,
            )

        pool_positions = _generate_uniform_surface_samples_warp(
            tri_vertices=tri_vertices,
            tri_edge1=tri_edge1,
            tri_edge2=tri_edge2,
            mesh_indices=mesh_indices,
            area_cdf=area_cdf,
            num_samples=pool_target,
            random_seed=random_seed,
        )
        output = _weighted_sample_elimination_warp(
            sample_positions=pool_positions,
            target_num_points=target_num_points,
            surface_area=total_area,
            hash_grid_resolution=grid_resolution,
        )
        return output

    # Allocate accepted/candidate buffers reused throughout dart throwing.
    accepted_positions = torch.empty(
        (max_points, 3),
        device=mesh_vertices.device,
        dtype=torch.float32,
    )
    accepted_radii = torch.empty(
        (max_points,), device=mesh_vertices.device, dtype=torch.float32
    )
    accepted_count = torch.zeros((1,), device=mesh_vertices.device, dtype=torch.int32)

    candidate_positions = torch.empty(
        (batch_size, 3),
        device=mesh_vertices.device,
        dtype=torch.float32,
    )
    candidate_radii = torch.empty(
        (batch_size,), device=mesh_vertices.device, dtype=torch.float32
    )
    candidate_priorities = torch.empty(
        (batch_size,),
        device=mesh_vertices.device,
        dtype=torch.float32,
    )
    candidate_alive = torch.ones(
        (batch_size,), device=mesh_vertices.device, dtype=torch.int32
    )

    # Cache adaptive-radius maximum once for dynamic search radius updates.
    adaptive_max_radius = (
        float(per_vertex_radius.max().item()) if per_vertex_radius.numel() > 0 else 0.0
    )

    wp_launch_device, wp_launch_stream = FunctionSpec.warp_launch_context(mesh_vertices)
    with wp.ScopedStream(wp_launch_stream):
        # Convert static input tensors once for repeated kernel launches.
        wp_triangle_vertices = wp.from_torch(
            tri_vertices, dtype=wp.vec3f, return_ctype=True
        )
        wp_triangle_edge1 = wp.from_torch(tri_edge1, dtype=wp.vec3f, return_ctype=True)
        wp_triangle_edge2 = wp.from_torch(tri_edge2, dtype=wp.vec3f, return_ctype=True)
        wp_triangle_vertex_indices = wp.from_torch(
            mesh_indices,
            dtype=wp.int32,
            return_ctype=True,
        )
        wp_area_cdf = wp.from_torch(area_cdf, dtype=wp.float32, return_ctype=True)
        wp_per_vertex_radius = wp.from_torch(
            per_vertex_radius,
            dtype=wp.float32,
            return_ctype=True,
        )

        # Convert mutable buffers reused each iteration.
        wp_candidate_positions = wp.from_torch(
            candidate_positions,
            dtype=wp.vec3f,
            return_ctype=True,
        )
        wp_candidate_radii = wp.from_torch(
            candidate_radii,
            dtype=wp.float32,
            return_ctype=True,
        )
        wp_candidate_priorities = wp.from_torch(
            candidate_priorities,
            dtype=wp.float32,
            return_ctype=True,
        )
        wp_candidate_alive = wp.from_torch(
            candidate_alive,
            dtype=wp.int32,
            return_ctype=True,
        )
        wp_accepted_positions = wp.from_torch(
            accepted_positions,
            dtype=wp.vec3f,
            return_ctype=True,
        )
        wp_accepted_radii = wp.from_torch(
            accepted_radii,
            dtype=wp.float32,
            return_ctype=True,
        )
        wp_accepted_count = wp.from_torch(
            accepted_count,
            dtype=wp.int32,
            return_ctype=True,
        )
        wp_candidate_positions_array = wp.from_torch(
            candidate_positions, dtype=wp.vec3f
        )

        # Reuse hash-grid objects across iterations to reduce object churn.
        accepted_grid = wp.HashGrid(
            dim_x=grid_resolution[0],
            dim_y=grid_resolution[1],
            dim_z=grid_resolution[2],
            device=str(mesh_vertices.device),
        )
        accepted_grid.reserve(max_points)
        candidate_grid = wp.HashGrid(
            dim_x=grid_resolution[0],
            dim_y=grid_resolution[1],
            dim_z=grid_resolution[2],
            device=str(mesh_vertices.device),
        )
        candidate_grid.reserve(batch_size)

        def _run_dart_throwing_pass(
            *,
            pass_distance: float,
            pass_seed: int,
            pass_limit: int,
        ) -> int:
            accepted_count.zero_()
            current_count = 0

            # Main iterative parallel dart-throwing loop.
            for iteration in range(max_iterations):
                candidate_alive.fill_(1)

                # Generate one candidate batch on the mesh surface.
                wp.launch(
                    kernel=_generate_surface_candidates,
                    dim=batch_size,
                    inputs=[
                        wp_triangle_vertices,
                        wp_triangle_edge1,
                        wp_triangle_edge2,
                        wp_triangle_vertex_indices,
                        wp_area_cdf,
                        wp_per_vertex_radius,
                        float(pass_distance),
                        int(pass_seed + iteration * 104729),
                        wp_candidate_positions,
                        wp_candidate_radii,
                        wp_candidate_priorities,
                    ],
                    device=wp_launch_device,
                    stream=wp_launch_stream,
                )

                # Reject candidates near previously accepted points.
                if current_count > 0:
                    search_radius = max(pass_distance, adaptive_max_radius)
                    accepted_view = accepted_positions[:current_count]
                    accepted_grid.build(
                        points=wp.from_torch(accepted_view, dtype=wp.vec3f),
                        radius=search_radius,
                    )
                    wp.launch(
                        kernel=_reject_candidates_vs_accepted,
                        dim=batch_size,
                        inputs=[
                            accepted_grid.id,
                            wp_candidate_positions,
                            wp_candidate_radii,
                            wp_candidate_alive,
                            wp.from_torch(
                                accepted_view,
                                dtype=wp.vec3f,
                                return_ctype=True,
                            ),
                        ],
                        device=wp_launch_device,
                        stream=wp_launch_stream,
                    )

                # Resolve conflicts among this iteration's candidates.
                search_radius = max(pass_distance, adaptive_max_radius)
                candidate_grid.build(
                    points=wp_candidate_positions_array,
                    radius=search_radius,
                )
                wp.launch(
                    kernel=_resolve_candidate_conflicts,
                    dim=batch_size,
                    inputs=[
                        candidate_grid.id,
                        wp_candidate_positions,
                        wp_candidate_radii,
                        wp_candidate_priorities,
                        wp_candidate_alive,
                    ],
                    device=wp_launch_device,
                    stream=wp_launch_stream,
                )

                # Commit surviving candidates to accepted arrays.
                wp.launch(
                    kernel=_commit_accepted_candidates,
                    dim=batch_size,
                    inputs=[
                        wp_candidate_positions,
                        wp_candidate_radii,
                        wp_candidate_alive,
                        wp_accepted_positions,
                        wp_accepted_radii,
                        wp_accepted_count,
                    ],
                    device=wp_launch_device,
                    stream=wp_launch_stream,
                )

                # Read accepted count once per iteration and reuse it next iteration.
                count_after = int(accepted_count[0].item())
                accepted_now = min(count_after, pass_limit) - min(
                    current_count, pass_limit
                )
                current_count = count_after

                # Stop on saturation or no-progress iterations.
                if accepted_now <= 0 or current_count >= pass_limit:
                    break

            return min(current_count, pass_limit)

        # Default mode: direct iterative dart throwing.
        final_count = _run_dart_throwing_pass(
            pass_distance=min_distance,
            pass_seed=random_seed,
            pass_limit=max_points,
        )
        return accepted_positions[:final_count].contiguous()


# Public alias used by the FunctionSpec wrapper.
mesh_poisson_disk_sample_warp = _mesh_poisson_disk_sample_warp


__all__ = [
    "mesh_poisson_disk_sample_warp",
    "_DART_THROWING_MODE",
    "_WEIGHTED_SAMPLE_ELIMINATION_MODE",
]
