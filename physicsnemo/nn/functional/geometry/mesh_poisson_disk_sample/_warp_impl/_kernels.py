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
#
# Kernel layout:
# 1) Shared Warp helper funcs
# 2) Dart-throwing candidate kernels
# 3) Weighted-sample-elimination kernels

from __future__ import annotations

import warp as wp


# ----------------------------------------------------------------------------
# Shared Warp Helper Functions
# ----------------------------------------------------------------------------
@wp.func
def _binary_search_cdf(cdf: wp.array(dtype=wp.float32), value: wp.float32) -> int:
    left = int(0)
    right = int(cdf.shape[0] - 1)

    while left < right:
        mid = (left + right) >> 1
        if cdf[mid] < value:
            left = mid + 1
        else:
            right = mid
    return left


# Generate uniform random barycentric coordinates via Turk's method.
@wp.func
def _uniform_barycentric_sample(u1: wp.float32, u2: wp.float32) -> wp.vec2f:
    sqrt_u1 = wp.sqrt(u1)
    return wp.vec2f(1.0 - sqrt_u1, u2 * sqrt_u1)


# Compute squared distance between two 3D points.
@wp.func
def _distance_squared(p1: wp.vec3f, p2: wp.vec3f) -> wp.float32:
    diff = p1 - p2
    return wp.dot(diff, diff)


# Check minimum-distance constraint between two points.
@wp.func
def _points_too_close(
    p1: wp.vec3f,
    p2: wp.vec3f,
    min_distance: wp.float32,
) -> bool:
    return _distance_squared(p1, p2) < (min_distance * min_distance)


# Generate candidate points from area-weighted random triangle samples.
@wp.kernel
def _generate_surface_candidates(
    triangle_vertices: wp.array(dtype=wp.vec3f),
    triangle_edge1: wp.array(dtype=wp.vec3f),
    triangle_edge2: wp.array(dtype=wp.vec3f),
    triangle_vertex_indices: wp.array(dtype=wp.int32),
    area_cdf: wp.array(dtype=wp.float32),
    per_vertex_radius: wp.array(dtype=wp.float32),
    constant_radius: wp.float32,
    seed_base: int,
    output_positions: wp.array(dtype=wp.vec3f),
    output_radii: wp.array(dtype=wp.float32),
    output_priorities: wp.array(dtype=wp.float32),
):
    candidate_idx = wp.tid()
    rng_state = wp.rand_init(seed_base, candidate_idx)

    # Sample one triangle from area CDF.
    random_value = wp.randf(rng_state)
    triangle_idx = _binary_search_cdf(area_cdf, random_value)

    # Sample one point uniformly over that triangle.
    u1 = wp.randf(rng_state)
    u2 = wp.randf(rng_state)
    bary = _uniform_barycentric_sample(u1, u2)
    bary_u = bary[0]
    bary_v = bary[1]
    bary_w = 1.0 - bary_u - bary_v

    point = (
        triangle_vertices[triangle_idx]
        + triangle_edge1[triangle_idx] * bary_v
        + triangle_edge2[triangle_idx] * bary_w
    )
    output_positions[candidate_idx] = point
    output_priorities[candidate_idx] = wp.randf(rng_state)

    # Use either constant radius or barycentric interpolation from vertices.
    if per_vertex_radius.shape[0] > 0:
        i0 = triangle_vertex_indices[triangle_idx * 3 + 0]
        i1 = triangle_vertex_indices[triangle_idx * 3 + 1]
        i2 = triangle_vertex_indices[triangle_idx * 3 + 2]
        radius = (
            bary_u * per_vertex_radius[i0]
            + bary_v * per_vertex_radius[i1]
            + bary_w * per_vertex_radius[i2]
        )
        output_radii[candidate_idx] = radius
    else:
        output_radii[candidate_idx] = constant_radius


# Reject candidates that conflict with already-accepted samples.
@wp.kernel
def _reject_candidates_vs_accepted(
    hashgrid_id: wp.uint64,
    candidate_positions: wp.array(dtype=wp.vec3f),
    candidate_radii: wp.array(dtype=wp.float32),
    candidate_alive: wp.array(dtype=wp.int32),
    accepted_positions: wp.array(dtype=wp.vec3f),
):
    candidate_idx = wp.tid()
    if candidate_alive[candidate_idx] == 0:
        return

    candidate_position = candidate_positions[candidate_idx]
    candidate_radius = candidate_radii[candidate_idx]

    neighbor_idx = int(0)
    query = wp.hash_grid_query(hashgrid_id, candidate_position, candidate_radius)
    while wp.hash_grid_query_next(query, neighbor_idx):
        if neighbor_idx < accepted_positions.shape[0]:
            accepted_position = accepted_positions[neighbor_idx]
            if _points_too_close(
                candidate_position, accepted_position, candidate_radius
            ):
                candidate_alive[candidate_idx] = 0
                return


# Resolve conflicts between candidate points with random-priority MIS.
@wp.kernel
def _resolve_candidate_conflicts(
    hashgrid_id: wp.uint64,
    candidate_positions: wp.array(dtype=wp.vec3f),
    candidate_radii: wp.array(dtype=wp.float32),
    candidate_priorities: wp.array(dtype=wp.float32),
    candidate_alive: wp.array(dtype=wp.int32),
):
    candidate_idx = wp.tid()
    if candidate_alive[candidate_idx] == 0:
        return

    candidate_position = candidate_positions[candidate_idx]
    candidate_radius = candidate_radii[candidate_idx]
    candidate_priority = candidate_priorities[candidate_idx]

    neighbor_idx = int(0)
    query = wp.hash_grid_query(hashgrid_id, candidate_position, candidate_radius)
    while wp.hash_grid_query_next(query, neighbor_idx):
        if neighbor_idx == candidate_idx:
            continue
        if neighbor_idx >= candidate_positions.shape[0]:
            continue
        if candidate_alive[neighbor_idx] == 0:
            continue

        neighbor_position = candidate_positions[neighbor_idx]
        neighbor_priority = candidate_priorities[neighbor_idx]
        min_radius = wp.min(candidate_radius, candidate_radii[neighbor_idx])

        # Keep the candidate with higher random priority (stable tiebreak).
        if _points_too_close(candidate_position, neighbor_position, min_radius):
            if neighbor_priority > candidate_priority or (
                neighbor_priority == candidate_priority and neighbor_idx > candidate_idx
            ):
                candidate_alive[candidate_idx] = 0
                return


# Commit surviving candidates into the accepted-sample arrays.
@wp.kernel
def _commit_accepted_candidates(
    candidate_positions: wp.array(dtype=wp.vec3f),
    candidate_radii: wp.array(dtype=wp.float32),
    candidate_alive: wp.array(dtype=wp.int32),
    accepted_positions: wp.array(dtype=wp.vec3f),
    accepted_radii: wp.array(dtype=wp.float32),
    accepted_count: wp.array(dtype=wp.int32),
):
    candidate_idx = wp.tid()
    if candidate_alive[candidate_idx] == 0:
        return

    accepted_idx = wp.atomic_add(accepted_count, 0, 1)
    if accepted_idx >= accepted_positions.shape[0]:
        return

    accepted_positions[accepted_idx] = candidate_positions[candidate_idx]
    accepted_radii[accepted_idx] = candidate_radii[candidate_idx]


# Compute Yuksel sample-elimination contribution for one pairwise distance.
@wp.func
def _wse_pair_weight(
    distance_squared: wp.float32,
    r_min: wp.float32,
    r_max: wp.float32,
    alpha: wp.float32,
) -> wp.float32:
    distance = wp.sqrt(distance_squared)
    if distance < r_min:
        distance = r_min
    value = 1.0 - distance / r_max
    if value <= 0.0:
        return wp.float32(0.0)
    return wp.pow(value, alpha)


# Count valid weighted-elimination neighbors for each sample (CSR row lengths).
@wp.kernel
def _count_wse_neighbors(
    hashgrid_id: wp.uint64,
    sample_positions: wp.array(dtype=wp.vec3f),
    r_max: wp.float32,
    output_counts: wp.array(dtype=wp.int32),
):
    sample_idx = wp.tid()
    center = sample_positions[sample_idx]
    radius_sq = r_max * r_max
    neighbor_count = int(0)
    neighbor_idx = int(0)
    query = wp.hash_grid_query(hashgrid_id, center, r_max)
    while wp.hash_grid_query_next(query, neighbor_idx):
        if neighbor_idx == sample_idx:
            continue
        if neighbor_idx >= sample_positions.shape[0]:
            continue
        d2 = _distance_squared(center, sample_positions[neighbor_idx])
        if d2 >= radius_sq:
            continue
        neighbor_count += 1
    output_counts[sample_idx] = neighbor_count


# Fill CSR adjacency and pair weights for weighted sample elimination.
@wp.kernel
def _write_wse_csr(
    hashgrid_id: wp.uint64,
    sample_positions: wp.array(dtype=wp.vec3f),
    row_ptr: wp.array(dtype=wp.int32),
    r_min: wp.float32,
    r_max: wp.float32,
    alpha: wp.float32,
    col_idx: wp.array(dtype=wp.int32),
    pair_weights: wp.array(dtype=wp.float32),
):
    sample_idx = wp.tid()
    center = sample_positions[sample_idx]
    radius_sq = r_max * r_max
    write_cursor = row_ptr[sample_idx]

    neighbor_idx = int(0)
    query = wp.hash_grid_query(hashgrid_id, center, r_max)
    while wp.hash_grid_query_next(query, neighbor_idx):
        if neighbor_idx == sample_idx:
            continue
        if neighbor_idx >= sample_positions.shape[0]:
            continue
        d2 = _distance_squared(center, sample_positions[neighbor_idx])
        if d2 >= radius_sq:
            continue

        col_idx[write_cursor] = neighbor_idx
        pair_weights[write_cursor] = _wse_pair_weight(d2, r_min, r_max, alpha)
        write_cursor += 1


# Initialize weighted-elimination scores from CSR pair weights.
@wp.kernel
def _initialize_wse_weights_from_csr(
    row_ptr: wp.array(dtype=wp.int32),
    pair_weights: wp.array(dtype=wp.float32),
    deleted: wp.array(dtype=wp.int32),
    output_weights: wp.array(dtype=wp.float32),
):
    sample_idx = wp.tid()
    if deleted[sample_idx] != 0:
        output_weights[sample_idx] = -1.0e30
        return

    row_start = row_ptr[sample_idx]
    row_end = row_ptr[sample_idx + 1]
    weight = wp.float32(0.0)
    for edge_idx in range(row_start, row_end):
        weight = weight + pair_weights[edge_idx]
    output_weights[sample_idx] = weight


# Subtract deleted-sample contributions for a whole batch from CSR neighbors.
@wp.kernel
def _subtract_deleted_wse_contribution_batch_csr(
    row_ptr: wp.array(dtype=wp.int32),
    col_idx: wp.array(dtype=wp.int32),
    pair_weights: wp.array(dtype=wp.float32),
    deleted_batch: wp.array(dtype=wp.int32),
    batch_count: int,
    max_row_size: int,
    deleted: wp.array(dtype=wp.int32),
    weights: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    if max_row_size <= 0:
        return

    batch_slot = tid // max_row_size
    local_edge_idx = tid - batch_slot * max_row_size
    if batch_slot >= batch_count:
        return

    deleted_index = deleted_batch[batch_slot]
    row_start = row_ptr[deleted_index]
    row_end = row_ptr[deleted_index + 1]
    edge_idx = row_start + local_edge_idx
    if edge_idx >= row_end:
        return

    neighbor_idx = col_idx[edge_idx]
    if deleted[neighbor_idx] != 0:
        return
    # Multiple deleted nodes can share neighbors, so use atomic accumulation.
    wp.atomic_add(weights, neighbor_idx, -pair_weights[edge_idx])


# Mark a batch of samples as deleted and set their weights to -inf.
@wp.kernel
def _mark_wse_deleted_batch(
    deleted_batch: wp.array(dtype=wp.int32),
    batch_count: int,
    deleted: wp.array(dtype=wp.int32),
    weights: wp.array(dtype=wp.float32),
    neg_inf: wp.float32,
):
    tid = wp.tid()
    if tid >= batch_count:
        return
    sample_idx = deleted_batch[tid]
    deleted[sample_idx] = 1
    weights[sample_idx] = neg_inf
