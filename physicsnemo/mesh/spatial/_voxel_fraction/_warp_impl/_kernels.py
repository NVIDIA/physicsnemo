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
# 1) Closed-mesh voxel classification kernel
# 2) Open-mesh winding-number voxel classification kernel

from __future__ import annotations

import warp as wp


# ----------------------------------------------------------------------------
# Voxelization Kernels
# ----------------------------------------------------------------------------
@wp.kernel
def _voxel_mesh_intersection_kernel(
    mesh_id: wp.uint64,
    origin: wp.vec3f,
    voxel_size: wp.float32,
    nx: int,
    ny: int,
    nz: int,
    n_samples: int,
    seed_offset: int,
    output: wp.array(dtype=wp.float32),
):
    # Compute voxel coordinates from launch index.
    i, j, k = wp.tid()
    if i >= nx or j >= ny or k >= nz:
        return

    # Compute flattened output index for this voxel.
    output_index = i + j * nx + k * nx * ny

    # Build voxel bounds and center in world coordinates.
    low = origin + wp.vec3f(
        wp.float32(i) * voxel_size,
        wp.float32(j) * voxel_size,
        wp.float32(k) * voxel_size,
    )
    high = low + wp.vec3f(voxel_size, voxel_size, voxel_size)
    center = (low + high) * wp.float32(0.5)

    # Query whether any triangles overlap this voxel AABB.
    query = wp.mesh_query_aabb(mesh_id, low, high)
    tri_index = wp.int32(0)

    # Use grid extent as a conservative distance scale for inside/outside queries.
    max_dim = wp.max(nx, wp.max(ny, nz))
    max_dist = voxel_size * wp.float32(max_dim)

    # Fast path: no triangle overlap -> classify voxel center only.
    if not wp.mesh_query_aabb_next(query, tri_index):
        hit = wp.mesh_query_point_sign_normal(mesh_id, center, max_dist, 1.0e-6)
        output[output_index] = wp.float32(1.0) if hit.result and hit.sign < 0.0 else 0.0
        return

    # Overlap path: estimate volume fraction with Monte Carlo samples.
    inside_count = wp.int32(0)
    rng_state = wp.rand_init(seed_offset + output_index)

    for _ in range(n_samples):
        rx = wp.randf(rng_state)
        ry = wp.randf(rng_state)
        rz = wp.randf(rng_state)
        sample = low + wp.vec3f(rx, ry, rz) * voxel_size
        hit = wp.mesh_query_point_sign_normal(mesh_id, sample, max_dist, 1.0e-6)
        if hit.result and hit.sign < 0.0:
            inside_count += 1

    output[output_index] = wp.float32(inside_count) / wp.float32(n_samples)


# Kernel for open meshes using sign-winding-number point queries.
@wp.kernel
def _voxel_open_mesh_intersection_kernel(
    mesh_id: wp.uint64,
    origin: wp.vec3f,
    voxel_size: wp.float32,
    nx: int,
    ny: int,
    nz: int,
    n_samples: int,
    seed_offset: int,
    winding_number_threshold: wp.float32,
    winding_number_accuracy: wp.float32,
    output: wp.array(dtype=wp.float32),
):
    # Compute voxel coordinates from launch index.
    i, j, k = wp.tid()
    if i >= nx or j >= ny or k >= nz:
        return

    # Compute flattened output index for this voxel.
    output_index = i + j * nx + k * nx * ny

    # Build voxel bounds and center in world coordinates.
    low = origin + wp.vec3f(
        wp.float32(i) * voxel_size,
        wp.float32(j) * voxel_size,
        wp.float32(k) * voxel_size,
    )
    high = low + wp.vec3f(voxel_size, voxel_size, voxel_size)
    center = (low + high) * wp.float32(0.5)

    # Query whether any triangles overlap this voxel AABB.
    query = wp.mesh_query_aabb(mesh_id, low, high)
    tri_index = wp.int32(0)

    # Use grid extent as a conservative distance scale for inside/outside queries.
    max_dim = wp.max(nx, wp.max(ny, nz))
    max_dist = voxel_size * wp.float32(max_dim)

    # Fast path: no triangle overlap -> classify voxel center only.
    if not wp.mesh_query_aabb_next(query, tri_index):
        hit = wp.mesh_query_point_sign_winding_number(
            mesh_id,
            center,
            max_dist,
            winding_number_accuracy,
            winding_number_threshold,
        )
        output[output_index] = wp.float32(1.0) if hit.result and hit.sign < 0.0 else 0.0
        return

    # Overlap path: estimate volume fraction with Monte Carlo samples.
    inside_count = wp.int32(0)
    rng_state = wp.rand_init(seed_offset + output_index)

    for _ in range(n_samples):
        rx = wp.randf(rng_state)
        ry = wp.randf(rng_state)
        rz = wp.randf(rng_state)
        sample = low + wp.vec3f(rx, ry, rz) * voxel_size
        hit = wp.mesh_query_point_sign_winding_number(
            mesh_id,
            sample,
            max_dist,
            winding_number_accuracy,
            winding_number_threshold,
        )
        if hit.result and hit.sign < 0.0:
            inside_count += 1

    output[output_index] = wp.float32(inside_count) / wp.float32(n_samples)
