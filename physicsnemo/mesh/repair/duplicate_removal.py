# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
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

"""Remove duplicate vertices from meshes.

Merges vertices that are coincident within a tolerance and updates cell
connectivity accordingly.
"""

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from physicsnemo.mesh.mesh import Mesh


def remove_duplicate_vertices(
    mesh: "Mesh",
    tolerance: float = 1e-6,
) -> tuple["Mesh", dict[str, int]]:
    """Merge coincident vertices and update cell connectivity.

    Identifies pairs of vertices closer than tolerance and merges them,
    updating all cell references to use the merged vertex indices.

    Args:
        mesh: Input mesh
        tolerance: Distance threshold for considering vertices duplicates

    Returns:
        Tuple of (cleaned_mesh, stats_dict) where stats_dict contains:
        - "n_duplicates_merged": Number of duplicate vertices merged
        - "n_points_original": Original number of points
        - "n_points_final": Final number of points

    Algorithm:
        Uses BVH spatial data structure for O(n log n) complexity.
        Constructs a BVH from point cloud, queries for nearby points, then
        checks exact distances only for candidates. All operations are fully
        vectorized with no Python loops over points.

    Example:
        >>> from physicsnemo.mesh.primitives.basic import two_triangles_2d
        >>> mesh = two_triangles_2d.load()
        >>> mesh_clean, stats = remove_duplicate_vertices(mesh, tolerance=1e-6)
        >>> assert mesh_clean.validate()["valid"]
    """
    n_original = mesh.n_points
    device = mesh.points.device

    if n_original == 0:
        return mesh, {
            "n_duplicates_merged": 0,
            "n_points_original": 0,
            "n_points_final": 0,
        }

    if n_original == 1:
        return mesh, {
            "n_duplicates_merged": 0,
            "n_points_original": 1,
            "n_points_final": 1,
        }

    ### Create 0-manifold mesh for BVH construction
    # Each point is a 0-cell (single vertex) with degenerate AABB
    from physicsnemo.mesh.mesh import Mesh as TempMesh
    from physicsnemo.mesh.spatial.bvh import BVH

    point_cells = torch.arange(n_original, device=device, dtype=torch.long).unsqueeze(
        1
    )  # (n_points, 1)

    # Create 0-manifold mesh: cells.shape[-1] - 1 = 1 - 1 = 0
    point_mesh = TempMesh(
        points=mesh.points,
        cells=point_cells,
    )

    ### Build BVH for efficient spatial queries
    bvh = BVH.from_mesh(point_mesh)

    ### Find candidate duplicates using BVH
    # For each point, find all points within tolerance (using L∞ distance with tolerance)
    candidate_lists = bvh.find_candidate_cells(
        query_points=mesh.points,
        max_candidates_per_point=100,  # Conservative upper bound
        aabb_tolerance=tolerance,
    )

    ### Extract candidate pairs and compute exact distances
    # Build list of (query_idx, candidate_idx) pairs
    pair_queries = []
    pair_candidates = []
    for query_idx, candidates in enumerate(candidate_lists):
        if len(candidates) > 0:
            pair_queries.append(torch.full_like(candidates, query_idx))
            pair_candidates.append(candidates)

    if len(pair_queries) == 0:
        # No candidates found
        return mesh, {
            "n_duplicates_merged": 0,
            "n_points_original": n_original,
            "n_points_final": n_original,
        }

    pair_queries = torch.cat(pair_queries)  # (n_pairs,)
    pair_candidates = torch.cat(pair_candidates)  # (n_pairs,)

    # Remove self-pairs and ensure query < candidate to avoid duplicate counting
    valid_pairs = pair_queries < pair_candidates
    pair_queries = pair_queries[valid_pairs]
    pair_candidates = pair_candidates[valid_pairs]

    if len(pair_queries) == 0:
        return mesh, {
            "n_duplicates_merged": 0,
            "n_points_original": n_original,
            "n_points_final": n_original,
        }

    # Compute exact L2 distances for candidate pairs
    distances = torch.norm(
        mesh.points[pair_queries] - mesh.points[pair_candidates],
        dim=-1,
    )

    # Filter to actual duplicates (within L2 tolerance)
    is_duplicate = distances < tolerance
    v1_orig = pair_queries[is_duplicate]
    v2_orig = pair_candidates[is_duplicate]

    if len(v1_orig) == 0:
        return mesh, {
            "n_duplicates_merged": 0,
            "n_points_original": n_original,
            "n_points_final": n_original,
        }

    ### Build union-find structure (vectorized)

    # Initialize parent array: each vertex is its own parent
    parent = torch.arange(n_original, device=device, dtype=torch.long)

    # Union operation: merge to smaller index for consistency
    merge_from = torch.maximum(v1_orig, v2_orig)
    merge_to = torch.minimum(v1_orig, v2_orig)

    # Apply unions using scatter with reduction to handle multiple merges
    # Use scatter_reduce with 'amin' to always keep smallest parent
    parent.scatter_reduce_(
        dim=0,
        index=merge_from,
        src=merge_to,
        reduce="amin",
    )

    # Path compression: iteratively follow parent pointers until convergence
    # Each iteration halves the tree depth (expected O(log log n) iterations)
    max_iterations = 20  # Conservative upper bound
    for _ in range(max_iterations):
        old_parent = parent
        parent = parent[parent]  # Follow parent pointers (vectorized)
        if torch.equal(parent, old_parent):
            break

    canonical_indices = parent

    ### Compute unique vertices
    unique_canonical = torch.unique(canonical_indices)
    n_unique = len(unique_canonical)
    n_merged = n_original - n_unique

    if n_merged == 0:
        # No duplicates found after union-find
        return mesh, {
            "n_duplicates_merged": 0,
            "n_points_original": n_original,
            "n_points_final": n_original,
        }

    ### Create mapping from old to new indices (fully vectorized)
    # Scatter to create old_to_new mapping
    old_to_new = torch.empty(n_original, device=device, dtype=torch.long)
    old_to_new[unique_canonical] = torch.arange(
        n_unique, device=device, dtype=torch.long
    )

    # Map all vertices through their canonical representative
    old_to_new = old_to_new[canonical_indices]

    ### Build new mesh
    new_points = mesh.points[unique_canonical]
    new_cells = old_to_new[mesh.cells]

    ### Transfer data (excluding cache)
    # Filter out cache before indexing to avoid transferring cached computations
    from tensordict import TensorDict

    from physicsnemo.mesh.mesh import Mesh

    point_data_filtered = mesh.point_data.exclude("_cache")
    new_point_data = TensorDict(
        point_data_filtered[unique_canonical], batch_size=[n_unique]
    )
    new_cell_data = TensorDict(
        mesh.cell_data.exclude("_cache"), batch_size=mesh.cell_data.batch_size
    )
    new_global_data = TensorDict(
        mesh.global_data, batch_size=mesh.global_data.batch_size
    )

    cleaned_mesh = Mesh(
        points=new_points,
        cells=new_cells,
        point_data=new_point_data,
        cell_data=new_cell_data,
        global_data=new_global_data,
    )

    stats = {
        "n_duplicates_merged": n_merged,
        "n_points_original": n_original,
        "n_points_final": n_unique,
    }

    return cleaned_mesh, stats
