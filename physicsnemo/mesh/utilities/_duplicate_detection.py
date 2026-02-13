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

"""Duplicate point detection using BVH-accelerated spatial queries.

This is the single source of truth for finding and clustering coincident
points in a mesh.  All higher-level operations (cleaning, repair,
validation) delegate here.

Algorithm
---------
1. Construct a 0-manifold ``Mesh`` where each point is its own cell.
2. Build a BVH over those degenerate cells.
3. Query the BVH with the same points, expanded by ``tolerance``.
4. Filter candidate pairs by exact L2 distance.
5. (Optional) Cluster pairs into equivalence classes via vectorised
   union-find with path compression.
"""

import warnings

import torch


def vectorized_connected_components(
    pairs: torch.Tensor, n_elements: int
) -> torch.Tensor:
    """Compute connected components from pairwise connections.

    Uses iterative vectorized union-find with path compression.

    Parameters
    ----------
    pairs : torch.Tensor
        Shape (n_pairs, 2). Each row is a pair of element indices that should
        be in the same component.
    n_elements : int
        Total number of elements.

    Returns
    -------
    torch.Tensor
        Shape (n_elements,). labels[i] is the canonical (smallest index)
        representative of element i's component.
    """
    device = pairs.device
    parent = torch.arange(n_elements, dtype=torch.long, device=device)

    if len(pairs) == 0:
        return parent

    ### Iterative union-find: repeat union + path compression until stable
    max_iterations = 100
    for _ in range(max_iterations):
        prev = parent.clone()

        # Union step: merge to smaller index
        merge_from = torch.maximum(pairs[:, 0], pairs[:, 1])
        merge_to = torch.minimum(pairs[:, 0], pairs[:, 1])
        # Also need to merge through current parent pointers
        parent_from = parent[pairs[:, 0]]
        parent_to = parent[pairs[:, 1]]
        all_merge_from = torch.cat([merge_from, torch.maximum(parent_from, parent_to)])
        all_merge_to = torch.cat([merge_to, torch.minimum(parent_from, parent_to)])

        parent.scatter_reduce_(
            dim=0,
            index=all_merge_from,
            src=all_merge_to,
            reduce="amin",
        )

        # Path compression
        parent = parent[parent]

        if torch.equal(parent, prev):
            break
    else:
        warnings.warn(
            f"Union-find did not converge in {max_iterations} iterations. "
            "This should not happen for valid meshes.",
            stacklevel=2,
        )

    return parent


def find_duplicate_pairs(
    points: torch.Tensor,
    tolerance: float,
) -> torch.Tensor:
    """Find all pairs of points whose L2 distance is below *tolerance*.

    Uses a BVH for O(n log n) candidate generation, then exact L2
    filtering.  Every returned pair satisfies ``i < j``.

    Parameters
    ----------
    points : torch.Tensor
        Point coordinates, shape (n_points, n_spatial_dims).
    tolerance : float
        Absolute distance threshold.

    Returns
    -------
    torch.Tensor
        Duplicate pairs, shape (n_pairs, 2) with ``pairs[:, 0] < pairs[:, 1]``.
        Empty (0, 2) tensor if no duplicates are found.
    """
    n_points = points.shape[0]
    device = points.device

    if n_points < 2:
        return torch.empty((0, 2), dtype=torch.long, device=device)

    ### Build a 0-manifold mesh so the BVH has cells to work with
    from physicsnemo.mesh.mesh import Mesh
    from physicsnemo.mesh.spatial.bvh import BVH

    point_cells = torch.arange(n_points, device=device, dtype=torch.long).unsqueeze(1)
    point_mesh = Mesh(points=points, cells=point_cells)

    bvh = BVH.from_mesh(point_mesh)

    ### BVH query: L∞ candidate pairs within tolerance
    candidate_adjacency = bvh.find_candidate_cells(
        query_points=points,
        max_candidates_per_point=100,
        aabb_tolerance=tolerance,
    )

    if candidate_adjacency.n_total_neighbors == 0:
        return torch.empty((0, 2), dtype=torch.long, device=device)

    pair_queries, pair_candidates = candidate_adjacency.expand_to_pairs()

    ### Keep only pairs with query < candidate (canonical order, no self-pairs)
    valid = pair_queries < pair_candidates
    pair_queries = pair_queries[valid]
    pair_candidates = pair_candidates[valid]

    if len(pair_queries) == 0:
        return torch.empty((0, 2), dtype=torch.long, device=device)

    ### Exact L2 distance filter
    distances = torch.linalg.vector_norm(
        points[pair_queries] - points[pair_candidates], dim=-1
    )
    within = distances < tolerance

    if not within.any():
        return torch.empty((0, 2), dtype=torch.long, device=device)

    return torch.stack([pair_queries[within], pair_candidates[within]], dim=1)


def compute_canonical_indices(
    points: torch.Tensor,
    tolerance: float,
) -> torch.Tensor:
    """Map each point to the smallest-index representative in its cluster.

    Two points belong to the same cluster when they are connected by a
    chain of pairwise distances below *tolerance* (transitive closure).

    Parameters
    ----------
    points : torch.Tensor
        Point coordinates, shape (n_points, n_spatial_dims).
    tolerance : float
        Absolute distance threshold.

    Returns
    -------
    torch.Tensor
        Shape (n_points,).  ``canonical[i]`` is the index of the
        canonical representative for point *i* (always the smallest
        index in the equivalence class).
    """
    n_points = points.shape[0]
    device = points.device

    if n_points < 2:
        return torch.arange(n_points, device=device, dtype=torch.long)

    pairs = find_duplicate_pairs(points, tolerance)

    return vectorized_connected_components(pairs, n_points)
