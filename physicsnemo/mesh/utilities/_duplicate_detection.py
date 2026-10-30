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
from jaxtyping import Float, Int


def vectorized_connected_components(
    pairs: Int[torch.Tensor, "n_pairs 2"], n_elements: int
) -> Int[torch.Tensor, " n_elements"]:
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
    # Invariant: parent[i] <= i, since parent starts as arange and both the
    # "amin" scatter and path compression only lower it. Hence merging the
    # roots of a pair needs only the scatter of the smaller root lo into the
    # larger root hi: the reverse scatter would be a no-op, as
    # parent[lo] <= lo <= hi. Merging the raw pairs as well
    # is not needed: in the first iteration parent[pairs] == pairs, and
    # afterwards parent[max(a, b)] <= min(a, b) already holds and stays true.
    first = pairs[:, 0]
    second = pairs[:, 1]
    max_iterations = 100
    for _ in range(max_iterations):
        prev = parent.clone()

        # Union step: merge the larger root of each pair to the smaller root
        root_first = parent[first]
        root_second = parent[second]
        smaller_root = torch.minimum(root_first, root_second)
        larger_root = root_first.clamp_(min=root_second)  # the maximum, in place
        del root_first, root_second
        parent.scatter_reduce_(0, larger_root, smaller_root, reduce="amin")
        del larger_root, smaller_root

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
    points: Float[torch.Tensor, "n_points n_spatial_dims"],
    tolerance: float,
) -> Int[torch.Tensor, "n_pairs 2"]:
    r"""Find all pairs of points whose L2 distance is below ``tolerance``.

    Uses a BVH for :math:`O(n \log n)` candidate generation, then exact
    L2 filtering.  Every returned pair satisfies ``i < j``.

    Parameters
    ----------
    points : torch.Tensor
        Point coordinates, shape ``(n_points, n_spatial_dims)``.
    tolerance : float
        Absolute distance threshold.

    Returns
    -------
    torch.Tensor
        Duplicate pairs, shape ``(n_pairs, 2)`` with
        ``pairs[:, 0] < pairs[:, 1]``. Empty ``(0, 2)`` tensor if no
        duplicates are found.
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
        max_candidates_per_point=None,
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
    points: Float[torch.Tensor, "n_points n_spatial_dims"],
    tolerance: float,
) -> Int[torch.Tensor, " n_points"]:
    """Map each point to the smallest-index representative in its cluster.

    Two points belong to the same cluster when they are connected by a
    chain of pairwise distances below ``tolerance`` (transitive closure).

    Parameters
    ----------
    points : torch.Tensor
        Point coordinates, shape ``(n_points, n_spatial_dims)``.
    tolerance : float
        Absolute distance threshold.

    Returns
    -------
    torch.Tensor
        Shape ``(n_points,)``.  ``canonical[i]`` is the index of the
        canonical representative for point ``i`` (always the smallest
        index in the equivalence class).
    """
    n_points = points.shape[0]
    device = points.device

    if n_points < 2:
        return torch.arange(n_points, device=device, dtype=torch.long)

    pairs = find_duplicate_pairs(points, tolerance)

    return vectorized_connected_components(pairs, n_points)
