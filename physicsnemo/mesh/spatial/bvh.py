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

"""Bounding Volume Hierarchy (BVH) for efficient spatial queries.

This module implements a GPU-compatible BVH using flat array storage for efficient
traversal on both CPU and GPU. The BVH enables O(log N) query time for finding
which cells contain query points, compared to O(N) for brute-force search.

Construction uses a morton-code-based Linear BVH (LBVH) algorithm that runs in
O(log N) Python iterations instead of the O(N) iterations required by a naive
sequential approach, enabling scalability to hundreds of millions of cells.
"""

from typing import TYPE_CHECKING

import torch
from jaxtyping import Bool, Float, Int
from tensordict import tensorclass

from physicsnemo.mesh.neighbors._adjacency import Adjacency, build_adjacency_from_pairs
from physicsnemo.mesh.spatial._lbvh import build_lbvh_topology
from physicsnemo.mesh.spatial._ragged import _ragged_arange

if TYPE_CHECKING:
    from physicsnemo.mesh.mesh import Mesh


# ---------------------------------------------------------------------------
# Morton code computation
# ---------------------------------------------------------------------------


def _compute_morton_codes(
    centroids: Float[torch.Tensor, "n_centroids n_spatial_dims"],
) -> Int[torch.Tensor, " n_centroids"]:
    """Compute morton codes (Z-order curve) for a set of points.

    Morton codes interleave the bits of quantized coordinates to produce a
    single integer that preserves spatial locality: points that are nearby in
    D-dimensional space tend to have nearby morton codes. Sorting by morton
    code therefore clusters spatially-adjacent primitives together, which is
    the foundation of Linear BVH (LBVH) construction.

    Parameters
    ----------
    centroids : torch.Tensor
        Point coordinates, shape ``(N, D)``, any float dtype.

    Returns
    -------
    torch.Tensor
        Morton codes, shape ``(N,)``, dtype int64. All values are non-negative
        (fit in 63 useful bits of signed int64).
    """
    if centroids.ndim != 2:
        raise ValueError(
            f"centroids must be 2D (N, D), got {centroids.ndim}D "
            f"with shape {tuple(centroids.shape)}"
        )
    if not centroids.is_floating_point():
        raise TypeError(
            f"centroids must be a floating-point tensor (got {centroids.dtype=!r}); "
            f"integer input would silently produce wrong quantization"
        )

    N, D = centroids.shape
    device = centroids.device

    ### Bits per dimension: 63 // D keeps total code <= 63 bits (non-negative int64)
    n_bits = 63 // D
    max_val = (1 << n_bits) - 1

    ### Quantize centroids to integer grid [0, 2^n_bits - 1]
    cmin = centroids.min(dim=0).values  # (D,)
    cmax = centroids.max(dim=0).values  # (D,)
    extent = (cmax - cmin).clamp(min=1e-30)  # avoid division by zero
    coords = ((centroids - cmin) / extent * max_val).long().clamp(0, max_val)  # (N, D)

    ### Bit-interleave all dimensions: bit b of dim d -> position b*D + d.
    if device.type == "cuda":
        # CUDA is launch-bound in the bit loop below. Materializing all bits at
        # once trades a modest temporary for far fewer kernel launches.
        bit_offsets = torch.arange(n_bits, dtype=torch.int64, device=device)
        dim_offsets = torch.arange(D, dtype=torch.int64, device=device)
        bits = (coords.unsqueeze(-1) >> bit_offsets) & 1  # (N, D, n_bits)
        shifts = bit_offsets.view(1, 1, -1) * D + dim_offsets.view(1, -1, 1)
        return (bits << shifts).reshape(N, -1).sum(dim=1)

    code = torch.zeros(N, dtype=torch.int64, device=device)
    dim_offsets = torch.arange(D, dtype=torch.int64, device=device)  # (D,)
    for b in range(n_bits):
        bits = (coords >> b) & 1  # (N, D) - extract bit b from every dim
        code += (bits << (b * D + dim_offsets)).sum(dim=1)  # (N,)
    return code


# ---------------------------------------------------------------------------
# Leaf expansion helper
# ---------------------------------------------------------------------------


def _expand_leaf_hits(
    leaf_query_indices: Int[torch.Tensor, " n_leaf_hits"],
    leaf_node_indices: Int[torch.Tensor, " n_leaf_hits"],
    leaf_start: Int[torch.Tensor, " n_leaves"],
    leaf_count: Int[torch.Tensor, " n_leaves"],
    sorted_cell_order: Int[torch.Tensor, " n_cells"],
) -> tuple[Int[torch.Tensor, " n_expanded"], Int[torch.Tensor, " n_expanded"]]:
    """Expand (query, leaf_node) hits into (query, cell) candidate pairs.

    Each leaf node may contain multiple cells. This performs a "ragged expand"
    to produce one ``(query_idx, cell_idx)`` pair for every cell in every hit
    leaf.

    Parameters
    ----------
    leaf_query_indices : torch.Tensor
        Query indices for leaf hits, shape ``(n_hits,)``.
    leaf_node_indices : torch.Tensor
        Node indices for leaf hits, shape ``(n_hits,)``.
    leaf_start : torch.Tensor
        Per-node start index into ``sorted_cell_order``, shape ``(n_nodes,)``.
    leaf_count : torch.Tensor
        Per-node cell count, shape ``(n_nodes,)``.
    sorted_cell_order : torch.Tensor
        Morton-sorted cell permutation, shape ``(n_cells,)``.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(expanded_query_indices, expanded_cell_indices)``
    """
    starts = leaf_start[leaf_node_indices]  # (n_hits,)
    counts = leaf_count[leaf_node_indices]  # (n_hits,)
    device = leaf_query_indices.device

    # repeat_interleave already materializes the expanded length, so reuse its shape
    # (no host sync) for both the empty-check and _ragged_arange's ``total`` -- this
    # replaces an int(counts.sum()) sync and a second sync inside _ragged_arange.
    expanded_queries = torch.repeat_interleave(leaf_query_indices, counts)
    total = expanded_queries.shape[0]

    if total == 0:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty

    sorted_positions, _ = _ragged_arange(starts, counts, total=total)
    expanded_cells = sorted_cell_order[sorted_positions]

    return expanded_queries, expanded_cells


# ---------------------------------------------------------------------------
# Segmented leaf AABB computation
# ---------------------------------------------------------------------------


def _compute_leaf_aabbs(
    leaf_seg_starts: Int[torch.Tensor, " n_leaves"],
    leaf_seg_sizes: Int[torch.Tensor, " n_leaves"],
    sorted_aabb_min: Float[torch.Tensor, "n_sorted n_spatial_dims"],
    sorted_aabb_max: Float[torch.Tensor, "n_sorted n_spatial_dims"],
) -> tuple[
    Float[torch.Tensor, "n_leaves n_spatial_dims"],
    Float[torch.Tensor, "n_leaves n_spatial_dims"],
]:
    """Compute AABBs for a batch of leaf segments via segmented reduction.

    Each leaf segment is a contiguous range ``[start, start + size)`` in the
    morton-sorted cell arrays.

    Parameters
    ----------
    leaf_seg_starts : torch.Tensor
        Start positions in the sorted cell array, shape ``(n_leaf_segs,)``.
    leaf_seg_sizes : torch.Tensor
        Number of cells per leaf segment, shape ``(n_leaf_segs,)``.
    sorted_aabb_min : torch.Tensor
        Per-cell AABB minima in sorted order, shape ``(n_cells, D)``.
    sorted_aabb_max : torch.Tensor
        Per-cell AABB maxima in sorted order, shape ``(n_cells, D)``.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(aabb_min, aabb_max)`` each of shape ``(n_leaf_segs, D)``.
    """
    device = leaf_seg_starts.device
    D = sorted_aabb_min.shape[1]
    dtype = sorted_aabb_min.dtype
    n_leaf_segs = len(leaf_seg_starts)

    if int(leaf_seg_sizes.sum()) == 0 or n_leaf_segs == 0:
        return (
            torch.empty((0, D), dtype=dtype, device=device),
            torch.empty((0, D), dtype=dtype, device=device),
        )

    cell_pos, seg_ids = _ragged_arange(leaf_seg_starts, leaf_seg_sizes)

    cell_mins = sorted_aabb_min[cell_pos]  # (total_cells, D)
    cell_maxs = sorted_aabb_max[cell_pos]  # (total_cells, D)

    ### Segmented min/max reduction
    seg_min = torch.full((n_leaf_segs, D), float("inf"), dtype=dtype, device=device)
    seg_max = torch.full((n_leaf_segs, D), float("-inf"), dtype=dtype, device=device)
    exp_ids = seg_ids.unsqueeze(1).expand_as(cell_mins)
    seg_min.scatter_reduce_(0, exp_ids, cell_mins, reduce="amin", include_self=True)
    seg_max.scatter_reduce_(0, exp_ids, cell_maxs, reduce="amax", include_self=True)

    return seg_min, seg_max


# ---------------------------------------------------------------------------
# BVH tensorclass
# ---------------------------------------------------------------------------


@tensorclass
class BVH:
    """Bounding Volume Hierarchy for efficient spatial queries.

    The BVH is stored as flat tensors for GPU compatibility, avoiding
    pointer-based tree structures. Each internal node has exactly two children
    (binary tree). Leaf nodes store a contiguous range of cells in
    morton-sorted order.

    Construction uses a Linear BVH (LBVH) algorithm: cells are sorted by
    morton code (Z-order curve), then the tree is built top-down by splitting
    sorted segments at their midpoints. This runs in O(log N) Python-level
    iterations with O(N log N) total GPU work, enabling scalability to
    hundreds of millions of cells.

    Attributes
    ----------
    node_aabb_min : torch.Tensor
        Minimum corner of axis-aligned bounding box for each node,
        shape ``(n_nodes, n_spatial_dims)``.
    node_aabb_max : torch.Tensor
        Maximum corner of AABB for each node,
        shape ``(n_nodes, n_spatial_dims)``.
    node_left_child : torch.Tensor
        Index of left child for each internal node,
        shape ``(n_nodes,)``, dtype int64. Value is -1 for leaf nodes.
    node_right_child : torch.Tensor
        Index of right child for each internal node,
        shape ``(n_nodes,)``, dtype int64. Value is -1 for leaf nodes.
    leaf_start : torch.Tensor
        Start index into ``sorted_cell_order`` for leaf nodes,
        shape ``(n_nodes,)``, dtype int64. Value is -1 for internal nodes.
    leaf_count : torch.Tensor
        Number of cells in each leaf node,
        shape ``(n_nodes,)``, dtype int64. Value is 0 for internal nodes.
    sorted_cell_order : torch.Tensor
        Morton-code-sorted permutation of cell indices,
        shape ``(n_cells,)``, dtype int64.

    Examples
    --------
    >>> bvh = BVH.from_mesh(mesh)
    >>> candidates = bvh.find_candidate_cells(query_points)
    """

    node_aabb_min: Float[torch.Tensor, "n_nodes n_spatial_dims"]
    node_aabb_max: Float[torch.Tensor, "n_nodes n_spatial_dims"]
    # Child indices: -1 for leaves
    node_left_child: Int[torch.Tensor, " n_nodes"]
    node_right_child: Int[torch.Tensor, " n_nodes"]
    # Leaf metadata: -1 / 0 for internal nodes
    leaf_start: Int[torch.Tensor, " n_nodes"]
    leaf_count: Int[torch.Tensor, " n_nodes"]
    sorted_cell_order: Int[torch.Tensor, " n_cells"]

    @property
    def n_nodes(self) -> int:
        """Number of nodes in the BVH."""
        return self.node_aabb_min.shape[0]

    @property
    def n_spatial_dims(self) -> int:
        """Dimensionality of the spatial space."""
        return self.node_aabb_min.shape[1]

    @property
    def device(self) -> torch.device:
        """Device where BVH tensors are stored."""
        return self.node_aabb_min.device

    @classmethod
    def from_mesh(cls, mesh: "Mesh", leaf_size: int = 1) -> "BVH":
        """Construct a BVH from a mesh using morton-code LBVH.

        Cells are sorted by the morton code of their centroids, then the tree
        is built top-down by recursively splitting sorted segments at their
        midpoints. AABBs are computed in two phases: leaf AABBs via segmented
        reduction over cell bounds, internal AABBs via a bottom-up pass from
        leaves to root. The entire construction runs in O(log N) Python-level
        iterations.

        Parameters
        ----------
        mesh : Mesh
            The mesh to build the BVH for.
        leaf_size : int, optional
            Maximum number of cells per leaf node. The default of 1 minimizes
            candidate cells per query hit but maximizes node count
            (``2 * n_cells - 1`` nodes) and tree depth. Larger values reduce
            build time and memory at the cost of more candidate cells per
            query hit.

        Returns
        -------
        BVH
            Constructed BVH ready for queries.

        Raises
        ------
        ValueError
            If ``leaf_size < 1``.
        """
        if leaf_size < 1:
            raise ValueError(f"leaf_size must be >= 1, got {leaf_size=!r}")

        n_cells = mesh.n_cells
        D = mesh.n_spatial_dims
        device = mesh.points.device
        dtype = mesh.points.dtype

        ### Handle empty mesh
        if n_cells == 0:
            empty_long = torch.empty(0, dtype=torch.long, device=device)
            return cls(
                node_aabb_min=torch.empty((0, D), dtype=dtype, device=device),
                node_aabb_max=torch.empty((0, D), dtype=dtype, device=device),
                node_left_child=empty_long,
                node_right_child=empty_long,
                leaf_start=empty_long,
                leaf_count=empty_long,
                sorted_cell_order=empty_long,
            )

        ### Compute per-cell bounding boxes; reuse cached centroids
        cell_vertices = mesh.points[mesh.cells]  # (n_cells, n_verts, D)
        cell_aabb_min = cell_vertices.min(dim=1).values  # (n_cells, D)
        cell_aabb_max = cell_vertices.max(dim=1).values  # (n_cells, D)
        cell_centroids = mesh.cell_centroids  # (n_cells, D)

        ### Sort cells by morton code for spatial coherence
        morton_codes = _compute_morton_codes(cell_centroids)
        sorted_order = morton_codes.argsort(stable=True)  # (n_cells,)
        sorted_aabb_min = cell_aabb_min[sorted_order]  # (n_cells, D)
        sorted_aabb_max = cell_aabb_max[sorted_order]  # (n_cells, D)

        ### Build the shared morton-LBVH node topology over the sorted cells.
        topo = build_lbvh_topology(n_cells, leaf_size, device)

        ### Fill leaf AABBs from per-cell bounds (segmented reduction), then
        # propagate bottom-up so each internal node bounds its two children.
        node_aabb_min_buf = torch.full(
            (topo.max_nodes, D), float("inf"), dtype=dtype, device=device
        )
        node_aabb_max_buf = torch.full(
            (topo.max_nodes, D), float("-inf"), dtype=dtype, device=device
        )
        seg_min, seg_max = _compute_leaf_aabbs(
            topo.leaf_starts, topo.leaf_sizes, sorted_aabb_min, sorted_aabb_max
        )
        node_aabb_min_buf[topo.leaf_node_ids] = seg_min
        node_aabb_max_buf[topo.leaf_node_ids] = seg_max

        for level_node_ids in reversed(topo.internal_nodes_per_level):
            left = topo.left_child[level_node_ids]
            right = topo.right_child[level_node_ids]
            node_aabb_min_buf[level_node_ids] = torch.minimum(
                node_aabb_min_buf[left], node_aabb_min_buf[right]
            )
            node_aabb_max_buf[level_node_ids] = torch.maximum(
                node_aabb_max_buf[left], node_aabb_max_buf[right]
            )

        ### Trim to actual node count
        node_count = topo.node_count
        return cls(
            node_aabb_min=node_aabb_min_buf[:node_count],
            node_aabb_max=node_aabb_max_buf[:node_count],
            node_left_child=topo.left_child[:node_count],
            node_right_child=topo.right_child[:node_count],
            leaf_start=topo.leaf_start[:node_count],
            leaf_count=topo.leaf_count[:node_count],
            sorted_cell_order=sorted_order,
        )

    def point_in_aabb(
        self,
        points: Float[torch.Tensor, "n_points n_spatial_dims"],
        aabb_min: Float[torch.Tensor, "n_boxes n_spatial_dims"],
        aabb_max: Float[torch.Tensor, "n_boxes n_spatial_dims"],
    ) -> Bool[torch.Tensor, "n_points n_boxes"]:
        """Test if points are inside axis-aligned bounding boxes.

        Parameters
        ----------
        points : torch.Tensor
            Query points, shape ``(n_points, n_spatial_dims)``.
        aabb_min : torch.Tensor
            Minimum corners, shape ``(n_boxes, n_spatial_dims)``.
        aabb_max : torch.Tensor
            Maximum corners, shape ``(n_boxes, n_spatial_dims)``.

        Returns
        -------
        torch.Tensor
            Boolean tensor of shape ``(n_points, n_boxes)`` indicating
            containment.
        """
        if points.ndim != 2 or aabb_min.ndim != 2 or aabb_max.ndim != 2:
            raise ValueError(
                f"All inputs must be 2D tensors; got points={tuple(points.shape)}, "
                f"aabb_min={tuple(aabb_min.shape)}, aabb_max={tuple(aabb_max.shape)}"
            )
        D = points.shape[1]
        if aabb_min.shape[1] != D or aabb_max.shape[1] != D:
            raise ValueError(
                f"Spatial dimension mismatch: points has {D} dims, "
                f"aabb_min has {aabb_min.shape[1]}, aabb_max has {aabb_max.shape[1]}"
            )

        points_exp = points.unsqueeze(1)  # (n_points, 1, D)
        aabb_min_exp = aabb_min.unsqueeze(0)  # (1, n_boxes, D)
        aabb_max_exp = aabb_max.unsqueeze(0)  # (1, n_boxes, D)

        inside = ((points_exp >= aabb_min_exp) & (points_exp <= aabb_max_exp)).all(
            dim=2
        )
        return inside

    def _validate_query_points(
        self,
        query_points: Float[torch.Tensor, "n_queries n_spatial_dims"],
    ) -> None:
        """Validate shape, dtype, and dimensionality of query points."""
        if query_points.ndim != 2:
            raise ValueError(
                f"query_points must be 2D (n_queries, n_spatial_dims), got "
                f"{query_points.ndim}D with shape {tuple(query_points.shape)}"
            )
        if not query_points.is_floating_point():
            raise TypeError(
                f"query_points must be a floating-point tensor "
                f"(got {query_points.dtype=!r})"
            )
        if self.n_nodes > 0 and query_points.shape[1] != self.n_spatial_dims:
            raise ValueError(
                f"query_points has {query_points.shape[1]} spatial dims, but "
                f"BVH has {self.n_spatial_dims}"
            )

    def _traverse(
        self,
        query_points: Float[torch.Tensor, "n_queries n_spatial_dims"],
        max_candidates_per_point: int | None,
        aabb_tolerance: float,
    ) -> tuple[
        Int[torch.Tensor, " n_pairs"],
        Int[torch.Tensor, " n_pairs"],
    ]:
        r"""Core BVH traversal returning raw ``(query_idx, cell_idx)`` pairs.

        All AABB tests and tree operations are fully vectorized across
        queries. The outer loop runs once per tree level
        (:math:`O(\log N)` iterations).
        """
        n_queries = query_points.shape[0]
        dev = self.device

        if self.n_nodes == 0 or n_queries == 0:
            empty = torch.empty(0, dtype=torch.long, device=dev)
            return empty, empty.clone()

        ### Initialize work queue: all queries start at root (node 0)
        current_query_indices = torch.arange(n_queries, dtype=torch.long, device=dev)
        current_node_indices = torch.zeros(n_queries, dtype=torch.long, device=dev)

        ### Track candidate counts per query (for max_candidates enforcement)
        candidates_count = torch.zeros(n_queries, dtype=torch.long, device=dev)

        ### Accumulate (query_idx, cell_idx) result pairs
        all_query_indices_list: list[torch.Tensor] = []
        all_cell_indices_list: list[torch.Tensor] = []

        ### Iterative traversal: one iteration per tree level
        while len(current_query_indices) > 0:
            ### Vectorized AABB containment test for all active pairs
            batch_points = query_points[current_query_indices]  # (n_active, D)
            batch_min = self.node_aabb_min[current_node_indices]
            batch_max = self.node_aabb_max[current_node_indices]

            inside = (
                (batch_points >= batch_min - aabb_tolerance)
                & (batch_points <= batch_max + aabb_tolerance)
            ).all(dim=1)

            ### Filter to intersecting pairs only
            hit_query = current_query_indices[inside]
            hit_node = current_node_indices[inside]

            if len(hit_query) == 0:
                break

            ### Separate leaf hits from internal-node hits
            hit_leaf_count = self.leaf_count[hit_node]
            is_leaf = hit_leaf_count > 0

            ### Handle leaf hits: expand to (query, cell) pairs
            if is_leaf.any():
                expanded_q, expanded_c = _expand_leaf_hits(
                    hit_query[is_leaf],
                    hit_node[is_leaf],
                    self.leaf_start,
                    self.leaf_count,
                    self.sorted_cell_order,
                )
                if len(expanded_q) > 0:
                    all_query_indices_list.append(expanded_q)
                    all_cell_indices_list.append(expanded_c)

                    candidates_count += torch.bincount(expanded_q, minlength=n_queries)

            ### Handle internal-node hits: expand to children
            is_internal = ~is_leaf
            int_query = hit_query[is_internal]
            int_node = hit_node[is_internal]

            # Enforce max_candidates limit
            if max_candidates_per_point is not None and len(int_query) > 0:
                under_limit = candidates_count[int_query] < max_candidates_per_point
                int_query = int_query[under_limit]
                int_node = int_node[under_limit]

            if len(int_query) == 0:
                break

            left_children = self.node_left_child[int_node]
            right_children = self.node_right_child[int_node]

            valid_left = left_children >= 0
            valid_right = right_children >= 0

            parts_q: list[torch.Tensor] = []
            parts_n: list[torch.Tensor] = []
            if valid_left.any():
                parts_q.append(int_query[valid_left])
                parts_n.append(left_children[valid_left])
            if valid_right.any():
                parts_q.append(int_query[valid_right])
                parts_n.append(right_children[valid_right])

            if parts_q:
                current_query_indices = torch.cat(parts_q)
                current_node_indices = torch.cat(parts_n)
            else:
                break

        if all_query_indices_list:
            return torch.cat(all_query_indices_list), torch.cat(all_cell_indices_list)
        empty = torch.empty(0, dtype=torch.long, device=dev)
        return empty, empty.clone()

    def find_candidate_cells(
        self,
        query_points: Float[torch.Tensor, "n_queries n_spatial_dims"],
        max_candidates_per_point: int | None = 32,
        aabb_tolerance: float = 1e-6,
    ) -> Adjacency:
        r"""Find candidate cells that might contain each query point.

        Uses batched iterative BVH traversal where all queries are processed
        simultaneously in a vectorized manner.

        Parameters
        ----------
        query_points : torch.Tensor
            Points to query, shape ``(n_queries, n_spatial_dims)``.
        max_candidates_per_point : int | None, optional
            Maximum number of candidate cells to return per query point.
            Prevents memory explosion for degenerate cases. If None, no
            limit is applied.
        aabb_tolerance : float, optional
            Tolerance for AABB intersection test. Important for degenerate
            cells (e.g., cells with duplicate vertices).

        Returns
        -------
        Adjacency
            Adjacency object where candidates for query ``i`` are at
            ``result.indices[result.offsets[i]:result.offsets[i+1]]``.
            Use ``result.to_list()`` for a list-of-tensors representation.

        Notes
        -----
        Complexity is :math:`O(M \log N)` where :math:`M` = queries and
        :math:`N` = cells. All AABB tests and tree operations are fully
        vectorized across queries - there are no Python-level loops over
        individual query points. The outer loop runs once per tree level
        (:math:`O(\log N)` iterations).
        """
        self._validate_query_points(query_points)
        all_q, all_c = self._traverse(
            query_points,
            max_candidates_per_point,
            aabb_tolerance,
        )
        n_queries = query_points.shape[0]
        adjacency = build_adjacency_from_pairs(
            source_indices=all_q,
            target_indices=all_c,
            n_sources=n_queries,
            n_targets=len(self.sorted_cell_order),
        )
        return adjacency.truncate_per_source(max_candidates_per_point)
