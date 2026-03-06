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

"""Spatial cluster tree for dual-tree Barnes-Hut acceleration of GLOBE kernels.

This module provides a GPU-compatible hierarchical spatial decomposition over a
set of points, designed for dual-tree Barnes-Hut O(N) kernel acceleration.
Trees are built over both source and target points.  The dual-tree traversal
classifies (target_node, source_node) pairs as near-field or far-field:

- **Near-field**: both nodes are leaves and nearby - expand to individual
  (target, source) pairs for exact kernel evaluation.
- **Far-field**: nodes are well-separated - evaluate the kernel ONCE at the
  node centroids and broadcast the result to all targets in the target node.

This reduces far-field kernel evaluations from O(N log N) (single-tree) to
O(N) (dual-tree), which is critical at large mesh scales (800k+ faces).

Construction uses a morton-code-based Linear BVH (LBVH) algorithm identical in
structure to :mod:`physicsnemo.mesh.spatial.bvh`, producing a binary radix tree
stored as flat tensors for GPU compatibility.
"""

from dataclasses import dataclass
from math import ceil

import torch
from jaxtyping import Float, Int
from tensordict import TensorDict, tensorclass
from torch.profiler import record_function

from physicsnemo.mesh.spatial.bvh import _compute_morton_codes


# ---------------------------------------------------------------------------
# InteractionPlan: the output of tree traversal
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DualInteractionPlan:
    r"""Result of a dual-tree Barnes-Hut traversal: near-field individual
    pairs and far-field node-to-node pairs that together cover all source
    contributions for every target point.

    Near-field pairs ``(near_target_ids[i], near_source_ids[i])`` are
    individual target-source pairs requiring exact kernel evaluation.

    Far-field pairs ``(far_target_node_ids[i], far_source_node_ids[i])``
    are node-to-node pairs where the kernel is evaluated ONCE at the
    node centroids and the result is broadcast to all individual targets
    in the target node.  This reduces far-field kernel evaluations from
    O(N log N) to O(N).

    All index tensors are ``int64`` on the same device as the tree.
    """

    near_target_ids: Int[torch.Tensor, " n_near"]
    near_source_ids: Int[torch.Tensor, " n_near"]
    far_target_node_ids: Int[torch.Tensor, " n_far_nodes"]
    far_source_node_ids: Int[torch.Tensor, " n_far_nodes"]

    @property
    def n_near(self) -> int:
        """Number of near-field (exact) individual interaction pairs."""
        return self.near_target_ids.shape[0]

    @property
    def n_far_nodes(self) -> int:
        """Number of far-field node-to-node pairs (each = one kernel eval)."""
        return self.far_target_node_ids.shape[0]


# ---------------------------------------------------------------------------
# Segmented reduction helpers
# ---------------------------------------------------------------------------


def _ragged_arange(
    starts: Int[torch.Tensor, " n_segments"],
    counts: Int[torch.Tensor, " n_segments"],
) -> tuple[Int[torch.Tensor, " total"], Int[torch.Tensor, " total"]]:
    r"""Expand segment descriptors ``(start, count)`` into flat index arrays.

    Given *N* segments where segment *i* spans positions
    ``[starts[i], starts[i] + counts[i])``, produces two flat tensors of
    length ``sum(counts)``:

    - ``positions[k]``: the absolute index for element *k*
    - ``seg_ids[k]``: the segment (``0..N-1``) that element *k* belongs to

    Conceptually, this concatenates ``arange(s, s+c)`` for each ``(s, c)``
    pair, along with the corresponding segment labels.

    Parameters
    ----------
    starts : torch.Tensor
        Start offset per segment, shape ``(N,)``, int64.
    counts : torch.Tensor
        Element count per segment, shape ``(N,)``, int64.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(positions, seg_ids)`` each with shape ``(sum(counts),)``.
    """
    total = int(counts.sum())
    device = starts.device
    n_segments = starts.shape[0]

    seg_ids = torch.repeat_interleave(
        torch.arange(n_segments, dtype=torch.long, device=device),
        counts,
    )
    # Within-segment offsets: [0, 1, ..., c0-1, 0, 1, ..., c1-1, ...]
    cum = counts.cumsum(0)
    offsets = torch.arange(total, dtype=torch.long, device=device)
    offsets = offsets - torch.repeat_interleave(cum - counts, counts)

    positions = torch.repeat_interleave(starts, counts) + offsets

    return positions, seg_ids


def _segmented_weighted_sum(
    values: Float[torch.Tensor, "n *features"],
    weights: Float[torch.Tensor, " n"],
    seg_ids: Int[torch.Tensor, " n"],
    n_segments: int,
) -> Float[torch.Tensor, "n_segments *features"]:
    """Compute weighted sum per segment via scatter_add.

    Parameters
    ----------
    values : torch.Tensor
        Values to aggregate, shape ``(N,)`` or ``(N, F)``.
    weights : torch.Tensor
        Per-element weights, shape ``(N,)``.
    seg_ids : torch.Tensor
        Segment assignment for each element, shape ``(N,)``, int64.
    n_segments : int
        Total number of output segments.

    Returns
    -------
    torch.Tensor
        Weighted sums, shape ``(n_segments,)`` or ``(n_segments, F)``.
    """
    weighted = values * (weights.unsqueeze(-1) if values.ndim > 1 else weights)
    out = torch.zeros(
        (n_segments,) + values.shape[1:],
        dtype=values.dtype,
        device=values.device,
    )
    idx = seg_ids.unsqueeze(-1).expand_as(weighted) if weighted.ndim > 1 else seg_ids
    out.scatter_add_(0, idx, weighted)
    return out


def _expand_leaf_hits(
    leaf_query_indices: Int[torch.Tensor, " n_hits"],
    leaf_node_indices: Int[torch.Tensor, " n_hits"],
    leaf_start: Int[torch.Tensor, " n_nodes"],
    leaf_count: Int[torch.Tensor, " n_nodes"],
    sorted_order: Int[torch.Tensor, " n_sources"],
) -> tuple[Int[torch.Tensor, " n_expanded"], Int[torch.Tensor, " n_expanded"]]:
    """Expand ``(query, leaf_node)`` hits into ``(query, source)`` pairs.

    Each leaf node contains a contiguous range of sources in morton-sorted
    order. This performs a ragged expand to produce one pair per source in
    every hit leaf.

    Parameters
    ----------
    leaf_query_indices : torch.Tensor
        Query indices for leaf hits, shape ``(n_hits,)``.
    leaf_node_indices : torch.Tensor
        Node indices for leaf hits, shape ``(n_hits,)``.
    leaf_start : torch.Tensor
        Per-node start offset into ``sorted_order``, shape ``(n_nodes,)``.
    leaf_count : torch.Tensor
        Per-node source count, shape ``(n_nodes,)``.
    sorted_order : torch.Tensor
        Morton-sorted source permutation, shape ``(n_sources,)``.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(expanded_query_ids, expanded_source_ids)``
    """
    starts = leaf_start[leaf_node_indices]
    counts = leaf_count[leaf_node_indices]
    total = int(counts.sum())
    device = leaf_query_indices.device

    if total == 0:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty.clone()

    expanded_queries = torch.repeat_interleave(leaf_query_indices, counts)

    sorted_positions, _ = _ragged_arange(starts, counts)
    expanded_sources = sorted_order[sorted_positions]

    return expanded_queries, expanded_sources


def _expand_dual_leaf_hits(
    target_leaf_ids: Int[torch.Tensor, " n_leaf_pairs"],
    source_leaf_ids: Int[torch.Tensor, " n_leaf_pairs"],
    target_tree: "ClusterTree",
    source_tree: "ClusterTree",
) -> tuple[Int[torch.Tensor, " n_expanded"], Int[torch.Tensor, " n_expanded"]]:
    """Expand ``(target_leaf, source_leaf)`` pairs into all individual pairs.

    For each leaf pair, forms the Cartesian product: every target in the
    target leaf paired with every source in the source leaf.  This is the
    near-field expansion for the dual-tree traversal.

    Parameters
    ----------
    target_leaf_ids : torch.Tensor
        Target tree node IDs for near-field leaf pairs, shape ``(n,)``.
    source_leaf_ids : torch.Tensor
        Source tree node IDs for near-field leaf pairs, shape ``(n,)``.
    target_tree : ClusterTree
        Tree over target points.
    source_tree : ClusterTree
        Tree over source points.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(expanded_target_ids, expanded_source_ids)`` in original
        (unsorted) point indices.
    """
    t_starts = target_tree.leaf_start[target_leaf_ids]
    t_counts = target_tree.leaf_count[target_leaf_ids]
    s_starts = source_tree.leaf_start[source_leaf_ids]
    s_counts = source_tree.leaf_count[source_leaf_ids]

    product_counts = t_counts * s_counts
    total = int(product_counts.sum())
    device = target_leaf_ids.device

    if total == 0:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty.clone()

    ### Expand pair indices and compute within-pair linear offsets
    pair_ids = torch.repeat_interleave(
        torch.arange(len(target_leaf_ids), dtype=torch.long, device=device),
        product_counts,
    )
    cum = product_counts.cumsum(0)
    linear = torch.arange(total, dtype=torch.long, device=device)
    linear = linear - torch.repeat_interleave(cum - product_counts, product_counts)

    ### Decompose linear offset into (target_within, source_within) via
    # row-major indexing into a |T_i| x |S_i| grid.
    s_counts_per = torch.repeat_interleave(s_counts, product_counts)
    target_within = linear // s_counts_per
    source_within = linear % s_counts_per

    ### Map within-leaf offsets to original point indices
    t_starts_per = torch.repeat_interleave(t_starts, product_counts)
    s_starts_per = torch.repeat_interleave(s_starts, product_counts)

    expanded_targets = target_tree.sorted_source_order[t_starts_per + target_within]
    expanded_sources = source_tree.sorted_source_order[s_starts_per + source_within]

    return expanded_targets, expanded_sources


# ---------------------------------------------------------------------------
# ClusterTree tensorclass
# ---------------------------------------------------------------------------


@tensorclass
class ClusterTree:
    r"""Hierarchical spatial decomposition for Barnes-Hut kernel acceleration.

    Stores a binary radix tree over source points as flat GPU-compatible tensors.
    The tree structure (positions, AABBs, children) is precomputable per mesh
    geometry. Per-node source-data aggregates are recomputed whenever the source
    features change (e.g., between communication hyperlayers).

    The tree supports both boundary face centroids and prediction point clouds
    (same construction algorithm, same data structure).

    Attributes
    ----------
    node_aabb_min : torch.Tensor
        AABB minimum corner per node, shape ``(n_nodes, D)``.
    node_aabb_max : torch.Tensor
        AABB maximum corner per node, shape ``(n_nodes, D)``.
    node_diameter_sq : torch.Tensor
        Squared AABB diagonal per node, shape ``(n_nodes,)``.
    node_left_child : torch.Tensor
        Left child index per node, ``-1`` for leaves, shape ``(n_nodes,)``.
    node_right_child : torch.Tensor
        Right child index per node, ``-1`` for leaves, shape ``(n_nodes,)``.
    leaf_start : torch.Tensor
        Start offset into ``sorted_source_order`` for leaf nodes,
        ``-1`` for internal nodes, shape ``(n_nodes,)``.
    leaf_count : torch.Tensor
        Number of sources in each leaf node, ``0`` for internal nodes,
        shape ``(n_nodes,)``.
    node_range_start : torch.Tensor
        Start offset into ``sorted_source_order`` for ALL nodes (both
        leaf and internal), shape ``(n_nodes,)``.  Each node's subtree
        covers a contiguous range in morton-sorted order.
    node_range_count : torch.Tensor
        Number of points in each node's subtree, shape ``(n_nodes,)``.
        For leaves this equals ``leaf_count``; for internal nodes it
        equals the sum of children's range counts.
    node_total_area : torch.Tensor
        Total source area in each node's subtree, shape ``(n_nodes,)``.
    sorted_source_order : torch.Tensor
        Morton-code-sorted permutation of source indices,
        shape ``(n_sources,)``.
    source_points : torch.Tensor
        Original source point coordinates, shape ``(n_sources, D)``.
    max_depth : torch.Tensor
        Scalar tensor storing the tree depth (for fixed-iteration traversal).
    """

    node_aabb_min: torch.Tensor
    node_aabb_max: torch.Tensor
    node_diameter_sq: torch.Tensor
    node_left_child: torch.Tensor
    node_right_child: torch.Tensor
    leaf_start: torch.Tensor
    leaf_count: torch.Tensor
    node_range_start: torch.Tensor
    node_range_count: torch.Tensor
    node_total_area: torch.Tensor
    sorted_source_order: torch.Tensor
    source_points: torch.Tensor
    max_depth: torch.Tensor

    @property
    def n_nodes(self) -> int:
        """Number of nodes in the tree."""
        return self.node_aabb_min.shape[0]

    @property
    def n_sources(self) -> int:
        """Number of source points."""
        return self.sorted_source_order.shape[0]

    @property
    def n_spatial_dims(self) -> int:
        """Spatial dimensionality."""
        return self.node_aabb_min.shape[1]

    @classmethod
    def from_points(
        cls,
        points: Float[torch.Tensor, "n_points n_dims"],
        *,
        leaf_size: int = 32,
        areas: Float[torch.Tensor, " n_points"] | None = None,
    ) -> "ClusterTree":
        r"""Build a cluster tree from a set of points via morton-code LBVH.

        Parameters
        ----------
        points : Float[torch.Tensor, "n_points n_dims"]
            Source point coordinates, shape :math:`(N, D)`.
        leaf_size : int
            Maximum sources per leaf node. Larger values produce shallower
            trees (fewer traversal iterations) at the cost of more exact
            near-field interactions per leaf hit.
        areas : Float[torch.Tensor, "n_points"] or None
            Per-source area weights used for aggregate computation. If
            ``None``, all areas default to 1.

        Returns
        -------
        ClusterTree
            Constructed tree ready for traversal and aggregate computation.
        """
        if leaf_size < 1:
            raise ValueError(f"leaf_size must be >= 1, got {leaf_size=!r}")

        n_points = points.shape[0]
        D = points.shape[1]
        device = points.device
        dtype = points.dtype

        if areas is None:
            areas = torch.ones(n_points, device=device, dtype=dtype)

        ### Handle empty point set
        if n_points == 0:
            empty_long = torch.empty(0, dtype=torch.long, device=device)
            return cls(
                node_aabb_min=torch.empty((0, D), dtype=dtype, device=device),
                node_aabb_max=torch.empty((0, D), dtype=dtype, device=device),
                node_diameter_sq=torch.empty(0, dtype=dtype, device=device),
                node_left_child=empty_long,
                node_right_child=empty_long,
                leaf_start=empty_long,
                leaf_count=empty_long,
                node_range_start=empty_long,
                node_range_count=empty_long,
                node_total_area=torch.empty(0, dtype=dtype, device=device),
                sorted_source_order=empty_long,
                source_points=points,
                max_depth=torch.tensor(0, dtype=torch.long, device=device),
                batch_size=torch.Size([]),
            )

        ### Sort points by morton code for spatial coherence
        with record_function("cluster_tree::morton_sort"):
            morton_codes = _compute_morton_codes(points)
            sorted_order = morton_codes.argsort(stable=True)  # (n_points,)
            sorted_points = points[sorted_order]  # (n_points, D)
            sorted_areas = areas[sorted_order]  # (n_points,)

        ### Pre-allocate node storage.
        # The midpoint split guarantees each child gets at least
        # floor(parent_size / 2) sources, so the minimum leaf occupancy
        # is ceil(leaf_size / 2).  From that we bound the maximum number
        # of leaves and apply the full-binary-tree identity (n_internal =
        # n_leaves - 1) to get max_nodes.
        min_per_leaf = max(1, (leaf_size + 1) // 2)
        max_leaves = (n_points + min_per_leaf - 1) // min_per_leaf
        max_nodes = max(1, 2 * max_leaves - 1)

        aabb_min_buf = torch.full(
            (max_nodes, D), float("inf"), dtype=dtype, device=device
        )
        aabb_max_buf = torch.full(
            (max_nodes, D), float("-inf"), dtype=dtype, device=device
        )
        left_child = torch.full((max_nodes,), -1, dtype=torch.long, device=device)
        right_child = torch.full((max_nodes,), -1, dtype=torch.long, device=device)
        leaf_start_buf = torch.full((max_nodes,), -1, dtype=torch.long, device=device)
        leaf_count_buf = torch.zeros(max_nodes, dtype=torch.long, device=device)
        range_start_buf = torch.zeros(max_nodes, dtype=torch.long, device=device)
        range_count_buf = torch.zeros(max_nodes, dtype=torch.long, device=device)
        total_area_buf = torch.zeros(max_nodes, dtype=dtype, device=device)

        # -----------------------------------------------------------
        # Phase 1: Top-down LBVH construction (O(log N) iterations)
        # -----------------------------------------------------------
        with record_function("cluster_tree::top_down_build"):
            seg_starts = torch.tensor([0], dtype=torch.long, device=device)
            seg_ends = torch.tensor([n_points], dtype=torch.long, device=device)
            seg_node_ids = torch.tensor([0], dtype=torch.long, device=device)
            node_count = 1
            actual_depth = 0

            internal_nodes_per_level: list[torch.Tensor] = []

            while len(seg_starts) > 0:
                seg_sizes = seg_ends - seg_starts

                ### Store the sorted-order range for ALL nodes at this level.
                # Each node covers a contiguous range [seg_start, seg_end)
                # in the morton-sorted order.  Used by dual-tree traversal
                # to expand node-level results to individual points.
                range_start_buf[seg_node_ids] = seg_starts
                range_count_buf[seg_node_ids] = seg_sizes

                ### Classify segments as leaf or internal
                is_leaf_seg = seg_sizes <= leaf_size
                is_internal_seg = ~is_leaf_seg

                ### Process leaf segments
                leaf_indices = torch.where(is_leaf_seg)[0]
                if len(leaf_indices) > 0:
                    leaf_nids = seg_node_ids[leaf_indices]
                    l_starts = seg_starts[leaf_indices]
                    l_sizes = seg_sizes[leaf_indices]

                    leaf_start_buf[leaf_nids] = l_starts
                    leaf_count_buf[leaf_nids] = l_sizes

                    # Compute leaf AABBs via segmented reduction
                    _fill_leaf_aabbs(
                        leaf_nids,
                        l_starts,
                        l_sizes,
                        sorted_points,
                        aabb_min_buf,
                        aabb_max_buf,
                    )

                    # Compute leaf total areas
                    _fill_leaf_total_areas(
                        leaf_nids, l_starts, l_sizes, sorted_areas, total_area_buf
                    )

                ### Process internal segments: split at the midpoint of the
                # morton-sorted range.  Because morton codes preserve spatial
                # locality, this approximates a spatial median split and produces
                # a balanced binary tree in O(log N) iterations.
                internal_indices = torch.where(is_internal_seg)[0]
                if len(internal_indices) == 0:
                    break

                actual_depth += 1
                int_starts = seg_starts[internal_indices]
                int_ends = seg_ends[internal_indices]
                int_sizes = seg_sizes[internal_indices]
                int_node_ids = seg_node_ids[internal_indices]

                midpoints = int_starts + int_sizes // 2

                n_internal = len(internal_indices)
                left_ids = (
                    node_count
                    + torch.arange(n_internal, dtype=torch.long, device=device) * 2
                )
                right_ids = left_ids + 1
                node_count += 2 * n_internal

                left_child[int_node_ids] = left_ids
                right_child[int_node_ids] = right_ids
                internal_nodes_per_level.append(int_node_ids)

                seg_starts = torch.cat([int_starts, midpoints])
                seg_ends = torch.cat([midpoints, int_ends])
                seg_node_ids = torch.cat([left_ids, right_ids])

        # -----------------------------------------------------------
        # Phase 2: Bottom-up AABB and area propagation
        # -----------------------------------------------------------
        with record_function("cluster_tree::bottom_up_aabb"):
            for level_node_ids in reversed(internal_nodes_per_level):
                left = left_child[level_node_ids]
                right = right_child[level_node_ids]
                aabb_min_buf[level_node_ids] = torch.minimum(
                    aabb_min_buf[left], aabb_min_buf[right]
                )
                aabb_max_buf[level_node_ids] = torch.maximum(
                    aabb_max_buf[left], aabb_max_buf[right]
                )
                total_area_buf[level_node_ids] = (
                    total_area_buf[left] + total_area_buf[right]
                )

        ### Compute squared AABB diagonals
        aabb_min_trimmed = aabb_min_buf[:node_count]
        aabb_max_trimmed = aabb_max_buf[:node_count]
        diameter_sq = (aabb_max_trimmed - aabb_min_trimmed).pow(2).sum(dim=-1)

        return cls(
            node_aabb_min=aabb_min_trimmed,
            node_aabb_max=aabb_max_trimmed,
            node_diameter_sq=diameter_sq,
            node_left_child=left_child[:node_count],
            node_right_child=right_child[:node_count],
            leaf_start=leaf_start_buf[:node_count],
            leaf_count=leaf_count_buf[:node_count],
            node_range_start=range_start_buf[:node_count],
            node_range_count=range_count_buf[:node_count],
            node_total_area=total_area_buf[:node_count],
            sorted_source_order=sorted_order,
            source_points=points,
            max_depth=torch.tensor(actual_depth, dtype=torch.long, device=device),
            batch_size=torch.Size([]),
        )

    def compute_source_aggregates(
        self,
        source_points: Float[torch.Tensor, "n_sources n_dims"],
        areas: Float[torch.Tensor, " n_sources"],
        source_data: TensorDict | None = None,
    ) -> "SourceAggregates":
        r"""Compute per-node aggregate source data for far-field approximation.

        Aggregates are area-weighted averages of source features within each
        node's subtree. The total weight for each node is the sum of per-source
        strengths (handled separately during kernel evaluation, not here).

        Parameters
        ----------
        source_points : Float[torch.Tensor, "n_sources n_dims"]
            Source coordinates, shape :math:`(N, D)`.
        areas : Float[torch.Tensor, "n_sources"]
            Per-source area weights, shape :math:`(N,)`.
        source_data : TensorDict or None
            Per-source features (normals, latents, etc.) with
            ``batch_size=(N,)``. ``None`` if no per-source features.

        Returns
        -------
        SourceAggregates
            Per-node aggregated centroids and source data.
        """
        if self.n_nodes == 0:
            D = source_points.shape[1]
            device = source_points.device
            dtype = source_points.dtype
            return SourceAggregates(
                node_centroid=torch.empty((0, D), dtype=dtype, device=device),
                node_source_data=None,
            )

        device = source_points.device
        dtype = source_points.dtype
        D = source_points.shape[1]
        n_nodes = self.n_nodes

        ### Leaf aggregation: compute per-leaf centroids and source data
        with record_function("cluster_tree::leaf_aggregation"):
            is_leaf = self.leaf_count > 0
            leaf_node_ids = torch.where(is_leaf)[0]

            if leaf_node_ids.numel() > 0:
                leaf_starts = self.leaf_start[leaf_node_ids]
                leaf_counts = self.leaf_count[leaf_node_ids]
                n_leaves = leaf_node_ids.shape[0]
                positions, compact_ids = _ragged_arange(leaf_starts, leaf_counts)

                seg_ids_compact = torch.zeros(
                    self.n_sources, dtype=torch.long, device=device
                )
                seg_ids_compact[positions] = compact_ids

            sorted_points = source_points[self.sorted_source_order]
            sorted_areas = areas[self.sorted_source_order]

            centroid_buf = torch.zeros(n_nodes, D, dtype=dtype, device=device)

            if leaf_node_ids.numel() > 0:
                leaf_centroids = _segmented_weighted_sum(
                    sorted_points, sorted_areas, seg_ids_compact, n_leaves
                )
                leaf_total_areas = self.node_total_area[leaf_node_ids]
                safe_areas = leaf_total_areas.clamp(min=1e-30)
                leaf_centroids = leaf_centroids / safe_areas.unsqueeze(-1)
                centroid_buf[leaf_node_ids] = leaf_centroids

            node_source_data: TensorDict | None = None
            if source_data is not None and leaf_node_ids.numel() > 0:
                sorted_source_data = source_data[self.sorted_source_order]
                node_source_data = _aggregate_source_data_leaves(
                    sorted_source_data,
                    sorted_areas,
                    seg_ids_compact,
                    n_leaves,
                    leaf_node_ids,
                    leaf_total_areas,
                    n_nodes,
                    device,
                )

        ### Bottom-up propagation: internal node centroids
        with record_function("cluster_tree::bottom_up_propagation"):
            _propagate_centroids_bottom_up(
                centroid_buf,
                node_source_data,
                self.node_left_child,
                self.node_right_child,
                self.node_total_area,
                n_nodes,
            )

        return SourceAggregates(
            node_centroid=centroid_buf,
            node_source_data=node_source_data,
        )

    def find_dual_interaction_pairs(
        self,
        target_tree: "ClusterTree",
        theta: float = 1.0,
    ) -> DualInteractionPlan:
        r"""Find near-field and far-field pairs via dual-tree traversal.

        Traverses both the source tree (``self``) and ``target_tree``
        simultaneously.  For well-separated node pairs, records a single
        far-field (target_node, source_node) entry - the kernel is evaluated
        ONCE at the node centroids and broadcast to all targets in the node.
        This reduces far-field kernel evaluations from O(N log N) to O(N).

        Uses a combined AABB-distance opening criterion:
        ``(D_T + D_S) / r < theta``, where D_T and D_S are the AABB
        diagonals and r is the minimum distance between the two AABBs.
        This accounts for approximation error on both the target and
        source sides.

        Parameters
        ----------
        target_tree : ClusterTree
            Tree over target points.  For self-interaction (communication
            layers), this is the same object as ``self``.
        theta : float
            Barnes-Hut opening angle.  Larger = more aggressive.
            ``theta = 0`` forces all interactions to be exact.

        Returns
        -------
        DualInteractionPlan
            Near-field individual pairs and far-field node-to-node pairs.
        """
        source_tree = self
        device = source_tree.node_aabb_min.device
        theta_sq = theta * theta

        ### Handle empty trees
        if source_tree.n_nodes == 0 or target_tree.n_nodes == 0:
            empty = torch.empty(0, dtype=torch.long, device=device)
            return DualInteractionPlan(
                near_target_ids=empty,
                near_source_ids=empty.clone(),
                far_target_node_ids=empty.clone(),
                far_source_node_ids=empty.clone(),
            )

        with record_function("cluster_tree::dual_traversal"):
            ### Initialize: root-to-root pair
            active_tgt_nodes = torch.zeros(1, dtype=torch.long, device=device)
            active_src_nodes = torch.zeros(1, dtype=torch.long, device=device)

            near_target_list: list[torch.Tensor] = []
            near_source_list: list[torch.Tensor] = []
            far_tgt_node_list: list[torch.Tensor] = []
            far_src_node_list: list[torch.Tensor] = []

            max_iters = int(target_tree.max_depth.item()) + int(source_tree.max_depth.item()) + 1

            for _ in range(max_iters):
                if active_tgt_nodes.numel() == 0:
                    break

                ### Combined opening criterion: minimum AABB-to-AABB gap.
                # For each dimension, the gap is the positive distance
                # between the two boxes (zero if they overlap).
                aabb_min_T = target_tree.node_aabb_min[active_tgt_nodes]
                aabb_max_T = target_tree.node_aabb_max[active_tgt_nodes]
                aabb_min_S = source_tree.node_aabb_min[active_src_nodes]
                aabb_max_S = source_tree.node_aabb_max[active_src_nodes]

                gap = torch.clamp(
                    torch.maximum(aabb_min_T - aabb_max_S, aabb_min_S - aabb_max_T),
                    min=0,
                )
                min_dist_sq = gap.pow(2).sum(dim=-1)

                diam_T = target_tree.node_diameter_sq[active_tgt_nodes].sqrt()
                diam_S = source_tree.node_diameter_sq[active_src_nodes].sqrt()
                combined_diam_sq = (diam_T + diam_S).pow(2)

                is_far = min_dist_sq * theta_sq > combined_diam_sq

                ### Classify active pairs
                is_leaf_T = target_tree.leaf_count[active_tgt_nodes] > 0
                is_leaf_S = source_tree.leaf_count[active_src_nodes] > 0

                ### 1. Far-field: well-separated node pairs
                if is_far.any():
                    far_tgt_node_list.append(active_tgt_nodes[is_far])
                    far_src_node_list.append(active_src_nodes[is_far])

                ### 2. Near-field, both leaves: Cartesian product expansion
                near_leaf_leaf = (~is_far) & is_leaf_T & is_leaf_S
                if near_leaf_leaf.any():
                    exp_tgts, exp_srcs = _expand_dual_leaf_hits(
                        active_tgt_nodes[near_leaf_leaf],
                        active_src_nodes[near_leaf_leaf],
                        target_tree,
                        source_tree,
                    )
                    near_target_list.append(exp_tgts)
                    near_source_list.append(exp_srcs)

                ### 3. Need to split: at least one is internal, not far
                need_split = (~is_far) & (~near_leaf_leaf)
                if not need_split.any():
                    break

                split_tgt = active_tgt_nodes[need_split]
                split_src = active_src_nodes[need_split]
                split_is_leaf_T = is_leaf_T[need_split]
                split_is_leaf_S = is_leaf_S[need_split]
                split_diam_sq_T = target_tree.node_diameter_sq[split_tgt]
                split_diam_sq_S = source_tree.node_diameter_sq[split_src]

                ### Splitting decision: split the larger node.
                # If equal (including self-interaction T==S), split both.
                # If one side is a leaf, can only split the other.
                do_split_T = (~split_is_leaf_T) & (
                    split_is_leaf_S | (split_diam_sq_T >= split_diam_sq_S)
                )
                do_split_S = (~split_is_leaf_S) & (
                    split_is_leaf_T | (split_diam_sq_S >= split_diam_sq_T)
                )

                ### Generate child pairs for each split case
                next_tgt_parts: list[torch.Tensor] = []
                next_src_parts: list[torch.Tensor] = []

                # Case A: split T only (T internal, S leaf or T strictly larger)
                case_T_only = do_split_T & (~do_split_S)
                if case_T_only.any():
                    t_ids = split_tgt[case_T_only]
                    s_ids = split_src[case_T_only]
                    left_T = target_tree.node_left_child[t_ids]
                    right_T = target_tree.node_right_child[t_ids]
                    for child_T in (left_T, right_T):
                        valid = child_T >= 0
                        if valid.any():
                            next_tgt_parts.append(child_T[valid])
                            next_src_parts.append(s_ids[valid])

                # Case B: split S only (S internal, T leaf or S strictly larger)
                case_S_only = do_split_S & (~do_split_T)
                if case_S_only.any():
                    t_ids = split_tgt[case_S_only]
                    s_ids = split_src[case_S_only]
                    left_S = source_tree.node_left_child[s_ids]
                    right_S = source_tree.node_right_child[s_ids]
                    for child_S in (left_S, right_S):
                        valid = child_S >= 0
                        if valid.any():
                            next_tgt_parts.append(t_ids[valid])
                            next_src_parts.append(child_S[valid])

                # Case C: split both (both internal, equal diameter or T==S)
                case_both = do_split_T & do_split_S
                if case_both.any():
                    t_ids = split_tgt[case_both]
                    s_ids = split_src[case_both]
                    left_T = target_tree.node_left_child[t_ids]
                    right_T = target_tree.node_right_child[t_ids]
                    left_S = source_tree.node_left_child[s_ids]
                    right_S = source_tree.node_right_child[s_ids]
                    for child_T in (left_T, right_T):
                        for child_S in (left_S, right_S):
                            valid = (child_T >= 0) & (child_S >= 0)
                            if valid.any():
                                next_tgt_parts.append(child_T[valid])
                                next_src_parts.append(child_S[valid])

                if next_tgt_parts:
                    active_tgt_nodes = torch.cat(next_tgt_parts)
                    active_src_nodes = torch.cat(next_src_parts)
                else:
                    break

            ### Concatenate accumulated pairs
            if near_target_list:
                near_tgt = torch.cat(near_target_list)
                near_src = torch.cat(near_source_list)
            else:
                near_tgt = torch.empty(0, dtype=torch.long, device=device)
                near_src = torch.empty(0, dtype=torch.long, device=device)

            if far_tgt_node_list:
                far_tgt_nid = torch.cat(far_tgt_node_list)
                far_src_nid = torch.cat(far_src_node_list)
            else:
                far_tgt_nid = torch.empty(0, dtype=torch.long, device=device)
                far_src_nid = torch.empty(0, dtype=torch.long, device=device)

            ### Sort near pairs by source index for coalesced gather
            if near_src.numel() > 0:
                sort_order = near_src.argsort(stable=True)
                near_tgt = near_tgt[sort_order]
                near_src = near_src[sort_order]

            ### Sort far pairs by source node for coalesced aggregate gather
            if far_src_nid.numel() > 0:
                sort_order = far_src_nid.argsort(stable=True)
                far_tgt_nid = far_tgt_nid[sort_order]
                far_src_nid = far_src_nid[sort_order]

        return DualInteractionPlan(
            near_target_ids=near_tgt,
            near_source_ids=near_src,
            far_target_node_ids=far_tgt_nid,
            far_source_node_ids=far_src_nid,
        )


# ---------------------------------------------------------------------------
# SourceAggregates: per-node aggregate data for far-field approximation
# ---------------------------------------------------------------------------


@dataclass
class SourceAggregates:
    """Per-node aggregated source data for far-field monopole approximation.

    Computed by :meth:`ClusterTree.compute_source_aggregates` and consumed
    by :class:`BarnesHutKernel` during kernel evaluation.
    """

    node_centroid: Float[torch.Tensor, "n_nodes n_dims"]
    """Area-weighted centroid per node."""

    node_source_data: TensorDict | None
    """Area-weighted average source features per node, or ``None`` if no
    per-source features. Has ``batch_size=(n_nodes,)``."""


# ---------------------------------------------------------------------------
# Internal helpers for tree construction
# ---------------------------------------------------------------------------


def _fill_leaf_aabbs(
    leaf_nids: Int[torch.Tensor, " n_leaves"],
    leaf_starts: Int[torch.Tensor, " n_leaves"],
    leaf_sizes: Int[torch.Tensor, " n_leaves"],
    sorted_points: Float[torch.Tensor, "n_sorted_sources n_dims"],
    aabb_min_buf: Float[torch.Tensor, "n_nodes n_dims"],
    aabb_max_buf: Float[torch.Tensor, "n_nodes n_dims"],
) -> None:
    """Fill AABB buffers for leaf nodes via segmented reduction (in-place)."""
    device = leaf_nids.device
    D = sorted_points.shape[1]
    dtype = sorted_points.dtype
    n_leaves = leaf_nids.shape[0]
    total = int(leaf_sizes.sum())

    if total == 0 or n_leaves == 0:
        return

    positions, seg_ids = _ragged_arange(leaf_starts, leaf_sizes)
    pts = sorted_points[positions]  # (total, D)

    seg_min = torch.full((n_leaves, D), float("inf"), dtype=dtype, device=device)
    seg_max = torch.full((n_leaves, D), float("-inf"), dtype=dtype, device=device)
    exp_ids = seg_ids.unsqueeze(1).expand_as(pts)
    seg_min.scatter_reduce_(0, exp_ids, pts, reduce="amin", include_self=True)
    seg_max.scatter_reduce_(0, exp_ids, pts, reduce="amax", include_self=True)

    aabb_min_buf[leaf_nids] = seg_min
    aabb_max_buf[leaf_nids] = seg_max


def _fill_leaf_total_areas(
    leaf_nids: Int[torch.Tensor, " n_leaves"],
    leaf_starts: Int[torch.Tensor, " n_leaves"],
    leaf_sizes: Int[torch.Tensor, " n_leaves"],
    sorted_areas: Float[torch.Tensor, " n_sorted_sources"],
    total_area_buf: Float[torch.Tensor, " n_nodes"],
) -> None:
    """Compute total area per leaf node (in-place)."""
    device = leaf_nids.device
    n_leaves = leaf_nids.shape[0]
    total = int(leaf_sizes.sum())

    if total == 0 or n_leaves == 0:
        return

    positions, seg_ids = _ragged_arange(leaf_starts, leaf_sizes)
    areas = sorted_areas[positions]

    leaf_areas = torch.zeros(n_leaves, dtype=areas.dtype, device=device)
    leaf_areas.scatter_add_(0, seg_ids, areas)

    total_area_buf[leaf_nids] = leaf_areas


def _aggregate_source_data_leaves(
    sorted_source_data: TensorDict,
    sorted_areas: Float[torch.Tensor, " n_sorted_sources"],
    seg_ids: Int[torch.Tensor, " n_sorted_sources"],
    n_leaves: int,
    leaf_node_ids: Int[torch.Tensor, " n_leaves"],
    leaf_total_areas: Float[torch.Tensor, " n_leaves"],
    n_nodes: int,
    device: torch.device,
) -> TensorDict:
    """Compute area-weighted average source data for leaf nodes.

    Returns a TensorDict with ``batch_size=(n_nodes,)`` where only
    leaf entries are populated (internal nodes are zeros, filled by
    bottom-up propagation).
    """
    safe_areas = leaf_total_areas.clamp(min=1e-30)

    def _aggregate_leaf(tensor: torch.Tensor) -> torch.Tensor:
        trailing_shape = tensor.shape[1:]
        flat = tensor.reshape(tensor.shape[0], -1)  # (n_sorted_sources, F)

        weighted_sum = _segmented_weighted_sum(
            flat, sorted_areas, seg_ids, n_leaves
        )
        avg = weighted_sum / safe_areas.unsqueeze(-1)

        out = torch.zeros(
            (n_nodes,) + trailing_shape,
            dtype=tensor.dtype,
            device=device,
        )
        out_flat = out.reshape(n_nodes, -1)
        out_flat[leaf_node_ids] = avg
        return out.reshape((n_nodes,) + trailing_shape)

    return sorted_source_data.apply(_aggregate_leaf, batch_size=[n_nodes])


def _propagate_centroids_bottom_up(
    centroid_buf: Float[torch.Tensor, "n_nodes n_dims"],
    node_source_data: TensorDict | None,
    left_child: Int[torch.Tensor, " n_nodes"],
    right_child: Int[torch.Tensor, " n_nodes"],
    total_area: Float[torch.Tensor, " n_nodes"],
    n_nodes: int,
) -> None:
    """Propagate centroids and source data from leaves to root (in-place).

    Internal node centroid = area-weighted average of its children's centroids.
    Internal node source data = area-weighted average of its children's data.
    """
    # Identify internal nodes: those with valid children
    is_internal = left_child[:n_nodes] >= 0
    internal_ids = torch.where(is_internal)[0]

    if internal_ids.numel() == 0:
        return

    # BFS from root discovers which internal nodes live at each depth.
    # We then process reversed(depth_levels) so the deepest internal
    # nodes (whose children are leaves with known values) are computed
    # first, and each shallower level can read correct children.
    device = centroid_buf.device
    depth_levels: list[torch.Tensor] = []
    current_level = torch.tensor([0], dtype=torch.long, device=device)

    while current_level.numel() > 0:
        # Filter to internal nodes at this level
        mask = is_internal[current_level]
        internal_at_level = current_level[mask]
        if internal_at_level.numel() > 0:
            depth_levels.append(internal_at_level)

        # Expand to children
        next_parts: list[torch.Tensor] = []
        if internal_at_level.numel() > 0:
            left = left_child[internal_at_level]
            right = right_child[internal_at_level]
            valid_l = left >= 0
            valid_r = right >= 0
            if valid_l.any():
                next_parts.append(left[valid_l])
            if valid_r.any():
                next_parts.append(right[valid_r])

        current_level = torch.cat(next_parts) if next_parts else torch.empty(
            0, dtype=torch.long, device=device
        )

    # Process from deepest level to root
    for level_ids in reversed(depth_levels):
        left = left_child[level_ids]
        right = right_child[level_ids]

        left_area = total_area[left]
        right_area = total_area[right]
        total = (left_area + right_area).clamp(min=1e-30)

        # 1D base weights; each consumer unsqueezes as needed for its rank
        w_left_1d = left_area / total   # (n,)
        w_right_1d = right_area / total  # (n,)

        centroid_buf[level_ids] = (
            centroid_buf[left] * w_left_1d.unsqueeze(-1)
            + centroid_buf[right] * w_right_1d.unsqueeze(-1)
        )

        if node_source_data is not None:
            for key in node_source_data.keys(include_nested=True, leaves_only=True):
                val_left = node_source_data[key][left]
                val_right = node_source_data[key][right]
                w_l = w_left_1d
                w_r = w_right_1d
                while w_l.ndim < val_left.ndim:
                    w_l = w_l.unsqueeze(-1)
                    w_r = w_r.unsqueeze(-1)
                node_source_data[key][level_ids] = (
                    val_left * w_l + val_right * w_r
                )
