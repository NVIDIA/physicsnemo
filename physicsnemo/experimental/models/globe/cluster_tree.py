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

"""Spatial cluster tree for Barnes-Hut acceleration of GLOBE kernel evaluation.

This module provides a GPU-compatible hierarchical spatial decomposition over a
set of points, designed for Barnes-Hut-style O(N log N) kernel acceleration.
The tree enables replacing the O(N^2) all-to-all source-target interaction with
a mix of exact near-field and approximate far-field evaluations.

Construction uses a morton-code-based Linear BVH (LBVH) algorithm identical in
structure to :mod:`physicsnemo.mesh.spatial.bvh`, producing a binary radix tree
stored as flat tensors for GPU compatibility.

The tree supports two key operations beyond construction:

1. **Aggregate computation**: bottom-up computation of per-node aggregate source
   data (centroids, normals, features) used as "virtual source" representations
   for far-field cluster approximations.

2. **Interaction pair finding**: breadth-first traversal that classifies each
   (target, node) pair as near-field (requiring exact kernel evaluation) or
   far-field (using the monopole approximation), returning an ``InteractionPlan``.
"""

from dataclasses import dataclass
from math import ceil

import torch
from jaxtyping import Float, Int
from tensordict import TensorDict, tensorclass

from physicsnemo.mesh.spatial.bvh import _compute_morton_codes


# ---------------------------------------------------------------------------
# InteractionPlan: the output of tree traversal
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InteractionPlan:
    r"""Result of a Barnes-Hut tree traversal: the sets of near-field and
    far-field interaction pairs that together cover all source contributions
    for every target point.

    Near-field pairs ``(near_target_ids[i], near_source_ids[i])`` require
    exact kernel evaluation. Far-field pairs ``(far_target_ids[i],
    far_node_ids[i])`` use the cluster's aggregate data as a monopole
    approximation.

    All index tensors are ``int64`` on the same device as the tree.
    Pairs are sorted (near by source index, far by node index) for
    memory-coalesced gather operations during kernel evaluation.
    """

    near_target_ids: Int[torch.Tensor, " n_near"]
    near_source_ids: Int[torch.Tensor, " n_near"]
    far_target_ids: Int[torch.Tensor, " n_far"]
    far_node_ids: Int[torch.Tensor, " n_far"]

    @property
    def n_near(self) -> int:
        """Number of near-field (exact) interaction pairs."""
        return self.near_target_ids.shape[0]

    @property
    def n_far(self) -> int:
        """Number of far-field (approximate) interaction pairs."""
        return self.far_target_ids.shape[0]

    @property
    def n_total(self) -> int:
        """Total number of interaction pairs."""
        return self.n_near + self.n_far


# ---------------------------------------------------------------------------
# Segmented weighted reduction helpers
# ---------------------------------------------------------------------------


def _segmented_weighted_sum(
    values: torch.Tensor,
    weights: torch.Tensor,
    seg_ids: torch.Tensor,
    n_segments: int,
) -> torch.Tensor:
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
    leaf_query_indices: torch.Tensor,
    leaf_node_indices: torch.Tensor,
    leaf_start: torch.Tensor,
    leaf_count: torch.Tensor,
    sorted_order: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
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

    cum = counts.cumsum(0)
    offsets = torch.arange(total, dtype=torch.long, device=device)
    offsets = offsets - torch.repeat_interleave(cum - counts, counts)

    sorted_positions = torch.repeat_interleave(starts, counts) + offsets
    expanded_sources = sorted_order[sorted_positions]

    return expanded_queries, expanded_sources


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
                node_total_area=torch.empty(0, dtype=dtype, device=device),
                sorted_source_order=empty_long,
                source_points=points,
                max_depth=torch.tensor(0, dtype=torch.long, device=device),
                batch_size=torch.Size([]),
            )

        ### Sort points by morton code for spatial coherence
        morton_codes = _compute_morton_codes(points)
        sorted_order = morton_codes.argsort(stable=True)  # (n_points,)
        sorted_points = points[sorted_order]  # (n_points, D)
        sorted_areas = areas[sorted_order]  # (n_points,)

        ### Pre-allocate node storage
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
        total_area_buf = torch.zeros(max_nodes, dtype=dtype, device=device)

        # -----------------------------------------------------------
        # Phase 1: Top-down LBVH construction (O(log N) iterations)
        # -----------------------------------------------------------
        seg_starts = torch.tensor([0], dtype=torch.long, device=device)
        seg_ends = torch.tensor([n_points], dtype=torch.long, device=device)
        seg_node_ids = torch.tensor([0], dtype=torch.long, device=device)
        node_count = 1
        actual_depth = 0

        internal_nodes_per_level: list[torch.Tensor] = []

        while len(seg_starts) > 0:
            seg_sizes = seg_ends - seg_starts

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

            ### Process internal segments: split at midpoint, assign children
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

        ### Identify leaf nodes and their source ranges
        is_leaf = self.leaf_count > 0
        leaf_node_ids = torch.where(is_leaf)[0]

        ### Build compact segment IDs: map each sorted source to its leaf node
        if leaf_node_ids.numel() > 0:
            leaf_starts = self.leaf_start[leaf_node_ids]
            leaf_counts = self.leaf_count[leaf_node_ids]
            # Compact mapping: leaf_node_ids[i] -> i for scatter, then remap
            n_leaves = leaf_node_ids.shape[0]
            compact_ids = torch.repeat_interleave(
                torch.arange(n_leaves, dtype=torch.long, device=device),
                leaf_counts,
            )
            cum = leaf_counts.cumsum(0)
            total = int(leaf_counts.sum())
            offsets = torch.arange(total, dtype=torch.long, device=device)
            offsets = offsets - torch.repeat_interleave(
                cum - leaf_counts, leaf_counts
            )
            positions = torch.repeat_interleave(leaf_starts, leaf_counts) + offsets

            # seg_ids maps sorted position -> compact leaf index
            seg_ids_compact = torch.zeros(
                self.n_sources, dtype=torch.long, device=device
            )
            seg_ids_compact[positions] = compact_ids

        ### Compute leaf centroids: area-weighted mean of source positions
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

        ### Compute leaf aggregate source data
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

    def find_interaction_pairs(
        self,
        target_points: Float[torch.Tensor, "n_targets n_dims"],
        theta: float = 0.5,
    ) -> InteractionPlan:
        r"""Find near-field and far-field interaction pairs via tree traversal.

        Uses the AABB-distance opening angle criterion: a target uses the
        far-field (monopole) approximation for a node when
        :math:`\text{dist}(target, AABB)^2 > \text{diameter}^2 \cdot \theta^2`.

        Parameters
        ----------
        target_points : Float[torch.Tensor, "n_targets n_dims"]
            Target coordinates, shape :math:`(N_{tgt}, D)`.
        theta : float
            Far-field distance threshold: a node is approximated when
            ``dist > diameter * theta``. Larger values are more
            conservative (more exact interactions, higher accuracy,
            slower). Typical values: ``0.3``--``1.0``.

        Returns
        -------
        InteractionPlan
            Near-field and far-field interaction pairs sorted for
            memory-coalesced access.
        """
        n_targets = target_points.shape[0]
        device = target_points.device
        theta_sq = theta * theta

        ### Handle empty tree or empty targets
        if self.n_nodes == 0 or n_targets == 0:
            empty = torch.empty(0, dtype=torch.long, device=device)
            return InteractionPlan(
                near_target_ids=empty,
                near_source_ids=empty.clone(),
                far_target_ids=empty.clone(),
                far_node_ids=empty.clone(),
            )

        ### Initialize: all targets start at root (node 0)
        active_target_ids = torch.arange(n_targets, dtype=torch.long, device=device)
        active_node_ids = torch.zeros(n_targets, dtype=torch.long, device=device)

        near_target_list: list[torch.Tensor] = []
        near_source_list: list[torch.Tensor] = []
        far_target_list: list[torch.Tensor] = []
        far_node_list: list[torch.Tensor] = []

        max_depth = int(self.max_depth.item())

        ### Breadth-first traversal, one iteration per tree level
        for _ in range(max_depth + 1):
            if active_target_ids.numel() == 0:
                break

            ### AABB-distance opening angle test
            pts = target_points[active_target_ids]  # (n_active, D)
            aabb_lo = self.node_aabb_min[active_node_ids]
            aabb_hi = self.node_aabb_max[active_node_ids]

            clamped = torch.clamp(pts, min=aabb_lo, max=aabb_hi)
            dist_sq = (pts - clamped).pow(2).sum(dim=-1)  # (n_active,)

            diam_sq = self.node_diameter_sq[active_node_ids]
            is_far = dist_sq > diam_sq * theta_sq

            ### Separate leaf from internal among non-far nodes
            hit_leaf_count = self.leaf_count[active_node_ids]
            is_leaf = hit_leaf_count > 0

            ### Far-field: accumulate (target, node) pairs
            if is_far.any():
                far_target_list.append(active_target_ids[is_far])
                far_node_list.append(active_node_ids[is_far])

            ### Near-field leaves: expand to (target, source) pairs
            near_leaf = ~is_far & is_leaf
            if near_leaf.any():
                expanded_tgts, expanded_srcs = _expand_leaf_hits(
                    active_target_ids[near_leaf],
                    active_node_ids[near_leaf],
                    self.leaf_start,
                    self.leaf_count,
                    self.sorted_source_order,
                )
                near_target_list.append(expanded_tgts)
                near_source_list.append(expanded_srcs)

            ### Internal near-field: expand to children for next iteration
            expand = ~is_far & ~is_leaf
            if not expand.any():
                break

            exp_targets = active_target_ids[expand]
            exp_nodes = active_node_ids[expand]
            left = self.node_left_child[exp_nodes]
            right = self.node_right_child[exp_nodes]

            valid_left = left >= 0
            valid_right = right >= 0

            parts_t: list[torch.Tensor] = []
            parts_n: list[torch.Tensor] = []
            if valid_left.any():
                parts_t.append(exp_targets[valid_left])
                parts_n.append(left[valid_left])
            if valid_right.any():
                parts_t.append(exp_targets[valid_right])
                parts_n.append(right[valid_right])

            if parts_t:
                active_target_ids = torch.cat(parts_t)
                active_node_ids = torch.cat(parts_n)
            else:
                break

        ### Concatenate accumulated pairs
        if near_target_list:
            near_tgt = torch.cat(near_target_list)
            near_src = torch.cat(near_source_list)
        else:
            near_tgt = torch.empty(0, dtype=torch.long, device=device)
            near_src = torch.empty(0, dtype=torch.long, device=device)

        if far_target_list:
            far_tgt = torch.cat(far_target_list)
            far_nid = torch.cat(far_node_list)
        else:
            far_tgt = torch.empty(0, dtype=torch.long, device=device)
            far_nid = torch.empty(0, dtype=torch.long, device=device)

        ### Sort pairs for memory-coalesced gather during kernel evaluation
        if near_src.numel() > 0:
            sort_order = near_src.argsort(stable=True)
            near_tgt = near_tgt[sort_order]
            near_src = near_src[sort_order]

        if far_nid.numel() > 0:
            sort_order = far_nid.argsort(stable=True)
            far_tgt = far_tgt[sort_order]
            far_nid = far_nid[sort_order]

        return InteractionPlan(
            near_target_ids=near_tgt,
            near_source_ids=near_src,
            far_target_ids=far_tgt,
            far_node_ids=far_nid,
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

    node_centroid: torch.Tensor
    """Area-weighted centroid per node, shape ``(n_nodes, D)``."""

    node_source_data: TensorDict | None
    """Area-weighted average source features per node, or ``None`` if no
    per-source features. Has ``batch_size=(n_nodes,)``."""


# ---------------------------------------------------------------------------
# Internal helpers for tree construction
# ---------------------------------------------------------------------------


def _fill_leaf_aabbs(
    leaf_nids: torch.Tensor,
    leaf_starts: torch.Tensor,
    leaf_sizes: torch.Tensor,
    sorted_points: torch.Tensor,
    aabb_min_buf: torch.Tensor,
    aabb_max_buf: torch.Tensor,
) -> None:
    """Fill AABB buffers for leaf nodes via segmented reduction (in-place)."""
    device = leaf_nids.device
    D = sorted_points.shape[1]
    dtype = sorted_points.dtype
    n_leaves = leaf_nids.shape[0]
    total = int(leaf_sizes.sum())

    if total == 0 or n_leaves == 0:
        return

    seg_ids = torch.repeat_interleave(
        torch.arange(n_leaves, dtype=torch.long, device=device),
        leaf_sizes,
    )
    cum = leaf_sizes.cumsum(0)
    offsets = torch.arange(total, dtype=torch.long, device=device)
    offsets = offsets - torch.repeat_interleave(cum - leaf_sizes, leaf_sizes)
    positions = torch.repeat_interleave(leaf_starts, leaf_sizes) + offsets

    pts = sorted_points[positions]  # (total, D)

    seg_min = torch.full((n_leaves, D), float("inf"), dtype=dtype, device=device)
    seg_max = torch.full((n_leaves, D), float("-inf"), dtype=dtype, device=device)
    exp_ids = seg_ids.unsqueeze(1).expand_as(pts)
    seg_min.scatter_reduce_(0, exp_ids, pts, reduce="amin", include_self=True)
    seg_max.scatter_reduce_(0, exp_ids, pts, reduce="amax", include_self=True)

    aabb_min_buf[leaf_nids] = seg_min
    aabb_max_buf[leaf_nids] = seg_max


def _fill_leaf_total_areas(
    leaf_nids: torch.Tensor,
    leaf_starts: torch.Tensor,
    leaf_sizes: torch.Tensor,
    sorted_areas: torch.Tensor,
    total_area_buf: torch.Tensor,
) -> None:
    """Compute total area per leaf node (in-place)."""
    device = leaf_nids.device
    n_leaves = leaf_nids.shape[0]
    total = int(leaf_sizes.sum())

    if total == 0 or n_leaves == 0:
        return

    seg_ids = torch.repeat_interleave(
        torch.arange(n_leaves, dtype=torch.long, device=device),
        leaf_sizes,
    )
    cum = leaf_sizes.cumsum(0)
    offsets = torch.arange(total, dtype=torch.long, device=device)
    offsets = offsets - torch.repeat_interleave(cum - leaf_sizes, leaf_sizes)
    positions = torch.repeat_interleave(leaf_starts, leaf_sizes) + offsets

    areas = sorted_areas[positions]

    leaf_areas = torch.zeros(n_leaves, dtype=areas.dtype, device=device)
    leaf_areas.scatter_add_(0, seg_ids, areas)

    total_area_buf[leaf_nids] = leaf_areas


def _aggregate_source_data_leaves(
    sorted_source_data: TensorDict,
    sorted_areas: torch.Tensor,
    seg_ids: torch.Tensor,
    n_leaves: int,
    leaf_node_ids: torch.Tensor,
    leaf_total_areas: torch.Tensor,
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
    centroid_buf: torch.Tensor,
    node_source_data: TensorDict | None,
    left_child: torch.Tensor,
    right_child: torch.Tensor,
    total_area: torch.Tensor,
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

    # Build depth ordering by BFS from root
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
