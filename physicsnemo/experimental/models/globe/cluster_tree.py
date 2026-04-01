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

Construction uses the same morton-code-based Linear BVH (LBVH) algorithm as
:mod:`physicsnemo.mesh.spatial.bvh` (morton sort, midpoint splits, bottom-up
AABB propagation), but the resulting data structure differs: ClusterTree stores
additional per-node fields (diameter, subtree ranges, area-weighted aggregates)
needed for the Barnes-Hut opening criterion, dual-tree traversal, and
far-field monopole approximation. The two classes share
:func:`~physicsnemo.mesh.spatial.bvh._compute_morton_codes` and
:func:`~physicsnemo.mesh.spatial._ragged._ragged_arange` but are otherwise
independent.
"""

import logging

import torch
from jaxtyping import Float, Int
from tensordict import TensorDict, tensorclass
from torch.profiler import record_function

from physicsnemo.mesh.spatial._ragged import _ragged_arange
from physicsnemo.mesh.spatial.bvh import _compute_morton_codes

logger = logging.getLogger("globe.cluster_tree")


# ---------------------------------------------------------------------------
# InteractionPlan: the output of tree traversal
# ---------------------------------------------------------------------------


@tensorclass
class DualInteractionPlan:
    r"""Result of a dual-tree Barnes-Hut traversal: four categories of
    interactions that together cover all source contributions for every
    target point.

    **(near, near)**: ``(near_target_ids[i], near_source_ids[i])`` are
    individual target-source pairs requiring exact kernel evaluation.

    **(far, far)**: ``(far_target_node_ids[i], far_source_node_ids[i])``
    are node-to-node pairs where the kernel is evaluated ONCE at the
    node centroids and the result is broadcast to all individual targets
    in the target node.

    **(near, far)**: ``(nf_target_ids[i], nf_source_node_ids[i])`` are
    individual target points paired with source nodes.  The kernel is
    evaluated at ``(target_point, source_centroid)`` using the source
    node's monopole approximation.  No target-side broadcast.

    **(far, near)**: ``(fn_target_node_ids[i], fn_source_ids[i])`` are
    target nodes paired with individual source points.  The kernel is
    evaluated at ``(target_centroid, source_point)`` using exact source
    data, then broadcast to stage-1 survivor targets via the
    ``fn_broadcast_*`` mapping.

    All index tensors are ``int64`` on the same device as the tree.
    """

    near_target_ids: Int[torch.Tensor, " n_near"]
    near_source_ids: Int[torch.Tensor, " n_near"]
    far_target_node_ids: Int[torch.Tensor, " n_far_nodes"]
    far_source_node_ids: Int[torch.Tensor, " n_far_nodes"]
    nf_target_ids: Int[torch.Tensor, " n_nf"]
    nf_source_node_ids: Int[torch.Tensor, " n_nf"]
    fn_target_node_ids: Int[torch.Tensor, " n_fn"]
    fn_source_ids: Int[torch.Tensor, " n_fn"]
    fn_broadcast_targets: Int[torch.Tensor, " n_fn_bcast"]
    fn_broadcast_starts: Int[torch.Tensor, " n_fn"]
    fn_broadcast_counts: Int[torch.Tensor, " n_fn"]

    @property
    def n_near(self) -> int:
        """Number of (near,near) exact individual interaction pairs."""
        return self.near_target_ids.shape[0]

    @property
    def n_far_nodes(self) -> int:
        """Number of (far,far) node-to-node pairs (each = one kernel eval)."""
        return self.far_target_node_ids.shape[0]

    @property
    def n_nf(self) -> int:
        """Number of (near,far) target-point-to-source-node pairs."""
        return self.nf_target_ids.shape[0]

    @property
    def n_fn(self) -> int:
        """Number of (far,near) target-node-to-source-point pairs."""
        return self.fn_target_node_ids.shape[0]

    def validate(self) -> None:
        """Check internal consistency of the interaction plan.

        Verifies shape pairing, non-negativity, and fn_broadcast bounds.
        Raises ``ValueError`` on any inconsistency.  Intended to be called
        behind a ``not torch.compiler.is_compiling()`` guard so it is
        zero-cost under ``torch.compile``.

        Raises
        ------
        ValueError
            If any internal consistency check fails.
        """
        ### Shape pairing: matched tensor pairs must have identical lengths
        pairs: list[tuple[str, torch.Tensor, str, torch.Tensor]] = [
            ("near_target_ids", self.near_target_ids,
             "near_source_ids", self.near_source_ids),
            ("far_target_node_ids", self.far_target_node_ids,
             "far_source_node_ids", self.far_source_node_ids),
            ("nf_target_ids", self.nf_target_ids,
             "nf_source_node_ids", self.nf_source_node_ids),
            ("fn_target_node_ids", self.fn_target_node_ids,
             "fn_source_ids", self.fn_source_ids),
        ]
        for name_a, a, name_b, b in pairs:
            if a.shape != b.shape:
                raise ValueError(
                    f"Shape mismatch: {name_a}.shape={a.shape!r} != "
                    f"{name_b}.shape={b.shape!r}"
                )

        ### fn_broadcast tensors must be consistently sized
        n_fn = self.fn_source_ids.shape[0]
        for name, tensor in [
            ("fn_broadcast_starts", self.fn_broadcast_starts),
            ("fn_broadcast_counts", self.fn_broadcast_counts),
        ]:
            if tensor.shape != (n_fn,):
                raise ValueError(
                    f"{name}.shape={tensor.shape!r}, expected ({n_fn},)"
                )

        ### Non-negativity
        for name, tensor in [
            ("fn_broadcast_starts", self.fn_broadcast_starts),
            ("fn_broadcast_counts", self.fn_broadcast_counts),
        ]:
            if tensor.numel() > 0 and (tensor < 0).any():
                raise ValueError(f"{name} contains negative values")

        ### fn_broadcast bounds: every (start, count) range with count > 0
        ### must fit within fn_broadcast_targets.  Zero-count entries are
        ### no-ops whose starts are never dereferenced.
        if n_fn > 0:
            nonzero = self.fn_broadcast_counts > 0
            if nonzero.any():
                ends = self.fn_broadcast_starts[nonzero] + self.fn_broadcast_counts[nonzero]
                max_end = ends.max().item()
                bcast_len = self.fn_broadcast_targets.shape[0]
                if max_end > bcast_len:
                    raise ValueError(
                        f"fn_broadcast out of bounds: max(starts + counts)="
                        f"{max_end} > fn_broadcast_targets.shape[0]={bcast_len}"
                    )


# ---------------------------------------------------------------------------
# Segmented reduction helpers
# ---------------------------------------------------------------------------


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


def _expand_dual_leaf_hits(
    target_leaf_ids: Int[torch.Tensor, " n_leaf_pairs"],
    source_leaf_ids: Int[torch.Tensor, " n_leaf_pairs"],
    target_tree: "ClusterTree",
    source_tree: "ClusterTree",
    theta: float,
) -> tuple[
    Int[torch.Tensor, " n_near"], Int[torch.Tensor, " n_near"],
    Int[torch.Tensor, " n_nf"], Int[torch.Tensor, " n_nf"],
    Int[torch.Tensor, " n_fn"], Int[torch.Tensor, " n_fn"],
    Int[torch.Tensor, " n_fn_bcast"],
    Int[torch.Tensor, " n_fn"], Int[torch.Tensor, " n_fn"],
]:
    """Expand ``(target_leaf, source_leaf)`` pairs with two-stage filtering.

    Applies two sequential per-point tests to classify each (target, source)
    interaction within a leaf pair:

    **Stage 1 (per-target)**: Test each target against the source leaf AABB.
    Targets that pass become **(near, far)** - they use the source monopole.
    Targets that fail are "survivors" and proceed to stage 2.

    **Stage 2 (per-source)**: Test each source against the target leaf AABB.
    Sources that pass become **(far, near)** - evaluated at the target
    centroid and broadcast to all survivors.  Sources that fail produce
    **(near, near)** Cartesian product pairs with the survivors.

    The two stages are independent (different AABBs) and sequential (stage 2
    only applies to survivors), so no (target, source) pair is double-counted.

    Returns
    -------
    near_target_ids, near_source_ids : torch.Tensor
        (near, near) individual target-source pairs.
    nf_target_ids, nf_source_node_ids : torch.Tensor
        (near, far) individual target to source-node pairs.
    fn_target_node_ids, fn_source_ids : torch.Tensor
        (far, near) target-node to individual source pairs.
    fn_broadcast_targets : torch.Tensor
        Survivor target IDs sorted by leaf pair, for (far, near) broadcast.
    fn_broadcast_starts, fn_broadcast_counts : torch.Tensor
        Per-fn-pair offset/count into ``fn_broadcast_targets``.
    """
    device = target_leaf_ids.device
    theta_sq = theta * theta
    n_pairs = target_leaf_ids.shape[0]

    def _empty_result():
        e = torch.empty(0, dtype=torch.long, device=device)
        return e, e.clone(), e.clone(), e.clone(), e.clone(), e.clone(), e.clone(), e.clone(), e.clone()

    if n_pairs == 0:
        return _empty_result()

    t_starts = target_tree.leaf_start[target_leaf_ids]
    t_counts = target_tree.leaf_count[target_leaf_ids]
    s_starts = source_tree.leaf_start[source_leaf_ids]
    s_counts = source_tree.leaf_count[source_leaf_ids]

    # ==================================================================
    # Stage 1: per-target test against source leaf AABBs
    # ==================================================================
    positions_t, leaf_pair_ids_t = _ragged_arange(t_starts, t_counts)
    target_point_ids = target_tree.sorted_source_order[positions_t]
    target_pts = target_tree.source_points[target_point_ids]

    src_leaf_per_target = source_leaf_ids[leaf_pair_ids_t]
    clamped_t = torch.clamp(
        target_pts,
        min=source_tree.node_aabb_min[src_leaf_per_target],
        max=source_tree.node_aabb_max[src_leaf_per_target],
    )
    dist_sq_t = (target_pts - clamped_t).pow(2).sum(dim=-1)
    target_is_far = dist_sq_t * theta_sq > source_tree.node_diameter_sq[src_leaf_per_target]

    ### (near, far) output
    nf_target_ids = target_point_ids[target_is_far]
    nf_source_node_ids = src_leaf_per_target[target_is_far]

    ### Survivors: targets that failed the per-target test
    surv_mask = ~target_is_far
    if not surv_mask.any():
        e = torch.empty(0, dtype=torch.long, device=device)
        return e, e.clone(), nf_target_ids, nf_source_node_ids, e.clone(), e.clone(), e.clone(), e.clone(), e.clone()

    surv_point_ids = target_point_ids[surv_mask]
    surv_lp_ids = leaf_pair_ids_t[surv_mask]

    # ==================================================================
    # Stage 2: per-source test against target leaf AABBs
    # ==================================================================
    positions_s, leaf_pair_ids_s = _ragged_arange(s_starts, s_counts)
    src_point_ids = source_tree.sorted_source_order[positions_s]
    src_pts = source_tree.source_points[src_point_ids]

    tgt_leaf_per_src = target_leaf_ids[leaf_pair_ids_s]
    clamped_s = torch.clamp(
        src_pts,
        min=target_tree.node_aabb_min[tgt_leaf_per_src],
        max=target_tree.node_aabb_max[tgt_leaf_per_src],
    )
    dist_sq_s = (src_pts - clamped_s).pow(2).sum(dim=-1)
    source_is_far = dist_sq_s * theta_sq > target_tree.node_diameter_sq[tgt_leaf_per_src]

    ### (far, near) output: source points far from the target leaf
    fn_source_ids = src_point_ids[source_is_far]
    fn_target_node_ids = tgt_leaf_per_src[source_is_far]
    fn_lp_ids = leaf_pair_ids_s[source_is_far]

    # ==================================================================
    # Build (far, near) broadcast mapping
    # ==================================================================
    # Group survivors by leaf pair so each fn source can look up its
    # broadcast targets (all survivors from the same leaf pair).
    # Only include survivors from leaf pairs that have fn sources;
    # survivors from all-close leaf pairs are not referenced by any
    # fn_broadcast_starts/counts entry.
    has_fn_source = torch.zeros(n_pairs, dtype=torch.bool, device=device)
    if fn_lp_ids.numel() > 0:
        has_fn_source[fn_lp_ids] = True
    fn_active_mask = has_fn_source[surv_lp_ids]

    active_surv_ids = surv_point_ids[fn_active_mask]
    active_surv_lp_ids = surv_lp_ids[fn_active_mask]

    surv_sort = active_surv_lp_ids.argsort(stable=True)
    fn_broadcast_targets = active_surv_ids[surv_sort]

    surv_counts_per_lp = torch.bincount(active_surv_lp_ids, minlength=n_pairs)
    surv_starts_per_lp = surv_counts_per_lp.cumsum(0) - surv_counts_per_lp

    fn_broadcast_starts = surv_starts_per_lp[fn_lp_ids]
    fn_broadcast_counts = surv_counts_per_lp[fn_lp_ids]

    # ==================================================================
    # Reduced Cartesian product: survivors × close sources only
    # ==================================================================
    close_mask = ~source_is_far
    close_src_ids = src_point_ids[close_mask]
    close_lp_ids = leaf_pair_ids_s[close_mask]

    if close_src_ids.numel() == 0 or surv_point_ids.numel() == 0:
        e = torch.empty(0, dtype=torch.long, device=device)
        return (
            e, e.clone(),
            nf_target_ids, nf_source_node_ids,
            fn_target_node_ids, fn_source_ids,
            fn_broadcast_targets, fn_broadcast_starts, fn_broadcast_counts,
        )

    ### Group close sources by leaf pair for contiguous access
    close_sort = close_lp_ids.argsort(stable=True)
    sorted_close_srcs = close_src_ids[close_sort]
    close_counts_per_lp = torch.bincount(close_lp_ids, minlength=n_pairs)
    close_starts_per_lp = close_counts_per_lp.cumsum(0) - close_counts_per_lp

    ### Each survivor expands against its leaf pair's close sources
    per_surv_close_counts = close_counts_per_lp[surv_lp_ids]
    total_nn = int(per_surv_close_counts.sum())

    if total_nn == 0:
        e = torch.empty(0, dtype=torch.long, device=device)
        return (
            e, e.clone(),
            nf_target_ids, nf_source_node_ids,
            fn_target_node_ids, fn_source_ids,
            fn_broadcast_targets, fn_broadcast_starts, fn_broadcast_counts,
        )

    expanded_near_tgts = torch.repeat_interleave(surv_point_ids, per_surv_close_counts)
    per_surv_close_starts = close_starts_per_lp[surv_lp_ids]
    src_positions_nn, _ = _ragged_arange(per_surv_close_starts, per_surv_close_counts)
    expanded_near_srcs = sorted_close_srcs[src_positions_nn]

    return (
        expanded_near_tgts, expanded_near_srcs,
        nf_target_ids, nf_source_node_ids,
        fn_target_node_ids, fn_source_ids,
        fn_broadcast_targets, fn_broadcast_starts, fn_broadcast_counts,
    )


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
    leaf_node_ids : torch.Tensor
        Indices of leaf nodes, shape ``(n_leaves,)``.  Precomputed during
        tree construction so ``compute_source_aggregates`` avoids a
        data-dependent ``torch.where`` that would break ``torch.compile``.
    leaf_seg_ids : torch.Tensor
        Per-source compact leaf segment ID in sorted order, shape
        ``(n_sources,)``.  Maps each source to the index of its
        containing leaf within ``leaf_node_ids``, used for segmented
        reductions in ``compute_source_aggregates``.
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
    internal_level_ids: torch.Tensor
    internal_level_offsets: torch.Tensor
    # internal_level_ids and internal_level_offsets store the tree's
    # internal node IDs in CSR-packed level order (shallowest first).
    # Computed once during from_points() and reused by all bottom-up
    # propagation routines (_propagate_centroids_bottom_up,
    # _compute_node_strengths) to avoid recomputing the BFS traversal
    # that discovers this ordering.  Stored as tensors (not a Python
    # list) so they participate in tensorclass .to(device) moves.
    leaf_node_ids: torch.Tensor
    leaf_seg_ids: torch.Tensor

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

    @property
    def n_leaves(self) -> int:
        """Number of leaf nodes in the tree."""
        return self.leaf_node_ids.shape[0]

    @property
    def internal_nodes_per_level(self) -> list[torch.Tensor]:
        """Internal node IDs grouped by tree depth, shallowest first.

        Reconstructed from CSR-packed ``internal_level_ids`` and
        ``internal_level_offsets`` tensors that are computed once during
        tree construction in :meth:`from_points`.
        """
        offsets = self.internal_level_offsets
        return [
            self.internal_level_ids[offsets[i] : offsets[i + 1]]
            for i in range(len(offsets) - 1)
        ]

    @classmethod
    def from_points(
        cls,
        points: Float[torch.Tensor, "n_points n_dims"],
        *,
        leaf_size: int = 1,
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
                internal_level_ids=empty_long,
                internal_level_offsets=torch.tensor([0], dtype=torch.long, device=device),
                leaf_node_ids=empty_long,
                leaf_seg_ids=empty_long,
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

        ### Precompute leaf indices and per-source segment IDs so that
        ### compute_source_aggregates() avoids a data-dependent
        ### torch.where() that would break torch.compile tracing.
        leaf_count_trimmed = leaf_count_buf[:node_count]
        _leaf_node_ids = torch.where(leaf_count_trimmed > 0)[0]
        _leaf_starts = leaf_start_buf[_leaf_node_ids]
        _leaf_counts = leaf_count_trimmed[_leaf_node_ids]
        _positions, _compact_ids = _ragged_arange(
            _leaf_starts, _leaf_counts, total=n_points,
        )
        _leaf_seg_ids = torch.zeros(n_points, dtype=torch.long, device=device)
        _leaf_seg_ids[_positions] = _compact_ids

        logger.debug(
            "ClusterTree: %d points -> %d nodes (%d leaves), "
            "depth %d, leaf_size=%d",
            n_points, node_count, _leaf_node_ids.shape[0], actual_depth,
            leaf_size,
        )

        ### Pack the per-level internal node IDs into CSR tensors so they
        ### survive as tensorclass attributes (device-safe, no BFS needed later).
        _level_ids = (
            torch.cat(internal_nodes_per_level)
            if internal_nodes_per_level
            else torch.empty(0, dtype=torch.long, device=device)
        )
        _level_lengths = torch.tensor(
            [len(t) for t in internal_nodes_per_level],
            dtype=torch.long,
            device=device,
        )
        _level_offsets = torch.cat([
            torch.zeros(1, dtype=torch.long, device=device),
            _level_lengths.cumsum(0),
        ])

        return cls(
            node_aabb_min=aabb_min_trimmed,
            node_aabb_max=aabb_max_trimmed,
            node_diameter_sq=diameter_sq,
            node_left_child=left_child[:node_count],
            node_right_child=right_child[:node_count],
            leaf_start=leaf_start_buf[:node_count],
            leaf_count=leaf_count_trimmed,
            node_range_start=range_start_buf[:node_count],
            node_range_count=range_count_buf[:node_count],
            node_total_area=total_area_buf[:node_count],
            sorted_source_order=sorted_order,
            source_points=points,
            max_depth=torch.tensor(actual_depth, dtype=torch.long, device=device),
            internal_level_ids=_level_ids,
            internal_level_offsets=_level_offsets,
            leaf_node_ids=_leaf_node_ids,
            leaf_seg_ids=_leaf_seg_ids,
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

        ### Leaf aggregation: compute per-leaf centroids and source data.
        ### leaf_node_ids and leaf_seg_ids were precomputed during tree
        ### construction (from_points) to avoid data-dependent torch.where
        ### and _ragged_arange calls that would break torch.compile.
        with record_function("cluster_tree::leaf_aggregation"):
            leaf_node_ids = self.leaf_node_ids
            n_leaves = leaf_node_ids.shape[0]
            seg_ids_compact = self.leaf_seg_ids

            sorted_points = source_points[self.sorted_source_order]
            sorted_areas = areas[self.sorted_source_order]

            centroid_buf = torch.zeros(n_nodes, D, dtype=dtype, device=device)

            leaf_centroids = _segmented_weighted_sum(
                sorted_points, sorted_areas, seg_ids_compact, n_leaves
            )
            leaf_total_areas = self.node_total_area[leaf_node_ids]
            safe_areas = leaf_total_areas.clamp(min=1e-30)
            leaf_centroids = leaf_centroids / safe_areas.unsqueeze(-1)
            centroid_buf[leaf_node_ids] = leaf_centroids

            node_source_data: TensorDict | None = None
            if source_data is not None:
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
                self.internal_nodes_per_level,
            )

        return SourceAggregates(
            node_centroid=centroid_buf,
            node_source_data=node_source_data,
        )

    def find_dual_interaction_pairs(
        self,
        target_tree: "ClusterTree",
        theta: float = 1.0,
        *,
        expand_far_targets: bool = False,
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
        expand_far_targets : bool, optional, default=False
            If ``True``, far-field node pairs are expanded to individual
            target points, converting ``(far, far)`` entries into
            ``(near, far)`` entries.  This eliminates the target-side
            centroid approximation (and the blocky spatial artifacts it
            produces) at the cost of more kernel evaluations while
            preserving the source-side monopole speedup.

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
                nf_target_ids=empty.clone(),
                nf_source_node_ids=empty.clone(),
                fn_target_node_ids=empty.clone(),
                fn_source_ids=empty.clone(),
                fn_broadcast_targets=empty.clone(),
                fn_broadcast_starts=empty.clone(),
                fn_broadcast_counts=empty.clone(),
            )

        with record_function("cluster_tree::dual_traversal"):
            ### Initialize: root-to-root pair
            active_tgt_nodes = torch.zeros(1, dtype=torch.long, device=device)
            active_src_nodes = torch.zeros(1, dtype=torch.long, device=device)

            near_target_list: list[torch.Tensor] = []
            near_source_list: list[torch.Tensor] = []
            far_tgt_node_list: list[torch.Tensor] = []
            far_src_node_list: list[torch.Tensor] = []
            nf_target_list: list[torch.Tensor] = []
            nf_source_node_list: list[torch.Tensor] = []
            fn_tgt_node_list: list[torch.Tensor] = []
            fn_src_list: list[torch.Tensor] = []
            fn_bcast_targets_list: list[torch.Tensor] = []
            fn_bcast_starts_list: list[torch.Tensor] = []
            fn_bcast_counts_list: list[torch.Tensor] = []
            fn_bcast_offset = 0

            max_iters = int(target_tree.max_depth.item()) + int(source_tree.max_depth.item()) + 1
            depth = 0

            for depth in range(max_iters):
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
                    if expand_far_targets:
                        # Expand target nodes to individual points,
                        # converting (far,far) → (near,far).
                        far_tgt_nids = active_tgt_nodes[is_far]
                        far_src_nids = active_src_nodes[is_far]
                        starts = target_tree.node_range_start[far_tgt_nids]
                        counts = target_tree.node_range_count[far_tgt_nids]
                        positions, pair_ids = _ragged_arange(starts, counts)
                        nf_target_list.append(
                            target_tree.sorted_source_order[positions]
                        )
                        nf_source_node_list.append(far_src_nids[pair_ids])
                    else:
                        far_tgt_node_list.append(active_tgt_nodes[is_far])
                        far_src_node_list.append(active_src_nodes[is_far])

                ### 2. Near-field, both leaves: two-stage filtered expansion.
                # Stage 1 (per-target) -> (near,far).
                # Stage 2 (per-source) -> (far,near).
                # Remainder -> (near,near).
                near_leaf_leaf = (~is_far) & is_leaf_T & is_leaf_S
                if near_leaf_leaf.any():
                    (
                        nn_tgts, nn_srcs,
                        nf_tgts, nf_snids,
                        fn_tnids, fn_sids,
                        fn_btgts, fn_bstarts, fn_bcounts,
                    ) = _expand_dual_leaf_hits(
                        active_tgt_nodes[near_leaf_leaf],
                        active_src_nodes[near_leaf_leaf],
                        target_tree,
                        source_tree,
                        theta,
                    )
                    near_target_list.append(nn_tgts)
                    near_source_list.append(nn_srcs)
                    nf_target_list.append(nf_tgts)
                    nf_source_node_list.append(nf_snids)
                    fn_tgt_node_list.append(fn_tnids)
                    fn_src_list.append(fn_sids)
                    fn_bcast_targets_list.append(fn_btgts)
                    fn_bcast_starts_list.append(fn_bstarts + fn_bcast_offset)
                    fn_bcast_counts_list.append(fn_bcounts)
                    fn_bcast_offset += fn_btgts.shape[0]

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

            if nf_target_list:
                nf_tgt = torch.cat(nf_target_list)
                nf_snid = torch.cat(nf_source_node_list)
            else:
                nf_tgt = torch.empty(0, dtype=torch.long, device=device)
                nf_snid = torch.empty(0, dtype=torch.long, device=device)

            if fn_tgt_node_list:
                fn_tnid = torch.cat(fn_tgt_node_list)
                fn_sid = torch.cat(fn_src_list)
                fn_btgts = torch.cat(fn_bcast_targets_list)
                fn_bstarts = torch.cat(fn_bcast_starts_list)
                fn_bcounts = torch.cat(fn_bcast_counts_list)
            else:
                fn_tnid = torch.empty(0, dtype=torch.long, device=device)
                fn_sid = torch.empty(0, dtype=torch.long, device=device)
                fn_btgts = torch.empty(0, dtype=torch.long, device=device)
                fn_bstarts = torch.empty(0, dtype=torch.long, device=device)
                fn_bcounts = torch.empty(0, dtype=torch.long, device=device)

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

            ### Sort (near,far) pairs by source node for coalesced gather
            if nf_snid.numel() > 0:
                sort_order = nf_snid.argsort(stable=True)
                nf_tgt = nf_tgt[sort_order]
                nf_snid = nf_snid[sort_order]

            ### Sort (far,near) pairs by source index for coalesced gather
            if fn_sid.numel() > 0:
                sort_order = fn_sid.argsort(stable=True)
                fn_tnid = fn_tnid[sort_order]
                fn_sid = fn_sid[sort_order]
                fn_bstarts = fn_bstarts[sort_order]
                fn_bcounts = fn_bcounts[sort_order]

        plan = DualInteractionPlan(
            near_target_ids=near_tgt,
            near_source_ids=near_src,
            far_target_node_ids=far_tgt_nid,
            far_source_node_ids=far_src_nid,
            nf_target_ids=nf_tgt,
            nf_source_node_ids=nf_snid,
            fn_target_node_ids=fn_tnid,
            fn_source_ids=fn_sid,
            fn_broadcast_targets=fn_btgts,
            fn_broadcast_starts=fn_bstarts,
            fn_broadcast_counts=fn_bcounts,
        )

        if not torch.compiler.is_compiling():
            plan.validate()

        is_self = target_tree is self
        logger.debug(
            "dual traversal: %d near + %d nf + %d fn + %d far_node pairs, "
            "theta=%.2f, self_interaction=%s, %d iterations",
            plan.n_near, plan.n_nf, plan.n_fn, plan.n_far_nodes,
            theta, is_self, depth,
        )

        return plan


# ---------------------------------------------------------------------------
# SourceAggregates: per-node aggregate data for far-field approximation
# ---------------------------------------------------------------------------


@tensorclass
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


### Disabled for torch.compile: this function iterates over a
### variable-length list (depth_levels), whose length equals the tree
### depth. Dynamo unrolls this loop and specializes on the length,
### causing recompilation every time a new tree depth is encountered
### (each airfoil mesh produces a different-depth tree). Disabling
### compilation here produces one clean graph break at the function
### boundary instead of per-depth-level recompilation storms.
@torch.compiler.disable
def _propagate_centroids_bottom_up(
    centroid_buf: Float[torch.Tensor, "n_nodes n_dims"],
    node_source_data: TensorDict | None,
    left_child: Int[torch.Tensor, " n_nodes"],
    right_child: Int[torch.Tensor, " n_nodes"],
    total_area: Float[torch.Tensor, " n_nodes"],
    depth_levels: list[torch.Tensor],
) -> None:
    """Propagate centroids and source data from leaves to root (in-place).

    Internal node centroid = area-weighted average of its children's centroids.
    Internal node source data = area-weighted average of its children's data.

    Parameters
    ----------
    centroid_buf : Float[torch.Tensor, "n_nodes n_dims"]
        Buffer of per-node centroids (leaf values pre-filled, internal values
        written by this function).
    node_source_data : TensorDict or None
        Per-node source data to propagate (same structure as centroid_buf).
    left_child : Int[torch.Tensor, "n_nodes"]
        Left child index per node (-1 for leaves).
    right_child : Int[torch.Tensor, "n_nodes"]
        Right child index per node (-1 for leaves).
    total_area : Float[torch.Tensor, "n_nodes"]
        Total source area in each node's subtree.
    depth_levels : list[torch.Tensor]
        Internal node IDs grouped by tree depth (shallowest first),
        from :attr:`ClusterTree.internal_nodes_per_level`.
    """
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
