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

    ### This function is intentionally written without ``if X.any():`` /
    ### ``int(X.sum())`` early-exit branches.  Each such branch was a
    ### CPU-GPU sync point in the dual-traversal hot loop; the sync count
    ### was the dominant CPU stall in profiling.  All downstream operations
    ### (boolean indexing, ``_ragged_arange``, ``argsort``, ``scatter``,
    ### ``torch.bincount``) handle zero-element inputs correctly, so we
    ### let empty intermediate tensors flow through unconditionally.

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

    ### (near, far) output.  ``target_is_far`` is consumed by two
    ### indexings; doing one ``nonzero`` and reusing the integer index
    ### saves one sync (each ``tensor[bool_mask]`` lowers to a
    ### synchronizing ``aten::nonzero`` to size the output).
    far_idx_t = target_is_far.nonzero(as_tuple=True)[0]
    nf_target_ids = target_point_ids[far_idx_t]
    nf_source_node_ids = src_leaf_per_target[far_idx_t]

    ### Survivors: targets that failed the per-target test.  Empty
    ### survivors are fine; downstream ops produce empty tensors.
    ### Same dedup trick as ``far_idx_t`` above.
    surv_idx = (~target_is_far).nonzero(as_tuple=True)[0]
    surv_point_ids = target_point_ids[surv_idx]
    surv_lp_ids = leaf_pair_ids_t[surv_idx]

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

    ### (far, near) output: source points far from the target leaf.
    ### Three indexings off the same mask collapse to one ``nonzero`` +
    ### integer indexing, saving two syncs.
    far_idx_s = source_is_far.nonzero(as_tuple=True)[0]
    fn_source_ids = src_point_ids[far_idx_s]
    fn_target_node_ids = tgt_leaf_per_src[far_idx_s]
    fn_lp_ids = leaf_pair_ids_s[far_idx_s]

    # ==================================================================
    # Build (far, near) broadcast mapping
    # ==================================================================
    # Group survivors by leaf pair so each fn source can look up its
    # broadcast targets (all survivors from the same leaf pair).
    # Only include survivors from leaf pairs that have fn sources;
    # survivors from all-close leaf pairs are not referenced by any
    # fn_broadcast_starts/counts entry.
    # ``index_put_`` with empty indices is a no-op, so no shape branch.
    has_fn_source = torch.zeros(n_pairs, dtype=torch.bool, device=device)
    has_fn_source[fn_lp_ids] = True
    fn_active_mask = has_fn_source[surv_lp_ids]

    ### Shared mask -> dedup the boolean indexing (saves one sync).
    fn_active_idx = fn_active_mask.nonzero(as_tuple=True)[0]
    active_surv_ids = surv_point_ids[fn_active_idx]
    active_surv_lp_ids = surv_lp_ids[fn_active_idx]

    surv_sort = active_surv_lp_ids.argsort(stable=True)
    fn_broadcast_targets = active_surv_ids[surv_sort]

    surv_counts_per_lp = torch.bincount(active_surv_lp_ids, minlength=n_pairs)
    surv_starts_per_lp = surv_counts_per_lp.cumsum(0) - surv_counts_per_lp

    fn_broadcast_starts = surv_starts_per_lp[fn_lp_ids]
    fn_broadcast_counts = surv_counts_per_lp[fn_lp_ids]

    # ==================================================================
    # Reduced Cartesian product: survivors × close sources only
    # ==================================================================
    ### Survivors of the per-source far test.  Shared mask + dedup as
    ### above.
    close_idx = (~source_is_far).nonzero(as_tuple=True)[0]
    close_src_ids = src_point_ids[close_idx]
    close_lp_ids = leaf_pair_ids_s[close_idx]

    ### Group close sources by leaf pair for contiguous access.  When
    ### either side is empty the per-segment counts are all zero and
    ### the expansion below produces empty (near, near) output - no
    ### early-exit needed.
    close_sort = close_lp_ids.argsort(stable=True)
    sorted_close_srcs = close_src_ids[close_sort]
    close_counts_per_lp = torch.bincount(close_lp_ids, minlength=n_pairs)
    close_starts_per_lp = close_counts_per_lp.cumsum(0) - close_counts_per_lp

    ### Each survivor expands against its leaf pair's close sources.
    per_surv_close_counts = close_counts_per_lp[surv_lp_ids]
    per_surv_close_starts = close_starts_per_lp[surv_lp_ids]

    expanded_near_tgts = torch.repeat_interleave(
        surv_point_ids, per_surv_close_counts
    )
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

            ### Defer leaf-segment processing to a single end-of-loop pass.
            # Per-iter ``torch.where(is_leaf_seg)[0]`` was a CPU-GPU sync
            # point on every level of every tree (~16 levels x 4 trees x
            # 28 samples).  We instead accumulate (node_id, start, size,
            # validity) for each segment seen during the loop and pay one
            # boolean compaction at the end.  ``torch.where(is_internal_seg)``
            # remains in-loop because the next iteration's active segments
            # are derived from this iteration's internals.
            leaf_seg_node_ids: list[torch.Tensor] = []
            leaf_seg_starts: list[torch.Tensor] = []
            leaf_seg_sizes: list[torch.Tensor] = []
            leaf_seg_validity: list[torch.Tensor] = []

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

                ### Defer leaf processing: accumulate, compact once at end.
                leaf_seg_node_ids.append(seg_node_ids)
                leaf_seg_starts.append(seg_starts)
                leaf_seg_sizes.append(seg_sizes)
                leaf_seg_validity.append(is_leaf_seg)

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

            ### Single-pass leaf fill: one boolean compaction across all
            ### levels, then one combined ``_fill_leaf_aggregates`` call
            ### instead of one per level.
            if leaf_seg_node_ids:
                all_leaf_validity = torch.cat(leaf_seg_validity)
                leaf_nids = torch.cat(leaf_seg_node_ids)[all_leaf_validity]
                l_starts = torch.cat(leaf_seg_starts)[all_leaf_validity]
                l_sizes = torch.cat(leaf_seg_sizes)[all_leaf_validity]

                leaf_start_buf[leaf_nids] = l_starts
                leaf_count_buf[leaf_nids] = l_sizes
                _fill_leaf_aggregates(
                    leaf_nids,
                    l_starts,
                    l_sizes,
                    sorted_points,
                    sorted_areas,
                    aabb_min_buf,
                    aabb_max_buf,
                    total_area_buf,
                )

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

        leaf_count_trimmed = leaf_count_buf[:node_count]
        logger.debug(
            "ClusterTree: %d points -> %d nodes, depth %d, leaf_size=%d",
            n_points, node_count, actual_depth, leaf_size,
        )

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
        device = source_points.device
        dtype = source_points.dtype
        D = source_points.shape[1]
        n_nodes = self.n_nodes
        if n_nodes == 0:
            return SourceAggregates(
                node_centroid=torch.empty((0, D), dtype=dtype, device=device),
                node_source_data=None,
            )

        ### Range-sum aggregation via morton-sorted prefix sums.
        # Each node covers a contiguous range
        # [node_range_start, node_range_start + node_range_count) in
        # morton-sorted source order, so any node-subtree sum is just
        # ``prefix[end] - prefix[start]``.  This replaces the old
        # leaf-aggregation + bottom-up Python loop, which were the
        # dominant CPU + GPU costs in ``compute_source_aggregates``
        # (~2 s combined per training step in profiling).
        #
        # The cumsum and the range subtract are done in fp64 because fp32
        # suffers catastrophic cancellation when ``range_sum << cumsum_total``,
        # which is the regime of small leaves (``leaf_size=1``) in a large
        # tree built over offset (e.g. all-positive) coordinates.  At
        # drivaer scale (``N=1M``, coords ~5 m), fp32 leaf centroids had
        # median ~2 % relative error and p99 ~100 % wrong.  Lifting the
        # cumsum to fp64 brings this back to fp32 epsilon (~1e-7) and adds
        # <1 % wall-clock to the training step (cumsum is ~2.3x slower in
        # fp64, but cumsum is a tiny fraction of step time).  CUDA fp32
        # cumsum is also non-deterministic across runs (pytorch#75240);
        # fp64 cumsum is much less affected.
        sorted_points = source_points[self.sorted_source_order]
        sorted_areas = areas[self.sorted_source_order]
        weighted_points_64 = (sorted_points * sorted_areas.unsqueeze(-1)).double()

        ### Leading-zero padding makes ``prefix[i]`` the sum of the first
        ### ``i`` elements, so subtraction gives the half-open range sum.
        cumsum_weighted_points = torch.nn.functional.pad(
            torch.cumsum(weighted_points_64, dim=0), (0, 0, 1, 0)
        )

        starts = self.node_range_start
        ends = starts + self.node_range_count
        node_total_weighted_pts = (
            cumsum_weighted_points[ends] - cumsum_weighted_points[starts]
        )
        ### ``self.node_total_area`` was filled during tree construction
        ### via the bottom-up AABB pass; reuse it instead of recomputing.
        ### Promote to fp64 here so the divide stays in fp64 alongside the
        ### range-sum; cast back to ``source_points.dtype`` at the end.
        safe_areas_64 = self.node_total_area.double().clamp(min=1e-30)
        with record_function("cluster_tree::node_centroids"):
            centroid_buf = (
                node_total_weighted_pts / safe_areas_64.unsqueeze(-1)
            ).to(source_points.dtype)

        node_source_data: TensorDict | None = None
        if source_data is not None:
            sorted_source_data = source_data[self.sorted_source_order]
            inv_safe_areas_64 = 1.0 / safe_areas_64

            def _aggregate_via_prefix_sum(tensor: torch.Tensor) -> torch.Tensor:
                trailing_shape = tensor.shape[1:]
                ### Flatten trailing dims so the prefix sum is over a
                ### single feature axis - avoids materialising a
                ### per-feature kernel chain inside ``cumsum``.  Same fp64
                ### upcast rationale as the centroid branch above.
                flat = tensor.reshape(tensor.shape[0], -1)
                weighted_64 = (flat * sorted_areas.unsqueeze(-1)).double()
                cumsum_weighted = torch.nn.functional.pad(
                    torch.cumsum(weighted_64, dim=0), (0, 0, 1, 0)
                )
                node_weighted_sum = cumsum_weighted[ends] - cumsum_weighted[starts]
                node_avg = node_weighted_sum * inv_safe_areas_64.unsqueeze(-1)
                return node_avg.reshape((n_nodes,) + trailing_shape).to(tensor.dtype)

            with record_function("cluster_tree::node_source_data"):
                node_source_data = sorted_source_data.apply(
                    _aggregate_via_prefix_sum, batch_size=[n_nodes]
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

            ### Output streams.  Far/(near,far)-from-far entries are appended
            ### unfiltered together with a per-iteration ``is_far`` validity
            ### mask; one boolean compaction at the very end of the function
            ### eliminates the per-iteration sync that ``active[is_far]``
            ### would otherwise force.
            far_tgt_unfiltered_list: list[torch.Tensor] = []
            far_src_unfiltered_list: list[torch.Tensor] = []
            far_validity_list: list[torch.Tensor] = []

            ### Outputs from leaf-leaf expansion still get appended already-
            ### filtered, since ``_expand_dual_leaf_hits`` works on a
            ### compacted (target_leaf_id, source_leaf_id) batch.  We accept
            ### one boolean indexing sync per iteration here.
            near_target_list: list[torch.Tensor] = []
            near_source_list: list[torch.Tensor] = []
            nf_target_list: list[torch.Tensor] = []
            nf_source_node_list: list[torch.Tensor] = []
            fn_tgt_node_list: list[torch.Tensor] = []
            fn_src_list: list[torch.Tensor] = []
            fn_bcast_targets_list: list[torch.Tensor] = []
            fn_bcast_starts_list: list[torch.Tensor] = []
            fn_bcast_counts_list: list[torch.Tensor] = []
            fn_bcast_offset = 0

            ### Loop bound: every iteration descends at least one tree level
            ### on at least one side, so ``2 * total_levels + safety`` is a
            ### hard upper bound that requires no GPU->CPU read.  Using
            ### ``int(max_depth.item())`` as before would force two syncs
            ### per call before we even start the loop.
            n_src_levels = max(1, int(source_tree.n_sources).bit_length())
            n_tgt_levels = max(1, int(target_tree.n_sources).bit_length())
            max_iters = 2 * (n_src_levels + n_tgt_levels) + 4
            depth = 0

            for depth in range(max_iters):
                ### ``numel()`` is a shape query (Python int), not a sync.
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

                diam_sq_T = target_tree.node_diameter_sq[active_tgt_nodes]
                diam_sq_S = source_tree.node_diameter_sq[active_src_nodes]
                diam_T = diam_sq_T.sqrt()
                diam_S = diam_sq_S.sqrt()
                combined_diam_sq = (diam_T + diam_S).pow(2)

                is_far = min_dist_sq * theta_sq > combined_diam_sq

                ### Classify active pairs (boolean masks over the full
                ### active set; combined later via ``need_split``).
                is_leaf_T = target_tree.leaf_count[active_tgt_nodes] > 0
                is_leaf_S = source_tree.leaf_count[active_src_nodes] > 0
                near_leaf_leaf = (~is_far) & is_leaf_T & is_leaf_S
                need_split = (~is_far) & (~near_leaf_leaf)

                ### 1. Far-field: deferred-compaction path.
                if expand_far_targets:
                    ### Mask counts to zero for non-far entries; the ragged
                    ### expansion then naturally skips them.  ``pair_ids``
                    ### indexes back into the *full* active set so we never
                    ### need a separate filtered ``far_src_nids`` tensor.
                    starts_full = target_tree.node_range_start[active_tgt_nodes]
                    counts_full = target_tree.node_range_count[active_tgt_nodes]
                    counts_masked = torch.where(
                        is_far, counts_full, torch.zeros_like(counts_full)
                    )
                    positions, pair_ids = _ragged_arange(
                        starts_full, counts_masked
                    )
                    nf_target_list.append(
                        target_tree.sorted_source_order[positions]
                    )
                    nf_source_node_list.append(active_src_nodes[pair_ids])
                else:
                    far_tgt_unfiltered_list.append(active_tgt_nodes)
                    far_src_unfiltered_list.append(active_src_nodes)
                    far_validity_list.append(is_far)

                ### 2. Near-field, both leaves: two-stage filtered expansion.
                # ``_expand_dual_leaf_hits`` operates on a packed batch of
                # (target_leaf_id, source_leaf_id) pairs.  Two indexings
                # share the ``near_leaf_leaf`` mask, so dedup the
                # ``nonzero`` call: one sync per iteration instead of two.
                nll_idx = near_leaf_leaf.nonzero(as_tuple=True)[0]
                (
                    nn_tgts, nn_srcs,
                    nf_tgts, nf_snids,
                    fn_tnids, fn_sids,
                    fn_btgts, fn_bstarts, fn_bcounts,
                ) = _expand_dual_leaf_hits(
                    active_tgt_nodes[nll_idx],
                    active_src_nodes[nll_idx],
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

                ### 3. Generate next iteration's active set.
                # We compute children over the FULL active set (n_active
                # entries) and use validity masks per (T,S) child slot to
                # encode the case-A / case-B / case-C splitting rules from
                # the original implementation.  After unioning the eight
                # potential child slots we pay ONE boolean compaction
                # instead of the original ~12 ``.any()``-gated indexings.
                do_split_T = (~is_leaf_T) & (
                    is_leaf_S | (diam_sq_T >= diam_sq_S)
                )
                do_split_S = (~is_leaf_S) & (
                    is_leaf_T | (diam_sq_S >= diam_sq_T)
                )
                case_T_only = need_split & do_split_T & (~do_split_S)
                case_S_only = need_split & do_split_S & (~do_split_T)
                case_both = need_split & do_split_T & do_split_S

                left_T = target_tree.node_left_child[active_tgt_nodes]
                right_T = target_tree.node_right_child[active_tgt_nodes]
                left_S = source_tree.node_left_child[active_src_nodes]
                right_S = source_tree.node_right_child[active_src_nodes]
                left_T_ok = left_T >= 0
                right_T_ok = right_T >= 0
                left_S_ok = left_S >= 0
                right_S_ok = right_S >= 0

                ### Eight child-pair slots: each is (t_ids, s_ids, validity)
                ### where every tensor has shape ``(n_active,)``.
                # 1: case_T_only, (left_T,  parent_S)
                # 2: case_T_only, (right_T, parent_S)
                # 3: case_S_only, (parent_T, left_S)
                # 4: case_S_only, (parent_T, right_S)
                # 5: case_both,   (left_T,  left_S)
                # 6: case_both,   (left_T,  right_S)
                # 7: case_both,   (right_T, left_S)
                # 8: case_both,   (right_T, right_S)
                slot_t = torch.stack([
                    left_T, right_T,
                    active_tgt_nodes, active_tgt_nodes,
                    left_T, left_T, right_T, right_T,
                ])
                slot_s = torch.stack([
                    active_src_nodes, active_src_nodes,
                    left_S, right_S,
                    left_S, right_S, left_S, right_S,
                ])
                slot_v = torch.stack([
                    case_T_only & left_T_ok,
                    case_T_only & right_T_ok,
                    case_S_only & left_S_ok,
                    case_S_only & right_S_ok,
                    case_both & left_T_ok & left_S_ok,
                    case_both & left_T_ok & right_S_ok,
                    case_both & right_T_ok & left_S_ok,
                    case_both & right_T_ok & right_S_ok,
                ])

                ### One sync per iteration: the boolean compaction below.
                flat_v = slot_v.reshape(-1)
                active_tgt_nodes = slot_t.reshape(-1)[flat_v]
                active_src_nodes = slot_s.reshape(-1)[flat_v]

            ### Concatenate accumulated pairs.
            ### Lists always have at least one element per iteration (we
            ### always append, gated by validity), so we don't need the
            ### empty-list fallbacks the previous implementation had.
            near_tgt = torch.cat(near_target_list) if near_target_list else \
                torch.empty(0, dtype=torch.long, device=device)
            near_src = torch.cat(near_source_list) if near_source_list else \
                torch.empty(0, dtype=torch.long, device=device)

            ### Far-field compaction: ONE sync for the entire traversal.
            if far_validity_list:
                far_tgt_full = torch.cat(far_tgt_unfiltered_list)
                far_src_full = torch.cat(far_src_unfiltered_list)
                far_validity_full = torch.cat(far_validity_list)
                far_tgt_nid = far_tgt_full[far_validity_full]
                far_src_nid = far_src_full[far_validity_full]
            else:
                far_tgt_nid = torch.empty(0, dtype=torch.long, device=device)
                far_src_nid = torch.empty(0, dtype=torch.long, device=device)

            nf_tgt = torch.cat(nf_target_list) if nf_target_list else \
                torch.empty(0, dtype=torch.long, device=device)
            nf_snid = torch.cat(nf_source_node_list) if nf_source_node_list else \
                torch.empty(0, dtype=torch.long, device=device)

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


def _fill_leaf_aggregates(
    leaf_nids: Int[torch.Tensor, " n_leaves"],
    leaf_starts: Int[torch.Tensor, " n_leaves"],
    leaf_sizes: Int[torch.Tensor, " n_leaves"],
    sorted_points: Float[torch.Tensor, "n_sorted_sources n_dims"],
    sorted_areas: Float[torch.Tensor, " n_sorted_sources"],
    aabb_min_buf: Float[torch.Tensor, "n_nodes n_dims"],
    aabb_max_buf: Float[torch.Tensor, "n_nodes n_dims"],
    total_area_buf: Float[torch.Tensor, " n_nodes"],
) -> None:
    """Fill leaf AABB and total-area buffers in one segmented reduction pass.

    AABB and area aggregations share the same per-source ``(positions,
    seg_ids)`` mapping from ``_ragged_arange``; doing them together
    halves the ragged-arange work and avoids a redundant
    ``int(leaf_sizes.sum())`` sync that the previous separate
    ``_fill_leaf_aabbs`` / ``_fill_leaf_total_areas`` helpers each paid.
    Empty inputs (``n_leaves == 0``) are a no-op via the early return.
    """
    n_leaves = leaf_nids.shape[0]
    if n_leaves == 0:
        return

    device = leaf_nids.device
    D = sorted_points.shape[1]
    dtype = sorted_points.dtype

    positions, seg_ids = _ragged_arange(leaf_starts, leaf_sizes)
    pts = sorted_points[positions]
    areas_per_pos = sorted_areas[positions]

    seg_min = torch.full((n_leaves, D), float("inf"), dtype=dtype, device=device)
    seg_max = torch.full((n_leaves, D), float("-inf"), dtype=dtype, device=device)
    exp_ids = seg_ids.unsqueeze(1).expand_as(pts)
    seg_min.scatter_reduce_(0, exp_ids, pts, reduce="amin", include_self=True)
    seg_max.scatter_reduce_(0, exp_ids, pts, reduce="amax", include_self=True)

    leaf_areas = torch.zeros(n_leaves, dtype=areas_per_pos.dtype, device=device)
    leaf_areas.scatter_add_(0, seg_ids, areas_per_pos)

    aabb_min_buf[leaf_nids] = seg_min
    aabb_max_buf[leaf_nids] = seg_max
    total_area_buf[leaf_nids] = leaf_areas


