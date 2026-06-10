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

"""Spatial cluster tree for dual-tree Barnes-Hut acceleration.

This module provides a GPU-compatible hierarchical spatial decomposition over a
set of points, designed for dual-tree Barnes-Hut O(N) acceleration of
kernel-summation and attention-style operators (e.g. GLOBE's field kernels and
mesh attention layers).
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
from typing import NamedTuple

import torch
from jaxtyping import Float, Int
from tensordict import TensorDict, tensorclass
from torch.profiler import record_function

from physicsnemo.mesh.spatial._lbvh import build_lbvh_topology
from physicsnemo.mesh.spatial._ragged import _ragged_arange
from physicsnemo.mesh.spatial.bvh import _compute_morton_codes

logger = logging.getLogger("mesh.spatial.cluster_tree")


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
            (
                "near_target_ids",
                self.near_target_ids,
                "near_source_ids",
                self.near_source_ids,
            ),
            (
                "far_target_node_ids",
                self.far_target_node_ids,
                "far_source_node_ids",
                self.far_source_node_ids,
            ),
            (
                "nf_target_ids",
                self.nf_target_ids,
                "nf_source_node_ids",
                self.nf_source_node_ids,
            ),
            (
                "fn_target_node_ids",
                self.fn_target_node_ids,
                "fn_source_ids",
                self.fn_source_ids,
            ),
        ]
        for name_a, a, name_b, b in pairs:
            if a.shape != b.shape:
                raise ValueError(
                    f"Shape mismatch: {name_a}.shape={a.shape!r} != "
                    f"{name_b}.shape={b.shape!r}"
                )

        ### fn_broadcast tensors must be consistently sized AND non-negative.
        n_fn = self.fn_source_ids.shape[0]
        for name, tensor in [
            ("fn_broadcast_starts", self.fn_broadcast_starts),
            ("fn_broadcast_counts", self.fn_broadcast_counts),
        ]:
            if tensor.shape != (n_fn,):
                raise ValueError(f"{name}.shape={tensor.shape!r}, expected ({n_fn},)")
            if tensor.numel() > 0 and (tensor < 0).any():
                raise ValueError(f"{name} contains negative values")

        ### fn_broadcast bounds: every (start, count) range with count > 0
        ### must fit within fn_broadcast_targets.  Zero-count entries are
        ### no-ops whose starts are never dereferenced.
        if n_fn > 0:
            nonzero = self.fn_broadcast_counts > 0
            if nonzero.any():
                ends = (
                    self.fn_broadcast_starts[nonzero]
                    + self.fn_broadcast_counts[nonzero]
                )
                max_end = ends.max().item()
                bcast_len = self.fn_broadcast_targets.shape[0]
                if max_end > bcast_len:
                    raise ValueError(
                        f"fn_broadcast out of bounds: max(starts + counts)="
                        f"{max_end} > fn_broadcast_targets.shape[0]={bcast_len}"
                    )


class _ExpandedLeafHits(NamedTuple):
    """Per-iteration output of :func:`_expand_dual_leaf_hits`.

    Three of the four interaction streams are returned in
    *deferred-compaction* form: the per-element tensor is unfiltered
    (length ``t_full`` or ``s_full``) and accompanied by a boolean
    validity mask, so the caller can amortise compaction across all
    traversal iterations into a single boolean indexing per stream.

    Fields
    ------
    near_tgts, near_srcs : Int[Tensor, " n_near"]
        (near, near) Cartesian-product pairs.  Already compacted -
        ``_ragged_arange`` sized the output by data anyway.
    nf_tgts, nf_snids : Int[Tensor, " t_full"]
        (near, far) target IDs / source-node IDs, length :math:`T_\\text{full}
        = \\sum t_\\text{counts}`.
    nf_validity : Bool[Tensor, " t_full"]
        Mask selecting the (near, far) entries.  Equals ``target_is_far``.
    fn_sids, fn_tnids : Int[Tensor, " s_full"]
        (far, near) source IDs / target-node IDs, length :math:`S_\\text{full}
        = \\sum s_\\text{counts}`.
    fn_validity : Bool[Tensor, " s_full"]
        Mask selecting the (far, near) entries.  Equals ``source_is_far``.
    fn_bcast_starts, fn_bcast_counts : Int[Tensor, " s_full"]
        Per-source start/count into ``fn_bcast_targets``.  Aligned with
        the fn stream; filter by ``fn_validity``.  Non-fn entries have
        arithmetically defined but unused values.
    fn_bcast_targets : Int[Tensor, " t_full"]
        Active survivor target IDs sorted by leaf pair, *sentinel-padded*
        at the tail.  Compact via ``fn_bcast_targets_validity``.
    fn_bcast_targets_validity : Bool[Tensor, " t_full"]
        Mask selecting active (non-sentinel) entries.
    """

    near_tgts: torch.Tensor
    near_srcs: torch.Tensor
    nf_tgts: torch.Tensor
    nf_snids: torch.Tensor
    nf_validity: torch.Tensor
    fn_sids: torch.Tensor
    fn_tnids: torch.Tensor
    fn_validity: torch.Tensor
    fn_bcast_starts: torch.Tensor
    fn_bcast_counts: torch.Tensor
    fn_bcast_targets: torch.Tensor
    fn_bcast_targets_validity: torch.Tensor

    @classmethod
    def empty(cls, device: torch.device) -> "_ExpandedLeafHits":
        """All-empty hits, used as the ``n_pairs == 0`` short-circuit."""
        el = torch.empty(0, dtype=torch.long, device=device)
        eb = torch.empty(0, dtype=torch.bool, device=device)
        return cls(
            near_tgts=el,
            near_srcs=el.clone(),
            nf_tgts=el.clone(),
            nf_snids=el.clone(),
            nf_validity=eb,
            fn_sids=el.clone(),
            fn_tnids=el.clone(),
            fn_validity=eb.clone(),
            fn_bcast_starts=el.clone(),
            fn_bcast_counts=el.clone(),
            fn_bcast_targets=el.clone(),
            fn_bcast_targets_validity=eb.clone(),
        )


def _expand_dual_leaf_hits(
    target_leaf_ids: Int[torch.Tensor, " n_leaf_pairs"],
    source_leaf_ids: Int[torch.Tensor, " n_leaf_pairs"],
    target_tree: "ClusterTree",
    source_tree: "ClusterTree",
    theta: float,
) -> _ExpandedLeafHits:
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

    Three of the four output streams are returned in **deferred-compaction**
    form on the result struct: the per-element tensor is unfiltered (length
    ``t_full`` or ``s_full``) and accompanied by a boolean validity mask.
    The caller accumulates these across traversal iterations and does ONE
    boolean compaction at the end - mirroring the pattern already used for
    the far-field stream in ``find_dual_interaction_pairs``.  This
    eliminates the five per-iter ``aten::nonzero`` syncs that the previous
    eagerly-filtered version paid (one each for ``target_is_far``,
    ``~target_is_far``, ``source_is_far``, ``fn_active_mask``, and
    ``~source_is_far``).

    Returns
    -------
    _ExpandedLeafHits
        See :class:`_ExpandedLeafHits` for the per-field shapes and
        deferred-compaction protocol.
    """
    device = target_leaf_ids.device
    theta_sq = theta * theta
    n_pairs = target_leaf_ids.shape[0]

    ### The early-return guard is a Python ``int`` comparison on a shape
    ### attribute - zero CUDA cost.  It matters because in the early
    ### traversal iterations (top-of-tree) there are typically no
    ### leaf-leaf pairs yet, and without this guard the three
    ### ``_ragged_arange`` calls below would each pay a
    ### ``torch.arange(scalar_tensor)`` host sync to size their empty
    ### output.  Saves ~3 syncs * (number of leaf-leaf-free early iters)
    ### per traversal.
    if n_pairs == 0:
        return _ExpandedLeafHits.empty(device)

    ### The rest of this function is intentionally written without
    ### ``if X.any():`` / ``int(X.sum())`` early-exit branches AND without
    ### ``X.nonzero()`` compactions.  Each such call was a CPU-GPU sync
    ### point in the dual-traversal hot loop; the sync count was the
    ### dominant CPU stall in profiling.  All downstream operations
    ### (``_ragged_arange``, ``argsort``, ``scatter_add_``, ``scatter_``)
    ### handle zero-element inputs correctly, so we let empty intermediate
    ### tensors flow through unconditionally.

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
    target_is_far = (
        dist_sq_t * theta_sq > source_tree.node_diameter_sq[src_leaf_per_target]
    )

    ### (near, far) stream is returned unfiltered.  ``target_point_ids``,
    ### ``src_leaf_per_target`` are length ``t_full``; the caller compacts
    ### them with ``target_is_far`` (== ``nf_validity``) at end-of-traversal.

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
    source_is_far = (
        dist_sq_s * theta_sq > target_tree.node_diameter_sq[tgt_leaf_per_src]
    )

    ### (far, near) stream is returned unfiltered: ``src_point_ids``,
    ### ``tgt_leaf_per_src`` are length ``s_full``; the caller compacts
    ### with ``source_is_far`` (== ``fn_validity``).

    # ==================================================================
    # Build (far, near) broadcast mapping (sync-free, sentinel-padded)
    # ==================================================================
    # ``has_fn_source[lp]`` is True iff leaf pair ``lp`` has at least one
    # fn source (i.e., a source that passed the per-source far test).
    # Sync-free construction: scatter ``True`` into ``has_fn_source[lp]``
    # for every fn entry and into a sentinel slot for every non-fn entry.
    # The original ``has_fn_source[fn_lp_ids] = True`` required a
    # ``nonzero`` on ``source_is_far`` to compute the filtered
    # ``fn_lp_ids``; ``torch.where`` + sentinel-slot scatter is
    # data-flow-equivalent and pays zero CUDA syncs.
    has_fn_source_buf = torch.zeros(n_pairs + 1, dtype=torch.bool, device=device)
    safe_fn_lp = torch.where(
        source_is_far,
        leaf_pair_ids_s,
        torch.full_like(leaf_pair_ids_s, n_pairs),
    )
    has_fn_source_buf.scatter_(0, safe_fn_lp, True)
    has_fn_source = has_fn_source_buf[:n_pairs]

    ### An "active" target is a stage-1 survivor whose leaf pair has at
    ### least one fn source.  Working on the unfiltered ``leaf_pair_ids_t``
    ### lets us build the validity mask without a ``nonzero`` over
    ### ``~target_is_far``.
    active_validity = (~target_is_far) & has_fn_source[leaf_pair_ids_t]

    ### Sort by ``(leaf_pair_id if active else n_pairs)`` so that within
    ### the sorted target_point_ids the active entries come first, grouped
    ### by leaf-pair, followed by all the inactive entries (which carry the
    ### sentinel key ``n_pairs``).  The caller drops the inactive tail via
    ### ``fn_broadcast_targets_validity`` at end-of-traversal.
    bcast_sort_key = torch.where(
        active_validity,
        leaf_pair_ids_t,
        torch.full_like(leaf_pair_ids_t, n_pairs),
    )
    bcast_sort_order = bcast_sort_key.argsort(stable=True)
    fn_broadcast_targets = target_point_ids[bcast_sort_order]
    fn_broadcast_targets_validity = active_validity[bcast_sort_order]

    ### Per-lp active count via weighted ``scatter_add_``.  Weight =
    ### ``active_validity.long()``, so non-active entries contribute zero.
    active_counts_per_lp = torch.zeros(n_pairs, dtype=torch.long, device=device)
    active_counts_per_lp.scatter_add_(0, leaf_pair_ids_t, active_validity.long())
    active_starts_per_lp = active_counts_per_lp.cumsum(0) - active_counts_per_lp

    ### Return broadcast_starts/counts aligned with the *full* per-source
    ### axis (length ``s_full``).  The caller filters by ``source_is_far``.
    fn_broadcast_starts_full = active_starts_per_lp[leaf_pair_ids_s]
    fn_broadcast_counts_full = active_counts_per_lp[leaf_pair_ids_s]

    # ==================================================================
    # Reduced Cartesian product: survivors × close sources only
    # ==================================================================
    ### Per-lp count of close sources via weighted ``scatter_add_``.
    close_counts_per_lp = torch.zeros(n_pairs, dtype=torch.long, device=device)
    close_counts_per_lp.scatter_add_(0, leaf_pair_ids_s, (~source_is_far).long())

    ### Sort sources by ``(leaf_pair_id, source_is_far)`` so within each
    ### lp's contiguous block the close sources come first (key
    ### ``2*lp + 0``) followed by the far sources (key ``2*lp + 1``).
    ### This avoids the per-iter ``(~source_is_far).nonzero()`` sync that
    ### the previous filtered-then-sort version paid.  Stable argsort
    ### preserves the original within-lp order of close sources, matching
    ### the previous implementation's output element-for-element.
    src_sort_key = leaf_pair_ids_s * 2 + source_is_far.long()
    src_sort_order = src_sort_key.argsort(stable=True)
    sorted_src_ids = src_point_ids[src_sort_order]

    ### Start of lp's block in ``sorted_src_ids`` is the exclusive cumsum
    ### of ``s_counts`` (the per-lp total source count, by construction).
    total_lp_starts = s_counts.cumsum(0) - s_counts

    ### Per-target close counts: gate by ``(~target_is_far).long()`` so
    ### non-survivors get count 0 and produce no Cartesian-product output.
    ### The block start does not depend on survivor-ness.
    per_target_close_counts = (
        close_counts_per_lp[leaf_pair_ids_t] * (~target_is_far).long()
    )
    per_target_close_starts = total_lp_starts[leaf_pair_ids_t]

    ### Reuse ``_ragged_arange``'s second output (``seg_ids_nn``) as the
    ### per-element survivor index instead of calling
    ### ``torch.repeat_interleave(surv_point_ids, per_target_close_counts)``.
    ### Both ops sync once to size their output; folding them into one
    ### ``_ragged_arange`` halves that cost.  Functionally identical:
    ### ``repeat_interleave(x, counts)[k] == x[seg_ids[k]]`` by
    ### definition of segment ids.
    src_positions_nn, seg_ids_nn = _ragged_arange(
        per_target_close_starts, per_target_close_counts
    )
    expanded_near_tgts = target_point_ids[seg_ids_nn]
    expanded_near_srcs = sorted_src_ids[src_positions_nn]

    return _ExpandedLeafHits(
        near_tgts=expanded_near_tgts,
        near_srcs=expanded_near_srcs,
        nf_tgts=target_point_ids,
        nf_snids=src_leaf_per_target,
        nf_validity=target_is_far,
        fn_sids=src_point_ids,
        fn_tnids=tgt_leaf_per_src,
        fn_validity=source_is_far,
        fn_bcast_starts=fn_broadcast_starts_full,
        fn_bcast_counts=fn_broadcast_counts_full,
        fn_bcast_targets=fn_broadcast_targets,
        fn_bcast_targets_validity=fn_broadcast_targets_validity,
    )


# ---------------------------------------------------------------------------
# Deferred-compaction helpers (used by find_dual_interaction_pairs)
# ---------------------------------------------------------------------------


def _compact_deferred(
    *tensor_lists: list[torch.Tensor],
    validity_list: list[torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    """Concatenate per-iter accumulators and boolean-index by validity.

    Each ``tensor_lists[i]`` is the per-iter accumulator for one output
    stream; ``validity_list`` is the shared per-iter validity mask.  All
    accumulators must be the same length within each iteration.  Pays
    exactly one ``aten::nonzero`` sync regardless of the number of
    output streams - the sync is amortised across them by computing
    the integer ``keep_idx`` once and reusing it for every stream.

    The empty-``validity_list`` case (no iteration ever contributed to
    this stream) is handled explicitly because ``torch.cat([])`` raises.
    """
    if not validity_list:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return tuple(empty.clone() for _ in tensor_lists)
    keep_idx = torch.cat(validity_list).nonzero(as_tuple=True)[0]
    return tuple(torch.cat(L)[keep_idx] for L in tensor_lists)


def _compact_sentinel_padded(
    padded_tensor: torch.Tensor,
    referencing_indices: torch.Tensor,
    validity: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compact a sentinel-padded buffer and remap referencing indices.

    ``padded_tensor`` has don't-care entries wherever ``~validity``;
    ``referencing_indices`` are integer indices into ``padded_tensor``
    that only ever reference *valid* positions.  This returns
    ``(padded_tensor[validity], remapped_indices)`` where the remapped
    indices point at the same elements in the compacted buffer.

    The remap is the exclusive cumsum of ``validity``: position ``p`` in
    the padded buffer maps to ``sum(validity[:p])`` in the compacted
    buffer, which is its slot in ``padded_tensor[validity]``.  Pays one
    sync (the boolean indexing); the cumsum and integer-indexing remap
    are sync-free.
    """
    valid_long = validity.long()
    new_pos = valid_long.cumsum(0) - valid_long
    return padded_tensor[validity], new_pos[referencing_indices]


def _sort_by_key(
    *tensors: torch.Tensor,
    key: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Stable-sort companion tensors by ``key``; no-op on empty input.

    Used at the end of ``find_dual_interaction_pairs`` to group each
    output stream by source index (or source node) for coalesced
    downstream gathers.
    """
    if key.numel() == 0:
        return tensors
    order = key.argsort(stable=True)
    return tuple(t[order] for t in tensors)


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

        ### Build the shared morton-LBVH node topology over the sorted points.
        with record_function("cluster_tree::top_down_build"):
            topo = build_lbvh_topology(n_points, leaf_size, device)

        ### Fill leaf AABBs + total areas from the source points/areas (single
        # combined pass over the compacted leaf segments), then propagate AABBs
        # and areas bottom-up so each internal node summarises its subtree.
        aabb_min_buf = torch.full(
            (topo.max_nodes, D), float("inf"), dtype=dtype, device=device
        )
        aabb_max_buf = torch.full(
            (topo.max_nodes, D), float("-inf"), dtype=dtype, device=device
        )
        total_area_buf = torch.zeros(topo.max_nodes, dtype=dtype, device=device)
        with record_function("cluster_tree::leaf_aggregates"):
            _fill_leaf_aggregates(
                topo.leaf_node_ids,
                topo.leaf_starts,
                topo.leaf_sizes,
                sorted_points,
                sorted_areas,
                aabb_min_buf,
                aabb_max_buf,
                total_area_buf,
            )

        with record_function("cluster_tree::bottom_up_aabb"):
            for level_node_ids in reversed(topo.internal_nodes_per_level):
                left = topo.left_child[level_node_ids]
                right = topo.right_child[level_node_ids]
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
        node_count = topo.node_count
        aabb_min_trimmed = aabb_min_buf[:node_count]
        aabb_max_trimmed = aabb_max_buf[:node_count]
        diameter_sq = (aabb_max_trimmed - aabb_min_trimmed).pow(2).sum(dim=-1)

        logger.debug(
            "ClusterTree: %d points -> %d nodes, depth %d, leaf_size=%d",
            n_points,
            node_count,
            topo.max_depth,
            leaf_size,
        )

        return cls(
            node_aabb_min=aabb_min_trimmed,
            node_aabb_max=aabb_max_trimmed,
            node_diameter_sq=diameter_sq,
            node_left_child=topo.left_child[:node_count],
            node_right_child=topo.right_child[:node_count],
            leaf_start=topo.leaf_start[:node_count],
            leaf_count=topo.leaf_count[:node_count],
            node_range_start=topo.range_start[:node_count],
            node_range_count=topo.range_count[:node_count],
            node_total_area=total_area_buf[:node_count],
            sorted_source_order=sorted_order,
            source_points=points,
            max_depth=torch.tensor(topo.max_depth, dtype=torch.long, device=device),
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
            centroid_buf = (node_total_weighted_pts / safe_areas_64.unsqueeze(-1)).to(
                source_points.dtype
            )

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

            ### Output streams.  All per-iteration outputs use a
            ### deferred-compaction protocol: the per-element tensor is
            ### accumulated unfiltered together with a boolean validity
            ### mask, and ONE compaction is paid per stream at the end of
            ### the traversal.  This trades five per-iteration ``nonzero``
            ### syncs inside the (near,far)/(far,near)/broadcast paths plus
            ### two per-iteration syncs in the far path for a fixed handful
            ### of end-of-loop syncs.
            far_tgt_unfiltered_list: list[torch.Tensor] = []
            far_src_unfiltered_list: list[torch.Tensor] = []
            far_validity_list: list[torch.Tensor] = []

            ### (near,near) output from leaf-leaf expansion is already
            ### compacted by the ``_ragged_arange`` inside the expansion
            ### (its output size is set by the Cartesian total anyway),
            ### so no per-stream validity mask is needed here.
            near_target_list: list[torch.Tensor] = []
            near_source_list: list[torch.Tensor] = []

            ### (near,far) stream has two append paths:
            ###  - ``expand_far_targets=True``: already-filtered entries
            ###    from the ``_ragged_arange``-with-masked-counts branch
            ###    below.  No validity mask needed.
            ###  - ``_expand_dual_leaf_hits``: unfiltered targets +
            ###    ``nf_validity`` mask (length ``t_full`` per iter).
            ### Kept in separate lists so the deferred path's compaction
            ### at end-of-traversal does not touch the already-filtered
            ### entries.
            nf_filtered_target_list: list[torch.Tensor] = []
            nf_filtered_source_node_list: list[torch.Tensor] = []
            nf_deferred_target_list: list[torch.Tensor] = []
            nf_deferred_source_node_list: list[torch.Tensor] = []
            nf_deferred_validity_list: list[torch.Tensor] = []

            ### (far,near) + broadcast mapping from ``_expand_dual_leaf_hits``.
            ### Both the per-source tensors and the per-source broadcast
            ### starts/counts are stored unfiltered against the same
            ### ``fn_validity`` (``= source_is_far``); the broadcast
            ### targets buffer carries its own ``fn_bcast_validity`` mask
            ### that drops the sentinel-padded tail of each iter.
            fn_deferred_tgt_node_list: list[torch.Tensor] = []
            fn_deferred_src_list: list[torch.Tensor] = []
            fn_deferred_validity_list: list[torch.Tensor] = []
            fn_bcast_starts_list: list[torch.Tensor] = []
            fn_bcast_counts_list: list[torch.Tensor] = []
            fn_bcast_targets_list: list[torch.Tensor] = []
            fn_bcast_validity_list: list[torch.Tensor] = []
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
                    positions, pair_ids = _ragged_arange(starts_full, counts_masked)
                    nf_filtered_target_list.append(
                        target_tree.sorted_source_order[positions]
                    )
                    nf_filtered_source_node_list.append(active_src_nodes[pair_ids])
                else:
                    far_tgt_unfiltered_list.append(active_tgt_nodes)
                    far_src_unfiltered_list.append(active_src_nodes)
                    far_validity_list.append(is_far)

                ### 2. Near-field, both leaves: two-stage deferred expansion.
                # ``_expand_dual_leaf_hits`` returns the (near,far),
                # (far,near), and broadcast streams unfiltered (with
                # validity masks); the caller compacts them once at the
                # end of the traversal.  Only the (near,near) Cartesian-
                # product output (whose size is data-dependent regardless)
                # is already compacted.
                nll_idx = near_leaf_leaf.nonzero(as_tuple=True)[0]
                hits = _expand_dual_leaf_hits(
                    active_tgt_nodes[nll_idx],
                    active_src_nodes[nll_idx],
                    target_tree,
                    source_tree,
                    theta,
                )
                near_target_list.append(hits.near_tgts)
                near_source_list.append(hits.near_srcs)
                nf_deferred_target_list.append(hits.nf_tgts)
                nf_deferred_source_node_list.append(hits.nf_snids)
                nf_deferred_validity_list.append(hits.nf_validity)
                fn_deferred_tgt_node_list.append(hits.fn_tnids)
                fn_deferred_src_list.append(hits.fn_sids)
                fn_deferred_validity_list.append(hits.fn_validity)
                fn_bcast_starts_list.append(hits.fn_bcast_starts + fn_bcast_offset)
                fn_bcast_counts_list.append(hits.fn_bcast_counts)
                fn_bcast_targets_list.append(hits.fn_bcast_targets)
                fn_bcast_validity_list.append(hits.fn_bcast_targets_validity)
                fn_bcast_offset += hits.fn_bcast_targets.shape[0]

                ### 3. Generate next iteration's active set.
                # We compute children over the FULL active set (n_active
                # entries) and use validity masks per (T,S) child slot to
                # encode the case-A / case-B / case-C splitting rules from
                # the original implementation.  After unioning the eight
                # potential child slots we pay ONE boolean compaction
                # instead of the original ~12 ``.any()``-gated indexings.
                do_split_T = (~is_leaf_T) & (is_leaf_S | (diam_sq_T >= diam_sq_S))
                do_split_S = (~is_leaf_S) & (is_leaf_T | (diam_sq_S >= diam_sq_T))
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
                slot_t = torch.stack(
                    [
                        left_T,
                        right_T,
                        active_tgt_nodes,
                        active_tgt_nodes,
                        left_T,
                        left_T,
                        right_T,
                        right_T,
                    ]
                )
                slot_s = torch.stack(
                    [
                        active_src_nodes,
                        active_src_nodes,
                        left_S,
                        right_S,
                        left_S,
                        right_S,
                        left_S,
                        right_S,
                    ]
                )
                slot_v = torch.stack(
                    [
                        case_T_only & left_T_ok,
                        case_T_only & right_T_ok,
                        case_S_only & left_S_ok,
                        case_S_only & right_S_ok,
                        case_both & left_T_ok & left_S_ok,
                        case_both & left_T_ok & right_S_ok,
                        case_both & right_T_ok & left_S_ok,
                        case_both & right_T_ok & right_S_ok,
                    ]
                )

                ### One sync per iteration: the boolean compaction below.
                ### Each ``tensor[bool_mask]`` lowers to ``aten::nonzero``;
                ### computing ``keep_idx`` explicitly once and integer-
                ### indexing both ``slot_t`` and ``slot_s`` collapses the
                ### two nonzero syncs into one.
                flat_v = slot_v.reshape(-1)
                keep_idx = flat_v.nonzero(as_tuple=True)[0]
                active_tgt_nodes = slot_t.reshape(-1)[keep_idx]
                active_src_nodes = slot_s.reshape(-1)[keep_idx]

            ### Concatenate accumulated pairs and pay one boolean
            ### compaction per output stream, all at end-of-traversal.
            ### See :func:`_compact_deferred` and
            ### :func:`_compact_sentinel_padded` for the protocol.
            empty_long = torch.empty(0, dtype=torch.long, device=device)

            near_tgt = (
                torch.cat(near_target_list) if near_target_list else empty_long.clone()
            )
            near_src = (
                torch.cat(near_source_list) if near_source_list else empty_long.clone()
            )

            ### Far-field stream: deferred (unfiltered + validity).
            far_tgt_nid, far_src_nid = _compact_deferred(
                far_tgt_unfiltered_list,
                far_src_unfiltered_list,
                validity_list=far_validity_list,
                device=device,
            )

            ### (near, far) stream: combine deferred entries from
            ### ``_expand_dual_leaf_hits`` with the already-filtered
            ### entries from the ``expand_far_targets=True`` branch.
            nf_def_tgt, nf_def_snid = _compact_deferred(
                nf_deferred_target_list,
                nf_deferred_source_node_list,
                validity_list=nf_deferred_validity_list,
                device=device,
            )
            nf_tgt = (
                torch.cat([nf_def_tgt, *nf_filtered_target_list])
                if nf_filtered_target_list
                else nf_def_tgt
            )
            nf_snid = (
                torch.cat([nf_def_snid, *nf_filtered_source_node_list])
                if nf_filtered_source_node_list
                else nf_def_snid
            )

            ### (far, near) + broadcast streams.  The fn tensors and
            ### the per-source ``fn_bcast_starts/counts`` are aligned
            ### with ``fn_validity`` (= ``source_is_far``) and compact
            ### together.  ``fn_broadcast_targets`` is sentinel-padded
            ### on the *t_full* axis and compacts separately via its
            ### own validity mask, with ``fn_bcast_starts`` remapped
            ### into the compacted space.
            if fn_deferred_validity_list:
                fn_tnid, fn_sid, fn_bstarts_padded, fn_bcounts = _compact_deferred(
                    fn_deferred_tgt_node_list,
                    fn_deferred_src_list,
                    fn_bcast_starts_list,
                    fn_bcast_counts_list,
                    validity_list=fn_deferred_validity_list,
                    device=device,
                )
                fn_btgts, fn_bstarts = _compact_sentinel_padded(
                    torch.cat(fn_bcast_targets_list),
                    fn_bstarts_padded,
                    torch.cat(fn_bcast_validity_list),
                )
            else:
                fn_tnid = empty_long.clone()
                fn_sid = empty_long.clone()
                fn_btgts = empty_long.clone()
                fn_bstarts = empty_long.clone()
                fn_bcounts = empty_long.clone()

            ### Group each output stream by source index (or source node)
            ### for coalesced downstream gathers.  See :func:`_sort_by_key`.
            near_tgt, near_src = _sort_by_key(near_tgt, near_src, key=near_src)
            far_tgt_nid, far_src_nid = _sort_by_key(
                far_tgt_nid, far_src_nid, key=far_src_nid
            )
            nf_tgt, nf_snid = _sort_by_key(nf_tgt, nf_snid, key=nf_snid)
            fn_tnid, fn_sid, fn_bstarts, fn_bcounts = _sort_by_key(
                fn_tnid, fn_sid, fn_bstarts, fn_bcounts, key=fn_sid
            )

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
            plan.n_near,
            plan.n_nf,
            plan.n_fn,
            plan.n_far_nodes,
            theta,
            is_self,
            depth,
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
