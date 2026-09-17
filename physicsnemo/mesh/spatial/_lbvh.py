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

"""Shared morton-code LBVH node-topology construction.

Both :class:`~physicsnemo.mesh.spatial.bvh.BVH` (over cell AABBs) and
:class:`~physicsnemo.mesh.spatial.cluster_tree.ClusterTree` (over source points)
build the *same* binary tree: morton-sort the items, then recursively split each
sorted range at its midpoint, classifying ranges of ``<= leaf_size`` items as
leaves. This module factors out that topology construction so the two structures
no longer duplicate it; each consumer separately fills its own per-node geometry
and aggregates (leaf AABBs, total areas, diameters, ...) using the returned node
ranges and leaf segments.

The split is purely by sorted-range midpoint (``start + size // 2``) and depends
only on the item *count* and ``leaf_size`` -- not on the items' coordinates -- so
the topology is identical for any consumer over the same number of morton-sorted
items.
"""

from collections import defaultdict
from typing import NamedTuple

import torch


class LBVHTopology(NamedTuple):
    """Topology of a midpoint-split LBVH over ``n_items`` morton-sorted items.

    All ``(max_nodes,)`` buffers are pre-allocated to the capacity bound; callers
    slice them to ``[:node_count]``. Leaf-only fields (``leaf_start``,
    ``leaf_count``) carry ``-1`` / ``0`` for internal nodes; ``range_start`` /
    ``range_count`` are populated for *all* nodes (each node's subtree spans a
    contiguous range in sorted order).
    """

    left_child: torch.Tensor
    right_child: torch.Tensor
    leaf_start: torch.Tensor
    leaf_count: torch.Tensor
    range_start: torch.Tensor
    range_count: torch.Tensor
    node_count: int
    max_nodes: int
    internal_nodes_per_level: list[torch.Tensor]
    leaf_node_ids: torch.Tensor
    leaf_starts: torch.Tensor
    leaf_sizes: torch.Tensor
    max_depth: int


def build_lbvh_topology(
    n_items: int, leaf_size: int, device: torch.device
) -> LBVHTopology:
    """Build the midpoint-split LBVH node topology over ``n_items`` sorted items.

    Parameters
    ----------
    n_items : int
        Number of morton-sorted items (cells or points). Must be ``>= 1`` --
        callers handle the empty case before calling.
    leaf_size : int
        Maximum items per leaf node (``>= 1``).
    device : torch.device
        Device for the allocated tensors.

    Returns
    -------
    LBVHTopology
        Parent/child links, per-node sorted-order ranges, the compacted leaf
        segments (node id / start / size) for downstream AABB or aggregate
        filling, the internal-node ids per level (for bottom-up passes), and the
        used node count / buffer capacity / tree depth.
    """
    if leaf_size < 1:
        raise ValueError(f"leaf_size must be >= 1, got {leaf_size=!r}")

    # Midpoint splits guarantee each child gets at least floor(parent / 2) items,
    # so the minimum leaf occupancy is ceil(leaf_size / 2); from that bound the
    # max leaf count and apply the full-binary-tree identity n_internal = n_leaves - 1.
    min_per_leaf = max(1, (leaf_size + 1) // 2)
    max_leaves = (n_items + min_per_leaf - 1) // min_per_leaf
    max_nodes = max(1, 2 * max_leaves - 1)

    # --- Host-side topology recurrence (sync-free) --------------------------
    # The midpoint split depends only on the segment *size* (not coordinates),
    # so the exact per-level segment population is determined by ``n_items`` and
    # ``leaf_size`` alone and can be enumerated on the host. Tracking the
    # multiset of segment sizes (``size -> count``) per level -- only O(depth)
    # distinct sizes ever appear -- yields, for every level, the frontier width
    # and the number of internal (splitting) segments. With those host integers
    # known up front, the device build below needs no data-dependent shapes
    # (no ``torch.where(mask)`` / ``nonzero`` / ``len(tensor)`` host readbacks):
    # it compacts with ``cumsum`` + masked scatter into exactly-sized buffers.
    level_sizes: dict[int, int] = {n_items: 1}
    # Per level: (frontier_width, n_internal). The final entry has n_internal=0
    # (the all-leaf frontier).
    levels_info: list[tuple[int, int]] = []
    node_count = 1
    # Each split strictly shrinks the maximum segment size (size > leaf_size >= 1
    # implies size >= 2, and both children are < size), so the maximum size at
    # least halves per level: the recurrence terminates in <= log2(n_items) + 1
    # split levels. The extra slot covers the trailing all-leaf level, and the
    # bound is a hard guard against an unexpected non-terminating split.
    max_levels = max(1, n_items.bit_length()) + 2
    for _ in range(max_levels):
        width = sum(level_sizes.values())
        n_internal = sum(c for s, c in level_sizes.items() if s > leaf_size)
        levels_info.append((width, n_internal))
        if n_internal == 0:
            break
        node_count += 2 * n_internal
        nxt: dict[int, int] = defaultdict(int)
        for s, c in level_sizes.items():
            if s > leaf_size:
                nxt[s // 2] += c
                nxt[s - s // 2] += c
        level_sizes = dict(nxt)
    else:
        # Unreachable given the strict size-decrease argument above; a violation
        # means the split rule changed without updating this bound.
        raise RuntimeError(
            f"LBVH topology recurrence exceeded {max_levels} levels for "
            f"{n_items=}, {leaf_size=}; the midpoint split should terminate in "
            "O(log n_items) levels."
        )

    actual_depth = len(levels_info) - 1  # number of split levels
    # Full binary tree identity: n_leaves = (n_nodes + 1) / 2.
    n_leaves = (node_count + 1) // 2

    left_child = torch.full((max_nodes,), -1, dtype=torch.long, device=device)
    right_child = torch.full((max_nodes,), -1, dtype=torch.long, device=device)
    leaf_start = torch.full((max_nodes,), -1, dtype=torch.long, device=device)
    leaf_count = torch.zeros(max_nodes, dtype=torch.long, device=device)
    range_start = torch.zeros(max_nodes, dtype=torch.long, device=device)
    range_count = torch.zeros(max_nodes, dtype=torch.long, device=device)

    ### Phase 1: top-down segment queue (O(log N) iterations), sync-free.
    # Each segment is a contiguous range [start, end) in sorted order, owned by
    # a node. Compaction of internal / leaf segments uses ``cumsum`` for the
    # destination index and routes the masked-out rows to a throwaway pad slot,
    # so no host-device synchronization occurs.
    seg_starts = torch.zeros(1, dtype=torch.long, device=device)
    seg_ends = torch.full((1,), n_items, dtype=torch.long, device=device)
    seg_node_ids = torch.zeros(1, dtype=torch.long, device=device)
    node_count_dev = 1  # running id base; mirrors the host recurrence exactly
    internal_nodes_per_level: list[torch.Tensor] = []

    # Compact leaf segments are filled in place across levels at a host-tracked
    # offset; the trailing pad slot (index ``n_leaves``) absorbs masked writes.
    leaf_node_ids_buf = torch.empty(n_leaves + 1, dtype=torch.long, device=device)
    leaf_starts_buf = torch.empty(n_leaves + 1, dtype=torch.long, device=device)
    leaf_sizes_buf = torch.empty(n_leaves + 1, dtype=torch.long, device=device)
    leaf_offset = 0

    for width, n_internal in levels_info:
        seg_sizes = seg_ends - seg_starts

        ### Every node (leaf or internal) covers this contiguous sorted range.
        range_start[seg_node_ids] = seg_starts
        range_count[seg_node_ids] = seg_sizes

        is_internal_seg = seg_sizes > leaf_size
        is_leaf_seg = ~is_internal_seg
        n_leaf = width - n_internal

        # --- Record this level's leaf segments into the compact leaf buffers.
        if n_leaf > 0:
            leaf_pos = leaf_offset + torch.cumsum(is_leaf_seg.long(), 0) - 1
            leaf_dst = torch.where(
                is_leaf_seg, leaf_pos, torch.full_like(leaf_pos, n_leaves)
            )
            leaf_node_ids_buf[leaf_dst] = seg_node_ids
            leaf_starts_buf[leaf_dst] = seg_starts
            leaf_sizes_buf[leaf_dst] = seg_sizes
            leaf_offset += n_leaf

        if n_internal == 0:
            break

        # --- Compact the internal segments via cumsum (no nonzero sync). The
        # pad slot at index ``n_internal`` absorbs the leaf rows' writes.
        int_pos = torch.cumsum(is_internal_seg.long(), 0) - 1
        int_dst = torch.where(
            is_internal_seg, int_pos, torch.full_like(int_pos, n_internal)
        )
        int_starts_b = torch.empty(n_internal + 1, dtype=torch.long, device=device)
        int_ends_b = torch.empty(n_internal + 1, dtype=torch.long, device=device)
        int_node_b = torch.empty(n_internal + 1, dtype=torch.long, device=device)
        int_starts_b[int_dst] = seg_starts
        int_ends_b[int_dst] = seg_ends
        int_node_b[int_dst] = seg_node_ids
        int_starts = int_starts_b[:n_internal]
        int_ends = int_ends_b[:n_internal]
        int_node_ids = int_node_b[:n_internal]
        int_sizes = int_ends - int_starts

        midpoints = int_starts + int_sizes // 2

        left_ids = (
            node_count_dev
            + torch.arange(n_internal, dtype=torch.long, device=device) * 2
        )
        right_ids = left_ids + 1
        node_count_dev += 2 * n_internal

        left_child[int_node_ids] = left_ids
        right_child[int_node_ids] = right_ids
        internal_nodes_per_level.append(int_node_ids)

        seg_starts = torch.cat([int_starts, midpoints])
        seg_ends = torch.cat([midpoints, int_ends])
        seg_node_ids = torch.cat([left_ids, right_ids])

    leaf_node_ids = leaf_node_ids_buf[:n_leaves]
    leaf_starts = leaf_starts_buf[:n_leaves]
    leaf_sizes = leaf_sizes_buf[:n_leaves]
    leaf_start[leaf_node_ids] = leaf_starts
    leaf_count[leaf_node_ids] = leaf_sizes

    return LBVHTopology(
        left_child=left_child,
        right_child=right_child,
        leaf_start=leaf_start,
        leaf_count=leaf_count,
        range_start=range_start,
        range_count=range_count,
        node_count=node_count,
        max_nodes=max_nodes,
        internal_nodes_per_level=internal_nodes_per_level,
        leaf_node_ids=leaf_node_ids,
        leaf_starts=leaf_starts,
        leaf_sizes=leaf_sizes,
        max_depth=actual_depth,
    )
