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

"""Morton-code LBVH node-topology construction.

Both builders take items (cells or points) in Morton-sorted order and split
each sorted range in two until ranges hold at most ``leaf_size`` items. Each
consumer then fills its own per-node geometry and aggregates (leaf AABBs, total
areas, diameters, ...) from the returned node ranges and leaf segments.

- :func:`build_lbvh_topology` splits every range at its count midpoint
  (``start + size // 2``). The topology depends only on the item count and
  ``leaf_size``, so it is computed on the host without synchronizing, and
  :class:`~physicsnemo.mesh.spatial.bvh.BVH` and a ClusterTree built with
  ``split="midpoint"`` share it.
- :func:`build_morton_topology` splits every range at a Morton-cell boundary,
  so nodes cover compact cells with tight bounding boxes. It is the default for
  :class:`~physicsnemo.mesh.spatial.cluster_tree.ClusterTree`, whose
  Barnes-Hut plans grow with the box sizes.

Both builders give the two children of a node the consecutive ids ``left`` and
``left + 1``, and every internal node has exactly two children.
"""

from collections import defaultdict
from typing import NamedTuple

import torch


class LBVHTopology(NamedTuple):
    """Topology of a binary tree over ``n_items`` morton-sorted items.

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


def build_morton_topology(
    codes: torch.Tensor, leaf_size: int, *, balance: int = 4
) -> LBVHTopology:
    """Build a binary tree over sorted Morton ``codes`` whose splits follow Morton cells.

    Each range ``[s, e)`` of more than ``leaf_size`` items is split at the
    coarsest Morton-grid boundary (the highest bit in which two of its codes
    differ) among the split positions that leave at least
    ``max(1, (e - s) // balance)`` items on each side. ``balance=0`` allows
    any position, which is the binary radix tree of Karras (2012); with
    ``balance=4`` every child holds at least a quarter of its parent, so the
    depth is at most ``log(n) / log(4 / 3)``. A range of identical codes is
    split at its midpoint.

    Nodes then cover Morton cells (or, near the balance limit, a few adjacent
    cells), so their bounding boxes are tight. Count-midpoint splits
    (:func:`build_lbvh_topology`) cut through Morton cells instead, and the
    boxes of the resulting nodes span both sides of every cut.

    The topology depends on the codes, so each level synchronizes once to
    size the next.

    Parameters
    ----------
    codes : torch.Tensor
        Morton codes sorted in non-decreasing order, shape ``(n_items,)``,
        int64, ``n_items >= 1``.
    leaf_size : int
        Maximum items per leaf (``>= 1``).
    balance : int, optional, default=4
        Minimum child size as a divisor of the parent size; ``0`` disables
        the bound.

    Returns
    -------
    LBVHTopology
        Same layout as :func:`build_lbvh_topology`. Leaves may hold fewer
        than ``ceil(leaf_size / 2)`` items, so ``max_nodes = 2 * n_items - 1``.
    """
    if leaf_size < 1:
        raise ValueError(f"leaf_size must be >= 1, got {leaf_size=!r}")
    if balance < 0 or balance == 1:
        raise ValueError(f"balance must be 0 or >= 2, got {balance=!r}")
    device = codes.device
    n_items = codes.shape[0]
    max_nodes = max(1, 2 * n_items - 1)

    ### Node tables: children ``[left, right]`` (-1 for leaves) and sorted-order
    ### ranges ``[start, count]``. Children of the k-th split at a level get
    ### the consecutive ids ``base + 2k`` and ``base + 2k + 1``, so each level
    ### writes its children's ranges as one contiguous slice.
    children = torch.full((2, max_nodes), -1, dtype=torch.long, device=device)
    ranges = torch.zeros((2, max_nodes), dtype=torch.long, device=device)
    ranges[1, 0] = n_items
    node_count = 1
    internal_nodes_per_level: list[torch.Tensor] = []

    ### Ranges still to split, packed as ``[start, end, node id]``.
    segments = torch.tensor([[0], [n_items], [0]], dtype=torch.long, device=device)
    width = 1 if n_items > leaf_size else 0
    # For d in [2**k, 2**(k+1)), searchsorted(powers, d, right=True) == k + 1
    # and prefix_masks[k + 1] == -(2**k) clears the bits below k, exactly for
    # all 63-bit codes. d == 0 gives index 0, whose mask keeps every bit.
    powers = torch.tensor([1 << k for k in range(63)], dtype=torch.long, device=device)
    prefix_masks = torch.cat([powers.new_full((1,), -1), -powers])
    # ``codes_before[i] == codes[i - 1]``: the code just left of split position i.
    codes_before = torch.cat([codes[:1], codes[:-1]])
    while width:
        start, end, node = segments[0], segments[1], segments[2]
        size = end - start
        # Split positions p (left child [start, p)) range over [lo, hi].
        margin = (size // balance).clamp_min(1) if balance else 1
        lo = start + margin
        hi = end - margin
        first = codes_before[lo]
        last = codes[hi]
        diff = first ^ last
        # The first code whose bits from the highest differing bit up match
        # ``last`` starts the right child: clearing the low bits gives its key.
        mask = prefix_masks[torch.searchsorted(powers, diff, right=True)]
        boundary = torch.searchsorted(codes, last & mask)
        split = torch.where(diff > 0, boundary, start + (size >> 1))

        ids = torch.arange(node_count, node_count + 2 * width, device=device)
        children[:, node] = ids.view(width, 2).t()
        child_start = torch.stack([start, split], dim=1).reshape(-1)
        child_end = torch.stack([split, end], dim=1).reshape(-1)
        child_count = child_end - child_start
        ranges[0, node_count : node_count + 2 * width] = child_start
        ranges[1, node_count : node_count + 2 * width] = child_count
        internal_nodes_per_level.append(node)
        node_count += 2 * width

        keep = (child_count > leaf_size).nonzero(as_tuple=True)[0]
        width = keep.shape[0]
        segments = torch.stack([child_start, child_end, ids]).index_select(1, keep)

    ### Every range with more than ``leaf_size`` items was split, so the
    ### leaves are exactly the nodes with at most ``leaf_size`` items.
    range_start, range_count = ranges[0], ranges[1]
    is_leaf = range_count[:node_count] <= leaf_size
    leaf_node_ids = is_leaf.nonzero(as_tuple=True)[0]
    leaf_start = torch.full((max_nodes,), -1, dtype=torch.long, device=device)
    leaf_count = torch.zeros(max_nodes, dtype=torch.long, device=device)
    leaf_starts = range_start[leaf_node_ids]
    leaf_sizes = range_count[leaf_node_ids]
    leaf_start[leaf_node_ids] = leaf_starts
    leaf_count[leaf_node_ids] = leaf_sizes
    return LBVHTopology(
        left_child=children[0],
        right_child=children[1],
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
        max_depth=len(internal_nodes_per_level),
    )
