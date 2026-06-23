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

    left_child = torch.full((max_nodes,), -1, dtype=torch.long, device=device)
    right_child = torch.full((max_nodes,), -1, dtype=torch.long, device=device)
    leaf_start = torch.full((max_nodes,), -1, dtype=torch.long, device=device)
    leaf_count = torch.zeros(max_nodes, dtype=torch.long, device=device)
    range_start = torch.zeros(max_nodes, dtype=torch.long, device=device)
    range_count = torch.zeros(max_nodes, dtype=torch.long, device=device)

    ### Phase 1: top-down segment queue (O(log N) iterations).
    # Each segment is a contiguous range [start, end) in sorted order, owned by a node.
    seg_starts = torch.tensor([0], dtype=torch.long, device=device)
    seg_ends = torch.tensor([n_items], dtype=torch.long, device=device)
    seg_node_ids = torch.tensor([0], dtype=torch.long, device=device)
    node_count = 1
    actual_depth = 0
    internal_nodes_per_level: list[torch.Tensor] = []

    # Defer leaf-segment processing to a single end-of-loop compaction, avoiding a
    # per-level torch.where(is_leaf_seg) host-device sync.
    leaf_seg_node_ids: list[torch.Tensor] = []
    leaf_seg_starts: list[torch.Tensor] = []
    leaf_seg_sizes: list[torch.Tensor] = []
    leaf_seg_validity: list[torch.Tensor] = []

    while len(seg_starts) > 0:
        seg_sizes = seg_ends - seg_starts

        ### Every node (leaf or internal) covers this contiguous sorted range.
        range_start[seg_node_ids] = seg_starts
        range_count[seg_node_ids] = seg_sizes

        is_leaf_seg = seg_sizes <= leaf_size
        is_internal_seg = ~is_leaf_seg

        leaf_seg_node_ids.append(seg_node_ids)
        leaf_seg_starts.append(seg_starts)
        leaf_seg_sizes.append(seg_sizes)
        leaf_seg_validity.append(is_leaf_seg)

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
            node_count + torch.arange(n_internal, dtype=torch.long, device=device) * 2
        )
        right_ids = left_ids + 1
        node_count += 2 * n_internal

        left_child[int_node_ids] = left_ids
        right_child[int_node_ids] = right_ids
        internal_nodes_per_level.append(int_node_ids)

        seg_starts = torch.cat([int_starts, midpoints])
        seg_ends = torch.cat([midpoints, int_ends])
        seg_node_ids = torch.cat([left_ids, right_ids])

    ### Single-pass leaf fill: one boolean compaction across all levels.
    if leaf_seg_node_ids:
        all_leaf_validity = torch.cat(leaf_seg_validity)
        leaf_node_ids = torch.cat(leaf_seg_node_ids)[all_leaf_validity]
        leaf_starts = torch.cat(leaf_seg_starts)[all_leaf_validity]
        leaf_sizes = torch.cat(leaf_seg_sizes)[all_leaf_validity]
        leaf_start[leaf_node_ids] = leaf_starts
        leaf_count[leaf_node_ids] = leaf_sizes
    else:  # unreachable for n_items >= 1, but keep the contract total
        empty = torch.empty(0, dtype=torch.long, device=device)
        leaf_node_ids = empty
        leaf_starts = empty.clone()
        leaf_sizes = empty.clone()

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
