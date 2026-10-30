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

"""Morton topology of a forest of trees in Warp kernels, with one host sync.

:func:`~physicsnemo.mesh.spatial._lbvh.build_morton_topology` splits one tree
level by level and synchronizes once per level to size the next. Here every
sorted position walks down from its tree's root, computing that function's
split at every node on its way, and emits the nodes whose range starts at it:
a count pass, one readback, and a fill pass. A sort of the nodes by (tree,
depth, first position) is the level-by-level numbering of the torch build, so
the topology, which is integer arithmetic, is that build's exactly.
:func:`node_boxes` fills the nodes' boxes bottom up in one kernel; they are
the torch build's except in the signs of tied zeros and the payloads of NaNs.

Imported lazily by :class:`~physicsnemo.mesh.spatial.cluster_tree.ClusterTree`,
so importing :mod:`physicsnemo.mesh.spatial` registers no kernels.
"""

from typing import Any, NamedTuple

import torch
import torch.nn.functional as F
import warp as wp

from physicsnemo.core.function_spec import FunctionSpec
from physicsnemo.mesh.spatial._lbvh import LBVHTopology

#: Depth bins of the readback. A node key holds the depth in 6 bits, so a
#: forest with internal nodes at depth ``_DEPTH_BINS - 2`` is left to torch.
_DEPTH_BINS = 64
#: Copies of the depth bins that the positions' counts are spread over, so
#: that the atomic adds to one bin do not serialize.
_LANES = 1024


@wp.func
def _morton_split(
    codes: wp.array(dtype=wp.int64), s: wp.int64, e: wp.int64, balance: wp.int64
):
    """The split of the range ``[s, e)`` of sorted ``codes``, as
    :func:`~physicsnemo.mesh.spatial._lbvh.build_morton_topology` makes it: the
    first position in ``[lo, hi]`` whose code reaches ``codes[hi]`` with the
    bits below the highest bit in which ``codes[lo - 1]`` and ``codes[hi]``
    differ cleared, or the midpoint when they are equal."""
    size = e - s
    margin = wp.int64(1)
    if balance > wp.int64(0):
        margin = wp.max(size // balance, wp.int64(1))
    lo = s + margin
    hi = e - margin
    last = codes[hi]
    diff = codes[lo - wp.int64(1)] ^ last
    q = s + size // wp.int64(2)
    if diff > wp.int64(0):
        m = diff
        m = m | (m >> wp.int64(1))
        m = m | (m >> wp.int64(2))
        m = m | (m >> wp.int64(4))
        m = m | (m >> wp.int64(8))
        m = m | (m >> wp.int64(16))
        m = m | (m >> wp.int64(32))
        key = last - (last & (m >> wp.int64(1)))
        a = lo
        b = hi
        while a < b:
            mid = a + (b - a) // wp.int64(2)
            if codes[mid] < key:
                a = mid + wp.int64(1)
            else:
                b = mid
        q = a
    return q


@wp.kernel(enable_backward=False)
def _node_count_kernel(
    codes: wp.array(dtype=wp.int64),
    tree: wp.array(dtype=wp.int64),
    tree_start: wp.array(dtype=wp.int64),
    tree_end: wp.array(dtype=wp.int64),
    leaf_size: wp.int64,
    balance: wp.int64,
    count: wp.array(dtype=wp.int64),
    leaf_depth: wp.array(dtype=wp.int64),
):
    """The number of nodes whose range starts at sorted position ``tid`` (a
    chain of left children ending at a leaf) and the depth of that leaf."""
    p = wp.int64(wp.tid())
    t = tree[p]
    s = tree_start[t]
    e = tree_end[t]
    depth = wp.int64(0)
    n = wp.int64(0)
    if s == p:
        n = wp.int64(1)
    while e - s > leaf_size:
        q = _morton_split(codes, s, e, balance)
        if p < q:
            e = q
        else:
            s = q
        depth = depth + wp.int64(1)
        if s == p:
            n = n + wp.int64(1)
    count[p] = n
    leaf_depth[p] = depth


@wp.kernel(enable_backward=False)
def _node_fill_kernel(
    codes: wp.array(dtype=wp.int64),
    tree: wp.array(dtype=wp.int64),
    tree_start: wp.array(dtype=wp.int64),
    tree_end: wp.array(dtype=wp.int64),
    leaf_size: wp.int64,
    balance: wp.int64,
    offset: wp.array(dtype=wp.int64),
    node_depth: wp.array(dtype=wp.int64),
    node_count: wp.array(dtype=wp.int64),
):
    """The depth and item count of the nodes whose range starts at sorted
    position ``tid``, written from ``offset[tid]`` on, from the root down."""
    p = wp.int64(wp.tid())
    t = tree[p]
    s = tree_start[t]
    e = tree_end[t]
    depth = wp.int64(0)
    j = offset[p]
    if s == p:
        node_depth[j] = depth
        node_count[j] = e - s
        j = j + wp.int64(1)
    while e - s > leaf_size:
        q = _morton_split(codes, s, e, balance)
        if p < q:
            e = q
        else:
            s = q
        depth = depth + wp.int64(1)
        if s == p:
            node_depth[j] = depth
            node_count[j] = e - s
            j = j + wp.int64(1)


def _launch(kernel, inputs: list, ref: torch.Tensor) -> None:
    """Launch ``kernel`` over ``ref.shape[0]`` threads on the device and current
    stream of ``ref``."""
    device, stream = FunctionSpec.warp_launch_context(ref)
    with FunctionSpec.warp_stream_scope(stream):
        wp.init()
        arrays = [
            wp.from_torch(x, dtype=wp.int64) if isinstance(x, torch.Tensor) else x
            for x in inputs
        ]
        wp.launch(kernel, dim=ref.shape[0], inputs=arrays, device=device, stream=stream)


class ForestTopology(NamedTuple):
    """Output of :func:`build_morton_forest_topology`.

    ``topology`` is the forest's: its nodes tree by tree, each tree's nodes
    numbered as :func:`~physicsnemo.mesh.spatial._lbvh.build_morton_topology`
    numbers them, its ``max_depth`` the largest tree depth. ``root_ids`` holds
    the root of each nonempty tree, ``node_tree`` the tree of each node and
    ``node_offset`` the first node of each tree; the lists hold the items and
    nodes of each tree, and ``tree_depths`` the depth of each nonempty tree
    (int64, shape ``(n_trees,)``; any value for an empty tree).
    """

    topology: LBVHTopology
    root_ids: torch.Tensor
    node_tree: torch.Tensor
    node_offset: torch.Tensor
    tree_items: list[int]
    tree_nodes: list[int]
    tree_depths: torch.Tensor


def build_morton_forest_topology(
    codes: torch.Tensor,
    tree: torch.Tensor,
    tree_counts: torch.Tensor,
    leaf_size: int,
    *,
    balance: int,
) -> ForestTopology | None:
    """Build the Morton topology of every tree of a forest at once.

    Parameters
    ----------
    codes : torch.Tensor
        Morton codes, int64, shape ``(n_items,)`` with ``1 <= n_items <
        2**31`` (Warp's thread ids are int32): each tree's codes sorted in non-decreasing order, the
        trees laid end to end in order.
    tree : torch.Tensor
        Tree of each position, int64, shape ``(n_items,)``.
    tree_counts : torch.Tensor
        Items per tree, int64, shape ``(n_trees,)``; trees may be empty.
    leaf_size : int
        Maximum items per leaf (``>= 1``).
    balance : int
        As in :func:`~physicsnemo.mesh.spatial._lbvh.build_morton_topology`.

    Returns
    -------
    ForestTopology or None
        The forest's topology, or ``None`` when a node key cannot hold the
        forest (too many trees, or internal nodes at depth
        ``_DEPTH_BINS - 2`` or deeper).
    """
    n = codes.shape[0]
    n_trees = tree_counts.shape[0]
    device = codes.device
    start_bits = n.bit_length()
    key_bits = n_trees.bit_length() + 6 + start_bits
    if key_bits > 63:
        return None
    tree_end = torch.cumsum(tree_counts, 0)
    tree_start = tree_end - tree_counts

    ### Count pass: the nodes starting at each position; then one readback.
    count = torch.empty(n, dtype=torch.long, device=device)
    leaf_depth = torch.empty(n, dtype=torch.long, device=device)
    fixed = [codes, tree, tree_start, tree_end, int(leaf_size), int(balance)]
    _launch(_node_count_kernel, [*fixed, count, leaf_depth], codes)
    ### Each tree's nodes from a prefix sum, not atomic adds, which would all
    ### land on one element for one tree.
    first_node = F.pad(torch.cumsum(count, 0), (1, 0))
    nodes_per = first_node[tree_end] - first_node[tree_start]
    ### The internal nodes starting at position p have the depths
    ### leaf_depth - count + 1 to leaf_depth - 1: counted per depth by a
    ### difference array, position p adding to copy p % _LANES of the bins.
    starts_here = (count > 0).long()
    lane = torch.arange(n, device=device) % _LANES * _DEPTH_BINS
    bins = torch.zeros(_LANES * _DEPTH_BINS, dtype=torch.long, device=device)
    bins.index_add_(
        0, lane + (leaf_depth - count + 1).clamp_max(_DEPTH_BINS - 1), starts_here
    )
    bins.index_add_(0, lane + leaf_depth.clamp_max(_DEPTH_BINS - 1), -starts_here)
    bins = bins.view(_LANES, _DEPTH_BINS).sum(0)
    stats = torch.cat([tree_counts, nodes_per, torch.cumsum(bins, 0)]).tolist()
    items_per_h = stats[:n_trees]
    nodes_per_h = stats[n_trees : 2 * n_trees]
    internal_per_depth = stats[2 * n_trees :]
    if internal_per_depth[_DEPTH_BINS - 2]:
        return None
    n_nodes = sum(nodes_per_h)
    n_internal = sum(internal_per_depth)
    n_levels = sum(1 for c in internal_per_depth if c)

    ### Fill pass, then the nodes in the torch build's numbering: tree by
    ### tree, each tree level by level from its root, a level by first
    ### position.
    depth = torch.empty(n_nodes, dtype=torch.long, device=device)
    size = torch.empty(n_nodes, dtype=torch.long, device=device)
    _launch(_node_fill_kernel, [*fixed, first_node[:-1], depth, size], codes)
    start = torch.repeat_interleave(
        torch.arange(n, device=device), count, output_size=n_nodes
    )
    key = ((tree[start] << 6 | depth) << start_bits) | start
    if key_bits <= 31:
        key = key.int()  # half the radix sort's passes
    key, order = torch.sort(key)
    range_start, range_count, depth = start[order], size[order], depth[order]
    is_leaf = range_count <= leaf_size
    ### A node's left child is the node one level deeper with its first
    ### position; the right child follows it.
    child = torch.searchsorted(key, key + (1 << start_bits))
    left = torch.where(is_leaf, -1, child)
    right = torch.where(is_leaf, -1, child + 1)
    leaf_ids = torch.nonzero_static(is_leaf, size=n_nodes - n_internal)[:, 0]
    leaf_starts = range_start[leaf_ids]
    leaf_sizes = range_count[leaf_ids]
    ### Internal nodes by depth, each depth in ascending id, as the torch
    ### build lists them (the order of the ids themselves for one tree).
    internal = torch.nonzero_static(~is_leaf, size=n_internal)[:, 0]
    if n_trees > 1:
        internal = internal[torch.argsort(depth[internal], stable=True)]
    by_depth = list(torch.split(internal, internal_per_depth[:n_levels]))
    topo = LBVHTopology(
        left_child=left,
        right_child=right,
        leaf_start=torch.where(is_leaf, range_start, -1),
        leaf_count=torch.where(is_leaf, range_count, 0),
        range_start=range_start,
        range_count=range_count,
        node_count=n_nodes,
        max_nodes=n_nodes,
        internal_nodes_per_level=by_depth,
        leaf_node_ids=leaf_ids,
        leaf_starts=leaf_starts,
        leaf_sizes=leaf_sizes,
        max_depth=n_levels,
    )
    node_tree = tree[range_start]
    n_roots = sum(1 for c in items_per_h if c)
    node_offset = torch.cumsum(nodes_per, 0) - nodes_per
    return ForestTopology(
        topology=topo,
        root_ids=torch.nonzero_static(depth == 0, size=n_roots)[:, 0],
        node_tree=node_tree,
        node_offset=node_offset,
        tree_items=items_per_h,
        tree_nodes=nodes_per_h,
        ### A tree's depth is its last node's, its nodes being by depth.
        tree_depths=depth[(node_offset + nodes_per - 1).clamp_min(0)],
    )


@wp.func_native(
    """
#if defined(__CUDA_ARCH__)
    __threadfence();
#endif
"""
)
def _fence(): ...


@wp.func
def _min(a: Any, b: Any):
    """The smaller of ``a`` and ``b``, NaN if either is, as ``torch.amin``."""
    r = b
    if wp.isnan(a) or a < b:
        r = a
    return r


@wp.func
def _max(a: Any, b: Any):
    """The larger of ``a`` and ``b``, NaN if either is, as ``torch.amax``."""
    r = b
    if wp.isnan(a) or a > b:
        r = a
    return r


@wp.kernel(enable_backward=False)
def _node_boxes_kernel(
    points: wp.array(dtype=Any),
    leaf_node: wp.array(dtype=wp.int64),
    leaf_start: wp.array(dtype=wp.int64),
    leaf_size: wp.array(dtype=wp.int64),
    parent: wp.array(dtype=wp.int64),
    left: wp.array(dtype=wp.int64),
    right: wp.array(dtype=wp.int64),
    d: int,
    arrivals: wp.array(dtype=wp.int32),
    box_min: wp.array(dtype=Any),
    box_max: wp.array(dtype=Any),
):
    """Thread ``tid``: the box of leaf ``tid`` from its points, then, for as long
    as it is the second child to arrive at a parent, the parent's box from its
    two children's (the first child to arrive stops)."""
    i = wp.tid()
    node = leaf_node[i]
    s = leaf_start[i]
    for k in range(d):
        lo = points[s * wp.int64(d) + wp.int64(k)]
        hi = lo
        for j in range(1, int(leaf_size[i])):
            x = points[(s + wp.int64(j)) * wp.int64(d) + wp.int64(k)]
            lo = _min(lo, x)
            hi = _max(hi, x)
        box_min[node * wp.int64(d) + wp.int64(k)] = lo
        box_max[node * wp.int64(d) + wp.int64(k)] = hi
    p = parent[node]
    while p >= wp.int64(0):
        ### The boxes written so far are visible before the arrival is.
        _fence()
        if wp.atomic_add(arrivals, int(p), 1) == 0:
            p = wp.int64(-1)
        else:
            a = left[p] * wp.int64(d)
            b = right[p] * wp.int64(d)
            for k in range(d):
                kk = wp.int64(k)
                box_min[p * wp.int64(d) + kk] = _min(box_min[a + kk], box_min[b + kk])
                box_max[p * wp.int64(d) + kk] = _max(box_max[a + kk], box_max[b + kk])
            p = parent[p]


def node_boxes(
    topo: LBVHTopology, sorted_points: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-node AABBs of ``topo`` over the Morton-sorted points, as
    :func:`~physicsnemo.mesh.spatial.cluster_tree._node_boxes` makes them, in
    one kernel launch: leaves from their points, then every internal node by
    the second of its children to finish. Minima and maxima are exact, so the
    boxes are that function's in every bit, whichever child finishes first,
    except where ``+0.0`` ties ``-0.0`` (the zero kept depends on the order of
    comparison) and in NaN payloads (both give NaN in the same entries). Half-precision points
    are widened to float32 and back exactly."""
    n_nodes = topo.node_count
    d = sorted_points.shape[1]
    device = sorted_points.device
    dtype = sorted_points.dtype
    work = torch.float64 if dtype == torch.float64 else torch.float32
    left = topo.left_child[:n_nodes]
    right = topo.right_child[:n_nodes]
    parent = torch.full((n_nodes,), -1, dtype=torch.long, device=device)
    if topo.internal_nodes_per_level:
        internal = torch.cat(topo.internal_nodes_per_level)
        parent[left[internal]] = internal
        parent[right[internal]] = internal
    box_min = torch.empty((n_nodes, d), dtype=work, device=device)
    box_max = torch.empty_like(box_min)
    wtype = wp.dtype_from_torch(work)
    wp_device, stream = FunctionSpec.warp_launch_context(sorted_points)
    with FunctionSpec.warp_stream_scope(stream):
        wp.init()
        wp.launch(
            _node_boxes_kernel,
            dim=topo.leaf_node_ids.shape[0],
            inputs=[
                wp.from_torch(
                    sorted_points.to(work).contiguous().view(-1), dtype=wtype
                ),
                *(
                    wp.from_torch(x.contiguous(), dtype=wp.int64)
                    for x in (
                        topo.leaf_node_ids,
                        topo.leaf_starts,
                        topo.leaf_sizes,
                        parent,
                        left,
                        right,
                    )
                ),
                d,
                wp.from_torch(
                    torch.zeros(n_nodes, dtype=torch.int32, device=device),
                    dtype=wp.int32,
                ),
                wp.from_torch(box_min.view(-1), dtype=wtype),
                wp.from_torch(box_max.view(-1), dtype=wtype),
            ],
            device=wp_device,
            stream=stream,
        )
    return box_min.to(dtype), box_max.to(dtype)
