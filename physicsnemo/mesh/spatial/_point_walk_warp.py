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

r"""The walk of :meth:`~physicsnemo.mesh.spatial.cluster_tree.ClusterTree.find_point_interaction_pairs`
in two Warp kernels, with the torch walk's pairs in its order.

The torch walk visits the tree level by level for every target at once: at
each level the opened nodes hand on their left children, then their right
children. Here one thread walks one tree of the forest for one target. A
depth-first count pass sizes each thread's segments of the two streams; a
breadth-first fill pass writes the entries in the torch walk's order, each
level a buffer of the opened nodes' (left, right) pairs read even entries
first, so no sort follows. The opening test is the torch walk's arithmetic,
operation by operation in the dtype of its traversal table, in a module
compiled without contracted multiply-adds, so every decision is torch's.
Under autocast, which steps of the test run in bfloat16 depends on the
device's autocast lists (CUDA's runs ``pow`` and ``sum`` in float32): the
dtype of each step is read from torch's dispatch, and the kernels compute in
float32 and round to bfloat16 or float16 where torch's kernels round (each
elementwise result; a sum once, after accumulating in float32).

Imported lazily by :class:`~physicsnemo.mesh.spatial.cluster_tree.ClusterTree`,
so importing :mod:`physicsnemo.mesh.spatial` registers no kernels.
"""

from typing import Any

import torch
import warp as wp

from physicsnemo.core.function_spec import FunctionSpec

### Each product and sum of the opening test is rounded on its own, as torch's
### separate kernels round it. These options apply to this module only.
wp.set_module_options({"fuse_fp": False, "enable_backward": False})

#: Entries of the count pass's depth-first stack: a walk deeper than about
#: this many levels is refused by the count pass.
WALK_STACK = 96
_WALK_STACK = wp.constant(WALK_STACK)
_walk_stack = wp.types.vector(length=WALK_STACK, dtype=wp.int32)


@wp.func
def _low(x: Any, low: int):
    """``x`` rounded to a storage precision: bfloat16 (``low = 1``), float16
    (``2``), or kept (``0``)."""
    y = x
    if low == 1:
        y = type(x)(wp.bfloat16(wp.float32(x)))
    elif low == 2:
        y = type(x)(wp.float16(wp.float32(x)))
    return y


@wp.func
def _gap_sq(
    pts: wp.array(dtype=Any),
    q: int,
    lo: wp.array(dtype=Any),
    hi: wp.array(dtype=Any),
    m: int,
    d: int,
    k: int,
    mode: int,
    zero: Any,
):
    """``clamp(maximum(p - max, min - p), 0) ** 2`` on axis ``k``, NaN where
    torch's maximum and clamp would propagate one."""
    p = pts[q * d + k]
    a = _low(p - hi[m * d + k], mode & 3)
    b = _low(lo[m * d + k] - p, mode & 3)
    g = wp.max(wp.max(a, b), zero)
    sq = _low(g * g, (mode >> 2) & 3)
    if wp.isnan(a) or wp.isnan(b):
        sq = type(zero)(wp.nan)
    return sq


@wp.func
def _is_far(
    pts: wp.array(dtype=Any),
    q: int,
    lo: wp.array(dtype=Any),
    hi: wp.array(dtype=Any),
    diam_sq: wp.array(dtype=Any),
    admissible: wp.array(dtype=wp.bool),
    m: int,
    d: int,
    theta_sq: Any,
    mode: int,
):
    """The torch walk's admission of node ``m`` for target ``q``: the squared
    gap from the point to the box summed over the axes in torch's order, times
    ``theta_sq``, above the squared diagonal, and the node admissible; a NaN
    gap is never far. ``mode`` packs the roundings of the differences, the
    squares and the sum and product (bits 0-1, 2-3, 4-5; see :func:`_low`)."""
    zero = type(theta_sq)(0.0)
    s0 = _gap_sq(pts, q, lo, hi, m, d, 0, mode, zero)
    s1 = _gap_sq(pts, q, lo, hi, m, d, 1, mode, zero)
    s2 = zero
    if d == 3:
        s2 = _gap_sq(pts, q, lo, hi, m, d, 2, mode, zero)
    dist_sq = (s0 + s1) + s2
    low = (mode >> 4) & 3
    far = bool(False)
    if _low(_low(dist_sq, low) * theta_sq, low) > diam_sq[m]:
        far = admissible[m]
    return far


@wp.kernel
def _count_kernel(
    pts: wp.array(dtype=Any),
    lo: wp.array(dtype=Any),
    hi: wp.array(dtype=Any),
    diam_sq: wp.array(dtype=Any),
    admissible: wp.array(dtype=wp.bool),
    left: wp.array(dtype=wp.int64),
    right: wp.array(dtype=wp.int64),
    leaf_count: wp.array(dtype=wp.int64),
    roots: wp.array(dtype=wp.int64),
    theta_sq: Any,
    mode: int,
    d: int,
    n_q: int,
    n_t: int,
    n_near: wp.array(dtype=wp.int64),
    n_far: wp.array(dtype=wp.int64),
    n_visit: wp.array(dtype=wp.int64),
    overflow: wp.array(dtype=wp.int64),
):
    """Thread ``t * n_q + q``: target ``q``'s near sources, far nodes and
    visited nodes in tree ``t``, written at ``q * n_t + t``. A walk whose stack
    would pass ``WALK_STACK`` entries stops and sets ``overflow``."""
    tid = wp.tid()
    t = tid // n_q
    q = tid - t * n_q
    stack = _walk_stack()
    stack[0] = wp.int32(roots[t])
    top = int(1)
    near = wp.int64(0)
    far = wp.int64(0)
    visit = wp.int64(0)
    while top > 0:
        top -= 1
        m = int(stack[top])
        visit += wp.int64(1)
        if _is_far(pts, q, lo, hi, diam_sq, admissible, m, d, theta_sq, mode):
            far += wp.int64(1)
        elif left[m] < wp.int64(0):
            near += leaf_count[m]
        elif top + 2 > _WALK_STACK:
            overflow[0] = wp.int64(1)
            top = 0
        else:
            stack[top] = wp.int32(right[m])
            stack[top + 1] = wp.int32(left[m])
            top += 2
    i = q * n_t + t
    n_near[i] = near
    n_far[i] = far
    n_visit[i] = visit


@wp.kernel
def _fill_kernel(
    pts: wp.array(dtype=Any),
    lo: wp.array(dtype=Any),
    hi: wp.array(dtype=Any),
    diam_sq: wp.array(dtype=Any),
    admissible: wp.array(dtype=wp.bool),
    left: wp.array(dtype=wp.int64),
    right: wp.array(dtype=wp.int64),
    leaf_count: wp.array(dtype=wp.int64),
    roots: wp.array(dtype=wp.int64),
    theta_sq: Any,
    mode: int,
    d: int,
    n_q: int,
    n_t: int,
    leaf_start: wp.array(dtype=wp.int64),
    order: wp.array(dtype=wp.int64),
    near_start: wp.array(dtype=wp.int64),
    far_start: wp.array(dtype=wp.int64),
    visit_start: wp.array(dtype=wp.int64),
    n_visit: wp.array(dtype=wp.int64),
    scratch: wp.array(dtype=wp.int32),
    near_target: wp.array(dtype=wp.int64),
    near_source: wp.array(dtype=wp.int64),
    far_target: wp.array(dtype=wp.int64),
    far_node: wp.array(dtype=wp.int64),
):
    """Thread ``t * n_q + q``: target ``q``'s entries in tree ``t``, level by
    level, near sources from ``near_start`` and far nodes from ``far_start``.
    A level's buffer holds the previous level's opened nodes as (left, right)
    pairs in its order, so its even entries then its odd ones are the torch
    walk's order at that depth. The levels alternate between the front of the
    thread's scratch (entry ``j`` at ``base + j``) and its back (at
    ``last - j``); two consecutive levels are disjoint sets of the thread's
    visited nodes, so they fit together."""
    tid = wp.tid()
    t = tid // n_q
    q = tid - t * n_q
    i = q * n_t + t
    near_at = near_start[i]
    far_at = far_start[i]
    base = visit_start[i]
    last = base + n_visit[i] - wp.int64(1)
    scratch[base] = wp.int32(roots[t])
    width = int(1)
    front = bool(True)
    q64 = wp.int64(q)
    while width > 0:
        opened = int(0)
        for parity in range(2):
            j = int(parity)
            while j < width:
                m = int(0)
                if front:
                    m = int(scratch[base + wp.int64(j)])
                else:
                    m = int(scratch[last - wp.int64(j)])
                if _is_far(pts, q, lo, hi, diam_sq, admissible, m, d, theta_sq, mode):
                    far_target[far_at] = q64
                    far_node[far_at] = wp.int64(m)
                    far_at += wp.int64(1)
                elif left[m] < wp.int64(0):
                    s = leaf_start[m]
                    for c in range(int(leaf_count[m])):
                        near_target[near_at] = q64
                        near_source[near_at] = order[s + wp.int64(c)]
                        near_at += wp.int64(1)
                else:
                    if front:
                        scratch[last - wp.int64(opened)] = wp.int32(left[m])
                        scratch[last - wp.int64(opened + 1)] = wp.int32(right[m])
                    else:
                        scratch[base + wp.int64(opened)] = wp.int32(left[m])
                        scratch[base + wp.int64(opened + 1)] = wp.int32(right[m])
                    opened += 2
                j += 2
        width = opened
        front = not front


#: The kernels' rounding (:func:`_low`) for each dtype a step of the test runs in.
_LOW = {torch.bfloat16: 1, torch.float16: 2}


def _opening_dtypes(tree, theta_sq: float) -> tuple[torch.dtype, ...]:
    """The dtypes of the torch walk's opening test, read from torch's own
    dispatch on empty tensors under the current autocast state: the traversal
    table's (``_traversal_tables``' concatenation of the boxes and the
    diagonals, which the targets are cast to), the squares' and the sum's."""
    d = tree.n_spatial_dims
    lo, hi, diam_sq = (
        x[:0] for x in (tree.node_aabb_min, tree.node_aabb_max, tree.node_diameter_sq)
    )
    box = torch.cat([lo.t(), hi.t(), diam_sq.sqrt()[None], diam_sq[None]], dim=0)
    p = box[:d]
    sq = torch.clamp(torch.maximum(p - box[d : 2 * d], box[:d] - p), min=0).pow(2)
    return box.dtype, sq.dtype, (sq.sum(dim=0) * theta_sq).dtype


def point_walk(
    tree,
    targets: torch.Tensor,
    theta: float,
    source_admissible: torch.Tensor | None,
    *,
    count_only: bool,
    max_elements: int,
) -> tuple[torch.Tensor, ...] | None:
    """The walk of ``find_point_interaction_pairs`` for a nonempty tree of 2
    or 3 dimensions, nonempty ``targets`` of its dimension on its device (the
    caller checks them) and ``theta > 0``: ``(near_target,
    near_source, far_target, far_node)``, or with ``count_only`` the counts
    ``(n_near, n_far)`` per target. ``None`` when a walk passes the stack or
    an array would hold ``max_elements`` entries or more. One host read."""
    device = tree.node_aabb_min.device
    n_q, d = targets.shape[0], tree.n_spatial_dims
    n_t = tree.root_ids.shape[0]
    theta_sq = theta * theta
    ### Reduced-precision values are widened exactly to float32 and the kernels
    ### round where torch's kernels round.
    dtype, sq_dtype, sum_dtype = _opening_dtypes(tree, theta_sq)
    work = torch.float64 if dtype == torch.float64 else torch.float32
    mode = _LOW.get(dtype, 0) | _LOW.get(sq_dtype, 0) << 2 | _LOW.get(sum_dtype, 0) << 4
    scalar = wp.float64 if work == torch.float64 else wp.float32
    wtype = wp.dtype_from_torch(work)

    def as_long(x):
        return wp.from_torch(x.contiguous(), dtype=wp.int64)

    def as_real(x):
        return wp.from_torch(x.to(dtype).to(work).contiguous().reshape(-1), dtype=wtype)

    if source_admissible is None:
        source_admissible = torch.ones(tree.n_nodes, dtype=torch.bool, device=device)
    walk = [
        as_real(targets.detach()),
        as_real(tree.node_aabb_min),
        as_real(tree.node_aabb_max),
        as_real(tree.node_diameter_sq),
        wp.from_torch(source_admissible.contiguous(), dtype=wp.bool),
        as_long(tree.node_left_child),
        as_long(tree.node_right_child),
        as_long(tree.leaf_count),
        as_long(tree.root_ids),
        scalar(theta_sq),
        mode,
        d,
        n_q,
        n_t,
    ]
    n_near, n_far, n_visit = (
        torch.empty(n_q * n_t, dtype=torch.long, device=device) for _ in range(3)
    )
    overflow = torch.zeros(1, dtype=torch.long, device=device)
    wp_device, stream = FunctionSpec.warp_launch_context(targets)
    with FunctionSpec.warp_stream_scope(stream):
        wp.init()
        wp.launch(
            _count_kernel,
            dim=n_q * n_t,
            inputs=[*walk, *(as_long(x) for x in (n_near, n_far, n_visit, overflow))],
            device=wp_device,
            stream=stream,
        )
    if count_only:
        if overflow.item():
            return None
        return n_near.view(n_q, n_t).sum(1), n_far.view(n_q, n_t).sum(1)
    near_end = n_near.cumsum(0)
    far_end = n_far.cumsum(0)
    visit_end = n_visit.cumsum(0)
    n_near_total, n_far_total, n_scratch, overflowed = torch.cat(
        [near_end[-1:], far_end[-1:], visit_end[-1:], overflow]
    ).tolist()
    if overflowed or max(n_near_total, n_far_total, n_scratch) >= max_elements:
        return None
    near_target, near_source = (
        torch.empty(n_near_total, dtype=torch.long, device=device) for _ in range(2)
    )
    far_target, far_node = (
        torch.empty(n_far_total, dtype=torch.long, device=device) for _ in range(2)
    )
    scratch = torch.empty(n_scratch, dtype=torch.int32, device=device)
    with FunctionSpec.warp_stream_scope(stream):
        wp.launch(
            _fill_kernel,
            dim=n_q * n_t,
            inputs=[
                *walk,
                as_long(tree.leaf_start),
                as_long(tree.sorted_source_order),
                as_long(near_end - n_near),
                as_long(far_end - n_far),
                as_long(visit_end - n_visit),
                as_long(n_visit),
                wp.from_torch(scratch, dtype=wp.int32),
                *(as_long(x) for x in (near_target, near_source, far_target, far_node)),
            ],
            device=wp_device,
            stream=stream,
        )
    return near_target, near_source, far_target, far_node
