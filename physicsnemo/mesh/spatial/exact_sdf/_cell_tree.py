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

"""The tree over a boundary's triangles and the searches that read it.

One Warp kernel builds a midpoint-split tree over the cells in the Morton order
of their centroids in the boundary's canonical frame, its nodes in depth-first
preorder, with float32 boxes rounded outward. Two traversals read it: a float32
nearest-cell seed and the certified ball search, which returns every cell within
a float64 radius of its query. Segments in two dimensions take a brute force.
"""

import torch
import warp as wp

from physicsnemo.core.function_spec import FunctionSpec
from physicsnemo.mesh import Mesh
from physicsnemo.mesh.spatial._frame import canonical_frame, frame_coordinates
from physicsnemo.mesh.spatial.bvh import _compute_morton_codes
from physicsnemo.mesh.spatial.exact_sdf._weights import (
    _ericson_error_wp,
    _segment_distance2,
    _thin_bounds,
)

#: The most entries Warp takes in one array.
_WARP_MAX_ELEMENTS = 2**31


### (Query, cell) pairs per chunk of the two-dimensional search, and (point,
### subset) pairs per chunk of the least-norm solve; bound their blocks. Both are in
### a custom op, so the loops over chunks do not specialize a dynamic query count
### under torch.compile.
_SEARCH_PAIRS = 2**23


def _morton_order(simplex: torch.Tensor) -> torch.Tensor:
    """The cells ``simplex`` ``(F, 3, 3)`` in the Morton order of their centroids in the
    boundary's canonical frame (:func:`canonical_frame`), stable at equal codes."""
    f = simplex.shape[0]
    e = simplex[:, 1:] - simplex[:, :1]
    frame = canonical_frame(
        simplex.mean(1), torch.linalg.cross(e[:, 0], e[:, 1], dim=-1).norm(dim=-1)
    )
    cells = torch.arange(3 * f, device=simplex.device).reshape(f, 3)
    ### Column products, not a GEMM (see frame_coordinates).
    return _compute_morton_codes(
        Mesh(
            points=frame_coordinates(simplex.reshape(-1, 3), frame), cells=cells
        ).cell_centroids
    ).argsort(stable=True)


def _bottom_up(kernel, levels, arrays, ref: torch.Tensor):
    """``kernel`` over the inner nodes of each level of a BVH, deepest first, one
    launch per level (``arrays`` follow the node ids)."""
    device, stream = FunctionSpec.warp_launch_context(ref)
    with FunctionSpec.warp_stream_scope(stream):
        wp.init()
        for inner in reversed(levels):
            wp.launch(
                kernel,
                dim=inner.shape[0],
                inputs=[wp.from_torch(inner, dtype=wp.int64), *arrays],
                device=device,
                stream=stream,
            )


@wp.kernel
def _node_boxes_kernel(
    inner: wp.array(dtype=wp.int64),
    left: wp.array(dtype=wp.int64),
    right: wp.array(dtype=wp.int64),
    lo: wp.array(dtype=wp.vec3d),
    hi: wp.array(dtype=wp.vec3d),
):
    """The box of each node ``inner[tid]`` from its two children's."""
    i = inner[wp.tid()]
    lo[i] = wp.min(lo[left[i]], lo[right[i]])
    hi[i] = wp.max(hi[left[i]], hi[right[i]])


def _tree_boxes(simplex, left, right, levels, leaf_cell):
    """Each node's box ``(N, 3)`` twice (least and greatest corner): its cell's at a
    leaf, its children's above (one launch per level), in float32 rounded outward."""
    leaf = leaf_cell >= 0
    lo, hi = (
        simplex.new_full((left.shape[0], 3), torch.inf),
        simplex.new_full((left.shape[0], 3), -torch.inf),
    )
    lo[leaf], hi[leaf] = (
        simplex[leaf_cell[leaf]].amin(1),
        simplex[leaf_cell[leaf]].amax(1),
    )
    children = [wp.from_torch(c.contiguous(), dtype=wp.int64) for c in (left, right)]
    _bottom_up(
        _node_boxes_kernel,
        levels,
        [
            *children,
            wp.from_torch(lo, dtype=wp.vec3d),
            wp.from_torch(hi, dtype=wp.vec3d),
        ],
        simplex,
    )
    lo32, hi32 = lo.float(), hi.float()
    lo32 = torch.where(
        lo32.double() > lo, lo32.nextafter(lo32.new_tensor(-torch.inf)), lo32
    )
    hi32 = torch.where(
        hi32.double() < hi, hi32.nextafter(hi32.new_tensor(torch.inf)), hi32
    )
    return lo32, hi32


@wp.kernel
def _preorder_kernel(
    f: wp.int64,
    left: wp.array(dtype=wp.int64),
    right: wp.array(dtype=wp.int64),
    escape: wp.array(dtype=wp.int64),
    leaf_cell: wp.array(dtype=wp.int64),
    depth: wp.array(dtype=wp.int32),
):
    """The midpoint-split tree over ``f`` sorted cells, its nodes in depth-first
    preorder: the thread of sorted position ``i`` descends from the root and writes
    every node whose range starts at ``i`` (a node at ``p`` over ``[s, s + n)`` has its
    left child at ``p + 1`` over ``[s, s + n // 2)``, its right child at ``p + 2 (n //
    2)``, and the node after its subtree at ``p + 2 n - 1``); leaf ``i`` holds the
    cell of sorted position ``i``."""
    i = wp.int64(wp.tid())
    total = wp.int64(2) * f - wp.int64(1)
    p = wp.int64(0)
    s = wp.int64(0)
    n = f
    d = wp.int32(0)
    while n > wp.int64(1):
        h = n // wp.int64(2)
        if i == s:
            left[p] = p + wp.int64(1)
            right[p] = p + wp.int64(2) * h
            e = p + wp.int64(2) * n - wp.int64(1)
            if e >= total:
                e = wp.int64(-1)
            escape[p] = e
            leaf_cell[p] = wp.int64(-1)
            depth[p] = d
        if i < s + h:
            p = p + wp.int64(1)
            n = h
        else:
            p = p + wp.int64(2) * h
            s = s + h
            n = n - h
        d += wp.int32(1)
    left[p] = wp.int64(-1)
    right[p] = wp.int64(-1)
    e = p + wp.int64(1)
    if e >= total:
        e = wp.int64(-1)
    escape[p] = e
    leaf_cell[p] = i
    depth[p] = d


def _inner_per_level(f: int) -> list[int]:
    """The inner nodes at each depth, root first, of the midpoint-split tree over
    ``f`` items with one per leaf (``build_lbvh_topology``'s host recurrence)."""
    sizes, counts = {f: 1}, []
    while any(s > 1 for s in sizes):
        counts.append(sum(c for s, c in sizes.items() if s > 1))
        nxt = {}
        for s, c in sizes.items():
            if s > 1:
                for t in (s // 2, s - s // 2):
                    nxt[t] = nxt.get(t, 0) + c
        sizes = nxt
    return counts


def _preorder_cell_tree(simplex: torch.Tensor):
    """A BVH over the cells ``simplex`` ``(F, 3, 3)``, one cell per leaf (the
    midpoint-split topology of :class:`physicsnemo.mesh.spatial.BVH`), built by one
    kernel, with what a stackless traversal needs: each node's two children, its
    escape pointer (the node after its subtree, -1 at the end), the inner nodes of
    each level, root first, each node's cell (-1 inside), and each node's box ``(N,
    3)`` twice (least and greatest corner), in float32 rounded outward for the
    traversals' pruning. The cells are put in Morton order in the boundary's
    canonical frame (:func:`_morton_order`), so the tree, and the far field of the
    winding number on it, is the same in every pose of the case; the boxes are in
    the queries' frame. Its nodes are in depth-first preorder, so a node's left child
    is the next node, and its leaves hold the cells in sorted order: leaf ``i`` holds
    row ``i`` of ``simplex[order]``, returned with ``order`` ``(F,)`` and
    ``simplex[order]``."""
    f, dev = simplex.shape[0], simplex.device
    order = _morton_order(simplex)
    n = 2 * f - 1
    left, right, escape, leaf_cell = (
        torch.empty(n, dtype=torch.int64, device=dev) for _ in range(4)
    )
    depth = torch.empty(n, dtype=torch.int32, device=dev)
    device, stream = FunctionSpec.warp_launch_context(simplex)
    with FunctionSpec.warp_stream_scope(stream):
        wp.init()
        wp.launch(
            _preorder_kernel,
            dim=f,
            inputs=[
                f,
                *(
                    wp.from_torch(t, dtype=wp.int64)
                    for t in (left, right, escape, leaf_cell)
                ),
                wp.from_torch(depth, dtype=wp.int32),
            ],
            device=device,
            stream=stream,
        )
    counts = _inner_per_level(f)
    inner = torch.argsort(
        torch.where(left >= 0, depth, torch.iinfo(torch.int32).max), stable=True
    )[: sum(counts)]
    levels = list(inner.split(counts))
    s_sorted = simplex[order]
    return (
        (
            left,
            right,
            escape,
            levels,
            leaf_cell,
            *_tree_boxes(s_sorted, left, right, levels, leaf_cell),
        ),
        order,
        s_sorted,
    )


#: Entries of the traversal stack of :func:`_nearest_cells`; a midpoint-split tree of
#: ``F`` cells is ``ceil(log2 F)`` deep and needs one more.
_NEAREST_STACK = 64


_nearest_stack = wp.types.vector(length=_NEAREST_STACK, dtype=wp.int32)


_nearest_gaps = wp.types.vector(length=_NEAREST_STACK, dtype=wp.float32)


@wp.func
def _vec3f(v: wp.vec3d) -> wp.vec3f:
    return wp.vec3f(wp.float32(v[0]), wp.float32(v[1]), wp.float32(v[2]))


@wp.func
def _box_gap2(lo: wp.vec3f, hi: wp.vec3f, x: wp.vec3f) -> wp.float32:
    """The squared distance from ``x`` to the box ``[lo, hi]``."""
    zero = wp.float32(0.0)
    g0 = wp.max(wp.max(lo[0] - x[0], x[0] - hi[0]), zero)
    g1 = wp.max(wp.max(lo[1] - x[1], x[1] - hi[1]), zero)
    g2 = wp.max(wp.max(lo[2] - x[2], x[2] - hi[2]), zero)
    return g0 * g0 + g1 * g1 + g2 * g2


@wp.func
def _triangle_distance2_32(
    p: wp.vec3f, a: wp.vec3f, b: wp.vec3f, c: wp.vec3f
) -> wp.float32:
    """:func:`_triangle_distance2` in float32 (NaN on a degenerate triangle)."""
    zero = wp.float32(0.0)
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = wp.dot(ab, ap)
    d2 = wp.dot(ac, ap)
    if d1 <= zero and d2 <= zero:
        return wp.dot(ap, ap)
    bp = p - b
    d3 = wp.dot(ab, bp)
    d4 = wp.dot(ac, bp)
    if d3 >= zero and d4 <= d3:
        return wp.dot(bp, bp)
    vc = d1 * d4 - d3 * d2
    if vc <= zero and d1 >= zero and d3 <= zero:
        r = ap - ab * (d1 / (d1 - d3))
        return wp.dot(r, r)
    cp = p - c
    d5 = wp.dot(ab, cp)
    d6 = wp.dot(ac, cp)
    if d6 >= zero and d5 <= d6:
        return wp.dot(cp, cp)
    vb = d5 * d2 - d1 * d6
    if vb <= zero and d2 >= zero and d6 <= zero:
        r = ap - ac * (d2 / (d2 - d6))
        return wp.dot(r, r)
    va = d3 * d6 - d5 * d4
    if va <= zero and d4 - d3 >= zero and d5 - d6 >= zero:
        r = bp - (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)))
        return wp.dot(r, r)
    denom = va + vb + vc
    r = ap - ab * (vb / denom) - ac * (vc / denom)
    return wp.dot(r, r)


@wp.kernel
def _nearest_cell_kernel(
    node_lo: wp.array(dtype=wp.vec3f),
    node_hi: wp.array(dtype=wp.vec3f),
    left: wp.array(dtype=wp.int32),
    right: wp.array(dtype=wp.int32),
    leaf_cell: wp.array(dtype=wp.int64),
    corners: wp.array(dtype=wp.vec3d),
    query32: wp.array(dtype=wp.vec3f),
    rows: wp.array(dtype=wp.int64),
    nearest: wp.array(dtype=wp.int64),
):
    """The cell nearest to the query ``rows[tid]`` in float32 (the first found at equal
    distances): a depth-first traversal of the BVH, nearer child first, that skips the
    nodes whose boxes lie no nearer than the nearest cell so far (each stacked with its
    box's distance). A NaN distance (a degenerate cell) is never the nearest; with no
    other cell, ``nearest`` keeps its value."""
    q = rows[wp.tid()]
    x = query32[q]
    best = wp.float32(wp.inf)
    found = wp.int64(-1)
    nodes = _nearest_stack()
    gaps = _nearest_gaps()
    nodes[0] = wp.int32(0)
    gaps[0] = _box_gap2(node_lo[0], node_hi[0], x)
    top = int(1)
    while top > 0:
        top -= 1
        if gaps[top] < best:
            node = nodes[top]
            c = leaf_cell[node]
            if c >= wp.int64(0):
                i = c * wp.int64(3)
                d2 = _triangle_distance2_32(
                    x,
                    _vec3f(corners[i]),
                    _vec3f(corners[i + wp.int64(1)]),
                    _vec3f(corners[i + wp.int64(2)]),
                )
                if d2 < best:
                    best = d2
                    found = c
            else:
                near = left[node]
                far = right[node]
                g_near = _box_gap2(node_lo[near], node_hi[near], x)
                g_far = _box_gap2(node_lo[far], node_hi[far], x)
                ### (at equal box distances, as inside both boxes, the nearer centre first)
                swap = g_far < g_near
                if g_far == g_near:
                    e_near = wp.float32(0.5) * (node_lo[near] + node_hi[near]) - x
                    e_far = wp.float32(0.5) * (node_lo[far] + node_hi[far]) - x
                    swap = wp.dot(e_far, e_far) < wp.dot(e_near, e_near)
                if swap:
                    near = right[node]
                    far = left[node]
                    g = g_near
                    g_near = g_far
                    g_far = g
                if g_far < best:
                    nodes[top] = far
                    gaps[top] = g_far
                    top += 1
                if g_near < best:
                    nodes[top] = near
                    gaps[top] = g_near
                    top += 1
    if found >= wp.int64(0):
        nearest[q] = found


def _nearest_cells(query: torch.Tensor, simplex: torch.Tensor, tree) -> torch.Tensor:
    """For each query ``(Q, 3)``, the index of a cell of ``simplex`` ``(F, 3, 3)``
    nearest to it in float32 ``(Q,)``, by a Warp traversal of the BVH ``tree`` over the
    cells (:func:`_preorder_cell_tree`); cell 0 when every distance is NaN. Its exact distance
    bounds the closest one, so any cell would do for the search radius; a near one
    keeps the radius, and so the candidate pairs, few. The queries are traversed in
    Morton order."""
    left, right, _, levels, leaf_cell, lo, hi = tree
    if len(levels) >= _NEAREST_STACK:
        raise ValueError(
            f"a tree {len(levels)} levels deep exceeds the nearest-cell stack ({_NEAREST_STACK})"
        )
    nearest = torch.zeros(query.shape[0], dtype=torch.int64, device=query.device)
    device, stream = FunctionSpec.warp_launch_context(query)
    with FunctionSpec.warp_stream_scope(stream):
        wp.init()
        wp.launch(
            _nearest_cell_kernel,
            dim=query.shape[0],
            inputs=[
                wp.from_torch(lo.contiguous(), dtype=wp.vec3f),
                wp.from_torch(hi.contiguous(), dtype=wp.vec3f),
                wp.from_torch(left.int().contiguous(), dtype=wp.int32),
                wp.from_torch(right.int().contiguous(), dtype=wp.int32),
                wp.from_torch(leaf_cell.contiguous(), dtype=wp.int64),
                wp.from_torch(simplex.reshape(-1, 3).contiguous(), dtype=wp.vec3d),
                wp.from_torch(query.float().contiguous(), dtype=wp.vec3f),
                wp.from_torch(_compute_morton_codes(query).argsort(), dtype=wp.int64),
                wp.from_torch(nearest, dtype=wp.int64),
            ],
            device=device,
            stream=stream,
        )
    return nearest


@wp.func
def _triangle_distance2(
    p: wp.vec3d, a: wp.vec3d, b: wp.vec3d, c: wp.vec3d
) -> wp.float64:
    """The squared distance from ``p`` to the triangle ``abc`` (Ericson 2005, Sec.
    5.1.5); NaN on a degenerate triangle, which the caller keeps."""
    zero = wp.float64(0.0)
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = wp.dot(ab, ap)
    d2 = wp.dot(ac, ap)
    if d1 <= zero and d2 <= zero:
        return wp.dot(ap, ap)
    bp = p - b
    d3 = wp.dot(ab, bp)
    d4 = wp.dot(ac, bp)
    if d3 >= zero and d4 <= d3:
        return wp.dot(bp, bp)
    vc = d1 * d4 - d3 * d2
    if vc <= zero and d1 >= zero and d3 <= zero:
        r = ap - ab * (d1 / (d1 - d3))
        return wp.dot(r, r)
    cp = p - c
    d5 = wp.dot(ab, cp)
    d6 = wp.dot(ac, cp)
    if d6 >= zero and d5 <= d6:
        return wp.dot(cp, cp)
    vb = d5 * d2 - d1 * d6
    if vb <= zero and d2 >= zero and d6 <= zero:
        r = ap - ac * (d2 / (d2 - d6))
        return wp.dot(r, r)
    va = d3 * d6 - d5 * d4
    if va <= zero and d4 - d3 >= zero and d5 - d6 >= zero:
        r = bp - (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)))
        return wp.dot(r, r)
    denom = va + vb + vc
    r = ap - ab * (vb / denom) - ac * (vc / denom)
    return wp.dot(r, r)


@wp.func
def _beyond(
    x: wp.vec3d,
    a: wp.vec3d,
    b: wp.vec3d,
    c: wp.vec3d,
    thin: wp.vec2d,
    r: wp.float64,
    r2: wp.float64,
) -> bool:
    """Whether the triangle ``abc`` lies provably farther than ``r`` (``r2`` its
    square) from ``x``: Ericson's float64 distance less the bound on its error
    (:func:`_ericson_error_wp`, ``thin`` the cell's ``kappa`` and longest edge,
    :func:`_thin_bounds`) exceeds it; never on a degenerate triangle (NaN)."""
    d2 = _triangle_distance2(x, a, b, c)
    far = d2 > r2
    if far:
        l2 = wp.max(
            wp.max(wp.length_sq(x - a), wp.length_sq(x - b)), wp.length_sq(x - c)
        )
        dt = wp.sqrt(d2)
        far = dt - _ericson_error_wp(dt, thin[0] * l2, thin[1]) > r
    return far


@wp.kernel
def _ball_candidates_kernel(
    node_lo: wp.array(dtype=wp.vec3f),
    node_hi: wp.array(dtype=wp.vec3f),
    left: wp.array(dtype=wp.int32),
    escape: wp.array(dtype=wp.int32),
    leaf_cell: wp.array(dtype=wp.int64),
    corners: wp.array(dtype=wp.vec3d),
    thin: wp.array(dtype=wp.vec2d),
    planes: wp.array(dtype=wp.vec4f),
    query: wp.array(dtype=wp.vec3d),
    query32: wp.array(dtype=wp.vec3f),
    radius: wp.array(dtype=wp.float64),
    radius32: wp.array(dtype=wp.float32),
    reach32: wp.array(dtype=wp.float32),
    rows: wp.array(dtype=wp.int64),
    cap: wp.int64,
    count: wp.array(dtype=wp.int64),
    cells_out: wp.array(dtype=wp.int64),
):
    """Every cell within ``radius`` of the query ``rows[tid]``: a stackless
    depth-first traversal of the BVH (``escape`` is the node after a node's
    subtree), pruning the nodes whose boxes lie farther than ``radius32`` (float32
    boxes rounded outward, a radius padded over float32 rounding), then the cells
    farther than ``radius`` (float64, :func:`_beyond`; ``corners`` holds each cell's
    three vertices, ``thin`` its ``kappa`` and longest edge), after a float32 lower
    bound on their distance rejects the far ones: the distance to the cell's plane
    and beyond its farthest edge, from the unit normals in ``planes`` (the plane's,
    then each edge's outward one in the plane, with their offsets), against
    ``reach32``. Writes the first ``cap`` of them from
    ``tid * cap`` on and counts them all."""
    tid = wp.tid()
    q = rows[tid]
    x = query[q]
    x32 = query32[q]
    r2 = radius[q] * radius[q]
    r2_32 = radius32[q] * radius32[q]
    reach2 = reach32[q] * reach32[q]
    zero = wp.float32(0.0)
    k = wp.int64(0)
    node = wp.int32(0)
    while node >= 0:
        lo = node_lo[node]
        hi = node_hi[node]
        g0 = wp.max(wp.max(lo[0] - x32[0], x32[0] - hi[0]), zero)
        g1 = wp.max(wp.max(lo[1] - x32[1], x32[1] - hi[1]), zero)
        g2 = wp.max(wp.max(lo[2] - x32[2], x32[2] - hi[2]), zero)
        if g0 * g0 + g1 * g1 + g2 * g2 <= r2_32:
            c = leaf_cell[node]
            if c >= wp.int64(0):
                j = c * wp.int64(4)
                h = planes[j]
                s = h[0] * x32[0] + h[1] * x32[1] + h[2] * x32[2] - h[3]
                o = zero
                for e in range(1, 4):
                    h = planes[j + wp.int64(e)]
                    o = wp.max(o, h[0] * x32[0] + h[1] * x32[1] + h[2] * x32[2] - h[3])
                ### (not ``<=``: a NaN, from a degenerate cell, keeps it)
                i = c * wp.int64(3)
                if not (s * s + o * o > reach2) and not _beyond(
                    x,
                    corners[i],
                    corners[i + wp.int64(1)],
                    corners[i + wp.int64(2)],
                    thin[c],
                    radius[q],
                    r2,
                ):
                    if k < cap:
                        cells_out[wp.int64(tid) * cap + k] = c
                    k += wp.int64(1)
                node = escape[node]
            else:
                node = left[node]
        else:
            node = escape[node]
    count[tid] = k


#: Candidate slots per query of the first pass of :func:`_ball_candidates`; a query
#: with more is searched again with as many as it needs. That second pass takes about
#: a whole traversal's time however few queries it holds (DrivAerML's surface: 45 ms
#: for 360 queries of 200k at 16 slots; none overflow 64).
_BALL_SLOTS = 64


def _ball_candidates(
    query: torch.Tensor,
    radius: torch.Tensor,
    simplex: torch.Tensor,
    tree,
    reach,
    flat=None,
):
    """The (query, cell) pairs ``(P,)``, ``(P,)`` of every cell of ``simplex`` ``(F,
    3, 3)`` within ``radius`` ``(Q,)`` of its query ``(Q, 3)``, all float64: a Warp
    traversal of the BVH ``tree`` over the cells (:func:`_preorder_cell_tree`),
    with its boxes and the queries in float32 for the pruning (rounded outward and
    padded) and the distances to the cells in float64. One pass fills
    ``_BALL_SLOTS`` slots per query; the queries that overflow are searched again
    with as many slots as they need, so no candidate is dropped. The queries are
    traversed in Morton order, so neighbouring threads visit neighbouring nodes.
    ``reach`` is the greatest norm of a vertex. The zero-area cells ``flat`` ``(F,)``
    (:func:`_zero_area`) have no plane, and the float64 distance is not defined on
    them: they are kept wherever their boxes are reached. Nor are the planes of a
    cell whose float64 normal is not certified to its float32 rounding (``delta >
    u32``, :func:`_thin_bounds`); its float64 distance is rejected only beyond the
    bound on its error (:func:`_beyond`)."""
    dev = query.device
    left, _, escape, _, leaf_cell, lo, hi = tree
    ### Each cell's unit normal and its edges' outward unit normals in its plane
    ### (outward by the vertex order alone, the edges taken cyclically), with their
    ### offsets: a point's distance to the cell is at least the root sum of squares
    ### of its height over the plane and of its largest edge offset.
    a, b, c = simplex.unbind(1)
    n = torch.nn.functional.normalize(
        torch.linalg.cross(b - a, c - a, dim=-1), dim=-1, eps=0.0
    )
    rows_ = [torch.cat([n, (n * a).sum(-1, keepdim=True)], -1)]
    for p0, p1 in ((a, b), (b, c), (c, a)):
        m = torch.nn.functional.normalize(
            torch.linalg.cross(p1 - p0, n, dim=-1), dim=-1, eps=0.0
        )
        rows_.append(torch.cat([m, (m * p0).sum(-1, keepdim=True)], -1))
    planes = torch.stack(rows_, 1).float().reshape(-1, 4)
    delta, kappa, longest = _thin_bounds(simplex)
    ### (NaN fails every rejection test of the kernel)
    unsure = delta > torch.finfo(torch.float32).eps / 2
    corners = simplex
    if flat is not None:
        unsure = unsure | flat
        corners = torch.where(flat[:, None, None], torch.nan, simplex)
    planes = torch.where(unsure.repeat_interleave(4)[:, None], torch.nan, planes)
    q32 = query.float()
    ### (the query's own rounding, then a relative pad far above float32's on the
    ### box test's few operations)
    r32 = ((radius + (query - q32.double()).norm(dim=-1)) * (1 + 1e-5)).float()
    r32 = r32.nextafter(r32.new_tensor(torch.inf))
    ### (the lower bound's float32 rounding: the products and offsets, against the
    ### coordinates' magnitude)
    reach32 = (
        (
            r32.double()
            + 16 * torch.finfo(torch.float32).eps * (query.norm(dim=-1) + reach)
        )
        * (1 + 1e-5)
    ).float()
    device, stream = FunctionSpec.warp_launch_context(query)
    fixed = [
        wp.from_torch(lo.contiguous(), dtype=wp.vec3f),
        wp.from_torch(hi.contiguous(), dtype=wp.vec3f),
        wp.from_torch(left.int().contiguous(), dtype=wp.int32),
        wp.from_torch(escape.int().contiguous(), dtype=wp.int32),
        wp.from_torch(leaf_cell.contiguous(), dtype=wp.int64),
        wp.from_torch(corners.reshape(-1, 3).contiguous(), dtype=wp.vec3d),
        wp.from_torch(torch.stack([kappa, longest], -1).contiguous(), dtype=wp.vec2d),
        wp.from_torch(planes.contiguous(), dtype=wp.vec4f),
        wp.from_torch(query.contiguous(), dtype=wp.vec3d),
        wp.from_torch(q32.contiguous(), dtype=wp.vec3f),
        wp.from_torch(radius.contiguous(), dtype=wp.float64),
        wp.from_torch(r32.contiguous(), dtype=wp.float32),
        wp.from_torch(reach32.contiguous(), dtype=wp.float32),
    ]

    def search(rows, cap):
        count = torch.zeros(rows.shape[0], dtype=torch.int64, device=dev)
        out = torch.empty(rows.shape[0] * cap, dtype=torch.int64, device=dev)
        ### Blocks of rows whose slots Warp takes as one array (under 2**31 entries:
        ### 33,554,431 rows of 64); one block below that.
        block = (_WARP_MAX_ELEMENTS - 1) // cap
        with FunctionSpec.warp_stream_scope(stream):
            wp.init()
            for s in range(0, rows.shape[0], block):
                e = min(s + block, rows.shape[0])
                wp.launch(
                    _ball_candidates_kernel,
                    dim=e - s,
                    inputs=[
                        *fixed,
                        wp.from_torch(rows[s:e], dtype=wp.int64),
                        cap,
                        wp.from_torch(count[s:e], dtype=wp.int64),
                        wp.from_torch(out[s * cap : e * cap], dtype=wp.int64),
                    ],
                    device=device,
                    stream=stream,
                )
        return count, out.reshape(-1, cap)

    rows = _compute_morton_codes(query).argsort()
    count, out = search(rows, _BALL_SLOTS)
    over = count > _BALL_SLOTS
    keep = (torch.arange(_BALL_SLOTS, device=dev) < count[:, None]) & ~over[:, None]
    li, pk = rows[:, None].expand_as(out)[keep], out[keep]
    if bool(over.any()):
        rows = rows[over]
        count, out = search(rows, int(count.max()))
        keep = torch.arange(out.shape[1], device=dev) < count[:, None]
        li, pk = (
            torch.cat([li, rows[:, None].expand_as(out)[keep]]),
            torch.cat([pk, out[keep]]),
        )
    return li, pk


def _segment_candidates(
    query: torch.Tensor, points: torch.Tensor, cells: torch.Tensor, margin: torch.Tensor
):
    """The (query, segment) pairs whose distance, computed in the queries'
    precision, is within ``margin`` ``(Q,)`` of the query's least: a brute force
    over the segments, chunked over the queries."""
    a, b = points[cells[:, 0]][None], points[cells[:, 1]][None]
    chunk = max(1, _SEARCH_PAIRS // cells.shape[0])
    li, pk = [], []
    for s in range(0, query.shape[0], chunk):
        d2 = _segment_distance2(query[s : s + chunk, None], a, b)
        reach = (
            d2.amin(-1, keepdim=True).sqrt() + margin[s : s + chunk, None]
        ).square()
        i, k = (d2 <= reach).nonzero(as_tuple=True)
        li.append(s + i)
        pk.append(k)
    return torch.cat(li), torch.cat(pk)
