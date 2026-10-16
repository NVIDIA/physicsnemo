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

"""Exact k-nearest-neighbor search with a Morton-order BVH and Warp kernels.

The points are sorted along a Morton (Z-order) curve and grouped into leaves
of ``LEAF_SIZE`` consecutive points. The tree over the leaves is a complete
binary tree in heap layout (the children of node ``i`` are ``2i + 1`` and
``2i + 2``), so it is built with a handful of torch reductions. One Warp
thread per query then walks the tree with a shrinking radius: it visits the
nearer child first and prunes every node whose box is no nearer than its
current k-th best distance. The initial radius is the k-th distance among the
``2k`` points next to the query on the Morton curve, which is a valid upper
bound. Queries are processed in Morton order for coherent memory access.

The search is exact up to float32 rounding. All coordinates are processed in
float32, with inputs of up to three dimensions padded with zeros.
"""

import torch
import warp as wp

from physicsnemo.core.function_spec import FunctionSpec

from .utils import validate_inputs

wp.config.log_level = wp.LOG_WARNING

LEAF_SIZE = 8
# The traversal stack holds at most one pending sibling per tree level, plus
# the node being expanded. With fewer than 2**31 points (int32 indices) and
# LEAF_SIZE = 8, the heap tree has at most 28 levels, so 32 entries suffice.
STACK_SIZE = 32
_MORTON_BITS = 21  # 3 x 21 bits fit in a non-negative int64

_MORTON_MAX = float((1 << _MORTON_BITS) - 1)


@wp.func
def _spread_bits_3d(v: wp.uint64) -> wp.uint64:
    # Move bit ``b`` of a 21-bit integer to position ``3 * b``.
    v = (v | (v << wp.uint64(32))) & wp.uint64(0x001F00000000FFFF)
    v = (v | (v << wp.uint64(16))) & wp.uint64(0x001F0000FF0000FF)
    v = (v | (v << wp.uint64(8))) & wp.uint64(0x100F00F00F00F00F)
    v = (v | (v << wp.uint64(4))) & wp.uint64(0x10C30C30C30C30C3)
    v = (v | (v << wp.uint64(2))) & wp.uint64(0x1249249249249249)
    return v


@wp.func
def _quantize(x: float, lo: float, extent: float) -> wp.uint64:
    return wp.uint64(wp.clamp((x - lo) / extent * _MORTON_MAX, 0.0, _MORTON_MAX))


@wp.kernel(enable_backward=False)
def _morton_kernel(
    x: wp.array(dtype=wp.vec3),
    box: wp.array(dtype=wp.vec3),
    codes: wp.array(dtype=wp.int64),
):
    i = wp.tid()
    p = x[i]
    lo = box[0]
    ext = box[1]
    code = (
        _spread_bits_3d(_quantize(p[0], lo[0], ext[0]))
        | (_spread_bits_3d(_quantize(p[1], lo[1], ext[1])) << wp.uint64(1))
        | (_spread_bits_3d(_quantize(p[2], lo[2], ext[2])) << wp.uint64(2))
    )
    codes[i] = wp.int64(code)


def morton_codes(
    x: torch.Tensor, lo: torch.Tensor, extent: torch.Tensor
) -> torch.Tensor:
    """Return 63-bit Morton codes of float32 ``(n, 3)`` coordinates.

    Each axis is quantized to 21 bits over ``[lo, lo + extent]``
    independently and clamped, so queries outside the points' bounding box
    are valid inputs. Bit ``b`` of axis ``d`` goes to position ``3 * b + d``.
    """
    codes = torch.empty(x.shape[0], dtype=torch.int64, device=x.device)
    if x.shape[0] == 0:
        return codes
    safe_extent = torch.where(extent != 0, extent, torch.ones_like(extent))
    box = torch.stack([lo, safe_extent]).contiguous()
    wp_device, wp_stream = FunctionSpec.warp_launch_context(x)
    with FunctionSpec.warp_stream_scope(wp_stream):
        wp.launch(
            _morton_kernel,
            dim=x.shape[0],
            inputs=[
                wp.from_torch(x, dtype=wp.vec3, return_ctype=True),
                wp.from_torch(box, dtype=wp.vec3, return_ctype=True),
                wp.from_torch(codes, return_ctype=True),
            ],
            device=wp_device,
            stream=wp_stream,
        )
    return codes


def build_morton_tree(points: torch.Tensor) -> dict:
    """Build a heap-layout BVH over Morton-sorted float32 ``(n, 3)`` points.

    Returns a dict of flat tensors: the points in tree order (``tree_points``),
    the permutation from tree order to input order (``order``), the sorted
    Morton codes (``codes``), the quantization box (``lo``, ``extent``), and
    per-node ``node_lo``/``node_hi`` boxes, ``left``/``right`` children (-1 for
    leaves) and ``leaf_start``/``leaf_count`` ranges into the tree order.
    """
    n = points.shape[0]
    device = points.device
    lo = points.min(dim=0).values
    extent = points.max(dim=0).values - lo
    codes = morton_codes(points, lo, extent)
    codes, order = torch.sort(codes)
    tree_points = points[order].contiguous()

    n_leaves = -(-n // LEAF_SIZE)
    n_leaves_pow2 = 1 << max(n_leaves - 1, 0).bit_length()
    n_internal = n_leaves_pow2 - 1

    ### Leaf boxes from padded chunks; empty padding leaves get inverted boxes,
    # whose distance to any query is infinite, so traversal prunes them.
    n_pad = n_leaves_pow2 * LEAF_SIZE - n
    pad_lo = tree_points.new_full((n_pad, 3), float("inf"))
    level_lo = torch.cat([tree_points, pad_lo]).view(-1, LEAF_SIZE, 3).amin(dim=1)
    level_hi = torch.cat([tree_points, -pad_lo]).view(-1, LEAF_SIZE, 3).amax(dim=1)
    levels_lo, levels_hi = [level_lo], [level_hi]
    while level_lo.shape[0] > 1:
        level_lo = level_lo.view(-1, 2, 3).amin(dim=1)
        level_hi = level_hi.view(-1, 2, 3).amax(dim=1)
        levels_lo.append(level_lo)
        levels_hi.append(level_hi)
    node_lo = torch.cat(levels_lo[::-1]).contiguous()
    node_hi = torch.cat(levels_hi[::-1]).contiguous()

    ### Heap topology: internal nodes first, then the leaves in Morton order.
    n_nodes = n_internal + n_leaves_pow2
    ids = torch.arange(n_nodes, dtype=torch.int32, device=device)
    is_leaf = ids >= n_internal
    left = torch.where(is_leaf, -1, 2 * ids + 1)
    right = torch.where(is_leaf, -1, 2 * ids + 2)
    leaf_start = ((ids - n_internal) * LEAF_SIZE).clamp(min=0)
    leaf_count = torch.where(is_leaf, (n - leaf_start).clamp(0, LEAF_SIZE), 0)
    return dict(
        tree_points=tree_points,
        order=order,
        codes=codes,
        lo=lo,
        extent=extent,
        node_lo=node_lo,
        node_hi=node_hi,
        left=left,
        right=right,
        leaf_start=leaf_start.int(),
        leaf_count=leaf_count.int(),
    )


_KERNELS: dict = {}

# Each k up to MAX_STATIC_K gets a kernel with k compiled in. On a GB300 with
# 1M queries, a kernel taking k at run time is 18% slower at k = 11 and 37%
# slower at k = 40. Larger k share one run-time kernel per power of two, so
# that callers whose k varies still compile few kernels.
MAX_STATIC_K = 64


def _kernel_key(k: int) -> tuple[int, bool]:
    """Return ``(capacity, static_k)`` for the kernel that serves ``k``."""
    if k <= MAX_STATIC_K:
        return k, True
    return 1 << (k - 1).bit_length(), False


def get_tree_knn_kernel(capacity: int, static_k: bool):
    """Return the Warp kernel for shrinking-radius kNN.

    With ``static_k`` the kernel serves ``k = capacity``, compiled in.
    Otherwise it takes any ``k <= capacity`` at run time. One thread handles
    query ``query_order[tid]``. Node arrays are flat: ``node_lo``/``node_hi``
    boxes, ``left``/``right`` children (-1 for a leaf, root at 0), and
    ``leaf_start``/``leaf_count`` ranges into ``tree_points``. ``order`` maps
    tree positions to output indices. The initial radius is the k-th distance
    among the ``2k`` tree-order points around ``window_pos[query]``.
    """
    key = (capacity, static_k)
    if key in _KERNELS:
        return _KERNELS[key]

    vec_f = wp.types.vector(length=capacity, dtype=wp.float32)
    vec_i = wp.types.vector(length=capacity, dtype=wp.int32)
    stack_i = wp.types.vector(length=STACK_SIZE, dtype=wp.int32)
    stack_f = wp.types.vector(length=STACK_SIZE, dtype=wp.float32)

    @wp.func
    def box_dist_sq(q: wp.vec3, lo: wp.vec3, hi: wp.vec3) -> float:
        d = wp.max(wp.max(lo - q, q - hi), wp.vec3(0.0))
        return wp.dot(d, d)

    @wp.kernel(enable_backward=False, module="unique")
    def tree_knn(
        tree_points: wp.array(dtype=wp.vec3),
        order: wp.array(dtype=wp.int64),
        node_lo: wp.array(dtype=wp.vec3),
        node_hi: wp.array(dtype=wp.vec3),
        left: wp.array(dtype=wp.int32),
        right: wp.array(dtype=wp.int32),
        leaf_start: wp.array(dtype=wp.int32),
        leaf_count: wp.array(dtype=wp.int32),
        queries: wp.array(dtype=wp.vec3),
        query_order: wp.array(dtype=wp.int32),
        window_pos: wp.array(dtype=wp.int32),
        k: int,
        out_idx: wp.array2d(dtype=wp.int64),
        out_dist: wp.array2d(dtype=wp.float32),
    ):
        if wp.static(static_k):
            kq = int(wp.static(capacity))
        else:
            kq = k
        qi = query_order[wp.tid()]
        q = queries[qi]
        if not wp.isfinite(q):
            for j in range(kq):
                out_idx[qi, j] = wp.int64(-1)
                out_dist[qi, j] = wp.inf
            return
        best_d = vec_f()
        best_i = vec_i()
        last = kq - 1
        n_points = tree_points.shape[0]

        # k-th distance among the 2k points around the query's Morton
        # position: an upper bound on the true k-th distance. ``kth`` keeps
        # the list's k-th entry in a register.
        for j in range(kq):
            best_d[j] = wp.inf
        kth = float(wp.inf)
        width = wp.min(2 * kq, n_points)
        start = wp.clamp(window_pos[qi] - kq, 0, n_points - width)
        for p in range(start, start + width):
            d2 = wp.length_sq(tree_points[p] - q)
            if d2 < kth:
                j = int(last)
                while j > 0 and best_d[j - 1] > d2:
                    best_d[j] = best_d[j - 1]
                    j -= 1
                best_d[j] = d2
                kth = best_d[last]
        # Inflate so the window's own k-th point still enters the k-list.
        kth = kth * 1.000001 + 1.0e-30

        for j in range(kq):
            best_d[j] = kth
            best_i[j] = -1

        st_node = stack_i()
        st_dist = stack_f()
        st_node[0] = 0
        st_dist[0] = box_dist_sq(q, node_lo[0], node_hi[0])
        sp = int(1)
        while sp > 0:
            sp -= 1
            node = st_node[sp]
            if st_dist[sp] >= kth:
                continue
            lc = int(left[node])
            if lc < 0:
                s = int(leaf_start[node])
                for p in range(s, s + int(leaf_count[node])):
                    d2 = wp.length_sq(tree_points[p] - q)
                    if d2 < kth:
                        j = int(last)
                        while j > 0 and best_d[j - 1] > d2:
                            best_d[j] = best_d[j - 1]
                            best_i[j] = best_i[j - 1]
                            j -= 1
                        best_d[j] = d2
                        best_i[j] = p
                        kth = best_d[last]
            else:
                rc = int(right[node])
                dl = box_dist_sq(q, node_lo[lc], node_hi[lc])
                dr = box_dist_sq(q, node_lo[rc], node_hi[rc])
                near = lc
                far = rc
                dn = dl
                df = dr
                if dr < dl:
                    near = rc
                    far = lc
                    dn = dr
                    df = dl
                # Push the farther child first so the nearer one pops first.
                if df < kth and sp < wp.static(STACK_SIZE):
                    st_node[sp] = far
                    st_dist[sp] = df
                    sp += 1
                if dn < kth and sp < wp.static(STACK_SIZE):
                    st_node[sp] = near
                    st_dist[sp] = dn
                    sp += 1

        # A slot stays empty only when no point has a finite distance.
        for j in range(kq):
            if best_i[j] >= 0:
                out_idx[qi, j] = order[best_i[j]]
                out_dist[qi, j] = wp.sqrt(best_d[j])
            else:
                out_idx[qi, j] = wp.int64(-1)
                out_dist[qi, j] = wp.inf

    _KERNELS[key] = tree_knn
    return tree_knn


def tree_knn_query(
    tree: dict,
    queries: torch.Tensor,
    k: int,
    query_order: torch.Tensor,
    window_pos: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the traversal kernel over a flat tree (see :func:`build_morton_tree`).

    ``queries`` are float32 ``(m, 3)``. ``query_order`` (int32) maps threads
    to queries and ``window_pos`` (int32) is each query's position in the tree
    order, both from :func:`morton_query_plan`.
    """
    m = queries.shape[0]
    device = queries.device
    out_idx = torch.empty((m, k), dtype=torch.int64, device=device)
    out_dist = torch.empty((m, k), dtype=torch.float32, device=device)
    if m == 0:
        return out_idx, out_dist
    kernel = get_tree_knn_kernel(*_kernel_key(k))

    def as_vec3(t):
        return wp.from_torch(t, dtype=wp.vec3, return_ctype=True)

    def as_array(t):
        return wp.from_torch(t, return_ctype=True)

    wp_device, wp_stream = FunctionSpec.warp_launch_context(queries)
    with FunctionSpec.warp_stream_scope(wp_stream):
        wp.launch(
            kernel,
            dim=m,
            inputs=[
                as_vec3(tree["tree_points"]),
                as_array(tree["order"]),
                as_vec3(tree["node_lo"]),
                as_vec3(tree["node_hi"]),
                as_array(tree["left"]),
                as_array(tree["right"]),
                as_array(tree["leaf_start"]),
                as_array(tree["leaf_count"]),
                as_vec3(queries),
                as_array(query_order),
                as_array(window_pos),
                k,
                as_array(out_idx),
                as_array(out_dist),
            ],
            device=wp_device,
            stream=wp_stream,
        )
    return out_idx, out_dist


def morton_query_plan(
    tree: dict, queries: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(query_order, window_pos)`` (int32) for float32 ``(m, 3)`` queries.

    ``query_order`` sorts the queries along the points' Morton curve and
    ``window_pos`` is each query's insertion position in the sorted codes.
    """
    q_codes = morton_codes(queries, tree["lo"], tree["extent"])
    window_pos = torch.searchsorted(tree["codes"], q_codes).int()
    query_order = torch.argsort(q_codes).int()
    return query_order, window_pos


def _as_float32_xyz(x: torch.Tensor) -> torch.Tensor:
    x = x.detach().to(torch.float32)
    if x.shape[1] < 3:
        x = torch.nn.functional.pad(x, (0, 3 - x.shape[1]))
    return x.contiguous()


@torch.library.custom_op("physicsnemo::knn_warp", mutates_args=())
def knn_impl(
    points: torch.Tensor, queries: torch.Tensor, k: int = 3
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the ``k`` nearest points to each query, nearest first.

    Parameters
    ----------
    points : torch.Tensor
        Points to search, shape ``(N, D)`` with ``1 <= D <= 3``.
    queries : torch.Tensor
        Query points, shape ``(M, D)``, same device and dtype as ``points``.
    k : int
        Number of neighbors, ``1 <= k <= N``.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(indices, distances)``, each ``(M, k)``: int64 indices into
        ``points`` and Euclidean distances (computed in float32, returned in
        the input dtype), sorted ascending per query. A query with a
        non-finite coordinate, or a slot that no point at a finite distance
        can fill, gets index -1 and distance inf.
    """
    validate_inputs(points, queries)
    if points.ndim != 2 or queries.ndim != 2 or points.shape[1] != queries.shape[1]:
        raise ValueError(
            "`knn` warp implementation expects points (N, D) and queries (M, D), "
            f"got {tuple(points.shape)=} and {tuple(queries.shape)=}"
        )
    if not 1 <= points.shape[1] <= 3:
        raise ValueError(
            f"`knn` warp implementation supports 1 to 3 dimensions, got {points.shape[1]}"
        )
    if not 1 <= k <= points.shape[0]:
        raise ValueError(
            f"`knn` warp implementation needs 1 <= k <= number of points, "
            f"got {k=} and {points.shape[0]} points"
        )
    if max(points.shape[0], queries.shape[0]) >= 2**31:
        raise ValueError(
            "`knn` warp implementation supports fewer than 2**31 points and "
            f"queries, got {points.shape[0]} and {queries.shape[0]}"
        )
    pts = _as_float32_xyz(points)
    qrs = _as_float32_xyz(queries)
    tree = build_morton_tree(pts)
    query_order, window_pos = morton_query_plan(tree, qrs)
    indices, distances = tree_knn_query(tree, qrs, k, query_order, window_pos)
    return indices, distances.to(points.dtype)


@knn_impl.register_fake
def _(
    points: torch.Tensor, queries: torch.Tensor, k: int = 3
) -> tuple[torch.Tensor, torch.Tensor]:
    if points.device != queries.device:
        raise RuntimeError("points and queries must be on the same device")

    dist_output = torch.empty(
        queries.shape[0], k, device=queries.device, dtype=queries.dtype
    )
    idx_output = torch.empty(
        queries.shape[0], k, device=queries.device, dtype=torch.int64
    )

    return idx_output, dist_output
