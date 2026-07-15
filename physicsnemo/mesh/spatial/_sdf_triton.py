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

"""Triton per-thread depth-first nearest-triangle search for the torch SDF.

This is the GPU fast path behind
:func:`physicsnemo.mesh.spatial.sdf.signed_distance_field`. It reproduces
the design of a hand-written CUDA mesh-query kernel (one thread per query, a
small per-thread stack, descend the nearer child first, prune subtrees whose
AABB is farther than the running best) -- the only structure that achieves
single-kernel, launch-overhead-free traversal -- but writes its results into
torch-allocated tensors so it never mixes a foreign allocator with torch's.

The tree itself is still built by :class:`physicsnemo.mesh.spatial.BVH`
(``BVH.from_mesh``, a balanced midpoint-split LBVH). Only the nearest-triangle
search runs in the kernel; signs are computed afterwards by the caller (via the
nearest face pseudo-normal, or the :class:`~physicsnemo.mesh.spatial.ClusterTree`
winding-number summation). The pure-PyTorch traversal in
:mod:`physicsnemo.mesh.spatial.sdf` remains the CPU / no-Triton reference
implementation and the parity oracle.
"""

from __future__ import annotations

import torch

from physicsnemo.core.version_check import OptionalImport
from physicsnemo.mesh.spatial import BVH
from physicsnemo.mesh.spatial.bvh import _compute_morton_codes

triton = OptionalImport("triton")
_libdevice = OptionalImport("triton.language.extra.libdevice")


def _morton_order(points: torch.Tensor) -> torch.Tensor:
    """Permutation that sorts ``points`` along a Z-order (Morton) curve.

    The interior query points arrive spatially shuffled (the mesh reader
    pre-shuffles on-disk point order so a contiguous block is representative).
    Reordering them so spatial neighbors land in the same warp makes the
    per-thread BVH DFS far more coherent -- lanes in a block follow similar
    root-to-leaf paths, so the block's synchronized traversal does much less
    masked / divergent work.
    """
    return torch.argsort(_compute_morton_codes(points))


# Per-query DFS stack depth. The midpoint-split LBVH is balanced, so its depth is
# ~log2(n_faces); 64 slots covers >1e18 faces with comfortable headroom for the
# transient two-child push before the next pop.
_STACK_SIZE = 64

# float32 smallest normal, matching ``torch.finfo(torch.float32).tiny`` used as
# the denominator clamp in the torch reference's region classification.
_TINY = 1.1754943508222875e-38


def available() -> bool:
    """Return ``True`` when the Triton fast path can be used."""
    return bool(triton.available)


if triton.available:
    tl = triton.language

    # Triton @jit functions may only read globals declared as constexpr.
    _TINY_C = tl.constexpr(_TINY)

    @triton.jit
    def _node_dist_sq(qx, qy, qz, aabb_ptr, node, valid):
        """Squared distance from each query to its node's AABB (0 if inside).

        ``aabb_ptr`` packs each node's bounds contiguously as
        ``(n_nodes, 6)`` -- ``min(xyz)`` then ``max(xyz)`` -- so a node's six
        floats land in one ~32-byte segment, giving the per-lane gather better
        cache locality than two separate ``(n_nodes, 3)`` arrays.
        """
        minx = tl.load(aabb_ptr + node * 6 + 0, mask=valid, other=0.0)
        miny = tl.load(aabb_ptr + node * 6 + 1, mask=valid, other=0.0)
        minz = tl.load(aabb_ptr + node * 6 + 2, mask=valid, other=0.0)
        maxx = tl.load(aabb_ptr + node * 6 + 3, mask=valid, other=0.0)
        maxy = tl.load(aabb_ptr + node * 6 + 4, mask=valid, other=0.0)
        maxz = tl.load(aabb_ptr + node * 6 + 5, mask=valid, other=0.0)
        dx = tl.maximum(qx - maxx, 0.0) + tl.maximum(minx - qx, 0.0)
        dy = tl.maximum(qy - maxy, 0.0) + tl.maximum(miny - qy, 0.0)
        dz = tl.maximum(qz - maxz, 0.0) + tl.maximum(minz - qz, 0.0)
        return dx * dx + dy * dy + dz * dz

    @triton.jit
    def _closest_point_on_triangle(px, py, pz, ax, ay, az, bx, by, bz, cx, cy, cz):
        """Closest point on triangle (a, b, c) to p (Ericson region table).

        Mirrors ``physicsnemo.mesh.spatial.sdf._closest_point_on_triangles``
        exactly, including the precedence in which the region results are
        layered (face, then the three vertex regions, then the three edge
        regions), so the two implementations agree on degenerate / boundary
        cases.
        """
        abx = bx - ax
        aby = by - ay
        abz = bz - az
        acx = cx - ax
        acy = cy - ay
        acz = cz - az
        apx = px - ax
        apy = py - ay
        apz = pz - az

        d1 = abx * apx + aby * apy + abz * apz
        d2 = acx * apx + acy * apy + acz * apz

        bpx = px - bx
        bpy = py - by
        bpz = pz - bz
        d3 = abx * bpx + aby * bpy + abz * bpz
        d4 = acx * bpx + acy * bpy + acz * bpz

        cpx = px - cx
        cpy = py - cy
        cpz = pz - cz
        d5 = abx * cpx + aby * cpy + abz * cpz
        d6 = acx * cpx + acy * cpy + acz * cpz

        vc = d1 * d4 - d3 * d2
        vb = d5 * d2 - d1 * d6
        va = d3 * d6 - d5 * d4

        denom = tl.maximum(va + vb + vc, _TINY_C)
        v_face = vb / denom
        w_face = vc / denom
        rx = ax + abx * v_face + acx * w_face
        ry = ay + aby * v_face + acy * w_face
        rz = az + abz * v_face + acz * w_face

        # Vertex region A: d1 <= 0 and d2 <= 0
        mask_a = (d1 <= 0.0) & (d2 <= 0.0)
        rx = tl.where(mask_a, ax, rx)
        ry = tl.where(mask_a, ay, ry)
        rz = tl.where(mask_a, az, rz)

        # Vertex region B: d3 >= 0 and d4 <= d3
        mask_b = (d3 >= 0.0) & (d4 <= d3)
        rx = tl.where(mask_b, bx, rx)
        ry = tl.where(mask_b, by, ry)
        rz = tl.where(mask_b, bz, rz)

        # Vertex region C: d6 >= 0 and d5 <= d6
        mask_c = (d6 >= 0.0) & (d5 <= d6)
        rx = tl.where(mask_c, cx, rx)
        ry = tl.where(mask_c, cy, ry)
        rz = tl.where(mask_c, cz, rz)

        # Edge AB
        mask_ab = (vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0) & (~mask_a) & (~mask_b)
        t_ab = d1 / tl.maximum(d1 - d3, _TINY_C)
        t_ab = tl.minimum(tl.maximum(t_ab, 0.0), 1.0)
        rx = tl.where(mask_ab, ax + abx * t_ab, rx)
        ry = tl.where(mask_ab, ay + aby * t_ab, ry)
        rz = tl.where(mask_ab, az + abz * t_ab, rz)

        # Edge AC
        mask_ac = (vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0) & (~mask_a) & (~mask_c)
        t_ac = d2 / tl.maximum(d2 - d6, _TINY_C)
        t_ac = tl.minimum(tl.maximum(t_ac, 0.0), 1.0)
        rx = tl.where(mask_ac, ax + acx * t_ac, rx)
        ry = tl.where(mask_ac, ay + acy * t_ac, ry)
        rz = tl.where(mask_ac, az + acz * t_ac, rz)

        # Edge BC
        mask_bc = (
            (va <= 0.0)
            & ((d4 - d3) >= 0.0)
            & ((d5 - d6) >= 0.0)
            & (~mask_b)
            & (~mask_c)
        )
        t_bc = (d4 - d3) / tl.maximum((d4 - d3) + (d5 - d6), _TINY_C)
        t_bc = tl.minimum(tl.maximum(t_bc, 0.0), 1.0)
        rx = tl.where(mask_bc, bx + (cx - bx) * t_bc, rx)
        ry = tl.where(mask_bc, by + (cy - by) * t_bc, ry)
        rz = tl.where(mask_bc, bz + (cz - bz) * t_bc, rz)

        return rx, ry, rz

    # Light, query-count-keyed autotuning. The kernel runs one query per lane
    # with a block-synchronized DFS, so the only meaningful launch knob is how
    # many lanes share a program (the divergence-vs-occupancy tradeoff). Each
    # config keeps ``BLOCK == num_warps * 32`` to preserve the one-query-per-
    # thread mapping while sweeping that tradeoff. The set is deliberately tiny
    # (3 configs) because a single launch can take seconds on large meshes and
    # the autotuner benchmarks every config. Keying on ``N`` alone means a fixed
    # query workload (e.g. a training loop, or a mesh-size sweep at constant
    # query count) tunes exactly once and then hits the cache. The ``stack``
    # scratch is indexed by the global query id, so it is independent of
    # ``BLOCK`` and safe to retune.
    @triton.autotune(
        configs=[
            triton.Config({"BLOCK": 64}, num_warps=2),
            triton.Config({"BLOCK": 128}, num_warps=4),
            triton.Config({"BLOCK": 256}, num_warps=8),
        ],
        key=["N"],
    )
    @triton.jit
    def _nearest_triangle_kernel(
        query_ptr,  # (N, 3) f32
        fv_ptr,  # (n_faces, 9) f32  -- leaf-sorted: a(xyz), b(xyz), c(xyz)
        aabb_ptr,  # (n_nodes, 6) f32  -- min(xyz), max(xyz)
        left_ptr,  # (n_nodes,) i32
        right_ptr,  # (n_nodes,) i32
        lstart_ptr,  # (n_nodes,) i32
        lcount_ptr,  # (n_nodes,) i32
        stack_ptr,  # (STACK_SIZE, N) i32 scratch (depth-major: coalesced lanes)
        out_dist_ptr,  # (N,) f32  best squared distance
        out_face_ptr,  # (N,) i32  best face index
        out_pt_ptr,  # (N, 3) f32 closest point
        N,
        max_dist_sq,
        BLOCK: tl.constexpr,
        STACK_SIZE: tl.constexpr,
        MAX_LEAF: tl.constexpr,
    ):
        """One query per lane; bounded-stack near-first DFS for nearest triangle."""
        pid = tl.program_id(0)
        # int64 index base: ``off`` feeds element-offset arithmetic (``off * 3``
        # for queries, ``depth * N + off`` for the per-lane stack). At tens of
        # millions of queries the default int32 product silently overflows and
        # the kernel reads/writes wrong addresses, so widen before the multiply.
        off = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
        m = off < N

        qx = tl.load(query_ptr + off * 3 + 0, mask=m, other=0.0)
        qy = tl.load(query_ptr + off * 3 + 1, mask=m, other=0.0)
        qz = tl.load(query_ptr + off * 3 + 2, mask=m, other=0.0)

        best = tl.zeros((BLOCK,), tl.float32) + max_dist_sq
        best_face = tl.zeros((BLOCK,), tl.int32)
        bpx = qx
        bpy = qy
        bpz = qz

        # Seed each lane's stack with the root node (0) and size 1. The stack is
        # depth-major (``depth * N + off``): at a shared depth, adjacent lanes
        # map to adjacent addresses, so coherent pushes/pops coalesce. Root sits
        # at depth 0, i.e. element ``off``.
        sp = tl.where(m, 1, 0).to(tl.int32)
        tl.store(stack_ptr + off, tl.zeros((BLOCK,), tl.int32), mask=m)

        # Each node is pushed at most once per lane (one parent per node), so the
        # DFS pops a finite number of nodes and the loop is guaranteed to
        # terminate without an explicit iteration cap.
        active = sp > 0
        while tl.sum(active.to(tl.int32)) > 0:
            # --- Pop the top node from every active lane.
            ptr = sp - 1
            node = tl.load(stack_ptr + ptr.to(tl.int64) * N + off, mask=active, other=0)
            sp = tl.where(active, ptr, sp)

            # --- Prune: skip nodes that can no longer beat the running bound.
            lower_sq = _node_dist_sq(qx, qy, qz, aabb_ptr, node, active)
            proceed = active & (lower_sq < best)

            lcount = tl.load(lcount_ptr + node, mask=proceed, other=0)
            is_leaf = proceed & (lcount > 0)
            is_internal = proceed & (lcount <= 0)

            # --- Leaf: evaluate exact point-to-triangle distance per cell.
            lstart = tl.load(lstart_ptr + node, mask=is_leaf, other=0)
            for ci in tl.static_range(0, MAX_LEAF):
                cell_valid = is_leaf & (ci < lcount)
                # ``fv`` is pre-sorted into leaf order, so the leaf position is
                # the triangle row directly -- no ``sorted_cell_order`` load. The
                # caller maps this leaf position back to the original face id.
                cell = lstart + ci
                ax = tl.load(fv_ptr + cell * 9 + 0, mask=cell_valid, other=0.0)
                ay = tl.load(fv_ptr + cell * 9 + 1, mask=cell_valid, other=0.0)
                az = tl.load(fv_ptr + cell * 9 + 2, mask=cell_valid, other=0.0)
                bx = tl.load(fv_ptr + cell * 9 + 3, mask=cell_valid, other=0.0)
                by = tl.load(fv_ptr + cell * 9 + 4, mask=cell_valid, other=0.0)
                bz = tl.load(fv_ptr + cell * 9 + 5, mask=cell_valid, other=0.0)
                cx = tl.load(fv_ptr + cell * 9 + 6, mask=cell_valid, other=0.0)
                cy = tl.load(fv_ptr + cell * 9 + 7, mask=cell_valid, other=0.0)
                cz = tl.load(fv_ptr + cell * 9 + 8, mask=cell_valid, other=0.0)

                cpx, cpy, cpz = _closest_point_on_triangle(
                    qx, qy, qz, ax, ay, az, bx, by, bz, cx, cy, cz
                )
                dsq = (
                    (qx - cpx) * (qx - cpx)
                    + (qy - cpy) * (qy - cpy)
                    + (qz - cpz) * (qz - cpz)
                )
                better = cell_valid & (dsq < best)
                best = tl.where(better, dsq, best)
                best_face = tl.where(better, cell, best_face)
                bpx = tl.where(better, cpx, bpx)
                bpy = tl.where(better, cpy, bpy)
                bpz = tl.where(better, cpz, bpz)

            # --- Internal: push both children, nearer one on top of the stack.
            left = tl.load(left_ptr + node, mask=is_internal, other=-1)
            right = tl.load(right_ptr + node, mask=is_internal, other=-1)
            left_valid = is_internal & (left >= 0)
            right_valid = is_internal & (right >= 0)

            d_left = _node_dist_sq(qx, qy, qz, aabb_ptr, left, left_valid)
            d_right = _node_dist_sq(qx, qy, qz, aabb_ptr, right, right_valid)
            inf = tl.full((BLOCK,), float("inf"), tl.float32)
            d_left = tl.where(left_valid, d_left, inf)
            d_right = tl.where(right_valid, d_right, inf)

            left_first = d_left <= d_right
            near = tl.where(left_first, left, right)
            far = tl.where(left_first, right, left)
            near_valid = tl.where(left_first, left_valid, right_valid)
            far_valid = tl.where(left_first, right_valid, left_valid)

            # Prune at push time: a child whose AABB lower bound already exceeds
            # the running best cannot hold a closer triangle, so never push it.
            # ``d_left``/``d_right`` are reused here (already computed for the
            # near-first ordering), so this is effectively free and keeps
            # prunable subtrees out of the stack entirely. ``best`` only shrinks
            # later, so the pop-time prune above still catches nodes that become
            # prunable after they were pushed.
            d_near = tl.where(left_first, d_left, d_right)
            d_far = tl.where(left_first, d_right, d_left)
            near_valid = near_valid & (d_near < best)
            far_valid = far_valid & (d_far < best)

            # Push the farther child first so it sits below the nearer child.
            tl.store(stack_ptr + sp.to(tl.int64) * N + off, far, mask=far_valid)
            sp = tl.where(far_valid, sp + 1, sp)
            tl.store(stack_ptr + sp.to(tl.int64) * N + off, near, mask=near_valid)
            sp = tl.where(near_valid, sp + 1, sp)

            active = sp > 0

        tl.store(out_dist_ptr + off, best, mask=m)
        tl.store(out_face_ptr + off, best_face, mask=m)
        tl.store(out_pt_ptr + off * 3 + 0, bpx, mask=m)
        tl.store(out_pt_ptr + off * 3 + 1, bpy, mask=m)
        tl.store(out_pt_ptr + off * 3 + 2, bpz, mask=m)


def nearest_triangle_triton(
    bvh: BVH,
    face_vertices: torch.Tensor,
    query: torch.Tensor,
    max_dist: float,
    leaf_size: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Nearest triangle per query via the Triton bounded-stack DFS kernel.

    Parameters
    ----------
    bvh : BVH
        BVH built over the triangle AABBs (``BVH.from_mesh``).
    face_vertices : torch.Tensor
        Per-face vertex positions, shape ``(n_faces, 3, 3)``.
    query : torch.Tensor
        Query points, shape ``(n_queries, 3)``, on a CUDA device.
    max_dist : float
        Search radius: triangles farther than this are ignored. Pass
        ``float("inf")`` for an unbounded, exact nearest search. A query with no
        triangle within ``max_dist`` returns ``best_dist_sq == max_dist ** 2``
        and ``best_point == query`` (an unchanged closest point), which the
        caller treats as a miss.
    leaf_size : int, optional
        The ``leaf_size`` the BVH was built with (``BVH.from_mesh``'s default is
        1). A midpoint-split leaf holds at most ``leaf_size`` cells, so this is a
        sync-free upper bound on ``MAX_LEAF`` -- avoiding a per-call host readback
        of ``bvh.leaf_count.max()``.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ``(best_dist_sq, best_face, best_point)`` per query: squared distance to,
        index (int64) of, and closest point on the nearest triangle.
    """
    device = query.device
    n_queries = query.shape[0]

    if n_queries == 0:
        return (
            torch.empty(0, dtype=torch.float32, device=device),
            torch.empty(0, dtype=torch.long, device=device),
            torch.empty(0, 3, dtype=torch.float32, device=device),
        )

    n_faces = face_vertices.shape[0]
    query_c = query.reshape(-1, 3).to(torch.float32).contiguous()
    # Reorder the triangle payload into BVH leaf order: leaf position ``i`` holds
    # original face ``cell_order[i]``. Storing triangles this way makes a leaf's
    # cells contiguous and lets the kernel index them by leaf position
    # (``leaf_start + ci``) -- dropping the per-cell ``sorted_cell_order``
    # indirection load and coalescing the 9-float triangle gather for
    # warp-coherent lanes. The kernel records the leaf position as the winning
    # "face"; we map it back to the original face id before returning.
    cell_order = bvh.sorted_cell_order.to(torch.long)
    fv = face_vertices.reshape(n_faces, 9).to(torch.float32)
    fv_sorted = fv[cell_order].contiguous()
    # Pack node bounds as (n_nodes, 6) = min(xyz) | max(xyz) so each node's six
    # floats are contiguous, improving the per-lane AABB gather's cache locality.
    node_aabb = torch.cat(
        [bvh.node_aabb_min.to(torch.float32), bvh.node_aabb_max.to(torch.float32)],
        dim=1,
    ).contiguous()
    left = bvh.node_left_child.to(torch.int32).contiguous()
    right = bvh.node_right_child.to(torch.int32).contiguous()
    lstart = bvh.leaf_start.to(torch.int32).contiguous()
    lcount = bvh.leaf_count.to(torch.int32).contiguous()

    # Reorder queries along a Morton curve for warp coherence; unsorted at the
    # end. Outputs are written/allocated in sorted order, then scattered back.
    perm = _morton_order(query_c)
    query_s = query_c[perm].contiguous()

    out_dist_s = torch.empty(n_queries, dtype=torch.float32, device=device)
    out_face_s = torch.empty(n_queries, dtype=torch.int32, device=device)
    out_pt_s = torch.empty(n_queries, 3, dtype=torch.float32, device=device)

    # Bounded inner leaf loop. A midpoint-split leaf holds at most ``leaf_size``
    # cells, so this static bound is correct without reading ``lcount.max()`` back
    # to the host (that readback stalled the prefetch stream).
    max_leaf = max(1, leaf_size)

    # Depth-major scratch: shape (STACK_SIZE, n_queries) so that, at a shared
    # DFS depth, adjacent lanes index adjacent memory and the push/pop traffic
    # coalesces. Indexed in the kernel as ``depth * N + off``.
    stack = torch.empty(_STACK_SIZE, n_queries, dtype=torch.int32, device=device)

    # ``BLOCK`` (and ``num_warps``) come from the autotuner; the grid must be a
    # meta-aware callable so it tracks the chosen block size.
    def grid(meta):
        return ((n_queries + meta["BLOCK"] - 1) // meta["BLOCK"],)

    _nearest_triangle_kernel[grid](
        query_s,
        fv_sorted,
        node_aabb,
        left,
        right,
        lstart,
        lcount,
        stack,
        out_dist_s,
        out_face_s,
        out_pt_s,
        n_queries,
        float(max_dist) ** 2,
        STACK_SIZE=_STACK_SIZE,
        MAX_LEAF=max_leaf,
    )

    best_dist_sq = torch.empty_like(out_dist_s)
    best_face = torch.empty_like(out_face_s)
    best_point = torch.empty_like(out_pt_s)
    best_dist_sq[perm] = out_dist_s
    best_face[perm] = out_face_s
    best_point[perm] = out_pt_s

    # ``out_face_s`` holds BVH leaf positions (the kernel runs on leaf-sorted
    # triangles); map them back to original face ids.
    return best_dist_sq, cell_order[best_face.long()], best_point
