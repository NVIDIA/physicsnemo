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
:func:`physicsnemo.datapipes.transforms._sdf_torch.signed_distance_field_mesh`.
It reproduces the design of a hand-written CUDA mesh-query kernel (one thread per
query, a small per-thread stack, descend the nearer child first, prune subtrees
whose AABB is farther than the running best) -- the only structure that achieves
single-kernel, launch-overhead-free traversal -- but writes its results into
torch-allocated tensors so it never mixes a foreign allocator with torch's.

The tree itself is still built by :class:`physicsnemo.mesh.spatial.BVH`
(``BVH.from_mesh``, a balanced midpoint-split LBVH). Only the nearest-triangle
search runs in the kernel; signs are computed afterwards in plain PyTorch by the
caller. The pure-PyTorch traversal in ``_sdf_torch`` remains the CPU / no-Triton
reference implementation and the parity oracle.
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


def _node_dipole_aggregates(
    bvh: BVH,
    face_vertices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-node expansion data for the fast (Barnes-Hut) winding number.

    For every BVH node, aggregates over the triangles in its subtree:

    * ``Nsum`` -- sum of area-weighted normals ``0.5 * cross(b-a, c-a)``; this
      is the dipole moment of the node's solid-angle field.
    * ``p`` -- the area-weighted centroid (the expansion center).
    * ``r2`` -- squared radius: the farthest distance from ``p`` to the node
      AABB, i.e. an upper bound on the distance from ``p`` to any contained
      triangle. The traversal opens a node only when the query lies outside
      ``beta`` times this radius.

    Leaf aggregates come from a scatter-add of per-face quantities; internal
    aggregates are formed bottom-up by summing children once both are ready
    (``O(tree_depth)`` vectorized passes). Mirrors the per-triangle solid-angle
    formula used by :func:`_sdf_torch._winding_number_sign`.

    Runs without host readbacks so it stays stream-ordered on the prefetch
    stream: the leaf scatter expands over *all* nodes with a sized
    ``repeat_interleave`` (internal nodes carry ``leaf_count == 0`` and so
    contribute nothing) instead of compacting leaves with ``nonzero``, and the
    bottom-up pass runs a fixed (depth-bounded) number of masked ``where``
    iterations instead of an ``any()``-gated ``while`` with per-iter ``nonzero``.
    """
    device = face_vertices.device
    n_nodes = int(bvh.n_nodes)

    a = face_vertices[:, 0, :]
    b = face_vertices[:, 1, :]
    c = face_vertices[:, 2, :]
    nvec = 0.5 * torch.linalg.cross(b - a, c - a)  # (F, 3) area-weighted normal
    area = nvec.norm(dim=-1)  # (F,)
    aw_centroid = area.unsqueeze(-1) * ((a + b + c) / 3.0)  # (F, 3)

    nsum = torch.zeros(n_nodes, 3, dtype=torch.float32, device=device)
    asum = torch.zeros(n_nodes, dtype=torch.float32, device=device)
    csum = torch.zeros(n_nodes, 3, dtype=torch.float32, device=device)

    # Leaf scatter over all nodes. ``n_cells`` is a host-side shape, so passing it
    # as ``output_size`` lets ``repeat_interleave`` skip its own size readback.
    # The per-position node id is recovered by ``searchsorted`` on the cumulative
    # leaf counts: internal nodes (count 0) share a cumulative offset with the
    # leaf that owns it, and ``right=True`` selects that highest-id owner, so the
    # mapping matches a leaf-only compaction exactly.
    n_cells = bvh.sorted_cell_order.shape[0]
    counts = bvh.leaf_count
    starts = bvh.leaf_start.clamp(min=0)
    node_ids = torch.arange(n_nodes, dtype=torch.long, device=device)
    expanded_node = torch.repeat_interleave(node_ids, counts, output_size=n_cells)
    seg_start = counts.cumsum(0) - counts
    flat = torch.arange(n_cells, dtype=torch.long, device=device)
    seg_ids = torch.searchsorted(seg_start, flat, right=True) - 1
    pos = starts[seg_ids] + (flat - seg_start[seg_ids])
    cell = bvh.sorted_cell_order[pos]
    nsum.index_add_(0, expanded_node, nvec[cell].to(torch.float32))
    asum.index_add_(0, expanded_node, area[cell].to(torch.float32))
    csum.index_add_(0, expanded_node, aw_centroid[cell].to(torch.float32))

    # Bottom-up: an internal node is ready once both children are done. The
    # balanced midpoint-split LBVH has depth ~log2(n_nodes), so bit_length()+1
    # passes always suffice; iterating a fixed host-side count avoids the
    # ``bool(ready.any())`` early-exit readback, and full-width masked ``where``
    # updates avoid the per-iter ``nonzero`` compaction.
    left = bvh.node_left_child
    right = bvh.node_right_child
    internal = (left >= 0) & (right >= 0)
    done = bvh.leaf_count > 0
    left_c = left.clamp(min=0)
    right_c = right.clamp(min=0)
    for _ in range(int(n_nodes).bit_length() + 1):
        ready = internal & (~done) & done[left_c] & done[right_c]
        ready3 = ready.unsqueeze(-1)
        nsum = torch.where(ready3, nsum[left_c] + nsum[right_c], nsum)
        asum = torch.where(ready, asum[left_c] + asum[right_c], asum)
        csum = torch.where(ready3, csum[left_c] + csum[right_c], csum)
        done = done | ready

    tiny = torch.finfo(torch.float32).tiny
    p = csum / asum.clamp(min=tiny).unsqueeze(-1)
    far_corner = torch.maximum(
        (bvh.node_aabb_min - p).abs(), (bvh.node_aabb_max - p).abs()
    )
    r2 = (far_corner * far_corner).sum(-1)
    return nsum.contiguous(), p.contiguous(), r2.contiguous()


if triton.available:
    tl = triton.language
    atan2 = _libdevice.atan2

    # Triton @jit functions may only read globals declared as constexpr.
    _TINY_C = tl.constexpr(_TINY)

    @triton.jit
    def _node_dist_sq(qx, qy, qz, nmin_ptr, nmax_ptr, node, valid):
        """Squared distance from each query to its node's AABB (0 if inside)."""
        minx = tl.load(nmin_ptr + node * 3 + 0, mask=valid, other=0.0)
        miny = tl.load(nmin_ptr + node * 3 + 1, mask=valid, other=0.0)
        minz = tl.load(nmin_ptr + node * 3 + 2, mask=valid, other=0.0)
        maxx = tl.load(nmax_ptr + node * 3 + 0, mask=valid, other=0.0)
        maxy = tl.load(nmax_ptr + node * 3 + 1, mask=valid, other=0.0)
        maxz = tl.load(nmax_ptr + node * 3 + 2, mask=valid, other=0.0)
        dx = tl.maximum(qx - maxx, 0.0) + tl.maximum(minx - qx, 0.0)
        dy = tl.maximum(qy - maxy, 0.0) + tl.maximum(miny - qy, 0.0)
        dz = tl.maximum(qz - maxz, 0.0) + tl.maximum(minz - qz, 0.0)
        return dx * dx + dy * dy + dz * dz

    @triton.jit
    def _closest_point_on_triangle(
        px, py, pz, ax, ay, az, bx, by, bz, cx, cy, cz
    ):
        """Closest point on triangle (a, b, c) to p (Ericson region table).

        Mirrors ``_sdf_torch._closest_point_on_triangles`` exactly, including the
        precedence in which the region results are layered (face, then the three
        vertex regions, then the three edge regions), so the two implementations
        agree on degenerate / boundary cases.
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
            (va <= 0.0) & ((d4 - d3) >= 0.0) & ((d5 - d6) >= 0.0) & (~mask_b) & (~mask_c)
        )
        t_bc = (d4 - d3) / tl.maximum((d4 - d3) + (d5 - d6), _TINY_C)
        t_bc = tl.minimum(tl.maximum(t_bc, 0.0), 1.0)
        rx = tl.where(mask_bc, bx + (cx - bx) * t_bc, rx)
        ry = tl.where(mask_bc, by + (cy - by) * t_bc, ry)
        rz = tl.where(mask_bc, bz + (cz - bz) * t_bc, rz)

        return rx, ry, rz

    @triton.jit
    def _nearest_triangle_kernel(
        query_ptr,  # (N, 3) f32
        fv_ptr,  # (n_faces, 9) f32  -- a(xyz), b(xyz), c(xyz)
        nmin_ptr,  # (n_nodes, 3) f32
        nmax_ptr,  # (n_nodes, 3) f32
        left_ptr,  # (n_nodes,) i32
        right_ptr,  # (n_nodes,) i32
        lstart_ptr,  # (n_nodes,) i32
        lcount_ptr,  # (n_nodes,) i32
        order_ptr,  # (n_cells,) i32
        stack_ptr,  # (N, STACK_SIZE) i32 scratch
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
        off = pid * BLOCK + tl.arange(0, BLOCK)
        m = off < N

        qx = tl.load(query_ptr + off * 3 + 0, mask=m, other=0.0)
        qy = tl.load(query_ptr + off * 3 + 1, mask=m, other=0.0)
        qz = tl.load(query_ptr + off * 3 + 2, mask=m, other=0.0)

        best = tl.zeros((BLOCK,), tl.float32) + max_dist_sq
        best_face = tl.zeros((BLOCK,), tl.int32)
        bpx = qx
        bpy = qy
        bpz = qz

        # Seed each lane's stack with the root node (0) and size 1.
        sp = tl.where(m, 1, 0).to(tl.int32)
        tl.store(stack_ptr + off * STACK_SIZE + 0, tl.zeros((BLOCK,), tl.int32), mask=m)

        # Each node is pushed at most once per lane (one parent per node), so the
        # DFS pops a finite number of nodes and the loop is guaranteed to
        # terminate without an explicit iteration cap.
        active = sp > 0
        while tl.sum(active.to(tl.int32)) > 0:
            # --- Pop the top node from every active lane.
            ptr = sp - 1
            node = tl.load(stack_ptr + off * STACK_SIZE + ptr, mask=active, other=0)
            sp = tl.where(active, ptr, sp)

            # --- Prune: skip nodes that can no longer beat the running bound.
            lower_sq = _node_dist_sq(qx, qy, qz, nmin_ptr, nmax_ptr, node, active)
            proceed = active & (lower_sq < best)

            lcount = tl.load(lcount_ptr + node, mask=proceed, other=0)
            is_leaf = proceed & (lcount > 0)
            is_internal = proceed & (lcount <= 0)

            # --- Leaf: evaluate exact point-to-triangle distance per cell.
            lstart = tl.load(lstart_ptr + node, mask=is_leaf, other=0)
            for ci in tl.static_range(0, MAX_LEAF):
                cell_valid = is_leaf & (ci < lcount)
                cell = tl.load(order_ptr + lstart + ci, mask=cell_valid, other=0)
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
                dsq = (qx - cpx) * (qx - cpx) + (qy - cpy) * (qy - cpy) + (
                    qz - cpz
                ) * (qz - cpz)
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

            d_left = _node_dist_sq(qx, qy, qz, nmin_ptr, nmax_ptr, left, left_valid)
            d_right = _node_dist_sq(qx, qy, qz, nmin_ptr, nmax_ptr, right, right_valid)
            inf = tl.full((BLOCK,), float("inf"), tl.float32)
            d_left = tl.where(left_valid, d_left, inf)
            d_right = tl.where(right_valid, d_right, inf)

            left_first = d_left <= d_right
            near = tl.where(left_first, left, right)
            far = tl.where(left_first, right, left)
            near_valid = tl.where(left_first, left_valid, right_valid)
            far_valid = tl.where(left_first, right_valid, left_valid)

            # Push the farther child first so it sits below the nearer child.
            tl.store(stack_ptr + off * STACK_SIZE + sp, far, mask=far_valid)
            sp = tl.where(far_valid, sp + 1, sp)
            tl.store(stack_ptr + off * STACK_SIZE + sp, near, mask=near_valid)
            sp = tl.where(near_valid, sp + 1, sp)

            active = sp > 0

        tl.store(out_dist_ptr + off, best, mask=m)
        tl.store(out_face_ptr + off, best_face, mask=m)
        tl.store(out_pt_ptr + off * 3 + 0, bpx, mask=m)
        tl.store(out_pt_ptr + off * 3 + 1, bpy, mask=m)
        tl.store(out_pt_ptr + off * 3 + 2, bpz, mask=m)

    @triton.jit
    def _winding_kernel(
        query_ptr,  # (N, 3) f32
        fv_ptr,  # (n_faces, 9) f32
        nsum_ptr,  # (n_nodes, 3) f32  dipole moments (area-weighted normals)
        p_ptr,  # (n_nodes, 3) f32  expansion centers
        r2_ptr,  # (n_nodes,) f32   squared node radius
        left_ptr,  # (n_nodes,) i32
        right_ptr,  # (n_nodes,) i32
        lstart_ptr,  # (n_nodes,) i32
        lcount_ptr,  # (n_nodes,) i32
        order_ptr,  # (n_cells,) i32
        stack_ptr,  # (N, STACK_SIZE) i32 scratch
        out_w_ptr,  # (N,) f32  winding sum (solid-angle units, /4pi applied later)
        N,
        beta_sq,
        BLOCK: tl.constexpr,
        STACK_SIZE: tl.constexpr,
        MAX_LEAF: tl.constexpr,
    ):
        """One query per lane; per-thread DFS accumulating the winding number.

        Far nodes (query outside ``beta`` * node radius) contribute a single
        dipole term; near/leaf nodes are refined / evaluated exactly. The
        accumulated solid angle is divided by ``4*pi`` by the caller.
        """
        pid = tl.program_id(0)
        off = pid * BLOCK + tl.arange(0, BLOCK)
        m = off < N

        qx = tl.load(query_ptr + off * 3 + 0, mask=m, other=0.0)
        qy = tl.load(query_ptr + off * 3 + 1, mask=m, other=0.0)
        qz = tl.load(query_ptr + off * 3 + 2, mask=m, other=0.0)

        w = tl.zeros((BLOCK,), tl.float32)

        sp = tl.where(m, 1, 0).to(tl.int32)
        tl.store(stack_ptr + off * STACK_SIZE + 0, tl.zeros((BLOCK,), tl.int32), mask=m)

        active = sp > 0
        while tl.sum(active.to(tl.int32)) > 0:
            ptr = sp - 1
            node = tl.load(stack_ptr + off * STACK_SIZE + ptr, mask=active, other=0)
            sp = tl.where(active, ptr, sp)

            # Expansion center -> query separation.
            px = tl.load(p_ptr + node * 3 + 0, mask=active, other=0.0)
            py = tl.load(p_ptr + node * 3 + 1, mask=active, other=0.0)
            pz = tl.load(p_ptr + node * 3 + 2, mask=active, other=0.0)
            ex = px - qx
            ey = py - qy
            ez = pz - qz
            rn2 = ex * ex + ey * ey + ez * ez
            rnode2 = tl.load(r2_ptr + node, mask=active, other=0.0)

            lcount = tl.load(lcount_ptr + node, mask=active, other=0)
            is_leaf = active & (lcount > 0)
            is_internal = active & (lcount <= 0)
            far = is_internal & (rn2 > beta_sq * rnode2)
            near = is_internal & (rn2 <= beta_sq * rnode2)

            # --- Far internal node: single dipole contribution.
            nx = tl.load(nsum_ptr + node * 3 + 0, mask=far, other=0.0)
            ny = tl.load(nsum_ptr + node * 3 + 1, mask=far, other=0.0)
            nz = tl.load(nsum_ptr + node * 3 + 2, mask=far, other=0.0)
            rn = tl.sqrt(rn2)
            rn3 = tl.maximum(rn2 * rn, _TINY_C)
            dip = (nx * ex + ny * ey + nz * ez) / rn3
            w = tl.where(far, w + dip, w)

            # --- Leaf node: exact solid angle of each triangle.
            lstart = tl.load(lstart_ptr + node, mask=is_leaf, other=0)
            for ci in tl.static_range(0, MAX_LEAF):
                cell_valid = is_leaf & (ci < lcount)
                cell = tl.load(order_ptr + lstart + ci, mask=cell_valid, other=0)
                ax = tl.load(fv_ptr + cell * 9 + 0, mask=cell_valid, other=0.0) - qx
                ay = tl.load(fv_ptr + cell * 9 + 1, mask=cell_valid, other=0.0) - qy
                az = tl.load(fv_ptr + cell * 9 + 2, mask=cell_valid, other=0.0) - qz
                bx = tl.load(fv_ptr + cell * 9 + 3, mask=cell_valid, other=0.0) - qx
                by = tl.load(fv_ptr + cell * 9 + 4, mask=cell_valid, other=0.0) - qy
                bz = tl.load(fv_ptr + cell * 9 + 5, mask=cell_valid, other=0.0) - qz
                cx = tl.load(fv_ptr + cell * 9 + 6, mask=cell_valid, other=0.0) - qx
                cy = tl.load(fv_ptr + cell * 9 + 7, mask=cell_valid, other=0.0) - qy
                cz = tl.load(fv_ptr + cell * 9 + 8, mask=cell_valid, other=0.0) - qz

                la = tl.sqrt(ax * ax + ay * ay + az * az)
                lb = tl.sqrt(bx * bx + by * by + bz * bz)
                lc = tl.sqrt(cx * cx + cy * cy + cz * cz)

                # numerator: a . (b x c)
                crx = by * cz - bz * cy
                cry = bz * cx - bx * cz
                crz = bx * cy - by * cx
                numer = ax * crx + ay * cry + az * crz
                dab = ax * bx + ay * by + az * bz
                dbc = bx * cx + by * cy + bz * cz
                dca = cx * ax + cy * ay + cz * az
                denom = la * lb * lc + dab * lc + dbc * la + dca * lb
                omega = 2.0 * atan2(numer, denom)
                w = tl.where(cell_valid, w + omega, w)

            # --- Near internal node: descend into both children.
            left = tl.load(left_ptr + node, mask=near, other=-1)
            right = tl.load(right_ptr + node, mask=near, other=-1)
            left_valid = near & (left >= 0)
            right_valid = near & (right >= 0)
            tl.store(stack_ptr + off * STACK_SIZE + sp, left, mask=left_valid)
            sp = tl.where(left_valid, sp + 1, sp)
            tl.store(stack_ptr + off * STACK_SIZE + sp, right, mask=right_valid)
            sp = tl.where(right_valid, sp + 1, sp)

            active = sp > 0

        tl.store(out_w_ptr + off, w, mask=m)


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
        Maximum search radius; queries with no triangle within this distance
        keep the (large) initial bound and an unchanged closest point.
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
    fv = face_vertices.reshape(n_faces, 9).to(torch.float32).contiguous()
    query_c = query.reshape(-1, 3).to(torch.float32).contiguous()
    nmin = bvh.node_aabb_min.to(torch.float32).contiguous()
    nmax = bvh.node_aabb_max.to(torch.float32).contiguous()
    left = bvh.node_left_child.to(torch.int32).contiguous()
    right = bvh.node_right_child.to(torch.int32).contiguous()
    lstart = bvh.leaf_start.to(torch.int32).contiguous()
    lcount = bvh.leaf_count.to(torch.int32).contiguous()
    cell_order = bvh.sorted_cell_order.to(torch.int32).contiguous()

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

    BLOCK = 128
    grid = ((n_queries + BLOCK - 1) // BLOCK,)
    stack = torch.empty(n_queries, _STACK_SIZE, dtype=torch.int32, device=device)

    _nearest_triangle_kernel[grid](
        query_s,
        fv,
        nmin,
        nmax,
        left,
        right,
        lstart,
        lcount,
        cell_order,
        stack,
        out_dist_s,
        out_face_s,
        out_pt_s,
        n_queries,
        float(max_dist) ** 2,
        BLOCK=BLOCK,
        STACK_SIZE=_STACK_SIZE,
        MAX_LEAF=max_leaf,
        num_warps=4,
    )

    best_dist_sq = torch.empty_like(out_dist_s)
    best_face = torch.empty_like(out_face_s)
    best_point = torch.empty_like(out_pt_s)
    best_dist_sq[perm] = out_dist_s
    best_face[perm] = out_face_s
    best_point[perm] = out_pt_s

    return best_dist_sq, best_face.long(), best_point


def winding_sign_triton(
    bvh: BVH,
    face_vertices: torch.Tensor,
    query: torch.Tensor,
    beta: float = 2.0,
    leaf_size: int = 1,
) -> torch.Tensor:
    """SDF sign via the tree-accelerated (Barnes-Hut) generalized winding number.

    Robust for non-watertight / soup geometry, unlike the pseudo-normal sign.
    Each query runs a per-thread DFS that approximates well-separated nodes by a
    single dipole term and evaluates only nearby triangles exactly, giving
    ``O(n_queries * log n_faces)`` work instead of the exact ``O(n_queries *
    n_faces)`` sum in :func:`_sdf_torch._winding_number_sign`.

    Parameters
    ----------
    bvh : BVH
        BVH built over the triangle AABBs (``BVH.from_mesh``).
    face_vertices : torch.Tensor
        Per-face vertex positions, shape ``(n_faces, 3, 3)``.
    query : torch.Tensor
        Query points, shape ``(n_queries, 3)``, on a CUDA device.
    beta : float, optional
        Opening factor: a node is approximated when the query lies farther than
        ``beta`` times the node radius from its expansion center. Larger is
        faster / less accurate; ``2.0`` is a robust default.
    leaf_size : int, optional
        The ``leaf_size`` the BVH was built with; a sync-free upper bound on
        ``MAX_LEAF`` (see :func:`nearest_triangle_triton`).

    Returns
    -------
    torch.Tensor
        Sign per query in ``{-1, +1}`` (``+1`` outside, ``-1`` inside),
        shape ``(n_queries,)``, float32.
    """
    device = query.device
    n_queries = query.shape[0]
    if n_queries == 0:
        return torch.ones(0, dtype=torch.float32, device=device)

    n_faces = face_vertices.shape[0]
    fv = face_vertices.reshape(n_faces, 9).to(torch.float32).contiguous()
    query_c = query.reshape(-1, 3).to(torch.float32).contiguous()

    nsum, p, r2 = _node_dipole_aggregates(bvh, face_vertices)
    left = bvh.node_left_child.to(torch.int32).contiguous()
    right = bvh.node_right_child.to(torch.int32).contiguous()
    lstart = bvh.leaf_start.to(torch.int32).contiguous()
    lcount = bvh.leaf_count.to(torch.int32).contiguous()
    cell_order = bvh.sorted_cell_order.to(torch.int32).contiguous()
    # Sync-free static leaf bound (see nearest_triangle_triton); no lcount.max()
    # host readback on the prefetch stream.
    max_leaf = max(1, leaf_size)

    # Morton reorder for warp coherence (see _morton_order); unsorted at the end.
    perm = _morton_order(query_c)
    query_s = query_c[perm].contiguous()

    winding_s = torch.empty(n_queries, dtype=torch.float32, device=device)
    stack = torch.empty(n_queries, _STACK_SIZE, dtype=torch.int32, device=device)
    BLOCK = 128
    grid = ((n_queries + BLOCK - 1) // BLOCK,)

    _winding_kernel[grid](
        query_s,
        fv,
        nsum,
        p,
        r2,
        left,
        right,
        lstart,
        lcount,
        cell_order,
        stack,
        winding_s,
        n_queries,
        float(beta) ** 2,
        BLOCK=BLOCK,
        STACK_SIZE=_STACK_SIZE,
        MAX_LEAF=max_leaf,
        num_warps=4,
    )

    winding = torch.empty_like(winding_s)
    winding[perm] = winding_s
    winding = winding / (4.0 * torch.pi)
    inside = winding.abs() > 0.5
    return torch.where(
        inside,
        torch.full_like(winding, -1.0),
        torch.ones_like(winding),
    )
