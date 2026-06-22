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

"""Pure-PyTorch signed distance field over a triangle surface mesh.

This is a Warp-free replacement for
:func:`physicsnemo.nn.functional.signed_distance_field`, intended for use in
the datapipes layer. It avoids NVIDIA Warp entirely (and therefore the
stream-ordered CUDA memory churn that arises from mixing Warp's and torch's
allocators) by building the spatial acceleration structure with
:class:`physicsnemo.mesh.spatial.BVH` and computing distances/signs with plain
PyTorch tensor ops.

Layering note
-------------
This module lives in ``physicsnemo.datapipes`` -- the top layer -- because it
imports :mod:`physicsnemo.mesh`. The ``physicsnemo.nn`` layer may not import
``physicsnemo.mesh``, so the Warp-backed primitive still lives in
``physicsnemo.nn.functional`` while this mesh-backed implementation lives here.

Algorithm
---------
1. **Nearest triangle**: a bounded-stack depth-first traversal of the
   morton-LBVH built over triangle AABBs. Each query descends the nearer child
   first with a per-query stack, pruning any node whose AABB lower-bound distance
   exceeds the running best exact triangle distance. Peak memory is
   ``O(n_queries * tree_depth)`` -- it never materializes a breadth-first
   ``(query, node)`` frontier.
2. **Exact distance + closest point**: standard point-to-triangle region
   classification (clamp barycentric coordinates to the triangle), giving the
   unsigned distance and the closest point on the surface.
3. **Sign**:
   - ``use_sign_winding_number=False`` (default): angle-weighted pseudo-normal
     at the closest feature (face / edge / vertex), matching Warp's
     ``mesh_query_point_sign_normal``. Robust for watertight meshes.
   - ``use_sign_winding_number=True``: the generalized winding number (solid
     angle sum, Jacobson et al. 2013) evaluated exactly against every triangle.
     Robust for non-watertight / self-intersecting meshes. This costs
     ``O(n_queries * n_faces)`` and is chunked to bound memory.
"""

from __future__ import annotations

import torch
from jaxtyping import Float, Int

from physicsnemo.datapipes import _timing
from physicsnemo.datapipes.transforms import _sdf_triton
from physicsnemo.mesh import Mesh
from physicsnemo.mesh.spatial import BVH

# Chunk sizes keep the pairwise tensors bounded for large inputs. These are
# product-of-counts limits (rows of the materialized intermediate), not raw
# point counts, so they translate directly to a peak-memory ceiling.
_NEAREST_QUERY_CHUNK = 1 << 18  # queries processed per BVH traversal batch
_WINDING_FACE_CHUNK = 1 << 22  # (query, face) pairs per winding-number tile

# Cells per BVH leaf. Single source of truth: passed to ``BVH.from_mesh`` and to
# the Triton kernels (as their static ``MAX_LEAF`` bound) so the GPU path never
# reads ``leaf_count.max()`` back to the host on the prefetch stream.
_BVH_LEAF_SIZE = 1


def _build_surface_mesh(
    mesh_vertices: Float[torch.Tensor, "n_vertices 3"],
    mesh_indices: Int[torch.Tensor, "..."],
) -> tuple[Mesh, Float[torch.Tensor, "n_faces 3 3"], Int[torch.Tensor, "n_faces 3"]]:
    """Construct a triangle :class:`Mesh` and return per-face vertex positions.

    Parameters
    ----------
    mesh_vertices : torch.Tensor
        Vertex coordinates, shape ``(n_vertices, 3)``.
    mesh_indices : torch.Tensor
        Triangle connectivity, either flattened ``(3 * n_faces,)`` or
        ``(n_faces, 3)``.

    Returns
    -------
    tuple[Mesh, torch.Tensor, torch.Tensor]
        ``(mesh, face_vertices, faces)`` where ``face_vertices`` has shape
        ``(n_faces, 3, 3)`` and ``faces`` has shape ``(n_faces, 3)`` (int64).
    """
    faces = mesh_indices.reshape(-1, 3).to(torch.long)
    mesh = Mesh(points=mesh_vertices, cells=faces)
    face_vertices = mesh_vertices[faces]  # (n_faces, 3, 3)
    return mesh, face_vertices, faces


def _closest_point_on_triangles(
    query: Float[torch.Tensor, "n 3"],
    tri: Float[torch.Tensor, "n 3 3"],
) -> Float[torch.Tensor, "n 3"]:
    """Closest point on each triangle to its paired query point.

    Vectorized region-classification (Ericson, *Real-Time Collision
    Detection*). Computes, for each ``(query, triangle)`` pair, the point on the
    (closed) triangle nearest to ``query``.

    Parameters
    ----------
    query : torch.Tensor
        Query points, shape ``(n, 3)``.
    tri : torch.Tensor
        Triangle vertices, shape ``(n, 3, 3)`` (vertex axis is dim 1).

    Returns
    -------
    torch.Tensor
        Closest points, shape ``(n, 3)``.
    """
    a = tri[:, 0, :]
    b = tri[:, 1, :]
    c = tri[:, 2, :]

    ab = b - a
    ac = c - a
    ap = query - a

    d1 = (ab * ap).sum(-1)
    d2 = (ac * ap).sum(-1)

    bp = query - b
    d3 = (ab * bp).sum(-1)
    d4 = (ac * bp).sum(-1)

    cp = query - c
    d5 = (ab * cp).sum(-1)
    d6 = (ac * cp).sum(-1)

    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4

    # Barycentric-region weights (computed unconditionally, selected by region).
    denom = (va + vb + vc).clamp(min=torch.finfo(query.dtype).tiny)
    v_face = vb / denom
    w_face = vc / denom

    result = a + ab * v_face.unsqueeze(-1) + ac * w_face.unsqueeze(-1)

    # Vertex region A: d1 <= 0 and d2 <= 0
    mask_a = (d1 <= 0) & (d2 <= 0)
    result = torch.where(mask_a.unsqueeze(-1), a, result)

    # Vertex region B: d3 >= 0 and d4 <= d3
    mask_b = (d3 >= 0) & (d4 <= d3)
    result = torch.where(mask_b.unsqueeze(-1), b, result)

    # Vertex region C: d6 >= 0 and d5 <= d6
    mask_c = (d6 >= 0) & (d5 <= d6)
    result = torch.where(mask_c.unsqueeze(-1), c, result)

    # Edge AB: vc <= 0, d1 >= 0, d3 <= 0
    mask_ab = (vc <= 0) & (d1 >= 0) & (d3 <= 0) & ~mask_a & ~mask_b
    t_ab = (d1 / (d1 - d3).clamp(min=torch.finfo(query.dtype).tiny)).clamp(0.0, 1.0)
    proj_ab = a + ab * t_ab.unsqueeze(-1)
    result = torch.where(mask_ab.unsqueeze(-1), proj_ab, result)

    # Edge AC: vb <= 0, d2 >= 0, d6 <= 0
    mask_ac = (vb <= 0) & (d2 >= 0) & (d6 <= 0) & ~mask_a & ~mask_c
    t_ac = (d2 / (d2 - d6).clamp(min=torch.finfo(query.dtype).tiny)).clamp(0.0, 1.0)
    proj_ac = a + ac * t_ac.unsqueeze(-1)
    result = torch.where(mask_ac.unsqueeze(-1), proj_ac, result)

    # Edge BC: va <= 0, (d4 - d3) >= 0, (d5 - d6) >= 0
    mask_bc = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0) & ~mask_b & ~mask_c
    denom_bc = ((d4 - d3) + (d5 - d6)).clamp(min=torch.finfo(query.dtype).tiny)
    t_bc = ((d4 - d3) / denom_bc).clamp(0.0, 1.0)
    proj_bc = b + (c - b) * t_bc.unsqueeze(-1)
    result = torch.where(mask_bc.unsqueeze(-1), proj_bc, result)

    return result


def _aabb_min_distance_sq(
    query: Float[torch.Tensor, "n 3"],
    aabb_min: Float[torch.Tensor, "n 3"],
    aabb_max: Float[torch.Tensor, "n 3"],
) -> Float[torch.Tensor, " n"]:
    """Squared distance from each query to its paired AABB (0 if inside)."""
    over = (query - aabb_max).clamp(min=0.0)
    under = (aabb_min - query).clamp(min=0.0)
    delta = over + under
    return (delta * delta).sum(-1)


def _reduce_candidates_into_best(
    n_queries: int,
    expanded_q: Int[torch.Tensor, " n_cand"],
    cand_dist_sq: Float[torch.Tensor, " n_cand"],
    cand_face: Int[torch.Tensor, " n_cand"],
    closest: Float[torch.Tensor, "n_cand 3"],
    best_dist_sq: Float[torch.Tensor, " n_queries"],
    best_face: Int[torch.Tensor, " n_queries"],
    best_point: Float[torch.Tensor, "n_queries 3"],
) -> None:
    """Scatter-min ``(query, candidate triangle)`` pairs into the running best.

    For every query, reduces its candidate distances to the minimum and, for
    queries that improved on ``best_dist_sq``, writes back the winning face and
    closest point. Ties resolve to the first occurrence. Mutates the three
    ``best_*`` tensors in place.
    """
    device = cand_dist_sq.device
    improved = torch.full(
        (n_queries,), float("inf"), dtype=cand_dist_sq.dtype, device=device
    )
    improved.scatter_reduce_(
        0, expanded_q, cand_dist_sq, reduce="amin", include_self=True
    )
    # A candidate "wins" if it equals the per-query min and beats the current
    # best. Ties resolve arbitrarily (one winning row).
    cand_is_min = cand_dist_sq <= improved[expanded_q]
    cand_beats = cand_dist_sq < best_dist_sq[expanded_q]
    winners = cand_is_min & cand_beats
    # No ``winners.any()`` early-out: that host readback would sync, and the
    # remaining gather/argsort/scatter is a no-op when ``winners`` is all False.
    win_q = expanded_q[winners]
    w_dist = cand_dist_sq[winners]
    w_face = cand_face[winners]
    w_point = closest[winners]
    # Deduplicate winners per query (keep first occurrence).
    order = torch.argsort(win_q, stable=True)
    win_q_sorted = win_q[order]
    first = torch.ones_like(win_q_sorted, dtype=torch.bool)
    first[1:] = win_q_sorted[1:] != win_q_sorted[:-1]
    sel = order[first]
    uq = win_q_sorted[first]
    best_dist_sq[uq] = w_dist[sel]
    best_face[uq] = w_face[sel]
    best_point[uq] = w_point[sel]


def _eval_leaf_candidates(
    bvh: BVH,
    face_vertices: Float[torch.Tensor, "n_faces 3 3"],
    query: Float[torch.Tensor, "n_queries 3"],
    leaf_q: Int[torch.Tensor, " n_leaf_pairs"],
    leaf_n: Int[torch.Tensor, " n_leaf_pairs"],
    n_queries: int,
    best_dist_sq: Float[torch.Tensor, " n_queries"],
    best_face: Int[torch.Tensor, " n_queries"],
    best_point: Float[torch.Tensor, "n_queries 3"],
) -> None:
    """Evaluate exact triangle distances for ``(query, leaf)`` pairs.

    Expands each leaf into its member triangles, computes the exact
    point-to-triangle distance, and folds the results into the running best via
    :func:`_reduce_candidates_into_best`. Mutates the ``best_*`` tensors.
    """
    device = query.device
    starts = bvh.leaf_start[leaf_n]
    counts = bvh.leaf_count[leaf_n]

    # Expand (query, leaf) -> (query, cell) candidate pairs.
    expanded_q = torch.repeat_interleave(leaf_q, counts)
    total = expanded_q.shape[0]
    if total == 0:
        return
    seg_start_flat = counts.cumsum(0) - counts
    flat_idx = torch.arange(total, dtype=torch.long, device=device)
    seg_ids = torch.searchsorted(seg_start_flat, flat_idx, right=True) - 1
    sorted_pos = starts[seg_ids] + (flat_idx - seg_start_flat[seg_ids])
    cand_face = bvh.sorted_cell_order[sorted_pos]

    cand_query_pts = query[expanded_q]
    cand_tris = face_vertices[cand_face]
    closest = _closest_point_on_triangles(cand_query_pts, cand_tris)
    diff = cand_query_pts - closest
    cand_dist_sq = (diff * diff).sum(-1)

    _reduce_candidates_into_best(
        n_queries,
        expanded_q,
        cand_dist_sq,
        cand_face,
        closest,
        best_dist_sq,
        best_face,
        best_point,
    )


def _nearest_face_bvh(
    bvh: BVH,
    face_vertices: Float[torch.Tensor, "n_faces 3 3"],
    query: Float[torch.Tensor, "n_queries 3"],
    max_dist: float,
) -> tuple[
    Float[torch.Tensor, " n_queries"],
    Int[torch.Tensor, " n_queries"],
    Float[torch.Tensor, "n_queries 3"],
]:
    r"""Nearest triangle per query via a bounded-stack depth-first BVH search.

    This is the standard closest-point-on-mesh traversal. Each query keeps an
    explicit fixed-size stack and descends the **nearer child first**, carrying a
    running squared distance ``best_dist_sq`` to its closest triangle so far; a
    subtree is pruned the instant its AABB lower-bound exceeds that bound. Diving
    straight to a leaf makes the bound tight on the very first descent, so the
    far sibling at each level is almost always pruned on the way back up and each
    query visits ``O(log n_faces)`` nodes.

    The data structure is the whole point. A breadth-first ``(query, node)``
    frontier materializes *every* live node for *every* query at once, costing
    ``O(n_queries * nodes_within_radius)`` memory -- catastrophic for interior
    volume points whose distance shell intersects a large patch of surface. A
    per-query stack is instead bounded by the tree depth, so peak memory is
    ``O(n_queries * tree_depth)``, independent of mesh complexity or how far the
    query sits from the surface.

    Parameters
    ----------
    bvh : BVH
        BVH built over the triangle AABBs (``BVH.from_mesh``). Must be a binary
        tree whose depth is logarithmic in ``n_faces`` (the midpoint-split LBVH
        is balanced), so a stack sized to a small multiple of the depth suffices.
    face_vertices : torch.Tensor
        Per-face vertex positions, shape ``(n_faces, 3, 3)``.
    query : torch.Tensor
        Query points, shape ``(n_queries, 3)``.
    max_dist : float
        Maximum search radius; queries with no triangle within this distance
        keep the (large) initial bound and an unchanged closest point.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ``(best_dist_sq, best_face, best_point)`` per query: squared distance to,
        index of, and closest point on the nearest triangle.
    """
    device = query.device
    dtype = query.dtype
    n_queries = query.shape[0]

    best_dist_sq = torch.full(
        (n_queries,), float(max_dist) ** 2, dtype=dtype, device=device
    )
    best_face = torch.zeros(n_queries, dtype=torch.long, device=device)
    best_point = query.clone()

    if bvh.n_nodes == 0 or n_queries == 0:
        return best_dist_sq, best_face, best_point

    # Per-query explicit DFS stack. Near-first descent means the live stack only
    # ever holds the far siblings along the current root-to-node path, i.e. at
    # most the tree depth; sizing to ~2x the depth bound leaves ample headroom
    # for the transient two-child push before the next pop. The balanced LBVH
    # keeps depth ~= log2(n_nodes), so this is a few tens of slots.
    max_depth = max(8, 2 * max(1, int(bvh.n_nodes).bit_length()) + 8)
    stack = torch.zeros((n_queries, max_depth), dtype=torch.long, device=device)
    stack_size = torch.ones(n_queries, dtype=torch.long, device=device)  # root pushed
    all_q = torch.arange(n_queries, dtype=torch.long, device=device)

    # Synchronized traversal: each iteration pops one node from every non-empty
    # stack, prunes/evaluates it, and pushes its children (nearer on top). The
    # loop runs until all stacks drain; the cap is a hard safety bound that never
    # triggers for a well-formed tree.
    # This fallback only runs on CPU / when Triton is unavailable. The
    # ``nonempty.any()`` break and the boolean-index compactions below
    # (``all_q[nonempty]``, ``aq[keep]``, ``pq[is_leaf]``) are intrinsic to a
    # variable-width DFS: the break bounds the iteration count to the deepest
    # live traversal (without it the loop would run the full ``n_nodes`` safety
    # cap every call), so it is kept. These readbacks are free on CPU and only
    # sync on the rare CUDA-without-Triton path, where the Triton kernel is the
    # stream-ordered alternative. The gratuitous ``is_leaf.any()`` /
    # ``internal.any()`` guards, by contrast, are dropped: the leaf/internal
    # branches are no-ops when their masked selection is empty.
    for _ in range(bvh.n_nodes + 1):
        nonempty = stack_size > 0
        if not bool(nonempty.any()):
            break
        aq = all_q[nonempty]
        ptr = stack_size[aq] - 1
        node = stack[aq, ptr]
        stack_size[aq] = ptr  # pop

        # Re-test against the (possibly tightened) bound: a node pushed earlier
        # may no longer be able to beat the current best.
        node_min = bvh.node_aabb_min[node]
        node_max = bvh.node_aabb_max[node]
        lower_sq = _aabb_min_distance_sq(query[aq], node_min, node_max)
        keep = lower_sq < best_dist_sq[aq]
        pq = aq[keep]
        pn = node[keep]

        is_leaf = bvh.leaf_count[pn] > 0

        # --- Leaf nodes: evaluate exact triangle distances, fold into best.
        # No ``is_leaf.any()`` guard: empty selection is a no-op.
        _eval_leaf_candidates(
            bvh,
            face_vertices,
            query,
            pq[is_leaf],
            pn[is_leaf],
            n_queries,
            best_dist_sq,
            best_face,
            best_point,
        )

        # --- Internal nodes: push both children, nearer one on top of stack.
        internal = ~is_leaf
        iq = pq[internal]
        inode = pn[internal]
        left = bvh.node_left_child[inode]
        right = bvh.node_right_child[inode]
        left_valid = left >= 0
        right_valid = right >= 0
        q_int = query[iq]
        inf = torch.full((iq.shape[0],), float("inf"), dtype=dtype, device=device)
        d_left = torch.where(
            left_valid,
            _aabb_min_distance_sq(
                q_int,
                bvh.node_aabb_min[left.clamp(min=0)],
                bvh.node_aabb_max[left.clamp(min=0)],
            ),
            inf,
        )
        d_right = torch.where(
            right_valid,
            _aabb_min_distance_sq(
                q_int,
                bvh.node_aabb_min[right.clamp(min=0)],
                bvh.node_aabb_max[right.clamp(min=0)],
            ),
            inf,
        )
        left_first = d_left <= d_right
        near = torch.where(left_first, left, right)
        far = torch.where(left_first, right, left)
        near_valid = torch.where(left_first, left_valid, right_valid)
        far_valid = torch.where(left_first, right_valid, left_valid)

        # Push the farther child first so it sits *below* the nearer child (which
        # is therefore popped next). Invalid children advance the pointer by 0,
        # so their sentinel slot is harmlessly overwritten. Each query appears
        # once in ``iq``, so these scatter writes never collide.
        sp = stack_size[iq]
        stack[iq, sp] = far
        stack_size[iq] = sp + far_valid.long()
        sp = stack_size[iq]
        stack[iq, sp] = near
        stack_size[iq] = sp + near_valid.long()

    return best_dist_sq, best_face, best_point


def _pseudo_normal_sign(
    mesh: Mesh,
    query: Float[torch.Tensor, "n_queries 3"],
    best_face: Int[torch.Tensor, " n_queries"],
    best_point: Float[torch.Tensor, "n_queries 3"],
) -> Float[torch.Tensor, " n_queries"]:
    """Sign of the SDF via the angle-weighted pseudo-normal of the hit face.

    Uses the face normal of the nearest triangle (consistent with Warp's
    ``mesh_query_point_sign_normal`` for the common face-interior case): the
    point is "outside" (positive) when it lies on the positive side of the hit
    triangle's outward normal.

    Parameters
    ----------
    mesh : Mesh
        Triangle surface mesh (provides cached ``cell_normals``).
    query : torch.Tensor
        Query points, shape ``(n_queries, 3)``.
    best_face : torch.Tensor
        Nearest face index per query, shape ``(n_queries,)``.
    best_point : torch.Tensor
        Closest surface point per query, shape ``(n_queries, 3)``.

    Returns
    -------
    torch.Tensor
        Sign per query in ``{-1, +1}`` (``+1`` outside, ``-1`` inside).
    """
    face_normals = mesh.cell_normals  # (n_faces, 3), unit normals
    hit_normals = face_normals[best_face]  # (n_queries, 3)
    direction = query - best_point
    dot = (direction * hit_normals).sum(-1)
    # Points exactly on the surface (dot == 0) are treated as outside (+1).
    return torch.where(dot < 0, -torch.ones_like(dot), torch.ones_like(dot))


def _winding_number_sign(
    face_vertices: Float[torch.Tensor, "n_faces 3 3"],
    query: Float[torch.Tensor, "n_queries 3"],
) -> Float[torch.Tensor, " n_queries"]:
    """Sign of the SDF via the generalized winding number (solid angle sum).

    For each query the signed solid angle subtended by every triangle is summed
    (Jacobson et al., "Robust Inside-Outside Segmentation using Generalized
    Winding Numbers", 2013) and normalized by ``4*pi``. A winding number near 1
    means inside (negative SDF); near 0 means outside (positive SDF). This is
    robust on non-watertight meshes but costs ``O(n_queries * n_faces)``; the
    face axis is tiled to bound peak memory.

    Parameters
    ----------
    face_vertices : torch.Tensor
        Per-face vertex positions, shape ``(n_faces, 3, 3)``.
    query : torch.Tensor
        Query points, shape ``(n_queries, 3)``.

    Returns
    -------
    torch.Tensor
        Sign per query in ``{-1, +1}`` (``+1`` outside, ``-1`` inside).
    """
    device = query.device
    dtype = query.dtype
    n_queries = query.shape[0]
    n_faces = face_vertices.shape[0]

    winding = torch.zeros(n_queries, dtype=dtype, device=device)
    if n_faces == 0 or n_queries == 0:
        return torch.ones(n_queries, dtype=dtype, device=device)

    # Tile faces so the (n_queries, tile) intermediates stay within budget.
    faces_per_tile = max(1, _WINDING_FACE_CHUNK // max(1, n_queries))
    for start in range(0, n_faces, faces_per_tile):
        end = min(start + faces_per_tile, n_faces)
        tri = face_vertices[start:end]  # (f, 3, 3)
        # Vectors from each query to each triangle vertex: (n_queries, f, 3, 3).
        a = tri[:, 0, :].unsqueeze(0) - query.unsqueeze(1)
        b = tri[:, 1, :].unsqueeze(0) - query.unsqueeze(1)
        c = tri[:, 2, :].unsqueeze(0) - query.unsqueeze(1)

        la = a.norm(dim=-1)
        lb = b.norm(dim=-1)
        lc = c.norm(dim=-1)

        # Numerator: triple product a . (b x c).
        triple = (a * torch.cross(b, c, dim=-1)).sum(-1)
        denom = (
            la * lb * lc
            + (a * b).sum(-1) * lc
            + (b * c).sum(-1) * la
            + (c * a).sum(-1) * lb
        )
        omega = 2.0 * torch.atan2(triple, denom)
        winding += omega.sum(dim=1)

    winding = winding / (4.0 * torch.pi)
    # Inside when winding number ~ 1 (use 0.5 threshold on |winding|).
    inside = winding.abs() > 0.5
    return torch.where(inside, -torch.ones(n_queries, dtype=dtype, device=device),
                       torch.ones(n_queries, dtype=dtype, device=device))


def signed_distance_field_mesh(
    mesh_vertices: Float[torch.Tensor, "n_vertices 3"],
    mesh_indices: Int[torch.Tensor, "..."],
    input_points: Float[torch.Tensor, "... 3"],
    max_dist: float = 1e8,
    use_sign_winding_number: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Signed distance field of a triangle mesh, computed without Warp.

    Drop-in replacement for
    :func:`physicsnemo.nn.functional.signed_distance_field` that uses
    :class:`physicsnemo.mesh.spatial.BVH` and plain PyTorch ops. The returned
    tuple matches the Warp implementation's contract.

    Parameters
    ----------
    mesh_vertices : torch.Tensor
        Mesh vertex coordinates, shape ``(n_vertices, 3)``.
    mesh_indices : torch.Tensor
        Triangle connectivity, flattened ``(3 * n_faces,)`` or ``(n_faces, 3)``.
    input_points : torch.Tensor
        Query points, shape ``(..., 3)``.
    max_dist : float, optional
        Maximum search distance for the nearest-triangle query. Default ``1e8``.
    use_sign_winding_number : bool, optional
        If ``True``, sign via the generalized winding number (robust for
        non-watertight meshes, ``O(n_queries * n_faces)``). If ``False``
        (default), sign via the nearest face's pseudo-normal. The mesh should be
        watertight for reliable signs in the ``False`` case.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(sdf, hit_points)``: signed distance per query (shape ``input.shape[:-1]``)
        and the closest point on the mesh per query (shape ``input.shape``).

    Raises
    ------
    ValueError
        If ``input_points`` does not have a trailing dimension of size 3, or if
        ``mesh_indices`` is not 1D-flattened or ``(n_faces, 3)``.
    """
    if input_points.shape[-1] != 3:
        raise ValueError("input_points must have last dimension of size 3")

    if mesh_indices.ndim == 2:
        if mesh_indices.shape[-1] != 3:
            raise ValueError(
                "mesh_indices with 2 dimensions must have shape (n_faces, 3)"
            )
    elif mesh_indices.ndim != 1:
        raise ValueError(
            "mesh_indices must be either 1D flattened indices or 2D (n_faces, 3)"
        )

    input_shape = input_points.shape
    out_dtype = input_points.dtype
    device = input_points.device

    # Compute internally in float32 for parity with the Warp kernel; the BVH
    # build path also assumes a float coordinate dtype.
    vertices = mesh_vertices.to(torch.float32)
    queries = input_points.reshape(-1, 3).to(torch.float32)
    n_queries = queries.shape[0]

    mesh, face_vertices, faces = _build_surface_mesh(vertices, mesh_indices)

    sdf = torch.zeros(n_queries, dtype=torch.float32, device=device)
    hit_points = queries.clone()

    if faces.shape[0] == 0 or n_queries == 0:
        sdf = sdf.reshape(input_shape[:-1]).to(out_dtype)
        hit_points = hit_points.reshape(input_shape).to(out_dtype)
        return sdf, hit_points

    with _timing.record("sdf/bvh_build"):
        bvh = BVH.from_mesh(mesh, leaf_size=_BVH_LEAF_SIZE)

    # Nearest triangle + closest point. On CUDA with Triton available we run the
    # single-kernel per-thread DFS (:func:`_sdf_triton.nearest_triangle_triton`),
    # which is the only way to get launch-overhead-free traversal. Otherwise we
    # fall back to the pure-PyTorch bounded-stack DFS (:func:`_nearest_face_bvh`),
    # which is also the parity oracle for the kernel. Both have peak memory
    # O(n_queries * tree_depth), independent of mesh size or query depth.
    with _timing.record("sdf/nearest"):
        if queries.is_cuda and _sdf_triton.available():
            _, best_face, best_point = _sdf_triton.nearest_triangle_triton(
                bvh, face_vertices, queries, max_dist, leaf_size=_BVH_LEAF_SIZE
            )
        else:
            # Queries are chunked so the per-iteration working tensors stay
            # modest for very large query sets.
            best_face = torch.zeros(n_queries, dtype=torch.long, device=device)
            best_point = queries.clone()
            for start in range(0, n_queries, _NEAREST_QUERY_CHUNK):
                end = min(start + _NEAREST_QUERY_CHUNK, n_queries)
                _, bf, bp = _nearest_face_bvh(
                    bvh, face_vertices, queries[start:end], max_dist
                )
                best_face[start:end] = bf
                best_point[start:end] = bp

    distance = (queries - best_point).norm(dim=-1)

    with _timing.record("sdf/sign"):
        if use_sign_winding_number:
            # Tree-accelerated winding number on CUDA+Triton (O(n_queries * log
            # n_faces)); the exact O(n_queries * n_faces) torch sum is the CPU /
            # no-Triton fallback and parity oracle.
            if queries.is_cuda and _sdf_triton.available():
                sign = _sdf_triton.winding_sign_triton(
                    bvh, face_vertices, queries, leaf_size=_BVH_LEAF_SIZE
                )
            else:
                sign = _winding_number_sign(face_vertices, queries)
        else:
            sign = _pseudo_normal_sign(mesh, queries, best_face, best_point)

    sdf = sign * distance
    hit_points = best_point

    sdf = sdf.reshape(input_shape[:-1]).to(out_dtype)
    hit_points = hit_points.reshape(input_shape).to(out_dtype)
    return sdf, hit_points
