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

"""Mesh-native signed distance field over a triangle surface mesh.

This signed distance field is built entirely on the spatial acceleration
structures of :mod:`physicsnemo.mesh.spatial`: the nearest-triangle
acceleration structure is a :class:`physicsnemo.mesh.spatial.BVH`, and
distances/signs are computed with plain PyTorch tensor ops. The winding-number
sign is computed with a :class:`physicsnemo.mesh.spatial.ClusterTree` Barnes-Hut
summation over the mesh, so the whole pipeline reuses the mesh's own spatial
data structures and runs identically on CPU and GPU.

:func:`signed_distance_field` returns the signed distance, the closest surface
point, and the nearest face index for each query (as a
:class:`SignedDistanceFieldResult`).

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
     at the closest feature (face / edge / vertex). Robust for watertight meshes.
   - ``use_sign_winding_number=True``: the generalized winding number (solid
     angle sum, Jacobson et al. 2013), evaluated with a
     :class:`physicsnemo.mesh.spatial.ClusterTree` dual-tree Barnes-Hut
     summation. Robust for non-watertight / self-intersecting meshes and scales
     to large meshes as ``O(n_queries * log n_faces)``. The exact
     ``O(n_queries * n_faces)`` sum (:func:`_winding_number_sign`) is retained
     as a reference oracle for testing.
"""

from __future__ import annotations

from typing import Literal, NamedTuple, TypeAlias, get_args

import torch
from jaxtyping import Float, Int
from tensordict import TensorDict
from torch.profiler import record_function

from physicsnemo.mesh.mesh import Mesh
from physicsnemo.mesh.spatial import _sdf_triton
from physicsnemo.mesh.spatial._ragged import _ragged_arange
from physicsnemo.mesh.spatial.bvh import BVH
from physicsnemo.mesh.spatial.cluster_tree import ClusterTree

# Winding-number summation backends for ``use_sign_winding_number=True``. The
# runtime validation tuple is derived from the typed ``Literal`` via
# ``get_args`` so the two can never drift apart.
WindingBackend: TypeAlias = Literal["clustertree", "bruteforce"]


class SignedDistanceFieldResult(NamedTuple):
    """Result of :func:`signed_distance_field`.

    A named tuple: positional unpacking works, and the fields are
    self-documenting at call sites that only need a subset.

    Attributes
    ----------
    sdf : torch.Tensor
        Signed distance per query, shape ``query_points.shape[:-1]`` (negative
        inside, positive outside). ``NaN`` for queries beyond a finite
        ``max_dist``.
    hit_points : torch.Tensor
        Closest point on the mesh per query, shape ``query_points.shape``.
        ``NaN`` for queries beyond a finite ``max_dist``.
    hit_faces : torch.Tensor
        Index into ``mesh.cells`` of the nearest face per query (int64), shape
        ``query_points.shape[:-1]``. ``-1`` for queries beyond a finite
        ``max_dist``.
    """

    sdf: torch.Tensor
    hit_points: torch.Tensor
    hit_faces: torch.Tensor


# Chunk sizes keep the pairwise tensors bounded for large inputs. These are
# product-of-counts limits (rows of the materialized intermediate), not raw
# point counts, so they translate directly to a peak-memory ceiling.
_NEAREST_QUERY_CHUNK = 1 << 18  # queries processed per BVH traversal batch
_WINDING_FACE_CHUNK = 1 << 22  # (query, face) pairs per winding-number tile

# Cells per BVH leaf. Single source of truth: passed to ``BVH.from_mesh`` and to
# the Triton kernels (as their static ``MAX_LEAF`` bound) so the GPU path never
# reads ``leaf_count.max()`` back to the host on the prefetch stream.
_BVH_LEAF_SIZE = 16

# ClusterTree winding-number summation parameters. ``_WINDING_THETA`` is the
# Barnes-Hut opening angle for the dual-tree traversal (smaller is more exact /
# slower; this value matches the accuracy regime of the previous bespoke kernel,
# whose opening factor ``beta=2.0`` corresponds to ``theta=0.5``).
# ``_WINDING_LEAF_SIZE`` controls how many primitives sit in a ClusterTree leaf.
_WINDING_THETA = 0.5
_WINDING_LEAF_SIZE = 8

# Relative-height threshold below which a face is treated as degenerate and
# repaired at build time (see ``_repair_degenerate_faces``): a triangle whose
# height is less than this fraction of its longest edge has its off-edge vertex
# displaced perpendicular to that edge by the same fraction of the edge length.
# Chosen ~1000x float32 epsilon so that float32 rounding of Ericson's region
# determinants can never misclassify the repaired face, while the repair moves
# the surface by at most this fraction of the edge length.
_DEGENERATE_TRI_REL_HEIGHT = 1e-4


def _repair_degenerate_faces(
    face_vertices: Float[torch.Tensor, "n_faces 3 3"],
) -> Float[torch.Tensor, "n_faces 3 3"]:
    r"""Displace the off-edge vertex of (near-)degenerate faces off their edge.

    Ericson's Voronoi-region closest-point classification
    (:func:`_closest_point_on_triangles` and its Triton mirror) assumes a
    non-degenerate triangle: on a (near-)zero-area face -- repeated vertices or
    collinear points, which real surface meshes do contain -- several region
    tests fire vacuously and the cascade can return the wrong feature, silently
    overestimating the distance. Rather than paying for a degenerate fallback
    on every ``(query, candidate)`` pair in the hot kernels (~20-25% measured),
    this repairs the geometry once per call: any face whose height is below
    :data:`_DEGENERATE_TRI_REL_HEIGHT` times its longest edge has its off-edge
    vertex moved to the edge midpoint plus a perpendicular offset ``h``, giving
    an equivalent thin-but-valid triangle over the same edge.

    ``h`` is ``max(rel * L, 8 * eps_f32 * max|coord|)`` with ``L`` the longest
    edge: the first term keeps the repaired face far above the float32 regime
    where the region determinants misclassify, and the second keeps the offset
    representable for small faces far from the origin (where ``rel * L`` would
    round away against the coordinate magnitude). The closest point, and hence
    the SDF and hit point, move by at most ``2 h`` -- ``h`` from the repair
    itself plus up to ``h`` of pruning slack where the repaired face protrudes
    from its BVH bounds (built from the original geometry) -- and only for
    queries whose nearest face was degenerate. Point-like faces (all vertices coincident,
    zero longest edge) are left untouched: every Ericson region returns the
    single point, so they are already handled exactly.

    Everything is a fixed-shape tensor pass over the faces -- no host
    readbacks, so the SDF prep stream stays sync-free.

    Parameters
    ----------
    face_vertices : torch.Tensor
        Per-face vertex positions, shape ``(n_faces, 3, 3)`` (float32).

    Returns
    -------
    torch.Tensor
        Repaired per-face vertex positions, shape ``(n_faces, 3, 3)``. Faces
        above the degeneracy threshold are bit-identical to the input.
    """
    a = face_vertices[:, 0, :]
    b = face_vertices[:, 1, :]
    c = face_vertices[:, 2, :]
    ab = b - a
    ac = c - a
    bc = c - b
    ab_sq = (ab * ab).sum(-1)
    ac_sq = (ac * ac).sum(-1)
    bc_sq = (bc * bc).sum(-1)

    # degenerate <=> height <= rel * longest edge <=> |ab x ac|^2 <= (rel * L^2)^2
    area_sq = (torch.linalg.cross(ab, ac, dim=-1) ** 2).sum(-1)
    scale_sq = torch.maximum(ab_sq, torch.maximum(ac_sq, bc_sq))
    degenerate = area_sq <= (_DEGENERATE_TRI_REL_HEIGHT * scale_sq) ** 2

    # The longest edge of a (near-)collinear face spans its extreme points, so
    # the face is (within its height) the segment (e0, e1); the remaining
    # "off-edge" vertex is the one displaced. 0 -> ab (off c), 1 -> ac (off b),
    # 2 -> bc (off a). Ties pick either longest edge; both are valid.
    longest = torch.stack([ab_sq, ac_sq, bc_sq], dim=-1).argmax(dim=-1)
    is_ab = (longest == 0).unsqueeze(-1)
    is_ac = (longest == 1).unsqueeze(-1)
    e0 = torch.where(is_ab | is_ac, a, b)
    e1 = torch.where(is_ab, b, c)

    # Unit perpendicular to the edge: cross against whichever of x-hat / y-hat
    # is less aligned with it (at least one of the two always works).
    edge = e1 - e0
    x_hat = torch.zeros_like(edge)
    x_hat[:, 0] = 1.0
    y_hat = torch.zeros_like(edge)
    y_hat[:, 1] = 1.0
    perp = torch.linalg.cross(edge, x_hat, dim=-1)
    perp_alt = torch.linalg.cross(edge, y_hat, dim=-1)
    edge_sq = (edge * edge).sum(-1)
    use_alt = (perp * perp).sum(-1) < 0.5 * edge_sq
    perp = torch.where(use_alt.unsqueeze(-1), perp_alt, perp)
    tiny = torch.finfo(face_vertices.dtype).tiny
    perp = perp / perp.norm(dim=-1, keepdim=True).clamp(min=tiny)

    eps = torch.finfo(face_vertices.dtype).eps
    coord_scale = face_vertices.abs().amax(dim=(1, 2))
    h = torch.maximum(
        _DEGENERATE_TRI_REL_HEIGHT * edge_sq.sqrt(), 8.0 * eps * coord_scale
    )
    # Point-like faces (zero longest edge) keep h = 0, i.e. stay untouched.
    h = torch.where(edge_sq > 0, h, torch.zeros_like(h))
    off_vertex = 0.5 * (e0 + e1) + perp * h.unsqueeze(-1)

    move_a = (degenerate & (longest == 2)).unsqueeze(-1)
    move_b = (degenerate & (longest == 1)).unsqueeze(-1)
    move_c = (degenerate & (longest == 0)).unsqueeze(-1)
    return torch.stack(
        [
            torch.where(move_a, off_vertex, a),
            torch.where(move_b, off_vertex, b),
            torch.where(move_c, off_vertex, c),
        ],
        dim=1,
    )


def _build_surface_mesh(
    mesh: Mesh,
) -> tuple[Mesh, Float[torch.Tensor, "n_faces 3 3"], Int[torch.Tensor, "n_faces 3"]]:
    """Normalize a triangle :class:`Mesh` into the float32 working representation.

    The BVH build and the Triton nearest-triangle kernel assume a float32
    coordinate dtype, so this returns a float32 copy of ``mesh`` alongside the
    per-face vertex positions and the int64 triangle connectivity consumed by
    the downstream tensor ops. (Near-)degenerate faces are repaired in the
    returned ``face_vertices`` (see :func:`_repair_degenerate_faces`) so the
    closest-point kernels never see a triangle their region classification
    cannot handle; ``work_mesh`` keeps the original vertices, since the
    pseudo-normal sign machinery consumes topology, not the repaired geometry.

    Parameters
    ----------
    mesh : Mesh
        Triangle surface mesh: ``mesh.points`` has shape ``(n_vertices, 3)`` and
        ``mesh.cells`` has shape ``(n_faces, 3)``.

    Returns
    -------
    tuple[Mesh, torch.Tensor, torch.Tensor]
        ``(mesh, face_vertices, faces)`` where ``mesh`` is the float32 working
        copy, ``face_vertices`` has shape ``(n_faces, 3, 3)`` and ``faces`` has
        shape ``(n_faces, 3)`` (int64).
    """
    faces = mesh.cells.to(torch.long)
    work_mesh = Mesh(points=mesh.points.to(torch.float32), cells=faces)
    face_vertices = _repair_degenerate_faces(work_mesh.points[faces])
    return work_mesh, face_vertices, faces


def _closest_point_on_triangles(
    query: Float[torch.Tensor, "n 3"],
    tri: Float[torch.Tensor, "n 3 3"],
) -> Float[torch.Tensor, "n 3"]:
    """Closest point on each triangle to its paired query point.

    Vectorized region-classification (Ericson, *Real-Time Collision
    Detection*). Computes, for each ``(query, triangle)`` pair, the point on the
    (closed) triangle nearest to ``query``.

    Ericson's Voronoi-region tests assume a non-degenerate triangle: on a
    (near-)zero-area face several region tests become vacuously true and the
    cascade can return the wrong feature (always an overestimate of the
    distance). The SDF pipeline therefore never feeds this routine a
    (near-)degenerate triangle: ``_build_surface_mesh`` repairs such faces up
    front (see :func:`_repair_degenerate_faces`). Point-like faces (all three
    vertices coincident) are the exception -- every region returns the single
    point, so they are correct here without repair.

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
        Search radius: triangles farther than this are ignored. Pass
        ``float("inf")`` for an unbounded, exact nearest search. A query with no
        triangle within ``max_dist`` returns ``best_dist_sq == max_dist ** 2``
        and ``best_point == query`` (an unchanged closest point), which the
        caller treats as a miss.

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


def _edge_pseudonormals(
    faces: Int[torch.Tensor, "n_faces 3"],
    face_normals: Float[torch.Tensor, "n_faces 3"],
) -> Float[torch.Tensor, "n_faces 3 3"]:
    r"""Per-face, per-edge pseudo-normals: the sum of the incident face normals.

    For each of a face's three edges, the edge pseudo-normal is
    :math:`\sum_f \mathbf{n}_f` over the faces sharing that edge (Baerentzen &
    Aanaes: the incident angle of an edge is :math:`\pi` for both faces, so the
    contributions are equal-weighted and reduce to a plain sum). The three
    undirected edges of every face are canonicalized (sorted endpoints) and
    de-duplicated so coincident edges accumulate together, then the result is
    gathered back into a ``(n_faces, 3, 3)`` table indexed by ``[face, local
    edge]`` with local edges ordered ``(v0, v1)``, ``(v1, v2)``, ``(v2, v0)``.

    Parameters
    ----------
    faces : torch.Tensor
        Triangle connectivity, shape :math:`(n_{faces}, 3)` (vertex indices).
    face_normals : torch.Tensor
        Outward unit face normals, shape :math:`(n_{faces}, 3)`.

    Returns
    -------
    torch.Tensor
        Edge pseudo-normals, shape :math:`(n_{faces}, 3, 3)`.
    """
    n_faces = faces.shape[0]
    v0, v1, v2 = faces[:, 0], faces[:, 1], faces[:, 2]
    # Three undirected edges per face in fixed local order, flattened
    # face-major: (f0,e0), (f0,e1), (f0,e2), (f1,e0), ...
    edges = torch.stack(
        [
            torch.stack([v0, v1], dim=1),  # local edge 0
            torch.stack([v1, v2], dim=1),  # local edge 1
            torch.stack([v2, v0], dim=1),  # local edge 2
        ],
        dim=1,
    ).reshape(-1, 2)
    edges, _ = torch.sort(edges, dim=1)  # canonical (lo, hi) per edge

    # Group coincident edges WITHOUT ``torch.unique(edges, dim=0)``: that call
    # reads the unique-row count back to the host to size its output, forcing a
    # D2H sync that stalls the SDF prep stream (it can no longer overlap the
    # compute stream). Instead, lexicographically sort the canonical ``(lo, hi)``
    # rows -- two stable ``argsort``s on int64 columns, both sync-free because
    # their output size is the static ``m`` -- so identical edges become
    # adjacent. Contiguous group ids then follow from a cumsum over a
    # "row changed" mask. No host readback anywhere on the path.
    m = edges.shape[0]
    lo, hi = edges[:, 0], edges[:, 1]
    order_hi = torch.argsort(hi, stable=True)
    order = order_hi[torch.argsort(lo[order_hi], stable=True)]  # lexsort by (lo, hi)
    lo_s, hi_s = lo[order], hi[order]
    is_new = torch.ones(m, dtype=torch.bool, device=edges.device)
    is_new[1:] = (lo_s[1:] != lo_s[:-1]) | (hi_s[1:] != hi_s[:-1])
    group = (
        torch.cumsum(is_new.to(torch.long), 0) - 1
    )  # contiguous edge id per sorted row

    # Each face donates its normal to all three of its edges (same face-major
    # order as ``edges``). Accumulate per group into a buffer sized to the static
    # upper bound ``m`` (>= number of unique edges; unused tail rows stay zero),
    # gather each instance's group sum, then scatter back to the original
    # (unsorted) edge order so the table lines up with ``[face, local edge]``.
    fn_per_edge = face_normals.repeat_interleave(3, dim=0)  # (3 n_faces, 3)
    edge_accum = torch.zeros(m, 3, dtype=face_normals.dtype, device=face_normals.device)
    edge_accum.index_add_(0, group, fn_per_edge[order])
    pseudo = torch.empty_like(fn_per_edge)
    pseudo[order] = edge_accum[group]  # each instance -> sum over its coincident edges
    return pseudo.reshape(n_faces, 3, 3)


def _pseudo_normal_sign(
    mesh: Mesh,
    query: Float[torch.Tensor, "n_queries 3"],
    best_face: Int[torch.Tensor, " n_queries"],
    best_point: Float[torch.Tensor, "n_queries 3"],
) -> Float[torch.Tensor, " n_queries"]:
    r"""Sign of the SDF via the angle-weighted pseudo-normal of the hit feature.

    The sign is :math:`\mathrm{sign}((\mathbf{q} - \mathbf{p}) \cdot \mathbf{N})`,
    where :math:`\mathbf{p}` is the closest surface point and :math:`\mathbf{N}`
    is the *angle-weighted pseudo-normal* of the mesh feature that :math:`p` lies
    on -- not merely the nearest face normal. Picking a single face normal is
    wrong whenever :math:`p` lands on an edge or vertex shared by several faces
    (sharp or non-convex geometry): the query can sit behind that one face's
    half-plane while still being outside the solid, which flips the sign (and,
    near edges, corrupts the signed magnitude). Resolving the feature removes the
    ambiguity:

    - **face interior** -> the face normal :math:`\mathbf{n}_f`;
    - **edge** -> the sum of the normals of the faces sharing it,
      :math:`\sum_f \mathbf{n}_f` (see :func:`_edge_pseudonormals`);
    - **vertex** :math:`v` -> the incident-angle-weighted sum
      :math:`\sum_f \alpha_f(v)\,\mathbf{n}_f`, via
      :meth:`~physicsnemo.mesh.Mesh.compute_point_normals` with ``"angle"``
      weighting.

    Only the *direction* of :math:`\mathbf{N}` affects the sign, so the
    per-vertex normalization applied by ``compute_point_normals`` is harmless.
    The feature is classified from the barycentric region of :math:`p` on the
    hit triangle (Ericson, *Real-Time Collision Detection*). This is robust on
    watertight meshes; for non-watertight surfaces use the winding-number sign.

    Parameters
    ----------
    mesh : Mesh
        Triangle surface mesh (provides ``cells``, ``points``, cached
        ``cell_normals``, and angle-weighted ``compute_point_normals``).
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

    References
    ----------
    J. A. Baerentzen and H. Aanaes, "Signed Distance Computation Using the Angle
    Weighted Pseudonormal", IEEE TVCG, 2005.
    """
    dtype = query.dtype
    faces = mesh.cells  # (n_faces, 3) vertex indices
    vertices = mesh.points.to(dtype)  # (n_vertices, 3)
    face_normals = mesh.cell_normals.to(dtype)  # (n_faces, 3), unit, outward

    # Precompute the feature pseudo-normals (once per call). Vertex normals are
    # angle-weighted; edge normals are the sum of the incident face normals.
    vertex_pn = mesh.compute_point_normals(weighting="angle").to(dtype)
    face_edge_pn = _edge_pseudonormals(faces, face_normals)  # (n_faces, 3, 3)

    # Classify which feature of the hit triangle ``best_point`` lies on, using
    # the same barycentric region test as the closest-point routine. ``a, b, c``
    # are the hit triangle's vertices (vertex axis 1); d1..d6 / va,vb,vc are the
    # edge-projection and region-determinant quantities (Ericson).
    hit_faces = faces[best_face]  # (n_queries, 3) vertex indices
    tri = vertices[hit_faces]  # (n_queries, 3, 3)
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
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

    region_a = (d1 <= 0) & (d2 <= 0)  # vertex v0
    region_b = (d3 >= 0) & (d4 <= d3)  # vertex v1
    region_c = (d6 >= 0) & (d5 <= d6)  # vertex v2
    region_ab = (vc <= 0) & (d1 >= 0) & (d3 <= 0)  # edge (v0, v1)
    region_bc = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)  # edge (v1, v2)
    region_ca = (vb <= 0) & (d2 >= 0) & (d6 <= 0)  # edge (v2, v0)

    # Select the pseudo-normal per query. Default to the face interior, override
    # with edge normals, then with vertex normals last so vertices win at
    # corners (matching Ericson's vertex-first region precedence).
    pn = face_normals[best_face]
    edge_pn = face_edge_pn[best_face]  # (n_queries, 3, 3): edges (ab, bc, ca)
    pn = torch.where(region_ab.unsqueeze(-1), edge_pn[:, 0], pn)
    pn = torch.where(region_bc.unsqueeze(-1), edge_pn[:, 1], pn)
    pn = torch.where(region_ca.unsqueeze(-1), edge_pn[:, 2], pn)
    vert_pn = vertex_pn[hit_faces]  # (n_queries, 3, 3): vertices (a, b, c)
    pn = torch.where(region_a.unsqueeze(-1), vert_pn[:, 0], pn)
    pn = torch.where(region_b.unsqueeze(-1), vert_pn[:, 1], pn)
    pn = torch.where(region_c.unsqueeze(-1), vert_pn[:, 2], pn)

    direction = query - best_point
    dot = (direction * pn).sum(-1)
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
        triple = (a * torch.linalg.cross(b, c, dim=-1)).sum(-1)
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
    return torch.where(
        inside,
        -torch.ones(n_queries, dtype=dtype, device=device),
        torch.ones(n_queries, dtype=dtype, device=device),
    )


def _triangle_solid_angles(
    query: Float[torch.Tensor, "n 3"],
    tri: Float[torch.Tensor, "n 3 3"],
) -> Float[torch.Tensor, " n"]:
    r"""Signed solid angle subtended by each triangle at its paired query point.

    Implements the Van Oosterom-Strackee formula

    .. math::

        \Omega = 2 \, \mathrm{atan2}\!\left(
            \mathbf{a}\cdot(\mathbf{b}\times\mathbf{c}),\;
            |\mathbf{a}||\mathbf{b}||\mathbf{c}|
            + (\mathbf{a}\cdot\mathbf{b})|\mathbf{c}|
            + (\mathbf{b}\cdot\mathbf{c})|\mathbf{a}|
            + (\mathbf{c}\cdot\mathbf{a})|\mathbf{b}|
        \right)

    where :math:`\mathbf{a}, \mathbf{b}, \mathbf{c}` are the vectors from the
    query point to the three triangle vertices. This is the exact per-triangle
    term summed by :func:`_winding_number_sign`; here it is evaluated for paired
    ``(query, triangle)`` rows produced by the dual-tree traversal.

    Parameters
    ----------
    query : torch.Tensor
        Query points, shape ``(n, 3)``.
    tri : torch.Tensor
        Triangle vertices, shape ``(n, 3, 3)`` (vertex axis is dim 1).

    Returns
    -------
    torch.Tensor
        Signed solid angle per pair, shape ``(n,)``.
    """
    a = tri[:, 0, :] - query
    b = tri[:, 1, :] - query
    c = tri[:, 2, :] - query

    la = a.norm(dim=-1)
    lb = b.norm(dim=-1)
    lc = c.norm(dim=-1)

    triple = (a * torch.linalg.cross(b, c, dim=-1)).sum(-1)
    denom = (
        la * lb * lc
        + (a * b).sum(-1) * lc
        + (b * c).sum(-1) * la
        + (c * a).sum(-1) * lb
    )
    return 2.0 * torch.atan2(triple, denom)


def _winding_number_sign_clustertree(
    face_vertices: Float[torch.Tensor, "n_faces 3 3"],
    query: Float[torch.Tensor, "n_queries 3"],
    *,
    theta: float = _WINDING_THETA,
    leaf_size: int = _WINDING_LEAF_SIZE,
) -> Float[torch.Tensor, " n_queries"]:
    r"""Sign of the SDF via a :class:`ClusterTree` Barnes-Hut winding number.

    Computes the generalized winding number (Jacobson et al., "Robust
    Inside-Outside Segmentation using Generalized Winding Numbers", 2013) as a
    dual-tree Barnes-Hut summation, reusing the mesh's own
    :class:`physicsnemo.mesh.spatial.ClusterTree`. This is robust on
    non-watertight / soup geometry and scales as ``O(n_queries * log n_faces)``
    instead of the exact ``O(n_queries * n_faces)`` sum in
    :func:`_winding_number_sign`, and -- unlike a hand-written CUDA/Triton
    kernel -- runs identically on CPU and GPU because the tree is plain PyTorch.

    The summation builds a source tree over the triangle (area-weighted)
    centroids and a target tree over the query points, then evaluates the
    dual-tree interaction plan
    (:meth:`ClusterTree.find_dual_interaction_pairs`) with
    ``expand_far_targets=True`` so the dominant far-field stream is expanded to
    individual query points (no target-side approximation). Each interaction
    stream contributes to the per-query winding number:

    - **(near, near)**: the exact triangle solid angle, evaluated at the real
      query point.
    - **(near, far)**: the source node's dipole (sum of area-weighted normals)
      evaluated at the real query point, ``N_node . (p_node - q) / |p_node -
      q|^3``.
    - **(far, near)**: the exact triangle solid angle evaluated at the target
      node centroid, broadcast to the survivor query points in that node.

    Parameters
    ----------
    face_vertices : torch.Tensor
        Per-face vertex positions, shape ``(n_faces, 3, 3)``.
    query : torch.Tensor
        Query points, shape ``(n_queries, 3)``.
    theta : float, optional
        Barnes-Hut opening angle for the dual-tree traversal. Smaller is more
        exact and slower. Default :data:`_WINDING_THETA`.
    leaf_size : int, optional
        Maximum primitives per :class:`ClusterTree` leaf. Default
        :data:`_WINDING_LEAF_SIZE`.

    Returns
    -------
    torch.Tensor
        Sign per query in ``{-1, +1}`` (``+1`` outside, ``-1`` inside),
        shape ``(n_queries,)``.

    Notes
    -----
    See :func:`signed_distance_field` for the end-to-end SDF that consumes
    this sign.
    """
    device = query.device
    dtype = query.dtype
    n_queries = query.shape[0]
    n_faces = face_vertices.shape[0]

    if n_faces == 0 or n_queries == 0:
        return torch.ones(n_queries, dtype=dtype, device=device)

    # Per-face geometry: area-weighted normal (the solid-angle dipole moment),
    # scalar area, unit normal, and centroid (the source points of the tree).
    a = face_vertices[:, 0, :]
    b = face_vertices[:, 1, :]
    c = face_vertices[:, 2, :]
    area_normal = 0.5 * torch.linalg.cross(b - a, c - a, dim=-1)  # (F, 3)
    area = area_normal.norm(dim=-1)  # (F,)
    tiny = torch.finfo(torch.float32).tiny
    unit_normal = area_normal / area.clamp(min=tiny).unsqueeze(-1)  # (F, 3)
    centroid = (a + b + c) / 3.0  # (F, 3)

    # Build the source tree over triangle centroids (area-weighted) and the
    # target tree over the query points.
    source_tree = ClusterTree.from_points(centroid, leaf_size=leaf_size, areas=area)
    target_tree = ClusterTree.from_points(query, leaf_size=leaf_size)

    # Per-node dipole moment ``N_node = sum_i area_i * unit_normal_i`` and
    # expansion center ``p_node`` (area-weighted centroid). ``compute_source_
    # aggregates`` returns the area-weighted *average* normal, so multiplying by
    # the node's total area recovers the *sum* of area-weighted normals.
    source_data = TensorDict(
        {"normal": unit_normal}, batch_size=[n_faces], device=device
    )
    aggregates = source_tree.compute_source_aggregates(
        source_points=centroid, areas=area, source_data=source_data
    )
    node_dipole = aggregates.node_source_data["normal"] * (
        source_tree.node_total_area.unsqueeze(-1)
    )  # (n_nodes, 3)
    node_center = aggregates.node_centroid  # (n_nodes, 3)

    # ``expand_far_targets=True`` converts the (far, far) stream into per-target
    # (near, far) entries, eliminating the target-side blocky approximation for
    # the dominant far-field term while keeping the source-side monopole.
    plan = source_tree.find_dual_interaction_pairs(
        target_tree=target_tree, theta=theta, expand_far_targets=True
    )

    # Target node centroids are only needed by the (far, near) stream, where the
    # source triangle's solid angle is evaluated at the target node's centroid
    # and broadcast to its survivor queries.
    target_centroid = target_tree.compute_source_aggregates(
        source_points=query,
        areas=torch.ones(n_queries, dtype=query.dtype, device=device),
        source_data=None,
    ).node_centroid  # (n_target_nodes, 3)

    winding = torch.zeros(n_queries, dtype=dtype, device=device)

    # (near, near): exact triangle solid angle at the real query point.
    if plan.n_near > 0:
        nn_q = plan.near_target_ids
        nn_s = plan.near_source_ids
        omega = _triangle_solid_angles(query[nn_q], face_vertices[nn_s])
        winding.scatter_add_(0, nn_q, omega.to(dtype))

    # (near, far): source-node dipole evaluated at the real query point.
    if plan.n_nf > 0:
        nf_q = plan.nf_target_ids
        nf_n = plan.nf_source_node_ids
        e = node_center[nf_n] - query[nf_q]  # (n_nf, 3)
        r2 = (e * e).sum(-1)
        r3 = (r2 * r2.sqrt()).clamp(min=tiny)
        dip = (node_dipole[nf_n] * e).sum(-1) / r3
        winding.scatter_add_(0, nf_q, dip.to(dtype))

    # (far, near): exact triangle solid angle evaluated at the target-node
    # centroid, broadcast to the survivor query points in that node.
    if plan.n_fn > 0:
        fn_n = plan.fn_target_node_ids
        fn_s = plan.fn_source_ids
        omega_fn = _triangle_solid_angles(target_centroid[fn_n], face_vertices[fn_s])
        positions, pair_ids = _ragged_arange(
            plan.fn_broadcast_starts, plan.fn_broadcast_counts
        )
        expanded_tgt = plan.fn_broadcast_targets[positions]
        winding.scatter_add_(0, expanded_tgt, omega_fn[pair_ids].to(dtype))

    winding = winding / (4.0 * torch.pi)
    inside = winding.abs() > 0.5
    return torch.where(
        inside,
        -torch.ones(n_queries, dtype=dtype, device=device),
        torch.ones(n_queries, dtype=dtype, device=device),
    )


def signed_distance_field(
    mesh: Mesh,
    query_points: Float[torch.Tensor, "... 3"],
    max_dist: float | None = None,
    use_sign_winding_number: bool = False,
    *,
    winding_backend: WindingBackend = "clustertree",
) -> SignedDistanceFieldResult:
    r"""Compute the signed distance to a triangle surface mesh.

    Returns the signed distance, the closest surface point, and the nearest
    face index for each query.

    Parameters
    ----------
    mesh : Mesh
        Triangle surface mesh embedded in 3D: ``mesh.points`` has shape
        ``(n_vertices, 3)`` and ``mesh.cells`` has shape ``(n_faces, 3)``.
    query_points : torch.Tensor
        Query points, shape ``(..., 3)``.
    max_dist : float or None, optional
        Maximum search radius for the nearest-triangle query. ``None``
        (default) searches without bound, so the true nearest triangle is
        always found; a finite value restricts the search to a band and
        reports queries beyond it as ``NaN`` (both ``sdf`` and ``hit_points``).
    use_sign_winding_number : bool, optional
        If ``True``, sign via the generalized winding number (robust for
        non-watertight meshes). If ``False`` (default), sign via the
        angle-weighted pseudo-normal of the closest mesh feature (face, edge, or
        vertex), which stays correct at sharp/non-convex edges where a single
        face normal would flip the sign (see :func:`_pseudo_normal_sign`). The
        mesh should be watertight for reliable signs in the ``False`` case.
    winding_backend : {"clustertree", "bruteforce"}, optional
        Winding-number summation backend when ``use_sign_winding_number=True``:
        ``"clustertree"`` (default) for the
        :class:`physicsnemo.mesh.spatial.ClusterTree` Barnes-Hut sum
        (``O(n_queries * log n_faces)``, best for large meshes), or
        ``"bruteforce"`` for the exact fused ``O(n_queries * n_faces)`` sum
        (faster for small/medium meshes).

    Returns
    -------
    SignedDistanceFieldResult
        Named tuple ``(sdf, hit_points, hit_faces)``: signed distance per query
        (shape ``query_points.shape[:-1]``), the closest point on the mesh per
        query (shape ``query_points.shape``), and the index into ``mesh.cells``
        of the nearest face per query (int64, shape
        ``query_points.shape[:-1]``; ``-1`` for queries beyond a finite
        ``max_dist``).

    Raises
    ------
    ValueError
        If ``mesh`` is not a triangle surface in 3D (``n_spatial_dims == 3`` and
        ``n_manifold_dims == 2``), if ``query_points`` does not have a trailing
        dimension of size 3, if the mesh has no faces (there is no surface to
        measure distance to), if ``mesh`` and ``query_points`` are on different
        devices, if ``winding_backend`` is not a supported backend, or if
        ``max_dist`` is negative.

    Notes
    -----
    A finite ``max_dist`` is an opt-in optimization/narrow-band mode: it prunes
    the search to the given radius and marks out-of-band queries as ``NaN`` so a
    far query is never silently reported as on-surface (``sdf == 0``). The
    unbounded default never produces ``NaN`` for a non-empty mesh.
    """
    if query_points.shape[-1] != 3:
        raise ValueError("query_points must have last dimension of size 3")

    # A triangle surface in 3D is required: the closest-point and solid-angle
    # math both assume 3-vertex cells with 3D coordinates. Validate here so a
    # mis-typed mesh fails loudly rather than deep inside the BVH/winding kernels.
    if mesh.n_spatial_dims != 3:
        raise ValueError(
            "signed_distance_field requires a 3D mesh "
            f"(n_spatial_dims == 3), but got {mesh.n_spatial_dims=}."
        )
    if mesh.n_manifold_dims != 2:
        raise ValueError(
            "signed_distance_field requires a triangle mesh "
            f"(n_manifold_dims == 2), but got {mesh.n_manifold_dims=}."
        )
    if mesh.n_cells == 0:
        raise ValueError(
            "mesh has no faces; there is no surface to measure distance to"
        )
    if mesh.points.device != query_points.device:
        raise ValueError(
            "mesh and query_points must be on the same device, but got "
            f"{mesh.points.device=} and {query_points.device=}."
        )
    if winding_backend not in get_args(WindingBackend):
        raise ValueError(
            f"winding_backend must be one of {get_args(WindingBackend)}, "
            f"got {winding_backend!r}"
        )
    if max_dist is not None and max_dist < 0:
        raise ValueError(f"max_dist must be None or non-negative, got {max_dist}")

    query_shape = query_points.shape
    out_dtype = query_points.dtype
    device = query_points.device

    queries = query_points.reshape(-1, 3).to(torch.float32)
    n_queries = queries.shape[0]

    # None -> unbounded exact search; a finite value is a narrow band.
    max_dist_eff = float("inf") if max_dist is None else float(max_dist)

    # Normalize the mesh to a float32 working copy; the BVH build and the Triton
    # nearest-triangle kernel assume a float32 coordinate dtype.
    work_mesh, face_vertices, _ = _build_surface_mesh(mesh)

    sdf = torch.zeros(n_queries, dtype=torch.float32, device=device)
    hit_points = queries.clone()

    if n_queries == 0:
        return SignedDistanceFieldResult(
            sdf=sdf.reshape(query_shape[:-1]).to(out_dtype),
            hit_points=hit_points.reshape(query_shape).to(out_dtype),
            # Always empty here; -1 keeps the "no valid face" sentinel uniform.
            hit_faces=torch.full(query_shape[:-1], -1, dtype=torch.long, device=device),
        )

    with record_function("sdf/bvh_build"):
        bvh = BVH.from_mesh(work_mesh, leaf_size=_BVH_LEAF_SIZE)

    # Nearest triangle + closest point. On CUDA with Triton available we run the
    # single-kernel per-thread DFS (:func:`_sdf_triton.nearest_triangle_triton`),
    # which is the only way to get launch-overhead-free traversal. Otherwise we
    # fall back to the pure-PyTorch bounded-stack DFS (:func:`_nearest_face_bvh`),
    # which is also the parity oracle for the kernel. Both have peak memory
    # O(n_queries * tree_depth), independent of mesh size or query depth.
    with record_function("sdf/nearest"):
        if queries.is_cuda and _sdf_triton.available():
            best_dist_sq, best_face, best_point = _sdf_triton.nearest_triangle_triton(
                bvh, face_vertices, queries, max_dist_eff, leaf_size=_BVH_LEAF_SIZE
            )
        else:
            # Queries are chunked so the per-iteration working tensors stay
            # modest for very large query sets.
            best_dist_sq = torch.empty(n_queries, dtype=torch.float32, device=device)
            best_face = torch.zeros(n_queries, dtype=torch.long, device=device)
            best_point = queries.clone()
            for start in range(0, n_queries, _NEAREST_QUERY_CHUNK):
                end = min(start + _NEAREST_QUERY_CHUNK, n_queries)
                bd, bf, bp = _nearest_face_bvh(
                    bvh, face_vertices, queries[start:end], max_dist_eff
                )
                best_dist_sq[start:end] = bd
                best_face[start:end] = bf
                best_point[start:end] = bp

    distance = (queries - best_point).norm(dim=-1)

    with record_function("sdf/sign"):
        if use_sign_winding_number:
            # Generalized winding number. The ClusterTree Barnes-Hut summation
            # (O(n_queries * log n_faces)) wins on large meshes; the exact fused
            # O(n_queries * n_faces) sum is faster for small/medium meshes where
            # the tree cannot prune. Both run on CPU and GPU and are equivalent
            # up to the Barnes-Hut approximation.
            if winding_backend == "bruteforce":
                sign = _winding_number_sign(face_vertices, queries)
            else:
                sign = _winding_number_sign_clustertree(face_vertices, queries)
        else:
            sign = _pseudo_normal_sign(work_mesh, queries, best_face, best_point)

    sdf = sign * distance
    hit_points = best_point

    if max_dist is not None:
        # Out-of-band queries keep the initial bound; flag them NaN, not 0
        # (and -1 for the face index, which has no NaN).
        missed = best_dist_sq >= max_dist_eff**2
        sdf = torch.where(missed, sdf.new_full((), float("nan")), sdf)
        hit_points = torch.where(
            missed.unsqueeze(-1), hit_points.new_full((), float("nan")), hit_points
        )
        best_face = torch.where(missed, best_face.new_full((), -1), best_face)

    return SignedDistanceFieldResult(
        sdf=sdf.reshape(query_shape[:-1]).to(out_dtype),
        hit_points=hit_points.reshape(query_shape).to(out_dtype),
        hit_faces=best_face.reshape(query_shape[:-1]),
    )


def _signed_distance_field_from_arrays(
    mesh_vertices: Float[torch.Tensor, "n_vertices 3"],
    mesh_indices: Int[torch.Tensor, "..."],
    query_points: Float[torch.Tensor, "... 3"],
    max_dist: float | None = None,
    use_sign_winding_number: bool = False,
    *,
    winding_backend: WindingBackend = "clustertree",
) -> SignedDistanceFieldResult:
    r"""[INTERNAL - DO NOT USE] Private array-based SDF helper.

    .. warning::

       **DON'T USE THIS ONE.** This is a private, temporary entry point. Use
       the public :func:`signed_distance_field`, which takes a
       :class:`~physicsnemo.mesh.Mesh`, instead.

       This helper is unexported, carries no backward-compatibility guarantee,
       and may be removed without notice.

    It wraps the arrays in a :class:`~physicsnemo.mesh.Mesh` and defers to
    :func:`signed_distance_field`, so the numerics are identical.

    Parameters
    ----------
    mesh_vertices : torch.Tensor
        Mesh vertex coordinates, shape ``(n_vertices, 3)``.
    mesh_indices : torch.Tensor
        Triangle connectivity, flattened ``(3 * n_faces,)`` or ``(n_faces, 3)``.
    query_points : torch.Tensor
        Query points, shape ``(..., 3)``.
    max_dist : float or None, optional
        Maximum search radius; see :func:`signed_distance_field`.
    use_sign_winding_number : bool, optional
        Sign method; see :func:`signed_distance_field`.
    winding_backend : {"clustertree", "bruteforce"}, optional
        Winding-number backend; see :func:`signed_distance_field`.

    Returns
    -------
    SignedDistanceFieldResult
        ``(sdf, hit_points, hit_faces)``; see :func:`signed_distance_field`.

    Raises
    ------
    ValueError
        If ``mesh_indices`` is not 1D-flattened or ``(n_faces, 3)``; the
        remaining validation is performed by :func:`signed_distance_field`.
    """
    if mesh_indices.ndim == 2:
        if mesh_indices.shape[-1] != 3:
            raise ValueError(
                "mesh_indices with 2 dimensions must have shape (n_faces, 3)"
            )
    elif mesh_indices.ndim != 1:
        raise ValueError(
            "mesh_indices must be either 1D flattened indices or 2D (n_faces, 3)"
        )
    mesh = Mesh(points=mesh_vertices, cells=mesh_indices.reshape(-1, 3))
    return signed_distance_field(
        mesh,
        query_points,
        max_dist,
        use_sign_winding_number,
        winding_backend=winding_backend,
    )
