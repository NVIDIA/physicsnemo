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

"""Signed distance field over a triangle surface mesh.

This is a :class:`~physicsnemo.mesh.Mesh`-typed wrapper around the Warp-backed
:func:`physicsnemo.nn.functional.signed_distance_field` custom op, which runs
NVIDIA Warp mesh queries (``wp.mesh_query_point_sign_normal`` /
``wp.mesh_query_point_sign_winding_number``) on CPU and CUDA. Warp launches are
made stream- and allocator-safe by
:meth:`physicsnemo.core.function_spec.FunctionSpec.warp_launch_context`, so the
op composes with torch streams (e.g. datapipe prefetch) without host
synchronization.

:func:`signed_distance_field` returns the signed distance, the closest surface
point, and the nearest face index for each query.
"""

from __future__ import annotations

import torch
from jaxtyping import Float, Int

from physicsnemo.mesh.mesh import Mesh
from physicsnemo.nn.functional.geometry.sdf import (
    signed_distance_field as _warp_signed_distance_field,
)

# Relative-height threshold below which a face is treated as degenerate and
# repaired at call time (see ``_repair_degenerate_faces``): a triangle whose
# height is less than this fraction of its longest edge has its off-edge vertex
# displaced perpendicular to that edge by the same fraction of the edge length.
_DEGENERATE_TRI_REL_HEIGHT = 1e-4


def _repair_degenerate_faces(
    face_vertices: Float[torch.Tensor, "n_faces 3 3"],
) -> Float[torch.Tensor, "n_faces 3 3"]:
    r"""Displace the off-edge vertex of (near-)degenerate faces off their edge.

    Warp's mesh closest-point query skips (near-)zero-area triangles --
    repeated vertices or collinear points, which real surface meshes do
    contain -- so a query nearest to such a face would silently report the
    distance to some farther valid face instead. This repairs the geometry
    once per call: any face whose height is below
    :data:`_DEGENERATE_TRI_REL_HEIGHT` times its longest edge has its off-edge
    vertex moved to the edge midpoint plus a perpendicular offset ``h``, giving
    an equivalent thin-but-valid triangle over the same edge.

    ``h`` is ``max(rel * L, 8 * eps_f32 * max|coord|)`` with ``L`` the longest
    edge: the first term keeps the repaired face far above the float32 regime
    where closest-point arithmetic misclassifies, and the second keeps the
    offset representable for small faces far from the origin (where ``rel * L``
    would round away against the coordinate magnitude). The closest point, and
    hence the SDF and hit point, move by at most ``h`` -- and only for queries
    whose nearest face was degenerate. Point-like faces (all vertices
    coincident, zero longest edge) are left untouched.

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


def _splice_repaired_faces(
    points: Float[torch.Tensor, "n_vertices 3"],
    cells: Int[torch.Tensor, "n_faces 3"],
) -> tuple[Float[torch.Tensor, "n_vertices_out 3"], Int[torch.Tensor, "n_faces 3"]]:
    """Splice repaired degenerate faces into the mesh, preserving welded topology.

    Applies :func:`_repair_degenerate_faces` and rewires only the faces it
    changed: each face's (single) moved vertex is appended as a new private
    vertex and that face's cell slot is remapped to it. Valid faces keep their
    original, shared vertices bit-identical, so Warp's angle-weighted
    pseudo-normal sign (which consumes the welded topology) is unaffected away
    from the repaired faces. Appended vertices of untouched faces are simply
    never referenced.

    Fixed-shape tensor ops throughout -- no host readbacks, so the repair is
    safe on a sync-free prefetch stream.
    """
    n_vertices = points.shape[0]
    n_faces = cells.shape[0]

    face_vertices = points[cells]
    repaired = _repair_degenerate_faces(face_vertices)

    # At most one vertex slot moves per face; argmax picks it (slot 0 for
    # untouched faces, where the remap below is a no-op).
    moved = (repaired != face_vertices).any(dim=-1)  # (n_faces, 3)
    face_ids = torch.arange(n_faces, device=cells.device)
    slot = moved.long().argmax(dim=1)
    new_vertices = repaired[face_ids, slot]  # (n_faces, 3)

    new_cells = cells.clone()
    new_cells[face_ids, slot] = torch.where(
        moved.any(dim=1), n_vertices + face_ids, cells[face_ids, slot]
    )
    return torch.cat([points, new_vertices]), new_cells


def signed_distance_field(
    mesh: Mesh,
    query_points: Float[torch.Tensor, "... 3"],
    max_dist: float | None = None,
    use_sign_winding_number: bool = False,
) -> tuple[
    Float[torch.Tensor, "..."],
    Float[torch.Tensor, "... 3"],
    Int[torch.Tensor, "..."],
]:
    r"""Compute the signed distance to a triangle surface mesh.

    Returns the signed distance, the closest surface point, and the nearest
    face index for each query. Delegates to the Warp-backed
    :func:`physicsnemo.nn.functional.signed_distance_field` op, which runs on
    CPU and CUDA.

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
        reports queries farther than it as ``NaN`` (both ``sdf`` and
        ``hit_points``) with a hit face of ``-1``.
    use_sign_winding_number : bool, optional
        If ``True``, sign via the generalized winding number
        (``wp.mesh_query_point_sign_winding_number``), robust for
        non-watertight meshes. If ``False`` (default), sign via the
        angle-weighted pseudo-normal of the closest mesh feature
        (``wp.mesh_query_point_sign_normal``), which stays correct at
        sharp/non-convex edges where a single face normal would flip the sign.
        The mesh should be watertight for reliable signs in the ``False`` case.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ``(sdf, hit_points, hit_faces)``: signed distance per query
        (shape ``query_points.shape[:-1]``; negative inside, positive
        outside), the closest point on the mesh per query (shape
        ``query_points.shape``), and the index into ``mesh.cells`` of the
        nearest face per query (int64, shape ``query_points.shape[:-1]``).
        Queries beyond a finite ``max_dist`` return ``NaN`` for the distance
        and hit point and ``-1`` for the hit face.

    Raises
    ------
    ValueError
        If ``mesh`` is not a triangle surface in 3D (``n_spatial_dims == 3`` and
        ``n_manifold_dims == 2``), if ``query_points`` does not have a trailing
        dimension of size 3, if the mesh has no faces (there is no surface to
        measure distance to), if ``mesh`` and ``query_points`` are on different
        devices, or if ``max_dist`` is negative.

    Notes
    -----
    A finite ``max_dist`` is an opt-in optimization/narrow-band mode: it prunes
    the search to the given radius and marks out-of-band queries as ``NaN`` so a
    far query is never silently reported as on-surface (``sdf == 0``). The
    unbounded default never produces ``NaN`` for a non-empty mesh.

    Distances are computed in float32 (inputs are cast as needed) and results
    are cast back to the ``query_points`` dtype.

    (Near-)degenerate faces -- repeated vertices or collinear points, which
    Warp's mesh query would otherwise skip -- are repaired into equivalent
    thin-but-valid triangles over the same longest edge before the query, so a
    query nearest to such a face reports the distance to its segment (to
    within the repair offset) rather than the distance to some farther valid
    face. See :func:`_repair_degenerate_faces`.
    """
    if query_points.shape[-1] != 3:
        raise ValueError("query_points must have last dimension of size 3")

    # A triangle surface in 3D is required: the Warp mesh queries assume
    # 3-vertex cells with 3D coordinates. Validate here so a mis-typed mesh
    # fails loudly rather than deep inside the Warp op.
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
    if max_dist is not None and max_dist < 0:
        raise ValueError(f"max_dist must be None or non-negative, got {max_dist}")

    # None -> unbounded exact search; a finite value is a narrow band.
    max_dist_eff = float("inf") if max_dist is None else float(max_dist)

    # Warp's mesh query skips (near-)zero-area triangles, so repair them into
    # equivalent thin-but-valid faces first (welded topology preserved; see
    # :func:`_splice_repaired_faces`). float32 up front: the Warp op computes
    # in float32 anyway, and the repair thresholds are float32-calibrated.
    points, cells = _splice_repaired_faces(
        mesh.points.to(torch.float32), mesh.cells.to(torch.long)
    )

    return _warp_signed_distance_field(
        points,
        cells,
        query_points,
        max_dist=max_dist_eff,
        use_sign_winding_number=use_sign_winding_number,
    )
