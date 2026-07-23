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

"""Tests for the mesh-native signed distance field.

The nearest-triangle query is backed by :class:`physicsnemo.mesh.spatial.BVH`
(Triton fast path on CUDA, bounded-stack PyTorch DFS as the reference); the
winding-number sign is computed with a
:class:`physicsnemo.mesh.spatial.ClusterTree` Barnes-Hut summation, with the
exact ``O(n_queries * n_faces)`` torch sum as the oracle.
"""

import math

import pytest
import torch

from physicsnemo.mesh import Mesh
from physicsnemo.mesh.spatial.sdf import (
    _signed_distance_field_from_arrays,
    signed_distance_field,
)


# Build a simple tetrahedron surface mesh as four triangles (a deterministic
# fixture with known SDF values).
def _tetrahedron_mesh() -> Mesh:
    """A tetrahedron surface mesh: 12 vertices, 4 triangles, known SDF values."""
    vertices = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )
    cells = torch.arange(12, dtype=torch.int64).reshape(-1, 3)
    return Mesh(points=vertices, cells=cells)


def _uv_sphere_mesh(n_rings: int = 40, n_segments: int = 80) -> Mesh:
    """Build a UV-sphere triangle mesh (unit radius) for analytic SDF checks."""
    phi = torch.linspace(0, math.pi, n_rings + 2)[1:-1]
    theta = torch.linspace(0, 2 * math.pi, n_segments + 1)[:-1]
    phi_g, theta_g = torch.meshgrid(phi, theta, indexing="ij")
    sin_phi = phi_g.sin()
    ring = torch.stack(
        [sin_phi * theta_g.cos(), sin_phi * theta_g.sin(), phi_g.cos()], dim=-1
    ).reshape(-1, 3)
    vertices = torch.cat(
        [torch.tensor([[0.0, 0.0, 1.0]]), ring, torch.tensor([[0.0, 0.0, -1.0]])]
    ).float()

    south = n_rings * n_segments + 1
    j = torch.arange(n_segments)
    j_next = (j + 1) % n_segments
    north = torch.stack([torch.zeros_like(j), 1 + j, 1 + j_next], dim=1)
    r = torch.arange(n_rings - 1).unsqueeze(1)
    base = 1 + r * n_segments
    p00, p01 = base + j, base + j_next
    p10, p11 = base + n_segments + j, base + n_segments + j_next
    body = torch.stack(
        [torch.stack([p00, p10, p11], -1), torch.stack([p00, p11, p01], -1)], dim=2
    ).reshape(-1, 3)
    last = south - n_segments
    south_fan = torch.stack([last + j, torch.full_like(j, south), last + j_next], dim=1)
    faces = torch.cat([north, body, south_fan]).to(torch.int32)
    return Mesh(points=vertices, cells=faces)


# ---------------------------------------------------------------------------
# L-prism: a non-convex, sharp-edged watertight surface for sign-correctness
# checks. The single nearest-face pseudo-normal is unreliable at sharp edges,
# whereas the winding number is robust; these helpers expose that difference.
# ---------------------------------------------------------------------------

_L_PRISM_HEIGHT = 0.6  # z-extent of the extruded L cross-section


def _l_prism_inside(points: torch.Tensor) -> torch.Tensor:
    r"""Exact inside test for the L-prism (strict interior).

    The L cross-section is the union of a bottom rectangle
    :math:`(0, 1) \times (0, 0.5)` and a top-left rectangle
    :math:`(0, 0.5) \times (0, 1)`, extruded over :math:`(0, H)` in ``z``; the
    notch (``x > 0.5`` and ``y > 0.5``) is outside. Used both as ground truth and
    to orient the mesh outward.

    Parameters
    ----------
    points : torch.Tensor
        Query points, shape :math:`(\dots, 3)`.

    Returns
    -------
    torch.Tensor
        Boolean tensor of shape :math:`(\dots,)`; ``True`` strictly inside.
    """
    x, y, z = points[..., 0], points[..., 1], points[..., 2]
    in_z = (z > 0) & (z < _L_PRISM_HEIGHT)
    bottom = (x > 0) & (x < 1.0) & (y > 0) & (y < 0.5)
    top_left = (x > 0) & (x < 0.5) & (y > 0) & (y < 1.0)
    return in_z & (bottom | top_left)


def _l_prism_mesh() -> Mesh:
    r"""Build a watertight, outward-oriented L-prism surface mesh.

    A non-convex L-shaped polygon (one reflex corner) extruded in ``z`` into a
    closed triangular surface with both convex and reflex sharp edges. Small and
    fully self-contained (12 vertices, 20 triangles).

    Returns
    -------
    Mesh
        Triangle surface mesh (12 vertices, 20 triangles); triangles are wound so
        normals point outward.
    """
    # L-polygon corners in CCW order; index 3 is the reflex corner. The polygon
    # is star-shaped from corner 0, so each cap is a simple fan from that corner.
    corners = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 0.5], [0.5, 0.5], [0.5, 1.0], [0.0, 1.0]],
        dtype=torch.float32,
    )
    bottom = torch.cat([corners, torch.zeros(6, 1)], dim=1)
    top = torch.cat([corners, torch.full((6, 1), _L_PRISM_HEIGHT)], dim=1)
    vertices = torch.cat([bottom, top], dim=0)  # top vertex i lives at index i + 6

    # Two caps (fans from corner 0 / 6) plus one quad -> two triangles per wall.
    faces = [
        [0, 1, 2],
        [0, 2, 3],
        [0, 3, 4],
        [0, 4, 5],
        [6, 7, 8],
        [6, 8, 9],
        [6, 9, 10],
        [6, 10, 11],
    ]
    for a, b in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]:
        faces += [[a, b, b + 6], [a, b + 6, a + 6]]
    faces = torch.tensor(faces, dtype=torch.int64)

    # Orient every face outward: reverse the winding of any triangle whose raw
    # normal points into the solid (tested by nudging the centroid along it).
    tri = vertices[faces]  # (n_faces, 3, 3)
    normals = torch.linalg.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    centroids = tri.mean(dim=1)
    points_inward = _l_prism_inside(centroids + 1e-4 * normals)
    faces[points_inward] = faces[points_inward][:, [0, 2, 1]]
    return Mesh(points=vertices, cells=faces)


def _open_uv_sphere_mesh(n_rings: int = 40, n_segments: int = 80) -> Mesh:
    """A UV sphere with the south polar cap removed (non-watertight surface)."""
    phi = torch.linspace(0, math.pi, n_rings + 2)[1:-1]
    theta = torch.linspace(0, 2 * math.pi, n_segments + 1)[:-1]
    phi_g, theta_g = torch.meshgrid(phi, theta, indexing="ij")
    sin_phi = phi_g.sin()
    ring = torch.stack(
        [sin_phi * theta_g.cos(), sin_phi * theta_g.sin(), phi_g.cos()], dim=-1
    ).reshape(-1, 3)
    vertices = torch.cat(
        [torch.tensor([[0.0, 0.0, 1.0]]), ring, torch.tensor([[0.0, 0.0, -1.0]])]
    ).float()

    j = torch.arange(n_segments)
    j_next = (j + 1) % n_segments
    # North fan + body, but drop the south fan so the surface has a hole.
    north = torch.stack([torch.zeros_like(j), 1 + j, 1 + j_next], dim=1)
    r = torch.arange(n_rings - 1).unsqueeze(1)
    base = 1 + r * n_segments
    p00, p01 = base + j, base + j_next
    p10, p11 = base + n_segments + j, base + n_segments + j_next
    body = torch.stack(
        [torch.stack([p00, p10, p11], -1), torch.stack([p00, p11, p01], -1)], dim=2
    ).reshape(-1, 3)
    faces = torch.cat([north, body]).to(torch.int32)
    return Mesh(points=vertices, cells=faces)


def _l_prism_thick_mesh(thickness: float = 1.0) -> Mesh:
    """Watertight, outward-wound L-shaped triangular prism (non-convex solid).

    The cross-section (in the xy-plane) is the hexagon ``(0,0) -> (2,0) ->
    (2,1) -> (1,1) -> (1,2) -> (0,2)`` with a *reflex* (concave) corner at
    ``(1, 1)``, extruded along ``z`` to ``thickness``. Caps are simple fans from
    a single vertex (the polygon is star-shaped from ``(0, 0)``), walls are one
    quad per boundary edge, and every triangle is wound so its normal points
    outward. The reflex edge at ``x = y = 1`` is the sharp feature where a single
    face normal is insufficient to sign the field.
    """
    boundary = torch.tensor(
        [[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 1.0], [1.0, 2.0], [0.0, 2.0]]
    )
    n = boundary.shape[0]
    bottom = torch.cat([boundary, torch.zeros(n, 1)], dim=1)
    top = torch.cat([boundary, torch.full((n, 1), thickness)], dim=1)
    vertices = torch.cat([bottom, top], dim=0).float()  # (2n, 3); top vertex = n + i

    # Bottom cap, outward normal -z: reversed fan from vertex 0.
    bottom_cap = [[0, i + 1, i] for i in range(1, n - 1)]
    # Top cap, outward normal +z: fan from vertex n.
    top_cap = [[n, n + i, n + i + 1] for i in range(1, n - 1)]
    # Side walls, outward in-plane normal: a quad (two triangles) per edge.
    walls = [
        tri
        for i in range(n)
        for tri in ([i, (i + 1) % n, n + (i + 1) % n], [i, n + (i + 1) % n, n + i])
    ]
    faces = bottom_cap + top_cap + walls

    return Mesh(points=vertices, cells=torch.tensor(faces, dtype=torch.int32))


def _inside_l(points: torch.Tensor, thickness: float = 1.0) -> torch.Tensor:
    """Exact inside/outside test for the :func:`_l_prism_thick` solid (the oracle)."""
    x, y, z = points[..., 0], points[..., 1], points[..., 2]
    in_z = (z >= 0.0) & (z <= thickness)
    in_xy = ((x >= 0.0) & (x <= 2.0) & (y >= 0.0) & (y <= 1.0)) | (
        (x >= 0.0) & (x <= 1.0) & (y >= 1.0) & (y <= 2.0)
    )
    return in_z & in_xy


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("use_winding", [False, True])
def test_sdf_tetrahedron_reference(dtype, use_winding, device):
    """Match the known deterministic tetrahedron SDF values."""
    device = torch.device(device)
    mesh = _tetrahedron_mesh().to(device=device, dtype=dtype)
    query_points = torch.tensor(
        [[1.0, 1.0, 1.0], [0.05, 0.1, 0.1]], device=device, dtype=dtype
    )

    sdf_out, hit_points, _ = signed_distance_field(
        mesh,
        query_points,
        use_sign_winding_number=use_winding,
    )

    torch.testing.assert_close(
        sdf_out,
        torch.tensor([1.1547, -0.05], device=device, dtype=dtype),
        atol=1e-4,
        rtol=1e-4,
    )
    torch.testing.assert_close(
        hit_points,
        torch.tensor(
            [[0.33333322, 0.33333334, 0.3333334], [0.0, 0.10, 0.10]],
            device=device,
            dtype=dtype,
        ),
        atol=1e-4,
        rtol=1e-4,
    )


def test_sdf_index_layout_compatibility(device):
    """Flat and (n_faces, 3) connectivity agree (private array helper)."""
    device = torch.device(device)
    tet = _tetrahedron_mesh().to(device)
    # The array helper is tested directly here, so pull its inputs from the mesh.
    mesh_indices_flat = tet.cells.reshape(-1)
    mesh_indices_faces = tet.cells
    query_points = torch.tensor([[0.1, 0.2, 0.3]], device=device, dtype=torch.float32)

    sdf_flat, hit_flat, _ = _signed_distance_field_from_arrays(
        tet.points, mesh_indices_flat, query_points
    )
    sdf_faces, hit_faces, _ = _signed_distance_field_from_arrays(
        tet.points, mesh_indices_faces, query_points
    )
    torch.testing.assert_close(sdf_flat, sdf_faces)
    torch.testing.assert_close(hit_flat, hit_faces)


@pytest.mark.parametrize("use_winding", [False, True])
def test_sdf_sphere_analytic(use_winding, device):
    """SDF of a tessellated unit sphere matches the analytic ``|r| - 1``."""
    device = torch.device(device)
    mesh = _uv_sphere_mesh().to(device)

    torch.manual_seed(0)
    query = (torch.rand(4096, 3, device=device) * 3.0 - 1.5).float()
    radius = query.norm(dim=-1)
    gt = radius - 1.0

    sdf_out, hit, _ = signed_distance_field(
        mesh, query, use_sign_winding_number=use_winding
    )

    # The error is dominated by the polygonal approximation of the sphere, not
    # the algorithm; a coarse tolerance captures that the magnitude is correct.
    torch.testing.assert_close(sdf_out, gt, atol=5e-3, rtol=0.0)

    # Sign must agree with the analytic field away from the surface.
    far = gt.abs() > 0.05
    assert torch.all(sdf_out[far].sign() == gt[far].sign())

    # Hit points lie (approximately) on the unit sphere.
    torch.testing.assert_close(
        hit.norm(dim=-1), torch.ones_like(radius), atol=5e-3, rtol=0.0
    )


def test_sdf_preserves_input_shape(device):
    """Output SDF/hit-point shapes follow the (possibly batched) query shape."""
    device = torch.device(device)
    mesh = _tetrahedron_mesh().to(device)
    query = torch.rand(4, 5, 3, device=device)

    sdf_out, hit, hit_faces = signed_distance_field(mesh, query)
    assert sdf_out.shape == (4, 5)
    assert hit.shape == (4, 5, 3)
    assert hit_faces.shape == (4, 5)
    assert hit_faces.dtype == torch.long


def test_sdf_public_matches_private_arrays(device):
    """The public Mesh API and the private array helper agree exactly."""
    device = torch.device(device)
    mesh = _uv_sphere_mesh().to(device)

    torch.manual_seed(0)
    query = (torch.rand(4096, 3, device=device) * 3.0 - 1.5).float()

    sdf_pub, hit_pub, _ = signed_distance_field(mesh, query)
    sdf_priv, hit_priv, _ = _signed_distance_field_from_arrays(
        mesh.points, mesh.cells, query
    )
    torch.testing.assert_close(sdf_pub, sdf_priv)
    torch.testing.assert_close(hit_pub, hit_priv)


def test_sdf_error_handling(device):
    """Input validation for the public (Mesh) SDF interface."""
    device = torch.device(device)
    mesh = _tetrahedron_mesh().to(device)
    query = torch.tensor([[0.1, 0.2, 0.3]], device=device, dtype=torch.float32)

    bad_queries = torch.randn(4, 2, device=device)
    with pytest.raises(ValueError, match="last dimension of size 3"):
        signed_distance_field(mesh, bad_queries)

    # Non-triangle connectivity (cells wider than 3) is rejected up front.
    quad_cells = torch.zeros(2, 4, device=device, dtype=torch.int64)
    with pytest.raises(ValueError, match="triangle mesh"):
        signed_distance_field(Mesh(points=mesh.points, cells=quad_cells), query)

    # A mesh embedded in 2D has no well-defined 3D signed distance.
    flat_points = mesh.points[:, :2].contiguous()
    with pytest.raises(ValueError, match="3D mesh"):
        signed_distance_field(Mesh(points=flat_points, cells=mesh.cells), query)

    # An unknown winding backend is rejected up front (before any BVH work),
    # even when the winding-number sign path is not selected.
    with pytest.raises(ValueError, match="winding_backend"):
        signed_distance_field(mesh, query, winding_backend="warp")

    # A negative search radius is rejected rather than silently behaving
    # like its absolute value.
    with pytest.raises(ValueError, match="max_dist"):
        signed_distance_field(mesh, query, max_dist=-1.0)


def test_sdf_winding_backend_selection(device):
    """Both winding backends agree through the public API on a closed surface."""
    device = torch.device(device)
    mesh = _uv_sphere_mesh().to(device)

    torch.manual_seed(0)
    query = (torch.rand(512, 3, device=device) * 3.0 - 1.5).float()

    sdf_tree, hit_tree, _ = signed_distance_field(
        mesh, query, use_sign_winding_number=True, winding_backend="clustertree"
    )
    sdf_brute, hit_brute, _ = signed_distance_field(
        mesh, query, use_sign_winding_number=True, winding_backend="bruteforce"
    )

    # The unsigned distance and hit point come from the same nearest-triangle
    # search; only the sign may differ, and away from the surface it must not.
    torch.testing.assert_close(hit_tree, hit_brute)
    near_surface = sdf_brute.abs() < 0.05
    assert torch.all(sdf_tree[~near_surface] == sdf_brute[~near_surface])


def test_sdf_hit_faces_identify_nearest_face(device):
    """``hit_faces`` indexes the face that realizes the reported distance.

    Recomputing the closest point on the returned face must reproduce both the
    hit point and the unsigned distance. On CUDA this also checks the Triton
    kernel's mapping from BVH-sorted cell order back to input face indices.
    """
    from physicsnemo.mesh.spatial.sdf import _closest_point_on_triangles

    device = torch.device(device)
    mesh = _uv_sphere_mesh().to(device)

    torch.manual_seed(0)
    query = (torch.rand(2048, 3, device=device) * 3.0 - 1.5).float()

    sdf_out, hit, hit_faces = signed_distance_field(mesh, query)

    assert hit_faces.dtype == torch.long
    assert hit_faces.min() >= 0
    assert hit_faces.max() < mesh.n_cells
    tri = mesh.points.float()[mesh.cells.long()[hit_faces]]
    closest = _closest_point_on_triangles(query, tri)
    torch.testing.assert_close(
        (query - closest).norm(dim=-1), sdf_out.abs(), atol=1e-6, rtol=1e-5
    )
    torch.testing.assert_close(closest, hit, atol=1e-6, rtol=1e-5)


def test_sdf_array_connectivity_validation(device):
    """The private array helper still validates raw connectivity layout."""
    device = torch.device(device)
    vertices = _tetrahedron_mesh().to(device).points
    query = torch.tensor([[0.1, 0.2, 0.3]], device=device, dtype=torch.float32)

    bad_connectivity_shape = torch.zeros(4, 4, device=device, dtype=torch.int32)
    with pytest.raises(ValueError, match=r"shape \(n_faces, 3\)"):
        _signed_distance_field_from_arrays(vertices, bad_connectivity_shape, query)

    bad_connectivity_rank = torch.zeros(1, 2, 3, device=device, dtype=torch.int32)
    with pytest.raises(ValueError, match="1D flattened indices or 2D"):
        _signed_distance_field_from_arrays(vertices, bad_connectivity_rank, query)


def test_sdf_empty_mesh_raises(device):
    """A mesh with no faces has no surface, so the query must raise."""
    device = torch.device(device)
    points = _tetrahedron_mesh().to(device).points
    empty_mesh = Mesh(
        points=points, cells=torch.zeros(0, 3, device=device, dtype=torch.int64)
    )
    query = torch.tensor([[0.1, 0.2, 0.3]], device=device, dtype=torch.float32)

    with pytest.raises(ValueError, match="no faces"):
        signed_distance_field(empty_mesh, query)


def test_repair_degenerate_faces(device):
    """Degenerate faces are repaired into valid thin triangles; others untouched.

    Ericson's Voronoi-region cascade assumes a non-degenerate triangle: on
    zero-area faces several region tests fire vacuously and the cascade can
    return the wrong feature (an overestimated distance). The build step must
    therefore replace every repeated-vertex or collinear face with a valid
    thin triangle spanning the same longest edge, moving the surface by no
    more than the documented offset ``h``, while leaving valid faces
    bit-identical and point-like faces (exact under Ericson) alone.
    """
    from physicsnemo.mesh.spatial.sdf import (
        _DEGENERATE_TRI_REL_HEIGHT,
        _repair_degenerate_faces,
    )

    device = torch.device(device)
    torch.manual_seed(0)
    n = 10_000
    a = torch.randn(n, 3, device=device)
    b = torch.randn(n, 3, device=device)
    t_mid = torch.rand(n, 1, device=device) * 2 - 0.5
    mid = a + (b - a) * t_mid  # collinear, inside and beyond the a-b span

    def seg_dist(p, s0, s1):
        d = s1 - s0
        denom = (d * d).sum(-1)
        t = ((p - s0) * d).sum(-1) / denom.clamp(min=1e-30)
        t = torch.where(denom > 0, t.clamp(0.0, 1.0), torch.zeros_like(t))
        return (p - (s0 + d * t.unsqueeze(-1))).norm(dim=-1)

    degenerate_tris = [
        torch.stack([a, a, b], dim=1),
        torch.stack([a, b, a], dim=1),
        torch.stack([b, a, a], dim=1),
        torch.stack([a, mid, b], dim=1),
    ]
    for tri in degenerate_tris:
        repaired = _repair_degenerate_faces(tri)
        ra, rb, rc = repaired[:, 0], repaired[:, 1], repaired[:, 2]
        # Valid now: relative height comfortably above the float32 danger zone.
        area2 = torch.linalg.cross(rb - ra, rc - ra, dim=-1).norm(dim=-1)
        edge_max = torch.stack(
            [(rb - ra).norm(dim=-1), (rc - ra).norm(dim=-1), (rc - rb).norm(dim=-1)],
            dim=-1,
        ).amax(dim=-1)
        rel_height = area2 / edge_max.clamp(min=1e-30) ** 2
        assert torch.all(rel_height > 0.5 * _DEGENERATE_TRI_REL_HEIGHT)
        # The repaired surface stays within h of the original geometry: every
        # repaired vertex lies within h of the union of the original edges.
        # (Individual vertices may travel far -- a repeated vertex moves to
        # the edge midpoint -- but never off the original segment by more
        # than h.)
        h = (_DEGENERATE_TRI_REL_HEIGHT * edge_max + 1e-5).unsqueeze(-1)
        dist_to_orig = torch.stack(
            [
                torch.minimum(
                    torch.minimum(
                        seg_dist(repaired[:, k], tri[:, 0], tri[:, 1]),
                        seg_dist(repaired[:, k], tri[:, 1], tri[:, 2]),
                    ),
                    seg_dist(repaired[:, k], tri[:, 2], tri[:, 0]),
                )
                for k in range(3)
            ],
            dim=-1,
        )
        assert torch.all(dist_to_orig <= h)

    # Point-like faces are exact under Ericson already: left untouched.
    point_tri = torch.stack([a, a, a], dim=1)
    assert torch.equal(_repair_degenerate_faces(point_tri), point_tri)

    # Valid faces come back bit-identical.
    c = torch.randn(n, 3, device=device)
    valid = torch.stack([a, b, c], dim=1)
    area2 = torch.linalg.cross(b - a, c - a, dim=-1).norm(dim=-1)
    edge_max = torch.stack(
        [(b - a).norm(dim=-1), (c - a).norm(dim=-1), (c - b).norm(dim=-1)], dim=-1
    ).amax(dim=-1)
    valid = valid[area2 / edge_max**2 > 10 * _DEGENERATE_TRI_REL_HEIGHT]
    assert torch.equal(_repair_degenerate_faces(valid), valid)


def test_sdf_degenerate_face_mesh(device):
    """An isolated degenerate face reports the distance to its segment.

    End-to-end regression: a repeated-vertex face spanning the segment
    (0,0,0)-(4,0,0) must report the distance to that segment (nearest point
    (4,0,0) here) to within the documented repair offset -- not the distance
    to one of its vertices (an error of ~4 here before the fix). On CUDA this
    exercises the Triton kernel path; on CPU the torch DFS.
    """
    device = torch.device(device)
    # One repeated-vertex face spanning a segment, plus a far valid triangle so
    # the mesh also contains non-degenerate geometry.
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [100.0, 0.0, 0.0],
            [101.0, 0.0, 0.0],
            [100.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
        device=device,
    )
    cells = torch.tensor([[0, 1, 1], [2, 3, 4]], dtype=torch.int64, device=device)
    mesh = Mesh(points=points, cells=cells)

    query = torch.tensor([[4.5, 0.2, 0.0]], dtype=torch.float32, device=device)
    sdf_out, hit, _ = signed_distance_field(mesh, query, use_sign_winding_number=True)

    # The repair moves the surface by at most 2 h = 2 * 1e-4 * 4; assert to
    # 1e-3 to leave headroom over float32 arithmetic on top of that bound.
    true_dist = math.hypot(0.5, 0.2)
    torch.testing.assert_close(
        sdf_out.abs(),
        torch.tensor([true_dist], device=device),
        atol=1e-3,
        rtol=0.0,
    )
    torch.testing.assert_close(
        hit,
        torch.tensor([[4.0, 0.0, 0.0]], device=device),
        atol=1e-3,
        rtol=0.0,
    )


def test_sdf_max_dist_unbounded_and_narrow_band(device):
    """Default is exact/unbounded; a finite ``max_dist`` is a narrow band.

    The unbounded default must resolve a far query to its true nearest triangle
    (never the silent ``sdf == 0`` / ``hit == query`` on-surface result), while a
    finite ``max_dist`` smaller than the true distance reports the query as
    ``NaN`` and leaves in-band queries identical to the unbounded result.
    """
    device = torch.device(device)
    mesh = _tetrahedron_mesh().to(device)

    far = torch.tensor([[100.0, 100.0, 100.0]], device=device, dtype=torch.float32)
    near = torch.tensor([[0.05, 0.1, 0.1]], device=device, dtype=torch.float32)

    # Unbounded default: the far query finds its true nearest triangle.
    sdf_far, hit_far, _ = signed_distance_field(mesh, far)
    assert torch.isfinite(sdf_far).all()
    assert sdf_far.abs().item() > 1.0
    assert not torch.allclose(hit_far, far)

    # Finite band below the true distance: the far query is out of band ->
    # NaN results and a -1 face index (int64 has no NaN).
    sdf_band, hit_band, faces_band = signed_distance_field(mesh, far, max_dist=1.0)
    assert torch.isnan(sdf_band).all()
    assert torch.isnan(hit_band).all()
    assert (faces_band == -1).all()

    # An in-band query with a finite max_dist matches the unbounded result.
    sdf_unbounded, _, faces_unbounded = signed_distance_field(mesh, near)
    sdf_in_band, _, faces_in_band = signed_distance_field(mesh, near, max_dist=10.0)
    assert torch.isfinite(sdf_in_band).all()
    torch.testing.assert_close(sdf_in_band, sdf_unbounded, atol=1e-5, rtol=1e-5)
    assert (faces_in_band == faces_unbounded).all()
    assert (faces_in_band >= 0).all()


def test_sdf_pseudo_normal_sign_wrong_at_sharp_edges(device):
    r"""Document the nearest-face pseudo-normal sign bug at sharp edges.

    The default sign method classifies a query as inside/outside using the
    outward normal of the *single* nearest triangle. Near a sharp convex or
    reflex edge the nearest feature is the edge itself - shared by two faces with
    very different normals - so picking one face's normal can flip the sign. A
    robust implementation uses the angle-weighted pseudo-normal or the
    generalized winding number.
    """
    device = torch.device(device)
    mesh = _l_prism_mesh().to(device)

    # Build the query set on CPU (device-independent point set), then move it.
    # The expanded box reaches into the exterior wedges of the convex edges; the
    # cluster densely probes the reflex edge at (x, y) = (0.5, 0.5).
    torch.manual_seed(0)
    lo = torch.tensor([-0.25, -0.25, -0.25])
    hi = torch.tensor([1.25, 1.25, _L_PRISM_HEIGHT + 0.25])
    box = lo + (hi - lo) * torch.rand(200_000, 3)
    reflex = torch.rand(100_000, 3)
    reflex[:, 0] = 0.35 + 0.30 * reflex[:, 0]
    reflex[:, 1] = 0.35 + 0.30 * reflex[:, 1]
    reflex[:, 2] = _L_PRISM_HEIGHT * reflex[:, 2]
    query = torch.cat([box, reflex], dim=0).to(device)

    sdf_out, _, _ = signed_distance_field(mesh, query, use_sign_winding_number=False)

    # The distance magnitude is correct; only the sign is in question. Compare to
    # the analytic interior away from the surface, where the sign is unambiguous.
    inside = _l_prism_inside(query)
    away = sdf_out.abs() > 1e-3
    wrong = ((sdf_out < 0) != inside) & away

    n_wrong = int(wrong.sum())
    assert n_wrong == 0, (
        f"Nearest-face pseudo-normal sign misclassified {n_wrong} of "
        f"{int(away.sum())} points near the L-prism's sharp edges. The single "
        f"nearest-face normal is unreliable at sharp convex/reflex edges; use an "
        f"angle-weighted pseudo-normal or the winding number for the sign."
    )


def test_sdf_winding_sign_correct_at_sharp_edges(device):
    r"""Control: the winding-number sign is correct on the same L-prism.

    Identical sharp-edged mesh as
    ``test_sdf_pseudo_normal_sign_wrong_at_sharp_edges`` but with
    ``use_sign_winding_number=True``. The generalized winding number is robust at
    sharp edges, so the sign matches the analytic interior. This confirms the mesh
    is valid and isolates the failure to the pseudo-normal method.
    """
    device = torch.device(device)
    mesh = _l_prism_mesh().to(device)

    torch.manual_seed(1)
    lo = torch.tensor([-0.2, -0.2, -0.2])
    hi = torch.tensor([1.2, 1.2, _L_PRISM_HEIGHT + 0.2])
    query = (lo + (hi - lo) * torch.rand(40_000, 3)).to(device)

    sdf_out, _, _ = signed_distance_field(mesh, query, use_sign_winding_number=True)

    # Exclude a near-surface band: the CUDA Barnes-Hut winding approximation is
    # only loose right at the surface (cf. test_winding_sign_triton_matches_exact).
    inside = _l_prism_inside(query)
    away = sdf_out.abs() > 0.05
    wrong = ((sdf_out < 0) != inside) & away
    assert int(wrong.sum()) == 0


# ---------------------------------------------------------------------------
# ClusterTree winding-number sign: the Barnes-Hut summation must agree with the
# exact O(n_queries * n_faces) torch sum (the oracle) away from the surface,
# where the winding number is unambiguous. Runs on both CPU and CUDA.
# ---------------------------------------------------------------------------


def test_clustertree_winding_sign_matches_exact_oracle(device):
    """Tree-accelerated winding sign agrees with the exact winding sign."""
    from physicsnemo.mesh.spatial.sdf import (
        _build_surface_mesh,
        _winding_number_sign,
        _winding_number_sign_clustertree,
    )

    device = torch.device(device)
    mesh = _uv_sphere_mesh().to(device)

    torch.manual_seed(0)
    query = (torch.rand(4096, 3, device=device) * 3.0 - 1.5).float()
    radius = query.norm(dim=-1)
    away = (radius - 1.0).abs() > 0.05  # exclude the near-surface shell

    _, face_vertices, _ = _build_surface_mesh(mesh)

    sign_fast = _winding_number_sign_clustertree(face_vertices, query)
    sign_exact = _winding_number_sign(face_vertices, query)

    assert torch.all(sign_fast[away] == sign_exact[away])


def test_clustertree_winding_sign_non_watertight(device):
    """On a holed (non-watertight) surface the winding sign is still robust.

    The generalized winding number degrades gracefully on open meshes; the
    ClusterTree Barnes-Hut summation must still match the exact oracle away from
    the surface and from the hole's rim.
    """
    from physicsnemo.mesh.spatial.sdf import (
        _build_surface_mesh,
        _winding_number_sign,
        _winding_number_sign_clustertree,
    )

    device = torch.device(device)
    mesh = _open_uv_sphere_mesh().to(device)

    torch.manual_seed(1)
    query = (torch.rand(4096, 3, device=device) * 3.0 - 1.5).float()
    radius = query.norm(dim=-1)
    away = (radius - 1.0).abs() > 0.1  # exclude near-surface / near-hole shell

    _, face_vertices, _ = _build_surface_mesh(mesh)

    sign_fast = _winding_number_sign_clustertree(face_vertices, query)
    sign_exact = _winding_number_sign(face_vertices, query)

    assert torch.all(sign_fast[away] == sign_exact[away])


# ---------------------------------------------------------------------------
# Sharp / non-convex sign correctness. On an L-prism the closest surface point
# frequently lands on an edge or vertex shared by several faces. A single
# nearest-face normal is then insufficient: the query can sit behind that one
# face's half-plane while still being outside the solid, flipping the sign. The
# angle-weighted pseudo-normal (face/edge/vertex feature) and the winding number
# both stay correct -- here checked against the exact analytic inside/outside.
# ---------------------------------------------------------------------------


def _l_prism_probe_grid(device: torch.device, thickness: float = 1.0):
    """A dense grid of probes straddling the L-prism's surface and reflex edge."""
    g = torch.linspace(-0.5, 2.5, 31)
    z = torch.linspace(-0.4, thickness + 0.4, 13)
    gx, gy, gz = torch.meshgrid(g, g, z, indexing="ij")
    return torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3).to(device)


def test_sdf_winding_sign_correct_at_sharp_edges_grid(device):
    """Winding-number sign matches the analytic L-prism field (companion check)."""
    device = torch.device(device)
    thickness = 1.0
    mesh = _l_prism_thick_mesh(thickness).to(device)

    query = _l_prism_probe_grid(device, thickness)
    gt_inside = _inside_l(query, thickness)

    sdf_out, _, _ = signed_distance_field(mesh, query, use_sign_winding_number=True)

    # Compare signs away from the surface. The default ClusterTree backend is a
    # Barnes-Hut approximation whose winding number is only unreliable in a thin
    # band hugging the (sharp) surface, so exclude points within 0.1 of it.
    away = sdf_out.abs() > 0.1
    expected = torch.where(
        gt_inside, -torch.ones_like(sdf_out), torch.ones_like(sdf_out)
    )
    assert torch.all(sdf_out[away].sign() == expected[away])


def test_sdf_pseudo_normal_sign_correct_at_sharp_edges(device):
    """Angle-weighted pseudo-normal sign matches the analytic L-prism field.

    Regression for the default (``use_sign_winding_number=False``) sign path:
    using only the nearest face normal misclassifies points whose closest point
    lies on the reflex edge / its vertices, whereas the feature pseudo-normal
    agrees with the exact field (and with the winding-number companion above).
    """
    device = torch.device(device)
    thickness = 1.0
    mesh = _l_prism_thick_mesh(thickness).to(device)

    query = _l_prism_probe_grid(device, thickness)
    gt_inside = _inside_l(query, thickness)

    sdf_out, _, _ = signed_distance_field(mesh, query, use_sign_winding_number=False)

    away = sdf_out.abs() > 0.05
    expected = torch.where(
        gt_inside, -torch.ones_like(sdf_out), torch.ones_like(sdf_out)
    )
    assert torch.all(sdf_out[away].sign() == expected[away])


# ---------------------------------------------------------------------------
# Edge pseudo-normal grouping: the default (pseudo-normal) sign path sums the
# incident face normals per edge. The grouping was rewritten to avoid
# ``torch.unique(edges, dim=0)`` (a host sync that stalled the SDF prep stream),
# so it must still match a direct ``torch.unique`` reference exactly.
# ---------------------------------------------------------------------------


def _edge_pseudonormals_unique_reference(
    tri_faces: torch.Tensor, face_normals: torch.Tensor
) -> torch.Tensor:
    """Reference edge pseudo-normals via ``torch.unique`` (the pre-rewrite path)."""
    n_faces = tri_faces.shape[0]
    v0, v1, v2 = tri_faces[:, 0], tri_faces[:, 1], tri_faces[:, 2]
    edges = torch.stack(
        [
            torch.stack([v0, v1], dim=1),
            torch.stack([v1, v2], dim=1),
            torch.stack([v2, v0], dim=1),
        ],
        dim=1,
    ).reshape(-1, 2)
    edges, _ = torch.sort(edges, dim=1)
    unique_edges, inverse = torch.unique(edges, dim=0, return_inverse=True)
    fn_per_edge = face_normals.repeat_interleave(3, dim=0)
    edge_accum = torch.zeros(
        unique_edges.shape[0], 3, dtype=face_normals.dtype, device=face_normals.device
    )
    edge_accum.index_add_(0, inverse, fn_per_edge)
    return edge_accum[inverse].reshape(n_faces, 3, 3)


def test_edge_pseudonormals_matches_unique_reference(device):
    """Sync-free edge-pseudonormal grouping equals the ``torch.unique`` reference.

    Exercises a closed surface with shared edges (every edge is incident to two
    faces) so the per-edge accumulation is non-trivial.
    """
    from physicsnemo.mesh.spatial.sdf import _build_surface_mesh, _edge_pseudonormals

    device = torch.device(device)
    mesh = _uv_sphere_mesh().to(device)

    work_mesh, _, tri_faces = _build_surface_mesh(mesh)
    face_normals = work_mesh.cell_normals.float()

    got = _edge_pseudonormals(tri_faces, face_normals)
    expected = _edge_pseudonormals_unique_reference(tri_faces, face_normals)
    torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# Triton GPU kernel parity (CUDA-only): the kernel is the nearest-triangle fast
# path, the pure-PyTorch bounded-stack DFS is the reference oracle.
# ---------------------------------------------------------------------------

_CUDA = torch.cuda.is_available()


def _triton_available() -> bool:
    if not _CUDA:
        return False
    from physicsnemo.mesh.spatial import _sdf_triton

    return _sdf_triton.available()


@pytest.mark.skipif(not _CUDA, reason="CUDA required for the Triton SDF kernel")
def test_sdf_triton_nearest_matches_torch_reference():
    """The Triton nearest-triangle kernel matches the torch DFS reference.

    Distances are unique, so they must agree tightly. The winning face / closest
    point can differ on exact ties, so those are compared via the query-to-point
    distance rather than the face index.
    """
    if not _triton_available():
        pytest.skip("triton not available")

    from physicsnemo.mesh.spatial import BVH, _sdf_triton
    from physicsnemo.mesh.spatial.sdf import _build_surface_mesh, _nearest_face_bvh

    device = torch.device("cuda")
    mesh = _uv_sphere_mesh().to(device)

    torch.manual_seed(0)
    query = (torch.rand(8192, 3, device=device) * 3.0 - 1.5).float()

    work_mesh, face_vertices, _ = _build_surface_mesh(mesh)
    bvh = BVH.from_mesh(work_mesh)

    ref_dist_sq, _, ref_pt = _nearest_face_bvh(bvh, face_vertices, query, 1e8)
    tri_dist_sq, _, tri_pt = _sdf_triton.nearest_triangle_triton(
        bvh, face_vertices, query, 1e8
    )

    torch.testing.assert_close(
        tri_dist_sq.sqrt(), ref_dist_sq.sqrt(), atol=1e-4, rtol=1e-4
    )
    d_ref = (query - ref_pt).norm(dim=-1)
    d_tri = (query - tri_pt).norm(dim=-1)
    torch.testing.assert_close(d_tri, d_ref, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not _CUDA, reason="CUDA required for the Triton SDF kernel")
@pytest.mark.parametrize("use_winding", [False, True])
def test_sdf_triton_end_to_end_matches_reference(use_winding, monkeypatch):
    """Full signed_distance_field: Triton path matches the torch fallback."""
    if not _triton_available():
        pytest.skip("triton not available")

    from physicsnemo.mesh.spatial import _sdf_triton

    device = torch.device("cuda")
    mesh = _uv_sphere_mesh().to(device)

    torch.manual_seed(0)
    query = (torch.rand(4096, 3, device=device) * 3.0 - 1.5).float()

    # Triton fast path (default dispatch on CUDA).
    sdf_triton, _, _ = signed_distance_field(
        mesh, query, use_sign_winding_number=use_winding
    )

    # Force the pure-PyTorch nearest-triangle reference by disabling the Triton
    # dispatch. The winding-number sign uses the (device-agnostic) ClusterTree
    # path in both cases.
    monkeypatch.setattr(_sdf_triton, "available", lambda: False)
    sdf_ref, _, _ = signed_distance_field(
        mesh, query, use_sign_winding_number=use_winding
    )

    torch.testing.assert_close(sdf_triton, sdf_ref, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not _CUDA, reason="CUDA required to check stream-sync-free SDF")
def test_sdf_no_winding_path_is_sync_free():
    """The default (pseudo-normal) SDF path issues no host<->device syncs.

    The SDF transform runs on the dataloader's preprocessing stream; any host
    sync (e.g. the former ``torch.unique`` in ``_edge_pseudonormals``) blocks the
    main thread mid-enqueue and prevents the prep-stream SDF kernels from
    overlapping the compute-stream model. ``set_sync_debug_mode("error")`` turns
    any synchronizing CUDA call into a ``RuntimeError``, so a clean second run
    proves the path is overlap-safe. The winding-number sign path is
    intentionally excluded -- its ClusterTree traversal still syncs.
    """
    device = torch.device("cuda")
    mesh = _uv_sphere_mesh().to(device)

    torch.manual_seed(0)
    query = (torch.rand(8192, 3, device=device) * 3.0 - 1.5).float()

    # Warm up OUTSIDE the guard: first-call costs (Triton autotune ``do_bench``,
    # lazy module loading, caching-allocator growth) legitimately synchronize.
    # The guarded run below reuses the same query shape so no new autotune key or
    # allocation is triggered.
    signed_distance_field(mesh, query, use_sign_winding_number=False)
    torch.cuda.synchronize()

    # ``error`` mode raises on *implicit* synchronizing ops (``.item()``,
    # ``.cpu()``, ``torch.unique``, ``nonzero`` ...) at enqueue time, so the
    # detection does not need a trailing ``torch.cuda.synchronize()`` inside the
    # guard -- and keeping the explicit sync outside avoids any version-specific
    # ambiguity about whether an intentional device sync is itself flagged.
    prev = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        signed_distance_field(mesh, query, use_sign_winding_number=False)
    finally:
        torch.cuda.set_sync_debug_mode(prev)
    torch.cuda.synchronize()
