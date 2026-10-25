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


"""The exact signed distance and direction against brute-force and exact references.

:func:`~physicsnemo.mesh.spatial.exact_signed_distance` finds the exact closest
point on a boundary mesh's cells and signs the distance by the generalized winding
number, positive on the side the normals point to. These tests pin it against
brute-force and exact references, its sign convention and convergence, its ties
and on-surface rule, the function's validation and shapes, the ball search's
blocked launches, and the Ericson weights it imports.
"""

import itertools
import math

import pytest
import torch

from physicsnemo.mesh import Mesh
from physicsnemo.mesh.primitives.curves import circle_2d
from physicsnemo.mesh.primitives.surfaces import cube_surface, sphere_icosahedral, torus
from physicsnemo.mesh.spatial import exact_signed_distance
from physicsnemo.mesh.spatial.exact_sdf import (
    _cell_tree,
    _predicates,
    _search,
    _weights,
    _winding,
)
from physicsnemo.nn.functional.geometry.sdf import _closest_point_weights
from test.mesh.spatial._exact_sdf_meshes import (
    cloud,
    exact_sdf,
    orthogonal,
    soup,
    transform,
)

D = torch.float64


# ---------------------------------------------------------------------------
# The exact geometry against the boundary surface (exact_sdf)
# ---------------------------------------------------------------------------


def _oriented(mesh: Mesh, outward) -> Mesh:
    """``mesh`` with every cell re-oriented so its normal has a non-negative dot with
    ``outward(centroid)``."""
    cells = mesh.cells.clone()
    flip = (mesh.cell_normals * outward(mesh.cell_centroids)).sum(-1) < 0
    cells[flip, -2:] = cells[flip, -2:].flip(-1)
    return Mesh(points=mesh.points, cells=cells)


def _polygon(verts) -> Mesh:
    """A closed polyline through ``verts`` (n, 2), oriented with outward normals."""
    n = verts.shape[0]
    cells = torch.stack([torch.arange(n), torch.arange(1, n + 1) % n], -1)
    mesh = Mesh(points=verts, cells=cells)
    ### the shoelace area is positive for a counterclockwise polygon, whose rotated
    ### edges (the cell normals) point inward
    x, y = verts[:, 0], verts[:, 1]
    if (x * y.roll(-1) - x.roll(-1) * y).sum() > 0:
        mesh = Mesh(points=verts, cells=cells.flip(-1))
    return mesh


def _polygon_sdf(q, verts):
    """The exact signed distance to a closed polygon (positive outside): the brute
    force over its segments in float64, signed by the crossing-number test."""
    a, b = verts, verts.roll(-1, 0)
    ab = b - a
    t = (((q[:, None] - a) * ab).sum(-1) / (ab * ab).sum(-1)).clamp(0, 1)
    dist = (q[:, None] - a - t[..., None] * ab).norm(dim=-1).min(-1).values
    ya, yb = a[:, 1], b[:, 1]
    crosses = (ya > q[:, None, 1]) != (yb > q[:, None, 1])
    x_at = a[:, 0] + (q[:, None, 1] - ya) / (yb - ya) * ab[:, 0]
    inside = (crosses & (q[:, None, 0] < x_at)).sum(-1) % 2 == 1
    return torch.where(inside, -dist, dist)


def _naca(n: int, thickness: float = 0.12):
    """A NACA 00xx section of unit chord with a sharp trailing edge, 2n - 2 cosine-
    clustered vertices (the open-trailing-edge coefficient replaced by -0.1036)."""
    beta = torch.linspace(0, torch.pi, n, dtype=D)
    x = 0.5 * (1 - torch.cos(beta))
    yt = (
        5
        * thickness
        * (
            0.2969 * x.sqrt()
            - 0.1260 * x
            - 0.3516 * x**2
            + 0.2843 * x**3
            - 0.1036 * x**4
        )
    )
    upper = torch.stack([x, yt], -1)
    lower = torch.stack([x, -yt], -1).flip(0)[1:-1]
    return torch.cat([upper, lower])


def _brute_force_surface(q, points, cells):
    """The unsigned distance to the closest point of any cell, float64, over every
    (query, cell) pair: segments in two dimensions, triangles in three."""
    q, simplex = q.double(), points.double()[cells]
    if q.shape[-1] == 2:
        a, b = simplex[None, :, 0], simplex[None, :, 1]
        ab = b - a
        t = (((q[:, None] - a) * ab).sum(-1) / (ab * ab).sum(-1)).clamp(0, 1)
        return (q[:, None] - a - t[..., None] * ab).norm(dim=-1).min(-1)
    nq, nf = q.shape[0], simplex.shape[0]
    qq = q[:, None].expand(nq, nf, 3).reshape(-1, 3)
    tri = simplex[None].expand(nq, nf, 3, 3).reshape(-1, 3, 3)
    p = (_closest_point_weights(qq, tri)[..., None] * tri).sum(1)
    return (qq - p).norm(dim=-1).reshape(nq, nf).min(-1)


def _sphere(subdivisions: int, radius: float = 1.0) -> Mesh:
    m = sphere_icosahedral.load(radius=radius, subdivisions=subdivisions)
    return _oriented(Mesh(points=m.points.double(), cells=m.cells), lambda c: c)


def _circle(n: int) -> Mesh:
    m = circle_2d.load(n_points=n)
    return _oriented(Mesh(points=m.points.double(), cells=m.cells), lambda c: c)


def _torus(n_major: int = 48, n_minor: int = 24) -> Mesh:
    m = torus.load(n_major=n_major, n_minor=n_minor)

    def outward(c):
        ring = torch.nn.functional.normalize(
            c * torch.tensor([1.0, 1.0, 0.0], dtype=c.dtype), dim=-1
        )
        return c - ring

    return _oriented(Mesh(points=m.points.double(), cells=m.cells), outward)


def _near_wall(dim: int, n: int, seed: int, band: float, radius: float = 1.0):
    g = torch.Generator().manual_seed(seed)
    dirs = torch.nn.functional.normalize(
        torch.randn(n, dim, dtype=D, generator=g), dim=-1
    )
    return dirs * (radius + band * (2 * torch.rand(n, 1, dtype=D, generator=g) - 1))


def _geometry(q, mesh):
    return exact_sdf(q, mesh.points, mesh.cells)


@pytest.mark.parametrize("dim", [2, 3])
def test_exact_geometry_matches_the_brute_force(dim):
    """|d| is the distance to the closest point of any cell (float64 brute force over
    every pair): exact in two dimensions, to float32 resolution in three, where the
    BVH searches in float32 (the closest cell differs only at a float32 tie); u is
    a unit vector and q - |d| u lies on the closest cell."""
    g = torch.Generator().manual_seed(dim)
    pts, nrm = cloud(150, dim, seed=dim)
    mesh = soup(pts, nrm)
    q = torch.randn(400, dim, dtype=D, generator=g) * 3.0
    d, u = _geometry(q, mesh)
    ref = _brute_force_surface(q, mesh.points, mesh.cells).values
    assert d.shape == (400,) and u.shape == (400, dim) and d.dtype == D
    tol = 0.0 if dim == 2 else 8 * torch.finfo(torch.float32).eps * 3.0
    assert (d.abs() - ref).abs().max() <= tol + 1e-14
    assert torch.allclose(u.norm(dim=-1), torch.ones(400, dtype=D), atol=1e-12)
    foot = q - d.abs()[:, None] * u
    assert (
        _brute_force_surface(foot, mesh.points, mesh.cells).values.max() <= tol + 1e-12
    )


@pytest.mark.parametrize("dim", [2, 3])
def test_sign_and_direction_on_a_closed_surface(dim):
    """A circle / sphere with outward normals: d > 0 outside and d < 0 inside beyond
    the mesh's own deviation from the curved surface, u radial (outward outside,
    inward inside)."""
    mesh = _circle(256) if dim == 2 else _sphere(4)
    q = _near_wall(dim, 3000, 0, band=0.2)
    d, u = _geometry(q, mesh)
    r = q.norm(dim=-1)
    off = (r - 1).abs() > 2e-3  # beyond the inscribed polygon's sagitta
    assert (torch.sign(d[off]) == torch.sign(r - 1)[off]).all()
    radial = (u * q / r[:, None]).sum(-1)
    assert (radial[off] * torch.sign(r - 1)[off] > 0.99).all()


def test_convergence_is_second_order_under_refinement():
    """The circle at 64, 128, 256 segments and the sphere at icosahedral
    subdivisions 2, 3, 4 (h halving each step): the largest |d - d_exact| over a
    near-wall band falls by at least 3.5x per halving, and the largest deviation of
    the direction (on the side the mesh puts the query) from the radial one by at
    least 1.7x (O(h), the cells' normals against the surface's)."""
    for dim, meshes in (
        (2, [_circle(n) for n in (64, 128, 256)]),
        (3, [_sphere(s) for s in (2, 3, 4)]),
    ):
        q = _near_wall(dim, 4000, 1, band=0.05)
        r = q.norm(dim=-1)
        errs, dir_errs = [], []
        for mesh in meshes:
            d, u = _geometry(q, mesh)
            errs.append((d - (r - 1)).abs().max())
            dir_errs.append(
                (torch.sign(d)[:, None] * u - q / r[:, None]).norm(dim=-1).max()
            )
        for coarse, fine in zip(errs, errs[1:]):
            assert coarse / fine >= 3.5, (dim, errs)
        for coarse, fine in zip(dir_errs, dir_errs[1:]):
            assert coarse / fine >= 1.7, (dim, dir_errs)


@pytest.mark.parametrize("dim", [2, 3])
def test_exact_geometry_is_covariant_and_scales(dim):
    """Under a rotation, a reflection (cells re-oriented so the normals follow it), a
    translation and a rescale: d is invariant (scales with the geometry) and u
    covariant, to float rounding."""
    mesh = _polygon(_naca(80)) if dim == 2 else _torus()
    g = torch.Generator().manual_seed(2)
    q = torch.randn(600, dim, dtype=D, generator=g) * 0.6 + (0.5 if dim == 2 else 0.0)
    d, u = _geometry(q, mesh)
    for improper in (False, True):
        R = orthogonal(dim, 3, improper)
        t = torch.arange(1, dim + 1, dtype=D)
        moved = transform(mesh, R, t, scale=2.5)
        d2, u2 = _geometry(2.5 * q @ R.T + t, moved)
        tol = (
            1e-12 if dim == 2 else 1e-5
        )  # the float32 search's ties in three dimensions
        assert torch.allclose(d2, 2.5 * d, atol=tol), improper
        assert torch.allclose(u2, u @ R.T, atol=tol), improper


@pytest.mark.parametrize("dim", [2, 3])
def test_exact_geometry_is_invariant_to_splitting_the_surface(dim):
    """The same surface split into two named pieces (each with its own copy of the
    seam's vertices), and with every cell split (segments in two, triangles in four
    by their midpoints), gives the same d and u to rounding: the geometry is a
    function of the surface, not of its cells or of how it is stored."""
    mesh = _polygon(_naca(60)) if dim == 2 else _torus(24, 12)
    g = torch.Generator().manual_seed(4)
    q = torch.randn(800, dim, dtype=D, generator=g) * 0.5 + (0.5 if dim == 2 else 0.0)
    d, u = _geometry(q, mesh)
    ### two pieces, re-indexed separately and concatenated
    half = mesh.n_cells // 2
    pieces = [
        mesh.slice_cells(torch.arange(half)),
        mesh.slice_cells(torch.arange(half, mesh.n_cells)),
    ]
    pieces = [p.slice_points(torch.unique(p.cells)) for p in pieces]
    pts = torch.cat([p.points for p in pieces])
    cells = torch.cat([pieces[0].cells, pieces[1].cells + pieces[0].n_points])
    d2, u2 = exact_sdf(q, pts, cells)
    assert torch.allclose(d2, d, atol=1e-12, rtol=0) and torch.allclose(
        u2, u, atol=1e-12, rtol=0
    )
    ### every cell split
    d3, u3 = _geometry(q, mesh.subdivide(levels=1, filter="linear"))
    tol = 1e-12 if dim == 2 else 1e-5
    assert torch.allclose(d3, d, atol=tol, rtol=0) and torch.allclose(
        u3, u, atol=tol, rtol=0
    )


def test_sharp_trailing_edge_thin_plate_and_concave_corner():
    """Polygons where a single cell's normal gives the wrong sign: the wake behind a
    sharp trailing edge (NACA 0012), a thin plate (a 1e-3-thick rectangle) and the
    concave corner of an L; d equals the exact polygon SDF everywhere."""
    plate = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1e-3], [0.0, 1e-3]], dtype=D)
    ell = torch.tensor(
        [[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 1.0], [1.0, 2.0], [0.0, 2.0]],
        dtype=D,
    )
    g = torch.Generator().manual_seed(5)
    for verts, centre in (
        (_naca(100), (1.0, 0.0)),
        (plate, (1.0, 5e-4)),
        (ell, (1.0, 1.0)),
    ):
        mesh = _polygon(verts)
        c = torch.tensor(centre, dtype=D)
        q = torch.cat(
            [
                c + 0.05 * torch.randn(2000, 2, dtype=D, generator=g),
                c + torch.randn(2000, 2, dtype=D, generator=g),
            ]
        )
        d, _ = _geometry(q, mesh)
        assert torch.allclose(d, _polygon_sdf(q, verts), atol=1e-12, rtol=0)


def test_cube_edges_and_corners_and_torus_saddles():
    """The exact box SDF at a cube's edges and corners (the vertex and edge pseudo-
    normals), and the sign around a torus, whose inner vertices are saddles."""
    cube = cube_surface.load()
    g = torch.Generator().manual_seed(6)
    q = torch.randn(5000, 3, dtype=D, generator=g) * 0.8
    d, _ = _geometry(q, Mesh(points=cube.points.double(), cells=cube.cells))
    qq = q.abs() - 0.5
    exact = qq.clamp_min(0).norm(dim=-1) + qq.amax(-1).clamp_max(0)
    assert torch.allclose(d, exact, atol=8 * torch.finfo(torch.float32).eps, rtol=0)
    tor = _torus()
    q = torch.randn(5000, 3, dtype=D, generator=g) * torch.tensor(
        [1.2, 1.2, 0.4], dtype=D
    )
    d, _ = _geometry(q, tor)
    exact = ((q[:, :2].norm(dim=-1) - 1.0) ** 2 + q[:, 2] ** 2).sqrt() - 0.3
    off = exact.abs() > 5e-3  # beyond the polyhedral torus's deviation
    assert (torch.sign(d[off]) == torch.sign(exact[off])).all()


@pytest.mark.parametrize("dim", [2, 3])
def test_on_wall_queries_take_the_least_norm_normal(dim):
    """Queries exactly at the vertices (and at cell centroids) of a circle / sphere:
    d = 0 and u is the least-norm point of the hull of the unit normals of the cells
    holding the query, radial outward: the cell's normal at a centroid, the
    midpoint of the two segments' normals at a polygon's vertex (|u| = cos(pi / n))."""
    mesh = _circle(64) if dim == 2 else _sphere(3)
    for q in (mesh.points, mesh.cell_centroids):
        d, u = _geometry(q, mesh)
        assert (d.abs() <= 1e-15).all()
        radial = (u * torch.nn.functional.normalize(q, dim=-1)).sum(-1)
        assert (radial > 0.99).all() and (u.norm(dim=-1) <= 1 + 1e-15).all()
    d, u = _geometry(mesh.points, mesh)
    assert torch.equal(d, torch.zeros_like(d))
    assert (u.norm(dim=-1) < 1 - 1e-6).all()  # the hull of distinct normals
    _, u = _geometry(mesh.cell_centroids, mesh)
    assert torch.allclose(u, mesh.cell_normals, atol=1e-15, rtol=0)
    if dim == 2:
        _, u = _geometry(mesh.points, mesh)
        n = mesh.cell_normals
        at = torch.zeros_like(u).index_add(
            0, mesh.cells.reshape(-1), n.repeat_interleave(2, 0)
        )
        assert torch.allclose(u, 0.5 * at, atol=1e-15, rtol=0)
        assert torch.allclose(
            u.norm(dim=-1), torch.full_like(d, math.cos(math.pi / 64)), atol=1e-15
        )


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("shift", [0.0, 1.0, 100.0])
def test_exact_near_a_translated_wall_in_float32(dim, shift):
    """Regression (AirfRANS posed inference, 2026-10-03): near-wall float32 queries
    1e-5 to 1e-3 off a unit circle / sphere translated by 0, 1 and 100 radii: |d|
    equals the float64 brute-force distance to the same float32 surface to within
    a few float32 ulp of the coordinates, and the sign is right wherever the offset
    exceeds that resolution (the exact sign of the convex polygon / polyhedron)."""
    mesh = _circle(2000) if dim == 2 else _sphere(4)
    g = torch.Generator().manual_seed(7)
    dirs = torch.nn.functional.normalize(
        torch.randn(1000, dim, dtype=D, generator=g), dim=-1
    )
    side = torch.where(torch.rand(1000, dtype=D, generator=g) < 0.5, 1.0, -1.0)
    off = torch.tensor([1e-5, 1e-4, 1e-3], dtype=D)[
        torch.randint(3, (1000,), generator=g)
    ]
    t = torch.full((dim,), shift, dtype=D)
    pts = (mesh.points + t).float()
    q = (dirs * (1 + side * off)[:, None] + t).float()
    d, u = exact_sdf(q, pts, mesh.cells)
    assert d.dtype == torch.float32
    ref = _brute_force_surface(q, pts, mesh.cells).values
    ulp = torch.finfo(torch.float32).eps * (1.0 + shift)
    assert (d.double().abs() - ref).abs().max() <= 4 * ulp
    ### the mesh is convex: a query is outside iff it is above some cell's plane
    simplex = pts.double()[mesh.cells]
    planes = (
        (q.double()[:, None] - simplex[None, :, 0]) * mesh.cell_normals.double()[None]
    ).sum(-1)
    outside = planes.amax(-1) > 0
    resolved = ref > 8 * ulp
    assert (
        torch.where(outside, 1.0, -1.0)[resolved] == torch.sign(d.double())[resolved]
    ).all()
    assert torch.allclose(u.norm(dim=-1), torch.ones_like(d), atol=1e-6)


@pytest.mark.parametrize("dim", [2, 3])
def test_exact_geometry_is_differentiable(dim):
    """The gradient of d in the query is sign(d) u (the envelope theorem), and the
    vertex positions receive a finite gradient."""
    mesh = _circle(64) if dim == 2 else _sphere(2)
    q = _near_wall(dim, 200, 8, band=0.3).requires_grad_(True)
    pts = mesh.points.clone().requires_grad_(True)
    d, u = exact_sdf(q, pts, mesh.cells)
    d.sum().backward()
    assert torch.allclose(
        q.grad, torch.sign(d.detach())[:, None] * u.detach(), atol=1e-12
    )
    assert (
        pts.grad is not None
        and torch.isfinite(pts.grad).all()
        and pts.grad.abs().sum() > 0
    )


def test_two_dimensional_search_chunks_over_the_queries():
    """More queries than one chunk of the 2-D search: the same answer as one block."""
    verts = _naca(50)
    g = torch.Generator().manual_seed(9)
    q = torch.randn(
        2 * (_cell_tree._SEARCH_PAIRS // verts.shape[0]) + 17, 2, dtype=D, generator=g
    )
    d, _ = _geometry(q, _polygon(verts))
    assert torch.allclose(d, _polygon_sdf(q, verts), atol=1e-12, rtol=0)


def test_exact_geometry_refuses_other_dimensions():
    pts, nrm = cloud(10, 4)
    mesh = soup(pts, nrm)
    with pytest.raises(NotImplementedError, match="two and three dimensions"):
        exact_sdf(pts, mesh.points, mesh.cells)


def _flipped(mesh: Mesh) -> Mesh:
    """``mesh`` with every cell's last two vertices swapped (each normal reversed)."""
    d = mesh.cells.shape[-1]
    return Mesh(points=mesh.points, cells=mesh.cells[:, [*range(d - 2), d - 1, d - 2]])


@pytest.mark.parametrize("dim", [2, 3])
def test_the_sign_follows_the_surface_s_orientation(dim):
    """The distance is positive on the side the surface's normals point to:
    reversing every cell reverses the sign of the distance and leaves the direction
    (from the surface to the point) as it is, to rounding."""
    surface = _polygon(_naca(200)) if dim == 2 else _sphere(3)
    q = torch.randn(80, dim, dtype=D, generator=torch.Generator().manual_seed(3)) * 0.7
    d, u = _geometry(q, surface)
    d2, u2 = _geometry(q, _flipped(surface))
    ### (to rounding: each closest point is rebuilt from its cell's vertices in the
    ### other order)
    assert torch.allclose(d2, -d, atol=1e-14, rtol=0) and torch.allclose(
        u2, u, atol=1e-14, rtol=0
    )


# ---------------------------------------------------------------------------
# The sign: the generalized winding number
# ---------------------------------------------------------------------------


def _exact_winding(q, points, cells):
    """The generalized winding number of the oriented cells at each query, summed
    exactly over every (query, cell) pair in float64: the signed solid angles over 4
    pi (Van Oosterom and Strackee 1983), the signed subtended angles over 2 pi in
    two dimensions; 0 outside a closed surface with outward normals, 1 inside."""
    q, s = q.double(), points.double()[cells]
    out = []
    for x in q.split(256):
        r = s[None] - x[:, None, None]  # (n, F, d, d)
        if q.shape[-1] == 2:
            a, b = r[:, :, 0], r[:, :, 1]
            theta = torch.atan2(
                a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0], (a * b).sum(-1)
            )
            out.append(-theta.sum(-1) / (2 * math.pi))
            continue
        a, b, c = r.unbind(2)
        la, lb, lc = a.norm(dim=-1), b.norm(dim=-1), c.norm(dim=-1)
        num = (a * torch.linalg.cross(b, c, dim=-1)).sum(-1)
        den = (
            la * lb * lc
            + (a * b).sum(-1) * lc
            + (a * c).sum(-1) * lb
            + (b * c).sum(-1) * la
        )
        out.append(2 * torch.atan2(num, den).sum(-1) / (4 * math.pi))
    return torch.cat(out)


def _exact_sign(q, mesh):
    """sign(sigma / 2 - w) with the exact winding number and sigma the sign of the
    enclosed signed volume, +1 at zero."""
    n, c = mesh.cell_normals.double(), mesh.cell_centroids.double()
    a = mesh.cell_areas.double()
    g = (a[:, None] * c).sum(0) / a.sum()
    sigma = 1.0 if float((a[:, None] * n * (c - g)).sum()) >= 0 else -1.0
    w = _exact_winding(q, mesh.points, mesh.cells)
    return torch.where(0.5 * sigma - w >= 0, 1.0, -1.0).to(D)


def _tree_winding(q, mesh):
    """The winding number the search evaluates on its BVH (far field at
    ``WINDING_NUMBER_BETA``, exact solid angles at the near leaves)."""
    pts = mesh.points.double()
    cells, parity = _predicates._canonical_cells(
        pts, _predicates._weld(pts)[mesh.cells]
    )
    s, par = pts[cells], parity.to(D)
    e = s[:, 1:] - s[:, :1]
    area = par[:, None] * 0.5 * torch.linalg.cross(e[:, 0], e[:, 1], dim=-1)
    tree, order, s_leaf = _cell_tree._preorder_cell_tree(s)
    w = _winding._packed_winding_sum(q.double(), s_leaf, par[order], area[order], tree)
    return w / (4 * torch.pi)


def _finned_cube() -> Mesh:
    """The unit cube (outward normals) with a fin on its +x face: a 1 x 0.5 square
    in the plane y = 0, stored as two coincident sheets of opposite orientation (a
    zero-thickness fold, as two roles welded back to back)."""
    cube = cube_surface.load()
    fin = torch.tensor(
        [[0.5, 0, -0.25], [1.5, 0, -0.25], [1.5, 0, 0.25], [0.5, 0, 0.25]], dtype=D
    )
    sheet = torch.tensor([[0, 1, 2], [0, 2, 3]]) + cube.n_points
    return Mesh(
        points=torch.cat([cube.points.double(), fin]),
        cells=torch.cat([cube.cells, sheet, sheet[:, [0, 2, 1]]]),
    )


_DEVICES = [
    "cpu",
    pytest.param(
        "cuda",
        marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA"),
    ),
]


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_a_fold_is_signed_by_the_region_around_it(dtype, device):
    """Both sides of a zero-thickness fin outside a cube are fluid: d > 0 there, in
    float32 and float64, on CPU and CUDA (its two sheets cancel in the winding
    number; a local sign, a pseudo-normal or the closest cell's own normal, is
    decided by which of the two coincident cells comes first)."""
    mesh = _finned_cube()
    g = torch.Generator().manual_seed(0)
    n = 600
    off = torch.tensor([1e-4, 1e-2, 0.1], dtype=D)[torch.randint(3, (n,), generator=g)]
    side = torch.where(torch.rand(n, dtype=D, generator=g) < 0.5, 1.0, -1.0)
    q = torch.stack(
        [
            0.7 + 0.7 * torch.rand(n, dtype=D, generator=g),
            side * off,
            0.4 * torch.rand(n, dtype=D, generator=g) - 0.2,
        ],
        -1,
    )
    d, _ = exact_sdf(
        q.to(device, dtype), mesh.points.to(device, dtype), mesh.cells.to(device)
    )
    assert d.dtype == dtype and (d > 0).all()
    assert torch.allclose(d.double().cpu(), off, rtol=1e-5, atol=0)


@pytest.mark.parametrize("device", _DEVICES)
def test_a_winding_flip_negates_the_distance_bitwise(device):
    """Reversing every cell of the finned cube negates d bitwise at every query, on
    CPU and CUDA, those in the fin's plane beyond its free edge included (where the
    fin's edge pseudo-normal cancels and a local test reads a zero)."""
    mesh = _finned_cube()
    g = torch.Generator().manual_seed(1)
    beyond = torch.stack(
        [
            1.6 + torch.rand(200, dtype=D, generator=g),
            torch.zeros(200, dtype=D),
            0.4 * torch.rand(200, dtype=D, generator=g) - 0.2,
        ],
        -1,
    )
    q = torch.cat([torch.randn(1000, 3, dtype=D, generator=g), beyond]).to(device)
    d, _ = _geometry(q, mesh.to(device))
    d2, _ = _geometry(q, _flipped(mesh).to(device))
    assert (d != 0).all() and torch.equal(d2, -d)


def test_an_open_surface_signs_its_far_field_consistently():
    """A cube with its top face removed (outward normals): every query far from it,
    above the hole included, is outside (d > 0); a local sign reads the rim's side
    walls there and calls the points above the hole inside."""
    cube = cube_surface.load().subdivide(levels=2, filter="linear")
    keep = cube.cell_centroids[:, 2] < 0.49
    mesh = Mesh(points=cube.points.double(), cells=cube.cells[keep])
    g = torch.Generator().manual_seed(2)
    dirs = torch.nn.functional.normalize(
        torch.randn(2000, 3, dtype=D, generator=g), dim=-1
    )
    q = dirs * (3 + 27 * torch.rand(2000, 1, dtype=D, generator=g))
    q = torch.cat([q, torch.tensor([[0.0, 0.0, 5.0], [0.1, -0.2, 3.0]], dtype=D)])
    d, _ = _geometry(q, mesh)
    assert (d > 0).all()


@pytest.mark.parametrize("shape", ["naca", "sphere", "torus"])
def test_closed_surfaces_take_the_sign_of_the_exact_winding_number(shape):
    """On watertight surfaces (a NACA 0012, a sphere, a torus), natively, in a
    generic pose, with every cell split and with every cell doubled, d takes
    sign(sigma / 2 - w) of the exact winding number at every query off the surface,
    so the inside-outside sign of the closed surface; in three dimensions the
    winding number of the BVH is within its far-field error of the exact one (a
    tenth of the margin of a closed surface, per copy of each cell)."""
    g = torch.Generator().manual_seed(3)
    if shape == "naca":
        mesh = _polygon(_naca(100))
        q = torch.randn(1500, 2, dtype=D, generator=g) * 0.6 + torch.tensor(
            [0.5, 0.0], dtype=D
        )
    elif shape == "sphere":
        mesh = _sphere(3)
        q = _near_wall(3, 1500, 3, band=0.5)
    else:
        mesh = _torus()
        q = torch.randn(1500, 3, dtype=D, generator=g) * torch.tensor(
            [1.5, 1.5, 0.5], dtype=D
        )
    dim = q.shape[-1]
    R, t = orthogonal(dim, 5), torch.arange(1, dim + 1, dtype=D) / 3
    doubled = Mesh(points=mesh.points, cells=mesh.cells.repeat_interleave(2, 0))
    for m, x, copies in (
        (mesh, q, 1),
        (transform(mesh, R, t), q @ R.T + t, 1),
        (mesh.subdivide(levels=1, filter="linear"), q, 1),
        (doubled, q, 2),
    ):
        d, _ = _geometry(x, m)
        off = _brute_force_surface(x, m.points, m.cells).values > 1e-9
        assert torch.equal(torch.sign(d)[off], _exact_sign(x, m)[off])
        if dim == 3:
            w = _exact_winding(x, m.points, m.cells)
            assert (_tree_winding(x, m) - w).abs().max() < 0.05 * copies


# ---------------------------------------------------------------------------
# Closest-feature ties (the medial surface)
# ---------------------------------------------------------------------------


def _gradient(d, u):
    """The gradient direction, sign(d) u with sign(0) = +1."""
    return torch.where((d >= 0)[:, None], u, -u)


def _box_ties(dim: int):
    """The unit square / the cube [-1/2, 1/2]^3 (outward normals) and points
    inside it equidistant from two of its sides, from three (the cube's corner
    diagonals) and from all of them (the centre), with the least-norm gradient of
    the signed distance at each: the mean of the tied sides' outward unit normals,
    zero at the centre."""
    a = torch.linspace(0.05, 0.45, 9, dtype=D) - 0.5
    z = torch.zeros_like(a)
    if dim == 2:
        mesh = _polygon(
            torch.tensor([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]], dtype=D)
        )
        q = torch.cat(
            [
                torch.stack([a, a], -1),
                torch.stack([-a, a], -1),
                torch.stack([a, -a], -1),
                torch.stack([-a, -a], -1),
            ]
        )
        nu = 0.5 * torch.sign(q)
    else:
        cube = cube_surface.load()
        mesh = Mesh(points=cube.points.double(), cells=cube.cells)
        q = torch.cat(
            [
                torch.stack([a, a, z], -1),
                torch.stack([-a, z, a], -1),
                torch.stack([z, a, -a], -1),
                torch.stack([a, a, a], -1),
                torch.stack([-a, a, -a], -1),
            ]
        )
        nu = torch.sign(q) / (q != 0).sum(-1, keepdim=True)
    return (
        mesh,
        torch.cat([q, q.new_zeros(1, dim)]),
        torch.cat([nu, nu.new_zeros(1, dim)]),
    )


def test_tie_direction_is_the_least_norm_gradient_and_independent_of_the_order():
    """On a square's diagonals each point is equidistant from two sides: sign(d) u is
    the least-norm element of the generalized gradient of the signed distance, the
    midpoint of the two sides' unit gradients, and zero at the centre, where all
    four cancel; it is bitwise the same under every renumbering of the vertices and
    every cyclic shift of the cell list (a first-found side would follow them)."""
    mesh, q, nu = _box_ties(2)
    d, u = _geometry(q, mesh)
    assert torch.equal(_gradient(d, u), nu)
    assert torch.equal(d, -(0.5 - q.abs().amax(-1)))
    g = torch.Generator().manual_seed(0)
    for _ in range(4):
        perm = torch.randperm(4, generator=g)
        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(4)
        d2, u2 = exact_sdf(q, mesh.points[perm], inv[mesh.cells])
        assert torch.equal(d2, d) and torch.equal(u2, u), perm
    for shift in range(1, 4):
        d2, u2 = exact_sdf(q, mesh.points, mesh.cells.roll(shift, 0))
        assert torch.equal(d2, d) and torch.equal(u2, u), shift


@pytest.mark.parametrize("dim", [2, 3])
def test_tie_direction_flips_exactly_with_the_winding(dim):
    """A box in a generic pose (its ties hold to rounding only): reversing every cell
    negates d and sign(d) u bitwise at every tie point, the same tie set read with
    the other orientation (a first-found feature, before, could change with the
    swapped vertex order)."""
    mesh, q, _ = _box_ties(dim)
    R, t = orthogonal(dim, 7), torch.arange(1, dim + 1, dtype=D) / 3
    moved, q = transform(mesh, R, t), q @ R.T + t
    d, u = _geometry(q, moved)
    d2, u2 = _geometry(q, _flipped(moved))
    assert torch.equal(d2, -d) and torch.equal(_gradient(d2, u2), -_gradient(d, u))


@pytest.mark.parametrize("dim", [2, 3])
def test_tie_direction_is_covariant_under_rigid_motions(dim):
    """Under random rotations, reflections and translations the tie points keep
    their tie set (it holds to the error bound of the distance, which covers the
    rounding of the moved coordinates), so sign(d) u moves with the map, to
    rounding, and is the least-norm gradient in every pose: the midpoint of two
    sides, the centroid of three at the cube's corner diagonals, zero at the
    centre."""
    mesh, q, nu = _box_ties(dim)
    d, u = _geometry(q, mesh)
    assert torch.allclose(_gradient(d, u), nu, atol=1e-12, rtol=0)
    g = torch.Generator().manual_seed(dim)
    for seed in range(3):
        for improper in (False, True):
            R, t = (
                orthogonal(dim, 10 + seed, improper),
                torch.randn(dim, dtype=D, generator=g),
            )
            d2, u2 = _geometry(q @ R.T + t, transform(mesh, R, t))
            assert torch.allclose(d2, d, atol=1e-12, rtol=0), (seed, improper)
            assert torch.allclose(_gradient(d2, u2), nu @ R.T, atol=1e-12, rtol=0), (
                seed,
                improper,
            )


def _box_wall(dim: int):
    """The unit square / the cube [-1/2, 1/2]^3 (outward normals), its sides split
    once, and points exactly on it: its vertices, its cells' edge midpoints and
    centroids, with the least-norm normal at each: the mean of the outward unit
    normals of the box's sides holding it (one inside a side, two on an edge of
    the box, three at a corner of the cube)."""
    if dim == 2:
        h = torch.tensor([-0.5, 0.0, 0.5, 0.5, 0.5, 0.0, -0.5, -0.5], dtype=D)
        mesh = _polygon(torch.stack([h, h.roll(-2)], -1))
        pairs = ((0, 1),)
    else:
        cube = cube_surface.load()
        mesh = Mesh(points=cube.points.double(), cells=cube.cells).subdivide(
            levels=1, filter="linear"
        )
        pairs = ((0, 1), (1, 2), (0, 2))
    edges = [mesh.points[mesh.cells[:, [a, b]]].mean(1) for a, b in pairs]
    q = torch.cat([mesh.points, mesh.cell_centroids, *edges])
    on = q.abs() == 0.5
    return mesh, q, torch.sign(q) * on / on.sum(-1, keepdim=True)


@pytest.mark.parametrize("dim", [2, 3])
def test_on_wall_direction_is_the_least_norm_normal_and_independent_of_the_order(dim):
    """On the surface, u is the least-norm point of the hull of the unit normals of
    the cells holding the query (the generalized gradient's rule with the normals
    for the directions): the cell's normal inside a side, the midpoint of two on an
    edge, the centroid of three at a cube's corner; it is bitwise the same under
    every renumbering of the vertices, shuffle of the cells and cyclic rotation of
    a triangle's vertices (a first-found feature's pseudo-normal would follow
    them)."""
    mesh, q, nu = _box_wall(dim)
    d, u = _geometry(q, mesh)
    assert torch.equal(d, torch.zeros_like(d))
    assert torch.allclose(u, nu, atol=1e-15, rtol=0)
    g = torch.Generator().manual_seed(dim)
    for _ in range(3):
        perm = torch.randperm(mesh.n_points, generator=g)
        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(mesh.n_points)
        cells = inv[mesh.cells][torch.randperm(mesh.n_cells, generator=g)]
        if dim == 3:
            cells = cells.roll(1, -1)
        d2, u2 = exact_sdf(q, mesh.points[perm], cells)
        assert torch.equal(d2, d) and torch.equal(u2, u)


@pytest.mark.parametrize("dim", [2, 3])
def test_on_wall_direction_flips_exactly_with_the_winding(dim):
    """A box in a generic pose, with points on it to rounding (at its vertices, on
    its cells' edges and inside them): reversing every cell negates u bitwise, the
    same cells holding each point read with the other orientation, and u is the
    posed least-norm normal."""
    mesh, q, nu = _box_wall(dim)
    R, t = orthogonal(dim, 7), torch.arange(1, dim + 1, dtype=D) / 3
    moved, q = transform(mesh, R, t), q @ R.T + t
    d, u = _geometry(q, moved)
    d2, u2 = _geometry(q, _flipped(moved))
    assert torch.equal(d, torch.zeros_like(d)) and torch.equal(d2, d)
    assert torch.equal(u2, -u)
    assert torch.allclose(u, nu @ R.T, atol=1e-12, rtol=0)


@pytest.mark.parametrize("surface", ["ellipse", "sphere", "torus"])
def test_on_wall_direction_in_float32_is_the_float64_one(surface):
    """Curved float32 surfaces in a generic pose, at their vertices and their cells'
    edge midpoints: on the surface, u is the least-norm normal that float64 copies
    of the same inputs give, to float32 rounding, never 0 where that one is not
    (the float32 normals are not unit in float64, and their least-norm point failed
    its own optimality test), and a winding flip negates it bitwise."""
    if surface == "ellipse":
        m = _circle(400)
        mesh = Mesh(points=m.points * torch.tensor([1.0, 0.12], dtype=D), cells=m.cells)
    else:
        mesh = _sphere(3) if surface == "sphere" else _torus()
    dim = mesh.points.shape[-1]
    R, t = orthogonal(dim, 7), torch.arange(1, dim + 1, dtype=D) / 3
    pts = (mesh.points @ R.T + t).float()
    q = torch.cat([pts, pts[mesh.cells[:, :2]].mean(1)])
    d, u = exact_sdf(q, pts, mesh.cells)
    d64, u64 = exact_sdf(q.double(), pts.double(), mesh.cells)
    wall = d64 == 0
    assert wall[: pts.shape[0]].all() and torch.equal(
        d[wall], torch.zeros_like(d[wall])
    )
    assert (u64[wall].norm(dim=-1) > 0.5).all() and (u[wall].norm(dim=-1) > 0.5).all()
    assert (u[wall].double() - u64[wall]).norm(dim=-1).max() <= 1e-5
    _, u2 = exact_sdf(q, pts, _flipped(Mesh(points=pts, cells=mesh.cells)).cells)
    assert torch.equal(u2[wall], -u[wall])


@pytest.mark.parametrize("surface", ["split", "icosphere"])
def test_on_wall_direction_is_a_function_of_the_input_mesh(surface):
    """Float32 meshes in a generic pose, at their vertices, edge midpoints and
    centroids: on the surface, u from the float32 inputs is bitwise u from float64
    copies of them after the cast. The normals are computed in float64 from the
    coordinates as given, whatever the queries' precision; float32 normals carried
    their rounding into the least-norm point, which amplifies it where a split face's
    two triangles have near-duplicate normals (``split``: a torus with its vertices
    moved by 1e-5 of their radius, so each quad's triangles are about 1e-5 apart,
    as on the lug's hex faces), and differed from float64's at 594 of 781 on-wall
    points of the icosphere."""
    if surface == "split":
        ### The torus in its winding before #2148 made its normals outward: the
        ### midpoints of cells[:, :2] are then quad sides, the points this test was
        ### written for; the new winding puts each quad's diagonal there
        ### (test_on_wall_direction_at_quad_diagonals).
        m = _flipped(torus.load(n_major=28, n_minor=12))
        g = torch.Generator().manual_seed(0)
        p = m.points.double() * (
            1 + 1e-5 * (2 * torch.rand(m.n_points, 1, dtype=D, generator=g) - 1)
        )
    else:
        m = sphere_icosahedral.load(radius=1.0, subdivisions=3)
        p = m.points.double()
    pts = (p @ orthogonal(3, 7).T + torch.arange(1, 4, dtype=D) / 3).float()
    q = torch.cat([pts, pts[m.cells[:, :2]].mean(1), pts[m.cells].mean(1)])
    d, u = exact_sdf(q, pts, m.cells)
    d64, u64 = exact_sdf(q.double(), pts.double(), m.cells)
    wall = (d == 0) & (d64 == 0)
    assert wall[: pts.shape[0]].all() and (u[wall].norm(dim=-1) > 0.9).all()
    assert torch.equal(u[wall], u64[wall].float())


def test_on_wall_direction_skips_zero_area_cells():
    """A zero-area triangle on an edge of a sphere has no normal: at the edge's
    vertices and its midpoint the on-wall rule reads the other cells holding the
    query, so d and u are bitwise those without it (its zero normal in the hull
    made u 0)."""
    mesh = _sphere(2)
    a, b = mesh.cells[0, :2].tolist()
    mid = 0.5 * (mesh.points[a] + mesh.points[b])
    pts = torch.cat([mesh.points, mid[None]])
    cells = torch.cat([mesh.cells, torch.tensor([[a, b, mesh.n_points]])])
    q = torch.stack([mesh.points[a], mesh.points[b], mid, 1.01 * mid])
    d, u = exact_sdf(q, pts, cells)
    d0, u0 = exact_sdf(q, pts, mesh.cells)
    assert torch.equal(d[:3], torch.zeros(3, dtype=D))
    assert torch.equal(d, d0) and torch.equal(u, u0)
    assert (u.norm(dim=-1) > 0.9).all()


def test_float32_tie_keeps_its_side_off_the_wall():
    """Float32 queries at a posed sphere's edge midpoints lie a few 1e-8 off the
    surface (the rounding of the midpoint), off it in float64: at a tie, |d| is the
    float64 distance, which float32 does not round to 0, so d keeps the side the
    winding number gives it (d = -0 lost it) and a winding flip negates sign(d) u."""
    mesh = _sphere(3)
    R, t = orthogonal(3, 7), torch.arange(1, 4, dtype=D) / 3
    moved = Mesh(points=(mesh.points @ R.T + t).float(), cells=mesh.cells)
    q = moved.points[mesh.cells[:, :2]].mean(1)
    d, u = _geometry(q, moved)
    d2, u2 = _geometry(q, _flipped(moved))
    assert torch.equal(_gradient(d2, u2), -_gradient(d, u))


@pytest.mark.parametrize("surface", ["sphere", "torus"])
def test_float32_points_within_the_bound_are_on_the_wall(surface):
    """Float32 meshes as built (an icosphere, a torus) in a generic pose, at their
    vertices, their cells' edge midpoints and centroids (on the surface to float32
    rounding: up to 7e-8 off it in float64, within the closest cell's error bound)
    and at centroids moved up to 1e-5 along the cell's normal: every point on the
    surface is on the wall, d = 0 and u the least-norm normal, never 0 (under a fixed
    cutoff of 1e-12 the midpoints were off it, with a float32 closest point equal to
    the query: u = 0); off the wall d keeps the side the point was moved to; a
    winding flip negates d and sign(d) u bitwise, and renumbering the vertices and
    reordering the cells leave d and u bitwise."""
    if surface == "sphere":
        m = sphere_icosahedral.load(radius=1.0, subdivisions=3)
    else:
        ### The winding before #2148, so that the midpoints of cells[:, :2] are quad
        ### sides, as this test was written for (test_on_wall_direction_at_quad_diagonals).
        m = _flipped(torus.load(n_major=48, n_minor=24))
    pts = (m.points.double() @ orthogonal(3, 7).T).float()
    tri = pts[m.cells].double()
    n = torch.nn.functional.normalize(
        torch.linalg.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=-1), dim=-1
    )
    g = torch.Generator().manual_seed(0)
    s = 1e-5 * (2 * torch.rand(m.n_cells, 1, dtype=D, generator=g) - 1)
    q = torch.cat(
        [
            pts,
            pts[m.cells[:, :2]].mean(1),
            pts[m.cells].mean(1),
            (tri.mean(1) + s * n).float(),
        ]
    )
    d, u = exact_sdf(q, pts, m.cells)
    wall, moved = d == 0, slice(q.shape[0] - m.n_cells, None)
    assert wall[: moved.start].all() and not wall[moved].all()
    assert (u[wall].norm(dim=-1) > 0.5).all()
    assert torch.equal(
        torch.sign(d[moved][~wall[moved]]), torch.sign(s[~wall[moved], 0]).float()
    )
    d2, u2 = exact_sdf(q, pts, _flipped(Mesh(points=pts, cells=m.cells)).cells)
    assert torch.equal(d2, -d) and torch.equal(_gradient(d2, u2), -_gradient(d, u))
    perm = torch.randperm(pts.shape[0], generator=g)
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(pts.shape[0])
    cells = inv[m.cells][torch.randperm(m.n_cells, generator=g)].roll(1, -1)
    d3, u3 = exact_sdf(q, pts[perm], cells)
    assert torch.equal(d3, d) and torch.equal(u3, u)


def _quad_diagonals(case: str):
    """Float32 queries at the midpoints of a posed torus's quad diagonals (the edge
    each quad's two triangles share, which near-duplicate normals meet across), the
    torus's points and its cells: the 48 x 24 torus (``renumbering``), or the split
    torus of test_on_wall_direction_is_a_function_of_the_input_mesh
    (``precision``)."""
    if case == "renumbering":
        m = torus.load(n_major=48, n_minor=24)
        pts = (m.points.double() @ orthogonal(3, 7).T).float()
    else:
        m = torus.load(n_major=28, n_minor=12)
        g = torch.Generator().manual_seed(0)
        p = m.points.double() * (
            1 + 1e-5 * (2 * torch.rand(m.n_points, 1, dtype=D, generator=g) - 1)
        )
        pts = (p @ orthogonal(3, 7).T + torch.arange(1, 4, dtype=D) / 3).float()
    ### torus.load writes each quad's two triangles at k and k + n_quads.
    n_quads = m.n_cells // 2
    a, b = m.cells[:n_quads], m.cells[n_quads:]
    shared = (a[:, :, None] == b[:, None, :]).any(-1)
    return pts[a[shared].view(n_quads, 2)].mean(1), pts, m.cells


@pytest.mark.parametrize("case", ["renumbering", "precision"])
def test_on_wall_direction_at_quad_diagonals(case):
    """At the midpoints of the torus's quad diagonals in a generic pose
    (:func:`_quad_diagonals`), the on-wall u is bitwise invariant to renumbering the
    vertices and reordering the cells (``renumbering``), and from float32 inputs it
    is bitwise u from float64 copies after the cast (``precision``). A quad is
    planar, so its two triangles' normals agree to rounding and are not certified
    distinct: taking one of them made u depend on the cells' order (at 51 of the
    1152 midpoints, by up to 1.79e-6) and, through the certificate, on the queries'
    precision (at 5 of the 35 on-wall midpoints of the split torus, by up to
    1.61e-6)."""
    q, pts, cells = _quad_diagonals(case)
    d, u = exact_sdf(q, pts, cells)
    if case == "renumbering":
        g = torch.Generator().manual_seed(0)
        perm = torch.randperm(pts.shape[0], generator=g)
        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(pts.shape[0])
        cells = inv[cells][torch.randperm(cells.shape[0], generator=g)].roll(1, -1)
        d3, u3 = exact_sdf(q, pts[perm], cells)
        assert torch.equal(d3, d) and torch.equal(u3, u)
    else:
        d64, u64 = exact_sdf(q.double(), pts.double(), cells)
        wall = (d == 0) & (d64 == 0)
        assert wall.any()
        assert torch.equal(u[wall], u64[wall].float())


@pytest.mark.parametrize("dim", [2, 3])
def test_float64_points_beyond_the_bound_are_off_the_wall(dim):
    """A posed float64 box, at its vertices, its cells' edge midpoints and
    centroids, and at its centroids moved along the outward normal by a quarter of
    the closest cell's error bound E and by four times it (E ~ 1e-14 here): the
    first are on the wall (d = 0, u the least-norm normal), the last off it, with d
    its distance and u the normal, since float64 resolves them (a fixed cutoff of
    1e-12 put every point within 1e-12 on the wall)."""
    mesh, q, nu = _box_wall(dim)
    R, t = orthogonal(dim, 7), torch.arange(1, dim + 1, dtype=D) / 3
    moved, q, nu = transform(mesh, R, t), q @ R.T + t, nu @ R.T
    rows = slice(mesh.n_points, mesh.n_points + mesh.n_cells)
    c, n = q[rows], nu[rows]
    u64, tri = torch.finfo(D).eps / 2, moved.points[moved.cells]
    origin = _predicates._centroid(tri)
    e = u64 * (c.norm(dim=-1) + tri.norm(dim=-1).amax(-1)) + _weights._BAND_UNITS[
        dim
    ] * u64 * ((c - origin).norm(dim=-1) + (tri - origin).norm(dim=-1).amax(-1))
    d, u = _geometry(
        torch.cat([q, c + e[:, None] / 4 * n, c + 4 * e[:, None] * n]), moved
    )
    k = q.shape[0] + c.shape[0]
    assert torch.equal(d[:k], torch.zeros_like(d[:k]))
    assert torch.allclose(u[: q.shape[0]], nu, atol=1e-12, rtol=0)
    assert ((d[k:] - 4 * e).abs() <= e).all()
    assert torch.allclose(u[k:], n, atol=1e-12, rtol=0)


def _least_norm(c, tol: float = 1e-14):
    """The point of the convex hull of each row of ``c`` ``(n, K, d)`` closest to
    the origin, by enumeration in float64: every face of at most d of the vectors
    gives its least-norm affine point (the KKT system), and the hull's is the
    least-norm one with non-negative weights that meets the optimality condition
    c_k . x >= |x|^2; zero when none does (the origin is in the hull)."""
    n, k, d = c.shape
    best, best_norm = c.new_zeros(n, d), c.new_full((n,), torch.inf)
    for m in range(1, min(k, d) + 1):
        for sub in itertools.combinations(range(k), m):
            p = c[:, list(sub)]
            a = c.new_zeros(n, m + 1, m + 1)
            a[:, :m, :m], a[:, :m, m], a[:, m, :m] = p @ p.mT, 1.0, 1.0
            rhs = c.new_zeros(n, m + 1)
            rhs[:, m] = 1.0
            sol, info = torch.linalg.solve_ex(a, rhs)
            lam = sol[:, :m]
            x = (lam[..., None] * p).sum(1)
            ok = (info == 0) & torch.isfinite(lam).all(-1) & (lam >= -tol).all(-1)
            ok &= ((c * x[:, None]).sum(-1) >= (x * x).sum(-1, keepdim=True) - tol).all(
                -1
            )
            ok &= x.norm(dim=-1) < best_norm
            best = torch.where(ok[:, None], x, best)
            best_norm = torch.where(ok, x.norm(dim=-1), best_norm)
    return best


@pytest.mark.parametrize("dtype", [D, torch.float32])
@pytest.mark.parametrize(
    "surface", ["sphere", "raw torus", "torus", "ellipse", "sliver"]
)
def test_tie_directions_take_the_exact_least_norm_point(surface, dtype, monkeypatch):
    """At the vertices and the cells' edge midpoints of curved surfaces in a generic
    pose (the raw torus with its stored winding, the sliver a sphere with a zero-area
    triangle on one edge), on the surface and off it (a float32 midpoint lies a few
    1e-8 off its edge), every tie's least-norm step returns the exact least-norm
    point of its directions (enumerated in float64), to 1e-6 from float64 inputs and
    1e-5 from float32 ones, and never 0 where that is not."""
    if surface == "ellipse":
        m = _circle(400)
        mesh = Mesh(points=m.points * torch.tensor([1.0, 0.12], dtype=D), cells=m.cells)
    elif surface == "raw torus":
        m = torus.load(n_major=48, n_minor=24)
        mesh = Mesh(points=m.points.double(), cells=m.cells)
    else:
        mesh = {
            "sphere": lambda: _sphere(3),
            "torus": _torus,
            "sliver": lambda: _sphere(2),
        }[surface]()
    pts, cells = mesh.points, mesh.cells
    if surface == "sliver":
        a, b = cells[0, :2].tolist()
        pts = torch.cat([pts, 0.5 * (pts[a] + pts[b])[None]])
        cells = torch.cat([cells, torch.tensor([[a, b, mesh.n_points]])])
    dim = pts.shape[-1]
    R, t = orthogonal(dim, 7), torch.arange(1, dim + 1, dtype=D) / 3
    pts = (pts @ R.T + t).to(dtype)
    q = torch.cat([pts[: mesh.n_points], pts[mesh.cells[:, :2]].mean(1)])
    calls = []
    min_norm = _search._min_norm_points

    def recorded(c):
        x = min_norm(c)
        calls.append((c, x))
        return x

    monkeypatch.setattr(_search, "_min_norm_points", recorded)
    exact_sdf(q, pts, cells)
    assert calls
    tol = 1e-6 if dtype == D else 1e-5
    for c, x in calls:
        exact = _least_norm(c)
        assert not ((x.norm(dim=-1) == 0) & (exact.norm(dim=-1) > 0)).any()
        assert (x - exact).norm(dim=-1).max() <= tol


@pytest.mark.parametrize("k, dim", [(2, 2), (3, 2), (2, 3), (3, 3), (6, 3)])
def test_least_norm_point_without_a_certified_candidate(k, dim):
    """Float32 unit vectors read in float64 are unit only to about 6e-8, beyond the
    optimality certificate's rounding slack, so often no candidate passes it. Where
    a candidate separates the origin from the hull, the routine returns the one
    with the least duality gap g = |x|^2 - min_k c_k . x, within sqrt(2 g) of the
    exact least-norm point; otherwise 0, exactly where the exact point is 0. Before,
    every row without a certified candidate gave 0."""
    g = torch.Generator().manual_seed(k * dim)
    c = torch.randn(200, 1, dim, generator=g) + 0.4 * torch.randn(
        200, k, dim, generator=g
    )
    c = torch.nn.functional.normalize(c, dim=-1).double()
    x, exact = _search._min_norm_points(c), _least_norm(c, tol=1e-12)
    assert torch.equal(x.norm(dim=-1) == 0, exact.norm(dim=-1) == 0)
    gap = (x * x).sum(-1) - (c * x[:, None]).sum(-1).amin(-1)
    assert ((x - exact).norm(dim=-1) <= (2 * gap.clamp_min(0)).sqrt() + 1e-12).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="compares CUDA with CPU")
@pytest.mark.parametrize("dim", [2, 3])
def test_tie_search_on_cuda_matches_cpu(dim):
    """The search on CUDA (the Warp traversal in three dimensions) finds the same
    tie sets as on CPU: at the box's tie points (posed, so the ties hold to
    rounding), on the box (vertices, edges, cells) and near a sphere / circle in
    float32, d and u agree to rounding and the same points have a short u."""
    R, t = orthogonal(dim, 7), torch.arange(1, dim + 1, dtype=D) / 3
    for mesh, q in (_box_ties(dim)[:2], _box_wall(dim)[:2]):
        moved, q = transform(mesh, R, t), q @ R.T + t
        d, u = _geometry(q, moved)
        d2, u2 = exact_sdf(q.cuda(), moved.points.cuda(), moved.cells.cuda())
        assert torch.allclose(d2.cpu(), d, atol=1e-12, rtol=0)
        assert torch.allclose(u2.cpu(), u, atol=1e-12, rtol=0)
    mesh = _circle(2000) if dim == 2 else _sphere(4)
    pts, q = mesh.points.float(), _near_wall(dim, 4000, 11, band=0.3).float()
    d, u = exact_sdf(q, pts, mesh.cells)
    d2, u2 = exact_sdf(q.cuda(), pts.cuda(), mesh.cells.cuda())
    assert torch.equal(torch.sign(d2.cpu()), torch.sign(d))
    assert torch.allclose(d2.cpu(), d, atol=1e-6, rtol=0)
    assert torch.allclose(u2.cpu(), u, atol=1e-5, rtol=0)
    assert torch.equal(u2.cpu().norm(dim=-1) < 1 - 1e-5, u.norm(dim=-1) < 1 - 1e-5)


# ---------------------------------------------------------------------------
# The function: validation, shapes, dtypes and gradients
# ---------------------------------------------------------------------------


def test_exact_signed_distance_validates_its_inputs():
    """A mesh of another codimension, queries of another dimension and a mesh in
    four dimensions are refused."""
    sphere = _sphere(1)
    curve = Mesh(points=sphere.points, cells=sphere.cells[:, :2])
    with pytest.raises(ValueError, match="codimension one"):
        exact_signed_distance(curve, torch.zeros(2, 3, dtype=D))
    with pytest.raises(ValueError, match="last dimension"):
        exact_signed_distance(sphere, torch.zeros(2, 2, dtype=D))
    pts, nrm = cloud(10, 4)
    with pytest.raises(NotImplementedError, match="two and three dimensions"):
        exact_signed_distance(soup(pts, nrm), pts)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")
def test_exact_signed_distance_refuses_a_mesh_on_another_device():
    with pytest.raises(ValueError, match="same device"):
        exact_signed_distance(_sphere(1), torch.zeros(2, 3, dtype=D, device="cuda"))


@pytest.mark.parametrize("dim", [2, 3])
def test_exact_signed_distance_keeps_leading_dimensions(dim):
    """Queries of shape (4, 5, d) give d (4, 5) and u (4, 5, d), bitwise the
    flattened queries' outputs, in the queries' dtype."""
    mesh = _circle(64) if dim == 2 else _sphere(2)
    q = _near_wall(dim, 20, 12, band=0.3)
    d, u = exact_signed_distance(mesh, q.reshape(4, 5, dim))
    d0, u0 = exact_signed_distance(mesh, q)
    assert d.shape == (4, 5) and u.shape == (4, 5, dim) and d.dtype == D
    assert torch.equal(d.reshape(-1), d0) and torch.equal(u.reshape(-1, dim), u0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_exact_signed_distance_of_half_precision_queries(dtype):
    """Half-precision queries are searched in float32 on the mesh rounded to their
    dtype, and the outputs come back in their dtype (rounded once from float64, so
    within a unit of the dtype of the float32 outputs)."""
    mesh = _sphere(2)
    q = _near_wall(3, 50, 13, band=0.3).to(dtype)
    d, u = exact_signed_distance(mesh, q)
    assert d.dtype == dtype and u.dtype == dtype and u.shape == q.shape
    rounded = Mesh(points=mesh.points.to(dtype).float(), cells=mesh.cells)
    d32, u32 = exact_signed_distance(rounded, q.float())
    eps = torch.finfo(dtype).eps
    assert torch.allclose(d.float(), d32, atol=eps, rtol=eps)
    assert torch.allclose(u.float(), u32, atol=1e-2, rtol=0)


@pytest.mark.parametrize("dim", [2, 3])
def test_exact_signed_distance_has_gradients_in_the_queries_and_the_mesh(dim):
    """Through the Mesh, the queries and the mesh's points receive the gradient of
    the distance: sign(d) u in the queries."""
    surface = _circle(64) if dim == 2 else _sphere(2)
    pts = surface.points.clone().requires_grad_(True)
    q = _near_wall(dim, 100, 14, band=0.3).requires_grad_(True)
    d, u = exact_signed_distance(Mesh(points=pts, cells=surface.cells), q)
    d.sum().backward()
    assert torch.allclose(q.grad, torch.sign(d.detach())[:, None] * u.detach())
    assert pts.grad is not None and bool(torch.isfinite(pts.grad).all())
    assert float(pts.grad.abs().sum()) > 0


def _frozen_closest_point_weights(query, tri):
    """``physicsnemo.nn.functional.geometry.sdf._closest_point_weights`` as the
    rounding analysis of ``_ERICSON_K`` reads it (the same operations in the same
    order)."""
    a, b, c = tri.unbind(dim=1)
    ab = b - a
    ac = c - a
    d1 = (ab * (query - a)).sum(-1)
    d2 = (ac * (query - a)).sum(-1)
    d3 = (ab * (query - b)).sum(-1)
    d4 = (ac * (query - b)).sum(-1)
    d5 = (ab * (query - c)).sum(-1)
    d6 = (ac * (query - c)).sum(-1)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2
    tiny = torch.finfo(query.dtype).tiny
    zero = torch.zeros_like(d1)
    one = torch.ones_like(d1)
    denom = (va + vb + vc).clamp(min=tiny)
    t_bc = (d4 - d3) / ((d4 - d3) + (d5 - d6)).clamp(min=tiny)
    t_ac = d2 / (d2 - d6).clamp(min=tiny)
    t_ab = d1 / (d1 - d3).clamp(min=tiny)
    weights = torch.stack([va / denom, vb / denom, vc / denom], dim=-1)
    regions = (
        ((va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0), (zero, 1.0 - t_bc, t_bc)),
        ((vb <= 0) & (d2 >= 0) & (d6 <= 0), (1.0 - t_ac, zero, t_ac)),
        ((d6 >= 0) & (d5 <= d6), (zero, zero, one)),
        ((vc <= 0) & (d1 >= 0) & (d3 <= 0), (1.0 - t_ab, t_ab, zero)),
        ((d3 >= 0) & (d4 <= d3), (zero, one, zero)),
        ((d1 <= 0) & (d2 <= 0), (one, zero, zero)),
    )
    for in_region, region_weights in regions:
        weights = torch.where(
            in_region.unsqueeze(-1), torch.stack(region_weights, dim=-1), weights
        )
    return weights


def test_ericson_weights_are_the_imported_ones():
    """The certified bounds analyse Ericson's weights as
    ``nn.functional.geometry.sdf._closest_point_weights`` computes them today: on
    random triangles, thin ones included, and queries in every region, it is bitwise
    a frozen copy, and ``_closest_weights`` keeps it bitwise at every pair whose
    point is not on an edge (recomputed there) and not guarded."""
    g = torch.Generator().manual_seed(0)
    tri = torch.randn(3000, 3, 3, dtype=D, generator=g)
    tri[:1000, 2] = tri[:1000, 0] + 0.5 * (tri[:1000, 1] - tri[:1000, 0])
    tri[:1000, 2] += 1e-7 * torch.randn(1000, 3, dtype=D, generator=g)
    q = torch.cat(
        [
            torch.randn(1500, 3, dtype=D, generator=g) * 2,
            (torch.rand(1500, 3, 1, dtype=D, generator=g) * tri[1500:]).sum(1),
        ]
    )
    w = _closest_point_weights(q, tri)
    assert torch.equal(
        w.view(torch.int64), _frozen_closest_point_weights(q, tri).view(torch.int64)
    )
    guarded = []
    accurate = _weights._accurate_weights

    def spy(q, s):
        guarded.append(q)
        return accurate(q, s)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(_weights, "_accurate_weights", spy)
        mine = _weights._closest_weights(q, tri)
    assert guarded and guarded[0].shape[0] < q.shape[0]
    edge = (w > 0).sum(-1) == 2
    assert bool(edge.any()) and bool((~edge).any())
    _, kappa, longest = _weights._thin_bounds(tri)
    dist = (q - (w[..., None] * tri).sum(1)).norm(dim=-1)
    unit = 2.0**-53 * (q.norm(dim=-1) + tri.norm(dim=-1).amax(-1))
    l2 = (q[:, None] - tri).square().sum(-1).amax(-1)
    unguarded = _weights._ericson_error(dist, kappa * l2, longest) < unit
    kept = ~edge & unguarded
    assert bool(kept.any())
    assert torch.equal(mine[kept].view(torch.int64), w[kept].view(torch.int64))


# ---------------------------------------------------------------------------
# The ball search's launches
# ---------------------------------------------------------------------------


def _run(q, points, cells):
    """d, u, their gradients in the queries and the vertices (fixed cotangents),
    every output of the search and the winding number."""
    seen = {}
    op, sign = _search._closest_features_op, _search._winding_sign

    def spy(*args):
        seen["search"] = op(*args)
        return seen["search"]

    def spy_sign(query, simplex, parity, sigma, w=None):
        seen["w"] = w
        return sign(query, simplex, parity, sigma, w)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(_search, "_closest_features_op", spy)
        mp.setattr(_search, "_winding_sign", spy_sign)
        q, points = q.clone().requires_grad_(), points.clone().requires_grad_()
        d, u = exact_sdf(q, points, cells)
        probe = torch.linspace(0, 1, u.numel(), dtype=u.dtype, device=u.device)
        (d.sum() + (u * probe.reshape(u.shape)).sum()).backward()
    return [d, u, q.grad, points.grad, *seen["search"], seen["w"]]


def _fan(n: int = 96):
    """A fan of ``n`` triangles around a hub, 50 queries above the hub (``n``
    candidates each, more than the 64 slots) and 400 elsewhere."""
    t = torch.arange(n, dtype=D) * (2 * torch.pi / n)
    rim = torch.stack([torch.cos(t), torch.sin(t), 0 * t], -1)
    points = torch.cat([rim.new_zeros(1, 3), rim])
    i = torch.arange(1, n + 1)
    cells = torch.stack([torch.zeros_like(i), i, i % n + 1], -1)
    g = torch.Generator().manual_seed(0)
    hub = torch.linspace(0.01, 0.5, 50, dtype=D)[:, None] * torch.tensor([0.0, 0, 1])
    q = torch.cat([hub, torch.rand(400, 3, dtype=D, generator=g) * 3 - 1.5])
    return q, points, cells


def _equal(a, b):
    """Bitwise equality, signed zeros and NaN payloads included."""
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    if a.is_floating_point():
        return torch.equal(a.view(torch.uint8), b.view(torch.uint8))
    return torch.equal(a, b)


def test_ball_search_in_blocks_of_rows_equals_one_launch():
    """The ball search launched over blocks of a few query rows, as it is from 2**25 =
    33,554,432 queries on (Warp refuses 2**31 candidate slots), gives every output of
    one launch bitwise, its second pass included: the hub queries of a fan have 96
    candidates each, more than the 64 slots."""
    q, points, cells = _fan()
    one = _run(q, points, cells)
    caps, launch = [], _cell_tree.wp.launch

    def spy(k, dim, *args, **kwargs):
        if k is _cell_tree._ball_candidates_kernel:
            caps.append((kwargs["inputs"][14], dim))  # (slots per row, rows)
        return launch(k, dim, *args, **kwargs)

    with pytest.MonkeyPatch.context() as mp:
        ### Blocks of 64 rows in the first pass, of 42 in the second.
        mp.setattr(_cell_tree, "_WARP_MAX_ELEMENTS", 2**12 + 1)
        mp.setattr(_cell_tree.wp, "launch", spy)
        blocks = _run(q, points, cells)
    assert caps == [(64, 64)] * 7 + [(64, 2), (96, 42), (96, 8)]
    for a, b in zip(one, blocks):
        assert _equal(a, b)


@pytest.mark.parametrize("case", ["list", "fallback"])
def test_ball_search_overflow_list_and_its_fallback_equal_the_default_slots(case):
    """With two candidate slots per query, the candidates past them go to the
    overflow list (``list``: a sphere's queries, whose few extra candidates fit in
    it) or, when it is full, to a second pass (``fallback``: a fan's hub queries,
    94 extra candidates each, against a list of 450 entries); every output is
    bitwise that of the default 64 slots."""
    if case == "list":
        q, mesh = _near_wall(3, 400, 15, band=0.5), _sphere(2)
        points, cells = mesh.points, mesh.cells
    else:
        q, points, cells = _fan()
    default = _run(q, points, cells)
    caps, pairs = [], []
    launch, ball = _cell_tree.wp.launch, _search._ball_candidates

    def spy(k, dim, *args, **kwargs):
        if k is _cell_tree._ball_candidates_kernel:
            caps.append(kwargs["inputs"][14])
        return launch(k, dim, *args, **kwargs)

    def spy_ball(*args, **kwargs):
        pairs.append(ball(*args, **kwargs))
        return pairs[-1]

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(_cell_tree, "_BALL_SLOTS", 2)
        mp.setattr(_cell_tree.wp, "launch", spy)
        mp.setattr(_search, "_ball_candidates", spy_ball)
        two = _run(q, points, cells)
    per_query = torch.bincount(pairs[0][0], minlength=q.shape[0])
    extra = int((per_query - 2).clamp_min(0).sum())
    assert extra > 0
    if case == "list":
        assert extra <= q.shape[0] and caps == [2]
    else:
        assert extra > q.shape[0] and caps == [2, int(per_query.max())]
    for a, b in zip(default, two):
        assert _equal(a, b)


@pytest.mark.cuda
@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")
def test_ball_search_past_2_31_slots():
    """2**25 + 64 queries, whose 64 candidate slots each pass Warp's 2**31-entry
    arrays: the last 64 queries' distances and directions equal those of the same
    queries alone, bitwise (each query's search, closest point and winding number are
    its own). Needs about 60 GiB of device memory (1.6-1.8 kB per query, measured on
    CPU at 2**20 queries): the 17.2 GB of slots, their masks and the candidates'
    closest points."""
    tail = 64
    if torch.cuda.mem_get_info()[0] < 96 * 2**30:
        pytest.skip("needs about 60 GiB of free device memory")
    sphere = _sphere(2)
    points, cells = sphere.points.cuda(), sphere.cells.cuda()
    g = torch.Generator(device="cuda").manual_seed(0)
    q = torch.rand(2**25 + tail, 3, generator=g, device="cuda", dtype=D) * 3 - 1.5
    whole = [t[-tail:] for t in exact_sdf(q, points, cells)]
    alone = exact_sdf(q[-tail:].clone(), points, cells)
    for a, b in zip(whole, alone):
        assert torch.equal(a, b)
