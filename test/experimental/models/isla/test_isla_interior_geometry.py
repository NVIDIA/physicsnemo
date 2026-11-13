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

"""The interior geometry of the prediction-point tokens.

For every prediction point the model derives its signed distance and direction
from the boundary surface itself (the exact closest point on the surface's cells,
signed by the closest feature's pseudo-normal, positive on the side the normals
point to); neither is an input, and neither depends on how the boundary tokens
were sampled from the surface. These tests pin the geometry against brute-force
and exact references, its sign convention and convergence, and the contracts of
the interior task.
"""

import itertools
import math

import pytest
import torch

from physicsnemo.experimental.models.isla import ISLA
from physicsnemo.experimental.models.isla import _geometry as geometry
from physicsnemo.experimental.models.isla import model as isla_model
from physicsnemo.experimental.models.isla._geometry import exact_boundary_geometry
from physicsnemo.mesh import Mesh
from physicsnemo.mesh.primitives.surfaces import cube_surface, sphere_icosahedral, torus
from physicsnemo.nn.functional.geometry.sdf import _closest_point_weights
from test.experimental.models.isla._geometry_cases import (
    KERNEL_DEVICES,
    _box_ties,
    _box_wall,
    _circle,
    _finned_cube,
    _interior,
    _naca,
    _near_wall,
    _polygon,
    _quad_diagonals,
    _renumbered,
    _sphere,
    _torus,
)
from test.experimental.models.isla._marks import requires_cuda
from test.experimental.models.isla._meshes import (
    cloud,
    live,
    orthogonal,
    outputs,
    soup,
    transform,
)

D = torch.float64
PV = {"p": {"rank": 0}, "v": {"rank": 1}}


WALL = {"wall": {}}


def _model(**kw):
    torch.manual_seed(0)
    kw.setdefault("boundary_schemas", WALL)
    return (
        ISLA(
            output_schema=PV,
            global_schema={"g": {"rank": 1}},
            hidden=32,
            n_layers=2,
            n_slices=8,
            interior=True,
            **kw,
        )
        .double()
        .eval()
    )


def test_interior_contracts():
    """Rotation and reflection covariance, translation invariance and geometric-scale
    invariance in the interior task, with the prediction-point directions computed
    inside; the distance channel is live, and so is the boundary read (``live``, at
    the default opening angle: the forest is built in the boundary's own frame and
    scale, so it is the same in every pose)."""
    pts, nrm, w, _ = cloud(200)
    mesh = soup(pts, nrm, w)
    drv = torch.tensor([0.3, 1.0, 0.2], dtype=D)
    q, _, _, _ = cloud(40, seed=1, scale=(4.0, 4.0, 4.0))
    m = live(_model())

    def call(R=None, t=None, k=1.0, q=q):
        R = torch.eye(3, dtype=D) if R is None else R
        t = torch.zeros(3, dtype=D) if t is None else t
        with torch.no_grad():
            return outputs(
                m(
                    boundary_meshes={"wall": transform(mesh, R, t, scale=k)},
                    prediction_points=k * q @ R.T + t,
                    global_data={"g": drv @ R.T},
                )
            )

    base = call()
    for improper in (False, True):
        R = orthogonal(3, 4, improper)
        out = call(R, torch.tensor([2.0, -5.0, 1.0], dtype=D), k=2.5)
        assert torch.allclose(out["p"], base["p"], atol=1e-11, rtol=0.0), improper
        assert torch.allclose(out["v"], base["v"] @ R.T, atol=1e-11, rtol=0.0), improper
    ### the distance channel is live: moving the queries off the boundary changes p
    far = call(q=q * 3.0)
    assert not torch.allclose(far["p"], base["p"], atol=1e-6)


def test_no_query_normals_argument():
    """The prediction-point direction is never an input."""
    pts, nrm, w, _ = cloud(20)
    m = _model()
    with pytest.raises(TypeError, match="query_normals"):
        m(
            boundary_meshes={"wall": soup(pts, nrm, w)},
            prediction_points=pts[:3],
            global_data={"g": nrm[0]},
            query_normals=nrm[:3],
        )


def test_fused_kernel_stays_available_with_the_built_in_distance():
    """The built-in distance enters the embedding, not the geometry region, so a model
    without rank-1 or rank-2 per-token inputs keeps the fused kernel."""
    m = ISLA(
        output_schema=PV,
        boundary_schemas=WALL,
        global_schema={"g": {"rank": 1}},
        hidden=32,
        n_layers=1,
        n_slices=8,
        interior=True,
    )
    assert m._n_query_features == 2 and m.qt.seed.in_features == 1 + 2
    assert not (m._n_block_vecs or m._n_block_tens)


# ---------------------------------------------------------------------------
# The exact geometry against the boundary surface (exact_boundary_geometry)
# ---------------------------------------------------------------------------


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


def _geometry(q, mesh):
    return exact_boundary_geometry(q, mesh.points, mesh.cells)


@pytest.mark.parametrize("dim", [2, 3])
def test_exact_geometry_matches_the_brute_force(dim):
    """|d| is the distance to the closest point of any cell (float64 brute force over
    every pair): exact in two dimensions, to float32 resolution in three, where the
    BVH searches in float32 (the closest cell differs only at a float32 tie); u is
    a unit vector and q - |d| u lies on the closest cell."""
    g = torch.Generator().manual_seed(dim)
    pts, nrm, _, _ = cloud(150, dim, seed=dim)
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


def test_convergence_is_second_order_under_refinement():
    """The circle at 64, 128, 256 segments and the sphere at icosahedral
    subdivisions 2, 3, 4 (h halving each step): the largest |d - d_exact| over a
    near-wall band falls by at least 3.5x per halving, and the largest deviation of
    the direction (on the side the mesh puts the query) from the radial one by at
    least 1.7x (O(h), the cells' normals against the surface's). On the finest mesh
    both are small, so d has the sign of r - 1 beyond 2e-3 of the wall and u is
    radial there to within 0.14 (outward outside, inward inside)."""
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
        ### measured 7.5e-5 and 1.2e-2 (2-D), 1.1e-3 and 4.6e-2 (3-D)
        assert errs[-1] < 2e-3 and dir_errs[-1] < 0.14, (dim, errs, dir_errs)


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


@pytest.mark.parametrize("kernel, device", KERNEL_DEVICES)
def test_exact_geometry_scales_at_any_length(kernel, device):
    """At lengths times 1e-12 and 1e12, d scales and u is unchanged, signs
    included: the winding number's float32 far-field moments (a length to the
    fifth power) are taken in a power-of-two unit of the surface, so they do not
    overflow and flip the signs at large lengths."""
    mesh = _torus()
    g = torch.Generator().manual_seed(2)
    q = (torch.randn(600, 3, dtype=D, generator=g) * 0.6).to(device)
    pts, cells = mesh.points.to(device), mesh.cells.to(device)
    d, u = exact_boundary_geometry(q, pts, cells, kernel=kernel)
    for s in (1e-12, 1e12):
        d2, u2 = exact_boundary_geometry(s * q, s * pts, cells, kernel=kernel)
        assert torch.allclose(d2 / s, d, atol=1e-5, rtol=0), s
        assert torch.allclose(u2, u, atol=1e-5, rtol=0), s


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
    ### two pieces, re-indexed separately, concatenated as the model concatenates them
    half = mesh.n_cells // 2
    pieces = [
        mesh.slice_cells(torch.arange(half)),
        mesh.slice_cells(torch.arange(half, mesh.n_cells)),
    ]
    pieces = [p.slice_points(torch.unique(p.cells)) for p in pieces]
    pts = torch.cat([p.points for p in pieces])
    cells = torch.cat([pieces[0].cells, pieces[1].cells + pieces[0].n_points])
    d2, u2 = exact_boundary_geometry(q, pts, cells)
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
        ### Pins the on-wall u at the vertices (a tie of two segments),
        ### relative on purpose (rtol 1e-5): circle_2d places its
        ### vertices in float32, so the segments' angle is pi / 64 only to about 1e-8
        ### relative (measured 2e-8), which is the mesh and not the geometry's roundoff
        assert torch.allclose(
            u.norm(dim=-1),
            torch.full_like(d, math.cos(math.pi / 64)),
            rtol=1e-5,
            atol=1e-15,
        )


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("shift", [0.0, 1.0, 100.0])
def test_exact_near_a_translated_wall_in_float32(dim, shift):
    """Near-wall float32 queries
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
    d, u = exact_boundary_geometry(q, pts, mesh.cells)
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


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_half_precision_query_above_a_small_triangle(dtype):
    """Above a triangle with edges of 1e-4, whose cross product underflows in
    float16, the query keeps the float32 distance (to a unit of its dtype) and a unit
    normal, where the float16 arithmetic gives d = 0 and u = 0."""
    points = torch.tensor([[0.0, 0.0, 0.0], [1e-4, 0.0, 0.0], [0.0, 1e-4, 0.0]])
    points, cells = points.to(dtype), torch.tensor([[0, 1, 2]])
    q = torch.tensor([[2.5e-5, 2.5e-5, 1e-3]], dtype=dtype)
    d, u = exact_boundary_geometry(q, points, cells)
    d32, u32 = exact_boundary_geometry(q.float(), points.float(), cells)
    eps = torch.finfo(dtype).eps
    assert torch.allclose(d.float(), d32, atol=0, rtol=eps)
    assert torch.equal(u.float(), u32) and float(u.float().norm()) == 1


@pytest.mark.parametrize("dim", [2, 3])
def test_exact_geometry_is_differentiable(dim):
    """The gradient of d in the query is sign(d) u (the envelope theorem), and the
    vertex positions receive a finite gradient."""
    mesh = _circle(64) if dim == 2 else _sphere(2)
    q = _near_wall(dim, 200, 8, band=0.3).requires_grad_(True)
    pts = mesh.points.clone().requires_grad_(True)
    d, u = exact_boundary_geometry(q, pts, mesh.cells)
    d.sum().backward()
    assert torch.allclose(
        q.grad, torch.sign(d.detach())[:, None] * u.detach(), atol=1e-12
    )
    assert (
        pts.grad is not None
        and torch.isfinite(pts.grad).all()
        and pts.grad.abs().sum() > 0
    )


def test_two_dimensional_search_chunks_over_the_queries(monkeypatch):
    """More queries than one chunk of the 2-D search: the same answer as one block."""
    monkeypatch.setattr(geometry, "_SEARCH_PAIRS", 2**12)
    verts = _naca(50)
    g = torch.Generator().manual_seed(9)
    q = torch.randn(
        2 * (geometry._SEARCH_PAIRS // verts.shape[0]) + 17, 2, dtype=D, generator=g
    )
    d, _ = _geometry(q, _polygon(verts))
    assert torch.allclose(d, _polygon_sdf(q, verts), atol=1e-12, rtol=0)


def test_exact_geometry_refuses_other_dimensions():
    """A 4-D surface is refused: the geometry is defined in two and three dimensions."""
    pts, nrm, _, _ = cloud(10, 4)
    mesh = soup(pts, nrm)
    with pytest.raises(NotImplementedError, match="two and three dimensions"):
        exact_boundary_geometry(pts, mesh.points, mesh.cells)


@pytest.mark.parametrize("dim", [2, 3])
def test_exact_geometry_refuses_a_boundary_with_no_cells(dim):
    """No cells, no surface: a ValueError up front, not an error deep in the
    search."""
    mesh = _circle(8) if dim == 2 else _sphere(0)
    with pytest.raises(ValueError, match="no cells"):
        exact_boundary_geometry(
            torch.zeros(4, dim, dtype=D), mesh.points, mesh.cells[:0]
        )


@pytest.mark.parametrize("dim", [2, 3])
def test_no_queries_give_empty_outputs(dim):
    """Zero queries give empty d and u in the queries' dtype, with a gradient path
    to the queries."""
    mesh = _circle(8) if dim == 2 else _sphere(0)
    q = torch.zeros(0, dim, dtype=torch.float32, requires_grad=True)
    d, u = exact_boundary_geometry(q, mesh.points.float(), mesh.cells)
    assert d.shape == (0,) and u.shape == (0, dim)
    assert d.dtype == u.dtype == q.dtype
    (d.sum() + u.sum()).backward()
    assert q.grad.shape == q.shape


# ---------------------------------------------------------------------------
# The model reads the full-resolution surface and draws its tokens from it
# ---------------------------------------------------------------------------


def _captured(m, **kw):
    """The boundary tokens and the built-in geometry the model's batched routine
    receives, and the outputs."""
    seen = {"d": [], "u": []}
    inner, geometry_of = m._case_inputs, isla_model.exact_boundary_geometry

    def spy(*args, **kwargs):
        c = inner(*args, **kwargs)
        seen["tokens"] = c.boundary[0].clone()
        return c

    def spy_geometry(*args, **kwargs):
        ### (each chunk of points measures its own)
        d, u = geometry_of(*args, **kwargs)
        seen["d"].append(d.clone())
        seen["u"].append(u.clone())
        return d, u

    m._case_inputs = spy
    isla_model.exact_boundary_geometry = spy_geometry
    try:
        with torch.no_grad():
            out = outputs(m(**kw))
    finally:
        del m._case_inputs
        isla_model.exact_boundary_geometry = geometry_of
    return seen["tokens"], torch.cat(seen["d"]), torch.cat(seen["u"]), out


def _airfoil_case(n: int = 400, nq: int = 300):
    surface = _polygon(_naca(n))
    g = torch.Generator().manual_seed(1)
    q = torch.tensor([0.5, 0.0], dtype=D) + torch.randn(
        nq, 2, dtype=D, generator=g
    ) * torch.tensor([0.6, 0.1], dtype=D)
    return surface, q, {"g": torch.tensor([1.0, 0.1], dtype=D)}


def _rel(a, b) -> float:
    return max(float((a[k] - b[k]).norm() / b[k].norm()) for k in b)


def test_builtin_geometry_is_the_surface_s_for_every_draw_of_the_tokens():
    """A 798-segment NACA 0012: whatever the token budget, and in training as in
    evaluation, the distance and direction the model computes are bitwise the exact
    geometry of the full surface; only the tokens (and so the outputs) change."""
    surface, q, gd = _airfoil_case()
    ref = exact_boundary_geometry(q, surface.points, surface.cells)
    seen = {}
    for budget in (50, 200, 10**6):
        m = _interior(2, n_boundary_tokens=budget)
        for train in (False, True):
            m.train(train)
            tokens, d, u, out = _captured(
                m,
                boundary_meshes={"airfoil": surface},
                prediction_points=q,
                global_data=gd,
            )
            assert torch.equal(d, ref[0]) and torch.equal(u, ref[1]), (budget, train)
            assert tokens.shape[0] == min(budget, surface.n_cells)
            assert all(torch.isfinite(v).all() for v in out.values())
            seen[budget, train] = out["p"]
    assert not torch.allclose(seen[50, False], seen[200, False])


def test_outputs_are_invariant_to_the_resolution_of_the_input_mesh():
    """The same geometry at two and four times the resolution (every cell split
    into equal pieces) at a budget of 400 tokens: the outputs move (relative L2) by
    less than three times the largest change between training-mode draws on the
    native mesh, so a change of resolution reads as another draw of the same
    quadrature. Kept whole (the default budget exceeds every cell count) the
    outputs move by under 1e-3: the resolution invariance without draw noise.

    The draws' spread depends on the random weights (0.8-18 % in 2-D and 0.8-6 % in
    3-D over init seeds 0-7), so no absolute cap is a bound. A tripwire of 0.37,
    twice the largest refined change or spread measured, still fails a regression
    that makes every draw noisy."""
    for dim in (2, 3):
        if dim == 2:
            surface, q, gd = _airfoil_case()
        else:
            surface = _sphere(3)
            q = (
                torch.randn(300, 3, dtype=D, generator=torch.Generator().manual_seed(1))
                * 0.8
            )
            gd = {"g": torch.ones(3, dtype=D)}
        fine = surface.subdivide(levels=1, filter="linear")
        finer = fine.subdivide(levels=1, filter="linear")
        m = _interior(dim, n_boundary_tokens=400)
        run = lambda s: outputs(  # noqa: E731
            m(boundary_meshes={"wall": s}, prediction_points=q, global_data=gd)
        )
        with torch.no_grad():
            base, a, b = run(surface), run(fine), run(finer)
            m.train()
            torch.manual_seed(7)
            spread = max(_rel(run(surface), base) for _ in range(8))
            whole = _interior(dim)
            w0 = outputs(
                whole(
                    boundary_meshes={"wall": surface},
                    prediction_points=q,
                    global_data=gd,
                )
            )
            w1 = outputs(
                whole(
                    boundary_meshes={"wall": fine}, prediction_points=q, global_data=gd
                )
            )
        assert spread < 0.37, (dim, spread)
        for refined in (a, b):
            assert _rel(refined, base) < min(0.37, 3 * spread), (
                dim,
                _rel(refined, base),
                spread,
            )
        assert _rel(w1, w0) < 1e-3, dim


def test_evaluation_is_deterministic_and_training_draws_from_the_global_rng():
    """In evaluation the draw starts at a fixed point: two calls agree bitwise. In
    training each call draws its start from the global torch RNG: two calls differ,
    and torch.manual_seed reproduces a call bitwise."""
    surface, q, gd = _airfoil_case()
    m = _interior(2, n_boundary_tokens=100)
    kw = dict(boundary_meshes={"airfoil": surface}, prediction_points=q, global_data=gd)
    t1, _, _, e1 = _captured(m, **kw)
    t2, _, _, e2 = _captured(m, **kw)
    assert torch.equal(t1["points"], t2["points"]) and all(
        torch.equal(e1[k], e2[k]) for k in e1
    )
    m.train()
    torch.manual_seed(3)
    r1 = _captured(m, **kw)
    r2 = _captured(m, **kw)
    torch.manual_seed(3)
    r3 = _captured(m, **kw)
    assert not torch.equal(r1[0]["points"], r2[0]["points"])
    assert torch.equal(r1[0]["points"], r3[0]["points"]) and all(
        torch.equal(r1[3][k], r3[3][k]) for k in r1[3]
    )


def test_the_tokens_are_an_exact_count_with_small_conditions_kept_whole():
    """Two conditions, a 798-segment airfoil and a 12-segment far-field polygon, at
    a total budget of 100: exactly 100 tokens, the 12 far-field cells all kept with
    their own measures (a condition below its share is whole), the airfoil's 88
    tokens carrying Horvitz-Thompson measures at least the cells' own; with a budget
    above the 810 cells every cell is a token with its own measure, in mesh order."""
    surface, q, gd = _airfoil_case()
    t = torch.linspace(0, 2 * math.pi, 13, dtype=D)[:-1]
    far = _polygon(torch.stack([torch.cos(t), torch.sin(t)], -1) * 5)
    kw = dict(
        boundary_meshes={"airfoil": surface, "far": far},
        prediction_points=q,
        global_data=gd,
    )
    tokens, _, _, _ = _captured(_interior(2, n_boundary_tokens=100), **kw)
    assert tokens.shape[0] == 100
    on_far = tokens["points"].norm(dim=-1) > 2
    assert int(on_far.sum()) == 12
    assert torch.allclose(
        tokens["measure"][on_far].sort().values,
        far.cell_areas.sort().values,
        rtol=1e-12,
        atol=0,
    )
    airfoil = tokens["measure"][~on_far]
    assert float(airfoil.sum()) == pytest.approx(
        float(surface.cell_areas.sum()), rel=0.05
    )
    assert bool((airfoil >= surface.cell_areas.min()).all())
    tokens, _, _, _ = _captured(_interior(2), **kw)
    assert torch.equal(
        tokens["points"], torch.cat([surface.cell_centroids, far.cell_centroids])
    )
    assert torch.equal(
        tokens["measure"], torch.cat([surface.cell_areas, far.cell_areas])
    )


def test_n_boundary_tokens_is_validated():
    """``n_boundary_tokens`` is a positive integer, and a surface model, which
    samples nothing, takes only the default. An interior model needs boundary cells,
    and the interior task is refused outside two and three dimensions."""
    for bad in (0, -3, 2.5, True):
        with pytest.raises(ValueError, match="positive integer"):
            ISLA(
                output_schema=PV,
                boundary_schemas=WALL,
                interior=True,
                n_boundary_tokens=bad,
            )
    with pytest.raises(ValueError, match="surface task"):
        ISLA(output_schema=PV, boundary_schemas=WALL, n_boundary_tokens=1000)
    ISLA(output_schema=PV, boundary_schemas=WALL, n_boundary_tokens=200_000)
    surface, q, gd = _airfoil_case(60, 10)
    empty = Mesh(points=surface.points, cells=surface.cells[:0])
    with torch.no_grad(), pytest.raises(ValueError, match="have no cells"):
        _interior(2)(
            boundary_meshes={"airfoil": empty}, prediction_points=q, global_data=gd
        )
    with pytest.raises(NotImplementedError, match="two and three dimensions"):
        ISLA(
            output_schema=PV,
            boundary_schemas=WALL,
            interior=True,
            n_spatial_dims=4,
        )


def _flipped(mesh: Mesh) -> Mesh:
    """``mesh`` with every cell's last two vertices swapped (each normal reversed)."""
    d = mesh.cells.shape[-1]
    return Mesh(points=mesh.points, cells=mesh.cells[:, [*range(d - 2), d - 1, d - 2]])


@pytest.mark.parametrize("dim", [2, 3])
def test_the_sign_follows_the_surface_s_orientation(dim):
    """The distance is positive on the side the surface's normals point to:
    reversing every cell reverses the sign of the distance and leaves the direction
    (from the surface to the point) as it is, to rounding."""
    m = _interior(dim)
    surface = _polygon(_naca(200)) if dim == 2 else _sphere(3)
    q = torch.randn(80, dim, dtype=D, generator=torch.Generator().manual_seed(3)) * 0.7
    kw = dict(prediction_points=q, global_data={"g": torch.ones(dim, dtype=D)})
    _, d, u, _ = _captured(m, boundary_meshes={"wall": surface}, **kw)
    _, d2, u2, _ = _captured(m, boundary_meshes={"wall": _flipped(surface)}, **kw)
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
    """The winding number the geometry evaluates on its BVH (far field at
    ``WINDING_NUMBER_BETA``, exact solid angles at the near leaves)."""
    pts = mesh.points.double()
    cells, parity = geometry._canonical_cells(pts, geometry._weld(pts)[mesh.cells])
    s, par = pts[cells], parity.to(D)
    e = s[:, 1:] - s[:, :1]
    area = par[:, None] * 0.5 * torch.linalg.cross(e[:, 0], e[:, 1], dim=-1)
    tree, order, s_leaf = geometry._preorder_cell_tree(s)
    w = geometry._packed_winding_sum(q.double(), s_leaf, par[order], area[order], tree)
    return w / (4 * math.pi)


_DEVICES = ["cpu", pytest.param("cuda", marks=[pytest.mark.cuda, requires_cuda])]


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
    d, _ = exact_boundary_geometry(
        q.to(device, dtype), mesh.points.to(device, dtype), mesh.cells.to(device)
    )
    assert d.dtype == dtype and (d > 0).all()
    ### relative on purpose: the offsets span three decades (1e-4 to 0.1) and the
    ### distance's error scales with them (measured 2.5e-8 relative in float64)
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


@pytest.mark.parametrize(
    "shape",
    [
        "naca",
        ### a 320-cell sphere and 300 queries: the three-dimensional sign and the
        ### BVH's far-field bound in the fast tier
        "small_sphere",
        ### 2-11 s (sphere) and 4-40 s (torus) on CPU: the exact winding number
        ### over every (query, cell) pair, memory-bound
        pytest.param("sphere", marks=pytest.mark.slow),
        pytest.param("torus", marks=pytest.mark.slow),
    ],
)
def test_closed_surfaces_take_the_sign_of_the_exact_winding_number(shape):
    """On watertight surfaces (a NACA 0012, two spheres, a torus), natively, in a
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
    elif shape == "small_sphere":
        mesh = _sphere(2)
        q = _near_wall(3, 300, 3, band=0.5)
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
    """The token's direction, sign(d) u with sign(0) = +1."""
    return torch.where((d >= 0)[:, None], u, -u)


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
        d2, u2 = exact_boundary_geometry(q, mesh.points[perm], inv[mesh.cells])
        assert torch.equal(d2, d) and torch.equal(u2, u), perm
    for shift in range(1, 4):
        d2, u2 = exact_boundary_geometry(q, mesh.points, mesh.cells.roll(shift, 0))
        assert torch.equal(d2, d) and torch.equal(u2, u), shift


@pytest.mark.parametrize("dim", [2, 3])
def test_tie_direction_flips_exactly_with_the_winding(dim):
    """A box in a generic pose (its ties hold to rounding only): reversing every cell
    negates d and sign(d) u bitwise at every tie point: the same tie set, read with
    the other orientation (a first-found feature could change with the swapped
    vertex order)."""
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
    for seed in range(3):
        d2, u2 = exact_boundary_geometry(q, *_renumbered(mesh.points, mesh.cells, seed))
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
    (the float32 normals are not unit in float64, so their least-norm point can fail
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
    d, u = exact_boundary_geometry(q, pts, mesh.cells)
    d64, u64 = exact_boundary_geometry(q.double(), pts.double(), mesh.cells)
    wall = d64 == 0
    assert wall[: pts.shape[0]].all() and torch.equal(
        d[wall], torch.zeros_like(d[wall])
    )
    assert (u64[wall].norm(dim=-1) > 0.5).all() and (u[wall].norm(dim=-1) > 0.5).all()
    assert (u[wall].double() - u64[wall]).norm(dim=-1).max() <= 1e-5
    _, u2 = exact_boundary_geometry(
        q, pts, _flipped(Mesh(points=pts, cells=mesh.cells)).cells
    )
    assert torch.equal(u2[wall], -u[wall])


@pytest.mark.parametrize("surface", ["split", "icosphere"])
def test_on_wall_direction_is_a_function_of_the_input_mesh(surface):
    """Float32 meshes in a generic pose, at their vertices, edge midpoints and
    centroids, and on the split torus at its quads' diagonals: on the surface, u from
    the float32 inputs is bitwise u from float64 copies of them after the cast. The
    normals are computed in float64 from the coordinates as given, whatever the
    queries' precision. Float32 normals would carry their rounding into the
    least-norm point, which amplifies it where a split face's two triangles have
    near-duplicate normals (``split``: a torus with its vertices moved by 1e-5 of
    their radius, so each quad's triangles are about 1e-5 apart). A quad's diagonal
    is where those normals meet."""
    if surface == "split":
        ### The torus flipped to the other winding: the midpoints of cells[:, :2] are
        ### then quad sides; the torus as loaded puts each quad's diagonal there.
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
    if surface == "split":
        ### and the midpoints of the quads' diagonals, on the same points
        diagonals, same_pts, _ = _quad_diagonals("precision")
        assert torch.equal(same_pts, pts)
        q = torch.cat([q, diagonals])
    d, u = exact_boundary_geometry(q, pts, m.cells)
    d64, u64 = exact_boundary_geometry(q.double(), pts.double(), m.cells)
    wall = (d == 0) & (d64 == 0)
    assert wall[: pts.shape[0]].all() and (u[wall].norm(dim=-1) > 0.9).all()
    if surface == "split":
        assert wall[-diagonals.shape[0] :].any()
    assert torch.equal(u[wall], u64[wall].float())


@pytest.mark.parametrize("surface", ["sphere", "torus"])
def test_float32_points_within_the_bound_are_on_the_wall(surface):
    """Float32 meshes as built (an icosphere, a torus) in a generic pose, at their
    vertices, their cells' edge midpoints and centroids (on the surface to float32
    rounding: up to 7e-8 off it in float64, within the closest cell's error bound)
    and at centroids moved up to 1e-5 along the cell's normal. Every point on the
    surface is on the wall: d = 0 and u the least-norm normal, never 0. Off the wall
    d keeps the side the point was moved to. A winding flip negates d and sign(d) u
    bitwise, and renumbering the vertices and reordering the cells leave d and u
    bitwise."""
    if surface == "sphere":
        m = sphere_icosahedral.load(radius=1.0, subdivisions=3)
    else:
        ### The torus flipped to the other winding, so that the midpoints of cells[:, :2]
        ### are quad sides, as this test is for (the torus as loaded puts each quad's
        ### diagonal there).
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
    d, u = exact_boundary_geometry(q, pts, m.cells)
    wall, moved = d == 0, slice(q.shape[0] - m.n_cells, None)
    assert wall[: moved.start].all() and not wall[moved].all()
    assert (u[wall].norm(dim=-1) > 0.5).all()
    assert torch.equal(
        torch.sign(d[moved][~wall[moved]]), torch.sign(s[~wall[moved], 0]).float()
    )
    d2, u2 = exact_boundary_geometry(
        q, pts, _flipped(Mesh(points=pts, cells=m.cells)).cells
    )
    assert torch.equal(d2, -d) and torch.equal(_gradient(d2, u2), -_gradient(d, u))
    d3, u3 = exact_boundary_geometry(q, *_renumbered(pts, m.cells))
    assert torch.equal(d3, d) and torch.equal(u3, u)


@pytest.mark.parametrize("dim", [2, 3])
def test_float64_points_beyond_the_bound_are_off_the_wall(dim):
    """A posed float64 box, at its vertices, its cells' edge midpoints and
    centroids, and at its centroids moved along the outward normal by a quarter of
    the closest cell's error bound E and by four times it (E ~ 1e-14 here): the
    first are on the wall (d = 0, u the least-norm normal), the last off it, with d
    its distance and u the normal, since float64 resolves them (a fixed cutoff of
    1e-12 would put every point within 1e-12 on the wall)."""
    mesh, q, nu = _box_wall(dim)
    R, t = orthogonal(dim, 7), torch.arange(1, dim + 1, dtype=D) / 3
    moved, q, nu = transform(mesh, R, t), q @ R.T + t, nu @ R.T
    rows = slice(mesh.n_points, mesh.n_points + mesh.n_cells)
    c, n = q[rows], nu[rows]
    u64, tri = torch.finfo(D).eps / 2, moved.points[moved.cells]
    origin = geometry._centroid(tri)
    e = u64 * (c.norm(dim=-1) + tri.norm(dim=-1).amax(-1)) + geometry._BAND_UNITS[
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
@pytest.mark.parametrize("surface", ["sphere", "torus", "ellipse", "sliver"])
def test_tie_directions_take_the_exact_least_norm_point(surface, dtype, monkeypatch):
    """At the vertices and the cells' edge midpoints of curved surfaces in a generic
    pose (the sliver a sphere with a zero-area triangle on one edge), on the surface
    and off it (a float32 midpoint lies a few 1e-8 off its edge), every tie's
    least-norm step returns the exact least-norm point of its directions (enumerated
    in float64), to 1e-6 from float64 inputs and 1e-5 from float32 ones, and never 0
    where that is not."""
    if surface == "ellipse":
        m = _circle(400)
        mesh = Mesh(points=m.points * torch.tensor([1.0, 0.12], dtype=D), cells=m.cells)
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
    min_norm = geometry._min_norm_points

    def recorded(c):
        x = min_norm(c)
        calls.append((c, x))
        return x

    monkeypatch.setattr(geometry, "_min_norm_points", recorded)
    exact_boundary_geometry(q, pts, cells)
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
    exact least-norm point; otherwise 0, exactly where the exact point is 0."""
    g = torch.Generator().manual_seed(k * dim)
    c = torch.randn(200, 1, dim, generator=g) + 0.4 * torch.randn(
        200, k, dim, generator=g
    )
    c = torch.nn.functional.normalize(c, dim=-1).double()
    x, exact = geometry._min_norm_points(c), _least_norm(c, tol=1e-12)
    assert torch.equal(x.norm(dim=-1) == 0, exact.norm(dim=-1) == 0)
    gap = (x * x).sum(-1) - (c * x[:, None]).sum(-1).amin(-1)
    assert ((x - exact).norm(dim=-1) <= (2 * gap.clamp_min(0)).sqrt() + 1e-12).all()


### K = 100 is slow (68 s on a shared, loaded CPU; the library's copy took 24 s: the
### single pass's 166,650 subsets times 100 directions, and the chunked pass's 167
### chunks per row); K = 20 runs the same chunked branch in the default tier.
@pytest.mark.parametrize("k", [20, pytest.param(100, marks=pytest.mark.slow)])
def test_least_norm_point_of_many_directions_in_chunks(k, monkeypatch):
    """A hub vertex of many cells whose normals differ by rounding ties them
    all: K directions give C(K, 3) triples, each compared with all K, more than one
    pass holds. Rows of K unit directions (a cap, the same read from float32,
    one spread by rounding only, and a sphere, around the origin) are taken one at a
    time over chunks of their subsets, each chunk within the budget, and give
    bitwise the single pass's result."""
    g = torch.Generator().manual_seed(9)
    rows = []
    for spread, f32 in ((1e-3, False), (1e-3, True), (1e-7, False), (3.0, False)):
        c = torch.tensor([0.0, 0.0, 1.0], dtype=D) + spread * torch.randn(
            k, 3, dtype=D, generator=g
        )
        c = torch.nn.functional.normalize(c, dim=-1)
        rows.append(torch.unique(c.float().double() if f32 else c, dim=0))
    c = torch.stack(rows)
    assert c.shape == (4, k, 3)
    monkeypatch.setattr(geometry, "_SEARCH_PAIRS", 2**40)
    single = torch.cat([geometry._min_norm_points(row[None]) for row in c])
    budget = 10 * k * k  # 10 K subsets a chunk
    monkeypatch.setattr(geometry, "_SEARCH_PAIRS", 8 * budget)
    sizes, candidates = [], geometry._hull_candidates

    def spy(cc, subsets):
        sizes.append(cc.shape[0] * sum(s.shape[0] for s in subsets) * cc.shape[1])
        return candidates(cc, subsets)

    monkeypatch.setattr(geometry, "_hull_candidates", spy)
    chunked = geometry._min_norm_points(c)
    assert len(sizes) > 4 and max(sizes) <= budget
    assert torch.equal(chunked.view(torch.int64), single.view(torch.int64))
    assert (chunked[3] == 0).all() and (chunked[:3].norm(dim=-1) > 0.9).all()


@pytest.mark.cuda
@requires_cuda
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
        d2, u2 = exact_boundary_geometry(
            q.cuda(), moved.points.cuda(), moved.cells.cuda()
        )
        assert torch.allclose(d2.cpu(), d, atol=1e-12, rtol=0)
        assert torch.allclose(u2.cpu(), u, atol=1e-12, rtol=0)
    mesh = _circle(2000) if dim == 2 else _sphere(4)
    pts, q = mesh.points.float(), _near_wall(dim, 4000, 11, band=0.3).float()
    d, u = exact_boundary_geometry(q, pts, mesh.cells)
    d2, u2 = exact_boundary_geometry(q.cuda(), pts.cuda(), mesh.cells.cuda())
    assert torch.equal(torch.sign(d2.cpu()), torch.sign(d))
    assert torch.allclose(d2.cpu(), d, atol=1e-6, rtol=0)
    assert torch.allclose(u2.cpu(), u, atol=1e-5, rtol=0)
    assert torch.equal(u2.cpu().norm(dim=-1) < 1 - 1e-5, u.norm(dim=-1) < 1 - 1e-5)
