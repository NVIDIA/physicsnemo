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

"""The error bound of the exact boundary geometry and the frame its arithmetic runs in.

The bound's arithmetic share is a worst case of the float64 pair distance in the
boundary's centred frame, and its storage share the rounding of the coordinates as
given: far from the origin the distance keeps the queries' relative precision, on
a posed planar star u is the plane's normal, near-equal directions off the surface
tie only where certified distinct, the derived share holds against exact rational
distances, the outputs do not depend on the numbering (exact distance ties
included) nor, on the surface, on the queries' precision, and the fast search gives
the reference's outputs bitwise.
"""

from fractions import Fraction

import pytest
import torch

from physicsnemo.experimental.models.isla import _geometry as geometry
from physicsnemo.experimental.models.isla._geometry import exact_boundary_geometry
from physicsnemo.mesh.geometry import compute_cell_normals
from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral
from test.experimental.models.isla._geometry_cases import (
    FAST_DEVICES,
    KERNEL_DEVICES,
    _bitwise,
    _exact_d2,
    _quad_diagonals,
    _renumbered,
)
from test.experimental.models.isla._marks import requires_cuda
from test.experimental.models.isla._meshes import orthogonal

D = torch.float64
U32, U64 = 2.0**-24, 2.0**-53

_DEVICES = ["cpu", pytest.param("cuda", marks=[pytest.mark.cuda, requires_cuda])]


def _search(q, p, c):
    """Every output of the reference search op."""
    cells, parity = geometry._canonical_cells(p, geometry._weld(p)[c])
    s = p[cells].double()
    n64 = parity[:, None].double() * compute_cell_normals(s[:, 1:] - s[:, :1])
    return geometry._closest_candidates(q, p, cells, parity, n64)


def _far():
    """An icosphere of radius 1 translated by about 1e3 of its size, and float32
    queries 1e-3 to 2e-3 off its cells' centroids along their normals, either side
    (outside the storage share of the bound, ``u32 (|x| + R) ~ 1.2e-4``)."""
    m = sphere_icosahedral.load(radius=1.0, subdivisions=1)
    p, c = m.points.double(), m.cells
    tri = p[c]
    n = torch.nn.functional.normalize(
        torch.linalg.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=-1), dim=-1
    )
    g = torch.Generator().manual_seed(0)
    s = (1 + torch.rand(c.shape[0], 1, dtype=D, generator=g)) * 1e-3
    s = s * torch.where(torch.rand(c.shape[0], 1, generator=g) < 0.5, -1.0, 1.0)
    t = torch.tensor([1000.0, -700.0, 400.0], dtype=D)
    return (tri.mean(1) + s * n + t).float(), (p + t).float(), c, t.float()


@pytest.mark.parametrize("device", _DEVICES)
def test_far_from_the_origin_the_distance_keeps_its_relative_precision(device):
    """A float32 surface 1e3 of its size from the origin, queried 1e-3 off it: the
    distance is within 4 float32 units of itself of the exact one of the stored
    coordinates (float64, after an exact shift to the origin), as it is at the
    origin; a closest point formed in float32 loses it to the rounding of the
    coordinates' magnitude, 3e-5."""
    q, p, c, t = _far()
    d, _ = exact_boundary_geometry(q.to(device), p.to(device), c.to(device))
    t = t.double()
    exact, _ = exact_boundary_geometry(q.double() - t, p.double() - t, c)
    d = d.cpu().double()
    assert bool((d != 0).all()), "queries off the surface taken on it"
    err = ((d - exact).abs() / exact.abs()).max()
    assert err <= 4 * U32, f"relative distance error {float(err):.3g} > 4 u32"


def _star(posed: bool):
    """A planar star, seven triangles around a centre in the plane z = 0, as float32,
    rotated and translated when ``posed``, its plane's unit normal, and queries at
    its centre and at its spokes' midpoints (on the surface, every cell there
    coplanar)."""
    k = 7
    a = torch.arange(k, dtype=D) * 2 * torch.pi / k
    p = torch.cat(
        [torch.zeros(1, 3, dtype=D), torch.stack([a.cos(), a.sin(), 0 * a], -1)]
    )
    c = torch.stack(
        [
            torch.zeros(k, dtype=torch.long),
            torch.arange(1, k + 1),
            torch.arange(1, k + 1) % k + 1,
        ],
        -1,
    )
    q = torch.cat([p[:1], 0.5 * p[1:]])
    n = torch.tensor([0.0, 0.0, 1.0], dtype=D)
    if posed:
        R, t = orthogonal(3, 5), torch.tensor([0.7, -0.4, 1.3], dtype=D)
        p, q, n = p @ R.T + t, q @ R.T + t, n @ R.T
    return q.float(), p.float(), c, n


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("posed", [False, True])
def test_on_a_planar_star_u_is_the_plane_s_normal(posed, device):
    """At the centre and on the spokes of a planar star stored in float32 the
    query is on the surface in either pose, d = 0 and u the plane's normal to
    float32 rounding: the cells' float64 normals differ there only by the rounding
    of the stored vertices, so their least-norm point is within it too."""
    q, p, c, n = _star(posed)
    q, p, c = q.to(device), p.to(device), c.to(device)
    _, _, _, _, on_wall, _ = _search(q, p, c)
    assert bool(on_wall.all())
    d, u = exact_boundary_geometry(q, p, c)
    assert bool((d == 0).all())
    assert float((u.cpu().double() - n).norm(dim=-1).max()) <= 4 * U32


@pytest.mark.parametrize("dtype", [torch.float32, D])
def test_near_equal_directions_off_the_surface_tie_only_where_certified(dtype):
    """A valley of two triangles folded by 2a (a = 2^-17) along the x axis, queried
    1e-3 above its fold: each triangle's closest point is interior to it, at the
    same distance, and the directions are their normals, 1.5e-5 apart. In float32
    that is within their certified bounds (the storage rounding over the distance,
    about 1e-4), so the query is not tied and u is one of the normals; in float64
    they are certified distinct, so it is tied and u their least-norm point."""
    a = 2.0**-17
    p = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, -1.0, a], [0.5, 1.0, a]], dtype=dtype
    )
    c = torch.tensor([[0, 2, 1], [0, 1, 3]])
    x = torch.linspace(0.3, 0.7, 9, dtype=D)
    q = torch.stack([x, 0 * x, 0 * x + 1e-3], -1).to(dtype)
    _, tied, _, _, on_wall, _ = _search(q, p, c)
    _, u = exact_boundary_geometry(q, p, c)
    assert not bool(on_wall.any())
    u = u.double()
    if dtype == torch.float32:
        assert not bool(tied.any())
        assert torch.allclose(u[:, 1].abs(), torch.full_like(u[:, 1], a), rtol=1e-6)
    else:
        assert bool(tied.all())
        assert float(u[:, 1].abs().max()) <= 1e-12


def _thin(g, n, rho):
    """``n`` triangles of shape ``rho`` (a cap when the apex is near the long edge's
    middle, a needle near an end), posed 100 of their size from the origin, and a
    query per triangle on it, just off it, or beside its long edge."""
    a = torch.rand(n, 3, dtype=D, generator=g)
    b = a + torch.nn.functional.normalize(
        torch.randn(n, 3, dtype=D, generator=g), dim=-1
    )
    side = torch.nn.functional.normalize(
        torch.linalg.cross(b - a, torch.randn(n, 3, dtype=D, generator=g), dim=-1),
        dim=-1,
    )
    s = torch.rand(n, 1, dtype=D, generator=g)
    apex = a + s * (b - a) + rho * side
    tri = torch.stack([a, b, apex], 1)
    w = torch.rand(n, 3, dtype=D, generator=g)
    on = ((w / w.sum(1, keepdim=True))[..., None] * tri).sum(1)
    nrm = torch.nn.functional.normalize(
        torch.linalg.cross(b - a, apex - a, dim=-1), dim=-1
    )
    h = torch.rand(n, 1, dtype=D, generator=g) * 2 - 1
    kind = torch.arange(n)[:, None] % 3
    q = torch.where(
        kind == 0,
        on,
        torch.where(kind == 1, on + 1e-3 * h * nrm, a + s * (b - a) - 1e-2 * h * side),
    )
    t = torch.tensor([60.0, -80.0, 20.0], dtype=D)
    return q + t, tri + t


def _exact_segment_d2(x, a, b):
    """The squared distance from ``x`` to the segment ``[a, b]``, exactly."""
    q, a, b = ([Fraction(v) for v in p] for p in (x, a, b))
    e = [b[i] - a[i] for i in range(2)]
    y = [q[i] - a[i] for i in range(2)]
    t = (y[0] * e[0] + y[1] * e[1]) / (e[0] * e[0] + e[1] * e[1])
    t = min(max(t, Fraction(0)), Fraction(1))
    r = [y[i] - t * e[i] for i in range(2)]
    return r[0] * r[0] + r[1] * r[1]


@pytest.mark.parametrize("dim", [2, 3])
def test_the_arithmetic_share_bounds_the_pair_distance(dim):
    """Float64 cells and queries posed far from the origin, centred as the search
    centres them: the pair distance from the weights of ``_closest_weights`` is
    within ``C u64 (|x'| + R'_k)`` (``_BAND_UNITS``) of the exact distance from the
    query to the cell as given (exact rational arithmetic), on thin caps and
    needles in three dimensions, many of them guarded or switched, and on segments
    in two."""
    g = torch.Generator().manual_seed(dim)
    if dim == 3:
        q, tri = map(
            torch.cat, zip(*(_thin(g, 60, rho) for rho in (1e-2, 1e-6, 1e-10)))
        )
    else:
        a = torch.rand(200, 2, dtype=D, generator=g) + torch.tensor(
            [50.0, -30.0], dtype=D
        )
        tri = torch.stack([a, a + torch.randn(200, 2, dtype=D, generator=g)], 1)
        q = tri.mean(1) + torch.randn(
            200, 2, dtype=D, generator=g
        ) * 10.0 ** -torch.randint(0, 8, (200, 1), generator=g)
    origin = geometry._centroid(tri)
    qc, tc = q - origin, tri - origin
    w = geometry._closest_weights(qc, tc)
    dist = (qc - (w[..., None] * tc).sum(1)).norm(dim=-1)
    if dim == 3:
        exact = [
            float(_exact_d2(x.tolist(), t.tolist())) ** 0.5 for x, t in zip(q, tri)
        ]
    else:
        exact = [
            float(_exact_segment_d2(x.tolist(), *t.tolist())) ** 0.5
            for x, t in zip(q, tri)
        ]
    unit = U64 * (qc.norm(dim=-1) + tc.norm(dim=-1).amax(-1))
    ratio = ((dist - torch.tensor(exact, dtype=D)).abs() / unit).max()
    assert ratio <= geometry._BAND_UNITS[dim], f"{float(ratio):.3g} units"
    assert geometry._BAND_UNITS[3] >= geometry._SWITCH_UNITS + geometry._FOOT_UNITS


def test_the_band_units_are_the_derived_constants():
    """The arithmetic share's units are pinned: the tests read ``_BAND_UNITS``, so a
    change to it must be a change to its derivation."""
    assert geometry._BAND_UNITS == {2: 17, 3: 91}, (
        "the band units are derived constants: change their derivation first"
    )


def _numbered():
    """A posed float32 icosphere, queries around it, on its vertices and its
    edges' midpoints, and the same surface with its vertices and cells permuted and
    each cell's vertex list rotated."""
    m = sphere_icosahedral.load(radius=1.0, subdivisions=2)
    R, t = orthogonal(3, 3), torch.tensor([30.0, 10.0, -20.0], dtype=D)
    p, c = m.points.double() @ R.T + t, m.cells
    g = torch.Generator().manual_seed(4)
    q = torch.cat(
        [
            torch.randn(300, 3, dtype=D, generator=g) * 0.8 @ R.T + t,
            p[:40],
            p[c[:40, :2]].mean(1),
        ]
    ).float()
    p = p.float()
    return q, p, c, *_renumbered(p, c, seed=4)


def _ellipse():
    """A float32 polyline of 4000 segments on an ellipse of semi-axes 1 and 0.12,
    posed, and its vertices: on its flat sides consecutive normals are about 2e-4
    apart, segments about 1.6e-3 long against a storage rounding of about 5e-8, so
    many are not certified distinct in float32 (with 3000 segments every pair is
    certified)."""
    n = 4000
    t = torch.arange(n, dtype=D) * 2 * torch.pi / n
    p = torch.stack([torch.cos(t), 0.12 * torch.sin(t)], -1)
    p = (p @ orthogonal(2, 7).T + torch.tensor([0.3, 0.7], dtype=D)).float()
    return p, p, torch.stack([torch.arange(n), (torch.arange(n) + 1) % n], -1)


def _off_the_wall():
    """Float64 queries 1e-3 off the posed torus of :func:`_quad_diagonals` along the
    normal at its quad diagonals' midpoints, on either side (closest points on edges,
    where the cells sharing one tie exactly for the least distance), and at Gaussian
    offsets (sigma 1e-3) from its vertices."""
    q, p, c = _quad_diagonals("renumbering")
    q, p = q.double(), p.double()
    tri = p[c[: q.shape[0]]]
    n = torch.nn.functional.normalize(
        torch.linalg.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=-1), dim=-1
    )
    g = torch.Generator().manual_seed(2)
    near = p + 1e-3 * torch.randn(p.shape, dtype=D, generator=g)
    return torch.cat([q + 1e-3 * n, q - 1e-3 * n, near]), p, c


@pytest.mark.parametrize("kernel, device", KERNEL_DEVICES)
@pytest.mark.parametrize("case", ["numbered", "ellipse", "diagonals", "off the wall"])
def test_exact_distance_ties_do_not_depend_on_the_numbering(case, device, kernel):
    """Permuting the vertices and the cells and rotating each triangle's vertex list
    leave d and u bitwise under either kernel: on a posed icosphere (the centred
    frame's origin is a function of the set of cells), and where cells tie exactly
    for the least distance and their directions are not certified distinct (on the
    surface at the fine ellipse's vertices and the torus's quad diagonals, off it at
    the torus's edge and vertex closest points). The closest cell is chosen by the
    cells' positions, and on the surface u is the least-norm point of every
    different normal."""
    build = {
        "numbered": lambda: _numbered()[:3],
        "ellipse": _ellipse,
        "off the wall": _off_the_wall,
    }
    q, p, c = build[case]() if case in build else _quad_diagonals("renumbering")
    p2, c2 = _renumbered(p, c)
    q, p, c, p2, c2 = (x.to(device) for x in (q, p, c, p2, c2))
    d, u = exact_boundary_geometry(q, p, c, kernel=kernel)
    d2, u2 = exact_boundary_geometry(q, p2, c2, kernel=kernel)
    assert torch.equal(d, d2) and torch.equal(u, u2)


def test_on_wall_direction_of_a_fine_polyline_in_float32_is_the_float64_one():
    """At the fine ellipse's vertices, on the surface in both precisions, u from the
    float32 inputs is bitwise u from float64 copies of them after the cast: both are
    the least-norm point of the same two float64 normals, which float64 certifies
    distinct where float32 does not."""
    q, p, c = _ellipse()
    d, u = exact_boundary_geometry(q, p, c)
    d64, u64 = exact_boundary_geometry(q.double(), p.double(), c)
    assert bool(((d == 0) & (d64 == 0)).all())
    assert torch.equal(u, u64.float())


def _band_cases():
    q, p, c, _ = _far()
    yield "far", q, p, c
    for posed in (False, True):
        q, p, c, _ = _star(posed)
        yield f"star {posed}", q, p, c
    g = torch.Generator().manual_seed(7)
    q, tri = map(torch.cat, zip(*(_thin(g, 30, rho) for rho in (1e-3, 1e-9))))
    yield "thin", q, tri.reshape(-1, 3), torch.arange(tri.shape[0] * 3).reshape(-1, 3)
    q, p, c, p2, c2 = _numbered()
    yield "numbered", q, p, c
    yield "renumbered", q, p2, c2
    yield "ellipse", *_ellipse()
    for case in ("renumbering", "precision"):
        yield f"diagonals {case}", *_quad_diagonals(case)
    yield "off the wall", *_off_the_wall()


@pytest.mark.parametrize("device", FAST_DEVICES)
@pytest.mark.parametrize("case", [c[0] for c in _band_cases()])
def test_the_fast_search_equals_the_reference_bitwise(case, device):
    """On the cases above, d, u and their gradients in the queries and the
    vertices are the reference's bitwise under the fast search."""
    _, q, p, c = next(x for x in _band_cases() if x[0] == case)
    out = []
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for kernel in ("reference", "fast"):
            qq = q.to(device).clone().requires_grad_()
            pp = p.to(device).clone().requires_grad_()
            d, u = exact_boundary_geometry(qq, pp, c.to(device), kernel=kernel)
            probe = torch.linspace(0, 1, u.numel(), dtype=u.dtype, device=u.device)
            gq, gp = torch.autograd.grad(
                d.sum() + (u * probe.reshape(u.shape)).sum(), (qq, pp)
            )
            out.append((d, u, gq, gp))
    finally:
        torch.set_num_threads(threads)
    for name, a, b in zip(("d", "u", "dq", "dp"), *out):
        assert _bitwise(a, b), name
