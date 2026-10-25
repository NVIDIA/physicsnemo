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

"""The exact signed distance on surfaces with thin cells of nonzero area.

A sliver, a cap or a needle (``rho = 2 A / L^2`` far below 1) is a cell like any
other: the distance, the closest cell and whether a query is on the wall are those
of an exact rational brute force, in float32 and float64, at queries on its face,
just off it and far; and a coplanar cap that splits a face of a closed surface
changes nothing.
"""

from fractions import Fraction

import pytest
import torch

from physicsnemo.mesh.geometry._cell_normals import compute_cell_normals
from physicsnemo.mesh.spatial.exact_sdf import _predicates, _search, _weights
from test.mesh.spatial._exact_sdf_meshes import exact_sdf

D = torch.float64
U64 = 2.0**-53

_DEVICES = [
    "cpu",
    pytest.param(
        "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")
    ),
]


def _exact_d2(x, tri):
    """The squared distance from ``x`` to the closed triangle ``tri``, exactly (a
    Fraction): the least over its edges and, where the projection falls inside it,
    its plane."""
    q = [Fraction(v) for v in x]
    a, b, c = ([Fraction(v) for v in p] for p in tri)

    def sub(u, v):
        return [u[i] - v[i] for i in range(3)]

    def dot(u, v):
        return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]

    def cross(u, v):
        return [
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        ]

    best = None
    for p0, p1 in ((a, b), (b, c), (a, c)):
        e = sub(p1, p0)
        t = min(max(dot(sub(q, p0), e) / dot(e, e), Fraction(0)), Fraction(1))
        r = [q[i] - p0[i] - t * e[i] for i in range(3)]
        best = dot(r, r) if best is None else min(best, dot(r, r))
    n = cross(sub(b, a), sub(c, a))
    nn = dot(n, n)
    h = dot(sub(q, a), n)
    p = [q[i] - h / nn * n[i] for i in range(3)]
    if all(
        dot(cross(sub(y, p), sub(z, p)), n) >= 0 for y, z in ((b, c), (c, a), (a, b))
    ):
        best = min(best, h * h / nn)
    return best


def _segment_d2(q, a, b):
    ab = b - a
    t = (((q - a) * ab).sum(-1) / (ab * ab).sum(-1)).clamp(0, 1)
    r = q - a - t[..., None] * ab
    return (r * r).sum(-1)


def _brute(q, tri):
    """For each query, the exact distance to the triangles (float64 of the exact
    value) and the set of cells at it: the cells whose box is no farther than the
    nearest edge (float64, padded) are evaluated exactly."""
    q64, t64 = q.double(), tri.double()
    lo, hi = t64.amin(1), t64.amax(1)
    gap = (lo[None] - q64[:, None]).clamp_min(0) + (q64[:, None] - hi[None]).clamp_min(
        0
    )
    lb = gap.norm(dim=-1)
    ub = (
        torch.stack(
            [
                _segment_d2(q64[:, None], t64[None, :, i], t64[None, :, j])
                for i, j in ((0, 1), (1, 2), (0, 2))
            ]
        )
        .amin(0)
        .sqrt()
        .amin(1)
    )
    pad = 1e-9 * (ub + q64.norm(dim=-1) + t64.norm(dim=-1).max())
    cand = lb * (1 - 1e-9) <= (ub + pad)[:, None]
    dist, cells = [], []
    for i in range(q.shape[0]):
        ks = cand[i].nonzero()[:, 0].tolist()
        d2 = {k: _exact_d2(q64[i].tolist(), t64[k].tolist()) for k in ks}
        m = min(d2.values())
        dist.append(float(m) ** 0.5)
        cells.append({k for k, v in d2.items() if v == m})
    return torch.tensor(dist, dtype=D), cells


def _closest_cell(q, p, c):
    """The search op's closest cell (a row of ``c``) and on-wall decision."""
    cells, parity = _predicates._canonical_cells(p, _predicates._weld(p)[c])
    s = p[cells].double()
    n64 = parity[:, None].double() * compute_cell_normals(s[:, 1:] - s[:, :1])
    idx, _, _, _, on_wall, _ = _search._closest_features_op(q, p, cells, parity, n64)
    return idx, on_wall


def _check(q, p, c, device):
    """Against the brute force: ``|d|`` within ``8 u (|x| + R)`` off the wall, the
    closest cell within the op's tie bound of the closest distance, and the query on
    the wall exactly when the brute force's distance says so (below ``E / 2`` it must
    be, beyond ``2 E`` it must not, ``E = u (|x| + R) + C u64 (|x'| + R')``, primes
    in the boundary's centred frame, ``C`` the derived ``_BAND_UNITS``); and the
    distance has finite gradients in the queries and the vertices."""
    dtype = q.dtype
    u = torch.finfo(dtype).eps / 2
    tri = p.double()[c]
    exact, holders = _brute(q, tri)
    qq, pp = (
        q.to(device).clone().requires_grad_(),
        p.to(device).clone().requires_grad_(),
    )
    d, uu = exact_sdf(qq, pp, c.to(device))
    (d.sum() + uu.sum()).backward()
    assert bool(torch.isfinite(qq.grad).all() and torch.isfinite(pp.grad).all())
    with torch.no_grad():
        idx, on_wall = _closest_cell(q.to(device), p.to(device), c.to(device))
    d, idx, on_wall = d.detach().cpu(), idx.cpu(), on_wall.cpu()
    xs = q.double().norm(dim=-1)
    reach = tri.norm(dim=-1).amax(-1)
    origin = _predicates._centroid(tri).to(dtype).double()
    xc = (q.double() - origin).norm(dim=-1)
    reach_c = (tri - origin).norm(dim=-1).amax(-1)
    share = _weights._BAND_UNITS[3] * U64

    def band(i, k):
        return u * (float(xs[i]) + float(reach[k])) + share * (
            float(xc[i]) + float(reach_c[k])
        )

    e_all = u * (xs + reach.max()) + share * (xc + reach_c.max())
    bad_d = ~on_wall & ((d.double().abs() - exact).abs() > 8 * u * (xs + reach.max()))
    assert int(bad_d.sum()) == 0
    e_near = torch.tensor(
        [min(band(i, k) for k in holders[i]) for i in range(q.shape[0])], dtype=D
    )
    assert int((on_wall & (exact > 2 * e_all)).sum()) == 0
    assert int((~on_wall & (exact <= 0.5 * e_near)).sum()) == 0
    assert bool((d[on_wall] == 0).all())
    wrong = 0
    for i in range(q.shape[0]):
        k = int(idx[i])
        if k not in holders[i]:
            e_k = band(i, k)
            wrong += float(
                _exact_d2(q[i].double().tolist(), tri[k].tolist())
            ) ** 0.5 - float(exact[i]) > e_k + float(e_near[i])
    assert wrong == 0


def _rotation(seed):
    g = torch.Generator().manual_seed(seed)
    qm, r = torch.linalg.qr(torch.randn(3, 3, dtype=D, generator=g))
    qm = qm * torch.sign(torch.diagonal(r))[None]
    return qm if torch.det(qm) > 0 else -qm


def _capped_cube(h, kind):
    """The unit cube's surface, outward, two triangles a face, except its bottom face
    z = 0, split at a vertex ``m`` beside its edge y = 0 into four cells: the first
    is the cap (0,0,0)-(1,0,0)-m of width ``h`` (``m = (0.5, h, 0)``) or a needle
    (``m = (1 - h, h, 0)``), ``rho = h``; and the cube as two triangles a face."""
    p = torch.tensor(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ],
        dtype=D,
    )
    m = {"cap": [0.5, h, 0.0], "needle": [1 - h, h, 0.0]}[kind]
    p = torch.cat([p, torch.tensor([m], dtype=D)])
    rest = [
        [4, 5, 6],
        [4, 6, 7],
        [0, 4, 5],
        [0, 5, 1],
        [1, 5, 6],
        [1, 6, 2],
        [2, 6, 7],
        [2, 7, 3],
        [3, 7, 4],
        [3, 4, 0],
    ]

    def outward(c):
        c = torch.tensor(c)
        v = p[c]
        n = torch.linalg.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0], dim=-1)
        flip = (n * (v.mean(1) - 0.5)).sum(-1) < 0
        return torch.where(flip[:, None], c[:, [0, 2, 1]], c)

    return (
        p,
        outward([[0, 1, 8], [0, 8, 3], [8, 2, 3], [8, 1, 2], *rest]),
        outward([[0, 1, 2], [0, 2, 3], *rest]),
    )


def _face_queries(g, tri, h, n):
    """``n`` points on the triangle ``tri`` (random barycentric), the same moved off
    its plane by ``h`` times a random factor in (-1, 1), and by far less (a few
    hundred float32 units of the coordinates, still clearly off the wall in
    float64), and points near its longest edge."""
    w = torch.rand(n, 3, dtype=D, generator=g)
    on = (w / w.sum(1, keepdim=True)) @ tri
    nrm = torch.linalg.cross(tri[1] - tri[0], tri[2] - tri[0])
    nrm = nrm / nrm.norm()
    side = torch.rand(n, 1, dtype=D, generator=g) * 2 - 1
    s = torch.rand(n, 1, dtype=D, generator=g)
    edge = tri[0] + s * (tri[1] - tri[0])
    return torch.cat(
        [
            on,
            on + h * side * nrm,
            on + 1e-5 * side * nrm,
            edge + h * torch.randn(n, 3, dtype=D, generator=g),
        ]
    )


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, D])
@pytest.mark.parametrize("rho", [1e-3, 1e-6])
@pytest.mark.parametrize("kind", ["cap", "needle"])
def test_rotated_thin_cells(kind, rho, dtype, device):
    """A cap or a needle in a face of a rotated, shifted cube: the distance, the
    closest cell and the on-wall decision are the brute force's at queries on the
    thin cell, just off it, beside its long edge and in the box."""
    p, c, _ = _capped_cube(rho, kind)
    g = torch.Generator().manual_seed(int(rho * 1e7) + len(kind))
    q = torch.cat(
        [
            _face_queries(g, p[c[0]], rho, 40),
            torch.rand(80, 3, dtype=D, generator=g) * 1.4 - 0.2,
        ]
    )
    rot, shift = _rotation(7), torch.tensor([0.3, -0.2, 0.7], dtype=D)
    _check((q @ rot.T + shift).to(dtype), (p @ rot.T + shift).to(dtype), c, device)


def _soup(seed, h):
    """A sliver soup: random triangles in the unit cube, the
    first eight slivers (the middle vertex ``h`` off the line of the other two), with
    random queries and queries around the slivers."""
    g = torch.Generator().manual_seed(500 + seed)
    tri = torch.rand(48, 3, 3, dtype=D, generator=g)
    tri[:8, 1] = 0.5 * (tri[:8, 0] + tri[:8, 2]) + h * torch.randn(
        8, 3, dtype=D, generator=g
    )
    mid = 0.5 * (tri[:8, 0] + tri[:8, 2])
    q = torch.cat(
        [
            torch.rand(80, 3, dtype=D, generator=g) * 1.4 - 0.2,
            (mid[:, None] + 0.01 * torch.randn(8, 10, 3, dtype=D, generator=g)).reshape(
                -1, 3
            ),
            tri[:8].reshape(-1, 3),
            mid,
        ]
    )
    return tri.reshape(-1, 3), torch.arange(144).reshape(-1, 3), q


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("h", [1e-16, 1e-13])
@pytest.mark.parametrize("seed", [0, 1])
def test_sliver_soups(seed, h, device):
    """Float64 sliver soups, whose slivers lie below the certainty of
    their float64 normal: the search keeps their cells, and the answers are the
    brute force's."""
    p, c, q = _soup(seed, h)
    _check(q, p, c, device)


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, D])
@pytest.mark.parametrize("h", [1e-4, 1e-7, 1e-10])
def test_coplanar_cap_changes_nothing(h, dtype, device):
    """A closed cube whose bottom face is four cells, one of them a cap of width
    ``h``, has the distance of the cube with that face as two: within the rounding
    of the precision, at queries on the cap, just off it, near the face and in the
    box, offset or not (where either takes the query on the wall, the other's
    distance is within the on-wall band, ``2 E``)."""
    p, c, base = _capped_cube(h, "cap")
    g = torch.Generator().manual_seed(3)
    q = torch.cat(
        [
            _face_queries(g, p[c[0]], h, 60),
            torch.rand(200, 3, dtype=D, generator=g) * 1.4 - 0.2,
        ]
    )
    u = torch.finfo(dtype).eps / 2
    for shift in (0.0, 1024.0):
        args = (q + shift).to(device, dtype), (p + shift).to(device, dtype)
        d0, _ = exact_sdf(*args, base.to(device))
        d, _ = exact_sdf(*args, c.to(device))
        scale = (q + shift).norm(dim=-1) + (p + shift).norm(dim=-1).max()
        qs, ps = (x.double().cpu() for x in args)
        origin = _predicates._centroid(ps[c]).to(dtype).double()
        scale_c = (qs - origin).norm(dim=-1) + (ps - origin).norm(dim=-1).max()
        d, d0 = d.double().cpu(), d0.double().cpu()
        wall = (d == 0) | (d0 == 0)
        share = _weights._BAND_UNITS[3] * U64
        tol = torch.where(wall, 2 * (u * scale + share * scale_c), 8 * u * scale)
        assert ((d - d0).abs() <= tol).all()


def test_certified_normal():
    """``delta`` bounds the angle between a cell's float64 normal and its exact one,
    on rotated slivers from ``rho = 1e-2`` to ``1e-15``; the ball search's float32
    planes are dropped exactly where it exceeds float32's unit roundoff, which no
    cell of ``rho >= 1e-6`` reaches."""
    rho = 10.0 ** -torch.arange(2, 16, dtype=D)
    base = torch.stack(
        [
            torch.zeros(14, 3, dtype=D),
            torch.tensor([1.0, 0, 0], dtype=D).expand(14, 3),
            torch.stack([0.5 + 0 * rho, rho, 0 * rho], -1),
        ],
        1,
    )
    tri = base @ _rotation(11).T + torch.tensor([0.3, -0.2, 0.7], dtype=D)
    delta, _, _ = _weights._thin_bounds(tri)
    n = torch.linalg.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=-1)
    for k in range(14):
        a, b, c = ([Fraction(v) for v in x] for x in tri[k].tolist())
        e1, e2 = [b[i] - a[i] for i in range(3)], [c[i] - a[i] for i in range(3)]
        exact = [
            e1[1] * e2[2] - e1[2] * e2[1],
            e1[2] * e2[0] - e1[0] * e2[2],
            e1[0] * e2[1] - e1[1] * e2[0],
        ]
        nk = [Fraction(v) for v in n[k].tolist()]
        ### (sin of the angle, squared, exactly: |n x n*|^2 / (|n|^2 |n*|^2))
        cr = [
            nk[1] * exact[2] - nk[2] * exact[1],
            nk[2] * exact[0] - nk[0] * exact[2],
            nk[0] * exact[1] - nk[1] * exact[0],
        ]
        sin2 = sum(x * x for x in cr) / (
            sum(x * x for x in nk) * sum(x * x for x in exact)
        )
        assert sin2 <= Fraction(float(delta[k])) ** 2
    unsure = delta > torch.finfo(torch.float32).eps / 2
    assert not unsure[rho >= 1e-6].any() and unsure[rho <= 1e-12].all()


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, D])
def test_switch_on_ordinary_cell_changes_nothing_visible(dtype, device, monkeypatch):
    """On a large cell of ordinary shape (rho about 0.4, as DrivAerML's tunnel floor)
    the guard covers on-wall queries, and the switch to the accurate point fires
    where the two points differ by rounding alone. The search's internal weights then
    differ by an ulp or two, but d, u and their gradients in the queries and the
    vertices are bitwise those of the unswitched path."""
    p0 = torch.tensor(
        [[0.2, -0.6, -0.0171], [1.1, -0.4, -0.0171], [0.5, 0.3, -0.0171]],
        dtype=dtype,
        device=device,
    )
    c = torch.tensor([[0, 1, 2]], device=device)
    g = torch.Generator().manual_seed(0)
    b = torch.rand(4000, 3, dtype=D, generator=g)
    q0 = ((b / b.sum(1, keepdim=True)) @ p0.double().cpu()).to(dtype).to(device)
    search = _search._closest_features_op
    weights = []

    def recorded(*args):
        out = search(*args)
        weights.append(out[5].clone())
        return out

    monkeypatch.setattr(_search, "_closest_features_op", recorded)

    def run():
        q, p = q0.clone().requires_grad_(), p0.clone().requires_grad_()
        d, u = exact_sdf(q, p, c)
        probe = (
            torch.randn(u.shape, generator=torch.Generator().manual_seed(1), dtype=D)
            .to(dtype)
            .to(device)
        )
        gq, gp = torch.autograd.grad(d.sum() + (u * probe).sum(), (q, p))
        return d, u, gq, gp

    switched = run()
    monkeypatch.setattr(_weights, "_SWITCH_UNITS", float("inf"))
    unswitched = run()
    assert bool((weights[0] != weights[1]).any(-1).any())
    for a, b_ in zip(switched, unswitched):
        assert torch.equal(a, b_)
