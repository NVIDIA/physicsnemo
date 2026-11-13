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

"""The exact boundary geometry on surfaces with thin cells of nonzero area.

A sliver, a cap or a needle (``rho = 2 A / L^2`` far below 1) is a cell like any
other: the distance, the closest cell and whether a query is on the wall are those
of an exact rational brute force, in float32 and float64, at queries on its face,
just off it and far; and a coplanar cap that splits a face of a closed surface
changes nothing.
"""

from fractions import Fraction

import pytest
import torch

from physicsnemo.experimental.models.isla import _geometry as geometry
from physicsnemo.experimental.models.isla._geometry import exact_boundary_geometry
from physicsnemo.mesh.geometry import compute_cell_normals
from test.experimental.models.isla._geometry_cases import (
    _capped_cube,
    _exact_d2,
    _face_queries,
    _rotation,
    _soup,
)
from test.experimental.models.isla._marks import DEVICES

D = torch.float64
U64 = 2.0**-53


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


def _search(q, p, c, kernel):
    """The search op's closest cell (a row of ``c``) and on-wall decision."""
    cells, parity = geometry._canonical_cells(p, geometry._weld(p)[c])
    s = p[cells].double()
    n64 = parity[:, None].double() * compute_cell_normals(s[:, 1:] - s[:, :1])
    op = {
        "reference": geometry._closest_candidates,
        "fast": geometry._closest_candidates_fast,
    }[kernel]
    idx, _, _, _, on_wall, _ = op(q, p, cells, parity, n64)
    return idx, on_wall


def _check(q, p, c, device):
    """Against the brute force: ``|d|`` within ``8 u (|x| + R)`` off the wall, the
    closest cell within the op's tie bound of the closest distance, and the query on
    the wall exactly when the brute force's distance says so (below ``E / 2`` it must
    be, beyond ``2 E`` it must not, ``E = u (|x| + R) + C u64 (|x'| + R')``, primes
    in the boundary's centred frame, ``C`` the derived ``_BAND_UNITS``); and on CUDA
    the fast search gives the reference's outputs and gradients bitwise (off CUDA it
    runs the reference's code)."""
    dtype = q.dtype
    u = torch.finfo(dtype).eps / 2
    tri = p.double()[c]
    exact, holders = _brute(q, tri)
    out = {}
    for kernel in ("reference", "fast") if device == "cuda" else ("reference",):
        qq, pp = (
            q.to(device).clone().requires_grad_(),
            p.to(device).clone().requires_grad_(),
        )
        d, uu = exact_boundary_geometry(qq, pp, c.to(device), kernel=kernel)
        (d.sum() + uu.sum()).backward()
        with torch.no_grad():
            idx, on_wall = _search(q.to(device), p.to(device), c.to(device), kernel)
        out[kernel] = (d.detach(), uu.detach(), qq.grad, pp.grad, idx, on_wall)
    for a, b in zip(out["reference"], out.get("fast", ())):
        assert torch.equal(a, b)
    d, _, _, _, idx, on_wall = (t.cpu() for t in out["reference"])
    xs = q.double().norm(dim=-1)
    reach = tri.norm(dim=-1).amax(-1)
    origin = geometry._centroid(tri).to(dtype).double()
    xc = (q.double() - origin).norm(dim=-1)
    reach_c = (tri - origin).norm(dim=-1).amax(-1)
    share = geometry._BAND_UNITS[3] * U64

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


@pytest.mark.parametrize("device", DEVICES)
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


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("h", [1e-16, 1e-13])
@pytest.mark.parametrize("seed", [0, 1])
def test_sliver_soups(seed, h, device):
    """Float64 sliver soups, whose slivers lie below the certainty of
    their float64 normal: the search keeps their cells, and the answers are the
    brute force's."""
    p, c, q = _soup(seed, h)
    _check(q, p, c, device)


@pytest.mark.parametrize("device", DEVICES)
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
        d0, _ = exact_boundary_geometry(*args, base.to(device))
        d, _ = exact_boundary_geometry(*args, c.to(device))
        scale = (q + shift).norm(dim=-1) + (p + shift).norm(dim=-1).max()
        qs, ps = (x.double().cpu() for x in args)
        origin = geometry._centroid(ps[c]).to(dtype).double()
        scale_c = (qs - origin).norm(dim=-1) + (ps - origin).norm(dim=-1).max()
        d, d0 = d.double().cpu(), d0.double().cpu()
        wall = (d == 0) | (d0 == 0)
        share = geometry._BAND_UNITS[3] * U64
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
    delta, _, _ = geometry._thin_bounds(tri)
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


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, D])
def test_switch_on_ordinary_cell_changes_nothing_visible(dtype, device, monkeypatch):
    """On a large cell of ordinary shape (rho about 0.4, as DrivAerML's tunnel floor)
    the guard covers on-wall queries, and the switch to the accurate point fires
    where the two points differ by rounding alone. The search's internal weights then
    differ by an ulp or two, but d, u and their gradients in the queries are bitwise
    those of the unswitched path. So is the gradient in the vertices, but on the wall:
    there it is the plane distance's, which reads the weights, and agrees to the
    rounding of its sum over 4,000 queries (64 ulps of the largest)."""
    p0 = torch.tensor(
        [[0.2, -0.6, -0.0171], [1.1, -0.4, -0.0171], [0.5, 0.3, -0.0171]],
        dtype=dtype,
        device=device,
    )
    c = torch.tensor([[0, 1, 2]], device=device)
    g = torch.Generator().manual_seed(0)
    b = torch.rand(4000, 3, dtype=D, generator=g)
    q0 = ((b / b.sum(1, keepdim=True)) @ p0.double().cpu()).to(dtype).to(device)
    search = geometry._closest_candidates
    weights = []

    def recorded(*args):
        out = search(*args)
        weights.append(out[5].clone())
        return out

    monkeypatch.setattr(geometry, "_closest_candidates", recorded)

    def run():
        q, p = q0.clone().requires_grad_(), p0.clone().requires_grad_()
        d, u = exact_boundary_geometry(q, p, c)
        probe = (
            torch.randn(u.shape, generator=torch.Generator().manual_seed(1), dtype=D)
            .to(dtype)
            .to(device)
        )
        gq, gp = torch.autograd.grad(d.sum() + (u * probe).sum(), (q, p))
        return d, u, gq, gp

    switched = run()
    monkeypatch.setattr(geometry, "_SWITCH_UNITS", float("inf"))
    unswitched = run()
    assert bool((weights[0] != weights[1]).any(-1).any())
    for a, b_ in zip(switched[:3], unswitched[:3]):
        assert torch.equal(a, b_)
    gp, gp_ref = switched[3], unswitched[3]
    ulps = 64 * torch.finfo(dtype).eps * float(gp_ref.abs().max())
    assert torch.allclose(gp, gp_ref, rtol=0, atol=ulps)
