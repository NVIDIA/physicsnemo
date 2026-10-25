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

"""The exact signed distance on surfaces with zero-area cells.

A zero-area cell (two or three coincident vertices, or three collinear ones) is the
segment or the point its vertices span: the distance is to that set, it has no
normal and no solid angle, so adding such cells to a surface changes nothing, and a
query held by zero-area cells alone has ``u = 0``. The coordinates lie on a dyadic
lattice, so the collinear cells stay exactly collinear in float32.
"""

import pytest
import torch

from physicsnemo.mesh.primitives.surfaces import cube_surface
from physicsnemo.mesh.spatial.exact_sdf import _predicates
from test.mesh.spatial._exact_sdf_meshes import exact_sdf

D = torch.float64
L = 2.0**-10
CASES = ("two coincident", "three coincident", "three collinear")

_DEVICES = [
    "cpu",
    pytest.param(
        "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")
    ),
]


def _lattice(g, *shape):
    return torch.randint(0, 1024, shape, generator=g).to(D) * L


def _zero_area_cells(g, n, case):
    """``n`` zero-area triangles ``(n, 3, 3)`` of the case, their vertices shuffled."""
    v = _lattice(g, n, 3)
    s = torch.randint(-64, 65, (n, 3), generator=g).to(D) * L
    if case == "two coincident":
        t = torch.stack([v, v, v + 8 * s], 1)
    elif case == "three coincident":
        t = torch.stack([v, v, v], 1)
    else:
        t = torch.stack([v, v + 4 * s, v + 12 * s], 1)
    perm = torch.stack([torch.randperm(3, generator=g) for _ in range(n)])
    return t.gather(1, perm[:, :, None].expand(-1, -1, 3))


def _segment_d2(q, a, b):
    ab = b - a
    l2 = (ab * ab).sum(-1)
    t = torch.where(l2 > 0, ((q - a) * ab).sum(-1) / torch.where(l2 > 0, l2, 1.0), 0.0)
    r = q - a - t.clamp(0, 1)[..., None] * ab
    return (r * r).sum(-1)


def _brute_distance(q, tri):
    """The distance from each query to the union of the triangles, float64: the
    least over their edges and, where the projection falls inside a triangle of
    nonzero area, its plane."""
    q = q.double()[:, None]
    a, b, c = (tri.double()[None, :, k] for k in range(3))
    d2 = torch.minimum(
        torch.minimum(_segment_d2(q, a, b), _segment_d2(q, b, c)), _segment_d2(q, a, c)
    )
    e1, e2, r = b - a, c - a, q - a
    n = torch.linalg.cross(e1.expand_as(r), e2.expand_as(r), dim=-1)
    g11, g12, g22 = (e1 * e1).sum(-1), (e1 * e2).sum(-1), (e2 * e2).sum(-1)
    det = g11 * g22 - g12 * g12
    ok = det > 1e-24
    r1, r2 = (r * e1).sum(-1), (r * e2).sum(-1)
    s = (g22 * r1 - g12 * r2) / torch.where(ok, det, 1.0)
    t = (g11 * r2 - g12 * r1) / torch.where(ok, det, 1.0)
    inside = ok & (s >= 0) & (t >= 0) & (s + t <= 1)
    h2 = (r * n).sum(-1) ** 2 / torch.where(ok, (n * n).sum(-1), 1.0)
    return torch.where(inside, torch.minimum(d2, h2), d2).min(1).values.sqrt()


def _box_with(case):
    """A closed box on the lattice, and its cells with zero-area cells of the case
    added along its edges, at its vertices or over straight runs of two edges."""
    box = cube_surface.load().subdivide(levels=2, filter="linear")
    p, c = box.points.double(), box.cells
    if case == "two coincident":
        extra = c[::8][:, [0, 0, 1]]
    elif case == "three coincident":
        extra = c[::8][:, [0, 0, 0]]
    else:
        at = {tuple(x): k for k, x in enumerate(p.tolist())}
        runs = []
        for a, m in torch.cat([c[:, [0, 1]], c[:, [1, 2]], c[:, [2, 0]]]).tolist():
            b = at.get(tuple((2 * p[m] - p[a]).tolist()))
            if b is not None:
                runs.append((a, b, m))
        extra = torch.tensor(runs[::7])
    return p, c, torch.cat([c, extra])


@pytest.mark.parametrize("dtype", [torch.float32, D])
def test_zero_area_flag(dtype):
    """The flag is exact: zero area for coincident and collinear vertices, nonzero
    for a triangle one lattice step off its line and for a tiny one."""
    a, b = (
        torch.tensor([0.25, 0.5, 0.75], dtype=D),
        torch.tensor([0.75, -0.5, 1.25], dtype=D),
    )
    off = torch.tensor([0.0, 0.0, 2.0**-20], dtype=D)
    tri = torch.stack(
        [
            torch.stack([a, a, b]),
            torch.stack([b, a, a]),
            torch.stack([a, a, a]),
            torch.stack([a, 0.5 * (a + b), b]),
            torch.stack([a, a + 0.25 * (b - a), b]),
            torch.stack([a, 0.5 * (a + b) + off, b]),
            torch.stack([a, b, a + off]) * 2.0**-60,
            torch.stack([a, b, torch.tensor([1.0, 0.0, 0.0], dtype=D)]),
        ]
    ).to(dtype)
    cells = torch.arange(24).reshape(8, 3)
    flat, any_flat = _predicates._zero_area(tri.double(), cells)
    assert any_flat
    assert flat.tolist() == [True] * 5 + [False] * 3


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, D])
@pytest.mark.parametrize("case", CASES)
def test_zero_area_cells_distance(case, dtype, device):
    """On a soup with zero-area cells, isolated, the distance is the brute force's
    to their segments and points, within the rounding of its precision, at queries
    on them, near them and far; nothing is non-finite."""
    g = torch.Generator().manual_seed(CASES.index(case))
    deg = _zero_area_cells(g, 12, case)
    tri = torch.cat([deg, _lattice(g, 60, 3, 3)])
    on = torch.cat([deg.reshape(-1, 3), deg[:, 0] + 0.5 * (deg[:, 2] - deg[:, 0])])
    q = torch.cat(
        [
            on,
            on + 1e-3 * torch.randn(on.shape[0], 3, dtype=D, generator=g),
            _lattice(g, 400, 3) * 1.4 - 0.2,
        ]
    )
    q, points, cells = (
        q.to(device, dtype),
        tri.reshape(-1, 3).to(device, dtype),
        torch.arange(216, device=device).reshape(72, 3),
    )
    qq, pp = q.clone().requires_grad_(), points.clone().requires_grad_()
    d, u = exact_sdf(qq, pp, cells)
    (d.sum() + u.sum()).backward()
    d, u, gq, gp = d.detach(), u.detach(), qq.grad, pp.grad
    assert all(torch.isfinite(t).all() for t in (d, u, gq, gp))
    exact = _brute_distance(q.cpu(), tri.to(dtype))
    scale = q.double().norm(dim=-1).cpu() + points.double().norm(dim=-1).max().cpu()
    assert (
        (d.abs().double().cpu() - exact).abs() <= 8 * torch.finfo(dtype).eps / 2 * scale
    ).all()


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, D])
@pytest.mark.parametrize("case", CASES)
def test_zero_area_cells_change_nothing(case, dtype, device):
    """Zero-area cells added to a closed surface, along its edges, at its vertices
    or over a straight run of two edges, change neither d nor u: bitwise, at its
    vertices, edge midpoints and centroids (on the wall), near them and inside and
    outside it, offset or not."""
    p, base, cells = _box_with(case)
    assert _predicates._zero_area(p[cells].double(), cells)[0][base.shape[0] :].all()
    v = p[base]
    g = torch.Generator().manual_seed(0)
    on = torch.cat([p, v[:, [0, 1]].mean(1), v.mean(1)])
    q = torch.cat(
        [
            on,
            on + 1e-3 * torch.randn(on.shape[0], 3, dtype=D, generator=g),
            torch.rand(500, 3, dtype=D, generator=g) * 2.4 - 1.2,
        ]
    )
    for shift in (0.0, 1024.0):
        args = (q + shift).to(device, dtype), (p + shift).to(device, dtype)
        d0, u0 = exact_sdf(*args, base.to(device))
        d, u = exact_sdf(*args, cells.to(device))
        assert torch.equal(d, d0) and torch.equal(u, u0)


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, D])
def test_query_on_zero_area_cells_alone(dtype, device):
    """A query held only by zero-area cells (a segment and a point beside a box)
    is on the wall with no normal: d = 0 and u = 0. Off them, d is the distance
    and u the unit direction from the segment."""
    box = cube_surface.load()
    p = torch.cat(
        [
            box.points.double(),
            torch.tensor([[1.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=D),
        ]
    )
    n = box.points.shape[0]
    cells = torch.cat([box.cells, torch.tensor([[n, n, n + 1], [n + 2, n + 2, n + 2]])])
    q = torch.tensor(
        [
            [1.0, 0, 0],
            [1.25, 0, 0],
            [1.5, 0, 0],
            [0, 1, 0],
            [1.25, 0.25, 0],
            [1.75, 0, 0],
        ],
        dtype=D,
    )
    d, u = exact_sdf(q.to(device, dtype), p.to(device, dtype), cells.to(device))
    d, u = d.cpu().double(), u.cpu().double()
    assert (d[:4] == 0).all() and (u[:4] == 0).all()
    assert torch.allclose(d[4:], torch.tensor([0.25, 0.25], dtype=D))
    assert torch.allclose(u[4:], torch.tensor([[0.0, 1, 0], [1, 0, 0]], dtype=D))


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, D])
def test_zero_length_segments_2d(dtype, device):
    """In two dimensions a zero-length segment is its point: one at a vertex of a
    closed square changes nothing; an isolated one holds its query with d = 0 and
    u = 0."""
    p = torch.tensor([[0.0, 0], [1, 0], [1, 1], [0, 1], [2, 2]], dtype=D)
    base = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0]])
    cells = torch.cat([base, torch.tensor([[1, 1], [4, 4]])])
    g = torch.Generator().manual_seed(0)
    q = torch.cat([p[:4], torch.rand(300, 2, dtype=D, generator=g) * 1.6 - 0.3])
    args = q.to(device, dtype), p.to(device, dtype)
    d0, u0 = exact_sdf(*args, base.to(device))
    d, u = exact_sdf(*args, cells[:5].to(device))
    assert torch.equal(d, d0) and torch.equal(u, u0)
    d, u = exact_sdf(p[4:].to(device, dtype), p.to(device, dtype), cells.to(device))
    assert (d == 0).all() and (u == 0).all()
