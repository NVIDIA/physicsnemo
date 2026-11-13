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

"""Geometry cases shared by the ISLA geometry tests: closed surfaces and polygons
with outward normals, an airfoil section, a folded fin, boxes with points at ties and
on the wall, thin cells and sliver soups with their exact squared distance, queries
near a wall, and the small interior model of the interior-geometry tests; also the
search kernels' test parameters, a renumbering of a surface and a bitwise
comparison."""

from fractions import Fraction

import pytest
import torch

from physicsnemo.experimental.models.isla import ISLA
from physicsnemo.mesh import Mesh
from physicsnemo.mesh.primitives.curves import circle_2d
from physicsnemo.mesh.primitives.surfaces import cube_surface, sphere_icosahedral, torus
from test.experimental.models.isla._marks import requires_cuda
from test.experimental.models.isla._meshes import orthogonal

D = torch.float64
PV = {"p": {"rank": 0}, "v": {"rank": 1}}
_CUDA = [pytest.mark.cuda, requires_cuda]
#: The ``kernel, device`` parameters of a test of both search kernels. Off CUDA the
#: fast kernel runs the reference's torch code, so it runs on CUDA only.
KERNEL_DEVICES = [
    ("reference", "cpu"),
    pytest.param("reference", "cuda", marks=_CUDA),
    pytest.param("fast", "cuda", marks=_CUDA),
]
#: The ``device`` parameter of a test of the fast kernel against the reference.
FAST_DEVICES = [pytest.param("cuda", marks=_CUDA)]


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


def _interior(dim: int, **kw) -> ISLA:
    torch.manual_seed(0)
    kw.setdefault("boundary_schemas", {"wall": {}, "airfoil": {}, "far": {}})
    return (
        ISLA(
            output_schema=PV,
            global_schema={"g": {"rank": 1}},
            hidden=32,
            n_layers=2,
            n_slices=8,
            interior=True,
            n_spatial_dims=dim,
            **kw,
        )
        .double()
        .eval()
    )


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


def _soup(seed, h):
    """A sliver soup, smaller: random triangles in the unit cube, the
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


def _renumbered(p, c, seed: int = 0):
    """The surface with its vertices and its cells permuted, each triangle's vertex
    list rotated (a segment's order is kept: reversing it flips its normal)."""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(p.shape[0], generator=g)
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(p.shape[0])
    c2 = inv[c][torch.randperm(c.shape[0], generator=g)]
    return p[perm], c2.roll(1, -1) if c.shape[1] == 3 else c2


def _bitwise(a, b) -> bool:
    """Whether two tensors are the same bit for bit, dtype included (``-0`` and NaN
    payloads too, where ``torch.equal`` passes ``-0 == 0`` and fails on NaN)."""
    if a.dtype != b.dtype:
        return False
    if a.is_floating_point():
        return torch.equal(a.view(torch.uint8), b.view(torch.uint8))
    return torch.equal(a, b)


def _quad_diagonals(case: str):
    """Float32 queries at the midpoints of a posed torus's quad diagonals (the edge
    each quad's two triangles share, which near-duplicate normals meet across), the
    torus's points and its cells: the 48 x 24 torus (``renumbering``), or the split
    torus of test_on_wall_direction_is_a_function_of_the_input_mesh (``precision``)."""
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


def _fan(n: int = 96):
    """A fan of ``n`` triangles around a hub, 50 queries above the hub (``n``
    candidates each, more than the 64 slots at the default 96) and 400 elsewhere."""
    t = torch.arange(n, dtype=D) * (2 * torch.pi / n)
    rim = torch.stack([torch.cos(t), torch.sin(t), 0 * t], -1)
    points = torch.cat([rim.new_zeros(1, 3), rim])
    i = torch.arange(1, n + 1)
    cells = torch.stack([torch.zeros_like(i), i, i % n + 1], -1)
    g = torch.Generator().manual_seed(0)
    hub = torch.linspace(0.01, 0.5, 50, dtype=D)[:, None] * torch.tensor([0.0, 0, 1])
    q = torch.cat([hub, torch.rand(400, 3, dtype=D, generator=g) * 3 - 1.5])
    return q, points, cells
