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

"""Gradient checks of ISLA's differentiable geometry and of its custom ops and
autograd Functions.

- ``torch.autograd.gradcheck`` in float64, against central finite differences, where
  the function takes float64: the exact boundary geometry (CPU), the fused geometry
  region's custom op (CUDA), and the fused read's basis-gradient sink (a CPU stand-in
  read). Finite differences use ``eps = 1e-6``; their truncation error is about
  ``eps**2 |f'''|`` (below 1e-9 here) and their roundoff about ``1e-16 / eps``, so the
  analytical and numerical Jacobians are compared at ``atol = 1e-7, rtol = 1e-6``.
  The exact geometry's checks run gradcheck's fast mode (a seeded random projection
  ``v^T J u`` of the Jacobian, from a few evaluations instead of one per input): its
  Jacobian is dense in the vertices, and the full one costs a minute.
- ``torch.library.opcheck`` on every custom op (schema, autograd registration, fake
  tensor and AOT dispatch with dynamic shapes), on the devices each supports.
- The Triton autograd Functions refuse float64, so each is held to the 2x-error rule
  against float64 eager autograd instead: per tensor, the fused path's distance to
  float64 is at most twice the eager path's in the same precision plus a floor at the
  precision's roundoff. Those tests are in test_isla_tree_read.py (the fused read, the
  fused basis and the node means) and test_isla_kernel_refs.py; here, gradcheck of
  the float64 eager references they measure against.

The geometry is checked at generic points: off the wall and off ties, each in one
region of its closest feature (a cell's interior, an edge, a vertex), so the finite
differences do not cross a region's boundary.
"""

import pytest
import torch

### The modules that own each function checked here.
from physicsnemo.experimental.models.isla import _fused_read as fused_read
from physicsnemo.experimental.models.isla import _geo_kernel as geo_kernel
from physicsnemo.experimental.models.isla import _geometry as geometry
from physicsnemo.experimental.models.isla import _tree_read as tree_read
from physicsnemo.experimental.models.isla._geometry import exact_boundary_geometry
from physicsnemo.mesh.geometry import compute_cell_normals
from test.experimental.models.isla._geometry_cases import _circle, _sphere
from test.experimental.models.isla._kernel_cases import _SinkRead
from test.experimental.models.isla._marks import DEVICES, requires_cuda_triton

D = torch.float64
GRADCHECK = dict(eps=1e-6, atol=1e-7, rtol=1e-6, check_batched_grad=False)


@pytest.fixture
def one_thread():
    """One CPU thread: the vertex gradient sums over the cells, in an order that
    depends on the thread count, and gradcheck asks two backward passes to agree."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


# --------------------------------------------------------------------------- exact geometry
def _regions(dim):
    """A closed convex surface (a 16-gon, an icosphere of 42 vertices) and generic
    queries 0.1 off it, by the region of their closest feature: a cell's interior
    (outside and inside, along the cell's normal from its centroid), a vertex
    (radially out of it, inside its normal cone) and, in 3-D, an edge (radially out
    of its midpoint, between its two cells' normals), every few of each."""
    mesh = _circle(16) if dim == 2 else _sphere(1)
    n = mesh.cell_normals
    regions = {
        "cell": torch.cat(
            [mesh.cell_centroids + 0.1 * n, mesh.cell_centroids - 0.1 * n]
        ),
        "vertex": 1.1 * mesh.points,
    }
    if dim == 3:
        e = (
            torch.cat(
                [mesh.cells[:, [0, 1]], mesh.cells[:, [1, 2]], mesh.cells[:, [2, 0]]]
            )
            .sort(-1)
            .values.unique(dim=0)
        )
        mid = mesh.points[e].mean(1)
        regions["edge"] = 1.1 * mid / mid.norm(dim=-1, keepdim=True)
    ### About 12 queries of each region (a failing gradcheck's report builds the full
    ### Jacobian).
    return mesh, {k: q[:: max(1, q.shape[0] // 12)] for k, q in regions.items()}


def _feature_size(q, mesh, monkeypatch):
    """The number of nonzero barycentric weights of each query's closest point (the
    search's own weights, read through a spy): ``dim`` in a cell's interior, 2 on an
    edge, 1 at a vertex."""
    seen = []
    search = geometry._closest_candidates

    def spy(*args):
        seen.append(search(*args))
        return seen[-1]

    with monkeypatch.context() as m:
        m.setattr(geometry, "_closest_candidates", spy)
        exact_boundary_geometry(q, mesh.points, mesh.cells)
    return (seen[0][5] > 0).sum(-1)


@pytest.mark.parametrize("dim", [2, 3])
def test_the_gradcheck_queries_lie_in_the_regions_they_name(dim, monkeypatch):
    """Each query set of :func:`_regions` has its closest point where it is said to:
    the search's weights have ``dim`` nonzero entries in a cell's interior, 2 on an
    edge, 1 at a vertex (so the gradchecks below test the region they name)."""
    mesh, regions = _regions(dim)
    want = {"cell": dim, "edge": 2, "vertex": 1}
    for name, q in regions.items():
        assert (_feature_size(q, mesh, monkeypatch) == want[name]).all(), name


def _check(fn, x):
    """gradcheck in fast mode, raising with the full Jacobians on a mismatch."""
    torch.manual_seed(0)  # the fast mode's random projections
    x = x.clone().requires_grad_(True)
    return torch.autograd.gradcheck(fn, (x,), fast_mode=True, **GRADCHECK)


@pytest.mark.parametrize("wrt", ["queries", "vertices"])
@pytest.mark.parametrize("dim", [2, 3])
def test_signed_distance_passes_gradcheck_in_every_region(dim, wrt, one_thread):
    """d is differentiable in the queries and the vertex positions with the gradient
    of the envelope theorem, which is d's derivative: gradcheck in float64 at
    generic queries of every region."""
    mesh, regions = _regions(dim)
    q = torch.cat(list(regions.values()))
    p, c = mesh.points, mesh.cells
    if wrt == "queries":
        assert _check(lambda x: exact_boundary_geometry(x, p, c)[0], q)
    else:
        assert _check(lambda x: exact_boundary_geometry(q, x, c)[0], p)


@pytest.mark.parametrize("wrt", ["queries", "vertices"])
@pytest.mark.parametrize("dim", [2, 3])
def test_signed_distance_keeps_its_gradient_on_the_wall(dim, wrt, one_thread):
    """A query on the wall (a cell's centroid) takes d = 0 exactly, with the gradient
    of the distance to the cell's plane: u, the cell's unit normal, in the query, as
    beside the wall; gradcheck in float64 in the queries and the vertex positions."""
    mesh = _circle(16) if dim == 2 else _sphere(1)
    q, p, c = mesh.cell_centroids[::3], mesh.points, mesh.cells
    x = q.clone().requires_grad_(True)
    d, u = exact_boundary_geometry(x, p, c)
    assert torch.equal(d, torch.zeros_like(d))
    grad = torch.autograd.grad(d.sum(), x)[0]
    assert torch.allclose(grad, u, rtol=0, atol=1e-12)
    assert torch.allclose(u, mesh.cell_normals[::3], rtol=0, atol=1e-12)
    if wrt == "queries":
        assert _check(lambda x: exact_boundary_geometry(x, p, c)[0], q)
    else:
        assert _check(lambda x: exact_boundary_geometry(q, x, c)[0], p)


def _direction_cases():
    for dim, region in (
        (2, "cell"),
        (2, "vertex"),
        (3, "cell"),
        (3, "vertex"),
        (3, "edge"),
    ):
        for wrt in ("queries", "vertices"):
            yield pytest.param(dim, region, wrt, id=f"{dim}d-{region}-{wrt}")


@pytest.mark.parametrize("dim, region, wrt", list(_direction_cases()))
def test_direction_passes_gradcheck(dim, region, wrt, one_thread):
    """u is differentiable in the queries and the vertex positions: gradcheck in
    float64 at generic queries of one region (a cell's interior, where u is the
    cell's normal; a vertex, where the closest point is fixed; an edge in 3-D,
    where it slides along the edge with the query and the vertices, and u takes
    the gradient of that motion, geometry._closest_features' sliding direction)."""
    mesh, regions = _regions(dim)
    q, p, c = regions[region], mesh.points, mesh.cells
    if wrt == "queries":
        assert _check(lambda x: exact_boundary_geometry(x, p, c)[1], q)
    else:
        assert _check(lambda x: exact_boundary_geometry(q, x, c)[1], p)


def _triangle_queries():
    """A triangle and queries off it in its edge region (the point (0.5, -1, 0) and a
    generic one), a vertex region and its interior, each away from the regions'
    boundaries."""
    g = torch.Generator().manual_seed(3)
    tri = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=D)
    generic = tri + 0.3 * torch.randn(3, 3, dtype=D, generator=g)
    a, b, c = generic.unbind(0)
    n = torch.linalg.cross(b - a, c - a)
    n = n / n.norm()
    out = torch.linalg.cross(b - a, n)  # in the plane, away from c
    out = out / out.norm() * (1 if (out * (a - c)).sum() > 0 else -1)
    return [
        (tri, torch.tensor([0.5, -1.0, 0.0], dtype=D)),
        (generic, a + 0.4 * (b - a) + 0.7 * out + 0.3 * n),
        (generic, a + 0.5 * (a - b) + 0.5 * (a - c) - 0.4 * n),
        (generic, (a + b + c) / 3 + 0.6 * n),
    ]


@pytest.mark.parametrize(
    "case", range(4), ids=["edge example", "edge", "vertex", "interior"]
)
def test_direction_jacobian_matches_finite_differences(case, one_thread):
    """The full Jacobian of d and u in the query and the vertices of one triangle
    matches central finite differences in float64: off its edge (where the closest
    point slides along it, so du_x/dx_x = 0 at (0.5, -1, 0)), its vertex and its
    interior."""
    tri, x = _triangle_queries()[case]
    cells = torch.tensor([[0, 1, 2]])

    def f(flat):
        d, u = exact_boundary_geometry(flat[None, :3], flat[3:].reshape(3, 3), cells)
        return torch.cat([d, u[0]])

    flat = torch.cat([x, tri.reshape(-1)])
    jac = torch.autograd.functional.jacobian(f, flat)
    h = 1e-6
    fd = torch.stack(
        [(f(flat + h * e) - f(flat - h * e)) / (2 * h) for e in torch.eye(12, dtype=D)],
        1,
    )
    assert torch.allclose(jac, fd, atol=1e-7, rtol=0)


# --------------------------------------------------------------------------- custom ops
def _search_args(dim, dtype, device):
    mesh, regions = _regions(dim)
    p = mesh.points.to(device, dtype)
    cells, parity = geometry._canonical_cells(
        p, geometry._weld(p)[mesh.cells.to(device)]
    )
    s = p[cells].double()
    n64 = parity[:, None].double() * compute_cell_normals(s[:, 1:] - s[:, :1])
    q = torch.cat(list(regions.values())).to(device, dtype)
    return q, p, cells, parity, n64


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize(
    "name", ["isla_closest_candidates", "isla_closest_candidates_fast"]
)
def test_closest_candidates_ops_pass_opcheck(name, dim, dtype, device):
    """The geometry search's two custom ops: schema, autograd registration (none:
    the search runs without gradient), the fake implementation's shapes and dtypes
    against the real outputs, and AOT dispatch with dynamic shapes."""
    op = getattr(torch.ops.physicsnemo, name).default
    torch.library.opcheck(op, _search_args(dim, dtype, device))


def _region_inputs(dtype, k, p, b=1, n=6, s=5, seed=0, device="cuda"):
    """Small inputs of the geometry region (on CUDA for the fused one): pre-logits,
    the points' anchors and normals, ``k`` global directions plus ``p`` token vectors,
    the slices' anchors and directions, and the routing Linear's weight and bias."""
    g = torch.Generator().manual_seed(seed)

    def rnd(*shape, scale=1.0, unit=False):
        x = torch.randn(*shape, generator=g, dtype=D)
        x = torch.nn.functional.normalize(x, dim=-1) if unit else scale * x
        return x.to(device, dtype)

    return [
        rnd(b, n, s),
        rnd(b, n, 3, scale=0.3),
        rnd(b, n, 3, unit=True),
        torch.cat([rnd(b, n, k, 3, unit=True), rnd(b, n, p, 3, scale=0.7)], 2),
        rnd(b, s, 3, scale=0.3),
        rnd(b, s, 3, unit=True),
        rnd(1, 5 + k + p, scale=0.5),
        rnd(1, scale=0.5),
    ]


_KP = pytest.mark.parametrize(
    "k, p", [(1, 0), (2, 1), (1, 3)], ids=["K1P0", "K2P1", "K1P3"]
)


@pytest.mark.cuda
@requires_cuda_triton
@_KP
def test_fused_geo_region_op_passes_gradcheck_in_float64(k, p):
    """``physicsnemo::isla_geo_region_fwd`` with its registered backward
    (``isla_geo_region_bwd``), float64: every output (routing bias, mix, pooled
    invariants) with respect to every tensor input, against finite differences."""
    ins = [t.requires_grad_(True) for t in _region_inputs(D, k, p)]

    def region(*xs):
        return geo_kernel._geo_region_fwd(*xs, 1e-12, False, k)

    assert torch.autograd.gradcheck(region, tuple(ins), **GRADCHECK)


@pytest.mark.cuda
@requires_cuda_triton
@_KP
def test_fused_geo_region_backward_skips_what_autograd_does_not_ask_for(k, p):
    """The fused backward leaves out the gradients of the positions, normals and
    per-point vectors when they require none, and reads no bias gradient when the
    routing bias is dropped: the gradients that remain are the full backward's
    (float64, to roundoff), and the skipped ones are absent."""
    ins = _region_inputs(D, k, p)
    g = torch.Generator(device="cuda").manual_seed(1)
    w_mix = torch.randn(ins[0].shape, generator=g, device="cuda", dtype=D)
    w_pooled = torch.randn(
        *ins[0].shape[:2], 5 + k + p, generator=g, device="cuda", dtype=D
    )

    def grads(full: bool):
        xs = [
            t.clone().requires_grad_(full or i not in (1, 2, 3))
            for i, t in enumerate(ins)
        ]
        bias, mix, pooled = geo_kernel._geo_region_fwd(*xs, 1e-12, False, k)
        loss = (mix * w_mix).sum() + (pooled * w_pooled).sum()
        if full:  # the bias kept, with a zero gradient
            loss = loss + (bias * 0.0).sum()
        loss.backward()
        return [x.grad for x in xs]

    full, part = grads(True), grads(False)
    assert all(part[i] is None for i in (1, 2, 3))
    for i in (0, 4, 5, 6, 7):
        torch.testing.assert_close(part[i], full[i], rtol=1e-10, atol=1e-13)


@pytest.mark.cuda
@requires_cuda_triton
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@_KP
def test_fused_geo_region_ops_pass_opcheck(k, p, dtype):
    """The forward op (with gradients, as it is called in training) and the backward
    op (no autograd: it has no double backward), float32 and float64."""
    ins = _region_inputs(dtype, k, p)
    fwd = torch.ops.physicsnemo.isla_geo_region_fwd.default
    torch.library.opcheck(
        fwd, (*[t.clone().requires_grad_(True) for t in ins], 1e-12, False, k)
    )
    outs = fwd(*ins, 1e-12, False, k)
    g = torch.Generator(device="cuda").manual_seed(1)
    grads = [
        torch.randn(o.shape, generator=g, device="cuda", dtype=dtype) for o in outs
    ]
    torch.library.opcheck(
        torch.ops.physicsnemo.isla_geo_region_bwd.default,
        (*ins, *grads, 1e-12, False, k),
    )


# --------------------------------------------------------------------------- the basis sink
def test_basis_sink_passes_gradcheck():
    """``_BasisSink``: three reads of one basis (``inv * theta``) through its sink,
    float64; the sink sums their basis gradients and hands them to the basis once, so
    the gradients in theta and in each read's weights are the derivative's (repeated
    backward passes of one graph included, as gradcheck runs them)."""
    g = torch.Generator().manual_seed(8)
    inv = torch.randn(20, 6, generator=g, dtype=D)
    theta = torch.randn(6, generator=g, dtype=D).requires_grad_(True)
    ws = [
        torch.randn(4, 6, generator=g, dtype=D).requires_grad_(True) for _ in range(3)
    ]

    def reads(theta, *ws):
        basis = inv * theta
        sink = fused_read.basis_sink(basis)
        return sum(_SinkRead.apply(basis.detach(), *sink, w) for w in ws).square()

    assert torch.autograd.gradcheck(reads, (theta, *ws), **GRADCHECK)


# --------------------------------------------------------------------------- the references
def test_the_eager_references_pass_gradcheck():
    """The float64 references the 2x-error rules measure against: the eager
    segment-softmax read (in ker, logm, q, k and v, on a ragged plan with a point and
    a source without pairs), the eager geometry region (every tensor input, with a
    token vector) and ``node_means`` (in the cells' rows), on CPU."""
    from physicsnemo.experimental.models.isla._blocks import _geo_region
    from physicsnemo.experimental.models.isla._segment_softmax import (
        PairPlan,
        _eager_segment_softmax_read,
    )

    g = torch.Generator().manual_seed(2)
    counts = torch.tensor([3, 0, 5, 1, 4])
    qi = torch.repeat_interleave(torch.arange(5), counts)
    bj = torch.randint(0, 5, (int(counts.sum()),), generator=g)
    plan = PairPlan.build(qi, bj, 5, 6)
    ins = [
        torch.randn(*s, generator=g, dtype=D)
        for s in ((13, 2), (13,), (5, 2, 4), (6, 2, 4), (6, 2, 3))
    ]
    ins = [t.requires_grad_(True) for t in ins]
    assert torch.autograd.gradcheck(
        lambda *x: _eager_segment_softmax_read(*x, plan, 0.5), tuple(ins), **GRADCHECK
    )

    region = [
        t.cpu().requires_grad_(True) for t in _region_inputs(D, 1, 1, device="cpu")
    ]

    def eager_region(logits, r, nh, d, z, ms, w, b):
        lin = torch.nn.Linear(7, 1).double()
        del lin.weight, lin.bias
        lin.weight, lin.bias = w, b
        return _geo_region(lin, logits, r, nh, d[:, :, :1], z, ms, 1e-12, d[:, :, 1:])

    assert torch.autograd.gradcheck(eager_region, tuple(region), **GRADCHECK)

    pts = torch.randn(60, 3, generator=g, dtype=D)
    nrm = torch.nn.functional.normalize(
        torch.randn(60, 3, generator=g, dtype=D), dim=-1
    )
    w = torch.rand(60, generator=g, dtype=D) + 0.5
    omega = w / w.sum()
    forest = tree_read.BoundaryForest.build(
        pts, omega, nrm, None, theta=1.0, leaf_size=4
    )
    x = torch.randn(60, 5, generator=g, dtype=D).requires_grad_(True)
    assert torch.autograd.gradcheck(
        lambda x: tree_read.node_means(x, forest, omega), (x,), **GRADCHECK
    )


# --------------------------------------------------------------------------- the whole model
def _ellipsoid_model(interior=True):
    """An interior model at ``tree_read_theta=0`` in float64 with every parameter
    shifted by seeded noise (some layers start at zero, the reads among them), an
    ellipsoid's boundary and three prediction points outside it (``interior=False``:
    the same model for the surface task, at its default angle)."""
    from physicsnemo.experimental.models.isla import ISLA
    from physicsnemo.mesh import Mesh
    from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral

    s = sphere_icosahedral.load(subdivisions=2)
    points = s.points.to(D) * torch.tensor([2.0, 1.0, 0.5], dtype=D)
    model = ISLA(
        output_schema={"p": {"rank": 0}},
        boundary_schemas={"wall": {}},
        hidden=32,
        n_layers=2,
        n_slices=8,
        **(dict(interior=True, tree_read_theta=0.0) if interior else {}),
    ).to(D)
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.2 * torch.randn(p.shape, generator=g, dtype=D))
    x = torch.tensor([[3.1, 0.4, 0.3], [-0.5, 1.9, 0.2], [0.6, -0.3, 1.1]], dtype=D)
    return model, {"wall": Mesh(points=points, cells=s.cells)}, x


def _point_grad(model, meshes, x):
    """The derivative of the summed output in the prediction points (each output
    depends on its own point only, so row ``i`` is output ``i``'s gradient)."""
    x = x.clone().requires_grad_(True)
    p = model(boundary_meshes=meshes, prediction_points=x).point_data["p"]
    return torch.autograd.grad(p.sum(), x)[0]


def test_interior_outputs_point_derivative_matches_central_differences():
    """The whole interior model at ``tree_read_theta=0`` in float64: the autograd
    derivative of the outputs in the prediction points is the central difference's
    (``h = 1e-5``: truncation and roundoff both near 1e-10 of the largest) to 1e-8
    of the largest. The read's pair invariants carry their gradient; without it the
    two differ by about 1e-2 of the largest."""
    model, meshes, x0 = _ellipsoid_model()
    auto = _point_grad(model, meshes, x0)
    h, fd = 1e-5, torch.zeros_like(x0)
    with torch.no_grad():
        for i in range(x0.shape[1]):
            e = torch.zeros(x0.shape[1], dtype=D)
            e[i] = h
            hi = model(boundary_meshes=meshes, prediction_points=x0 + e)
            lo = model(boundary_meshes=meshes, prediction_points=x0 - e)
            fd[:, i] = (hi.point_data["p"] - lo.point_data["p"]) / (2 * h)
    assert float((auto - fd).abs().max()) <= 1e-8 * float(fd.abs().max())


@pytest.mark.parametrize("interior", [True, False], ids=["interior", "surface"])
def test_outputs_vertex_derivative_matches_central_differences(interior):
    """The whole model at ``tree_read_theta=0`` in float64, on an asymmetric closed
    body (an ellipsoid sheared along one axis: distinct variances and third moments,
    so no tie of its frame): the autograd derivative of the summed outputs in the
    boundary vertices, projected on seeded random directions, is the central
    difference's (``h = 1e-5``, truncation near 1e-7 of the largest) to 5e-7 of the
    largest. In the interior task every cell is a token, and its Horvitz-Thompson
    weight carries its measure's gradient; without it the two differ by orders of
    magnitude."""
    from physicsnemo.mesh import Mesh
    from test.experimental.models.isla._meshes import outputs

    model, meshes, x = _ellipsoid_model(interior)
    p, cells = meshes["wall"].points.clone(), meshes["wall"].cells
    p[:, 0] += 0.3 * p[:, 1] ** 2

    def f(points):
        wall = {"wall": Mesh(points=points, cells=cells)}
        if interior:
            out = model(boundary_meshes=wall, prediction_points=x)
        else:
            out = model(boundary_meshes=wall, prediction_boundaries=["wall"])
        return outputs(out)["p"].sum()

    pg = p.clone().requires_grad_(True)
    grad = torch.autograd.grad(f(pg), pg)[0]
    g, h = torch.Generator().manual_seed(0), 1e-5
    auto, fd = [], []
    for _ in range(3):
        v = torch.randn(p.shape, generator=g, dtype=D)
        auto.append(float((grad * v).sum()))
        with torch.no_grad():
            fd.append(float((f(p + h * v) - f(p - h * v)) / (2 * h)))
    auto, fd = torch.tensor(auto, dtype=D), torch.tensor(fd, dtype=D)
    assert float((auto - fd).abs().max()) <= 5e-7 * float(fd.abs().max())


def test_tree_read_basis_passes_gradcheck_in_the_geometry():
    """The tree read's kernel basis at ``theta=0`` (every pair a cell), float64 on
    CPU, in the points' positions, directions and wall distances and in the cells'
    positions and normals: its pair invariants carry their gradient when the geometry
    requires one."""
    from test.experimental.models.isla._meshes import cloud

    g = torch.Generator().manual_seed(3)
    pts, nrm, w, _ = cloud(30, seed=3)
    q = cloud(8, seed=7, scale=(4.0,) * 3)[0]
    basis = tree_read._KernelBasis(tree_read.kernel_geo_width(1, 0)).to(D)
    g_unit = torch.nn.functional.normalize(
        torch.randn(1, 1, 3, generator=g, dtype=D), dim=-1
    )

    def read(q_r, q_n, q_wall, b_r, b_n):
        pairs = tree_read.tree_read_pairs(
            basis, query_r=q_r[None], query_n=q_n[None], query_wall=q_wall[None], boundary_r=b_r[None],
            boundary_n=b_n[None], boundary_vecs=None, log_frac=torch.log(w / w.sum())[None], g_unit=g_unit,
            theta=0.0, eps=1e-12,
        )  # fmt: skip
        return pairs.basis

    ins = (
        q,
        torch.nn.functional.normalize(q, dim=-1),
        0.5 + torch.rand(8, generator=g, dtype=D),
        pts,
        nrm,
    )
    torch.manual_seed(0)  # the fast mode's random projections
    ins = tuple(t.clone().requires_grad_(True) for t in ins)
    assert torch.autograd.gradcheck(read, ins, fast_mode=True, **GRADCHECK)
