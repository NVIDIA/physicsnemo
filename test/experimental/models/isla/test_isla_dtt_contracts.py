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

"""Model-level contracts of ISLA's interior task (``interior=True``), with the
boundary read live.

Every model here goes through ``live`` (``_meshes.py``), which seeds the boundary
read's zero-initialized output projection, so each contract holds for a read that
contributes. The read walks a forest of cluster trees built in the boundary's own
frame at the opening angle ``TREE_READ_THETA``; the tests set the model's
``tree_read_theta`` to 0 for the exact read (every cell pair read). The in-model
draw keeps ``n_boundary_tokens`` of the cells when the boundary has more. Each test
states its property and the reason for its tolerance; every tolerance is absolute
(``rtol=0``) unless a relative term is the point.
"""

import logging

import pytest
import torch

import physicsnemo.experimental.models.isla.model as isla_model
from physicsnemo.experimental.models.isla import ISLA
from physicsnemo.experimental.models.isla._tree_read import TREE_READ_THETA
from physicsnemo.mesh import Mesh
from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral
from test.experimental.models.isla._marks import requires_cuda_triton
from test.experimental.models.isla._meshes import (
    cloud,
    law,
    live,
    orthogonal,
    outputs,
    points_mesh,
    same,
    soup,
    transform,
)

PV = {"p": {"rank": 0}, "v": {"rank": 1}}
G = {"g": {"rank": 1}}
D = torch.float64
#: Every rank and parity the heads declare (as test_isla_heads.py's ``OUTPUTS``).
EVERY_RANK = {
    "s": {"rank": 0},
    "v": {"rank": 1},
    "pv": {"rank": 1, "parity": "odd"},
    "t": {"rank": 2},
    "ts": {"rank": 2, "symmetric": True},
    "pt": {"rank": 2, "parity": "odd"},
    "t3": {"rank": 3},
    "t3o": {"rank": 3, "parity": "odd"},
    "t3s": {"rank": 3, "symmetric": True},
    "t4s": {"rank": 4, "symmetric": True},
}
THETAS = pytest.mark.parametrize(
    "theta", [0.0, TREE_READ_THETA], ids=["exact_read", "default_theta"]
)


def _model(dim=3, theta=TREE_READ_THETA, output_schema=PV, seed=0, **kw):
    """A float64 interior model in eval mode, its read live at ``theta``."""
    torch.manual_seed(seed)
    kw = {
        "hidden": 32,
        "n_layers": 2,
        "n_slices": 8,
        "boundary_schemas": {"wall": {}},
        **kw,
    }
    m = ISLA(
        output_schema=output_schema,
        global_schema=G,
        interior=True,
        n_spatial_dims=dim,
        tree_read_theta=theta,
        **kw,
    )
    return live(m.double().eval())


def _case(dim=3, n=300, nq=120, seed=0):
    """A generic anisotropic boundary soup of ``n`` cells, its global direction, and
    ``nq`` prediction points: near the cells, and ten far away."""
    pts, nrm, w, drv = cloud(n, d=dim, seed=seed)
    g = torch.Generator().manual_seed(seed + 1)
    near = pts[: nq - 10] + 0.15 * torch.randn(nq - 10, dim, generator=g, dtype=D)
    far = 40.0 + torch.randn(10, dim, generator=g, dtype=D)
    return soup(pts, nrm, w), drv, torch.cat([near, far])


def _run(m, mesh, drv, q):
    with torch.no_grad():
        return outputs(
            m(
                boundary_meshes={"wall": mesh},
                prediction_points=points_mesh(q),
                global_data={"g": drv},
            )
        )


def _spy(m, monkeypatch):
    """Record, for every forward pass of ``m``, the draw's kept cells and the boundary
    read's walk (its forest and the points' lists, of one chunk of points)."""
    seen = {"kept": [], "walks": []}
    draw, walk = isla_model.sample_boundary_cells, isla_model.walk_points_range

    def spy_draw(*a, **k):
        kept, weights = draw(*a, **k)
        seen["kept"].append(kept)
        return kept, weights

    def spy_walk(*a, **k):
        out = walk(*a, **k)
        seen["walks"].append(out)
        return out

    monkeypatch.setattr(isla_model, "sample_boundary_cells", spy_draw)
    monkeypatch.setattr(isla_model, "walk_points_range", spy_walk)
    return seen


def _renumbered(mesh: Mesh, seed: int):
    """``mesh`` with its cells and its vertices renumbered: new cell j is old cell
    ``c[j]`` (each cell's vertices in their order), new vertex j old vertex ``p[j]``."""
    g = torch.Generator().manual_seed(seed)
    p = torch.randperm(mesh.n_points, generator=g)
    c = torch.randperm(mesh.n_cells, generator=g)
    new_id = torch.empty_like(p).scatter_(0, p, torch.arange(mesh.n_points))
    return Mesh(
        points=mesh.points[p], cells=new_id[mesh.cells[c]], cell_data=mesh.cell_data[c]
    ), c


@THETAS
@pytest.mark.parametrize("budget", [None, 150], ids=["every_cell", "drawn"])
def test_renumbering_the_boundary_cells_changes_no_interior_prediction(
    theta, budget, monkeypatch
):
    """Renumbering the boundary's cells and vertices changes no interior prediction:
    the draw keeps the same physical cells, the forest holds them in the same order
    with the same nodes, every point's list names the same cells and nodes (bitwise,
    the cells mapped through the renumbering), and the outputs agree to 1e-12 (the
    only difference is the summation order of the frame's and the length unit's
    sums). With ``budget`` below the cell count the draw subsamples."""
    kw = {} if budget is None else {"n_boundary_tokens": budget}
    m = _model(theta=theta, **kw)
    seen = _spy(m, monkeypatch)
    mesh, drv, q = _case()
    renumbered, c = _renumbered(mesh, seed=7)
    base = _run(m, mesh, drv, q)
    out = _run(m, renumbered, drv, q)
    assert same(out, base, atol=1e-12)
    ### the same physical cells drawn: run 0's tokens are old cells, run 1's new ones
    old0, old1 = seen["kept"][0], c[seen["kept"][1]]
    k = old0.shape[0]
    assert k == (mesh.n_cells if budget is None else budget)
    assert torch.equal(old1.sort().values, old0.sort().values)
    ### run 1's token t is run 0's token map10[t]
    pos0 = torch.full((mesh.n_cells,), -1, dtype=torch.long)
    pos0[old0] = torch.arange(k)
    map10 = pos0[old1]
    (f0,), (f1,) = (w.forests for w in seen["walks"])
    assert torch.equal(map10[f1.cells], f0.cells)
    for name in ("range_start", "range_count", "left", "right", "coherent"):
        assert torch.equal(getattr(f1, name), getattr(f0, name)), name
    (qi0, si0, n0), (qi1, si1, n1) = (w.lists[0] for w in seen["walks"])
    si1 = torch.where(si1 < k, map10[si1.clamp(max=k - 1)], si1)
    assert n0 == n1 and torch.equal(qi1, qi0)
    if theta == 0.0:
        ### the exact read lists each point's cells in token order, which follows the
        ### numbering: the same pairs, compared as sets
        def pairs(qi, si):
            return (qi * (k + f0.n_nodes) + si).sort().values

        assert torch.equal(pairs(qi1, si1), pairs(qi0, si0))
    else:
        assert torch.equal(si1, si0)


@THETAS
def test_query_chunks_with_the_draw_active_equal_the_forward_pass(theta, monkeypatch):
    """ISLA.query_chunks when the draw subsamples (128 tokens of 300 cells), in
    training mode from the same seed: the same cells are drawn as by forward, the
    chunks' outputs equal forward's to 1e-11 absolute (other summation order), and
    every parameter's gradient of a loss summed over the chunks equals the unchunked
    step's to 1e-9 relative (a gradient's size varies over the parameters by
    decades; a gradient that vanishes analytically is roundoff on both sides, hence
    the floor of 1e-12 of the largest)."""
    m = _model(theta=theta, n_boundary_tokens=128).train()
    with torch.no_grad():
        for p in m.blocks.parameters():
            p.add_(0.05 * torch.randn_like(p))
    seen = _spy(m, monkeypatch)
    mesh, drv, q = _case()
    kw = {
        "boundary_meshes": {"wall": mesh},
        "prediction_points": points_mesh(q),
        "global_data": {"g": drv},
    }
    target = torch.randn(
        q.shape[0], 4, generator=torch.Generator().manual_seed(9), dtype=D
    )

    def loss_of(out, rows):
        return (out["p"] - target[rows, 0]).square().sum() + (
            out["v"] - target[rows, 1:]
        ).square().sum()

    torch.manual_seed(3)
    full = outputs(m(**kw))
    loss_of(full, slice(None)).backward()
    ref = {n: p.grad.clone() for n, p in m.named_parameters()}
    m.zero_grad(set_to_none=True)
    torch.manual_seed(3)
    chunks = m.query_chunks(**kw)
    parts = []
    for a in range(0, chunks.n_points, 37):
        rows = slice(a, min(a + 37, chunks.n_points))
        out = outputs(chunks(rows.start, rows.stop))
        loss_of(out, rows).backward()
        parts.append({k: v.detach() for k, v in out.items()})
    chunks.backward()
    chunked = {k: torch.cat([p[k] for p in parts]) for k in parts[0]}
    assert [kept.shape[0] for kept in seen["kept"]] == [128, 128]
    assert torch.equal(seen["kept"][0], seen["kept"][1])
    assert same(chunked, {k: v.detach() for k, v in full.items()}, atol=1e-11)
    scale = max(float(g.abs().max()) for g in ref.values())
    for name, p in m.named_parameters():
        assert p.grad is not None, name
        assert torch.allclose(p.grad, ref[name], rtol=1e-9, atol=1e-12 * scale), name


def test_query_chunks_carry_the_gradients_of_the_boundary_points_and_global_data():
    """With the boundary points and the global direction requiring gradients, three
    chunks' backward passes and QueryChunks.backward give both inputs the gradients of
    one forward pass, to 1e-12 of the largest (float64, another summation order): every
    tensor the chunks share is a leaf, so no chunk's backward frees another's graph."""
    m = _model()
    mesh, drv, q = _case()

    def grads(chunked):
        pts = mesh.points.detach().clone().requires_grad_()
        g = drv.detach().clone().requires_grad_()
        kw = {
            "boundary_meshes": {
                "wall": Mesh(points=pts, cells=mesh.cells, cell_data=mesh.cell_data)
            },
            "prediction_points": points_mesh(q),
            "global_data": {"g": g},
        }
        if chunked:
            chunks = m.query_chunks(**kw)
            for a, b in ((0, 40), (40, 80), (80, chunks.n_points)):
                out = outputs(chunks(a, b))
                sum(v.square().sum() for v in out.values()).backward()
            chunks.backward()
        else:
            out = outputs(m(**kw))
            sum(v.square().sum() for v in out.values()).backward()
        return pts.grad, g.grad

    for got, ref in zip(grads(chunked=True), grads(chunked=False)):
        assert got is not None and float(ref.abs().max()) > 0
        assert torch.allclose(got, ref, rtol=0, atol=1e-12 * float(ref.abs().max()))


def test_query_chunks_carry_the_prediction_points_gradient_through_the_read():
    """At ``theta=0``, with a boundary vector field (rank-1 sources of the read, from a
    learned map) and prediction points requiring gradients: three chunks' backward
    passes and QueryChunks.backward give the points and every parameter the gradients
    of one forward pass, to 1e-12 of the largest (float64), and the points' gradient
    is the central difference's (``h = 1e-5``) to 1e-7 of the largest. The read's pair
    invariants carry the points' gradient, and their sources are leaves, so no
    chunk's backward frees another's graph."""
    m = _model(theta=0.0, boundary_schemas={"wall": {"u": {"rank": 1}}})
    pts, nrm, w, drv = cloud(200, seed=4)
    u = torch.randn(200, 3, generator=torch.Generator().manual_seed(5), dtype=D)
    mesh = soup(pts, nrm, w, cell_data={"u": u})
    q = pts[:60] + 0.3 * nrm[:60]
    kw = {"boundary_meshes": {"wall": mesh}, "global_data": {"g": drv}}

    def loss(out):
        return sum(v.square().sum() for v in outputs(out).values())

    def grads(chunked):
        m.zero_grad()
        x = q.clone().requires_grad_()
        if chunked:
            chunks = m.query_chunks(prediction_points=points_mesh(x), **kw)
            for a, b in ((0, 20), (20, 40), (40, chunks.n_points)):
                loss(chunks(a, b)).backward()
            chunks.backward()
        else:
            loss(m(prediction_points=points_mesh(x), **kw)).backward()
        return [x.grad] + [p.grad for p in m.parameters()]

    ref = grads(chunked=False)
    for got, want in zip(grads(chunked=True), ref):
        assert (got is None) == (want is None)
        if want is not None:
            assert torch.allclose(
                got, want, rtol=0, atol=1e-12 * float(want.abs().max())
            )
    h, fd = 1e-5, torch.zeros(2, 3, dtype=D)
    with torch.no_grad():
        for i in range(2):
            for j in range(3):
                e = torch.zeros_like(q)
                e[i, j] = h
                hi = loss(m(prediction_points=points_mesh(q + e), **kw))
                lo = loss(m(prediction_points=points_mesh(q - e), **kw))
                fd[i, j] = (hi - lo) / (2 * h)
    assert torch.allclose(ref[0][:2], fd, rtol=0, atol=1e-7 * float(fd.abs().max()))


def test_query_chunks_outputs_differentiate_in_space():
    """A chunk's outputs stay in the graph of the caller's prediction points: their
    derivative in the points (as a PDE-residual loss takes it) is that of one forward
    pass at the same rows, to 1e-12 of the largest (float64), for every chunk."""
    m = _model()
    mesh, drv, q = _case()
    kw = {"boundary_meshes": {"wall": mesh}, "global_data": {"g": drv}}

    def space_grad(out, pts):
        s = sum(v.square().sum() for v in outputs(out).values())
        return torch.autograd.grad(s, pts)[0]

    pts = q.detach().clone().requires_grad_()
    ref = space_grad(m(prediction_points=points_mesh(pts), **kw), pts)
    pts = q.detach().clone().requires_grad_()
    chunks = m.query_chunks(prediction_points=points_mesh(pts), **kw)
    for a, b in ((0, 40), (40, chunks.n_points)):
        got = space_grad(chunks(a, b), pts)
        assert (
            float(got[a:b].abs().max()) > 0 and not got[:a].any() and not got[b:].any()
        )
        assert torch.allclose(
            got[a:b], ref[a:b], rtol=0, atol=1e-12 * float(ref.abs().max())
        )


def test_forward_runs_its_points_in_chunks_of_at_most_query_chunk_pairs(
    monkeypatch, caplog
):
    """Above ``query_chunk_pairs`` walk pairs the interior forward runs its prediction
    points in consecutive chunks of at most that many pairs each (a point with more
    alone), the fewest from the start, and logs the choice once; its outputs and every
    parameter's gradient are those of the forward in one chunk (float64, another
    summation order: outputs to 1e-11 absolute, gradients to 1e-9 relative with a
    floor of 1e-12 of the largest)."""
    m = _model()
    mesh, drv, q = _case()
    kw = {
        "boundary_meshes": {"wall": mesh},
        "prediction_points": points_mesh(q),
        "global_data": {"g": drv},
    }

    def run():
        m.zero_grad(set_to_none=True)
        out = outputs(m(**kw))
        sum(v.square().sum() for v in out.values()).backward()
        grads = {n: p.grad.clone() for n, p in m.named_parameters()}
        return {k: v.detach() for k, v in out.items()}, grads

    whole, ref = run()
    with torch.no_grad():
        counts = m.query_chunks(**kw).pair_counts
    budget = int(counts.sum()) // 5
    spans = []
    fields = isla_model.QueryChunks._fields

    def spy(self, start, stop):
        spans.append((start, stop))
        return fields(self, start, stop)

    monkeypatch.setattr(m, "query_chunk_pairs", budget)
    monkeypatch.setattr(isla_model.QueryChunks, "_fields", spy)
    with caplog.at_level(logging.INFO, logger=isla_model.__name__):
        chunked, grads = run()
        run()
    assert (
        len(spans) >= 2 * 5
        and spans[0][0] == 0
        and spans[len(spans) // 2 - 1][1] == q.shape[0]
    )
    first = spans[: len(spans) // 2]
    assert all(a == b for (_, b), (a, _) in zip(first, first[1:]))
    sizes = [int(counts[a:b].sum()) for a, b in first]
    assert all(n <= budget or b - a == 1 for n, (a, b) in zip(sizes, first))
    assert all(n + int(counts[b]) > budget for n, (_, b) in zip(sizes, first[:-1]))
    assert sum("chunks of at most" in r.getMessage() for r in caplog.records) == 1
    assert same(chunked, whole, atol=1e-11)
    scale = max(float(g.abs().max()) for g in ref.values())
    for name, g in grads.items():
        assert torch.allclose(g, ref[name], rtol=1e-9, atol=1e-12 * scale), name


@THETAS
@pytest.mark.parametrize("improper", [False, True], ids=["rotation", "reflection"])
@pytest.mark.parametrize("dim", [3, 2])
def test_every_rank_and_parity_is_covariant_in_the_interior_task(dim, improper, theta):
    """Under a random orthogonal map R (proper or improper) with a translation, moving
    the boundary, the global direction and the prediction points together, every
    output of the interior task maps by its rank (R applied on each axis) and an
    odd-parity one also by det R, ranks 0 to 4, in two and three dimensions. The read
    is exact at theta 0, and at the default theta the same approximation in every
    pose (the forest lives in the boundary's own frame). Float64 roundoff of outputs
    of order one: 1e-11 absolute."""
    m = _model(dim, theta, output_schema=EVERY_RANK)
    mesh, drv, q = _case(dim)
    base = _run(m, mesh, drv, q)
    R = orthogonal(dim, 4, improper)
    t = torch.arange(1.0, dim + 1.0, dtype=D)
    out = _run(m, transform(mesh, R, t), drv @ R.T, q @ R.T + t)
    for name, spec in m.output_schema.items():
        sign = torch.det(R) if spec.parity == "odd" else 1.0
        want = law(base[name], R, spec.rank) * sign
        assert torch.allclose(out[name], want, atol=1e-11, rtol=0.0), name
    ### not vacuous: every output is nonzero, the odd ones included
    assert all(base[name].abs().max() > 1e-3 for name in EVERY_RANK)


def _ellipsoid(perturb=0.0, axes=(2.0, 1.0, 0.5)):
    """An icosphere of 320 cells scaled to the semi-axes ``axes``. At (2, 1, 0.5):
    distinct principal spreads and three mirror planes, so every axis of the
    boundary's canonical frame has a vanishing third moment and takes its sign from
    the lab axes. At (1, 1, 1), a sphere: equal spreads as well, so the axes' very
    directions come from the lab axes. With ``perturb`` each vertex moves radially by
    a seeded relative amount below ``perturb / 2``, which breaks the mirrors: the
    frame is unique."""
    sphere = sphere_icosahedral.load(radius=1.0, subdivisions=2)
    pts = sphere.points.to(D)
    g = torch.Generator().manual_seed(5)
    pts = pts * (
        1 + perturb * (torch.rand(pts.shape[0], 1, generator=g, dtype=D) - 0.5)
    )
    return Mesh(points=pts * torch.tensor(axes, dtype=D), cells=sphere.cells)


@pytest.mark.parametrize("case", ["mirror_planes", "sphere", "perturbed"])
def test_a_boundary_with_mirror_planes_is_covariant_where_documented(case, monkeypatch):
    """Under three random rotations and a reflection, each with a translation, the
    float64 interior model's outputs map by their ranks to 1e-12 (roundoff of outputs
    of order one) on a boundary whose canonical frame ties (an ellipsoid with three
    mirror planes, or a sphere) when every cell is a token and the read is exact,
    and on a perturbed ellipsoid, whose frame is unique, with 100 of its 320 cells
    drawn at the default theta (the same cells in every pose).

    A tied frame with fewer tokens than cells or a finite theta, and an open
    boundary, are the documented exceptions (module docstring, *Where the covariance
    is exact*): there the outputs agree only approximately, and are not asserted."""
    tied = case != "perturbed"
    if case == "mirror_planes":
        m, mesh = _model(theta=0.0), _ellipsoid()
    elif case == "sphere":
        m, mesh = _model(theta=0.0), _ellipsoid(axes=(1.0, 1.0, 1.0))
    else:
        m, mesh = _model(n_boundary_tokens=100), _ellipsoid(perturb=0.1)
    seen = _spy(m, monkeypatch)
    g = torch.Generator().manual_seed(1)
    q = 1.5 * torch.randn(60, 3, generator=g, dtype=D)
    drv = torch.nn.functional.normalize(torch.tensor([1.0, 0.3, 0.2], dtype=D), dim=0)
    base = _run(m, mesh, drv, q)
    t = torch.tensor([1.0, -2.0, 3.0], dtype=D)
    for seed, improper in ((10, False), (11, False), (12, False), (13, True)):
        R = orthogonal(3, seed, improper)
        out = _run(m, transform(mesh, R, t), drv @ R.T, q @ R.T + t)
        assert torch.allclose(out["p"], base["p"], atol=1e-12, rtol=0.0), seed
        assert torch.allclose(out["v"], base["v"] @ R.T, atol=1e-12, rtol=0.0), seed
    assert all(k.shape[0] == (320 if tied else 100) for k in seen["kept"])
    assert all(torch.equal(k, seen["kept"][0]) for k in seen["kept"])
    assert base["p"].abs().max() > 0.1


def _float32_interior_case(device):
    """A float32 interior model on ``device`` (its read live) and a case of 500 cells
    with 150 prediction points, as float32 tensors on ``device``."""
    m = _model().float().to(device)
    mesh, drv, q = _case(n=500, nq=150)
    mesh = Mesh(
        points=mesh.points.float(),
        cells=mesh.cells,
        cell_data=mesh.cell_data.apply(lambda t: t.float()),
    )
    return m, mesh.to(device), drv.float().to(device), q.float().to(device)


@pytest.mark.parametrize(
    "device",
    [
        ### on CPU the padding tokens' zero states once gave 0/0 in the routing
        ### logits' float16 normalize
        "cpu",
        pytest.param(
            "cuda",
            marks=[
                pytest.mark.cuda,
                ### with triton: auto must not pick the fused region, which refuses
                ### float16 autocast
                requires_cuda_triton,
            ],
        ),
    ],
)
def test_a_float16_autocast_forward_is_finite(device):
    """Under float16 autocast, a forward pass of the interior model with every kernel
    at ``auto`` gives finite outputs, as it does under bfloat16 (and as the eager
    kernels do on CUDA)."""
    m, mesh, drv, q = _float32_interior_case(device)
    with torch.no_grad(), torch.autocast(device, dtype=torch.float16):
        out = outputs(
            m(
                boundary_meshes={"wall": mesh},
                prediction_points=q,
                global_data={"g": drv},
            )
        )
    assert all(torch.isfinite(o).all() for o in out.values())


@pytest.mark.cuda
@requires_cuda_triton  # the bitwise comparison needs the Triton kernels' fixed order
def test_mixed_devices_are_refused_and_global_data_follows_the_boundary():
    """Documented behaviour, as it is: inputs on another device than the model are
    refused, by torch's own device check deep in the forward pass (ISLA's validation
    does not name the argument), and ``global_data`` is moved to the boundary's
    device."""
    m, mesh, drv, q = _float32_interior_case("cuda")
    gd = {"g": drv}
    for kw in (
        dict(
            boundary_meshes={"wall": mesh.to("cpu")},
            prediction_points=q,
            global_data=gd,
        ),
        dict(boundary_meshes={"wall": mesh}, prediction_points=q.cpu(), global_data=gd),
    ):
        with torch.no_grad(), pytest.raises(RuntimeError, match="same device"):
            m(**kw)
    with torch.no_grad():
        out = outputs(
            m(
                boundary_meshes={"wall": mesh},
                prediction_points=q,
                global_data={"g": drv.cpu()},
            )
        )
        ref = outputs(
            m(boundary_meshes={"wall": mesh}, prediction_points=q, global_data=gd)
        )
    assert all(torch.equal(out[k], ref[k]) for k in ref)
