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

"""Per-condition quadrature. With more than one declared condition, every boundary
token's log weight is log a_i - log A_k + beta_k, where A_k is the full-resolution
measure of its condition and beta_k a learned log share
(``condition_log_shares.beta``, zero at construction, after every other parameter).
The shift applies in the blocks, the head frame and the interior task's per-pair
log m (a node's cells are of one condition, so it takes that condition's shift); the
read's forest keeps the raw measures. A model of one declared condition has no
shares and no shift. On CPU the read is the eager one; on CUDA the fused read
differentiates log m too.
"""

import math

import pytest
import torch

import physicsnemo.experimental.models.isla.model as isla_model
from physicsnemo.experimental.models.isla import ISLA
from physicsnemo.experimental.models.isla import _tree_read as tree_read
from physicsnemo.experimental.models.isla._tree_read import BoundaryForest
from physicsnemo.mesh.calculus.measure import cell_measures
from test.experimental.models.isla._marks import requires_cuda, requires_cuda_triton
from test.experimental.models.isla._meshes import outputs, points_mesh, same, soup

F = torch.float32
D = torch.float64
GD = {"g": torch.tensor([0.8, 0.6])}


def _model(conditions=("airfoil", "freestream"), n_boundary_tokens=150, **kw):
    """A small interior model; its reads' weights randomized (their output projection starts
    at zero, so the read would start as no update)."""
    torch.manual_seed(0)
    m = ISLA(
        output_schema={"p": {"rank": 0}, "v": {"rank": 1}},
        boundary_schemas={c: {} for c in conditions},
        global_schema={"g": {"rank": 1}},
        interior=True,
        n_spatial_dims=2,
        hidden=32,
        n_layers=2,
        n_slices=8,
        n_boundary_tokens=n_boundary_tokens,
        **kw,
    ).eval()
    with torch.no_grad():
        for p in [*m.kernel_reads.parameters(), *m.kernel_basis.parameters()]:
            p.normal_(0.0, 0.3)
    return m


def _polygon(v, outward):
    """Cells of a closed polyline ``v``: centroids, unit normals (outward from the
    origin, or inward) and lengths."""
    e = torch.roll(v, -1, 0) - v
    n = torch.stack([e[:, 1], -e[:, 0]], -1) / e.norm(dim=-1, keepdim=True)
    mid = v + 0.5 * e
    flip = ((n * mid).sum(-1) < 0) != outward
    return mid, torch.where(flip[:, None], -n, n), e.norm(dim=-1)


def _meshes(scale=1.0, n_body=80, n_far=60, split=False, dtype=F):
    """A small closed body (chord 1) inside a large circle (radius 30), float64
    geometry cast once; ``split`` halves the body's first cell into two."""
    t = torch.linspace(0, 2 * math.pi, n_body + 1, dtype=D)[:-1]
    body = torch.stack(
        [0.5 * torch.cos(t), 0.07 * torch.sin(t) + 0.02 * torch.sin(2 * t)], -1
    )
    if split:
        body = torch.cat([body[:1], 0.5 * (body[:1] + body[1:2]), body[1:]])
    s = torch.linspace(0, 2 * math.pi, n_far + 1, dtype=D)[:-1]
    far = torch.stack([30 * torch.cos(s) + 4.0, 30 * torch.sin(s) - 1.5], -1)
    out = {}
    for name, v, outward in (("airfoil", body, True), ("freestream", far, False)):
        mid, n, w = _polygon(v * scale, outward)
        out[name] = soup(mid, n, w).to(dtype)
    return out


def _queries(scale=1.0, dtype=F):
    g = torch.Generator().manual_seed(7)
    return points_mesh(
        (torch.randn(50, 2, generator=g, dtype=D) * 1.5 * scale).to(dtype)
    )


def _first_block_log_w(m):
    """The first block's log weights of the boundary tokens of the next forward."""
    seen = {}

    def hook(mod, a):
        seen.setdefault("log_w", a[1].detach().clone())

    handle = m.blocks[0].register_forward_pre_hook(hook)
    return seen, handle


def _no_shift(m, monkeypatch):
    """The code path without the log share shift."""
    monkeypatch.setattr(m, "_log_shift", lambda boundary, dtype: None)
    return m


def _kept(m, meshes, monkeypatch):
    """Each token's condition (canonical order) and HT weight, from the draw the
    model makes."""
    seen = {}
    sample = isla_model.sample_boundary_cells

    def spy(*a, **k):
        seen["kept"], seen["w"] = sample(*a, **k)
        return seen["kept"], seen["w"]

    monkeypatch.setattr(isla_model, "sample_boundary_cells", spy)
    names = [n for n in m.boundary_names if n in meshes]
    cond = torch.cat(
        [torch.full((meshes[n].n_cells,), m.boundary_names.index(n)) for n in names]
    )
    return seen, cond


def test_the_shares_are_zeros_after_every_other_parameter_and_one_condition_has_none():
    """beta starts at zero and comes last in parameters(), so a loop over the
    parameters (a seeded fill, an optimizer's groups) sees every other parameter in
    its order; a model of one declared condition has no shares, its parameters those
    of the same model before the shares were added."""
    names = [n for n, _ in _model().named_parameters()]
    assert names[-1] == "condition_log_shares.beta"
    assert names.count("condition_log_shares.beta") == 1
    torch.manual_seed(0)
    m = ISLA(
        output_schema={"p": {"rank": 0}},
        boundary_schemas={"a": {}, "b": {}, "c": {}},
        n_spatial_dims=3,
        hidden=32,
        n_layers=2,
        n_slices=8,
    )
    assert torch.equal(m.condition_log_shares.beta, torch.zeros(3))
    one = _model(conditions=("airfoil",))
    assert one.condition_log_shares is None
    assert not [n for n, _ in one.named_parameters() if "condition_log_shares" in n]


@pytest.mark.parametrize("budget, split", [(60, False), (1000, False), (1000, True)])
def test_each_condition_s_total_weight_is_its_share(budget, split, monkeypatch):
    """Per condition, the tokens' total weight is e^{beta_k} (HT total / A_k); kept
    whole (budget above the cell count) it is e^{beta_k}, and splitting a cell into
    two changes no condition's total (refinement)."""
    beta = torch.tensor([0.7, -1.3])
    m = _model(n_boundary_tokens=budget)
    with torch.no_grad():
        m.condition_log_shares.beta.copy_(beta)
    meshes = _meshes(split=split)
    draw, cond = _kept(m, meshes, monkeypatch)
    seen, _ = _first_block_log_w(m)
    with torch.no_grad():
        m(boundary_meshes=meshes, prediction_points=_queries(), global_data=GD)
    cond = cond[draw["kept"]]
    log_w = seen["log_w"][0, : cond.shape[0], 0].double()
    for k, name in enumerate(m.boundary_names):
        area = cell_measures(meshes[name]).double().sum()
        ht = draw["w"][cond == k].double().sum()
        total = log_w[cond == k].exp().sum()
        torch.testing.assert_close(
            total, math.exp(beta[k]) * ht / area, rtol=2e-6, atol=0
        )
        if budget == 1000:
            torch.testing.assert_close(
                total, torch.tensor(math.exp(beta[k]), dtype=D), rtol=2e-6, atol=0
            )


def test_a_uniform_rescale_leaves_every_log_weight_unchanged():
    """Scaling the whole case by any factor leaves every token's log weight unchanged:
    log a_i - log A_k cancels the scale, and the shares do not see it."""
    m = _model()
    with torch.no_grad():
        m.condition_log_shares.beta.copy_(torch.tensor([0.4, -0.2]))
    out = []
    for c in (1.0, 3.7, 0.021):
        seen, handle = _first_block_log_w(m)
        with torch.no_grad():
            m(
                boundary_meshes=_meshes(scale=c),
                prediction_points=_queries(c),
                global_data=GD,
            )
        handle.remove()
        out.append(seen["log_w"][0, :150, 0])
    for o in out[1:]:
        torch.testing.assert_close(o, out[0], rtol=0, atol=1e-5)


@pytest.mark.parametrize(
    "dtype, tol", [(F, 1e-5), (D, 1e-12)], ids=["float32", "float64"]
)
def test_one_given_condition_with_zero_shares_is_the_unshifted_function(
    dtype, tol, monkeypatch
):
    """One condition given of two, beta = 0: the shift is the constant -log A, which
    cancels in every softmax up to rounding (max error over max output; float64
    shows it is rounding only)."""
    meshes = {"airfoil": _meshes(dtype=dtype)["airfoil"]}
    q = _queries(dtype=dtype)
    gd = {"g": GD["g"].to(dtype)}
    with torch.no_grad():
        shifted = (
            _model()
            .to(dtype)(boundary_meshes=meshes, prediction_points=q, global_data=gd)
            .point_data
        )
        base = _no_shift(_model().to(dtype), monkeypatch)(
            boundary_meshes=meshes, prediction_points=q, global_data=gd
        ).point_data
    for k in ("p", "v"):
        a, b = shifted[k].double(), base[k].double()
        assert float((a - b).abs().max()) <= tol * float(b.abs().max())


def test_the_shares_gradient_is_finite_and_nonzero_through_the_read_too(monkeypatch):
    """The shares receive a finite, nonzero gradient, and so does the boundary read's
    part of it alone (the full gradient minus the one with the read's shift
    detached)."""
    m = _model().train()

    def step():
        m.zero_grad(set_to_none=True)
        torch.manual_seed(3)
        out = m(
            boundary_meshes=_meshes(), prediction_points=_queries(), global_data=GD
        ).point_data
        (out["p"].square().mean() + out["v"].square().mean()).backward()
        return m.condition_log_shares.beta.grad.clone()

    full = step()
    assert torch.isfinite(full).all() and bool((full != 0).all())
    ### The read's part alone: the full gradient minus the one with the read's shift
    ### detached (the blocks' and head frame's parts are the same in both).
    walk_range_pairs = isla_model.walk_range_pairs

    def detached(*a, log_shift=None, condition_shift=None, **k):
        return walk_range_pairs(
            *a,
            log_shift=None if log_shift is None else log_shift.detach(),
            condition_shift=None
            if condition_shift is None
            else condition_shift.detach(),
            **k,
        )

    monkeypatch.setattr(isla_model, "walk_range_pairs", detached)
    read = full - step()
    assert torch.isfinite(read).all() and bool((read != 0).all())


def test_an_absent_condition_s_share_gets_a_zero_gradient():
    """A declared condition absent from the case still has a gradient on its share,
    exactly zero, so DDP finds every parameter used."""
    m = _model().train()
    out = m(
        boundary_meshes={"airfoil": _meshes()["airfoil"]},
        prediction_points=_queries(),
        global_data=GD,
    )
    out.point_data["p"].square().mean().backward()
    g = m.condition_log_shares.beta.grad
    assert g is not None and float(g[1]) == 0.0


def test_the_forest_is_the_unshifted_path_s_bitwise(monkeypatch):
    """The shift must not reach the forest: with nonzero shares the forest built
    inside forward, the read's cell fractions, its pairs and their basis are bitwise
    those of the unshifted path. Each pair's log m is the raw one plus its source's
    shift: a cell's own, bitwise; a node's, bitwise its condition's, every node
    holding one condition (one set of face trees per condition)."""
    seen, pairs = [], []
    build = BoundaryForest.build
    walk_range_pairs = isla_model.walk_range_pairs

    def spy(*a, **k):
        seen.append(build(*a, **k))
        return seen[-1]

    def spy_pairs(*a, **k):
        pairs.append(
            (walk_range_pairs(*a, **k), k.get("log_shift"), k.get("condition_shift"))
        )
        return pairs[-1][0]

    monkeypatch.setattr(BoundaryForest, "build", spy)
    monkeypatch.setattr(isla_model, "walk_range_pairs", spy_pairs)
    conds = []
    for shifted in (True, False):
        m = _model()
        with torch.no_grad():
            m.condition_log_shares.beta.copy_(torch.tensor([1.1, -0.6]))
        if not shifted:
            m._log_shift = lambda boundary, dtype: None
        draw, cond = _kept(m, _meshes(), monkeypatch)
        with torch.no_grad():
            m(boundary_meshes=_meshes(), prediction_points=_queries(), global_data=GD)
        conds.append(cond[draw["kept"]])
    assert torch.equal(conds[0], conds[1])
    a, b = seen
    ### ``per_tree`` makes ``trees`` and ``tree_cells``, compared in its place.
    names = [k for k in a.__dataclass_fields__ if k != "per_tree"]
    for k in [*names, "trees", "tree_cells"]:
        x, y = getattr(a, k), getattr(b, k)
        if k == "trees":
            assert all(
                xa.keys() == ya.keys() and all(torch.equal(xa[j], ya[j]) for j in xa)
                for xa, ya in zip((t.to_dict() for t in x), (t.to_dict() for t in y))
            )
        elif isinstance(x, torch.Tensor):
            assert torch.equal(x, y), k
        elif isinstance(x, list) and x and isinstance(x[0], torch.Tensor):
            assert all(torch.equal(u, v) for u, v in zip(x, y)), k
        elif k != "levels":
            assert x == y, k
    (pa, shift, cshift), (pb, none, _) = pairs
    assert none is None
    assert torch.equal(pa.omega, pb.omega) and torch.equal(pa.basis, pb.basis)
    assert torch.equal(pa.plan.qi, pb.plan.qi) and torch.equal(pa.plan.bj, pb.plan.bj)
    forest, n, cond = pa.forests[0], pa.n_cells, conds[0]
    shift, cshift = shift[0], cshift[0]
    assert torch.equal(pa.src_log_m[:n], pb.src_log_m[:n] + shift)
    assert torch.equal(pa.log_m, pa.src_log_m[pa.plan.bj.long()])
    assert forest.group_origin.shape[0] == 2
    for node in range(forest.n_nodes):
        a_, c_ = int(forest.range_start[node]), int(forest.range_count[node])
        cells = forest.cells[a_ : a_ + c_]
        got = pa.src_log_m[n + node]
        assert cond[cells].unique().numel() == 1, node
        assert torch.equal(got, pb.src_log_m[n + node] + cshift[cond[cells[0]]])
        assert torch.equal(cshift[cond[cells[0]]], shift[cells[0]])


def _logsumexp_node_shift(forest, omega, conditions, condition_shift):
    """The logsumexp node shift, for nodes that may mix conditions:
    logsumexp_k(log f_{m,k} + s_k), with f_{m,k} the node's fraction of its measure
    in condition k (float64 prefix sums without gradient, -inf where the node holds
    none)."""
    c = condition_shift.shape[0]
    with torch.no_grad():
        cells = forest.cells.long()
        mass = torch.zeros(cells.shape[0] + 1, c, dtype=D, device=omega.device)
        one_hot = torch.nn.functional.one_hot(conditions[cells].long(), c).double()
        mass[1:] = one_hot * omega[cells].double()[:, None]
        prefix = mass.cumsum(0)
        start = forest.range_start.long()
        per = prefix[start + forest.range_count.long()] - prefix[start]
        total = per.sum(-1, keepdim=True)
        frac = torch.where(
            total > 0, per / total.clamp_min(torch.finfo(D).tiny), 1.0 / c
        )
        log_frac = torch.where(frac > 0, torch.log(frac), -math.inf)
    lf = log_frac.to(condition_shift.dtype)
    terms = torch.where(torch.isneginf(lf), lf, lf + condition_shift[None, :])
    return torch.logsumexp(terms, dim=-1)


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=[pytest.mark.cuda, requires_cuda])]
)
@pytest.mark.parametrize("dtype", [F, D], ids=["float32", "float64"])
def test_a_node_s_shift_is_the_logsumexp_form_s_bitwise(device, dtype):
    """With one set of face trees per condition no node mixes conditions, so a node
    takes its condition's shift s_k: the read's source log measures, and beta's
    gradient through them (a loss with random weights on every source), are bitwise
    those of the logsumexp form over the nodes' condition fractions on the same
    forest; of three declared conditions two are given, so the absent one's share
    takes a zero gradient."""
    meshes = _meshes()
    names = ("airfoil", "freestream")
    pts = torch.cat([meshes[k].cell_centroids for k in names]).to(device, dtype)
    nrm = torch.cat([meshes[k].cell_normals for k in names]).to(device, dtype)
    w = torch.cat([cell_measures(meshes[k]) for k in names]).to(device, dtype)
    cond = torch.cat(
        [
            torch.full((meshes[k].n_cells,), 2 * i, device=device)
            for i, k in enumerate(names)
        ]
    )
    n = pts.shape[0]
    g = torch.Generator().manual_seed(5)
    q = (torch.randn(40, 2, generator=g, dtype=D) * 3).to(device, dtype)
    walk = tree_read.tree_read_walk(
        query_r=q[None],
        query_n=torch.nn.functional.normalize(q, dim=-1)[None],
        query_wall=q.norm(dim=-1)[None],
        boundary_r=pts[None],
        boundary_n=nrm[None],
        boundary_vecs=None,
        log_frac=torch.log(w / w.sum())[None],
        g_unit=torch.tensor([[[0.6, 0.8]]], device=device, dtype=dtype),
        theta=1.0,
        eps=1e-12,
        conditions=cond[None],
    )
    forest = walk.forests[0]
    assert forest.group_origin.shape[0] == 2 and forest.n_nodes > 10
    basis = tree_read._KernelBasis(tree_read.kernel_geo_width(1, 0)).to(device, dtype)
    neg = (torch.rand(3, generator=g, dtype=D) * 5).to(device, dtype)
    loss_w = torch.randn(n + forest.n_nodes, generator=g, dtype=D).to(device, dtype)

    def read(node_shift):
        beta = torch.tensor(
            [0.4, -0.7, 1.3], device=device, dtype=dtype, requires_grad=True
        )
        cshift = neg + beta
        one_hot = torch.nn.functional.one_hot(cond, 3).to(dtype)
        log_shift = (one_hot * cshift).sum(-1)
        if node_shift is None:
            src = tree_read.walk_range_pairs(
                basis, walk, log_shift=log_shift[None], condition_shift=cshift[None]
            ).src_log_m
        else:
            shifts = torch.cat([log_shift, node_shift(cshift)])
            src = walk.src_logm[0] + shifts.to(dtype)
        (grad,) = torch.autograd.grad((src * loss_w).sum(), beta)
        return src.detach(), grad

    omega = torch.exp(walk.src_logm[0][:n])
    got = read(None)
    ref = read(lambda s: _logsumexp_node_shift(forest, omega, cond, s))
    assert torch.equal(got[0], ref[0])
    assert torch.equal(got[1], ref[1]), (got[1], ref[1])
    assert float(got[1][1]) == 0.0 and bool((got[1][[0, 2]] != 0).all())


def test_query_chunks_carry_the_shares_gradient_as_forward_does():
    """ISLA.query_chunks with two conditions and nonzero shares: the chunks' outputs
    equal forward's, and beta's gradient (and every other parameter's) of a loss summed
    over the chunks equals the unchunked step's (the read's shift held as a leaf)."""
    m = _model(n_boundary_tokens=100).double().train()
    with torch.no_grad():
        m.condition_log_shares.beta.copy_(torch.tensor([0.5, -0.9], dtype=D))
    kw = dict(
        boundary_meshes=_meshes(dtype=D),
        prediction_points=_queries(dtype=D),
        global_data={"g": GD["g"].to(D)},
    )
    torch.manual_seed(3)
    full = outputs(m(**kw))
    sum(v.square().sum() for v in full.values()).backward()
    ref = {n: p.grad.clone() for n, p in m.named_parameters()}
    m.zero_grad(set_to_none=True)
    torch.manual_seed(3)
    chunks = m.query_chunks(**kw)
    parts = []
    for a in range(0, chunks.n_points, 17):
        out = outputs(chunks(a, min(a + 17, chunks.n_points)))
        sum(v.square().sum() for v in out.values()).backward()
        parts.append({k: v.detach() for k, v in out.items()})
    chunks.backward()
    chunked = {k: torch.cat([p[k] for p in parts]) for k in parts[0]}
    assert same(chunked, {k: v.detach() for k, v in full.items()}, atol=1e-11)
    scale = max(float(g.abs().max()) for g in ref.values())
    assert bool((ref["condition_log_shares.beta"] != 0).all())
    for name, p in m.named_parameters():
        assert p.grad is not None, name
        assert torch.allclose(p.grad, ref[name], rtol=1e-9, atol=1e-12 * scale), name


@pytest.mark.cuda
@requires_cuda_triton
def test_the_fused_read_carries_the_shares_gradient_as_the_reference_read():
    """On CUDA in float32 (no autocast), beta's gradient with the fused read (the
    default) against the same step with the reference read (Triton's
    segment-softmax read, which differentiates log m per pair): within 1e-4 relative
    per component."""
    meshes = {k: v.to("cuda") for k, v in _meshes().items()}
    q = points_mesh(_queries().points.cuda())
    gd = {"g": GD["g"].cuda()}

    def grad(kernel):
        m = _model().cuda().train()
        m._tree_read_kernel = kernel
        with torch.no_grad():
            m.condition_log_shares.beta.copy_(torch.tensor([0.4, -0.7]))
        assert m._resolve_tree_read_kernel(q.points) == (
            "fused" if kernel == "auto" else "reference"
        )
        torch.manual_seed(3)
        out = m(boundary_meshes=meshes, prediction_points=q, global_data=gd).point_data
        (out["p"].square().mean() + out["v"].square().mean()).backward()
        return m.condition_log_shares.beta.grad.double().cpu()

    fused, ref = grad("auto"), grad("reference")
    rel = float(((fused - ref).abs() / ref.abs()).max())
    print(
        f"beta grad fused {fused.tolist()} reference {ref.tolist()} max rel {rel:.2e}"
    )
    assert rel < 1e-4
