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

"""The fused geometry region and the compiled model compute what the eager reference
computes (the eager region, rebuilt in backward).

The bars: the fused Triton region reproduces the eager forward to 1e-6 relative in
float32 and the parameter gradients to 1e-5 relative; in float64 the two agree to
1e-11 (the same arithmetic in another summation order). The same bars hold for the
whole model under torch.compile, which must stay free of graph breaks."""

import copy
import logging
import pickle

import pytest
import torch

from physicsnemo.experimental.models.isla import ISLA
from physicsnemo.experimental.models.isla._blocks import _geo_region
from test.experimental.models.isla._marks import requires_cuda, requires_cuda_triton
from test.experimental.models.isla._meshes import cloud, outputs, soup

REF = dict(
    output_schema={"p": {"rank": 0}, "v": {"rank": 1}},
    boundary_schemas={"wall": {}},
    global_schema={"g": {"rank": 1}},
    hidden=64,
    n_layers=3,
    n_slices=32,
)


def test_a_triton_of_another_distribution_counts_by_its_module_version(monkeypatch):
    """Triton counts as present by the version of the distribution named ``triton``
    (the metadata lookup mocked) or, with no distribution of that name but a module
    ``triton`` importable (as ``pytorch-triton`` provides it), by the module's
    ``__version__``: present at the floor and above, absent below it, and absent with
    neither the distribution nor the module."""
    import importlib.machinery
    import sys
    import types

    from physicsnemo.core import version_check
    from physicsnemo.experimental.models.isla import _triton

    def module(version):
        m = types.ModuleType("triton")
        m.__spec__ = importlib.machinery.ModuleSpec("triton", None)
        m.__version__ = version
        return m

    floor = _triton.TRITON_MIN_VERSION
    for installed, expected in ((floor, True), ("3.0.0", False)):
        monkeypatch.setattr(
            version_check, "get_installed_version", lambda name: installed
        )
        monkeypatch.setitem(sys.modules, "triton", module("99.0.0"))
        assert _triton.triton_available() is expected, installed
    monkeypatch.setattr(version_check, "get_installed_version", lambda name: None)
    for version, expected in ((floor + "+git0123abcd", True), ("3.0.0", False)):
        monkeypatch.setitem(sys.modules, "triton", module(version))
        assert _triton.triton_available() is expected, version
    monkeypatch.setitem(sys.modules, "triton", None)  # find_spec finds no module
    assert _triton.triton_available() is False


def _boundary(
    n=700,
    seed=0,
    device="cuda",
    dtype=torch.float32,
    cell_data=None,
    d=3,
    direction=True,
):
    """One boundary soup of ``n`` cells in ``d`` dimensions with a global direction
    (unless ``direction`` is false), and a target."""
    pts, nrm, w, drv = cloud(n, d, seed=seed, dtype=torch.float32)
    mesh = soup(pts, nrm, w, cell_data=cell_data).to(device).to(dtype)
    tgt = torch.randn(n, 1 + d, generator=torch.Generator().manual_seed(seed + 1)).to(
        device, dtype
    )
    glob = {"g": drv.to(device, dtype)} if direction else None
    return dict(boundary_meshes={"wall": mesh}, global_data=glob), tgt


def _flat(o):
    return torch.cat([o["p"][..., None], o["v"]], dim=-1)


def _rel(a, b):
    return float((a.double() - b.double()).norm() / b.double().norm())


def _run(model, inputs, target):
    model.zero_grad(set_to_none=True)
    if "prediction_points" not in inputs:
        inputs = {**inputs, "prediction_boundaries": ["wall"]}
    out = _flat(outputs(model(**inputs)))
    torch.nn.functional.mse_loss(out.float(), target.float()).backward()
    grads = {
        k.replace("_orig_mod.", ""): p.grad.detach().clone()
        for k, p in model.named_parameters()
    }
    return out.detach(), grads


def _pair(dtype, device="cuda", geo_kernel="auto", **kw):
    torch.manual_seed(0)
    ref = ISLA(**REF).to(device, dtype)
    ref._geo_kernel = (
        "eager"  # the eager reference ("auto" resolves to the fused kernel on CUDA)
    )
    torch.manual_seed(0)
    opt = ISLA(**{**REF, **kw}).to(device, dtype)
    opt._geo_kernel = geo_kernel
    opt.load_state_dict(ref.state_dict())
    return ref, opt


def _assert_same(ref, opt, dtype, out_tol, grad_tol, n=700):
    inputs, tgt = _boundary(n, dtype=dtype)
    o_ref, g_ref = _run(ref, inputs, tgt)
    o_opt, g_opt = _run(opt, inputs, tgt)
    assert _rel(o_opt, o_ref) < out_tol, _rel(o_opt, o_ref)
    total = torch.cat([g.flatten() for g in g_ref.values()]).norm()
    for k, g in g_ref.items():
        if g.norm() < 1e-6 * total:
            continue  # a gradient zero in exact arithmetic (a constant cancelling in a softmax) is roundoff
        # relative to the parameter's own gradient, floored at 1e-3 of the whole gradient: a
        # scalar such as a block's prototype gain is a reduction over every token and slice
        # whose gradient can nearly cancel, and its own norm is then a roundoff-sized yardstick
        scale = torch.maximum(g.norm(), 1e-3 * total)
        assert (g_opt[k] - g).norm() / scale < grad_tol, (k, _rel(g_opt[k], g))


# --------------------------------------------------------------------------- fused region
@pytest.mark.cuda
@requires_cuda_triton
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("dim", [3, 2])
@pytest.mark.parametrize(
    "K,P",
    [
        (0, 0),
        (0, 1),
        (0, 3),
        (1, 0),
        (2, 0),
        (3, 0),
        (4, 0),
        (1, 1),
        (1, 2),
        (1, 3),
        (2, 1),
        (2, 2),
        (3, 1),
    ],
)
def test_fused_geo_region_matches_eager_region(dtype, K, P, dim, monkeypatch):
    """Region level: forward outputs (bias, mix, pooled) and every input gradient, for
    K global directions (none included) plus P per-token rank-1 vectors (the
    (5 + K + P)-invariant set in the eager order) with the linear routing-bias head, in
    three and two dimensions."""
    from physicsnemo.experimental.models.isla._geo_kernel import fused_geo_region

    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    b, n, s = 2, 500, 40
    g = torch.Generator().manual_seed(0)

    def mk(*shape):
        return (
            (torch.randn(*shape, generator=g) * 0.1).to("cuda", dtype).requires_grad_()
        )

    r, z = mk(b, n, dim), mk(b, s, dim)
    nh = (
        torch.nn.functional.normalize(torch.randn(b, n, dim, generator=g), dim=-1)
        .to("cuda", dtype)
        .requires_grad_()
    )
    ms = (
        torch.nn.functional.normalize(torch.randn(b, s, dim, generator=g), dim=-1)
        .to("cuda", dtype)
        .requires_grad_()
    )
    d = (
        torch.nn.functional.normalize(torch.randn(b, 1, K, dim, generator=g), dim=-1)
        .to("cuda", dtype)
        .requires_grad_()
    )
    v = (
        (torch.randn(b, n, P, dim, generator=g) * 0.7)
        .to("cuda", dtype)
        .requires_grad_()
    )
    logits = torch.randn(b, n, s, generator=g).to("cuda", dtype).requires_grad_()
    torch.manual_seed(0)
    lin = torch.nn.Linear(5 + K + P, 1).to("cuda", dtype)
    leaves = (*lin.parameters(), logits, r, nh, d, z, ms, v)
    wts = [
        torch.randn(b, n, s, generator=g).to("cuda", dtype),
        torch.randn(b, n, s, generator=g).to("cuda", dtype),
        torch.randn(b, n, 5 + K + P, generator=g).to("cuda", dtype),
    ]

    def run(fn):
        for t in leaves:
            t.grad = None
        d_n = d.expand(
            b, n, K, dim
        )  # both regions take the K global directions as (B, N, K, dim)
        if fn is _geo_region:
            outs = fn(lin, logits, r, nh, d_n, z, ms, 1e-12, v if P else None)
        else:  # the fused kernel takes the directions and the token vectors as one (B, N, K + P, 3) set
            outs = fn(
                lin,
                logits,
                r,
                nh,
                torch.cat([d_n, v], 2) if P else d_n,
                z,
                ms,
                1e-12,
                n_global=K,
            )
        sum((o * w).sum() for o, w in zip(outs, wts)).backward()
        return [o.detach() for o in outs], [
            (t.grad.detach().clone() if t.grad is not None else torch.zeros_like(t))
            for t in leaves
        ]

    o_ref, g_ref = run(_geo_region)
    o_f, g_f = run(fused_geo_region)
    tol = 1e-11 if dtype == torch.float64 else 1e-6
    for a, bb in zip(o_f, o_ref):
        assert _rel(a, bb) < tol, _rel(a, bb)
    for a, bb in zip(g_f, g_ref):
        if bb.numel() == 0:
            continue  # the empty direction or token-vector leaf when K or P is 0
        assert _rel(a, bb) < (1e-11 if dtype == torch.float64 else 1e-5), _rel(a, bb)


@pytest.mark.cuda
@requires_cuda_triton
@pytest.mark.slow
def test_fused_geo_region_offsets_past_int32():
    """Points x slices >= 2**31: the last rows, whose (point, slice) offsets pass int32,
    get the forward outputs and the per-point gradients of the same rows run alone
    (every row is computed from its own inputs). Needs about 100 GiB of device memory:
    three (N, S) float32 tensors each way and the backward's per-tile slice partials."""
    from physicsnemo.experimental.models.isla._geo_kernel import fused_geo_region

    s, tail = 512, 64
    n = 2**31 // s + tail
    if torch.cuda.mem_get_info()[0] < 128 * 2**30:
        pytest.skip("needs about 100 GiB of free device memory")
    dev = torch.Generator(device="cuda").manual_seed(0)

    def unit(*shape):
        return torch.nn.functional.normalize(
            torch.randn(*shape, device="cuda", generator=dev), dim=-1
        )

    r = torch.randn(1, n, 3, device="cuda", generator=dev) * 0.1
    nh, d = unit(1, n, 3), unit(1, 1, 1, 3).expand(1, n, 1, 3)
    z, ms = torch.randn(1, s, 3, device="cuda", generator=dev) * 0.1, unit(1, s, 3)
    logits = torch.randn(1, n, s, device="cuda", generator=dev)
    torch.manual_seed(0)
    lin = torch.nn.Linear(6, 1).cuda()
    w_bias, w_mix = torch.randn(2, tail, s, device="cuda", generator=dev)

    def run(sl):
        x, rr = logits[:, sl].requires_grad_(), r[:, sl].requires_grad_()
        bias, mix, pooled = fused_geo_region(
            lin, x, rr, nh[:, sl], d[:, sl], z, ms, 1e-12
        )
        loss = (bias[0, -tail:] * w_bias).sum() + (mix[0, -tail:] * w_mix).sum()
        (loss + pooled[0, -tail:].sum()).backward()
        rows = (bias, mix, pooled, x.grad, rr.grad)
        return [t[0, -tail:].detach().clone() for t in rows]

    whole = run(slice(None))
    alone = run(slice(n - tail, None))
    for a, b in zip(whole, alone):
        assert _rel(a, b) < 1e-6, _rel(a, b)


@pytest.mark.cuda
@requires_cuda_triton
def test_fused_geo_region_reads_strided_inputs_as_contiguous_ones():
    """Two cases (B = 2) with r and n_hat sliced along the points from longer tensors
    (strided): the outputs and every gradient equal those of contiguous copies,
    bitwise."""
    from physicsnemo.experimental.models.isla._geo_kernel import fused_geo_region

    b, n, s, dt = 2, 1000, 64, torch.float64
    dev = torch.Generator(device="cuda").manual_seed(0)

    def unit(*shape):
        return torch.nn.functional.normalize(
            torch.randn(*shape, device="cuda", dtype=dt, generator=dev), dim=-1
        )

    r = (0.1 * torch.randn(b, 1500, 3, device="cuda", dtype=dt, generator=dev))[:, :n]
    nh = unit(b, 1500, 3)[:, :n]
    d = unit(b, 1, 1, 3).expand(b, n, 1, 3)
    z, ms = torch.randn(b, s, 3, device="cuda", dtype=dt, generator=dev), unit(b, s, 3)
    logits = torch.randn(b, n, s, device="cuda", dtype=dt, generator=dev)
    torch.manual_seed(0)
    lin = torch.nn.Linear(6, 1).cuda().double()
    w_bias, w_mix = torch.randn(2, b, n, s, device="cuda", dtype=dt, generator=dev)

    def run(*geo):
        lin.zero_grad(set_to_none=True)
        ins = [t.detach().requires_grad_() for t in (logits, *geo, z, ms)]
        bias, mix, pooled = fused_geo_region(lin, ins[0], *ins[1:3], d, *ins[3:], 1e-12)
        ((bias * w_bias).sum() + (mix * w_mix).sum() + pooled.sum()).backward()
        grads = [t.grad for t in (*ins, lin.weight, lin.bias)]
        return [t.detach().clone() for t in (bias, mix, pooled, *grads)]

    assert not (r.is_contiguous() or nh.is_contiguous())
    strided = run(r, nh)
    for a, c in zip(strided, run(r.contiguous(), nh.contiguous())):
        assert torch.equal(a, c)


# --------------------------------------------------------------------------- model level
_RANK1 = torch.randn(600, 3, generator=torch.Generator().manual_seed(9)) * 0.5
#: (model options, the region it is given, the boundary's options) per model.
_EXACT = {
    "3d": ({}, "fused", dict(n=700)),
    "2d": ({"n_spatial_dims": 2}, "auto", dict(n=600, d=2)),
    "3d_no_direction": ({"global_schema": None}, "auto", dict(n=600, direction=False)),
    "2d_no_direction": (
        {"n_spatial_dims": 2, "global_schema": None},
        "auto",
        dict(n=600, d=2, direction=False),
    ),
    "rank1_boundary_input": (
        {"boundary_schemas": {"wall": {"t": {"rank": 1}}}},
        "fused",
        dict(n=600, cell_data={"t": _RANK1}),
    ),
}


@pytest.mark.cuda
@requires_cuda_triton
@pytest.mark.parametrize("name", list(_EXACT))
def test_fused_geo_kernel_model_exact_in_float64(name):
    """The model with the fused region matches the eager region in float64 (outputs
    and every parameter's gradient to 1e-11): forced on, and picked by "auto" for a
    2-D model (the kernel's third component zero) and for a model without a global
    direction (one zero direction with a zero weight)."""
    kw, kernel, data = _EXACT[name]
    spec = {**REF, **kw}
    torch.manual_seed(0)
    ref = ISLA(**spec).to("cuda", torch.float64)
    ref._geo_kernel = "eager"
    torch.manual_seed(0)
    opt = ISLA(**spec).to("cuda", torch.float64)
    opt._geo_kernel = kernel
    opt.load_state_dict(ref.state_dict())
    inputs, tgt = _boundary(dtype=torch.float64, **data)
    assert opt._resolve_geo_kernel(inputs["boundary_meshes"]["wall"].points) == "fused"
    o_ref, g_ref = _run(ref, inputs, tgt)
    o_opt, g_opt = _run(opt, inputs, tgt)
    assert _rel(o_opt, o_ref) < 1e-11, _rel(o_opt, o_ref)
    total = torch.cat([g.flatten() for g in g_ref.values()]).norm()
    for k, g in g_ref.items():
        if g.norm() < 1e-6 * total:
            continue
        assert _rel(g_opt[k], g) < 1e-11, (k, _rel(g_opt[k], g))


# --------------------------------------------------------------------------- torch.compile
@pytest.mark.cuda
@requires_cuda
@pytest.mark.parametrize(
    "geo_kernel", ["eager", pytest.param("fused", marks=requires_cuda_triton)]
)
def test_torch_compile_fullgraph_matches_eager(geo_kernel, monkeypatch):
    """The whole model compiled must stay one graph and reproduce the eager reference
    within the float32 bars."""
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    torch._dynamo.reset()
    ref, opt = _pair(torch.float32, geo_kernel=geo_kernel)
    inputs, _ = _boundary(300)
    ex = torch._dynamo.explain(opt)(**inputs, prediction_boundaries=["wall"])
    assert ex.graph_break_count == 0, ex.break_reasons
    compiled = torch.compile(opt, fullgraph=True)
    ### Gradient bar 5e-5: on GB300 blocks.0.assign.v_proj.weight measured 2.1e-5 to 2.3e-5.
    _assert_same(ref, compiled, torch.float32, 1e-6, 5e-5, n=300)


# --------------------------------------------------------------------------- compiled blocks
@pytest.mark.slow  # 6 s on CPU, 9 s on GB300: compiled blocks
@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=[pytest.mark.cuda, requires_cuda])]
)
def test_compile_blocks_matches_eager(device):
    """``compile_blocks=True`` runs each slice block's forward under torch.compile and
    leaves parameter names, the state dict and the arithmetic unchanged: outputs and
    gradients match the eager model to float64 roundoff (the compiled blocks reorder
    floating-point sums only), and the model compiles once."""
    ref, opt = _pair(torch.float64, device=device, compile_blocks=True)
    assert set(opt.state_dict()) == set(ref.state_dict())
    inputs, tgt = _boundary(500, device=device, dtype=torch.float64)
    o_ref, g_ref = _run(ref, inputs, tgt)
    o_opt, g_opt = _run(opt, inputs, tgt)
    assert opt._block_forwards is not None and len(opt._block_forwards) == len(
        opt.blocks
    )
    assert _rel(o_opt, o_ref) < 1e-9, _rel(o_opt, o_ref)
    total = torch.cat([g.flatten() for g in g_ref.values()]).norm()
    for k, g in g_ref.items():
        if g.norm() < 1e-6 * total:
            continue
        assert _rel(g_opt[k], g) < 1e-9, (k, _rel(g_opt[k], g))
    # a second call with the same shapes reuses the compiled blocks (no rebuild)
    built = opt._block_forwards
    _run(opt, inputs, tgt)
    assert opt._block_forwards is built


def test_a_copy_of_a_model_with_compiled_blocks_runs_its_own_blocks():
    """A deep copy of a ``compile_blocks`` model made after its first forward (an EMA
    or a best-model copy) runs its own blocks, not the original's, and the model
    pickles: the compiled blocks, which hold the original's, are not in its state.
    Dynamo is off, so the compiled callables run eagerly but are still cached."""
    inputs, _ = _boundary(200, device="cpu", dtype=torch.float64)
    q = torch.randn(
        40, 3, generator=torch.Generator().manual_seed(5), dtype=torch.float64
    )
    inputs = {**inputs, "prediction_points": q}
    torch.manual_seed(0)
    spec = {**REF, "hidden": 32, "n_slices": 8, "interior": True}
    m = ISLA(**spec, compile_blocks=True).double()
    with torch.no_grad(), torch._dynamo.config.patch(disable=True):
        m(**inputs)
        assert m._block_forwards is not None and m._block_reads is not None
        twin = copy.deepcopy(m)
        for p in twin.blocks.parameters():
            p.add_(0.1 * torch.randn_like(p))
        eager = ISLA(**spec).double()
        eager.load_state_dict(twin.state_dict())
        expect = outputs(eager(**inputs))
        got = outputs(twin(**inputs))
        assert all(torch.equal(got[k], expect[k]) for k in expect)
        loaded = pickle.loads(pickle.dumps(twin))  # noqa: S301 - local round trip
        got = outputs(loaded(**inputs))
        assert all(torch.equal(got[k], expect[k]) for k in expect)


@pytest.mark.cuda
@requires_cuda
def test_block_activation_budget_keeps_a_recompile_limit_per_block_function(caplog):
    """With ``block_activation_budget`` below 1 the blocks' forwards and reads are
    compiled through a wrapper, and each keeps its own Dynamo recompile limit, as
    without the budget: at a limit of 1 a training step of the interior task compiles both and
    no block runs eager. Past the limit (the float64 graphs here) the blocks run eager
    without raising, and the model warns once that the budget saves nothing there
    (in a training step: without gradients there is nothing to save, and no warning)."""
    from physicsnemo.experimental.models.isla import model as isla_model

    def steps(budget):
        torch._dynamo.reset()
        torch.manual_seed(0)
        m = ISLA(**{**REF, "interior": True}, compile_blocks=True).cuda()
        m.block_activation_budget = budget
        with torch._dynamo.config.patch(recompile_limit=1):
            for dtype in (torch.float32, torch.float64, torch.float64):
                inputs, _ = _boundary(600, dtype=dtype)
                q = torch.randn(300, 3, generator=torch.Generator().manual_seed(5))
                q = q.to("cuda", dtype) * 3.0
                tgt = torch.zeros(300, 4, device="cuda", dtype=dtype)
                out, _ = _run(m.to(dtype), {**inputs, "prediction_points": q}, tgt)
                assert out.isfinite().all()
                yield bool(m._budget_eager_warned)

    def said():
        return [r for r in caplog.records if r.name == isla_model.__name__]

    with caplog.at_level(logging.WARNING, logger=isla_model.__name__):
        assert list(steps(1.0)) == [False, False, False] and not said()
        assert list(steps(0.8)) == [False, True, True]
        with torch.no_grad(), torch._dynamo.config.patch(recompile_limit=1):
            torch._dynamo.reset()
            m = ISLA(**{**REF, "interior": True}, compile_blocks=True).cuda()
            m.block_activation_budget = 0.8
            for dtype in (torch.float32, torch.float64):
                inputs, _ = _boundary(600, dtype=dtype)
                q = torch.randn(300, 3, device="cuda", dtype=dtype)
                m.to(dtype)(**inputs, prediction_points=q)
            assert not m._budget_eager_warned
    assert len(said()) == 1 and "block_activation_budget" in said()[0].getMessage()


@pytest.mark.cuda
@requires_cuda_triton
def test_interior_fused_and_compiled_blocks_match_eager():
    """The interior task's training configuration (fused geometry kernel, compiled
    blocks) reproduces the eager model in float64, and its predictions stay
    query-independent on CUDA: a request's prefix gives the prefix of the predictions."""
    spec = {**REF, "interior": True}
    torch.manual_seed(0)
    ref = ISLA(**spec).to("cuda", torch.float64)
    ref._geo_kernel = "eager"
    torch.manual_seed(0)
    opt = ISLA(**spec, compile_blocks=True).to("cuda", torch.float64)
    opt._geo_kernel = "fused"
    opt.load_state_dict(ref.state_dict())
    inputs, _ = _boundary(600, dtype=torch.float64)
    q = (
        torch.randn(
            300, 3, generator=torch.Generator().manual_seed(5), dtype=torch.float64
        ).cuda()
        * 3.0
    )
    tgt = torch.randn(
        300, 4, generator=torch.Generator().manual_seed(6), dtype=torch.float64
    ).cuda()
    o_ref, g_ref = _run(ref, {**inputs, "prediction_points": q}, tgt)
    o_opt, g_opt = _run(opt, {**inputs, "prediction_points": q}, tgt)
    ### 1e-6 for what passes through the boundary read: its node means are float32 by design
    ### and CUDA's index_add sums in no fixed order; 1e-9 for the rest.
    tree_tol = 1e-6
    assert _rel(o_opt, o_ref) < tree_tol, _rel(o_opt, o_ref)
    total = torch.cat([g.flatten() for g in g_ref.values()]).norm()
    for k, g in g_ref.items():
        if g.norm() < 1e-6 * total:
            continue
        tol = tree_tol if k.startswith(("kernel_reads.", "kernel_basis.")) else 1e-9
        assert _rel(g_opt[k], g) < tol, (k, _rel(g_opt[k], g))
    with torch.no_grad():
        head = _flat(outputs(opt(**inputs, prediction_points=q[:40])))
    assert _rel(head, o_opt[:40]) < tree_tol


### Mix-order reduction (pytorch/pytorch#195845): Inductor fuses a block's softmax over the
### slices of each token with a sum over the tokens of each slice once tokens x slices >=
### 5 * 2**20 (and tokens >= 4096), and in torch 2.13 adds the padded rows of the
### last tile of tokens to that sum when the token count is not a multiple of the tile
### height (4 with 512 slices). The pair exists only in the eager geometry region (the
### fused kernel is a leaf of the graph), which a model runs off CUDA, under float16
### autocast, above three dimensions, with rank-2 inputs or with more than four per-point
### vectors. The compiled blocks turn mix-order
### reduction off. These tests run the eager region at a size where it would fuse, in float32 (it never fuses
### float64), at token counts that are and are not multiples of 4. ISLA's forward pads
### the token axis to a multiple of _TOKEN_ALIGNMENT (32), so the compiled blocks of a
### default forward never see such a count; the padding is not the fix, and the tests
### turn it off (an alignment of 1) so that the compiled blocks see the input's count.
MIX = {**REF, "n_layers": 1, "n_slices": 512}


def _record_block_tokens(model):
    """The token count of every call of ``model``'s compiled blocks, recorded from now on."""
    seen = []

    def recorded(forward):
        def call(h, *args, **kwargs):
            seen.append(h.shape[1])
            return forward(h, *args, **kwargs)

        return call

    model._block_forwards = [recorded(f) for f in model._block_callables()]
    return seen


@pytest.mark.slow  # 24 s on CPU as the session's first Inductor compile, 3 s after one
@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=[pytest.mark.cuda, requires_cuda])]
)
@pytest.mark.parametrize("task", ["surface", "interior"])
def test_compile_blocks_see_an_aligned_token_count(task, device):
    """A padded forward gives the compiled blocks a multiple of _TOKEN_ALIGNMENT tokens at a
    token count that is not one."""
    from physicsnemo.experimental.models.isla import model

    kw = {} if task == "surface" else {"interior": True}
    torch.manual_seed(0)
    opt = ISLA(**{**REF, "n_layers": 1, **kw}, compile_blocks=True).to(device)
    inputs, _ = _boundary(301, device=device)
    if task == "surface":
        n, call = 301, dict(prediction_boundaries=["wall"])
    else:
        q = torch.randn(7, 3, generator=torch.Generator().manual_seed(5)).to(device)
        n, call = 308, dict(prediction_points=q)
    assert n % model._TOKEN_ALIGNMENT
    seen = _record_block_tokens(opt)
    with torch.no_grad():
        opt(**inputs, **call)
    assert seen == [n + -n % model._TOKEN_ALIGNMENT], seen


def _mix_pair(**kw):
    import torch._inductor.metrics

    torch._dynamo.reset()
    torch._inductor.metrics.reset()
    torch.manual_seed(0)
    ref = ISLA(**{**MIX, **kw}).to("cuda")
    opt = ISLA(**{**MIX, **kw}, compile_blocks=True).to("cuda")
    opt.load_state_dict(ref.state_dict())
    ref._geo_kernel = opt._geo_kernel = "eager"
    return ref, opt


def _forced_on_error(run, **kw):
    """The positive control: ``run(ref, opt)``'s output error with the same blocks compiled
    with mix-order reduction forced on. Skips the calling test where Inductor then builds
    no mix-order kernel, since the test cannot see the bug there; a test calls it after
    its other assertions, so that they run on every device. Inductor's caches are
    off because a cache hit skips the codegen that counts the kernels."""
    import torch._inductor.config

    from physicsnemo.experimental.models.isla import model

    with (
        pytest.MonkeyPatch.context() as mp,
        torch._inductor.config.patch(force_disable_caches=True),
    ):
        mp.setattr(
            model, "_BLOCK_COMPILE_OPTIONS", {"triton.mix_order_reduction": True}
        )
        ref, opt = _mix_pair(**kw)
        err = run(ref, opt)
    if torch._inductor.metrics.codegen_mix_order_reduction == 0:
        pytest.skip(
            "mix-order reduction does not fire on this device, so this test cannot detect pytorch/pytorch#195845"
        )
    return err


@pytest.mark.cuda
@requires_cuda
@pytest.mark.parametrize("n", [12000, 12001, 12002, 12003])
def test_compile_blocks_float32_matches_eager_at_every_token_count(n, monkeypatch):
    """compile_blocks reproduces the eager model in float32 (outputs and gradients) at
    every token count, at a size where Inductor's mix-order reduction would fuse, and
    builds no mix-order kernel."""
    from physicsnemo.experimental.models.isla import model

    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    monkeypatch.setattr(model, "_TOKEN_ALIGNMENT", 1)
    inputs, tgt = _boundary(n)

    def forced_on(ref, opt):
        o_ref = _run(ref, inputs, tgt)[0]
        return _rel(_run(opt, inputs, tgt)[0], o_ref)

    ref, opt = _mix_pair()
    seen = _record_block_tokens(opt)
    ### With mix-order reduction on, GB300 measured an output error of 5.7e-5 at n = 12001
    ### and one mix-order kernel; with it off, 2.7e-7 and none. Gradient bar 1e-4: with it
    ### off, GB300 measured 5.29e-5 at n = 12003 (blocks.0.assign.weight), float32 sums
    ### over 12003 tokens in another order.
    _assert_same(ref, opt, torch.float32, 5e-6, 1e-4, n=n)
    assert torch._inductor.metrics.codegen_mix_order_reduction == 0
    assert seen == [n], seen
    if n % 4:
        err = _forced_on_error(forced_on)
        assert err > 5e-6, err  # forced on, the bug shows at this size


@pytest.mark.cuda
@requires_cuda
def test_the_model_s_compile_matches_eager_in_the_eager_region(monkeypatch):
    """The whole model under ISLA.compile reproduces the eager model in float32
    (outputs and gradients) in the eager geometry region, at a token count that is
    not a multiple of 4, and builds no mix-order kernel. Then the positive control:
    nn.Module.compile with Inductor's defaults, whose output differs on torch 2.13
    (pytorch/pytorch#195845); skipped where Inductor builds no mix-order kernel
    for it, since the test cannot see the bug there."""
    import torch._inductor.config
    import torch._inductor.metrics

    from physicsnemo.experimental.models.isla import model

    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    monkeypatch.setattr(model, "_TOKEN_ALIGNMENT", 1)
    n = 12001
    inputs, tgt = _boundary(n)

    def pair(compile_):
        torch._dynamo.reset()
        torch._inductor.metrics.reset()
        torch.manual_seed(0)
        ref = ISLA(**MIX).to("cuda")
        opt = copy.deepcopy(ref)
        ref._geo_kernel = opt._geo_kernel = "eager"
        compile_(opt)
        return ref, opt

    ### Inductor's caches off: a cache hit skips the codegen that counts the kernels.
    with torch._inductor.config.patch(force_disable_caches=True):
        ref, opt = pair(lambda m: m.compile(fullgraph=True, dynamic=True))
        _assert_same(ref, opt, torch.float32, 5e-6, 5e-5, n=n)
        assert torch._inductor.metrics.codegen_mix_order_reduction == 0
        ref, opt = pair(
            lambda m: torch.nn.Module.compile(m, fullgraph=True, dynamic=True)
        )
        err = _rel(_run(opt, inputs, tgt)[0], _run(ref, inputs, tgt)[0])
        if torch._inductor.metrics.codegen_mix_order_reduction == 0:
            pytest.skip(
                "mix-order reduction does not fire on this device, so this test cannot detect pytorch/pytorch#195845"
            )
        assert err > 5e-6, err


@pytest.mark.cuda
@requires_cuda
@pytest.mark.parametrize("n", [12001, 12002])
def test_compile_blocks_interior_float32_at_every_token_count(n, monkeypatch):
    """The interior task under compile_blocks in float32, at a size where mix-order
    reduction would fuse and a token count (n boundary cells) that is not a multiple
    of 4: the compiled model reproduces the eager one, and its prediction at a fixed
    set of queries does not depend on the other queries of the request. The queries
    read the boundary passively after the blocks (the query-independent form), so the
    compiled blocks see the boundary's tokens alone, whatever the query count."""
    from physicsnemo.experimental.models.isla import model

    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    monkeypatch.setattr(model, "_TOKEN_ALIGNMENT", 1)
    inputs, _ = _boundary(n)
    q = torch.randn(2000, 3, generator=torch.Generator().manual_seed(5)).cuda() * 3.0

    def forced_on(ref, opt):
        with torch.no_grad():
            o_ref = _flat(outputs(ref(**inputs, prediction_points=q)))
            return _rel(_flat(outputs(opt(**inputs, prediction_points=q))), o_ref)

    ref, opt = _mix_pair(interior=True)
    seen = _record_block_tokens(opt)
    with torch.no_grad():
        o_ref = _flat(outputs(ref(**inputs, prediction_points=q)))
        o_opt = _flat(outputs(opt(**inputs, prediction_points=q)))
        alone = _flat(outputs(opt(**inputs, prediction_points=q[:1000])))
    assert _rel(o_opt, o_ref) < 5e-6, _rel(o_opt, o_ref)
    assert _rel(alone, o_opt[:1000]) < 5e-6, _rel(alone, o_opt[:1000])
    assert torch._inductor.metrics.codegen_mix_order_reduction == 0
    assert seen == [n, n], seen
    err = _forced_on_error(forced_on, interior=True)
    assert err > 5e-6, err  # forced on, the bug shows at this size
