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

"""ISLA's kernels against their references, where the other tests leave a gap.

- The fused geometry region: its bfloat16 backward against the eager region under the
  same autocast, slice counts past each launch branch (257, 513, 1024), and every
  tile size.
- The segment-softmax read: its bfloat16 gradients, rounded once.
- The fused read's gradient in the log measure.
- The fused basis under float16 autocast (``unsupported_reason`` accepts it).
- Paths no other test runs: the forest build's depth-bin fallback, the walk of an
  empty forest or of no points and its first size limit,
  ``fused_basis.unsupported_reason``'s branches, and the geometry kernel's
  disqualifiers on CUDA.

The 2x-error rule (as in test_isla_tree_read.py): per tensor, the fast path's largest
distance to float64 eager autograd, relative to the float64 tensor's largest entry,
is at most twice the eager path's in the same precision plus a floor at that
precision's roundoff: 1e-6 in float32, 1e-3 in bfloat16 and float16 (whose unit
roundoffs are 2**-8 and 2**-11; the floor stands for the eager path's own error
landing near zero by chance).
"""

import logging

import pytest
import torch

### The modules that own each function compared here.
from physicsnemo.experimental.models.isla import ISLA
from physicsnemo.experimental.models.isla import _fused_basis as fused_basis
from physicsnemo.experimental.models.isla import _fused_read as fused_read
from physicsnemo.experimental.models.isla import _geo_kernel as geo_kernel
from physicsnemo.experimental.models.isla import _segment_softmax as segment_softmax
from physicsnemo.experimental.models.isla import _tree_read as tree_read
from physicsnemo.experimental.models.isla._blocks import _geo_region
from physicsnemo.experimental.models.isla._segment_softmax import PairPlan
from physicsnemo.experimental.models.isla._tree_read import BoundaryForest
from test.experimental.models.isla._forced import ran
from test.experimental.models.isla._kernel_cases import (
    _basis_case,
    _basis_step,
    _read_case,
    _to,
    _two_x,
)
from test.experimental.models.isla._marks import requires_cuda_triton
from test.experimental.models.isla._meshes import cloud

D = torch.float64
_LOW = {torch.float32: 1e-6, torch.bfloat16: 1e-3, torch.float16: 1e-3}


def _rel(a, b) -> float:
    return float((a.double() - b.double()).norm() / b.double().norm())


# --------------------------------------------------------------------------- fused geometry region
def _region_case(b, n, s, k, p, dtype, seed=0):
    """Leaves of one geometry region on CUDA (pre-logits, point and slice geometry,
    ``k`` global directions and ``p`` token vectors, the routing Linear) and the
    weights of a scalar loss over its three outputs."""
    g = torch.Generator().manual_seed(seed)

    def mk(*shape, scale=1.0, unit=False):
        x = torch.randn(*shape, generator=g, dtype=D)
        x = torch.nn.functional.normalize(x, dim=-1) if unit else scale * x
        return x.to("cuda", dtype).requires_grad_()

    leaves = dict(
        logits=mk(b, n, s),
        r=mk(b, n, 3, scale=0.1),
        nh=mk(b, n, 3, unit=True),
        d=mk(b, 1, k, 3, unit=True),
        v=mk(b, n, p, 3, scale=0.7),
        z=mk(b, s, 3, scale=0.1),
        ms=mk(b, s, 3, unit=True),
        weight=mk(1, 5 + k + p, scale=0.3),
        bias=mk(1, scale=0.3),
    )
    wts = [
        torch.randn(b, n, s, generator=g, dtype=D),
        torch.randn(b, n, s, generator=g, dtype=D),
    ]
    wts.append(torch.randn(b, n, 5 + k + p, generator=g, dtype=D))
    return leaves, wts


def _run_region(fused, leaves, wts, k, p, autocast=False, logits_bf16=False):
    """One region's outputs and every leaf's gradient (zeros for an empty leaf)."""
    x = {name: t.detach().clone().requires_grad_() for name, t in leaves.items()}
    lin = torch.nn.Linear(5 + k + p, 1).to(x["r"].device, x["r"].dtype)
    with torch.no_grad():
        lin.weight.copy_(x["weight"])
        lin.bias.copy_(x["bias"])
    lin.weight.requires_grad_(True)
    lin.bias.requires_grad_(True)
    b, n = x["r"].shape[:2]
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
        logits = x["logits"].bfloat16() if logits_bf16 else x["logits"]
        d_n = x["d"].expand(b, n, k, 3)
        if fused:
            vecs = torch.cat([d_n, x["v"]], 2) if p else d_n
            outs = geo_kernel.fused_geo_region(
                lin, logits, x["r"], x["nh"], vecs, x["z"], x["ms"], 1e-12, n_global=k
            )
        else:
            outs = _geo_region(
                lin,
                logits,
                x["r"],
                x["nh"],
                d_n,
                x["z"],
                x["ms"],
                1e-12,
                x["v"] if p else None,
            )
    sum((o.double() * w.to(o.device)).sum() for o, w in zip(outs, wts)).backward()
    grads = {
        name: t.grad if t.grad is not None else torch.zeros_like(t)
        for name, t in x.items()
        if name not in ("weight", "bias")
    }
    grads["weight"], grads["bias"] = lin.weight.grad, lin.bias.grad
    return [o.detach() for o in outs], grads


@pytest.mark.cuda
@requires_cuda_triton
@pytest.mark.parametrize("bf16", [True, False], ids=["bf16_logits", "fp32_logits"])
@pytest.mark.parametrize("k, p", [(1, 0), (2, 1)], ids=["K1P0", "K2P1"])
def test_fused_geo_region_bf16_backward_obeys_the_2x_error_rule(k, p, bf16):
    """Under bfloat16 autocast (bfloat16 or float32 pre-logits, the model's is
    float32), the fused region's three outputs and the gradients of every leaf are as
    close to the float64 eager region on the same inputs as the eager region under the
    same autocast is (the 2x-error rule, floor 1e-3). The fused backward accumulates in
    float32 where autocast's GEMM backward rounds to bfloat16, so it is the closer."""
    leaves, wts = _region_case(1, 800, 64, k, p, torch.float32)
    exact_leaves = {n: t.detach().double() for n, t in leaves.items()}
    if bf16:
        exact_leaves["logits"] = leaves["logits"].detach().bfloat16().double()
    exact = _run_region(False, exact_leaves, wts, k, p)
    eager = _run_region(False, leaves, wts, k, p, autocast=True, logits_bf16=bf16)
    fused = _run_region(True, leaves, wts, k, p, autocast=True, logits_bf16=bf16)
    assert [t.dtype for t in fused[0]] == [t.dtype for t in eager[0]]
    for i, name in enumerate(("bias", "mix", "pooled")):
        _two_x(name, fused[0][i], eager[0][i], exact[0][i], 1e-3)
    for name in exact[1]:
        if p == 0 and name == "v":
            continue
        _two_x(f"grad {name}", fused[1][name], eager[1][name], exact[1][name], 1e-3)


def _assert_region_exact(fused, eager, tol=1e-11):
    """float64: outputs and every gradient at ``tol`` relative in norm (the same
    arithmetic in another summation order)."""
    for a, b in zip(fused[0], eager[0]):
        assert _rel(a, b) < tol, _rel(a, b)
    for name, g in eager[1].items():
        if g.numel() and g.norm() > 0:
            assert _rel(fused[1][name], g) < tol, (name, _rel(fused[1][name], g))


@pytest.mark.cuda
@requires_cuda_triton
@pytest.mark.parametrize("s", [257, 513, 1024])
def test_fused_geo_region_matches_eager_past_every_slice_branch(s):
    """Slice counts past the launch branches at 256 and 512 slices, and the most the
    kernel takes (``MAX_SLICES``): forward and backward in float64 at 1e-11 relative."""
    assert s <= geo_kernel.MAX_SLICES
    leaves, wts = _region_case(1, 64, s, 2, 1, D, seed=s)
    _assert_region_exact(
        _run_region(True, leaves, wts, 2, 1), _run_region(False, leaves, wts, 2, 1)
    )


@pytest.mark.cuda
@requires_cuda_triton
@pytest.mark.parametrize("block", [1, 2, 4, 8])
def test_fused_geo_region_matches_eager_at_every_tile_size(block, monkeypatch):
    """``FWD_BLOCK_N`` and ``BWD_BLOCK_N`` set (the launch overrides): every tile size
    gives the eager region's outputs and gradients in float64 at 1e-11 relative, on 100
    points (a partial last tile at each size but 1 and 4)."""
    monkeypatch.setattr(geo_kernel, "FWD_BLOCK_N", block)
    monkeypatch.setattr(geo_kernel, "BWD_BLOCK_N", block)
    leaves, wts = _region_case(2, 101, 40, 1, 1, D, seed=block)
    with ran(geo_kernel._fwd_block, geo_kernel._bwd_block) as r:
        fused = _run_region(True, leaves, wts, 1, 1)
    assert r.returns(geo_kernel._fwd_block) == [block] and r.returns(
        geo_kernel._bwd_block
    ) == [block]
    _assert_region_exact(fused, _run_region(False, leaves, wts, 1, 1))


# --------------------------------------------------------------------------- segment softmax
def _ssr_case(h, kc, c, n_q=37, n_b=50, seed=3):
    """A random ragged pair list (segments longer than a source chunk, sources read by
    none, one source read by half the pairs) and float64 leaves."""
    g = torch.Generator().manual_seed(seed)
    counts = torch.randint(0, 5000, (n_q,), generator=g)
    counts[5] = 0
    p = int(counts.sum())
    qi = torch.repeat_interleave(torch.arange(n_q), counts)
    bj = torch.randint(0, n_b - 5, (p,), generator=g)
    bj[: p // 2] = 0
    leaves = [
        torch.randn(p, h, generator=g, dtype=D),
        torch.randn(n_q, h, kc, generator=g, dtype=D),
        torch.randn(n_b, h, kc, generator=g, dtype=D),
        torch.randn(n_b, h, c, generator=g, dtype=D),
    ]
    logm = -torch.rand(p, generator=g, dtype=D) * 8
    seed_ = torch.randn(n_q, h, c, generator=g, dtype=D)
    return (qi, bj, n_q, n_b), leaves, logm, seed_, counts > 0


def _ssr_run(read, plan_args, leaves, logm, seed, device, dtype):
    """``read``'s output and the gradients of ker, q, k, v for the loss
    ``(out * seed).sum()``."""
    plan = PairPlan.build(
        plan_args[0].to(device), plan_args[1].to(device), *plan_args[2:]
    )
    ins = [t.detach().to(device, dtype, copy=True).requires_grad_(True) for t in leaves]
    lm = logm.to(device, D if dtype == D else torch.float32)
    out = read(ins[0], lm, ins[1], ins[2], ins[3], plan, 0.5)
    (out.double() * seed.to(device)).sum().backward()
    return [out.detach()] + [t.grad for t in ins]


@pytest.mark.parametrize("h, kc, c", [(8, 4, 8), (4, 8, 16)])
def test_the_eager_segment_softmax_read_in_bfloat16_rounds_its_gradients_once(h, kc, c):
    """The eager read with bfloat16 ker, q, k and v (the reference read on every
    device) against the float64 read of the same inputs: its output and the gradients
    of ker, q, k and v are within one bfloat16 ulp of each float64 tensor's largest
    entry, 2**-7. The output stays float32; the gradients are computed in float32 and
    rounded once to bfloat16 (measured at most 3.0e-3). A cast after the gathers
    would sum each source's pair gradients in bfloat16 (an error of 0.58 for dv
    here)."""
    plan_args, leaves, logm, seed, _ = _ssr_case(h, kc, c, n_q=20)
    leaves = [t.bfloat16().double() for t in leaves]
    eager = segment_softmax._eager_segment_softmax_read
    low = _ssr_run(eager, plan_args, leaves, logm, seed, "cpu", torch.bfloat16)
    exact = _ssr_run(eager, plan_args, leaves, logm, seed, "cpu", D)
    for name, a, e in zip(("out", "ker", "q", "k", "v"), low, exact):
        err = float((a.double() - e).abs().max() / e.abs().max())
        assert err <= 2**-7, (name, err)


@pytest.mark.cuda
@requires_cuda_triton
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_the_fused_read_differentiates_the_log_measure(dtype):
    """``fused_kernel_read`` is differentiable in ``logm`` (per source, as the
    interior task's per-condition log shares need). Its gradient, each source's
    pairs' logit gradients summed over the heads, obeys the 2x-error rule against the
    reference read's (``segment_softmax_read`` of ``logm[plan.bj]``), with the float64
    eager read on CPU as the exact one. Asking for it leaves the output and every
    other gradient bitwise."""
    case = _read_case(5)

    def run(kind, device, dt, logm_grad=True):
        plan = PairPlan.build(
            case.qi.to(device), case.bj.to(device), case.n_q, case.n_b
        )
        b, w, q, k, v = ins = [
            t.detach().to(device, dt, copy=True).requires_grad_(True)
            for t in case.leaves
        ]
        lm = case.logm.to(device, D if dt == D else torch.float32, copy=True)
        lm.requires_grad_(logm_grad)
        if kind == "fused":
            assert fused_read.supported(b, q, k, v)
            out = fused_read.fused_kernel_read(b, w, lm, q, k, v, plan, 0.5)
        else:
            read = (
                segment_softmax.segment_softmax_read
                if kind == "reference"
                else segment_softmax._eager_segment_softmax_read
            )
            out = read(b @ w.t(), lm[plan.bj.long()], q, k, v, plan, 0.5)
        (out.double() * case.out_grad.to(device)).sum().backward()
        return [out.detach()] + [t.grad for t in ins] + [lm.grad]

    exact = run("eager", "cpu", D)
    fused, ref = run("fused", "cuda", dtype), run("reference", "cuda", dtype)
    plain = run("fused", "cuda", dtype, logm_grad=False)
    assert fused[-1] is not None and fused[-1].shape == (case.n_b,)
    assert plain[-1] is None
    _two_x("logm", fused[-1], ref[-1], exact[-1], _LOW[dtype])
    for i, name in enumerate(("out", "basis", "w", "q", "k", "v")):
        assert torch.equal(fused[i], plain[i]), name


# --------------------------------------------------------------------------- float16 narrow paths
@pytest.mark.cuda
@requires_cuda_triton
@pytest.mark.parametrize("dim, n_vecs", [(3, 1), (2, 0)])
def test_fused_kernel_basis_under_float16_autocast_obeys_the_2x_error_rule(
    dim, n_vecs, monkeypatch
):
    """The fused basis under float16 autocast (``unsupported_reason`` accepts it with
    the weight cache on) against the reference's chunked basis under the same
    autocast: the basis and every MLP gradient each as close to the CPU float64
    basis as the reference's (floor 1e-3)."""
    monkeypatch.setattr(tree_read, "TREE_READ_PAIR_CHUNK", 4096)
    geometry_, plan, basis = _basis_case(dim, n_vecs)
    seed = 1e-3 * torch.randn(20000, 16, generator=torch.Generator().manual_seed(1))
    seed = seed.half().double()
    exact = _basis_step(basis, plan, geometry_, seed)
    dev = torch.device("cuda")
    geo_c, plan_c = _to(geometry_, dev, torch.float32), _to(plan, dev, torch.float32)
    basis_c = basis.float().to(dev)

    def step(weights):
        basis_c.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            assert fused_basis.unsupported_reason(basis_c, geo_c) is None
            w = fused_basis.mlp_weights(basis_c) if weights else None
            out = tree_read._chunked_basis_from_plan(
                basis_c, plan_c, 0, int(plan_c.qi.shape[0]), 0, 0, geo_c, w
            )
        out.backward(seed.to(dev, out.dtype))
        return [out.detach()] + [p.grad.clone() for p in basis_c.parameters()]

    with ran(fused_basis._FusedBasis) as r:
        fused = step(True)
    assert r.count(fused_basis._FusedBasis) > 0
    ref = step(False)
    assert fused[0].dtype == ref[0].dtype == torch.float16
    names = ["basis"] + [n for n, _ in basis.named_parameters()]
    for name, a, b, e in zip(names, fused, ref, exact):
        _two_x(name, a, b, e, 1e-3)


# --------------------------------------------------------------------------- never-run paths
def test_the_fast_walk_of_no_points():
    """The fast walk of no points gives empty lists, and zero counts with
    ``count_only``."""
    pts, nrm, w, _ = cloud(300, d=3, seed=1)
    f = BoundaryForest.build(pts, w, nrm, None, theta=1.0)
    q = f.frame_points(cloud(40, d=3, seed=2, scale=(3.0,) * 3)[0])
    qi, si, n_node = f.frame_pairs(q[:0], walk="fast")
    assert qi.shape == si.shape == (0,) and n_node == 0
    assert torch.equal(
        f.frame_pairs(q[:0], walk="fast", count_only=True),
        torch.zeros(0, dtype=torch.long),
    )


def _basis_geometry(device, dim=3):
    geometry_, _, basis = _basis_case(dim, 1)
    return _to(geometry_, device, torch.float32), basis.float().to(device)


def test_fused_basis_unsupported_reasons_off_cuda(monkeypatch):
    """``unsupported_reason`` on CPU: CPU float32 geometry is refused, and without
    Triton that is the reason given first."""
    geometry_, basis = _basis_geometry("cpu")
    monkeypatch.setattr(fused_basis, "HAS_TRITON", True)
    assert "not CUDA float32" in fused_basis.unsupported_reason(basis, geometry_)
    monkeypatch.setattr(fused_basis, "HAS_TRITON", False)
    assert "Triton" in fused_basis.unsupported_reason(basis, geometry_)


@pytest.mark.cuda
@requires_cuda_triton
def test_fused_basis_unsupported_reasons_on_cuda():
    """``unsupported_reason`` on CUDA float32: None for the supported case; a
    float64 parameter, an autocast without its weight cache or in float32, a 4-D
    geometry and a hidden width that is not a power of two are each refused with
    their reason."""
    geometry_, basis = _basis_geometry("cuda")
    assert fused_basis.unsupported_reason(basis, geometry_) is None
    with torch.autocast("cuda", dtype=torch.bfloat16):
        assert fused_basis.unsupported_reason(basis, geometry_) is None
    with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
        assert "autocast" in fused_basis.unsupported_reason(basis, geometry_)
    with torch.autocast("cuda", dtype=torch.float32):
        assert "autocast" in fused_basis.unsupported_reason(basis, geometry_)
    ### (Module.double() and .float() convert the basis in place: back to float32 here)
    assert "not CUDA float32" in fused_basis.unsupported_reason(
        basis.double(), geometry_
    )
    basis.float()
    four = {
        k: (
            torch.cat([v, v[:, :1]], 1)
            if k in ("query_r", "query_n", "src_r", "src_n", "g_unit")
            else v
        )
        for k, v in geometry_.items()
    }
    four["src_vecs"] = torch.cat(
        [geometry_["src_vecs"], geometry_["src_vecs"][..., :1]], -1
    )
    assert "two- or three-dimensional" in fused_basis.unsupported_reason(basis, four)
    basis.net[0] = torch.nn.Linear(basis.net[0].in_features, 24).cuda()
    basis.net[2] = torch.nn.Linear(24, basis.net[2].out_features).cuda()
    assert "power of two" in fused_basis.unsupported_reason(basis, geometry_)


_SURFACE = dict(output_schema={"p": {"rank": 0}}, hidden=16, n_layers=1)


def _geo_model(**kw):
    kw = {
        "boundary_schemas": {"wall": {}},
        "global_schema": {"g": {"rank": 1}},
        "n_slices": 8,
        **kw,
    }
    return ISLA(**_SURFACE, **kw)


@pytest.mark.cuda
@requires_cuda_triton
def test_geo_kernel_auto_disqualifiers_on_cuda():
    """On CUDA with Triton, ``_geo_kernel="auto"`` picks the fused region for a model
    it fits and the eager region for each disqualifier: five per-point vectors (five
    global directions, or two and three rank-1 boundary inputs), more slices than
    ``MAX_SLICES``, a rank-2 boundary input, more than three dimensions (the kernel
    takes at most three components), float16 autocast (the fused region's only
    autocast dtype is bfloat16; the choice is re-made when the autocast dtype changes)."""
    pts = torch.zeros(4, 3, device="cuda")
    m = _geo_model()
    assert m._resolve_geo_kernel(pts) == "fused"
    for dtype, want in (
        (torch.float16, "eager"),
        (torch.bfloat16, "fused"),
        (torch.float16, "eager"),
    ):
        with torch.autocast("cuda", dtype=dtype):
            assert m._resolve_geo_kernel(pts) == want, dtype
    assert m._resolve_geo_kernel(pts) == "fused"
    five = {f"g{i}": {"rank": 1} for i in range(5)}
    cases = {
        "five global directions": _geo_model(global_schema=five),
        "two directions, three boundary vectors": _geo_model(
            global_schema={"a": {"rank": 1}, "b": {"rank": 1}},
            boundary_schemas={"wall": {f"t{i}": {"rank": 1} for i in range(3)}},
        ),
        "slices past MAX_SLICES": _geo_model(n_slices=geo_kernel.MAX_SLICES + 1),
        "rank-2 boundary input": _geo_model(
            boundary_schemas={"wall": {"tau": {"rank": 2, "symmetric": True}}}
        ),
    }
    assert cases["five global directions"]._n_global_vectors == 5
    m = cases["two directions, three boundary vectors"]
    assert m._n_global_vectors + m._n_block_vecs == 5
    m = cases["rank-2 boundary input"]
    ### (symmetric: its two derived vectors keep K + P at 3, so the rank-2 input alone
    ### disqualifies; a general one adds four and passes MAX_DIRECTIONS too)
    assert m._n_block_tens == 1
    assert m._n_global_vectors + m._n_block_vecs <= geo_kernel.MAX_DIRECTIONS
    for name, m in cases.items():
        assert m._geo_kernel == "auto" and m._resolve_geo_kernel(pts) == "eager", name
    m = _geo_model(n_spatial_dims=4)
    assert m._resolve_geo_kernel(torch.zeros(4, 4, device="cuda")) == "eager"


@pytest.mark.cuda
@requires_cuda_triton
def test_a_float64_model_under_bfloat16_autocast_runs_the_eager_region(caplog):
    """Autocast leaves float64 parameters as they are, while the fused region under
    bfloat16 autocast returns bfloat16: a float64 model there runs the eager region,
    with a warning that names the reason, and its forward pass runs as it does with
    ``reference_kernels``. Without autocast it keeps the fused region."""
    from physicsnemo.mesh import Mesh
    from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral

    s = sphere_icosahedral.load(subdivisions=2)
    mesh = {"wall": Mesh(points=s.points.double().cuda(), cells=s.cells.cuda())}
    m = ISLA(
        output_schema={"p": {"rank": 0}},
        boundary_schemas={"wall": {}},
        hidden=32,
        n_layers=2,
        n_slices=8,
    )
    m = m.double().cuda().eval()
    pts = mesh["wall"].points
    assert m._resolve_geo_kernel(pts) == "fused"
    outs = {}
    with caplog.at_level(
        logging.WARNING, logger="physicsnemo.experimental.models.isla.model"
    ):
        for reference in (True, False):
            m.reference_kernels = reference
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                outs[reference] = m(
                    boundary_meshes=mesh, prediction_boundaries=["wall"]
                ).point_data["p"]
                assert reference or m._resolve_geo_kernel(pts) == "eager"
    assert torch.equal(outs[True], outs[False])
    assert [r.getMessage() for r in caplog.records] == [
        "ISLA: the fused geometry kernel does not take this forward pass (float64 parameters under "
        "autocast); running the eager region."
    ]
