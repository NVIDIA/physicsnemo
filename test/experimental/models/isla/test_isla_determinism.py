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

"""Run-to-run determinism of ISLA's eager paths, as they are.

No test sets a global deterministic mode (``torch.use_deterministic_algorithms``):
that would hide what a user meets by default. Each test runs a path twice (or five
times) in one process on the same inputs and states what repeats:

- bitwise: the fast exact geometry (d, u, both gradients), the fused geometry
  region, the fused node means and the fused read with their backward passes, and
  the whole model (the surface task and the interior task with its draw, in 3-D and
  2-D, in float32 and under bfloat16 autocast, forward and backward) on CPU with one
  thread and on CUDA with kernels auto;
- to a stated tolerance, where the code sums in an order CUDA does not fix: the
  reference (unfused) tree read on CUDA (``node_means``' and the segment-softmax
  read's ``index_add``).

The draw's line is exact int64 arithmetic, so its CUDA runs are bitwise
(test_isla_sampling.py); its planted ties are here.
"""

import pytest
import torch

### The modules that own each function run here.
from physicsnemo.experimental.models.isla import ISLA
from physicsnemo.experimental.models.isla import _fused_read as fused_read
from physicsnemo.experimental.models.isla import _geo_kernel as geo_kernel
from physicsnemo.experimental.models.isla import _sampling as sampling
from physicsnemo.experimental.models.isla import _segment_softmax as segment_softmax
from physicsnemo.experimental.models.isla import _tree_build as tree_build
from physicsnemo.experimental.models.isla import _tree_read as tree_read
from physicsnemo.experimental.models.isla._geometry import exact_boundary_geometry
from physicsnemo.experimental.models.isla._segment_softmax import PairPlan
from physicsnemo.experimental.models.isla._tree_read import (
    BoundaryForest,
    _KernelBasis,
    _TreeKernelRead,
    kernel_geo_width,
)
from physicsnemo.mesh.sampling.systematic_sampling import _cumsum, _systematic_sample
from physicsnemo.mesh.spatial._point_walk_warp import point_walk
from test.experimental.models.isla._forced import ran
from test.experimental.models.isla._geometry_cases import (
    _near_wall,
    _sphere,
)
from test.experimental.models.isla._marks import requires_cuda, requires_cuda_triton
from test.experimental.models.isla._meshes import (
    cloud,
    live,
    outputs,
    points_mesh,
    soup,
)

D = torch.float64


def _bytes(t):
    return t.detach().contiguous().reshape(-1).view(torch.uint8)


def _bitwise(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return (
        a.dtype == b.dtype and a.shape == b.shape and torch.equal(_bytes(a), _bytes(b))
    )


def _rel_norm(a, b) -> float:
    a, b = a.double(), b.double()
    return float((a - b).norm() / b.norm().clamp_min(1e-300))


def _rel_max(a, b) -> float:
    a, b = a.double(), b.double()
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-300))


# --------------------------------------------------------------------------- the draw
def _draw_case(n=300_000, seed=0):
    """About 300k float64 cells over 4 conditions (one smaller than its share),
    measures spread over three decades, anisotropic centroids."""
    sizes = (150_000, 90_000, 57_000, 3_000)
    g = torch.Generator().manual_seed(seed)
    measures = [(torch.rand(m, generator=g, dtype=D) * 3).exp() * 1e-3 for m in sizes]
    scale = torch.tensor([4.0, 2.0, 1.0], dtype=D)
    centroids = [torch.randn(m, 3, generator=g, dtype=D) * scale for m in sizes]
    return [m.cuda() for m in measures], [x.cuda() for x in centroids]


BUDGET = 100_000


def _planted_hits(measures, centroids):
    """Starts that plant a hit exactly on a cell boundary of the CPU draw's line, in
    its first condition (cells of pi < 1). The line is int64 in units of 2**-Q; for
    the boundary ``line[i]`` (the end of the line's ``i``-th cell),
    ``U = line[i] mod 2**Q`` and the start ``U / 2**Q`` (exact in float64, so the
    draw's ``floor(start 2**Q)`` is ``U``) put the hit ``j = line[i] >> Q`` exactly
    on it, when ``U`` and ``U - 1`` lie inside the start's clamp by the line's
    residual ``D = line[-1] - k 2**Q``. Returns ``Q`` and ``(start, earlier, later)``
    per boundary: the start and the cells at the line's positions ``i`` and
    ``i + 1``, each shorter than one unit, so each holds at most one hit."""
    with ran(_cumsum, _systematic_sample) as r:
        sampling.sample_boundary_cells(
            measures, centroids, BUDGET, torch.tensor(0.5, dtype=D)
        )
    line = r.returns(_cumsum)[-1]  # the draw's last scan is the line
    order = r.calls(_systematic_sample)[-1]["order"]
    assert line.dtype == torch.int64
    assert line.shape == order.shape == (sum(m.shape[0] for m in measures),)
    Q = 62 - BUDGET.bit_length()
    unit = 1 << Q
    residual = int(line[-1]) - BUDGET * unit
    planted = []
    for i in torch.linspace(1000, measures[0].shape[0] - 1000, 8).long().tolist():
        end = int(line[i])
        U, j = end % unit, end >> Q
        if not max(0, residual) + 1 <= U <= min(unit, unit + residual) - 1:
            continue
        assert j < BUDGET  # one of the k hits
        assert 0 < end - int(line[i - 1]) < unit and 0 < int(line[i + 1]) - end < unit
        planted.append(
            (torch.tensor(U / unit, dtype=D), int(order[i]), int(order[i + 1]))
        )
    assert len(planted) >= 6, planted
    return Q, planted


@pytest.mark.cuda
@requires_cuda
def test_hits_planted_on_line_boundaries_keep_one_of_the_two_cells_there():
    """Hits planted exactly on cell boundaries of the systematic line
    (:func:`_planted_hits`): the hit takes the later of the two cells beside the
    boundary (half-open intervals), and one unit of the line's grid earlier the
    earlier one; and every CUDA run keeps the CPU's cells, with bitwise the same
    weights, since the line is exact int64 arithmetic on every device."""
    measures, centroids = _draw_case(seed=1)
    cpu_m, cpu_x = [m.cpu() for m in measures], [x.cpu() for x in centroids]
    Q, planted = _planted_hits(cpu_m, cpu_x)
    for start, earlier, later in planted:
        kept_cpu, w_cpu = sampling.sample_boundary_cells(cpu_m, cpu_x, BUDGET, start)
        kept = set(kept_cpu.tolist())
        assert later in kept and earlier not in kept, (start, earlier, later)
        below = sampling.sample_boundary_cells(cpu_m, cpu_x, BUDGET, start - 2.0**-Q)
        below = set(below[0].tolist())
        assert earlier in below and later not in below, (start, earlier, later)
        for _ in range(3):
            kept, w = sampling.sample_boundary_cells(
                measures, centroids, BUDGET, start.cuda()
            )
            assert torch.equal(kept.cpu(), kept_cpu)
            assert torch.equal(w.cpu().view(torch.int64), w_cpu.view(torch.int64))


# --------------------------------------------------------------------------- module level
@pytest.mark.cuda
@requires_cuda_triton
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_the_fast_geometry_repeats_on_cuda(dtype):
    """``exact_boundary_geometry(kernel="fast")`` twice on CUDA: d, u and the query
    and vertex gradients bitwise (the gather's backward is CUDA's sort-based
    ``index_put_``, which sums in a fixed order)."""
    mesh = _sphere(3)
    q0 = _near_wall(3, 20000, 2, band=0.5).to("cuda", dtype)
    p0, cells = mesh.points.to("cuda", dtype), mesh.cells.cuda()
    g = torch.Generator().manual_seed(0)
    probe = torch.randn(q0.shape, generator=g, dtype=D).to("cuda", dtype)

    def run():
        q, p = q0.clone().requires_grad_(), p0.clone().requires_grad_()
        d, u = exact_boundary_geometry(q, p, cells, kernel="fast")
        gq, gp = torch.autograd.grad(d.sum() + (u * probe).sum(), (q, p))
        return d, u, gq, gp

    a, b = run(), run()
    for name, x, y in zip(("d", "u", "dq", "dp"), a, b):
        if not _bitwise(x, y):
            print(f"{name}: run-to-run {_rel_max(x, y):.2e}")
        assert _bitwise(x, y), name


@pytest.mark.cuda
@requires_cuda_triton
@pytest.mark.parametrize("autocast", [False, True], ids=["fp32", "bf16_autocast"])
def test_the_fused_geo_region_repeats_on_cuda(autocast):
    """The fused geometry region twice on CUDA, forward and every input gradient:
    bitwise (its per-slice reductions are per-tile partial sums reduced by torch, no
    atomics)."""
    g = torch.Generator().manual_seed(0)
    b, n, s = 2, 3000, 64
    x0 = [
        torch.randn(b, n, s, generator=g),
        torch.randn(b, n, 3, generator=g) * 0.1,
        torch.nn.functional.normalize(torch.randn(b, n, 3, generator=g), dim=-1),
        torch.nn.functional.normalize(torch.randn(b, n, 2, 3, generator=g), dim=-1),
        torch.randn(b, s, 3, generator=g) * 0.1,
        torch.nn.functional.normalize(torch.randn(b, s, 3, generator=g), dim=-1),
    ]
    wts = [
        torch.randn(b, n, s, generator=g).cuda(),
        torch.randn(b, n, s, generator=g).cuda(),
        torch.randn(b, n, 7, generator=g).cuda(),
    ]
    torch.manual_seed(0)
    lin0 = torch.nn.Linear(7, 1).cuda()

    def run():
        x = [t.cuda().requires_grad_() for t in x0]
        lin = torch.nn.Linear(7, 1).cuda()
        lin.load_state_dict(lin0.state_dict())
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
            logits = x[0].bfloat16() if autocast else x[0]
            outs = geo_kernel.fused_geo_region(lin, logits, *x[1:], 1e-12)
        sum((o.float() * w).sum() for o, w in zip(outs, wts)).backward()
        return (
            [o.detach() for o in outs]
            + [t.grad for t in x]
            + [lin.weight.grad, lin.bias.grad]
        )

    for i, (x, y) in enumerate(zip(run(), run())):
        assert _bitwise(x, y), i


def _read_case(dtype, seed=0):
    """A forest over 4000 cells on CUDA, a random plan of the read's shape, and the
    leaves of one fused read with node means: the cells' keys and values ``(N, 96)``,
    the basis, the kernel weights and the points' queries."""
    g = torch.Generator().manual_seed(seed)
    pts, nrm, w, _ = cloud(4000, seed=seed)
    dev = torch.device("cuda")
    omega = (w / w.sum()).to(dev)
    forest = BoundaryForest.build(pts.to(dev), omega, nrm.to(dev), None, theta=1.0)
    n_src = 4000 + forest.n_nodes
    n_q, h, kc, c, f = 400, 8, 4, 8, 16
    counts = torch.randint(1, 300, (n_q,), generator=g)
    qi = torch.repeat_interleave(torch.arange(n_q), counts).to(dev)
    ### a hot source (cell 7, every third pair: split over many programs) and three
    ### sources no point reads (the last three nodes)
    bj = torch.randint(0, n_src - 3, (qi.shape[0],), generator=g)
    bj[::3] = 7
    bj = bj.to(dev)
    plan = PairPlan.build(qi, bj, n_q, n_src)
    leaves = [
        torch.randn(4000, h * (kc + c), generator=g),
        torch.randn(qi.shape[0], f, generator=g),
        0.3 * torch.randn(h, f, generator=g),
        torch.randn(n_q, h, kc, generator=g),
    ]
    logm = -8 * torch.rand(n_src, generator=g)
    seed_ = torch.randn(n_q, h, c, generator=g)
    return (
        forest,
        omega,
        plan,
        [t.to(dev, dtype) for t in leaves],
        logm.to(dev),
        seed_.to(dev),
        (h, kc, c),
    )


@pytest.mark.cuda
@requires_cuda_triton
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_the_fused_read_with_its_node_means_repeats_on_cuda(dtype):
    """The fused node means of the cells' keys and values, then the fused read over
    the cells and the nodes, twice on CUDA: the output and the gradients of the
    cells' keys and values, the sources' keys and values, the basis, the kernel
    weights and the queries, bitwise (fixed-order Triton sums, no atomics), with a
    source read by many points (its parts summed in order over several programs) and
    sources read by none (zero gradients)."""
    forest, omega, plan, leaves, logm, seed, (h, kc, c) = _read_case(dtype)
    assert int((plan.bj == 7).sum()) > 20 * fused_read.SOURCE_CHUNK

    def run():
        kv, basis, w, q = [t.clone().requires_grad_() for t in leaves]
        assert fused_read.node_means_supported(kv)
        src = torch.cat(
            [kv, fused_read.fused_node_means(kv, forest, omega.to(kv.dtype))]
        )
        src.retain_grad()
        k, v = src[:, : h * kc].reshape(-1, h, kc), src[:, h * kc :].reshape(-1, h, c)
        assert fused_read.supported(basis, q, k, v)
        out = fused_read.fused_kernel_read(basis, w, logm, q, k, v, plan, 0.5)
        (out * seed).sum().backward()
        return [out.detach(), kv.grad, src.grad, basis.grad, w.grad, q.grad]

    first, second = run(), run()
    for name, x, y in zip(("out", "kv", "src", "basis", "w", "q"), first, second):
        assert _bitwise(x, y), name
    assert not first[2][-3:].any()


@pytest.mark.cuda
@requires_cuda
def test_the_reference_tree_read_on_cuda_spreads_within_its_bound():
    """The reference (unfused) tree read on CUDA in float32 repeats within its bounds
    over five runs of one block's read, each tensor against the first run's, relative
    in norm: the output within 1e-5, the gradient of ``kernel.weight`` within 3.3e-4
    and every other parameter gradient within 3e-5. The node means sum by
    ``index_add``, whose atomic order CUDA does not fix.

    The gradient bounds are twice the largest spreads measured over 60 fresh processes
    (torch 2.13): 1.61e-4 at ``kernel.weight`` and 1.48e-5 at the others; the output's
    was 8.3e-7. ``kernel.weight`` spreads about ten times more because its gradient's
    terms nearly cancel: the softmax makes each point's logit gradients sum to zero
    over its sources, and the kernel basis changes little between them, so the terms'
    absolute sum is about 1,000 times the gradient's norm."""
    torch.manual_seed(0)
    dev = torch.device("cuda")
    pts, nrm, w, g = (t.float().to(dev) for t in cloud(3000, seed=3))
    q = cloud(500, seed=7, scale=(4.0,) * 3)[0].float().to(dev)
    kw = dict(
        query_r=q[None],
        query_n=torch.nn.functional.normalize(q, dim=-1)[None],
        query_wall=0.1 + torch.rand(1, 500, device=dev),
        boundary_r=pts[None],
        boundary_n=nrm[None],
        boundary_vecs=None,
        log_frac=torch.log(w / w.sum())[None],
        g_unit=torch.nn.functional.normalize(g, dim=-1)[None, None],
        theta=0.5,
        eps=1e-12,
    )
    basis = _KernelBasis(kernel_geo_width(1, 0)).to(dev)
    with torch.no_grad():
        pairs = tree_read.tree_read_pairs(basis, kernel="reference", **kw)
    assert pairs.kernel == "reference" and pairs.n_pairs_node > 0
    read = _TreeKernelRead(64).to(dev)
    with torch.no_grad():
        for p in read.parameters():
            p.normal_(0.0, 0.3)
    gen = torch.Generator().manual_seed(1)
    h_b = torch.randn(3000, 64, generator=gen).to(dev)
    h_q = torch.randn(500, 64, generator=gen).to(dev)
    seed = torch.randn(500, 64, generator=gen).to(dev)

    def run():
        read.zero_grad(set_to_none=True)
        with ran(tree_read.node_means, fused_read.fused_node_means) as r:
            out = read(h_b, h_q, pairs)
        assert (
            r.count(tree_read.node_means) == 1
            and r.count(fused_read.fused_node_means) == 0
        )
        (out * seed).sum().backward()
        return [out.detach()] + [p.grad.clone() for p in read.parameters()]

    runs = [run() for _ in range(5)]
    names = ["out"] + [n for n, _ in read.named_parameters()]
    spread = [
        max(_rel_norm(r[i], runs[0][i]) for r in runs[1:]) for i in range(len(names))
    ]
    print(
        "reference tree read, five-run spread in norm:",
        {n: f"{x:.1e}" for n, x in zip(names, spread)},
    )
    bound = {"out": 1e-5, "kernel.weight": 3.3e-4}
    over = {n: x for n, x in zip(names, spread) if x >= bound.get(n, 3e-5)}
    assert not over, over


# --------------------------------------------------------------------------- the whole model
#: The whole-model cases: the surface task and the interior task (its read live, its
#: draw keeping 300 of 400 cells), each in 3-D and in 2-D. On CUDA the interior task
#: also runs with every cell kept (``dtt_every_cell``, the default budget), where the
#: draw still runs its line over the inclusion probabilities.
TASKS = ["surface", "dtt"]
DIMS = [3, 2]
PRECISIONS = [
    pytest.param(False, id="fp32"),
    pytest.param(True, id="bf16_autocast"),
]


def _model(task, dims):
    torch.manual_seed(0)
    kw = dict(
        output_schema={"p": {"rank": 0}, "v": {"rank": 1}},
        boundary_schemas={"wall": {}},
        global_schema={"g": {"rank": 1}},
        n_spatial_dims=dims,
        hidden=32,
        n_layers=2,
        n_slices=16,
    )
    if task != "surface":
        kw.update(interior=True)
    if task == "dtt":
        kw.update(n_boundary_tokens=300)
    return live(ISLA(**kw))


def _step(model, task, device, autocast, n=400, nq=120):
    """One training step of ``model`` on a boundary soup of ``n`` cells (and, for
    the interior task, ``nq`` prediction points), the global RNG seeded first (the draw's
    start in training): the outputs, every parameter's gradient and the draw's kept
    cells and weights, by name."""
    dims = model.n_spatial_dims
    pts, nrm, w, drv = cloud(n, d=dims, seed=0, dtype=torch.float32)
    mesh = soup(pts, nrm, w).to(device)
    if task != "surface":
        q = cloud(nq, d=dims, seed=1, dtype=torch.float32, scale=(3.0,) * dims)[0]
        where = {"prediction_points": points_mesh(q.to(device))}
    else:
        where = {"prediction_boundaries": ["wall"]}
    model.zero_grad(set_to_none=True)
    torch.manual_seed(1)
    with (
        ran(sampling.sample_boundary_cells) as r,
        torch.autocast(device, dtype=torch.bfloat16, enabled=autocast),
    ):
        out = outputs(
            model(
                boundary_meshes={"wall": mesh},
                global_data={"g": drv.to(device)},
                **where,
            )
        )
        sum(o.double().square().mean() for o in out.values()).backward()
    draws = r.returns(sampling.sample_boundary_cells)
    assert len(draws) == (task != "surface")
    draw = {}
    if draws:
        draw = dict(zip(("draw.kept", "draw.weights"), draws[0]))
        assert draw["draw.kept"].shape == (300 if task == "dtt" else n,)
    return {
        **{k: v.detach() for k, v in out.items()},
        **{n: p.grad for n, p in model.named_parameters()},
        **draw,
    }


def _not_bitwise(a, b) -> list:
    assert a.keys() == b.keys()
    bad = [k for k in a if not _bitwise(a[k], b[k])]
    for k in bad:
        print(f"{k}: run-to-run {_rel_max(a[k], b[k]):.2e}")
    return bad


@pytest.mark.parametrize("autocast", PRECISIONS)
@pytest.mark.parametrize("dims", DIMS)
@pytest.mark.parametrize("task", TASKS)
def test_the_model_repeats_bitwise_on_cpu_with_one_thread(task, dims, autocast):
    """The surface and interior models (read live, draw active), in 3-D and 2-D, in
    float32 and under bfloat16 autocast: one training step twice on CPU with one
    thread, the outputs, every parameter gradient and the draw bitwise."""
    model = _model(task, dims)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        a, b = (_step(model, task, "cpu", autocast) for _ in range(2))
    finally:
        torch.set_num_threads(threads)
    assert not _not_bitwise(a, b)


def _resolved(model, task, device):
    """The kernels "auto" resolves to for ``model`` on ``device`` (float32 points):
    the geometry region's, and for the interior task the exact geometry's and the
    read's."""
    p = torch.zeros(1, model.n_spatial_dims, device=device)
    kernels = {"geo": model._resolve_geo_kernel(p)}
    if task != "surface":
        kernels["geometry"] = model._resolve_geometry_kernel(p)
        kernels["tree_read"] = model._resolve_tree_read_kernel(p)
    return kernels


@pytest.mark.cuda
@requires_cuda_triton
@pytest.mark.parametrize(
    "task, dims, autocast",
    [
        pytest.param(t, d, a.values[0], id=f"{t}-{d}-{a.id}")
        for t in TASKS
        for d in DIMS
        for a in PRECISIONS
    ]
    + [pytest.param("dtt_every_cell", 3, False, id="dtt_every_cell-3-fp32")],
)
def test_the_model_repeats_bitwise_on_cuda_with_kernels_auto(task, dims, autocast):
    """The surface and interior models (read live, draw active or keeping every cell),
    in 3-D and 2-D, in float32 and under bfloat16 autocast, on CUDA with every kernel
    "auto": one training step twice, the outputs, every parameter gradient and the
    draw (its float64 weights included) bitwise. The test also asserts which kernels run: the
    fused geometry region, and in the interior task the fast exact geometry, the
    fused read and node means (not the segment softmax read), the fast forest build
    and the fast walk."""
    model = _model(task, dims).cuda()
    reads = (
        fused_read.fused_kernel_read,
        fused_read.fused_node_means,
        segment_softmax.segment_softmax_read,
    )
    with ran(tree_build.forest_tables, point_walk, *reads) as r:
        a, b = (_step(model, task, "cuda", autocast) for _ in range(2))
    kernels = _resolved(model, task, "cuda")
    if task != "surface":
        tables, walks = (
            r.returns(tree_build.forest_tables),
            r.returns(point_walk),
        )
        kernels["build"] = "fast" if tables and None not in tables else "reference"
        kernels["walk"] = "fast" if walks and None not in walks else "reference"
        kernels["read_ran"] = {f.__name__ for f in reads if r.count(f)}
    print("resolved:", kernels)
    want = {"geo": "fused"}
    if task != "surface":
        want.update(geometry="fast", tree_read="fused", build="fast", walk="fast")
        want["read_ran"] = {"fused_kernel_read", "fused_node_means"}
    assert kernels == want
    assert not _not_bitwise(a, b)
