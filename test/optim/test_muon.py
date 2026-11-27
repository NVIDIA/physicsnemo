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

import copy
import io
import math
import pickle
import threading
import warnings

import pytest
import torch

from physicsnemo.optim import CombinedOptimizer, Muon

# torch.optim.Muon is the per-parameter reference implementation we batch.
_TORCH_MUON = getattr(torch.optim, "Muon", None)
_HAS_TORCH_MUON = _TORCH_MUON is not None


def _make_params(shapes, device, seed=0):
    """Create a list of 2-D parameters with reproducible random values."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    return [
        torch.nn.Parameter(torch.randn(*s, generator=gen).to(device)) for s in shapes
    ]


def _make_grads(shapes, device, seed):
    """Create a list of gradients aligned with ``shapes``."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    return [torch.randn(*s, generator=gen).to(device) for s in shapes]


@pytest.mark.skipif(not _HAS_TORCH_MUON, reason="torch.optim.Muon unavailable")
@pytest.mark.parametrize("nesterov", [True, False])
@pytest.mark.parametrize("adjust_lr_fn", ["original", "match_rms_adamw"])
def test_matches_torch_muon(device, nesterov, adjust_lr_fn):
    """Fused Muon matches torch.optim.Muon step-for-step within tolerance.

    Shapes include square, wide, tall, and a repeated shape so the batched
    Newton-Schulz path (group size > 1) is exercised.
    """
    shapes = [(8, 8), (8, 8), (8, 16), (16, 8)]

    ref_params = _make_params(shapes, device, seed=1)
    fused_params = [torch.nn.Parameter(p.detach().clone()) for p in ref_params]

    kwargs = dict(
        lr=0.02,
        weight_decay=0.01,
        momentum=0.95,
        nesterov=nesterov,
        adjust_lr_fn=adjust_lr_fn,
    )
    ref_opt = _TORCH_MUON(ref_params, **kwargs)
    fused_opt = Muon(fused_params, **kwargs)

    for step in range(5):
        grads = _make_grads(shapes, device, seed=100 + step)
        for p, g in zip(ref_params, grads):
            p.grad = g.clone()
        for p, g in zip(fused_params, grads):
            p.grad = g.clone()
        ref_opt.step()
        fused_opt.step()

    for ref_p, fused_p in zip(ref_params, fused_params):
        torch.testing.assert_close(fused_p, ref_p, atol=1e-3, rtol=1e-3)


def test_adjust_lr_matches_torch():
    """The local learning-rate adjustment (the formulas of torch.optim.Muon's
    documentation) equals torch's private helper, where torch has it, bit for bit."""
    from physicsnemo.optim.muon import _adjust_lr

    private = getattr(pytest.importorskip("torch.optim._muon"), "_adjust_lr", None)
    if private is None:
        pytest.skip("this torch has no torch.optim._muon._adjust_lr")
    for fn in (None, "original", "match_rms_adamw", "unknown"):
        for shape in ((8, 8), (8, 24), (96, 16), (3, 7, 5)):
            for lr in (0.02, 1e-3, 0.1):
                assert _adjust_lr(lr, fn, torch.Size(shape)) == private(
                    lr, fn, torch.Size(shape)
                )


def test_group_params_by_shape(device):
    """Equally-shaped params bucket together; distinct shapes stay separate."""
    shapes = [(8, 8), (8, 8), (8, 16), (16, 8), (8, 8)]
    params = _make_params(shapes, device, seed=2)

    groups = Muon._group_params_by_shape(params)

    sizes = sorted(len(idxs) for idxs in groups.values())
    # (8,8) x3, (8,16) x1, (16,8) x1
    assert sizes == [1, 1, 3]
    # The repeated (8,8) shape collapses to one group with the right indices.
    eight = [idxs for key, idxs in groups.items() if key[0] == (8, 8)]
    assert eight == [[0, 1, 4]]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_update_matches_transposed_layout(device, dtype, monkeypatch):
    """The fused update equals adding each parameter's update in the layout
    Newton-Schulz returns it in (transposed for a tall group) bit for bit, for
    square, wide and tall groups; float32 updates reach the add contiguous."""
    shapes = [(16, 16), (16, 16), (8, 32), (8, 32), (32, 8), (32, 8)]
    fused = [
        torch.nn.Parameter(p.detach().to(dtype))
        for p in _make_params(shapes, device, seed=3)
    ]
    looped = [torch.nn.Parameter(p.detach().clone()) for p in fused]
    kwargs = dict(lr=0.02, weight_decay=0.1, adjust_lr_fn="match_rms_adamw")
    fused_opt, looped_opt = Muon(fused, **kwargs), Muon(looped, **kwargs)
    foreach_add = torch._foreach_add_
    contiguous = []

    def recording_add(tensors, others, alpha=1):
        contiguous.extend(o.is_contiguous() for o in others)
        foreach_add(tensors, others, alpha=alpha)

    def add_transposed(tensors, others, alpha=1):
        for t, o in zip(tensors, others):
            if o.shape[0] > o.shape[1]:  # tall: as Newton-Schulz returns it
                o = o.mT.contiguous().mT
            t.add_(o, alpha=alpha)

    for step in range(3):
        grads = _make_grads(shapes, device, seed=200 + step)
        for p, q, g in zip(fused, looped, grads):
            p.grad, q.grad = g.to(dtype), g.to(dtype)
        monkeypatch.setattr(torch, "_foreach_add_", recording_add)
        fused_opt.step()
        monkeypatch.setattr(torch, "_foreach_add_", add_transposed)
        looped_opt.step()
        monkeypatch.setattr(torch, "_foreach_add_", foreach_add)
        for p, q in zip(fused, looped):
            assert torch.equal(p, q)
    # bf16 updates need no cast, so a tall group's stay transposed.
    assert all(contiguous) == (dtype == torch.float32)


def test_state_dict_roundtrip(device):
    """Saving and restoring state (incl. momentum buffers) resumes identically."""
    shapes = [(8, 8), (8, 16)]
    params = _make_params(shapes, device, seed=3)
    opt = Muon(params, lr=0.02, weight_decay=0.01, adjust_lr_fn="match_rms_adamw")

    # Two steps to populate momentum buffers.
    for step in range(2):
        grads = _make_grads(shapes, device, seed=200 + step)
        for p, g in zip(params, grads):
            p.grad = g.clone()
        opt.step()

    saved_state = copy.deepcopy(opt.state_dict())
    snapshot = [p.detach().clone() for p in params]

    # Continue one more step on the original optimizer (the reference).
    final_grads = _make_grads(shapes, device, seed=999)
    for p, g in zip(params, final_grads):
        p.grad = g.clone()
    opt.step()
    reference_final = [p.detach().clone() for p in params]

    # Fresh optimizer restored from the snapshot + saved state must match.
    restored_params = [torch.nn.Parameter(s.clone()) for s in snapshot]
    restored_opt = Muon(
        restored_params, lr=0.02, weight_decay=0.01, adjust_lr_fn="match_rms_adamw"
    )
    restored_opt.load_state_dict(saved_state)
    for p, g in zip(restored_params, final_grads):
        p.grad = g.clone()
    restored_opt.step()

    for ref_p, restored_p in zip(reference_final, restored_params):
        torch.testing.assert_close(restored_p, ref_p, atol=1e-6, rtol=1e-6)


def test_rejects_non_2d_params(device):
    """Muon only supports 2-D parameters."""
    param_1d = torch.nn.Parameter(torch.randn(8).to(device))
    with pytest.raises(ValueError, match="2D"):
        Muon([param_1d], lr=0.02)


_requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA graphs need CUDA"
)

# Repeated, wide and tall shapes, plus a larger group for realistic cuBLAS
# GEMMs: four Newton-Schulz shape groups.
_GRAPH_SHAPES = [
    (8, 8),
    (8, 8),
    (8, 8),
    (8, 32),
    (32, 8),
    (256, 1024),
    (256, 1024),
    (1024, 256),
]


def _set_grads(params, seed):
    """Give each parameter a fresh gradient, as a backward after set_to_none does."""
    grads = _make_grads([p.shape for p in params], params[0].device, seed)
    for p, g in zip(params, grads):
        p.grad = g


def _reload(opt, state_dict):
    """Load *state_dict* through torch.save/torch.load, so its tensors are new."""
    buffer = io.BytesIO()
    torch.save(state_dict, buffer)
    buffer.seek(0)
    opt.load_state_dict(torch.load(buffer))


@_requires_cuda
def test_cuda_graph_matches_eager_bitwise():
    """The graph replays the eager step bit for bit, also across an lr and
    weight decay change (which replays the same graph) and a state_dict round
    trip (which recaptures)."""
    eager_params = _make_params(_GRAPH_SHAPES, "cuda", seed=4)
    graph_params = [torch.nn.Parameter(p.detach().clone()) for p in eager_params]
    kwargs = dict(lr=0.02, weight_decay=0.01, adjust_lr_fn="match_rms_adamw")
    eager_opt = Muon(eager_params, **kwargs)
    graph_opt = Muon(graph_params, cuda_graph=True, **kwargs)

    for step in range(8):
        if step == 4:
            for opt in (eager_opt, graph_opt):
                opt.param_groups[0]["lr"] *= 0.1
                opt.param_groups[0]["weight_decay"] *= 3
        if step == 6:
            state_dict = eager_opt.state_dict()
            for opt in (eager_opt, graph_opt):
                _reload(opt, state_dict)
        _set_grads(eager_params, seed=300 + step)
        _set_grads(graph_params, seed=300 + step)
        eager_opt.step()
        graph_opt.step()
        assert graph_opt.cuda_graph
        assert (graph_opt._graph is None) == (step == 0)
        for pe, pg in zip(eager_params, graph_params):
            assert torch.equal(pe, pg)
            assert torch.equal(
                eager_opt.state[pe]["momentum_buffer"],
                graph_opt.state[pg]["momentum_buffer"],
            )


@_requires_cuda
def test_cuda_graph_recaptures_on_key_change():
    """A new set of parameters with gradients, loaded state, momentum or group
    dict recaptures; new gradients, lr, weight decay or adjust_lr_fn alone
    replay the same graph."""
    params = _make_params(_GRAPH_SHAPES, "cuda", seed=5)
    opt = Muon(params, lr=0.02, cuda_graph=True)

    def step(seed, with_grad=params):
        opt.zero_grad(set_to_none=True)
        _set_grads(with_grad, seed)
        opt.step()
        return opt._graph

    assert step(0) is None  # eager: creates the momentum buffers
    graph = step(1)
    assert graph is not None
    assert step(2) is graph
    opt.param_groups[0]["lr"] = 0.01
    opt.param_groups[0]["weight_decay"] = 0.2
    opt.param_groups[0]["adjust_lr_fn"] = "match_rms_adamw"
    assert step(3) is graph
    opt.param_groups[0] = {**opt.param_groups[0], "lr": 0.005}  # a new dict
    assert step(3) is not graph
    graph = opt._graph
    opt.param_groups[0]["momentum"] = 0.9
    assert step(3) is not graph
    graph = opt._graph
    assert step(4, with_grad=params[1:]) is not graph
    graph = opt._graph
    _reload(opt, opt.state_dict())
    assert opt.cuda_graph
    assert step(5, with_grad=params[1:]) is not graph


@_requires_cuda
def test_cuda_graph_under_torch_compile():
    """Under torch.compile the graph step runs outside the compiled region, so
    the graph is still captured and replayed, matching eager bit for bit."""
    eager_params = _make_params(_GRAPH_SHAPES[:5], "cuda", seed=9)
    graph_params = [torch.nn.Parameter(p.detach().clone()) for p in eager_params]
    eager_opt = Muon(eager_params, lr=0.02)
    graph_opt = Muon(graph_params, lr=0.02, cuda_graph=True)
    compiled_step = torch.compile(graph_opt.step)
    for step in range(3):
        _set_grads(eager_params, seed=400 + step)
        _set_grads(graph_params, seed=400 + step)
        eager_opt.step()
        compiled_step()
    assert graph_opt._graph is not None
    for pe, pg in zip(eager_params, graph_params):
        assert torch.equal(pe, pg)


def _loader_cuda_work(stop, ready, iterations, errors):
    """Until *stop* is set, do the CUDA work of a data loader thread on a
    stream of its own: pinned and device allocations of varying sizes,
    host-to-device copies and blocking device-to-host copies. Sets *ready*
    after the first round."""
    try:
        with torch.cuda.stream(torch.cuda.Stream()):
            while not stop.is_set():
                n = 2**14 * (1 + len(iterations) % 7)
                device_copy = torch.empty(n, pin_memory=True).to(
                    "cuda", non_blocking=True
                )
                torch.empty(n * (1 + len(iterations) % 5), device="cuda")
                device_copy[:1].cpu()
                iterations.append(None)
                ready.set()
    except Exception as e:
        errors.append(e)


@_requires_cuda
def test_cuda_graph_stream_outside_default_pool():
    """Muon captures on a stream of the high-priority pool, so the default
    pool, from which loaders draw their streams, never hands it out."""
    params = _make_params(_GRAPH_SHAPES[:5], "cuda", seed=12)
    opt = Muon(params, lr=0.02, cuda_graph=True)
    for step in range(2):  # the second step captures the graph
        _set_grads(params, seed=step)
        opt.step()
    assert opt._graph is not None
    handles = {torch.cuda.Stream().cuda_stream for _ in range(64)}
    assert opt._graph_stream.cuda_stream not in handles


@_requires_cuda
def test_cuda_graph_capture_with_loader_thread():
    """CUDA work in another thread, as a data loader does, neither invalidates
    the capture nor lands in it; every step recaptures (a new momentum) and
    matches eager bit for bit. The 40 captures outnumber torch's 32 pooled streams, so
    a new stream per capture would reach the loader's stream."""
    eager_params = _make_params(_GRAPH_SHAPES, "cuda", seed=10)
    graph_params = [torch.nn.Parameter(p.detach().clone()) for p in eager_params]
    eager_opt = Muon(eager_params, lr=0.02)
    graph_opt = Muon(graph_params, lr=0.02, cuda_graph=True)
    stop, ready, iterations, errors = threading.Event(), threading.Event(), [], []
    thread = threading.Thread(
        target=_loader_cuda_work, args=(stop, ready, iterations, errors)
    )
    thread.start()
    try:
        assert ready.wait(timeout=60), errors
        before, previous_graph = len(iterations), None
        for step in range(41):
            for opt in (eager_opt, graph_opt):
                opt.param_groups[0]["momentum"] = 0.95 - 0.005 * step
            _set_grads(eager_params, seed=500 + step)
            _set_grads(graph_params, seed=500 + step)
            eager_opt.step()
            graph_opt.step()
            # Step 0 runs eagerly; every later step captures a new graph.
            assert (graph_opt._graph is None) == (step == 0)
            assert step == 0 or graph_opt._graph is not previous_graph
            previous_graph = graph_opt._graph
        during = len(iterations) - before
    finally:
        stop.set()
        thread.join()
    assert not errors
    assert during > 0
    for pe, pg in zip(eager_params, graph_params):
        assert torch.equal(pe, pg)


def _fail_once(monkeypatch, name, after, error=RuntimeError):
    """Make ``torch.cuda.CUDAGraph.<name>`` raise *error* once, after running
    or before it."""
    original = getattr(torch.cuda.CUDAGraph, name)
    calls = []

    def failing(self, *args, **kwargs):
        if calls:
            return original(self, *args, **kwargs)
        calls.append(None)
        if after:
            original(self, *args, **kwargs)
        raise error(f"{name} failed")

    monkeypatch.setattr(torch.cuda.CUDAGraph, name, failing)


def _assert_no_capture_open(opt, stream, params, grads):
    """No capture is left open, and the user's current stream and gradients
    are in place."""
    assert torch.cuda.current_stream() == stream
    with torch.cuda.stream(opt._graph_stream):
        assert not torch.cuda.is_current_stream_capturing()
    torch.cuda.synchronize()
    assert all(p.grad is g for p, g in zip(params, grads))
    assert opt._graph is None


@_requires_cuda
@pytest.mark.parametrize("where", ["capture_begin", "step", "capture_end"])
def test_cuda_graph_capture_failure(monkeypatch, where):
    """A capture that fails at its start (as capture_begin does under
    torch.cuda.set_sync_debug_mode("error")), in the step (a host sync, which
    cannot be captured) or at its end warns once, leaves no capture open, and
    leaves the user's gradients and current stream in place. That step and
    every later one run eagerly, equal to an eager optimizer's bit for bit."""
    import physicsnemo.optim.muon as muon_module

    params = _make_params(_GRAPH_SHAPES, "cuda", seed=6)
    eager_params = [p.detach().clone().requires_grad_() for p in params]
    opt = Muon(params, lr=0.02, cuda_graph=True)
    eager = Muon(eager_params, lr=0.02)
    for ps, o in ((params, opt), (eager_params, eager)):
        _set_grads(ps, seed=0)
        o.step()

    if where == "step":
        newton_schulz = muon_module._batched_newton_schulz

        def syncing_newton_schulz(updates, *args):
            updates.sum().item()
            return newton_schulz(updates, *args)

        monkeypatch.setattr(
            muon_module, "_batched_newton_schulz", syncing_newton_schulz
        )
    else:
        _fail_once(monkeypatch, where, after=where == "capture_begin")
    stream = torch.cuda.current_stream()
    for seed in (1, 2, 3):
        _set_grads(params, seed=seed)
        _set_grads(eager_params, seed=seed)
        grads = [p.grad for p in params]
        if seed == 1:
            with pytest.warns(RuntimeWarning, match="steps eagerly"):
                opt.step()
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                opt.step()
        eager.step()
        _assert_no_capture_open(opt, stream, params, grads)
        assert not opt.cuda_graph
        assert all(torch.equal(p, q) for p, q in zip(params, eager_params))


@_requires_cuda
def test_cuda_graph_capture_out_of_memory(monkeypatch):
    """Running out of memory in a capture is not a capture failure: it raises,
    leaves no capture open, and keeps the graph on; the next step captures."""
    params = _make_params(_GRAPH_SHAPES, "cuda", seed=6)
    opt = Muon(params, lr=0.02, cuda_graph=True)
    _set_grads(params, seed=0)
    opt.step()

    _fail_once(monkeypatch, "capture_end", after=False, error=torch.OutOfMemoryError)
    _set_grads(params, seed=1)
    grads = [p.grad for p in params]
    stream = torch.cuda.current_stream()
    with pytest.raises(torch.OutOfMemoryError):
        opt.step()
    _assert_no_capture_open(opt, stream, params, grads)
    assert opt.cuda_graph
    monkeypatch.undo()
    opt.step()
    assert opt._graph is not None


@_requires_cuda
def test_cuda_graph_step_inside_user_capture():
    """Inside a capture of the caller's own, the plain step runs, so the
    caller's graph captures it; its replay matches the eager step bit for bit."""
    eager_params = _make_params(_GRAPH_SHAPES[:5], "cuda", seed=11)
    graph_params = [torch.nn.Parameter(p.detach().clone()) for p in eager_params]
    eager_opt = Muon(eager_params, lr=0.02)
    graph_opt = Muon(graph_params, lr=0.02, cuda_graph=True)
    for step in range(3):  # the second step captures Muon's own graph
        _set_grads(eager_params, seed=600 + step)
        _set_grads(graph_params, seed=600 + step)
        eager_opt.step()
        if step < 2:
            graph_opt.step()
            continue
        user_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(user_graph):
            graph_opt.step()
        user_graph.replay()
    for pe, pg in zip(eager_params, graph_params):
        assert torch.equal(pe, pg)


@_requires_cuda
def test_cuda_graph_tensor_lr():
    """A tensor lr is read after the replay, as the eager step reads it, so
    changing it in place replays the same graph and matches eager bit for bit."""
    eager_params = _make_params(_GRAPH_SHAPES, "cuda", seed=7)
    graph_params = [torch.nn.Parameter(p.detach().clone()) for p in eager_params]
    eager_lr, graph_lr = torch.tensor(0.02), torch.tensor(0.02)
    eager_opt = Muon(eager_params, lr=eager_lr)
    graph_opt = Muon(graph_params, lr=graph_lr, cuda_graph=True)
    graphs = []
    for step in range(5):
        for lr in (eager_lr, graph_lr):
            lr.fill_(0.02 / (1 + step))
        _set_grads(eager_params, seed=600 + step)
        _set_grads(graph_params, seed=600 + step)
        eager_opt.step()
        graph_opt.step()
        graphs.append(graph_opt._graph)
        for pe, pg in zip(eager_params, graph_params):
            assert torch.equal(pe, pg)
    assert graphs[0] is None and all(g is graphs[1] for g in graphs[1:])


def _warmup_cosine(warmup, total):
    """The recipe's kind of schedule: linear warmup, then cosine decay."""

    def factor(epoch):
        if epoch < warmup:
            return (epoch + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (epoch - warmup) / (total - warmup)))

    return factor


@_requires_cuda
@pytest.mark.parametrize("period", [1, 3])
@pytest.mark.parametrize("adjust_lr_fn", ["original", "match_rms_adamw"])
def test_cuda_graph_follows_a_schedule_bitwise(monkeypatch, period, adjust_lr_fn):
    """Under a warmup-plus-cosine schedule stepped every *period* steps, with an
    AdamW fallback for the 1-D parameters (as the recipe combines them), the
    graph is captured once and every parameter and momentum buffer equals the
    eager step's bit for bit after every step."""
    captures = []
    graph = torch.cuda.graph

    def counting_graph(*args, **kwargs):
        captures.append(None)
        return graph(*args, **kwargs)

    monkeypatch.setattr(torch.cuda, "graph", counting_graph)
    shapes_1d = [(8,), (32,), (1024,)]
    factor = _warmup_cosine(4, 40)
    runs = []
    for cuda_graph in (False, True):
        params = _make_params(_GRAPH_SHAPES, "cuda", seed=11)
        gen = torch.Generator(device="cpu").manual_seed(12)
        vectors = [
            torch.nn.Parameter(torch.randn(*s, generator=gen).cuda()) for s in shapes_1d
        ]
        opt = CombinedOptimizer(
            [
                Muon(
                    params,
                    lr=1.125e-3,
                    weight_decay=1e-4,
                    adjust_lr_fn=adjust_lr_fn,
                    cuda_graph=cuda_graph,
                ),
                torch.optim.AdamW(vectors, lr=1.125e-3, weight_decay=1e-4),
            ]
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, factor)
        runs.append((params, vectors, opt, scheduler))

    lrs = set()
    for step in range(30):
        lrs.add(runs[1][2].param_groups[0]["lr"])
        for params, vectors, opt, scheduler in runs:
            _set_grads(params + vectors, seed=700 + step)
            opt.step()
            if (step + 1) % period == 0:
                scheduler.step()
        (eager_params, eager_vectors, eager_opt, _), (params, vectors, opt, _) = runs
        for pe, pg in zip(eager_params + eager_vectors, params + vectors):
            assert torch.equal(pe, pg)
        eager_muon, muon = eager_opt.optimizers[0], opt.optimizers[0]
        for pe, pg in zip(eager_params, params):
            assert torch.equal(
                eager_muon.state[pe]["momentum_buffer"],
                muon.state[pg]["momentum_buffer"],
            )
    assert lrs == {1.125e-3 * factor(epoch) for epoch in range(30 // period)}
    assert len(captures) == 1


@pytest.mark.parametrize(
    "cuda_graph", [False, pytest.param(True, marks=_requires_cuda)]
)
def test_copy_and_pickle(cuda_graph):
    """deepcopy and pickle round trips keep cuda_graph and step as the original
    does; a copy of a graphed optimizer captures a graph of its own."""
    device = "cuda" if cuda_graph else "cpu"
    params = _make_params(_GRAPH_SHAPES[:5], device, seed=8)
    opt = Muon(params, lr=0.02, cuda_graph=cuda_graph)
    for step in range(2):  # the second step captures the graph
        _set_grads(params, seed=step)
        opt.step()
    copies = [copy.deepcopy(opt), pickle.loads(pickle.dumps(opt))]  # noqa: S301
    for o in [opt, *copies]:
        _set_grads(o.param_groups[0]["params"], seed=2)
        o.step()
    for copied in copies:
        assert copied.cuda_graph is cuda_graph
        assert (copied._graph is not None) == cuda_graph
        assert copied._graph is None or copied._graph is not opt._graph
        for p, pc in zip(params, copied.param_groups[0]["params"]):
            assert torch.equal(p, pc)


def test_cuda_graph_refuses_cpu_params():
    """cuda_graph=True refuses parameters that are not on CUDA."""
    with pytest.raises(ValueError, match="CUDA"):
        Muon(_make_params([(8, 8)], "cpu"), cuda_graph=True)
