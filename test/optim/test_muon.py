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
import pickle
import threading

import pytest
import torch

from physicsnemo.optim import Muon

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


def test_lazy_adjust_lr_proxy_resolves():
    """The private torch._adjust_lr is reachable via the lazy OptionalImport.

    physicsnemo.optim.muon imports torch.optim._muon lazily so a future
    rename/removal fails at step() runtime rather than at module import time.
    This asserts the proxy resolves the symbol on the installed torch.
    """
    from physicsnemo.optim.muon import _torch_muon_internal

    assert callable(_torch_muon_internal._adjust_lr)


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
    """The graph replays the eager step bit for bit, also across an lr change
    and a state_dict round trip (both of which recapture)."""
    eager_params = _make_params(_GRAPH_SHAPES, "cuda", seed=4)
    graph_params = [torch.nn.Parameter(p.detach().clone()) for p in eager_params]
    kwargs = dict(lr=0.02, weight_decay=0.01, adjust_lr_fn="match_rms_adamw")
    eager_opt = Muon(eager_params, **kwargs)
    graph_opt = Muon(graph_params, cuda_graph=True, **kwargs)

    for step in range(8):
        if step == 4:
            for opt in (eager_opt, graph_opt):
                opt.param_groups[0]["lr"] *= 0.1
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
    """A new lr, set of parameters with gradients, or loaded state recaptures;
    new gradients alone replay the same graph."""
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
def test_cuda_graph_capture_with_loader_thread():
    """CUDA work in another thread, as a data loader does, does not invalidate
    the capture; every step recaptures (a new lr) and matches eager bit for bit."""
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
        before, graphs = len(iterations), []
        for step in range(6):
            for opt in (eager_opt, graph_opt):
                opt.param_groups[0]["lr"] = 0.02 / (1 + step)
            _set_grads(eager_params, seed=500 + step)
            _set_grads(graph_params, seed=500 + step)
            eager_opt.step()
            graph_opt.step()
            graphs.append(graph_opt._graph)
        during = len(iterations) - before
    finally:
        stop.set()
        thread.join()
    assert not errors
    assert during > 0
    assert graphs[0] is None and None not in graphs[1:]
    assert len({id(g) for g in graphs[1:]}) == 5
    for pe, pg in zip(eager_params, graph_params):
        assert torch.equal(pe, pg)


@_requires_cuda
def test_cuda_graph_capture_failure(monkeypatch):
    """A host sync inside the step fails the capture with a RuntimeError and
    leaves the user's gradients in place."""
    import physicsnemo.optim.muon as muon_module

    params = _make_params(_GRAPH_SHAPES, "cuda", seed=6)
    opt = Muon(params, lr=0.02, cuda_graph=True)
    _set_grads(params, seed=0)
    opt.step()

    newton_schulz = muon_module._batched_newton_schulz

    def syncing_newton_schulz(updates, *args):
        updates.sum().item()
        return newton_schulz(updates, *args)

    monkeypatch.setattr(muon_module, "_batched_newton_schulz", syncing_newton_schulz)
    _set_grads(params, seed=1)
    grads = [p.grad for p in params]
    with pytest.raises(RuntimeError, match="cuda_graph=False"):
        opt.step()
    assert all(p.grad is g for p, g in zip(params, grads))
    assert opt._graph is None


@_requires_cuda
def test_cuda_graph_rejects_tensor_lr():
    """A tensor lr would be baked into the graph, so capture refuses it."""
    params = _make_params(_GRAPH_SHAPES, "cuda", seed=7)
    opt = Muon(params, lr=torch.tensor(0.02), cuda_graph=True)
    _set_grads(params, seed=0)
    opt.step()  # eager: creates the momentum buffers
    with pytest.raises(ValueError, match="float learning rates"):
        opt.step()


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
