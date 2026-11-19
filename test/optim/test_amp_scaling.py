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


"""Tests for torch.amp.GradScaler's sync-free step protocol in Muon and CombinedOptimizer."""

import copy

import pytest
import torch

from physicsnemo.optim import CombinedOptimizer, Muon

_SHAPES = [(8, 8), (8, 8), (8, 16), (16, 8)]


def _make_params(device, seed=0):
    """Create 2-D parameters with reproducible random values."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    return [
        torch.nn.Parameter(torch.randn(*s, generator=gen).to(device)) for s in _SHAPES
    ]


def _set_grads(params, seed, scale=1.0):
    """Assign reproducible random gradients, optionally multiplied by ``scale``."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    for p in params:
        p.grad = (torch.randn(p.shape, generator=gen) * scale).to(p.device)


def _muon(params):
    return Muon(params, lr=0.02, weight_decay=0.01, momentum=0.95)


def _step_with(opt, grad_scale=None, found_inf=None):
    """Mimic GradScaler.step: set the attributes, step, delete them."""
    opt.grad_scale = grad_scale
    opt.found_inf = found_inf
    try:
        opt.step()
    finally:
        del opt.grad_scale
        del opt.found_inf


def _bufs(opt, params):
    return [opt.state[p]["momentum_buffer"].clone() for p in params]


def test_muon_advertises_amp_scaling():
    assert Muon._step_supports_amp_scaling is True


def test_found_inf_skips_update(device):
    """found_inf=1 leaves parameters and momentum state bitwise unchanged."""
    params = _make_params(device)
    opt = _muon(params)
    _set_grads(params, seed=1)
    opt.step()  # populate momentum state
    before_params = [p.detach().clone() for p in params]
    before_bufs = _bufs(opt, params)

    _set_grads(params, seed=2)
    params[0].grad[0, 0] = float("inf")
    _step_with(opt, found_inf=torch.ones((), device=device))

    for p, ref in zip(params, before_params):
        torch.testing.assert_close(p.detach(), ref, rtol=0, atol=0)
    for b, ref in zip(_bufs(opt, params), before_bufs):
        torch.testing.assert_close(b, ref, rtol=0, atol=0)


def test_found_inf_skips_first_step(device):
    """A skipped first step leaves zero momentum buffers and unchanged params."""
    params = _make_params(device)
    before = [p.detach().clone() for p in params]
    opt = _muon(params)
    _set_grads(params, seed=1)
    _step_with(opt, found_inf=torch.ones((), device=device))
    for p, ref in zip(params, before):
        torch.testing.assert_close(p.detach(), ref, rtol=0, atol=0)
    for b in _bufs(opt, params):
        assert torch.count_nonzero(b) == 0


def test_found_inf_zero_equals_plain_step(device):
    """found_inf=0 gives exactly the plain step."""
    ref_params = _make_params(device)
    params = [torch.nn.Parameter(p.detach().clone()) for p in ref_params]
    ref_opt, opt = _muon(ref_params), _muon(params)
    for seed in range(1, 4):
        _set_grads(ref_params, seed)
        _set_grads(params, seed)
        ref_opt.step()
        _step_with(opt, found_inf=torch.zeros((), device=device))
    for p, ref in zip(params, ref_params):
        torch.testing.assert_close(p, ref, rtol=0, atol=0)
    for b, ref in zip(_bufs(opt, params), _bufs(ref_opt, ref_params)):
        torch.testing.assert_close(b, ref, rtol=0, atol=0)


def test_grad_scale_unscales(device):
    """Scaled grads + grad_scale give the same step as unscaled grads."""
    scale = 1024.0  # power of two: unscaling is exact
    ref_params = _make_params(device)
    params = [torch.nn.Parameter(p.detach().clone()) for p in ref_params]
    ref_opt, opt = _muon(ref_params), _muon(params)
    for seed in range(1, 4):
        _set_grads(ref_params, seed)
        _set_grads(params, seed, scale=scale)
        ref_opt.step()
        _step_with(
            opt,
            grad_scale=torch.tensor(scale, device=device),
            found_inf=torch.zeros((), device=device),
        )
        for p, ref in zip(params, ref_params):
            torch.testing.assert_close(p.grad, ref.grad, rtol=0, atol=0)
    for p, ref in zip(params, ref_params):
        torch.testing.assert_close(p, ref, rtol=0, atol=0)


def test_no_attrs_unchanged(device):
    """Without the attributes, step matches a deep-copied optimizer's step."""
    params = _make_params(device)
    opt = _muon(params)
    _set_grads(params, seed=1)
    ref_params = [torch.nn.Parameter(p.detach().clone()) for p in params]
    ref_opt = _muon(ref_params)
    _set_grads(ref_params, seed=1)
    opt.step()
    ref_opt.step()
    assert not hasattr(opt, "grad_scale") and not hasattr(opt, "found_inf")
    for p, ref in zip(params, ref_params):
        torch.testing.assert_close(p, ref, rtol=0, atol=0)


class _RecordingSGD(torch.optim.SGD):
    """Optimizer stub that records the amp attributes seen during step()."""

    _step_supports_amp_scaling = True

    def step(self, closure=None):
        self.seen = (
            getattr(self, "grad_scale", "unset"),
            getattr(self, "found_inf", "unset"),
        )
        return None


def test_combined_advertises_only_if_all_support(device):
    p1, p2 = _make_params(device)[:2]
    assert CombinedOptimizer(
        [_muon([p1]), _RecordingSGD([p2], lr=0.1)]
    )._step_supports_amp_scaling
    assert not CombinedOptimizer(
        [_muon([p1]), torch.optim.SGD([p2], lr=0.1)]
    )._step_supports_amp_scaling


def test_combined_forwards_attrs(device):
    p1, p2 = _make_params(device)[:2]
    subs = [_RecordingSGD([p1], lr=0.1), _RecordingSGD([p2], lr=0.1)]
    opt = CombinedOptimizer(subs)
    _set_grads([p1, p2], seed=1)
    gs, fi = torch.tensor(2.0, device=device), torch.zeros((), device=device)
    _step_with(opt, grad_scale=gs, found_inf=fi)
    for sub in subs:
        assert sub.seen[0] is gs and sub.seen[1] is fi
        assert not hasattr(sub, "grad_scale") and not hasattr(sub, "found_inf")
    # Without attributes nothing is forwarded.
    opt.step()
    for sub in subs:
        assert sub.seen == ("unset", "unset")


def test_combined_found_inf_skips_muon(device):
    params = _make_params(device)
    before = [p.detach().clone() for p in params]
    opt = CombinedOptimizer([_muon(params[:2]), _muon(params[2:])])
    _set_grads(params, seed=1)
    _step_with(opt, found_inf=torch.ones((), device=device))
    for p, ref in zip(params, before):
        torch.testing.assert_close(p.detach(), ref, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("combined", [False, True])
def test_grad_scaler_fp16_no_sync(combined):
    """GradScaler + fp16 autocast: steps without host sync and skips an inf step."""
    device = "cuda"
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(16, 32), torch.nn.GELU(), torch.nn.Linear(32, 4)
    ).to(device)
    weights = [p for p in model.parameters() if p.ndim == 2]
    others = [p for p in model.parameters() if p.ndim != 2]
    if combined:
        opt = CombinedOptimizer(
            [_muon(weights), torch.optim.Adam(others, lr=1e-3, fused=True)]
        )
    else:
        for p in others:
            p.requires_grad_(False)
        opt = _muon(weights)
    assert opt._step_supports_amp_scaling
    scaler = torch.amp.GradScaler("cuda", init_scale=2.0**10)
    x = torch.randn(8, 16, device=device)

    def run_step(inject_inf):
        opt.zero_grad()
        with torch.autocast("cuda", dtype=torch.float16):
            loss = model(x).float().pow(2).mean()
        scaler.scale(loss).backward()
        if inject_inf:
            weights[0].grad[0, 0] = float("inf")
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("error")
        try:
            scaler.step(opt)
        finally:
            torch.cuda.set_sync_debug_mode("default")
        scaler.update()

    run_step(False)
    state = copy.deepcopy(opt.state_dict())
    before = [p.detach().clone() for p in model.parameters()]
    scale = scaler.get_scale()
    run_step(True)
    for p, ref in zip(model.parameters(), before):
        torch.testing.assert_close(p.detach(), ref, rtol=0, atol=0)
    assert scaler.get_scale() == scale * 0.5
    after = opt.state_dict()
    if not combined:
        for k, v in state["state"].items():
            torch.testing.assert_close(
                after["state"][k]["momentum_buffer"],
                v["momentum_buffer"],
                rtol=0,
                atol=0,
            )
    run_step(False)
    assert any(not torch.equal(p, ref) for p, ref in zip(model.parameters(), before))
