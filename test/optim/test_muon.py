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
