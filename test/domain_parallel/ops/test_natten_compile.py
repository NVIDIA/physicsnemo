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

r"""``torch.compile`` over sharded neighborhood attention (issue #1986).

Dynamo treats ``ShardTensor`` as a traceable wrapper subclass and folds
``has_torch_function`` to ``False`` for it, so a traced ``na2d`` would call the
raw kernel on the ShardTensor instead of the halo-exchanging handler. The raw
kernel here is a pure-torch reference that, like an opaque kernel, refuses
distributed inputs; it also lets the test run without NATTEN installed.
"""

import pytest
import torch
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import Shard

from physicsnemo.distributed import DistributedManager
from physicsnemo.domain_parallel import ShardTensor, scatter_tensor

from .utils import collective_assert_close, sharded_to_local


def _na2d_reference(q, k, v, kernel_size, dilation=1, **_):
    r"""NATTEN-semantics 2D neighborhood attention on plain ``(B, H, W, h, D)``
    tensors: each query attends to the ``kernel_size`` window clamped to the
    image, not zero-padded."""
    if isinstance(q, (ShardTensor, DTensor)):
        raise RuntimeError("raw neighborhood-attention kernel got a distributed tensor")
    assert dilation == 1
    r = kernel_size

    def window(n):
        start = torch.arange(n, device=q.device).clamp(r // 2, n - 1 - r // 2) - r // 2
        return start[:, None] + torch.arange(r, device=q.device)

    ih, iw = window(q.shape[1]), window(q.shape[2])
    kk = k[:, ih][:, :, :, iw]  # (B, H, r, W, r, h, D)
    vv = v[:, ih][:, :, :, iw]
    logits = torch.einsum("bxyhd,bxiyjhd->bxyhij", q, kk) * q.shape[-1] ** -0.5
    p = logits.flatten(-2).softmax(-1).view_as(logits)
    return torch.einsum("bxyhij,bxiyjhd->bxyhd", p, vv)


@pytest.fixture
def natten_reference(monkeypatch):
    r"""Route the raw NATTEN call (eager and sharded paths) to the reference."""
    from types import SimpleNamespace

    from physicsnemo.domain_parallel.shard_utils import natten_patches
    from physicsnemo.nn.functional import natten as natten_functional

    stub = SimpleNamespace(functional=SimpleNamespace(na2d=_na2d_reference))
    monkeypatch.setattr(natten_functional, "_natten", stub)
    monkeypatch.setattr(natten_patches, "_natten", stub)


@pytest.mark.multigpu_static
@pytest.mark.timeout(300)
@pytest.mark.parametrize("backward", [False, True])
def test_na2d_compiled_on_shard_tensor(distributed_mesh, natten_reference, backward):
    r"""Compiled ``na2d`` on height-sharded q/k/v runs the halo-exchange handler
    and matches the single-device result, forward and backward."""
    from physicsnemo.nn.functional.natten import na2d

    dm = DistributedManager()
    kernel_size = 5
    shape = (1, 8 * distributed_mesh.size(0), 12, 2, 8)  # (B, H, W, heads, D)
    torch.manual_seed(0)
    q, k, v = (torch.randn(shape, device=dm.device) for _ in range(3))
    sq, sk, sv = (
        scatter_tensor(t, 0, distributed_mesh, (Shard(1),), requires_grad=backward)
        for t in (q, k, v)
    )

    torch._dynamo.reset()
    compiled = torch.compile(
        lambda q, k, v: na2d(q, k, v, kernel_size=kernel_size),
        backend="aot_eager",
        dynamic=False,
        fullgraph=True,  # the handler must be a graph node, not a graph break
    )
    out = compiled(sq, sk, sv)

    if backward:
        q, k, v = (t.detach().requires_grad_(True) for t in (q, k, v))
    ref = _na2d_reference(q, k, v, kernel_size)

    assert isinstance(out, ShardTensor)
    assert out._spec.placements == (Shard(1),)
    collective_assert_close(ref, sharded_to_local(out), atol=1e-4, rtol=1e-4, msg="out")

    if backward:
        out.square().sum().backward()
        ref.square().sum().backward()
        for name, st, t in (("q", sq, q), ("k", sk, k), ("v", sv, v)):
            collective_assert_close(
                t.grad, sharded_to_local(st.grad), atol=1e-3, rtol=1e-3, msg=name
            )
