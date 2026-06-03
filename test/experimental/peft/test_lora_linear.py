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

"""LoRALinear (and TE wrapper) unit tests (CPU-only; TE test skips cleanly)."""

import pytest
import torch
import torch.nn as nn

from physicsnemo.experimental.peft.lora import LoRALinear
from test.conftest import requires_module  # noqa: E402


def test_forward_equals_base_at_init():
    torch.manual_seed(0)
    base = nn.Linear(16, 32)
    x = torch.randn(4, 16)
    lora = LoRALinear(base, rank=4, alpha=4)
    # lora_B is zero at init → the delta is exactly zero.
    assert torch.count_nonzero(lora.lora_B) == 0
    assert torch.allclose(lora(x), base(x), atol=1e-6)


def test_gradient_flows_to_lora_only():
    torch.manual_seed(0)
    base = nn.Linear(16, 8)
    lora = LoRALinear(base, rank=4, alpha=8)
    # Perturb B so the delta (and gradients) are nonzero.
    with torch.no_grad():
        lora.lora_B.add_(torch.randn_like(lora.lora_B) * 0.01)
    lora(torch.randn(4, 16)).pow(2).sum().backward()
    assert lora.lora_A.grad is not None and lora.lora_A.grad.abs().sum() > 0
    assert lora.lora_B.grad is not None and lora.lora_B.grad.abs().sum() > 0
    assert all(not p.requires_grad for p in lora.base_layer.parameters())
    assert lora.base_layer.weight.grad is None


def test_device_dtype_inherited_from_base():
    base = nn.Linear(8, 8).to(torch.float64)
    lora = LoRALinear(base, rank=2, alpha=2)
    assert lora.lora_A.dtype == torch.float64
    assert lora.lora_B.dtype == torch.float64
    assert lora.lora_A.device == base.weight.device


def test_merge_matches_explicit_transpose():
    # Guards the (A @ B).T transpose: the merged weight must equal
    # W + scaling*(A @ B).T, and the base forward must then equal the pre-merge
    # LoRA forward.
    torch.manual_seed(0)
    base = nn.Linear(16, 32)
    lora = LoRALinear(base, rank=4, alpha=8)
    with torch.no_grad():
        lora.lora_B.copy_(torch.randn_like(lora.lora_B))
    x = torch.randn(4, 16)
    before = lora(x).detach().clone()
    expected = base.weight.detach() + lora.scaling * (lora.lora_A @ lora.lora_B).t()
    lora.merge_into_base()
    assert torch.allclose(base.weight, expected, atol=1e-5)
    assert torch.allclose(base(x), before, atol=1e-4)


def test_disable_recovers_base():
    torch.manual_seed(0)
    base = nn.Linear(8, 8)
    lora = LoRALinear(base, rank=2, alpha=2)
    with torch.no_grad():
        lora.lora_B.copy_(torch.randn_like(lora.lora_B))
    x = torch.randn(2, 8)
    assert not torch.allclose(lora(x), base(x), atol=1e-6)  # delta is active
    lora.enabled = False
    assert torch.allclose(lora(x), base(x), atol=1e-6)


@requires_module("transformer_engine")
def test_te_linear_wraps_and_init_parity():
    if not torch.cuda.is_available():
        pytest.skip("TE requires CUDA.")
    import transformer_engine.pytorch as te

    from physicsnemo.experimental.peft.lora import LoRA_te_Linear

    base = te.Linear(16, 32).cuda()
    lora = LoRA_te_Linear(base, rank=4, alpha=4).cuda()
    x = torch.randn(4, 16, device="cuda")
    assert torch.count_nonzero(lora.lora_B) == 0
    assert torch.allclose(lora(x), base(x), atol=1e-4)


@requires_module("transformer_engine")
def test_te_layernorm_mlp_residual():
    if not torch.cuda.is_available():
        pytest.skip("TE requires CUDA.")
    import transformer_engine.pytorch as te

    from physicsnemo.experimental.peft.lora import (
        LoRA_te_LayerNormMLP,
        wrappable_types,
    )

    assert te.LayerNormMLP in wrappable_types()  # registered wrapper type

    base = te.LayerNormMLP(hidden_size=64, ffn_hidden_size=256).cuda()
    lora = LoRA_te_LayerNormMLP(base, rank=4, alpha=4).cuda()
    x = torch.randn(2, 16, 64, device="cuda")

    # B=0 at init → residual is zero → output equals the fused base output.
    assert torch.count_nonzero(lora.lora_B) == 0
    with torch.no_grad():
        b = base(x)
        b = b[0] if isinstance(b, tuple) else b
        y = lora(x)
        y = y[0] if isinstance(y, tuple) else y
    assert torch.allclose(b, y, atol=1e-4)

    # Grad flows to the residual; the fused base params stay frozen.
    with torch.no_grad():
        lora.lora_B.copy_(torch.randn_like(lora.lora_B))
    out = lora(x)
    (out[0] if isinstance(out, tuple) else out).pow(2).sum().backward()
    assert lora.lora_A.grad is not None and lora.lora_A.grad.abs().sum() > 0
    assert not base.fc1_weight.requires_grad
    assert lora.mergeable is False
