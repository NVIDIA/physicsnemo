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

"""merge_lora tests (CPU-only)."""

import torch
import torch.nn as nn

from physicsnemo.experimental.peft import (
    LoRAConfig,
    apply_lora,
    is_lora_layer,
    merge_lora,
)


class _Net(nn.Module):
    def __init__(self, d=16):
        super().__init__()
        self.fc1 = nn.Linear(d, d)
        self.fc2 = nn.Linear(d, d)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def _trained_net():
    torch.manual_seed(0)
    m = _Net()
    apply_lora(m, LoRAConfig(rank=4, alpha=8, target_pattern="fc"))
    # Move B off zero so the adapter actually changes the output.
    for mod in m.modules():
        if is_lora_layer(mod):
            with torch.no_grad():
                mod.lora_B.copy_(torch.randn_like(mod.lora_B))
    return m


def test_merge_preserves_forward():
    m = _trained_net()
    x = torch.randn(3, 16)
    before = m(x).detach().clone()
    out = merge_lora(m)
    assert out is m
    # wrappers gone, plain Linears restored
    assert not any(is_lora_layer(mod) for mod in m.modules())
    assert isinstance(m.fc1, nn.Linear) and isinstance(m.fc2, nn.Linear)
    assert torch.allclose(before, m(x), atol=1e-4)


def test_merge_idempotent():
    m = _trained_net()
    x = torch.randn(3, 16)
    merge_lora(m)
    once = m(x).detach().clone()
    merge_lora(m)  # no-op, must not raise or change anything
    assert torch.allclose(once, m(x), atol=1e-6)
    assert not any(is_lora_layer(mod) for mod in m.modules())


def test_merge_skips_non_mergeable(caplog):
    # A non-mergeable wrapper (mergeable=False, like the te.LayerNormMLP residual)
    # is left in place and warned about, while mergeable wrappers are folded.
    import logging

    from physicsnemo.experimental.peft import register_lora_wrapper
    from physicsnemo.experimental.peft.lora import _LORA_WRAPPERS, LoRALinear

    class _ResidualLinear(nn.Linear):
        pass

    class _NonMergeableLoRA(LoRALinear):
        mergeable = False

    class _Mixed(nn.Module):
        def __init__(self):
            super().__init__()
            self.res = _ResidualLinear(8, 8)
            self.fc = nn.Linear(8, 8)

    register_lora_wrapper(_ResidualLinear, _NonMergeableLoRA)
    try:
        m = _Mixed()
        apply_lora(m, LoRAConfig(rank=2, alpha=2, target_pattern=r"res|fc"))
        with caplog.at_level(logging.WARNING, logger="experimental.peft"):
            merge_lora(m)
        assert is_lora_layer(m.res)  # non-mergeable left wrapped
        assert not is_lora_layer(m.fc)  # mergeable folded + unwrapped
        assert "non-mergeable" in caplog.text
    finally:
        _LORA_WRAPPERS.pop(_ResidualLinear, None)  # don't leak into other tests
