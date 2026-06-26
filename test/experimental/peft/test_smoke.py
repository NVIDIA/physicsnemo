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

"""CI smoke test: the full PEFT recipe via real training (CPU-only, fast).

apply_lora → split_params → AdamW train a few steps → assert the adapter
actually moved and loss dropped → save_adapter → load into a fresh base →
outputs match → merge_lora → output preserved. This is the end-to-end
round-trip that the example (transformer_models/src/finetune.py) performs,
shrunk to a toy model so it runs in CI without data or a GPU.
"""

import torch
import torch.nn as nn

from physicsnemo.experimental.peft import (
    LoRAConfig,
    apply_lora,
    is_lora_layer,
    load_adapter,
    merge_lora,
    save_adapter,
    split_params_for_optimizer,
)


class _Net(nn.Module):
    def __init__(self, d=32):
        super().__init__()
        self.fc1 = nn.Linear(d, d)
        self.fc2 = nn.Linear(d, d)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def test_peft_train_save_load_merge_roundtrip(tmp_path):
    torch.manual_seed(0)
    model = _Net()
    apply_lora(model, LoRAConfig(rank=4, alpha=8, target_pattern="fc"))

    # Optimizer over LoRA params only (the recipe's AdamW-not-Muon routing).
    groups = split_params_for_optimizer(model)
    optimizer = torch.optim.AdamW(groups["lora"] + groups["extras"], lr=1e-2)

    base_w0 = model.fc1.base_layer.weight.detach().clone()
    assert torch.count_nonzero(model.fc1.lora_B) == 0  # zero at init

    # Tiny synthetic regression task.
    torch.manual_seed(1)
    x = torch.randn(64, 32)
    y = torch.randn(64, 32)
    model.train()
    losses = []
    for _ in range(100):
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(model(x), y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    # Training actually trained: loss dropped, the adapter moved off zero, and
    # the frozen base weights are untouched.
    assert losses[-1] < losses[0]
    assert torch.count_nonzero(model.fc1.lora_B) > 0
    assert torch.allclose(model.fc1.base_layer.weight, base_w0)

    trained_out = model(x).detach().clone()

    # Save the adapter, reload into a fresh (identically-initialized) base.
    adapter = tmp_path / "adapter.lora"
    save_adapter(model, adapter)
    torch.manual_seed(0)
    fresh = _Net()
    load_adapter(fresh, adapter)
    assert torch.allclose(fresh(x), trained_out, atol=1e-5)

    # Merge → forward preserved, no LoRA layers remain (plain model again).
    merge_lora(fresh)
    assert not any(is_lora_layer(m) for m in fresh.modules())
    assert torch.allclose(fresh(x), trained_out, atol=1e-4)
