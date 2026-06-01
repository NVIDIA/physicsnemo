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

"""save_adapter / load_adapter tests (CPU-only)."""

import io
import json
import zipfile

import pytest
import torch
import torch.nn as nn

from physicsnemo.experimental.peft import (
    LoRAConfig,
    apply_lora,
    is_lora_layer,
    load_adapter,
    save_adapter,
)


class _Net(nn.Module):
    def __init__(self, d=16):
        super().__init__()
        self.fc1 = nn.Linear(d, d)
        self.fc2 = nn.Linear(d, d)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def _trained_net(d=16):
    torch.manual_seed(0)
    m = _Net(d)
    apply_lora(m, LoRAConfig(rank=4, alpha=8, target_pattern="fc"))
    for mod in m.modules():
        if is_lora_layer(mod):
            with torch.no_grad():
                mod.lora_B.copy_(torch.randn_like(mod.lora_B))
    return m


def test_save_load_roundtrip(tmp_path):
    m = _trained_net()
    x = torch.randn(3, 16)
    trained_out = m(x).detach().clone()
    p = tmp_path / "adapter.mdlus"
    save_adapter(m, p)

    # Fresh base with identical init (same seed) → load should reproduce output.
    torch.manual_seed(0)
    fresh = _Net()
    load_adapter(fresh, p)
    assert torch.allclose(fresh(x), trained_out, atol=1e-5)
    # the fresh model is now LoRA-wrapped on the same modules
    assert is_lora_layer(fresh.fc1) and is_lora_layer(fresh.fc2)


def test_archive_structure(tmp_path):
    m = _trained_net()
    p = tmp_path / "adapter.mdlus"
    save_adapter(m, p)
    with zipfile.ZipFile(p) as z:
        names = set(z.namelist())
        assert names == {"adapter_config.json", "adapter_model.pt", "metadata.json"}
        meta = json.loads(z.read("metadata.json"))
        cfg = json.loads(z.read("adapter_config.json"))
    assert meta["kind"] == "lora_adapter"
    assert meta["n_wrapped"] == 2
    assert sorted(cfg["target_modules"]) == ["fc1", "fc2"]


def test_save_requires_apply_first(tmp_path):
    m = _Net()  # never went through apply_lora
    with pytest.raises(ValueError, match="no stashed LoRA config"):
        save_adapter(m, tmp_path / "x.mdlus")


def test_save_requires_mdlus_extension(tmp_path):
    m = _trained_net()
    with pytest.raises(ValueError, match=r"\.mdlus"):
        save_adapter(m, tmp_path / "adapter.zip")


def test_kind_check_rejects_non_adapter(tmp_path):
    # Hand-craft a .mdlus whose metadata.kind is not "lora_adapter".
    p = tmp_path / "notadapter.mdlus"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr(
            "adapter_config.json",
            json.dumps(
                {
                    "rank": 4,
                    "alpha": 4.0,
                    "lora_dropout": 0.0,
                    "target_modules": ["fc1"],
                    "extras_trainable": [],
                    "init": "default",
                }
            ),
        )
        z.writestr("metadata.json", json.dumps({"kind": "model", "base_fingerprint": ""}))
        b = io.BytesIO()
        torch.save({}, b)
        z.writestr("adapter_model.pt", b.getvalue())

    m = _Net()
    with pytest.raises(ValueError, match="not a LoRA adapter"):
        load_adapter(m, p)


def test_fingerprint_mismatch_raises(tmp_path):
    m = _trained_net(d=16)
    p = tmp_path / "adapter.mdlus"
    save_adapter(m, p)
    # Different architecture → different fingerprint → strict load must refuse
    # before touching weights.
    other = _Net(d=8)
    with pytest.raises(ValueError, match="fingerprint"):
        load_adapter(other, p)
