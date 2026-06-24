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

"""LoRAConfig validation tests (CPU-only)."""

import pytest

from physicsnemo.experimental.peft import LoRAConfig


def test_requires_exactly_one_selector():
    with pytest.raises(ValueError, match="Exactly one"):
        LoRAConfig()  # none set
    with pytest.raises(ValueError, match="Exactly one"):
        LoRAConfig(target_modules=["a"], target_pattern="b")  # two set


def test_valid_single_selector():
    assert LoRAConfig(target_pattern=r"blocks\.\d+").target_pattern is not None
    assert LoRAConfig(target_modules=["head"]).target_modules == ["head"]
    assert LoRAConfig(target_filter=lambda n, m: True).target_filter is not None


def test_scaling_and_alpha_default():
    cfg = LoRAConfig(rank=8, target_modules=["x"])
    assert cfg.effective_alpha == 8.0
    assert cfg.scaling == 1.0
    cfg2 = LoRAConfig(rank=8, alpha=16, target_modules=["x"])
    assert cfg2.scaling == 2.0


@pytest.mark.parametrize("bad", [0, -4])
def test_rank_must_be_positive(bad):
    with pytest.raises(ValueError, match="rank"):
        LoRAConfig(rank=bad, target_modules=["x"])


@pytest.mark.parametrize("bad", [1.0, 1.5, -0.1])
def test_dropout_range(bad):
    with pytest.raises(ValueError, match="lora_dropout"):
        LoRAConfig(lora_dropout=bad, target_modules=["x"])


def test_empty_selector_rejected():
    with pytest.raises(ValueError, match="empty"):
        LoRAConfig(target_modules=[])
    with pytest.raises(ValueError, match="empty"):
        LoRAConfig(target_pattern="")


def test_init_accepts_default_and_callable():
    assert LoRAConfig(target_modules=["x"], init="default").init == "default"

    def my_init(t):
        return None

    assert LoRAConfig(target_modules=["x"], init=my_init).init is my_init


@pytest.mark.parametrize("bad", ["gaussian", "pissa", "kaiming"])
def test_init_rejects_unknown_string(bad):
    # Only "default" is a named strategy; any other string must be a callable.
    with pytest.raises(ValueError, match="init"):
        LoRAConfig(target_modules=["x"], init=bad)
