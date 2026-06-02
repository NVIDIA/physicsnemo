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

"""merge_lora — fold LoRA deltas into base weights for zero-overhead inference."""

from __future__ import annotations

import logging

import torch.nn as nn

from physicsnemo.experimental.peft.lora import is_lora_layer

logger = logging.getLogger("experimental.peft")


def merge_lora(model: nn.Module) -> nn.Module:
    """In-place: for each *mergeable* LoRA-wrapped module, fold its delta into
    the base weight and replace the wrapper with the (now-updated) underlying
    ``nn.Linear`` / ``te.Linear``. Returns ``model`` for chaining.

    Non-mergeable adapters (the fused ``te.LayerNormMLP`` residual, whose update
    can't be folded into the fused weights) are **left in place** and a warning
    is logged. After merging, mergeable wrappers are gone, so a model with only
    those can be saved as a normal ``.mdlus`` and served with zero adapter
    overhead. Idempotent — a second call is a no-op for merged layers.
    """
    # Snapshot names first: we mutate the tree (replace leaf wrappers) as we go.
    n_merged = 0
    n_skipped = 0
    for name, module in list(model.named_modules()):
        if not is_lora_layer(module):
            continue
        if not getattr(module, "mergeable", True):
            n_skipped += 1
            continue
        module.merge_into_base()
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child, module.base_layer)
        n_merged += 1
    if n_skipped:
        logger.warning(
            "merge_lora: merged %d adapters; left %d non-mergeable adapter(s) "
            "(e.g. te.LayerNormMLP residuals) in place.",
            n_merged,
            n_skipped,
        )
    return model
