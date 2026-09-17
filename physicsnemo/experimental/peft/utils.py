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

"""PEFT utilities: base fingerprinting, optimizer param splitting, reporting."""

from __future__ import annotations

import hashlib
import logging

import torch.nn as nn

from physicsnemo.experimental.peft.lora import is_lora_layer

logger = logging.getLogger("experimental.peft")


def compute_base_fingerprint(model: nn.Module) -> str:
    """Stable hash of module class names + state_dict keys/shapes.

    Used only internally to detect architecture (in)compatibility between an
    adapter and a base model — compared exact-match, not shown to users — so the
    full SHA-256 digest is returned (no truncation, no collision risk).

    MUST be computed on the pristine base, *before* ``apply_lora`` wraps
    anything: after wrapping, keys gain ``base_layer.`` prefixes and the
    original structure can no longer be recovered.
    """
    h = hashlib.sha256()
    for name, module in model.named_modules():
        h.update(f"{name}:{type(module).__name__}\n".encode())
    for key, tensor in model.state_dict().items():
        h.update(f"{key}:{tuple(tensor.shape)}\n".encode())
    return h.hexdigest()


def split_params_for_optimizer(model: nn.Module) -> dict[str, list]:
    """Split parameters into ``{'lora', 'extras', 'frozen'}``.

    Route ``lora + extras`` to AdamW — NOT to optimizers like Muon whose
    Newton-Schulz orthogonalization is degenerate on low-rank factors. ``frozen``
    is returned for reporting only.
    """
    lora: list = []
    extras: list = []
    frozen: list = []
    lora_ids: set[int] = set()
    for module in model.modules():
        if is_lora_layer(module):
            lora.extend([module.lora_A, module.lora_B])
            lora_ids.update({id(module.lora_A), id(module.lora_B)})
    for p in model.parameters():
        if id(p) in lora_ids:
            continue
        if p.requires_grad:
            extras.append(p)
        else:
            frozen.append(p)
    return {"lora": lora, "extras": extras, "frozen": frozen}


def print_trainable_parameters(model: nn.Module, use_logger: bool = False) -> str:
    """Emit a one-line ``trainable params: N (X% of M total)`` summary and
    return the message string."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = (100.0 * trainable / total) if total else 0.0
    msg = f"trainable params: {trainable:,} ({pct:.2f}% of {total:,} total)"
    if use_logger:
        logger.info(msg)
    else:
        print(msg)
    return msg


def set_adapter_enabled(model: nn.Module, enabled: bool) -> None:
    """Enable/disable all LoRA deltas in ``model``.

    Lets you run a base-only forward (e.g. for an adapter-vs-base comparison)
    without merging or reloading.
    """
    for module in model.modules():
        if is_lora_layer(module):
            module.enabled = enabled
