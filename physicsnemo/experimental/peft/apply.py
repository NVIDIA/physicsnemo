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

"""``apply_lora`` and ``resolve_targets`` — in-place LoRA injection.

``apply_lora`` mutates the model in place: it replaces each matched
``nn.Linear`` / ``te.Linear`` with a LoRA-wrapped version and freezes the base
(except ``extras_trainable``). The model keeps its class and identity — only the
matched leaf layers change — so existing checkpoint/inference tooling still
works. Returns an :class:`ApplyResult` report; the mutated model is the input
object itself.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Callable

import torch.nn as nn

from physicsnemo.experimental.peft.config import LoRAConfig
from physicsnemo.experimental.peft.lora import get_wrapper_for, is_lora_layer
from physicsnemo.experimental.peft.utils import compute_base_fingerprint

logger = logging.getLogger("experimental.peft")

# (full_name, parent_module, child_attr_name, target_module)
Target = tuple[str, nn.Module, str, nn.Module]


@dataclass
class ApplyResult:
    """Summary report from :func:`apply_lora` (the model is mutated in place)."""

    n_wrapped: int
    n_trainable: int
    n_frozen: int
    trainable_names: list[str] = field(default_factory=list)
    base_fingerprint: str = ""


def _build_matcher(config: LoRAConfig) -> Callable[[str, nn.Module], bool]:
    """Build a ``(name, module) -> bool`` predicate from the config's selector.
    Exactly one selector is set (validated in ``LoRAConfig.__post_init__``)."""
    if config.target_modules is not None:
        names = set(config.target_modules)
        return lambda name, module: name in names
    if config.target_pattern is not None:
        pattern = re.compile(config.target_pattern)
        return lambda name, module: pattern.search(name) is not None
    return config.target_filter  # type: ignore[return-value]


# Default feed-forward MLP target patterns for known PhysicsNeMo transformer
# blocks (e.g. GALE_block / TransolverBlock). Regex over fully-qualified names;
# editable by users for custom architectures. The registry decides whether each
# match is the fused te.LayerNormMLP or plain Linears, and non-Linear matches
# (norms, activations) are filtered out by type.
_MLP_TARGET_PATTERNS: list[str] = [
    r"\.ln_mlp1$",  # GALE_block FFN under TE: fused te.LayerNormMLP
    r"\.ln_mlp1\.\d+\.layers\.\d+$",  # GALE_block FFN non-TE: Sequential(LayerNorm, Mlp(layers=...))
]


def resolve_targets(model: nn.Module, config: LoRAConfig) -> list[Target]:
    """Return ``(full_name, parent, child_attr, module)`` for every wrappable,
    matched layer.

    Matches against *fully-qualified* names. Skips already-wrapped layers
    (double-apply guard), module types not in the wrapper registry, and instances
    the matched wrapper's ``is_compatible`` rejects. When ``config.wrap_mlp`` is
    set, also matches the default MLP feed-forward patterns (additive to the
    primary selector).
    """
    matcher = _build_matcher(config)
    mlp_patterns = (
        [re.compile(p) for p in _MLP_TARGET_PATTERNS] if config.wrap_mlp else []
    )

    def _is_target(name: str, module: nn.Module) -> bool:
        """True if ``name`` matches the config selector, or (when ``wrap_mlp``) a
        default feed-forward pattern."""
        return matcher(name, module) or any(p.search(name) for p in mlp_patterns)

    modules = dict(model.named_modules())
    targets: list[Target] = []
    for name, module in modules.items():
        if name == "" or is_lora_layer(module):
            continue
        wrapper = get_wrapper_for(module)
        if wrapper is None:
            # Skip layers with no registered wrapper, but warn if the user's own
            # selector matched one (e.g. an nn.Embedding) so it isn't a silent
            # no-op. wrap_mlp pattern matches are excluded — they intentionally
            # sweep over norms/activations that are filtered out by type.
            if matcher(name, module):
                logger.warning(
                    "LoRA selector matched %s (%s), but no wrapper is registered "
                    "for that type; skipping it. Register one via "
                    "register_lora_wrapper.",
                    name,
                    type(module).__name__,
                )
            continue
        if not _is_target(name, module):
            continue
        # Type is wrappable and selected — let the wrapper veto this specific
        # instance (e.g. an equivariant adapter needs shared in/out irreps). Warn
        # on a user-selected veto so it isn't a silent skip.
        is_compatible = getattr(wrapper, "is_compatible", None)
        if is_compatible is not None and not is_compatible(module):
            if matcher(name, module):
                logger.warning(
                    "LoRA selector matched %s (%s), but %s.is_compatible reports "
                    "it can't be adapted; skipping it.",
                    name,
                    type(module).__name__,
                    getattr(wrapper, "__name__", type(wrapper).__name__),
                )
            continue
        parent_name, _, child = name.rpartition(".")
        parent = modules[parent_name] if parent_name else model
        targets.append((name, parent, child, module))
    return targets


def _freeze_base_except_extras(
    model: nn.Module, extras_trainable: list[str]
) -> None:
    """Freeze all parameters, then unfreeze LoRA params and any modules whose
    fully-qualified name is in ``extras_trainable``."""
    for p in model.parameters():
        p.requires_grad = False
    for module in model.modules():
        if is_lora_layer(module):
            module.lora_A.requires_grad = True
            module.lora_B.requires_grad = True
    if extras_trainable:
        # If an extras_trainable name is a *container* that also holds
        # LoRA-wrapped layers, do NOT re-enable grad on their frozen
        # base_layer weights (that would train the base and bloat the adapter,
        # which save_adapter slices by requires_grad).
        lora_base_param_ids = {
            id(p)
            for module in model.modules()
            if is_lora_layer(module)
            for p in module.base_layer.parameters()
        }
        extras = set(extras_trainable)
        for name, module in model.named_modules():
            if name in extras:
                for p in module.parameters():
                    if id(p) not in lora_base_param_ids:
                        p.requires_grad = True


def apply_lora(model: nn.Module, config: LoRAConfig) -> ApplyResult:
    """In-place: wrap matched ``Linear`` / ``te.Linear`` layers with LoRA and
    freeze the base (except ``extras_trainable``).

    Raises
    ------
    ValueError
        If the model already contains LoRA layers (not re-entrant), or if zero
        layers match the selector (silent-miss prevention).
    """
    if any(is_lora_layer(m) for m in model.modules()):
        raise ValueError(
            "model already contains LoRA layers; apply_lora is not re-entrant. "
            "Start from a fresh base model (or call merge_lora first)."
        )

    # Fingerprint the PRISTINE base before any mutation — once layers are
    # wrapped, the original structure can no longer be recovered.
    fingerprint = compute_base_fingerprint(model)

    targets = resolve_targets(model, config)
    if not targets:
        raise ValueError(
            "apply_lora matched 0 wrappable layers. Check target_modules / "
            "target_pattern / target_filter, and verify the model uses "
            "nn.Linear or te.Linear (not other Linear-like classes)."
        )

    for _name, parent, child, module in targets:
        wrapper = get_wrapper_for(module)
        wrapped = wrapper(
            module,
            rank=config.rank,
            alpha=config.effective_alpha,
            dropout=config.lora_dropout,
            init=config.init,
        )
        # Enforce the wrapper contract: freeze/save/merge identify LoRA layers by
        # isinstance(module, LoRALayer), so a wrapper that does not subclass it
        # would be silently ignored downstream. Fail loudly instead.
        if not is_lora_layer(wrapped):
            raise TypeError(
                f"LoRA wrapper for {type(module).__name__} returned a "
                f"{type(wrapped).__name__}, which does not subclass LoRALayer. "
                "Custom wrappers registered via register_lora_wrapper must "
                "subclass LoRALayer (see its docstring for the contract)."
            )
        setattr(parent, child, wrapped)

    _freeze_base_except_extras(model, config.extras_trainable)
    # Stash the pristine-base fingerprint and the metadata save_adapter serializes.
    # This is a plain, picklable dict — NOT the LoRAConfig, which may hold callable
    # target_filter/init — so a wrapped model can still be torch.save'd whole.
    # target_modules is omitted (save_adapter derives it live from the model). init
    # is recorded by name if it is a named strategy, else "custom": a callable can't
    # be serialized and is irrelevant on reload (the saved weights overwrite the
    # seed), but "custom" honestly signals that a non-default init was used.
    model._lora_base_fingerprint = fingerprint
    model._lora_adapter_config = {
        "rank": config.rank,
        "alpha": config.effective_alpha,
        "lora_dropout": config.lora_dropout,
        "extras_trainable": list(config.extras_trainable),
        "init": config.init if isinstance(config.init, str) else "custom",
    }

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
    logger.info(
        "apply_lora: wrapped %d layers; %d trainable params, %d frozen.",
        len(targets),
        n_trainable,
        n_frozen,
    )
    return ApplyResult(
        n_wrapped=len(targets),
        n_trainable=n_trainable,
        n_frozen=n_frozen,
        trainable_names=trainable_names,
        base_fingerprint=fingerprint,
    )
