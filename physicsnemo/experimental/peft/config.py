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

"""LoRA configuration for ``physicsnemo.experimental.peft``.

LoRA (Low-Rank Adaptation) fine-tunes a frozen model by adding a small trainable
low-rank update ``B @ A`` beside selected linear layers. ``LoRAConfig`` declares
*which* layers to adapt and the adapter's capacity (``rank``) and strength
(``alpha``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Literal, Union

if TYPE_CHECKING:
    import torch
    import torch.nn as nn

# Accepted lora_A initialization spec: the named "default" strategy
# (kaiming_uniform, matching nn.Linear / the common PEFT default) or a callable
# that initializes the tensor in place. Extend the Literal as named strategies
# are added (e.g. SVD-based PiSSA/OLoRA).
LoRAInit = Union[Literal["default"], Callable[["torch.Tensor"], None]]


@dataclass
class LoRAConfig:
    """Configuration for applying LoRA to a model.

    Exactly one of ``target_modules``, ``target_pattern`` or ``target_filter``
    must be provided. They select layers by *fully-qualified* module name
    (e.g. ``blocks.3.Attn.qkv_project``), NOT bare leaf names — leaf names are
    not unique (the same short name can appear in many submodules).

    Parameters
    ----------
    rank : int
        Low-rank dimension ``r``. Must be positive.
    alpha : float | None
        LoRA scaling numerator; ``scaling = alpha / rank``. ``None`` defaults
        ``alpha`` to ``rank`` (scaling 1.0).
    target_modules : list[str] | None
        Exact fully-qualified module names to wrap.
    target_pattern : str | None
        Regex (``re.search``) matched against fully-qualified module names.
    target_filter : Callable[[str, nn.Module], bool] | None
        Predicate ``(name, module) -> bool`` (most flexible selector).
    lora_dropout : float
        Dropout on the LoRA input path; ``0.0`` disables it. In ``[0.0, 1.0)``.
    extras_trainable : list[str]
        Additional fully-qualified module names to leave fully trainable
        (not low-rank), e.g. a final head or norm.
    wrap_mlp : bool
        Convenience flag to *also* adapt the transformer **feed-forward (FFN)**
        sub-block — the position-wise ``Linear -> activation -> Linear`` that
        follows attention in a transformer block (NOT arbitrary or standalone
        MLPs, and not the model as a whole). In PhysicsNeMo transformer blocks
        this is the ``ln_mlp1`` module: under Transformer Engine the fused
        ``te.LayerNormMLP``, otherwise a ``Sequential(LayerNorm, Mlp)``. Matched
        by the known feed-forward naming of those blocks, so it is a no-op on
        models without that structure. Additive to the selector above.
    init : {"default"} or callable
        Initialization for the ``lora_A`` factor (``lora_B`` is always zero, so the
        adapter is identity at init). ``"default"`` uses ``kaiming_uniform_(a=√5)``
        — matching ``nn.Linear`` and the common PEFT default. Pass a callable
        ``(tensor) -> None`` to initialize ``lora_A`` in place with a custom scheme
        (e.g. ``lambda t: nn.init.normal_(t, std=0.01)`` for a Gaussian with a
        scale you control). Honored by wrappers built on ``_make_lora_params``;
        wrappers with their own parameterization (e.g. equivariant layers)
        initialize themselves and ignore this.
    """

    rank: int = 16
    alpha: float | None = None
    target_modules: list[str] | None = None
    target_pattern: str | None = None
    target_filter: Callable[[str, "nn.Module"], bool] | None = None
    lora_dropout: float = 0.0
    extras_trainable: list[str] = field(default_factory=list)
    wrap_mlp: bool = False
    init: LoRAInit = "default"

    def __post_init__(self) -> None:
        selectors = {
            "target_modules": self.target_modules,
            "target_pattern": self.target_pattern,
            "target_filter": self.target_filter,
        }
        set_selectors = [k for k, v in selectors.items() if v is not None]
        if len(set_selectors) != 1:
            raise ValueError(
                "Exactly one of target_modules, target_pattern, target_filter "
                f"must be set, got {len(set_selectors)} ({set_selectors})."
            )
        if self.target_modules is not None and len(self.target_modules) == 0:
            raise ValueError("target_modules is an empty list — nothing to select.")
        if self.target_pattern is not None and self.target_pattern == "":
            raise ValueError("target_pattern is an empty string — nothing to select.")
        if self.rank <= 0:
            raise ValueError(f"rank must be a positive integer, got {self.rank}.")
        if not (0.0 <= self.lora_dropout < 1.0):
            raise ValueError(
                f"lora_dropout must be in [0.0, 1.0), got {self.lora_dropout}."
            )
        if not (callable(self.init) or self.init == "default"):
            raise ValueError(
                f"init={self.init!r} is not supported; use 'default' or a callable "
                "that initializes the lora_A tensor in place."
            )

    @property
    def effective_alpha(self) -> float:
        """``alpha`` if set, else equal to ``rank`` (→ scaling 1.0)."""
        return float(self.alpha) if self.alpha is not None else float(self.rank)

    @property
    def scaling(self) -> float:
        """The LoRA scaling factor ``alpha / rank``."""
        return self.effective_alpha / self.rank
