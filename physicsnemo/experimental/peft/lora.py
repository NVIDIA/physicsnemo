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

"""LoRA layer wrappers and the type→wrapper registry.

``LoRALayer`` is a small stateful mixin; ``LoRALinear`` / ``LoRA_te_Linear``
combine it with a concrete base layer type. The module-level ``_LORA_WRAPPERS``
registry is the single extension seam (plan §14.1): supporting a new
architecture means registering one ``(layer_type, wrapper)`` pair, with no
changes to targeting / apply / save / merge.
"""

from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn as nn

try:  # Transformer Engine is optional.
    import transformer_engine.pytorch as te

    _TE_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on environment
    _TE_AVAILABLE = False


class LoRALayer:
    """Stateful mixin holding ``lora_A``, ``lora_B``, scaling, dropout and the
    enable flag. Combined with a base layer type by the wrapper subclasses.

    Math (plan §5.1): with ``lora_A: (in, r)`` and ``lora_B: (r, out)`` the
    forward adds ``((dropout(x) @ A) @ B) * scaling``. ``B`` is zero at init so
    the delta is exactly zero (the wrapped forward equals the base forward).
    """

    # Whether merge_lora can fold this adapter into base weights. False for the
    # fused te.LayerNormMLP residual wrapper (plan §5.3.2).
    mergeable: bool = True

    def _make_lora_params(
        self,
        in_features: int,
        out_features: int,
        ref_weight: torch.Tensor,
        rank: int,
        alpha: float,
        dropout: float,
    ) -> None:
        """Create lora_A/lora_B + dropout. ``ref_weight`` supplies device/dtype
        (inherited so DDP doesn't hit the CPU/CUDA mismatch — plan §5.2)."""
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.enabled = True
        self.lora_A = nn.Parameter(
            torch.empty(
                in_features, rank, device=ref_weight.device, dtype=ref_weight.dtype
            )
        )
        self.lora_B = nn.Parameter(
            torch.zeros(
                rank, out_features, device=ref_weight.device, dtype=ref_weight.dtype
            )
        )
        self.lora_dropout: nn.Module = (
            nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def _init_lora(
        self, base_layer: nn.Module, rank: int, alpha: float, dropout: float
    ) -> None:
        """Init for Linear-like bases: infer in/out + device/dtype from .weight."""
        in_features, out_features = self._infer_in_out_features(base_layer)
        self._make_lora_params(
            in_features, out_features, base_layer.weight, rank, alpha, dropout
        )

    @staticmethod
    def _infer_in_out_features(base_layer: nn.Module) -> tuple[int, int]:
        in_f = getattr(base_layer, "in_features", None)
        out_f = getattr(base_layer, "out_features", None)
        if in_f is None or out_f is None:
            # Fall back to the weight shape (out, in), as for nn.Linear.
            w = base_layer.weight
            out_f, in_f = int(w.shape[0]), int(w.shape[1])
        return int(in_f), int(out_f)

    def lora_delta(self, x: torch.Tensor) -> torch.Tensor:
        return ((self.lora_dropout(x) @ self.lora_A) @ self.lora_B) * self.scaling

    @torch.no_grad()
    def merge_into_base(self) -> None:
        """Fold ``scaling * (lora_A @ lora_B).T`` into ``base_layer.weight``
        (shape ``(out, in)``). Accumulate in fp32 then cast (plan §5.1).

        Note the transpose: ``lora_A @ lora_B`` is ``(in, out)``; the weight
        delta is its transpose. ``B @ A`` would be non-conformant.
        """
        delta = (self.lora_A.float() @ self.lora_B.float()).t() * self.scaling
        self.base_layer.weight.add_(delta.to(self.base_layer.weight.dtype))


class LoRALinear(nn.Module, LoRALayer):
    """LoRA wrapper for ``torch.nn.Linear``."""

    def __init__(
        self, base_layer: nn.Linear, rank: int, alpha: float, dropout: float = 0.0
    ) -> None:
        nn.Module.__init__(self)
        self.base_layer = base_layer
        for p in self.base_layer.parameters():
            p.requires_grad = False
        self._init_lora(base_layer, rank, alpha, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base_layer(x)
        if self.enabled:
            out = out + self.lora_delta(x)
        return out


if _TE_AVAILABLE:

    class LoRA_te_Linear(nn.Module, LoRALayer):
        """LoRA wrapper for ``transformer_engine.pytorch.Linear``.

        Passes TE-specific kwargs (e.g. ``is_first_microbatch`` for fp8)
        through to the frozen base layer.
        """

        def __init__(
            self,
            base_layer: "te.Linear",
            rank: int,
            alpha: float,
            dropout: float = 0.0,
        ) -> None:
            nn.Module.__init__(self)
            self.base_layer = base_layer
            for p in self.base_layer.parameters():
                p.requires_grad = False
            self._init_lora(base_layer, rank, alpha, dropout)

        def forward(self, x: torch.Tensor, **te_kwargs) -> torch.Tensor:
            out = self.base_layer(x, **te_kwargs)
            if self.enabled:
                out = out + self.lora_delta(x)
            return out

    class LoRA_te_LayerNormMLP(nn.Module, LoRALayer):
        """Option-B residual LoRA for the *fused* ``te.LayerNormMLP`` (plan §5.3.2).

        ``te.LayerNormMLP`` fuses ``LayerNorm → fc1 → act → fc2`` into one op with
        flat params (``layer_norm_weight``, ``fc1_weight``, ``fc2_weight``) — there
        is no child Linear to wrap and no way to inject a per-matrix LoRA between
        fc1 and the activation without abandoning the fused kernel. So this adds a
        single rank-r residual across the whole sub-block (hidden→hidden):
        ``y = te_layernorm_mlp(x) + ((dropout(x) @ A) @ B) * scaling``.

        Keeps the fused/fp8 kernel; NOT mergeable into the fused weights
        (``mergeable = False`` → merge_lora leaves it in place).
        """

        mergeable = False

        def __init__(
            self,
            base_layer: "te.LayerNormMLP",
            rank: int,
            alpha: float,
            dropout: float = 0.0,
        ) -> None:
            nn.Module.__init__(self)
            self.base_layer = base_layer
            for p in self.base_layer.parameters():
                p.requires_grad = False
            # hidden dim and device/dtype come from the fused LayerNorm weight.
            hidden = base_layer.layer_norm_weight.shape[0]
            self._make_lora_params(
                hidden, hidden, base_layer.layer_norm_weight, rank, alpha, dropout
            )

        def forward(self, x: torch.Tensor, **te_kwargs):
            out = self.base_layer(x, **te_kwargs)
            if not self.enabled:
                return out
            delta = self.lora_delta(x)
            if isinstance(out, tuple):  # e.g. when return_bias=True
                return (out[0] + delta, *out[1:])
            return out + delta

        def merge_into_base(self) -> None:  # pragma: no cover - guarded by mergeable
            raise NotImplementedError(
                "LoRA_te_LayerNormMLP is a sub-block residual and cannot be merged "
                "into the fused te.LayerNormMLP weights; keep the adapter un-merged."
            )


# --- type → wrapper registry (the §14.1 extension seam) --------------------
_LORA_WRAPPERS: dict[type, Callable[..., nn.Module]] = {nn.Linear: LoRALinear}
if _TE_AVAILABLE:
    _LORA_WRAPPERS[te.Linear] = LoRA_te_Linear
    _LORA_WRAPPERS[te.LayerNormMLP] = LoRA_te_LayerNormMLP


def register_lora_wrapper(
    layer_type: type, wrapper_factory: Callable[..., nn.Module]
) -> None:
    """Register a LoRA wrapper for ``layer_type``.

    This is how new architectures (equivariant, tensor, MoE — plan §14) plug in
    without touching the targeting / apply / merge core. ``wrapper_factory`` is
    called as ``wrapper_factory(base_layer, rank=, alpha=, dropout=)``.
    """
    _LORA_WRAPPERS[layer_type] = wrapper_factory


def get_wrapper_for(module: nn.Module) -> Callable[..., nn.Module] | None:
    """Return the registered wrapper for ``module`` (walking the MRO so that
    subclasses of a registered type are handled), or ``None`` if not wrappable.
    """
    for klass in type(module).__mro__:
        if klass in _LORA_WRAPPERS:
            return _LORA_WRAPPERS[klass]
    return None


def wrappable_types() -> tuple[type, ...]:
    """The layer types currently registered as wrappable."""
    return tuple(_LORA_WRAPPERS)


def is_lora_layer(module: nn.Module) -> bool:
    """Whether ``module`` is a LoRA wrapper."""
    return isinstance(module, LoRALayer)
