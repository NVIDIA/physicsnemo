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

A LoRA wrapper holds a frozen base layer and adds a trainable low-rank update
``((x @ A) @ B) * scaling`` to its output. ``LoRALayer`` is a small stateful
mixin (holds ``lora_A``/``lora_B`` and the math); ``LoRALinear`` /
``LoRA_te_Linear`` combine it with a concrete base layer type.

New layer types plug in through the module-level ``_LORA_WRAPPERS`` registry:
register one ``(layer_type, wrapper)`` pair and the targeting / apply / save /
merge machinery picks it up unchanged.
"""

from __future__ import annotations

import importlib
import math
from typing import Callable

import torch
import torch.nn as nn

from physicsnemo.core.version_check import check_version_spec
from physicsnemo.experimental.peft.config import LoRAInit

# Transformer Engine is an optional dependency.
_TE_AVAILABLE = check_version_spec("transformer_engine", "0.1.0", hard_fail=False)
te = importlib.import_module("transformer_engine.pytorch") if _TE_AVAILABLE else None


def _resolve_lora_a_init(init: LoRAInit) -> Callable[[torch.Tensor], None]:
    """Map an ``init`` spec to a callable that initializes the ``lora_A`` tensor
    in place. ``lora_B`` stays zero regardless, so the adapter is identity at init.

    - ``"default"``: ``kaiming_uniform_(a=sqrt(5))`` — matches ``nn.Linear`` and
      the common PEFT default (the ``a=sqrt(5)`` is PyTorch's historical value).
    - a callable: used as-is (must initialize the passed tensor in place); this is
      how you select a custom scheme, e.g. a Gaussian with a chosen scale.
    """
    if callable(init):
        return init
    if init == "default":
        return lambda t: nn.init.kaiming_uniform_(t, a=math.sqrt(5))
    raise ValueError(
        f"unknown init strategy {init!r}; use 'default' or a callable."
    )


class LoRALayer:
    """Generic LoRA mixin: holds ``lora_A``/``lora_B``, scaling, dropout and the
    enable flag, and computes the low-rank delta. Makes **no assumption** about
    the base layer's parameter shapes — combined with a base layer type by the
    wrapper subclasses.

    Math: with ``lora_A: (in, r)`` and ``lora_B: (r, out)`` the forward adds
    ``((dropout(x) @ A) @ B) * scaling``. ``B`` is zero at init so the delta is
    exactly zero — the wrapped forward equals the base forward until trained.

    Wrapper contract
    ----------------
    Every LoRA wrapper — the built-ins below and any registered via
    :func:`register_lora_wrapper` — is an ``nn.Module`` that subclasses
    ``LoRALayer`` and exposes the surface the apply / freeze / save / merge /
    enable utilities depend on:

    - **Constructor**: ``__init__(self, base_layer, *, rank, alpha, dropout=0.0,
      init="default")``. ``apply_lora`` instantiates wrappers as
      ``wrapper(base_layer, rank=, alpha=, dropout=, init=)`` — accept ``**kwargs``
      if you want to be forward-compatible with options added later.
    - **Attributes**: ``base_layer`` (the wrapped, frozen module); ``lora_A`` /
      ``lora_B`` (trainable ``nn.Parameter``\\ s — the only params left with
      ``requires_grad=True``, which is how ``save_adapter`` slices the adapter;
      these may be plain attributes or ``@property``\\ s that resolve to the
      Parameters, e.g. for layers whose factors are submodules); ``enabled`` (bool
      toggling the delta); ``mergeable`` (bool; ``False`` by default — opt in only
      if you also implement ``merge_into_base``).
    - **Methods**: ``forward`` (adds the low-rank delta to the base output when
      ``enabled``); ``merge_into_base`` (folds the delta into the base weight) —
      required only when ``mergeable`` is ``True``. Optionally override the
      classmethod ``is_compatible(base_layer)`` to veto instances of a registered
      type this wrapper can't adapt (defaults to accepting all).

    ``_make_lora_params(...)`` and ``lora_delta(...)`` are **optional conveniences**
    for the standard tensor case (2-D ``lora_A``/``lora_B`` with the
    ``((x @ A) @ B) * scaling`` delta): a wrapper may instead create its own
    parameters, init, and delta (e.g. an equivariant wrapper whose factors are
    themselves equivariant layers) as long as it ends up satisfying the contract
    above. Wrappers for Linear-like bases (``.weight`` shaped ``(out, in)``, or
    exposing ``in_features``/``out_features``) should subclass
    :class:`_LinearLoRALayer` instead — it adds in/out inference at init and a
    weight-folding ``merge_into_base``. Only generic, non-Linear wrappers (e.g. the
    fused ``te.LayerNormMLP`` residual) inherit ``LoRALayer`` directly.
    """

    # Whether merge_lora can fold this adapter into base weights. Generic wrappers
    # are non-mergeable by default; Linear-like wrappers (_LinearLoRALayer) opt in.
    mergeable: bool = False

    def _make_lora_params(
        self,
        in_features: int,
        out_features: int,
        ref_weight: torch.Tensor,
        rank: int,
        alpha: float,
        dropout: float,
        init: LoRAInit = "default",
    ) -> None:
        """Create lora_A/lora_B + dropout. ``ref_weight`` supplies device/dtype,
        inherited so the LoRA params live wherever the base weight does (avoids a
        device mismatch under DDP). ``init`` selects how ``lora_A`` is initialized
        (see :func:`_resolve_lora_a_init`); ``lora_B`` is always zero so the delta
        is exactly zero at init."""
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
        _resolve_lora_a_init(init)(self.lora_A)

    def lora_delta(self, x: torch.Tensor) -> torch.Tensor:
        """The low-rank update added to the base output:
        ``((dropout(x) @ lora_A) @ lora_B) * scaling``. Zero at init since
        ``lora_B`` starts at zero."""
        return ((self.lora_dropout(x) @ self.lora_A) @ self.lora_B) * self.scaling

    @classmethod
    def is_compatible(cls, base_layer: nn.Module) -> bool:
        """Whether this wrapper can adapt ``base_layer`` beyond simple type match.

        ``resolve_targets`` calls this on a selected, registered-type layer before
        wrapping it and skips the layer if it returns ``False`` — letting a wrapper
        veto instances it can't actually handle (e.g. an equivariant adapter that
        needs at least one shared input/output irrep). Defaults to ``True`` (every
        instance of a registered type is adaptable); override in subclasses that
        need an instance-level check.
        """
        return True


class _LinearLoRALayer(LoRALayer):
    """LoRA mixin specialized for Linear-like bases — those whose ``.weight`` is
    shaped ``(out, in)`` and/or expose ``in_features``/``out_features`` (e.g.
    ``nn.Linear``, ``te.Linear``). Adds in/out inference at init and a merge that
    folds the delta into ``base_layer.weight``.

    Non-Linear wrappers must NOT use this — they inherit :class:`LoRALayer`
    directly so they don't pick up these weight-shaped assumptions.
    """

    mergeable: bool = True

    @staticmethod
    def _is_linear_like(base_layer: nn.Module) -> bool:
        """True if ``base_layer`` exposes ``in_features``/``out_features`` or a 2-D
        ``weight``, so a ``(out, in)`` LoRA factorization can be inferred (e.g.
        ``nn.Linear``, ``te.Linear``)."""
        if (
            getattr(base_layer, "in_features", None) is not None
            and getattr(base_layer, "out_features", None) is not None
        ):
            return True
        return getattr(getattr(base_layer, "weight", None), "ndim", None) == 2

    @classmethod
    def is_compatible(cls, base_layer: nn.Module) -> bool:
        """True if ``base_layer`` is Linear-like (see ``_is_linear_like``).
        Non-Linear bases are skipped by ``resolve_targets`` rather than failing at
        wrap time. (This won't catch ``nn.Embedding``, which has a 2-D weight —
        embeddings need a dedicated index-lookup wrapper registered separately.)"""
        return cls._is_linear_like(base_layer)

    def _init_lora(
        self,
        base_layer: nn.Module,
        rank: int,
        alpha: float,
        dropout: float,
        init: LoRAInit = "default",
    ) -> None:
        """Infer in/out + device/dtype from the base ``.weight`` and create the
        LoRA params."""
        in_features, out_features = self._infer_in_out_features(base_layer)
        self._make_lora_params(
            in_features, out_features, base_layer.weight, rank, alpha, dropout, init
        )

    @staticmethod
    def _infer_in_out_features(base_layer: nn.Module) -> tuple[int, int]:
        """Infer ``(in_features, out_features)`` from a Linear-like base via its
        ``in_features``/``out_features`` attributes, or its 2-D ``.weight`` shape
        ``(out, in)``.

        Raises
        ------
        TypeError
            If ``base_layer`` is not Linear-like (see ``_is_linear_like``).
            ``resolve_targets`` normally skips such layers via ``is_compatible``;
            this guard defends direct construction, pointing to
            ``register_lora_wrapper`` for a dedicated wrapper.
        """
        if not _LinearLoRALayer._is_linear_like(base_layer):
            raise TypeError(
                f"{type(base_layer).__name__} is not Linear-like (no in_features/"
                "out_features and no 2-D weight), so its LoRA factor shapes can't "
                "be inferred; register a dedicated wrapper via register_lora_wrapper."
            )
        in_f = getattr(base_layer, "in_features", None)
        out_f = getattr(base_layer, "out_features", None)
        if in_f is not None and out_f is not None:
            return int(in_f), int(out_f)
        w = base_layer.weight  # guaranteed 2-D by _is_linear_like
        return int(w.shape[1]), int(w.shape[0])

    @torch.no_grad()
    def merge_into_base(self) -> None:
        """Fold ``scaling * (lora_A @ lora_B).T`` into ``base_layer.weight``
        (shape ``(out, in)``). Accumulate in fp32 then cast to the base dtype.

        Note the transpose: ``lora_A @ lora_B`` is ``(in, out)``; the weight
        delta is its transpose. ``B @ A`` would be non-conformant.
        """
        delta = (self.lora_A.float() @ self.lora_B.float()).t() * self.scaling
        self.base_layer.weight.add_(delta.to(self.base_layer.weight.dtype))


class LoRALinear(nn.Module, _LinearLoRALayer):
    """LoRA wrapper for ``torch.nn.Linear``.

    Wraps a frozen ``nn.Linear`` and adds a trainable low-rank update to its
    output (``base(x) + lora_delta(x)``). Only ``lora_A``/``lora_B`` train; the
    base layer's weight and bias are frozen in place.

    Parameters
    ----------
    base_layer : nn.Linear
        The linear layer to wrap; its parameters are frozen.
    rank : int
        Low-rank dimension ``r`` of the adapter.
    alpha : float
        LoRA scaling numerator; the delta is scaled by ``alpha / rank``.
    dropout : float, optional
        Dropout applied to the adapter input path. Defaults to ``0.0``.
    init : str or callable, optional
        How ``lora_A`` is initialized (``lora_B`` is always zero). See
        :class:`~physicsnemo.experimental.peft.config.LoRAConfig`. Defaults to
        ``"default"`` (``kaiming_uniform_``).
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
        init: LoRAInit = "default",
    ) -> None:
        nn.Module.__init__(self)
        self.base_layer = base_layer
        for p in self.base_layer.parameters():
            p.requires_grad = False
        self._init_lora(base_layer, rank, alpha, dropout, init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Frozen base output plus the LoRA delta (when ``enabled``)."""
        out = self.base_layer(x)
        if self.enabled:
            out = out + self.lora_delta(x)
        return out


if _TE_AVAILABLE:

    class LoRA_te_Linear(nn.Module, _LinearLoRALayer):
        """LoRA wrapper for ``transformer_engine.pytorch.Linear``.

        Adds a trainable low-rank update to the frozen TE linear's output, passing
        TE-specific kwargs (e.g. ``is_first_microbatch`` for fp8) through to the
        base layer.

        Parameters
        ----------
        base_layer : te.Linear
            The Transformer Engine linear layer to wrap; its parameters are frozen.
        rank : int
            Low-rank dimension ``r`` of the adapter.
        alpha : float
            LoRA scaling numerator; the delta is scaled by ``alpha / rank``.
        dropout : float, optional
            Dropout applied to the adapter input path. Defaults to ``0.0``.
        init : str or callable, optional
            How ``lora_A`` is initialized (``lora_B`` is always zero). See
            :class:`~physicsnemo.experimental.peft.config.LoRAConfig`. Defaults to
            ``"default"`` (``kaiming_uniform_``).
        """

        def __init__(
            self,
            base_layer: "te.Linear",
            rank: int,
            alpha: float,
            dropout: float = 0.0,
            init: LoRAInit = "default",
        ) -> None:
            nn.Module.__init__(self)
            self.base_layer = base_layer
            for p in self.base_layer.parameters():
                p.requires_grad = False
            self._init_lora(base_layer, rank, alpha, dropout, init)

        def forward(self, x: torch.Tensor, **te_kwargs) -> torch.Tensor:
            """Frozen base output (with TE kwargs) plus the LoRA delta when
            ``enabled``."""
            out = self.base_layer(x, **te_kwargs)
            if self.enabled:
                out = out + self.lora_delta(x)
            return out

    class LoRA_te_LayerNormMLP(nn.Module, LoRALayer):
        """Residual LoRA for the *fused* ``te.LayerNormMLP``.

        ``te.LayerNormMLP`` fuses ``LayerNorm → fc1 → act → fc2`` into one op with
        flat params (``layer_norm_weight``, ``fc1_weight``, ``fc2_weight``) — there
        is no child Linear to wrap and no way to inject a per-matrix LoRA between
        fc1 and the activation without abandoning the fused kernel. So this adds a
        single rank-r residual across the whole sub-block (hidden→hidden):
        ``y = te_layernorm_mlp(x) + ((dropout(x) @ A) @ B) * scaling``.

        Keeps the fused/fp8 kernel; NOT mergeable into the fused weights
        (``mergeable = False`` → merge_lora leaves it in place).

        Parameters
        ----------
        base_layer : te.LayerNormMLP
            The fused TE LayerNorm-MLP block to wrap; its parameters are frozen.
        rank : int
            Low-rank dimension ``r`` of the residual adapter.
        alpha : float
            LoRA scaling numerator; the residual is scaled by ``alpha / rank``.
        dropout : float, optional
            Dropout applied to the adapter input path. Defaults to ``0.0``.
        init : str or callable, optional
            How ``lora_A`` is initialized (``lora_B`` is always zero). See
            :class:`~physicsnemo.experimental.peft.config.LoRAConfig`. Defaults to
            ``"default"`` (``kaiming_uniform_``).
        """

        mergeable = False

        def __init__(
            self,
            base_layer: "te.LayerNormMLP",
            rank: int,
            alpha: float,
            dropout: float = 0.0,
            init: LoRAInit = "default",
        ) -> None:
            nn.Module.__init__(self)
            self.base_layer = base_layer
            for p in self.base_layer.parameters():
                p.requires_grad = False
            # hidden dim and device/dtype come from the fused LayerNorm weight.
            hidden = base_layer.layer_norm_weight.shape[0]
            self._make_lora_params(
                hidden, hidden, base_layer.layer_norm_weight, rank, alpha, dropout, init
            )

        def forward(self, x: torch.Tensor, **te_kwargs):
            """Fused base output plus the rank-r residual when ``enabled``;
            preserves a tuple output (e.g. ``return_bias=True``)."""
            out = self.base_layer(x, **te_kwargs)
            if not self.enabled:
                return out
            delta = self.lora_delta(x)
            if isinstance(out, tuple):  # e.g. when return_bias=True
                return (out[0] + delta, *out[1:])
            return out + delta

        def merge_into_base(self) -> None:  # pragma: no cover - guarded by mergeable
            """Not supported: the residual can't be folded into the fused weights
            (so ``mergeable`` is ``False`` and merge_lora never calls this)."""
            raise NotImplementedError(
                "LoRA_te_LayerNormMLP is a sub-block residual and cannot be merged "
                "into the fused te.LayerNormMLP weights; keep the adapter un-merged."
            )


# --- type → wrapper registry (the extension seam for new layer types) ------
_LORA_WRAPPERS: dict[type, Callable[..., nn.Module]] = {nn.Linear: LoRALinear}
if _TE_AVAILABLE:
    _LORA_WRAPPERS[te.Linear] = LoRA_te_Linear
    _LORA_WRAPPERS[te.LayerNormMLP] = LoRA_te_LayerNormMLP


def register_lora_wrapper(
    layer_type: type, wrapper_factory: Callable[..., nn.Module]
) -> None:
    """Register a LoRA wrapper for ``layer_type``.

    This is how new architectures (e.g. equivariant, tensor, or MoE layers) plug
    in without touching the targeting / apply / merge core.

    Parameters
    ----------
    layer_type : type
        The base layer class to wrap (e.g. a custom ``nn.Module`` subclass).
        Matched against each module's MRO, so subclasses are handled too.
    wrapper_factory : Callable[..., nn.Module]
        Called as ``wrapper_factory(base_layer, rank=, alpha=, dropout=, init=)``
        and must return an ``nn.Module`` that subclasses :class:`LoRALayer` (see
        its docstring for the full attribute/method contract). The subclass
        requirement is enforced by ``apply_lora``: freeze/save/merge identify LoRA
        layers via ``isinstance(module, LoRALayer)``, so a wrapper that does not
        subclass it would otherwise be silently skipped.
    """
    _LORA_WRAPPERS[layer_type] = wrapper_factory


def get_wrapper_for(module: nn.Module) -> Callable[..., nn.Module] | None:
    """Return the registered LoRA wrapper factory for ``module``, or ``None``.

    Parameters
    ----------
    module : nn.Module
        The candidate base layer to wrap.

    Returns
    -------
    Callable[..., nn.Module] or None
        The registered wrapper factory for ``module``'s type, found by walking its
        MRO (so subclasses of a registered type are handled), or ``None`` if no
        registered type matches (i.e. the module is not wrappable).
    """
    for _class in type(module).__mro__:
        if _class in _LORA_WRAPPERS:
            return _LORA_WRAPPERS[_class]
    return None


def wrappable_types() -> tuple[type, ...]:
    """Return the layer types currently registered as wrappable.

    Returns
    -------
    tuple[type, ...]
        The registered base layer types (e.g. ``nn.Linear`` and, when available,
        the Transformer Engine types).
    """
    return tuple(_LORA_WRAPPERS)


def is_lora_layer(module: nn.Module) -> bool:
    """Return whether ``module`` is a LoRA wrapper.

    Parameters
    ----------
    module : nn.Module
        The module to test.

    Returns
    -------
    bool
        ``True`` if ``module`` is a :class:`LoRALayer` instance.
    """
    return isinstance(module, LoRALayer)
