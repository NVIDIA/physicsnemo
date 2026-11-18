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

r"""FLARE and FLARE++ models.

Standalone pre-norm transformer backbones whose token mixers are
:class:`~physicsnemo.nn.FLARE` or :class:`~physicsnemo.nn.FLAREPlusPlus`
attention:

.. code-block:: text

    x = Mlp(cat(embedding, fx))                       # C_in -> 2C -> C
    for each block:
        x = x + Mixer(Norm(x))                        # FLARE / FLARE++
        x = x + Mlp(Norm(x))                          # C -> mlp_ratio * C -> C
    y = Linear(Norm(x))                               # C -> C_out
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from jaxtyping import Float

from physicsnemo.core.meta import ModelMetaData
from physicsnemo.core.module import Module
from physicsnemo.models.utils.activation_checkpointing import (
    resolve_checkpointing_ratio,
    run_checkpoint,
    should_checkpoint_interleaved_block,
)
from physicsnemo.nn import FLARE as FLAREAttention
from physicsnemo.nn import FLAREPlusPlus as FLAREPlusPlusAttention
from physicsnemo.nn import Mlp


@dataclass
class MetaData(ModelMetaData):
    r"""Metadata for the FLARE and FLARE++ models."""

    # Optimization
    jit: bool = False
    cuda_graphs: bool = False
    amp: bool = True
    # Inference
    onnx_cpu: bool = False
    onnx_gpu: bool = True
    onnx_runtime: bool = True
    # Physics informed
    var_dim: int = 1
    func_torch: bool = False
    auto_grad: bool = False


def _norm(dim: int, use_rmsnorm: bool) -> nn.Module:
    return nn.RMSNorm(dim, eps=1e-6) if use_rmsnorm else nn.LayerNorm(dim)


class _FLAREBlock(nn.Module):
    r"""Pre-norm residual block: ``x + mixer(norm1(x))``, then ``x + mlp(norm2(x))``."""

    def __init__(
        self,
        mixer: nn.Module,
        hidden_dim: int,
        mlp_ratio: int,
        act: str,
        use_rmsnorm: bool,
    ) -> None:
        super().__init__()
        self.norm1 = _norm(hidden_dim, use_rmsnorm)
        self.mixer = mixer
        self.norm2 = _norm(hidden_dim, use_rmsnorm)
        self.mlp = Mlp(
            in_features=hidden_dim,
            hidden_features=hidden_dim * mlp_ratio,
            out_features=hidden_dim,
            act_layer=act,
        )

    def forward(self, x: Float[torch.Tensor, "B N C"]) -> Float[torch.Tensor, "B N C"]:
        r"""Apply the mixer and MLP residual branches."""
        x = x + self.mixer(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class _FLAREBackbone(Module):
    r"""Shared FLARE / FLARE++ backbone; subclasses provide the token mixer."""

    def __init__(
        self,
        functional_dim: int,
        out_dim: int,
        embedding_dim: int | None,
        n_layers: int,
        n_hidden: int,
        dropout: float,
        n_head: int,
        act: str,
        mlp_ratio: int,
        slice_num: int,
        use_rmsnorm: bool,
        activation_checkpointing: bool,
        checkpointing_ratio: float,
    ) -> None:
        super().__init__(meta=MetaData())
        if n_layers <= 0:
            raise ValueError(f"n_layers must be positive, got {n_layers}")
        if n_hidden % n_head != 0:
            raise ValueError(
                f"n_hidden must be divisible by n_head, got n_hidden={n_hidden}, "
                f"n_head={n_head}"
            )
        self.functional_dim = functional_dim
        self.embedding_dim = embedding_dim
        self.out_dim = out_dim
        self.n_hidden = n_hidden
        self._activation_checkpointing_ratio = resolve_checkpointing_ratio(
            activation_checkpointing, checkpointing_ratio
        )

        in_dim = functional_dim + (embedding_dim or 0)
        self.preprocess = Mlp(
            in_features=in_dim,
            hidden_features=2 * n_hidden,
            out_features=n_hidden,
            act_layer=act,
        )
        self.blocks = nn.ModuleList(
            [
                _FLAREBlock(
                    self._build_mixer(
                        dim=n_hidden,
                        heads=n_head,
                        dropout=dropout,
                        n_global_queries=slice_num,
                        use_rmsnorm=use_rmsnorm,
                    ),
                    hidden_dim=n_hidden,
                    mlp_ratio=mlp_ratio,
                    act=act,
                    use_rmsnorm=use_rmsnorm,
                )
                for _ in range(n_layers)
            ]
        )
        self.head = nn.Sequential(
            _norm(n_hidden, use_rmsnorm), nn.Linear(n_hidden, out_dim)
        )
        self.initialize_weights()

    def _build_mixer(self, **kwargs) -> nn.Module:
        r"""Return one block's token mixer (implemented by subclasses)."""
        raise NotImplementedError

    def initialize_weights(self) -> None:
        r"""Truncated-normal (std 0.02) linear weights, zero biases, unit norms.

        Linear layers flagged ``_skip_weight_init`` (the mixers' identity-plus-
        noise projections) and the mixers' latent parameters keep their own
        initialization.
        """
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            if getattr(m, "_skip_weight_init", False):
                return
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.RMSNorm)):
            if m.weight is not None:
                nn.init.ones_(m.weight)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)

    def no_weight_decay(self) -> set[str]:
        r"""Fully qualified names of parameters to train without weight decay.

        Collected from every submodule that defines ``no_weight_decay()``
        (the FLARE / FLARE++ mixers' latent queries and gate logits).
        """
        names = set()
        for prefix, module in self.named_modules():
            if module is self or not callable(getattr(module, "no_weight_decay", None)):
                continue
            names.update(f"{prefix}.{name}" for name in module.no_weight_decay())
        return names

    def forward(
        self,
        fx: Float[torch.Tensor, "B N C_fx"],
        embedding: Float[torch.Tensor, "B N C_emb"] | None = None,
    ) -> Float[torch.Tensor, "B N C_out"]:
        r"""Run the model.

        Parameters
        ----------
        fx : torch.Tensor
            Functional input of shape :math:`(B, N, C_{fx})`.
        embedding : torch.Tensor | None, optional
            Embedding (e.g. coordinates) of shape :math:`(B, N, C_{emb})`,
            concatenated in front of ``fx``. Required when the model was
            built with ``embedding_dim``.

        Returns
        -------
        torch.Tensor
            Output of shape :math:`(B, N, C_{out})`.
        """
        if not torch.compiler.is_compiling():
            if fx.ndim != 3:
                raise ValueError(
                    f"Expected fx of shape (B, N, C), got shape {tuple(fx.shape)}"
                )
            if self.embedding_dim and embedding is None:
                raise ValueError(
                    f"embedding is required (embedding_dim={self.embedding_dim})"
                )
            if not self.embedding_dim and embedding is not None:
                raise ValueError(
                    "embedding was given but the model was built without embedding_dim"
                )
        if embedding is not None:
            fx = torch.cat((embedding, fx), dim=-1)

        x = self.preprocess(fx)
        for idx, block in enumerate(self.blocks):
            if should_checkpoint_interleaved_block(
                idx,
                len(self.blocks),
                self._activation_checkpointing_ratio,
                training=self.training,
            ):
                x = run_checkpoint(block, x, use_te=False, te_module=None)
            else:
                x = block(x)
        return self.head(x)


_COMMON_PARAMS_DOC = r"""
    Parameters
    ----------
    functional_dim : int
        Number of functional input channels :math:`C_{fx}`.
    out_dim : int
        Number of output channels :math:`C_{out}`.
    embedding_dim : int | None, optional
        Number of embedding channels :math:`C_{emb}` (e.g. coordinates),
        concatenated in front of ``fx``. ``None`` means no embedding input.
        Default is ``None``.
    n_layers : int, optional
        Number of blocks. Default is 4.
    n_hidden : int, optional
        Hidden width :math:`C`. Default is 256.
    dropout : float, optional
        Dropout after each mixer's output projection. Default is 0.0.
    n_head : int, optional
        Number of attention heads; ``n_hidden`` must be divisible by it.
        Default is 8.
    act : str, optional
        Activation of the input projection and block MLPs (any
        :func:`~physicsnemo.nn.get_activation` name). Default is
        ``"gelu_tanh"`` (tanh-approximate GELU).
    mlp_ratio : int, optional
        Block MLP hidden width is ``mlp_ratio * n_hidden``. Default is 4.
    slice_num : int, optional
        Number of latent (routing) queries per head. Default is 32.
    use_rmsnorm : bool, optional
        Use RMSNorm (``eps=1e-6``) for every norm (block, output and the
        mixers' per-head norms); ``False`` uses LayerNorm. Default is ``True``.
    activation_checkpointing : bool, optional
        Checkpoint blocks during training. Default is ``False``.
    checkpointing_ratio : float, optional
        Fraction of blocks to checkpoint, spread evenly across the stack, when
        ``activation_checkpointing=True``. Default is 1.0."""

_COMMON_IO_DOC = r"""
    Forward
    -------
    fx : torch.Tensor
        Functional input of shape :math:`(B, N, C_{fx})`.
    embedding : torch.Tensor | None, optional
        Embedding of shape :math:`(B, N, C_{emb})`; required iff
        ``embedding_dim`` is set.

    Outputs
    -------
    torch.Tensor
        Output of shape :math:`(B, N, C_{out})`.

    Notes
    -----
    :meth:`no_weight_decay` returns the fully qualified names of the
    parameters (latent queries, gate logits) to train without weight decay."""


class FLARE(_FLAREBackbone):
    __doc__ = (
        r"""FLARE model: pre-norm transformer with FLARE attention mixers.

    Each block applies :class:`~physicsnemo.nn.FLARE` attention and an MLP,
    both as pre-norm residual branches. The input projection is a two-layer
    MLP (width :math:`2C`) and the output head is a norm plus a linear layer.
    See the `FLARE paper <https://arxiv.org/abs/2508.12594>`_.
"""
        + _COMMON_PARAMS_DOC
        + r"""
    qk_norm : bool, optional
        Normalize latent queries and keys per head in each mixer. Default is
        ``False``.
"""
        + _COMMON_IO_DOC
        + r"""

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.models.flare import FLARE
    >>> model = FLARE(functional_dim=3, embedding_dim=3, out_dim=4, n_hidden=64)
    >>> fx, emb = torch.randn(2, 100, 3), torch.randn(2, 100, 3)
    >>> model(fx, embedding=emb).shape
    torch.Size([2, 100, 4])
    """
    )

    def __init__(
        self,
        functional_dim: int,
        out_dim: int,
        embedding_dim: int | None = None,
        n_layers: int = 4,
        n_hidden: int = 256,
        dropout: float = 0.0,
        n_head: int = 8,
        act: str = "gelu_tanh",
        mlp_ratio: int = 4,
        slice_num: int = 32,
        use_rmsnorm: bool = True,
        qk_norm: bool = False,
        activation_checkpointing: bool = False,
        checkpointing_ratio: float = 1.0,
    ) -> None:
        self._qk_norm = qk_norm
        super().__init__(
            functional_dim=functional_dim,
            out_dim=out_dim,
            embedding_dim=embedding_dim,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout=dropout,
            n_head=n_head,
            act=act,
            mlp_ratio=mlp_ratio,
            slice_num=slice_num,
            use_rmsnorm=use_rmsnorm,
            activation_checkpointing=activation_checkpointing,
            checkpointing_ratio=checkpointing_ratio,
        )

    def _build_mixer(self, **kwargs) -> nn.Module:
        return FLAREAttention(**kwargs, qk_norm=self._qk_norm)


class FLAREPlusPlus(_FLAREBackbone):
    __doc__ = (
        r"""FLARE++ model: pre-norm transformer with FLARE++ attention mixers.

    Same backbone as :class:`FLARE`, with
    :class:`~physicsnemo.nn.FLAREPlusPlus` mixers, whose routing queries are
    synthesized from each block's input. See the `FLARE++ paper
    <https://arxiv.org/abs/2608.11519>`_.
"""
        + _COMMON_PARAMS_DOC
        + r"""
    k_norm : bool, optional
        Normalize routing keys per head. Default is ``True``.
    share_k0_v0 : bool, optional
        Share keys and values of the query-synthesis pass. Default is ``True``.
    q_fixed_norm : bool, optional
        Normalize the fixed anchor queries. Default is ``True``.
    gate_logit_init : float, optional
        Initial per-head gate logit of the dynamic queries. Default is 0.25.
"""
        + _COMMON_IO_DOC
        + r"""

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.models.flare import FLAREPlusPlus
    >>> model = FLAREPlusPlus(functional_dim=3, out_dim=4, n_hidden=64)
    >>> model(torch.randn(2, 100, 3)).shape
    torch.Size([2, 100, 4])
    """
    )

    def __init__(
        self,
        functional_dim: int,
        out_dim: int,
        embedding_dim: int | None = None,
        n_layers: int = 4,
        n_hidden: int = 256,
        dropout: float = 0.0,
        n_head: int = 8,
        act: str = "gelu_tanh",
        mlp_ratio: int = 4,
        slice_num: int = 32,
        use_rmsnorm: bool = True,
        k_norm: bool = True,
        share_k0_v0: bool = True,
        q_fixed_norm: bool = True,
        gate_logit_init: float = 0.25,
        activation_checkpointing: bool = False,
        checkpointing_ratio: float = 1.0,
    ) -> None:
        self._mixer_kwargs = dict(
            k_norm=k_norm,
            share_k0_v0=share_k0_v0,
            q_fixed_norm=q_fixed_norm,
            gate_logit_init=gate_logit_init,
        )
        super().__init__(
            functional_dim=functional_dim,
            out_dim=out_dim,
            embedding_dim=embedding_dim,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout=dropout,
            n_head=n_head,
            act=act,
            mlp_ratio=mlp_ratio,
            slice_num=slice_num,
            use_rmsnorm=use_rmsnorm,
            activation_checkpointing=activation_checkpointing,
            checkpointing_ratio=checkpointing_ratio,
        )

    def _build_mixer(self, **kwargs) -> nn.Module:
        return FLAREPlusPlusAttention(**kwargs, **self._mixer_kwargs)
