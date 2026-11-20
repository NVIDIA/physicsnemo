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

r"""FLAREv2 and FLARE++ models.

Standalone pre-norm transformer backbones whose token mixers are
:class:`~physicsnemo.nn.FLAREv2` or :class:`~physicsnemo.nn.FLAREPlusPlus`
attention:

.. code-block:: text

    x = Mlp(cat(embedding, fx))                       # C_in -> 2C -> C
    for each block:
        x = x + Mixer(Norm(x))                        # FLAREv2 / FLARE++
        x = x + Mlp(Norm(x))                          # C -> mlp_ratio * C -> C
    y = Linear(Norm(x))                               # C -> C_out

With ``use_te=True`` the linear layers, norms and attention run on Transformer
Engine, mirroring :class:`~physicsnemo.models.transolver.Transolver`'s TE
blocks: ``Norm + Mlp`` is one ``te.LayerNormMLP`` and the head is one
``te.LayerNormLinear``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from jaxtyping import Float

from physicsnemo.core.meta import ModelMetaData
from physicsnemo.core.module import Module
from physicsnemo.core.version_check import OptionalImport
from physicsnemo.models.utils.activation_checkpointing import (
    resolve_checkpointing_ratio,
    run_checkpoint,
    should_checkpoint_interleaved_block,
)
from physicsnemo.nn import FLAREPlusPlus as FLAREPlusPlusAttention
from physicsnemo.nn import FLAREv2 as FLAREv2Attention
from physicsnemo.nn import Mlp

te = OptionalImport("transformer_engine.pytorch")

# Transformer Engine LayerNormMLP activation names; TE's "gelu" is the tanh
# approximation (as in TransolverBlock, whose TE MLP always uses it).
_TE_ACTIVATIONS = {"gelu_tanh": "gelu", "gelu": "gelu", "relu": "relu", "silu": "silu"}


@dataclass
class MetaData(ModelMetaData):
    r"""Metadata for the FLAREv2 and FLARE++ models."""

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


def _norm(dim: int, use_rmsnorm: bool, use_te: bool = False) -> nn.Module:
    if use_te:
        return te.RMSNorm(dim, eps=1e-6) if use_rmsnorm else te.LayerNorm(dim)
    return nn.RMSNorm(dim, eps=1e-6) if use_rmsnorm else nn.LayerNorm(dim)


def _te_norm_kwargs(use_rmsnorm: bool) -> dict:
    r"""``normalization`` / ``eps`` for Transformer Engine fused norm modules."""
    if use_rmsnorm:
        return {"normalization": "RMSNorm", "eps": 1e-6}
    return {"normalization": "LayerNorm", "eps": 1e-5}


def _te_activation(act: str) -> str:
    if act not in _TE_ACTIVATIONS:
        raise ValueError(
            f"act={act!r} is not supported with use_te=True; choose one of "
            f"{sorted(_TE_ACTIVATIONS)}"
        )
    return _TE_ACTIVATIONS[act]


class _FLAREBlock(nn.Module):
    r"""Pre-norm residual block: ``x + mixer(norm1(x))``, then ``x + mlp(norm2(x))``.

    With ``use_te=True`` the MLP branch is one fused ``te.LayerNormMLP``
    (``ln_mlp``) instead of ``norm2`` + ``mlp``.
    """

    def __init__(
        self,
        mixer: nn.Module,
        hidden_dim: int,
        mlp_ratio: int,
        act: str,
        use_rmsnorm: bool,
        use_te: bool = False,
    ) -> None:
        super().__init__()
        self.use_te = use_te
        self.norm1 = _norm(hidden_dim, use_rmsnorm, use_te)
        self.mixer = mixer
        if use_te:
            self.ln_mlp = te.LayerNormMLP(
                hidden_size=hidden_dim,
                ffn_hidden_size=hidden_dim * mlp_ratio,
                activation=_te_activation(act),
                **_te_norm_kwargs(use_rmsnorm),
            )
        else:
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
        if self.use_te:
            return x + self.ln_mlp(x)
        return x + self.mlp(self.norm2(x))


class _FLAREBackbone(Module):
    r"""Shared FLAREv2 / FLARE++ backbone; subclasses provide the token mixer."""

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
        use_te: bool,
        activation_checkpointing: bool,
        checkpointing_ratio: float,
    ) -> None:
        super().__init__(meta=MetaData())
        if use_te and not te.available:
            raise ImportError(
                f"{type(self).__name__}(use_te=True) requires Transformer Engine. "
                "Install it with `pip install transformer-engine`, or set use_te=False."
            )
        if use_te:
            _te_activation(act)
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
        self.use_te = use_te
        self._activation_checkpointing_ratio = resolve_checkpointing_ratio(
            activation_checkpointing, checkpointing_ratio
        )

        in_dim = functional_dim + (embedding_dim or 0)
        self.preprocess = Mlp(
            in_features=in_dim,
            hidden_features=2 * n_hidden,
            out_features=n_hidden,
            act_layer=act,
            use_te=use_te,
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
                        use_te=use_te,
                    ),
                    hidden_dim=n_hidden,
                    mlp_ratio=mlp_ratio,
                    act=act,
                    use_rmsnorm=use_rmsnorm,
                    use_te=use_te,
                )
                for _ in range(n_layers)
            ]
        )
        if use_te:
            self.head = te.LayerNormLinear(
                in_features=n_hidden,
                out_features=out_dim,
                **_te_norm_kwargs(use_rmsnorm),
            )
        else:
            self.head = nn.Sequential(
                _norm(n_hidden, use_rmsnorm), nn.Linear(n_hidden, out_dim)
            )
        self.initialize_weights()

    def _build_mixer(self, **kwargs) -> nn.Module:
        r"""Return one block's token mixer (implemented by subclasses).

        Parameters
        ----------
        **kwargs
            Mixer options shared by every block (``dim``, ``heads``, ...).

        Returns
        -------
        nn.Module
            Token mixer mapping :math:`(B, N, C)` to :math:`(B, N, C)`.
        """
        raise NotImplementedError

    def initialize_weights(self) -> None:
        r"""Truncated-normal (std 0.02) linear weights, zero biases, unit norms.

        Linear layers flagged ``_skip_weight_init`` (the mixers' identity-plus-
        noise projections) and the mixers' latent parameters keep their own
        initialization.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        r"""Initialize one module's parameters in place (used by ``self.apply``).

        Parameters
        ----------
        m : nn.Module
            Module to initialize. Linear weights get truncated-normal
            (std 0.02) values and zero biases; norm weights get ones and zero
            biases. Linear layers flagged ``_skip_weight_init`` are skipped.

        Returns
        -------
        None
        """
        linear_layers: tuple[type, ...] = (nn.Linear,)
        norm_layers: tuple[type, ...] = (nn.LayerNorm, nn.RMSNorm)
        if self.use_te:
            linear_layers += (te.Linear,)
            norm_layers += (te.LayerNorm, te.RMSNorm)
        if isinstance(m, linear_layers):
            if getattr(m, "_skip_weight_init", False):
                return
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, norm_layers):
            if m.weight is not None:
                nn.init.ones_(m.weight)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)
        elif self.use_te and isinstance(m, (te.LayerNormMLP, te.LayerNormLinear)):
            # Fused TE modules hold their parameters directly (no Linear
            # children): same trunc_normal / zero-bias / unit-norm init.
            if isinstance(m, te.LayerNormMLP):
                weights, biases = ("fc1_weight", "fc2_weight"), ("fc1_bias", "fc2_bias")
            else:
                weights, biases = ("weight",), ("bias",)
            for name in weights:
                nn.init.trunc_normal_(getattr(m, name), std=0.02)
            for name in (*biases, "layer_norm_bias"):
                param = getattr(m, name, None)
                if isinstance(param, nn.Parameter) and param.numel():
                    nn.init.zeros_(param)
            nn.init.ones_(m.layer_norm_weight)

    def no_weight_decay(self) -> set[str]:
        r"""Fully qualified names of parameters to train without weight decay.

        Collected from every submodule that defines ``no_weight_decay()``
        (the FLAREv2 / FLARE++ mixers' latent queries and gate logits).

        Parameters
        ----------
        None

        Returns
        -------
        set[str]
            Fully qualified parameter names, e.g.
            ``"blocks.0.mixer.latent_q"``.
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
            built with a nonzero ``embedding_dim``.

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
            width = 0 if embedding is None else embedding.shape[-1]
            if width != (self.embedding_dim or 0):
                raise ValueError(
                    f"Expected an embedding with {self.embedding_dim or 0} channels "
                    f"(embedding_dim={self.embedding_dim}), got "
                    f"{'none' if embedding is None else width}"
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
                x = run_checkpoint(block, x, use_te=self.use_te, te_module=te)
            else:
                x = block(x)
        return self.head(x)


class FLAREv2(_FLAREBackbone):
    r"""FLAREv2 model: pre-norm transformer with FLAREv2 attention mixers.

    Each block applies :class:`~physicsnemo.nn.FLAREv2` attention and an MLP,
    both as pre-norm residual branches. The input projection is a two-layer
    MLP (width :math:`2C`) and the output head is a norm plus a linear layer.
    See the `FLARE paper <https://arxiv.org/abs/2508.12594>`_.

    Parameters
    ----------
    functional_dim : int
        Number of functional input channels :math:`C_{fx}`.
    out_dim : int
        Number of output channels :math:`C_{out}`.
    embedding_dim : int | None, optional, default=None
        Number of embedding channels :math:`C_{emb}` (e.g. coordinates),
        concatenated in front of ``fx``. ``None`` means no embedding input.
    n_layers : int, optional, default=4
        Number of blocks.
    n_hidden : int, optional, default=256
        Hidden width :math:`C`.
    dropout : float, optional, default=0.0
        Dropout after each mixer's output projection.
    n_head : int, optional, default=8
        Number of attention heads; ``n_hidden`` must be divisible by it.
    act : str, optional, default="gelu_tanh"
        Activation of the input projection and block MLPs (any
        :func:`~physicsnemo.nn.get_activation` name). The default is
        tanh-approximate GELU.
    mlp_ratio : int, optional, default=4
        Block MLP hidden width is ``mlp_ratio * n_hidden``.
    slice_num : int, optional, default=32
        Number of latent (routing) queries per head.
    use_rmsnorm : bool, optional, default=True
        Use RMSNorm (``eps=1e-6``) for every norm (block, output and the
        mixers' per-head norms); ``False`` uses LayerNorm.
    use_te : bool, optional, default=False
        Use Transformer Engine layers (linear, norms, fused norm + MLP, fused
        norm + linear head, ``DotProductAttention``; CUDA only). With
        ``use_te=True``, ``act`` must be one of ``"gelu_tanh"``, ``"gelu"``
        (both tanh-approximate GELU in Transformer Engine), ``"relu"`` or
        ``"silu"``.
    activation_checkpointing : bool, optional, default=False
        Checkpoint blocks during training.
    checkpointing_ratio : float, optional, default=1.0
        Fraction of blocks to checkpoint, spread evenly across the stack, when
        ``activation_checkpointing=True``.
    qk_norm : bool, optional, default=True
        Normalize latent queries and keys per head in each mixer (see
        :class:`~physicsnemo.nn.FLAREv2`).

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
    parameters (latent queries, gate logits) to train without weight decay.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.models.flare import FLAREv2
    >>> model = FLAREv2(functional_dim=3, embedding_dim=3, out_dim=4, n_hidden=64)
    >>> fx, emb = torch.randn(2, 100, 3), torch.randn(2, 100, 3)
    >>> model(fx, embedding=emb).shape
    torch.Size([2, 100, 4])
    """

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
        qk_norm: bool = True,
        activation_checkpointing: bool = False,
        checkpointing_ratio: float = 1.0,
        use_te: bool = False,
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
            use_te=use_te,
            activation_checkpointing=activation_checkpointing,
            checkpointing_ratio=checkpointing_ratio,
        )

    def _build_mixer(self, **kwargs) -> nn.Module:
        return FLAREv2Attention(**kwargs, qk_norm=self._qk_norm)


class FLAREPlusPlus(_FLAREBackbone):
    r"""FLARE++ model: pre-norm transformer with FLARE++ attention mixers.

    Same backbone as :class:`FLAREv2`, with
    :class:`~physicsnemo.nn.FLAREPlusPlus` mixers, whose routing queries are
    synthesized from each block's input. See the `FLARE++ paper
    <https://arxiv.org/abs/2608.11519>`_.

    Parameters
    ----------
    functional_dim : int
        Number of functional input channels :math:`C_{fx}`.
    out_dim : int
        Number of output channels :math:`C_{out}`.
    embedding_dim : int | None, optional, default=None
        Number of embedding channels :math:`C_{emb}` (e.g. coordinates),
        concatenated in front of ``fx``. ``None`` means no embedding input.
    n_layers : int, optional, default=4
        Number of blocks.
    n_hidden : int, optional, default=256
        Hidden width :math:`C`.
    dropout : float, optional, default=0.0
        Dropout after each mixer's output projection.
    n_head : int, optional, default=8
        Number of attention heads; ``n_hidden`` must be divisible by it.
    act : str, optional, default="gelu_tanh"
        Activation of the input projection and block MLPs (any
        :func:`~physicsnemo.nn.get_activation` name). The default is
        tanh-approximate GELU.
    mlp_ratio : int, optional, default=4
        Block MLP hidden width is ``mlp_ratio * n_hidden``.
    slice_num : int, optional, default=32
        Number of latent (routing) queries per head.
    use_rmsnorm : bool, optional, default=True
        Use RMSNorm (``eps=1e-6``) for every norm (block, output and the
        mixers' per-head norms); ``False`` uses LayerNorm.
    use_te : bool, optional, default=False
        Use Transformer Engine layers (linear, norms, fused norm + MLP, fused
        norm + linear head, ``DotProductAttention``; CUDA only). With
        ``use_te=True``, ``act`` must be one of ``"gelu_tanh"``, ``"gelu"``
        (both tanh-approximate GELU in Transformer Engine), ``"relu"`` or
        ``"silu"``.
    activation_checkpointing : bool, optional, default=False
        Checkpoint blocks during training.
    checkpointing_ratio : float, optional, default=1.0
        Fraction of blocks to checkpoint, spread evenly across the stack, when
        ``activation_checkpointing=True``.
    k_norm : bool, optional, default=True
        Normalize routing keys per head.
    share_k0_v0 : bool, optional, default=True
        Share keys and values of the query-synthesis pass.
    q_fixed_norm : bool, optional, default=True
        Normalize the fixed anchor queries.
    gate_logit_init : float, optional, default=0.25
        Initial per-head gate logit of the dynamic queries.

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
    parameters (latent queries, gate logits) to train without weight decay.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.models.flare import FLAREPlusPlus
    >>> model = FLAREPlusPlus(functional_dim=3, out_dim=4, n_hidden=64)
    >>> model(torch.randn(2, 100, 3)).shape
    torch.Size([2, 100, 4])
    """

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
        use_te: bool = False,
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
            use_te=use_te,
            activation_checkpointing=activation_checkpointing,
            checkpointing_ratio=checkpointing_ratio,
        )

    def _build_mixer(self, **kwargs) -> nn.Module:
        return FLAREPlusPlusAttention(**kwargs, **self._mixer_kwargs)
