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

r"""ArGEnT — Arbitrary Geometry-encoded Transformer for Operator Learning.

Port of the cross-attention variant of ArGEnT (Chen et al. 2026;
`ArGEnT: Arbitrary Geometry-encoded Transformer for Operator Learning
<https://arxiv.org/abs/2602.11626>`_) into the PhysicsNeMo framework. The
model composes a geometry-aware transformer trunk with a DeepONet-style branch
over non-geometric operator parameters and predicts field values at arbitrary
query points inside the domain.

Only the cross-attention variant is implemented in v1. Passing
``attention_type='self'`` or ``attention_type='hybrid'`` to :class:`ArGEnT`
or :class:`ArGEnTDeepONet` raises :class:`NotImplementedError`; a future
implementer only needs to fill in the corresponding branch of
:meth:`ArGEnT.__init__` and :meth:`ArGEnT.forward` without renaming or
changing the public API.
"""

from __future__ import annotations

from typing import Callable, List, Literal, Optional, Union

import torch
import torch.nn as nn
from jaxtyping import Float

from physicsnemo.core import Module
from physicsnemo.core.meta import ModelMetaData

from .argent_attention import MultiLayerCrossAttention

# Activation dictionary — mirrors argent_exp/ArGEnT/nn_module/basis.py:7-15
# so the same string identifiers ("relu", "gelu", etc.) work interchangeably.
_ACTIVATION_DICT: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "relu": torch.relu,
    "selu": torch.selu,
    "gelu": nn.GELU(),
    "sigmoid": torch.sigmoid,
    "sin": torch.sin,
    "swish": lambda x: x * torch.sigmoid(x),
    "tanh": torch.tanh,
}


def _resolve_activation(
    activation: Union[str, Callable[[torch.Tensor], torch.Tensor]],
) -> Callable[[torch.Tensor], torch.Tensor]:
    r"""Resolve an activation identifier (string or callable) to a callable.

    Parameters
    ----------
    activation : str or callable
        Either a key of the internal activation dictionary (``"relu"``,
        ``"gelu"``, ``"selu"``, ``"sigmoid"``, ``"sin"``, ``"swish"``,
        ``"tanh"``) or a direct callable.

    Returns
    -------
    Callable[[torch.Tensor], torch.Tensor]
        The activation function.
    """
    if callable(activation):
        return activation
    if isinstance(activation, str) and activation.lower() in _ACTIVATION_DICT:
        return _ACTIVATION_DICT[activation.lower()]
    raise ValueError(f"Unsupported activation function: {activation!r}")


class _FNN(nn.Module):
    r"""Simple feed-forward stack used for embeddings and the DeepONet branch.

    Faithful port of argent_exp's ``FNN`` ([basis.py:25-73]). Kept as an
    internal helper (leading underscore) so this file stays self-contained
    per **MOD-004** — physicsnemo's :class:`~physicsnemo.nn.FCLayer` uses a
    different default init (Xavier uniform vs PyTorch's default Kaiming
    uniform in ``nn.Linear``) which would silently perturb trained-model
    reproducibility, so we don't compose it here.

    Applies ``activation`` after every linear except the last.

    Parameters
    ----------
    layers : list of int
        Sequence of layer widths ``[in, hidden_0, hidden_1, ..., out]``.
    activation : callable
        Activation function applied between hidden layers.

    Forward
    -------
    x : torch.Tensor
        Input tensor of shape :math:`(\ldots, \text{layers}[0])`.

    Outputs
    -------
    torch.Tensor
        Output tensor of shape :math:`(\ldots, \text{layers}[-1])`.
    """

    def __init__(
        self,
        layers: List[int],
        activation: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        super().__init__()
        self.activation = activation
        self.net = nn.ModuleList(
            [nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for linear in self.net[:-1]:
            x = self.activation(linear(x))
        return self.net[-1](x)


class _MFNN(nn.Module):
    r"""Modified feed-forward network from Wang et al. (2021).

    Faithful port of argent_exp's ``MFNN`` ([basis.py:75-129]). Selected when
    the argent constructor is called with ``use_MFNN=True``. Interpolates
    between two side-branch projections ``U`` and ``V`` using the main-branch
    activations ``Z_l`` at each hidden layer, following
    `Wang, Teng & Perdikaris (2021)
    <https://epubs.siam.org/doi/abs/10.1137/20M1318043>`_ (SIAM SISC).

    Parameters
    ----------
    layers : list of int
        Sequence of layer widths ``[in, hidden_0, hidden_1, ..., out]``.
    activation : callable
        Activation function applied at every hidden layer.

    Forward
    -------
    x : torch.Tensor
        Input tensor of shape :math:`(\ldots, \text{layers}[0])`.

    Outputs
    -------
    torch.Tensor
        Output tensor of shape :math:`(\ldots, \text{layers}[-1])`.
    """

    def __init__(
        self,
        layers: List[int],
        activation: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        super().__init__()
        self.activation = activation
        self.Znet = nn.ModuleList(
            [nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)]
        )
        self.Unet = nn.Linear(layers[0], layers[1])
        self.Vnet = nn.Linear(layers[0], layers[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        U = self.activation(self.Unet(x))
        V = self.activation(self.Vnet(x))
        for linear in self.Znet[:-1]:
            Z = self.activation(linear(x))
            x = Z * U + (1.0 - Z) * V
        return self.Znet[-1](x)


class ArGEnTMetaData(ModelMetaData):
    r"""Meta descriptor for :class:`ArGEnT`."""

    name: str = "ArGEnT"
    var_dim: int = -1
    experimental: bool = True


class ArGEnTDeepONetMetaData(ModelMetaData):
    r"""Meta descriptor for :class:`ArGEnTDeepONet`."""

    name: str = "ArGEnTDeepONet"
    var_dim: int = -1
    experimental: bool = True


class ArGEnT(Module):
    r"""Cross-attention ArGEnT trunk network (no branch).

    Faithful port of argent_exp's ``ArGEnT`` ([ArGEnT.py:10-83]) with
    ``attention_type='cross'``. Given query tokens (evaluation points) and
    key/value tokens (a geometry point cloud), produces per-query output
    features via a stack of linear cross-attention layers with per-token
    continuous-coordinate RoPE.

    The first ``dim_pos`` channels of ``x_query`` and ``x_kv`` are interpreted
    as spatial coordinates and consumed by the RoPE inside the attention
    (they still participate in the embedding MLP through the full input
    channel count).

    Parameters
    ----------
    dim_query : int, optional, default=2
        Number of input channels of ``x_query``.
    dim_kv : int, optional, default=2
        Number of input channels of ``x_kv``.
    dim_pos : int, optional, default=2
        Number of leading input channels used as continuous coordinates for
        RoPE. Both ``x_query`` and ``x_kv`` are expected to have at least
        ``dim_pos`` channels.
    dim_out : int, optional, default=1
        Number of output channels per query token.
    dim_emb : int, optional, default=128
        Feature dimension used throughout the trunk (embeddings and attention
        head sum).
    num_heads : int, optional, default=4
        Number of attention heads inside each attention layer.
    depth : int, optional, default=2
        Number of stacked cross-attention layers.
    max_freq : float, optional, default=64.0
        Base RoPE frequency. Layer ``d`` uses ``scale = max_freq / 2**d``.
    use_MFNN : bool, optional, default=False
        If ``True``, use the modified FNN of Wang et al. (2021) for the
        query/KV embeddings and output projection; if ``False``, use a plain
        stack of ``Linear`` + activation.
    activation : str or callable, optional, default="relu"
        Activation function for the embedding and output MLPs. String
        identifiers accepted: ``"relu"``, ``"selu"``, ``"gelu"``,
        ``"sigmoid"``, ``"sin"``, ``"swish"``, ``"tanh"``.
    use_relu_mlp : bool, optional, default=False
        If ``True``, use ReLU inside the feed-forward blocks of the attention
        stack; if ``False``, use GEGLU.
    axis_slices : list of int, optional
        Per-axis channel slice within ``dim_head`` (= ``dim_emb``); each entry
        must be even. If ``None``, defaults to
        ``[dim_emb // dim_pos] * dim_pos`` (matches argent_exp
        [ArGEnT.py:44-45]).
    attention_type : {"cross", "self", "hybrid"}, optional, default="cross"
        Which ArGEnT variant to build. Only ``"cross"`` is implemented in
        v1; ``"self"`` and ``"hybrid"`` raise :class:`NotImplementedError`.

    Forward
    -------
    x_query : torch.Tensor
        Query tokens of shape :math:`(B, N_q, \text{dim\_query})`. The first
        ``dim_pos`` channels are the query coordinates.
    x_kv : torch.Tensor
        Key/value tokens of shape :math:`(B, N_{kv}, \text{dim\_kv})`. The
        first ``dim_pos`` channels are the KV coordinates.
    padding_kv : torch.Tensor, optional
        Mask of shape :math:`(B, N_{kv})` or :math:`(B, N_{kv}, 1)`; ``True``
        (or nonzero) keeps the token, ``False`` (or zero) excludes it from
        :math:`K^\top V` and the normalizer.

    Outputs
    -------
    torch.Tensor
        Output tokens of shape :math:`(B, N_q, \text{dim\_out})`.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.experimental.models.argent import ArGEnT
    >>> model = ArGEnT(dim_query=3, dim_kv=3, dim_pos=2,
    ...                     dim_out=3, dim_emb=64, num_heads=2, depth=2)
    >>> x_query = torch.randn(2, 500, 3)
    >>> x_kv = torch.randn(2, 200, 3)
    >>> out = model(x_query, x_kv)
    >>> out.shape
    torch.Size([2, 500, 3])
    """

    def __init__(
        self,
        dim_query: int = 2,
        dim_kv: int = 2,
        dim_pos: int = 2,
        dim_out: int = 1,
        dim_emb: int = 128,
        num_heads: int = 4,
        depth: int = 2,
        max_freq: float = 64.0,
        use_MFNN: bool = False,
        activation: Union[str, Callable[[torch.Tensor], torch.Tensor]] = "relu",
        use_relu_mlp: bool = False,
        axis_slices: Optional[List[int]] = None,
        attention_type: Literal["cross", "self", "hybrid"] = "cross",
    ) -> None:
        super().__init__(meta=ArGEnTMetaData())
        attention_type = attention_type.lower()
        if attention_type != "cross":
            raise NotImplementedError(
                f"attention_type={attention_type!r} is not yet implemented in the "
                f"PhysicsNeMo port. Only 'cross' is supported in v1."
            )
        self.dim_query = int(dim_query)
        self.dim_kv = int(dim_kv)
        self.dim_pos = int(dim_pos)
        self.dim_out = int(dim_out)
        self.dim_emb = int(dim_emb)
        self.num_heads = int(num_heads)
        self.depth = int(depth)
        self.max_freq = float(max_freq)
        self.use_MFNN = bool(use_MFNN)
        self.use_relu_mlp = bool(use_relu_mlp)
        self.attention_type = attention_type

        self.activation = _resolve_activation(activation)

        # Argent default: equal-sized per-axis slices of head_dim across dim_pos axes.
        if axis_slices is None:
            axis_slices = [self.dim_emb // self.dim_pos] * self.dim_pos
        self.axis_slices = list(axis_slices)

        # Embedding stacks: match argent_exp/ArGEnT/nn_module/ArGEnT.py:41-43.
        mlp_cls = _MFNN if self.use_MFNN else _FNN
        self.query_to_embedding = mlp_cls(
            layers=[self.dim_query] + [self.dim_emb] * 4,
            activation=self.activation,
        )
        self.kv_to_embedding = mlp_cls(
            layers=[self.dim_kv] + [self.dim_emb] * 4,
            activation=self.activation,
        )
        self.to_out = mlp_cls(
            layers=[self.dim_emb] * 4 + [self.dim_out],
            activation=self.activation,
        )

        self.transformer = MultiLayerCrossAttention(
            dim_inout=self.dim_emb,
            depth=self.depth,
            num_heads=self.num_heads,
            dim_head=self.dim_emb,
            dim_mlp=self.dim_emb,
            max_freq=self.max_freq,
            axis_slices=self.axis_slices,
            use_relu_mlp=self.use_relu_mlp,
        )

    def forward(
        self,
        x_query: Float[torch.Tensor, "batch nq dim_query"],
        x_kv: Float[torch.Tensor, "batch nkv dim_kv"],
        padding_kv: Optional[torch.Tensor] = None,
    ) -> Float[torch.Tensor, "batch nq dim_out"]:
        # Input validation, skipped under torch.compile per MOD-005.
        if not torch.compiler.is_compiling():
            if x_query.ndim != 3 or x_query.shape[-1] != self.dim_query:
                raise ValueError(
                    f"Expected x_query shape (B, Nq, {self.dim_query}), "
                    f"got {tuple(x_query.shape)}"
                )
            if x_kv.ndim != 3 or x_kv.shape[-1] != self.dim_kv:
                raise ValueError(
                    f"Expected x_kv shape (B, Nkv, {self.dim_kv}), "
                    f"got {tuple(x_kv.shape)}"
                )
            if x_query.shape[-1] < self.dim_pos or x_kv.shape[-1] < self.dim_pos:
                raise ValueError(
                    f"Both x_query and x_kv must have at least dim_pos={self.dim_pos} "
                    f"channels; got x_query dim {x_query.shape[-1]}, x_kv dim "
                    f"{x_kv.shape[-1]}"
                )

        # First dim_pos channels serve as continuous coordinates for RoPE
        # (matches argent_exp/ArGEnT/nn_module/ArGEnT.py:72-73).
        pos_query = x_query[..., : self.dim_pos]
        pos_kv = x_kv[..., : self.dim_pos]

        # Embed query and KV — the activation is applied AFTER the last linear
        # of each embedding stack (matches argent_exp/ArGEnT.py:75-76).
        x_kv_emb = self.activation(self.kv_to_embedding(x_kv))
        x_query_emb = self.activation(self.query_to_embedding(x_query))

        # Cross-attention transformer.
        x_query_emb = self.transformer(
            x_query_emb,
            x_kv_emb,
            pos_q=pos_query,
            pos_kv=pos_kv,
            padding_kv=padding_kv,
        )

        return self.to_out(x_query_emb)


class ArGEnTDeepONet(Module):
    r"""ArGEnT trunk fused with a DeepONet-style branch over operator parameters.

    Faithful port of argent_exp's ``ArGEnT_DeepONet``
    ([ArGEnT_DeepONet.py:8-109]) restricted to ``attention_type='cross'``.
    Wraps :class:`ArGEnT` as the trunk and a small FNN as the branch
    over non-geometric operator parameters (e.g. Reynolds number for the
    lid-driven cavity example). Fuses trunk and branch multiplicatively:

    .. math::

        \text{out}(x) = \sum_{j=1}^{J} c_j(\mu) \, f_j(x, \Omega)

    where :math:`c_j(\mu)` is the branch output for operator parameter
    :math:`\mu`, :math:`f_j(x, \Omega)` is the trunk output for query point
    :math:`x` given geometry :math:`\Omega`, and the sum runs over the
    ``dim_emb`` inner dimension per output channel.

    When ``dim_Op < 1`` the branch is disabled and the model degenerates to
    :class:`ArGEnT` (matches argent_exp's behavior at
    ArGEnT_DeepONet.py:39-40 and forward at line 104-105).

    Parameters
    ----------
    dim_Op : int, optional, default=1
        Dimension of the operator-parameter input. If ``<1``, the branch net
        is skipped entirely and the model reduces to the trunk alone.
    dim_query : int, optional, default=2
        See :class:`ArGEnT`.
    dim_kv : int, optional, default=2
        See :class:`ArGEnT`.
    dim_pos : int, optional, default=2
        See :class:`ArGEnT`.
    dim_out : int, optional, default=1
        Number of output channels per query token.
    dim_emb : int, optional, default=128
        See :class:`ArGEnT`.
    num_heads : int, optional, default=4
        See :class:`ArGEnT`.
    depth : int, optional, default=2
        See :class:`ArGEnT`.
    max_freq : float, optional, default=64.0
        See :class:`ArGEnT`.
    use_MFNN : bool, optional, default=False
        See :class:`ArGEnT`. Also applied to the branch FNN.
    activation : str or callable, optional, default="relu"
        See :class:`ArGEnT`.
    use_relu_mlp : bool, optional, default=False
        See :class:`ArGEnT`.
    axis_slices : list of int, optional
        See :class:`ArGEnT`.
    attention_type : {"cross", "self", "hybrid"}, optional, default="cross"
        See :class:`ArGEnT`.

    Forward
    -------
    x_query : torch.Tensor
        Query tokens of shape :math:`(B, N_q, \text{dim\_query})`.
    x_kv : torch.Tensor
        Key/value tokens of shape :math:`(B, N_{kv}, \text{dim\_kv})`.
    OPParas : torch.Tensor, optional
        Operator parameters of shape :math:`(B, 1, \text{dim\_Op})`. Required
        when ``dim_Op >= 1``; ignored otherwise.
    padding_kv : torch.Tensor, optional
        KV padding mask; see :class:`ArGEnT`.

    Outputs
    -------
    torch.Tensor
        Predicted field values at query points, of shape
        :math:`(B, N_q, \text{dim\_out})`.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.experimental.models.argent import ArGEnTDeepONet
    >>> model = ArGEnTDeepONet(dim_Op=1, dim_query=3, dim_kv=3, dim_pos=2,
    ...                        dim_out=3, dim_emb=64, num_heads=2, depth=2)
    >>> x_query = torch.randn(2, 500, 3)
    >>> x_kv = torch.randn(2, 200, 3)
    >>> Re = torch.randn(2, 1, 1)
    >>> out = model(x_query, x_kv, Re)
    >>> out.shape
    torch.Size([2, 500, 3])
    """

    def __init__(
        self,
        dim_Op: int = 1,
        dim_query: int = 2,
        dim_kv: int = 2,
        dim_pos: int = 2,
        dim_out: int = 1,
        dim_emb: int = 128,
        num_heads: int = 4,
        depth: int = 2,
        max_freq: float = 64.0,
        use_MFNN: bool = False,
        activation: Union[str, Callable[[torch.Tensor], torch.Tensor]] = "relu",
        use_relu_mlp: bool = False,
        axis_slices: Optional[List[int]] = None,
        attention_type: Literal["cross", "self", "hybrid"] = "cross",
    ) -> None:
        super().__init__(meta=ArGEnTDeepONetMetaData())
        # If dim_Op < 1, disable the branch (matches ArGEnT_DeepONet.py:39-40).
        self.dim_Op = int(dim_Op) if int(dim_Op) >= 1 else 0
        self.dim_out = int(dim_out)
        self.dim_emb = int(dim_emb)

        # Trunk: when a branch exists, its output feature dim is dim_out * dim_emb
        # so the multiplicative fusion has enough channels; otherwise dim_out.
        trunk_dim_out = self.dim_out * self.dim_emb if self.dim_Op > 0 else self.dim_out
        self.argent = ArGEnT(
            dim_query=dim_query,
            dim_kv=dim_kv,
            dim_pos=dim_pos,
            dim_out=trunk_dim_out,
            dim_emb=dim_emb,
            num_heads=num_heads,
            depth=depth,
            max_freq=max_freq,
            use_MFNN=use_MFNN,
            activation=activation,
            use_relu_mlp=use_relu_mlp,
            axis_slices=axis_slices,
            attention_type=attention_type,
        )

        if self.dim_Op > 0:
            act = _resolve_activation(activation)
            mlp_cls = _MFNN if bool(use_MFNN) else _FNN
            # Branch layer sequence — matches ArGEnT_DeepONet.py:90.
            self.branch_net = mlp_cls(
                layers=[self.dim_Op] + [self.dim_emb] * 5 + [self.dim_out * self.dim_emb],
                activation=act,
            )

    def forward(
        self,
        x_query: Float[torch.Tensor, "batch nq dim_query"],
        x_kv: Float[torch.Tensor, "batch nkv dim_kv"],
        OPParas: Optional[Float[torch.Tensor, "batch 1 dim_op"]] = None,
        padding_kv: Optional[torch.Tensor] = None,
    ) -> Float[torch.Tensor, "batch nq dim_out"]:
        # Input validation (skipped under torch.compile per MOD-005).
        if not torch.compiler.is_compiling():
            if self.dim_Op > 0:
                if OPParas is None:
                    raise ValueError(
                        f"OPParas is required when dim_Op={self.dim_Op}>0"
                    )
                if (
                    OPParas.ndim != 3
                    or OPParas.shape[-1] != self.dim_Op
                    or OPParas.shape[-2] != 1
                ):
                    raise ValueError(
                        f"Expected OPParas shape (B, 1, {self.dim_Op}), "
                        f"got {tuple(OPParas.shape)}"
                    )

        trunk_out = self.argent(x_query, x_kv, padding_kv=padding_kv)

        if self.dim_Op == 0:
            # Branch is disabled; trunk_out is (B, Nq, dim_out) already.
            return trunk_out

        # Multiplicative fusion (matches ArGEnT_DeepONet.py:106-108):
        #   trunk (B, Nq, dim_out*dim_emb) * branch (B, 1, dim_out*dim_emb)
        #   -> reshape to (B, Nq, dim_emb, dim_out) and sum along dim_emb.
        branch_out = self.branch_net(OPParas)  # (B, 1, dim_out * dim_emb)
        fused = trunk_out * branch_out  # (B, Nq, dim_out * dim_emb)
        fused = fused.view(
            trunk_out.shape[0], trunk_out.shape[1], -1, self.dim_out
        )  # (B, Nq, dim_emb, dim_out)
        return fused.sum(dim=2)  # (B, Nq, dim_out)


