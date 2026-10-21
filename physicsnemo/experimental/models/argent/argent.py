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

r"""ArGEnT: Arbitrary Geometry-encoded Transformer for operator learning.

Implements the cross-attention variant of ArGEnT (`Chen et al., 2026
<https://arxiv.org/abs/2602.11626>`_): a geometry-aware transformer trunk that
attends from query points to a geometry point cloud, optionally fused with a
DeepONet-style branch over non-geometric operator parameters.
"""

from __future__ import annotations

from typing import List, Literal, Optional

import torch
import torch.nn as nn
from jaxtyping import Float, Shaped

from physicsnemo.core import Module
from physicsnemo.core.meta import ModelMetaData
from physicsnemo.nn import get_activation

from .argent_attention import MultiLayerCrossAttention


class _FNN(nn.Module):
    r"""Feed-forward stack with ``activation`` after every linear except the last.

    Parameters
    ----------
    layers : list of int
        Layer widths ``[in, hidden_0, ..., out]``.
    activation : nn.Module
        Activation applied between layers.

    Forward
    -------
    x : torch.Tensor
        Input tensor of shape :math:`(\ldots, \text{layers}[0])`.

    Outputs
    -------
    torch.Tensor
        Output tensor of shape :math:`(\ldots, \text{layers}[-1])`.
    """

    def __init__(self, layers: List[int], activation: nn.Module) -> None:
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
    r"""Modified feed-forward network of `Wang, Teng & Perdikaris (2021)
    <https://epubs.siam.org/doi/abs/10.1137/20M1318043>`_.

    Each hidden layer output :math:`Z` gates two input projections:
    :math:`x \leftarrow Z \odot U + (1 - Z) \odot V`.

    Parameters
    ----------
    layers : list of int
        Layer widths ``[in, hidden_0, ..., out]``.
    activation : nn.Module
        Activation applied at every hidden layer and to :math:`U`, :math:`V`.

    Forward
    -------
    x : torch.Tensor
        Input tensor of shape :math:`(\ldots, \text{layers}[0])`.

    Outputs
    -------
    torch.Tensor
        Output tensor of shape :math:`(\ldots, \text{layers}[-1])`.
    """

    def __init__(self, layers: List[int], activation: nn.Module) -> None:
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
    r"""ArGEnT trunk network with cross-attention from query points to geometry.

    Query tokens (evaluation points) and key/value tokens (a geometry point
    cloud) are embedded by MLPs, passed through a stack of linear
    cross-attention layers with continuous-coordinate RoPE
    (:class:`~physicsnemo.experimental.models.argent.argent_attention.MultiLayerCrossAttention`),
    and projected to ``dim_out`` channels per query. The first ``dim_pos``
    channels of ``x_query`` and ``x_kv`` are the spatial coordinates used by
    RoPE; all channels are fed to the embedding MLPs.

    Parameters
    ----------
    dim_query : int, optional, default=2
        Number of input channels of ``x_query``.
    dim_kv : int, optional, default=2
        Number of input channels of ``x_kv``.
    dim_pos : int, optional, default=2
        Number of leading coordinate channels used by RoPE. Must not exceed
        ``dim_query`` or ``dim_kv``.
    dim_out : int, optional, default=1
        Number of output channels per query token.
    dim_emb : int, optional, default=128
        Embedding and per-head attention dimension.
    num_heads : int, optional, default=4
        Number of attention heads.
    depth : int, optional, default=2
        Number of cross-attention layers.
    max_freq : float, optional, default=64.0
        RoPE frequency scale. At layer :math:`d`, coordinates are multiplied by
        :math:`\text{max\_freq}^2 / 2^d` before the RoPE frequency schedule.
    use_mfnn : bool, optional, default=False
        If ``True``, use the modified FNN of Wang et al. (2021) for the
        embedding and output MLPs instead of a plain feed-forward stack.
    activation : str, optional, default="relu"
        Activation of the embedding and output MLPs; any name accepted by
        :func:`~physicsnemo.nn.get_activation`.
    use_relu_mlp : bool, optional, default=False
        If ``True``, use ReLU feed-forward blocks in the attention stack;
        otherwise GEGLU.
    axis_slices : list of int, optional
        Channels of each attention head assigned to each coordinate axis. Must
        have ``dim_pos`` even entries summing to at most ``dim_emb``. Defaults
        to ``[dim_emb // dim_pos] * dim_pos``.
    attention_type : {"cross", "self", "hybrid"}, optional, default="cross"
        ArGEnT variant. Only ``"cross"`` is currently supported; other values
        raise :class:`NotImplementedError`.

    Forward
    -------
    x_query : torch.Tensor
        Query tokens of shape :math:`(B, N_q, D_q)`.
    x_kv : torch.Tensor
        Key/value tokens of shape :math:`(B, N_{kv}, D_{kv})`.
    padding_kv : torch.Tensor, optional
        Boolean or float mask of shape :math:`(B, N_{kv})` or
        :math:`(B, N_{kv}, 1)`. Nonzero entries mark valid tokens; padded
        tokens are excluded from the attention.

    Outputs
    -------
    torch.Tensor
        Output tokens of shape :math:`(B, N_q, D_{out})`.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.experimental.models.argent import ArGEnT
    >>> model = ArGEnT(dim_query=3, dim_kv=3, dim_pos=2,
    ...                dim_out=3, dim_emb=64, num_heads=2, depth=2)
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
        use_mfnn: bool = False,
        activation: str = "relu",
        use_relu_mlp: bool = False,
        axis_slices: Optional[List[int]] = None,
        attention_type: Literal["cross", "self", "hybrid"] = "cross",
    ) -> None:
        super().__init__(meta=ArGEnTMetaData())
        attention_type = attention_type.lower()
        if attention_type != "cross":
            raise NotImplementedError(
                f"attention_type={attention_type!r} is not supported. "
                "Only attention_type='cross' is currently supported."
            )
        if dim_pos > min(dim_query, dim_kv):
            raise ValueError(
                f"Expected dim_pos <= min(dim_query, dim_kv), got dim_pos={dim_pos}, "
                f"dim_query={dim_query}, dim_kv={dim_kv}"
            )
        if axis_slices is None:
            axis_slices = [dim_emb // dim_pos] * dim_pos
        if len(axis_slices) != dim_pos:
            raise ValueError(
                f"Expected axis_slices with {dim_pos} entries (dim_pos), "
                f"got {list(axis_slices)}"
            )

        self.dim_query = dim_query
        self.dim_kv = dim_kv
        self.dim_pos = dim_pos
        self.dim_out = dim_out
        self.dim_emb = dim_emb
        self.num_heads = num_heads
        self.depth = depth
        self.max_freq = float(max_freq)
        self.use_mfnn = use_mfnn
        self.use_relu_mlp = use_relu_mlp
        self.axis_slices = list(axis_slices)
        self.attention_type = attention_type

        self.activation = get_activation(activation)

        mlp_cls = _MFNN if use_mfnn else _FNN
        self.query_to_embedding = mlp_cls(
            layers=[dim_query] + [dim_emb] * 4, activation=self.activation
        )
        self.kv_to_embedding = mlp_cls(
            layers=[dim_kv] + [dim_emb] * 4, activation=self.activation
        )
        self.to_out = mlp_cls(
            layers=[dim_emb] * 4 + [dim_out], activation=self.activation
        )

        self.transformer = MultiLayerCrossAttention(
            dim_inout=dim_emb,
            depth=depth,
            num_heads=num_heads,
            dim_head=dim_emb,
            dim_mlp=dim_emb,
            max_freq=self.max_freq,
            axis_slices=self.axis_slices,
            use_relu_mlp=use_relu_mlp,
        )

    def forward(
        self,
        x_query: Float[torch.Tensor, "batch nq dim_query"],
        x_kv: Float[torch.Tensor, "batch nkv dim_kv"],
        padding_kv: Optional[Shaped[torch.Tensor, "batch nkv ..."]] = None,
    ) -> Float[torch.Tensor, "batch nq dim_out"]:
        ### Input validation
        if not torch.compiler.is_compiling():
            if x_query.ndim != 3 or x_query.shape[-1] != self.dim_query:
                raise ValueError(
                    f"Expected x_query of shape (B, N_q, {self.dim_query}) "
                    f"but got tensor of shape {tuple(x_query.shape)}"
                )
            B = x_query.shape[0]
            if x_kv.ndim != 3 or x_kv.shape[0] != B or x_kv.shape[-1] != self.dim_kv:
                raise ValueError(
                    f"Expected x_kv of shape ({B}, N_kv, {self.dim_kv}) "
                    f"but got tensor of shape {tuple(x_kv.shape)}"
                )
            if padding_kv is not None and padding_kv.numel() != B * x_kv.shape[1]:
                raise ValueError(
                    f"Expected padding_kv of shape ({B}, {x_kv.shape[1]}) or "
                    f"({B}, {x_kv.shape[1]}, 1) but got tensor of shape "
                    f"{tuple(padding_kv.shape)}"
                )

        pos_query = x_query[..., : self.dim_pos]
        pos_kv = x_kv[..., : self.dim_pos]

        # The activation is also applied after the last embedding layer.
        x_kv_emb = self.activation(self.kv_to_embedding(x_kv))
        x_query_emb = self.activation(self.query_to_embedding(x_query))

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

    The :class:`ArGEnT` trunk produces ``dim_emb`` basis features per output
    channel at each query point, and a feed-forward branch maps the operator
    parameters :math:`\mu` (e.g. a Reynolds number) to matching coefficients:

    .. math::

        u_k(x) = \sum_{j=1}^{\text{dim\_emb}} c_{jk}(\mu) \, f_{jk}(x, \Omega)

    where :math:`f` is the trunk output for query point :math:`x` and geometry
    :math:`\Omega`. With ``dim_op < 1`` the branch is disabled and the model
    reduces to :class:`ArGEnT`.

    Parameters
    ----------
    dim_op : int, optional, default=1
        Number of operator parameters. Values below 1 disable the branch.
    dim_query : int, optional, default=2
        Number of input channels of ``x_query``.
    dim_kv : int, optional, default=2
        Number of input channels of ``x_kv``.
    dim_pos : int, optional, default=2
        Number of leading coordinate channels used by RoPE.
    dim_out : int, optional, default=1
        Number of output channels per query token.
    dim_emb : int, optional, default=128
        Embedding dimension; also the number of basis functions per output
        channel.
    num_heads : int, optional, default=4
        Number of attention heads.
    depth : int, optional, default=2
        Number of cross-attention layers.
    max_freq : float, optional, default=64.0
        RoPE frequency scale; see :class:`ArGEnT`.
    use_mfnn : bool, optional, default=False
        If ``True``, use the modified FNN for the trunk MLPs and the branch.
    activation : str, optional, default="relu"
        Activation of the trunk MLPs and the branch; see :class:`ArGEnT`.
    use_relu_mlp : bool, optional, default=False
        If ``True``, use ReLU feed-forward blocks in the attention stack.
    axis_slices : list of int, optional
        Per-axis RoPE channel split; see :class:`ArGEnT`.
    attention_type : {"cross", "self", "hybrid"}, optional, default="cross"
        ArGEnT variant; see :class:`ArGEnT`.

    Forward
    -------
    x_query : torch.Tensor
        Query tokens of shape :math:`(B, N_q, D_q)`.
    x_kv : torch.Tensor
        Key/value tokens of shape :math:`(B, N_{kv}, D_{kv})`.
    op_params : torch.Tensor, optional
        Operator parameters of shape :math:`(B, 1, D_{op})`. Required when
        ``dim_op >= 1``; ignored otherwise.
    padding_kv : torch.Tensor, optional
        Key/value padding mask; see :class:`ArGEnT`.

    Outputs
    -------
    torch.Tensor
        Predicted fields at the query points, of shape :math:`(B, N_q, D_{out})`.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.experimental.models.argent import ArGEnTDeepONet
    >>> model = ArGEnTDeepONet(dim_op=1, dim_query=3, dim_kv=3, dim_pos=2,
    ...                        dim_out=3, dim_emb=64, num_heads=2, depth=2)
    >>> x_query = torch.randn(2, 500, 3)
    >>> x_kv = torch.randn(2, 200, 3)
    >>> re = torch.randn(2, 1, 1)
    >>> out = model(x_query, x_kv, re)
    >>> out.shape
    torch.Size([2, 500, 3])
    """

    def __init__(
        self,
        dim_op: int = 1,
        dim_query: int = 2,
        dim_kv: int = 2,
        dim_pos: int = 2,
        dim_out: int = 1,
        dim_emb: int = 128,
        num_heads: int = 4,
        depth: int = 2,
        max_freq: float = 64.0,
        use_mfnn: bool = False,
        activation: str = "relu",
        use_relu_mlp: bool = False,
        axis_slices: Optional[List[int]] = None,
        attention_type: Literal["cross", "self", "hybrid"] = "cross",
    ) -> None:
        super().__init__(meta=ArGEnTDeepONetMetaData())
        self.dim_op = max(dim_op, 0)
        self.dim_out = dim_out
        self.dim_emb = dim_emb

        # With a branch, the trunk emits dim_emb basis features per output channel.
        trunk_dim_out = dim_out * dim_emb if self.dim_op > 0 else dim_out
        self.argent = ArGEnT(
            dim_query=dim_query,
            dim_kv=dim_kv,
            dim_pos=dim_pos,
            dim_out=trunk_dim_out,
            dim_emb=dim_emb,
            num_heads=num_heads,
            depth=depth,
            max_freq=max_freq,
            use_mfnn=use_mfnn,
            activation=activation,
            use_relu_mlp=use_relu_mlp,
            axis_slices=axis_slices,
            attention_type=attention_type,
        )

        if self.dim_op > 0:
            mlp_cls = _MFNN if use_mfnn else _FNN
            self.branch_net = mlp_cls(
                layers=[self.dim_op] + [dim_emb] * 5 + [dim_out * dim_emb],
                activation=get_activation(activation),
            )

    def forward(
        self,
        x_query: Float[torch.Tensor, "batch nq dim_query"],
        x_kv: Float[torch.Tensor, "batch nkv dim_kv"],
        op_params: Optional[Float[torch.Tensor, "batch 1 dim_op"]] = None,
        padding_kv: Optional[Shaped[torch.Tensor, "batch nkv ..."]] = None,
    ) -> Float[torch.Tensor, "batch nq dim_out"]:
        ### Input validation (x_query, x_kv, padding_kv are checked by the trunk)
        if not torch.compiler.is_compiling() and self.dim_op > 0:
            B = x_query.shape[0]
            if op_params is None or op_params.shape != (B, 1, self.dim_op):
                raise ValueError(
                    f"Expected op_params of shape ({B}, 1, {self.dim_op}) but got "
                    f"{None if op_params is None else tuple(op_params.shape)}"
                )

        trunk_out = self.argent(x_query, x_kv, padding_kv=padding_kv)
        if self.dim_op == 0:
            return trunk_out

        # Multiplicative fusion: (B, Nq, dim_emb*dim_out) * (B, 1, dim_emb*dim_out),
        # then sum over the dim_emb basis functions of each output channel.
        branch_out = self.branch_net(op_params)
        fused = (trunk_out * branch_out).view(
            trunk_out.shape[0], trunk_out.shape[1], -1, self.dim_out
        )
        return fused.sum(dim=2)
