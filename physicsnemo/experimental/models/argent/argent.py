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

from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch.nn as nn
from jaxtyping import Float, Shaped

from physicsnemo.core import Module
from physicsnemo.core.meta import ModelMetaData
from physicsnemo.nn import Mlp, get_activation

from .argent_attention import _MultiLayerCrossAttention


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

    def __init__(self, layers: list[int], activation: nn.Module) -> None:
        super().__init__()
        self.activation = activation
        self.Znet = nn.ModuleList(
            [nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)]
        )
        self.Unet = nn.Linear(layers[0], layers[1])
        self.Vnet = nn.Linear(layers[0], layers[1])

    def forward(
        self, x: Float[torch.Tensor, "... dim_in"]
    ) -> Float[torch.Tensor, "... dim_out"]:
        U = self.activation(self.Unet(x))
        V = self.activation(self.Vnet(x))
        for linear in self.Znet[:-1]:
            Z = self.activation(linear(x))
            x = Z * U + (1.0 - Z) * V
        return self.Znet[-1](x)


def _make_mlp(
    dim_in: int, hidden: list[int], dim_out: int, activation: nn.Module, mfnn: bool
) -> nn.Module:
    r"""Build a plain :class:`~physicsnemo.nn.Mlp` or a :class:`_MFNN`."""
    if mfnn:
        return _MFNN([dim_in, *hidden, dim_out], activation=activation)
    return Mlp(dim_in, hidden, dim_out, act_layer=activation)


class ArGEnTGeometry(NamedTuple):
    r"""Encoded geometry returned by :meth:`ArGEnT.encode_geometry`.

    Attributes
    ----------
    tokens : torch.Tensor
        Embedded geometry tokens of shape :math:`(B, N_{kv}, \text{dim\_emb})`.
    pos : torch.Tensor
        Geometry coordinates used by RoPE, of shape
        :math:`(B, N_{kv}, \text{dim\_pos})`.
    padding : torch.Tensor or None
        The key/value padding mask passed to :meth:`ArGEnT.encode_geometry`.
    """

    tokens: torch.Tensor
    pos: torch.Tensor
    padding: torch.Tensor | None


@dataclass
class ArGEnTMetaData(ModelMetaData):
    r"""Meta descriptor for :class:`ArGEnT`."""

    amp: bool = True
    bf16: bool = True


@dataclass
class ArGEnTDeepONetMetaData(ModelMetaData):
    r"""Meta descriptor for :class:`ArGEnTDeepONet`."""

    amp: bool = True
    bf16: bool = True


class ArGEnT(Module):
    r"""ArGEnT trunk network with cross-attention from query points to geometry.

    This is the cross-attention variant of ArGEnT. Query tokens (evaluation
    points) and key/value tokens (a geometry point cloud) are embedded by
    MLPs, passed through a stack of linear cross-attention layers with
    continuous-coordinate RoPE, and projected to ``dim_out`` channels per
    query. The first ``dim_pos`` channels of ``x_query`` and ``x_kv`` are the
    spatial coordinates used by RoPE. All channels are fed to the embedding
    MLPs.

    The geometry can be encoded once with :meth:`encode_geometry` and reused
    across query chunks with :meth:`decode_queries`. :meth:`forward` runs both.

    Parameters
    ----------
    dim_query : int, optional, default=2
        Number of input channels of ``x_query``.
    dim_kv : int, optional, default=2
        Number of input channels of ``x_kv``.
    dim_pos : int, optional, default=2
        Number of leading coordinate channels used by RoPE. Must be at least 1
        and must not exceed ``dim_query`` or ``dim_kv``.
    dim_out : int, optional, default=1
        Number of output channels per query token.
    dim_emb : int, optional, default=128
        Embedding dimension, also used as the width of each attention head.
        The total attention width is ``dim_emb * num_heads``. This matches the
        configuration used for the results in `Chen et al., 2026
        <https://arxiv.org/abs/2602.11626>`_.
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
        Activation of the embedding and output MLPs. Any name accepted by
        :func:`~physicsnemo.nn.get_activation`.
    use_relu_mlp : bool, optional, default=False
        If ``True``, use ReLU feed-forward blocks in the attention stack.
        Otherwise use GEGLU.
    axis_slices : list of int, optional
        Channels of each attention head assigned to each coordinate axis. Must
        have ``dim_pos`` positive even entries summing to at most ``dim_emb``.
        Remaining channels are not rotated. Defaults to
        ``[2 * (dim_emb // (2 * dim_pos))] * dim_pos``.

    Forward
    -------
    x_query : torch.Tensor
        Query tokens of shape :math:`(B, N_q, D_q)`.
    x_kv : torch.Tensor
        Key/value tokens of shape :math:`(B, N_{kv}, D_{kv})`.
    padding_kv : torch.Tensor, optional
        Boolean or float mask of shape :math:`(B, N_{kv})` or
        :math:`(B, N_{kv}, 1)`. Any nonzero entry marks a valid token. Padded
        tokens are excluded from the attention. Padded entries of ``x_kv``
        may hold any value, including NaN, because they are zeroed before
        embedding.

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

    Encode the geometry once and evaluate query chunks against it:

    >>> geometry = model.encode_geometry(x_kv)
    >>> chunks = [model.decode_queries(q, geometry) for q in x_query.split(250, dim=1)]
    >>> torch.allclose(torch.cat(chunks, dim=1), out, atol=1e-6)
    True
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
        axis_slices: list[int] | None = None,
    ) -> None:
        super().__init__(meta=ArGEnTMetaData())
        if dim_pos < 1:
            raise ValueError(f"Expected dim_pos >= 1, got dim_pos={dim_pos}")
        if dim_pos > min(dim_query, dim_kv):
            raise ValueError(
                f"Expected dim_pos <= min(dim_query, dim_kv), got dim_pos={dim_pos}, "
                f"dim_query={dim_query}, dim_kv={dim_kv}"
            )
        if axis_slices is None:
            axis_slices = [2 * (dim_emb // (2 * dim_pos))] * dim_pos
        if len(axis_slices) != dim_pos:
            raise ValueError(
                f"Expected axis_slices with {dim_pos} entries (dim_pos), "
                f"got {list(axis_slices)}"
            )
        if any(s < 1 for s in axis_slices):
            raise ValueError(
                f"Expected every entry of axis_slices to be positive, got "
                f"{list(axis_slices)}. The default split needs dim_emb >= "
                f"2 * dim_pos (got dim_emb={dim_emb}, dim_pos={dim_pos})."
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

        self.activation = get_activation(activation)

        hidden = [dim_emb] * 3
        self.query_to_embedding = _make_mlp(
            dim_query, hidden, dim_emb, self.activation, use_mfnn
        )
        self.kv_to_embedding = _make_mlp(
            dim_kv, hidden, dim_emb, self.activation, use_mfnn
        )
        self.to_out = _make_mlp(dim_emb, hidden, dim_out, self.activation, use_mfnn)

        self.transformer = _MultiLayerCrossAttention(
            dim_inout=dim_emb,
            depth=depth,
            num_heads=num_heads,
            dim_head=dim_emb,
            dim_mlp=dim_emb,
            max_freq=self.max_freq,
            axis_slices=self.axis_slices,
            use_relu_mlp=use_relu_mlp,
        )

    def encode_geometry(
        self,
        x_kv: Float[torch.Tensor, "batch nkv dim_kv"],
        padding_kv: Shaped[torch.Tensor, "batch nkv ..."] | None = None,
    ) -> ArGEnTGeometry:
        r"""Embed a geometry point cloud for reuse across query evaluations.

        Parameters
        ----------
        x_kv : torch.Tensor
            Key/value tokens of shape :math:`(B, N_{kv}, D_{kv})`.
        padding_kv : torch.Tensor, optional
            Key/value padding mask. See :class:`ArGEnT`.

        Returns
        -------
        ArGEnTGeometry
            Embedded geometry tokens, their coordinates and the padding mask.
        """
        ### Input validation
        if not torch.compiler.is_compiling():
            if x_kv.ndim != 3 or x_kv.shape[-1] != self.dim_kv:
                raise ValueError(
                    f"Expected x_kv of shape (B, N_kv, {self.dim_kv}) "
                    f"but got tensor of shape {tuple(x_kv.shape)}"
                )
            B, N_kv = x_kv.shape[:2]
            if padding_kv is not None and padding_kv.shape not in (
                (B, N_kv),
                (B, N_kv, 1),
            ):
                raise ValueError(
                    f"Expected padding_kv of shape ({B}, {N_kv}) or ({B}, {N_kv}, 1) "
                    f"but got tensor of shape {tuple(padding_kv.shape)}"
                )

        # Zero padded geometry so that non-finite padding values cannot reach
        # the embeddings, the RoPE tables, or the gradients.
        if padding_kv is not None:
            x_kv = x_kv.masked_fill(padding_kv.reshape(*x_kv.shape[:2], 1) == 0, 0.0)

        # The activation is also applied after the last embedding layer.
        tokens = self.activation(self.kv_to_embedding(x_kv))
        return ArGEnTGeometry(tokens, x_kv[..., : self.dim_pos], padding_kv)

    def decode_queries(
        self,
        x_query: Float[torch.Tensor, "batch nq dim_query"],
        geometry: ArGEnTGeometry,
    ) -> Float[torch.Tensor, "batch nq dim_out"]:
        r"""Predict fields at query points against an encoded geometry.

        Parameters
        ----------
        x_query : torch.Tensor
            Query tokens of shape :math:`(B, N_q, D_q)`.
        geometry : ArGEnTGeometry
            Output of :meth:`encode_geometry`.

        Returns
        -------
        torch.Tensor
            Output tokens of shape :math:`(B, N_q, D_{out})`.
        """
        ### Input validation
        if not torch.compiler.is_compiling():
            B = geometry.tokens.shape[0]
            if (
                x_query.ndim != 3
                or x_query.shape[0] != B
                or x_query.shape[-1] != self.dim_query
            ):
                raise ValueError(
                    f"Expected x_query of shape ({B}, N_q, {self.dim_query}) "
                    f"but got tensor of shape {tuple(x_query.shape)}"
                )

        # The activation is also applied after the last embedding layer.
        x_query_emb = self.activation(self.query_to_embedding(x_query))
        x_query_emb = self.transformer(
            x_query_emb,
            geometry.tokens,
            x_query[..., : self.dim_pos],
            geometry.pos,
            padding_kv=geometry.padding,
        )
        return self.to_out(x_query_emb)

    def forward(
        self,
        x_query: Float[torch.Tensor, "batch nq dim_query"],
        x_kv: Float[torch.Tensor, "batch nkv dim_kv"],
        padding_kv: Shaped[torch.Tensor, "batch nkv ..."] | None = None,
    ) -> Float[torch.Tensor, "batch nq dim_out"]:
        return self.decode_queries(x_query, self.encode_geometry(x_kv, padding_kv))


class ArGEnTDeepONet(Module):
    r"""ArGEnT trunk fused with a DeepONet-style branch over operator parameters.

    The :class:`ArGEnT` trunk produces ``dim_emb`` basis features per output
    channel at each query point, and a feed-forward branch maps the operator
    parameters :math:`\mu` (e.g. a Reynolds number) to matching coefficients:

    .. math::

        u_k(x) = \sum_{j=1}^{\text{dim\_emb}} c_{jk}(\mu) \, f_{jk}(x, \Omega)

    where :math:`f` is the trunk output for query point :math:`x` and geometry
    :math:`\Omega`. Use :class:`ArGEnT` directly when there are no operator
    parameters.

    Parameters
    ----------
    dim_op : int, optional, default=1
        Number of operator parameters. Must be at least 1.
    dim_query : int, optional, default=2
        Number of input channels of ``x_query``.
    dim_kv : int, optional, default=2
        Number of input channels of ``x_kv``.
    dim_pos : int, optional, default=2
        Number of leading coordinate channels used by RoPE.
    dim_out : int, optional, default=1
        Number of output channels per query token.
    dim_emb : int, optional, default=128
        Embedding dimension and width of each attention head. Also the number
        of basis functions per output channel.
    num_heads : int, optional, default=4
        Number of attention heads.
    depth : int, optional, default=2
        Number of cross-attention layers.
    max_freq : float, optional, default=64.0
        RoPE frequency scale. See :class:`ArGEnT`.
    use_mfnn : bool, optional, default=False
        If ``True``, use the modified FNN for the trunk MLPs and the branch.
    activation : str, optional, default="relu"
        Activation of the trunk MLPs and the branch. See :class:`ArGEnT`.
    use_relu_mlp : bool, optional, default=False
        If ``True``, use ReLU feed-forward blocks in the attention stack.
    axis_slices : list of int, optional
        Per-axis RoPE channel split. See :class:`ArGEnT`.

    Forward
    -------
    x_query : torch.Tensor
        Query tokens of shape :math:`(B, N_q, D_q)`.
    x_kv : torch.Tensor
        Key/value tokens of shape :math:`(B, N_{kv}, D_{kv})`.
    op_params : torch.Tensor
        Operator parameters of shape :math:`(B, D_{op})` or
        :math:`(B, 1, D_{op})`.
    padding_kv : torch.Tensor, optional
        Key/value padding mask. See :class:`ArGEnT`.

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
    >>> re = torch.randn(2, 1)
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
        axis_slices: list[int] | None = None,
    ) -> None:
        super().__init__(meta=ArGEnTDeepONetMetaData())
        if dim_op < 1:
            raise ValueError(
                f"Expected dim_op >= 1, got dim_op={dim_op}. "
                "Use ArGEnT directly when there are no operator parameters."
            )
        self.dim_op = dim_op
        self.dim_out = dim_out
        self.dim_emb = dim_emb

        # The trunk emits dim_emb basis features per output channel.
        self.argent = ArGEnT(
            dim_query=dim_query,
            dim_kv=dim_kv,
            dim_pos=dim_pos,
            dim_out=dim_out * dim_emb,
            dim_emb=dim_emb,
            num_heads=num_heads,
            depth=depth,
            max_freq=max_freq,
            use_mfnn=use_mfnn,
            activation=activation,
            use_relu_mlp=use_relu_mlp,
            axis_slices=axis_slices,
        )
        self.branch_net = _make_mlp(
            dim_op,
            [dim_emb] * 5,
            dim_out * dim_emb,
            get_activation(activation),
            use_mfnn,
        )

    def encode_geometry(
        self,
        x_kv: Float[torch.Tensor, "batch nkv dim_kv"],
        padding_kv: Shaped[torch.Tensor, "batch nkv ..."] | None = None,
    ) -> ArGEnTGeometry:
        r"""Embed a geometry point cloud. See :meth:`ArGEnT.encode_geometry`."""
        return self.argent.encode_geometry(x_kv, padding_kv)

    def decode_queries(
        self,
        x_query: Float[torch.Tensor, "batch nq dim_query"],
        geometry: ArGEnTGeometry,
        op_params: Float[torch.Tensor, "batch ..."],
    ) -> Float[torch.Tensor, "batch nq dim_out"]:
        r"""Predict fields at query points against an encoded geometry.

        Parameters
        ----------
        x_query : torch.Tensor
            Query tokens of shape :math:`(B, N_q, D_q)`.
        geometry : ArGEnTGeometry
            Output of :meth:`encode_geometry`.
        op_params : torch.Tensor
            Operator parameters of shape :math:`(B, D_{op})` or
            :math:`(B, 1, D_{op})`.

        Returns
        -------
        torch.Tensor
            Predicted fields of shape :math:`(B, N_q, D_{out})`.
        """
        ### Input validation (x_query is checked by the trunk)
        if not torch.compiler.is_compiling():
            B = x_query.shape[0]
            if op_params.shape not in ((B, self.dim_op), (B, 1, self.dim_op)):
                raise ValueError(
                    f"Expected op_params of shape ({B}, {self.dim_op}) or "
                    f"({B}, 1, {self.dim_op}) but got {tuple(op_params.shape)}"
                )

        trunk_out = self.argent.decode_queries(x_query, geometry)
        branch_out = self.branch_net(op_params.reshape(-1, 1, self.dim_op))

        # Multiplicative fusion: (B, Nq, dim_emb*dim_out) * (B, 1, dim_emb*dim_out),
        # then sum over the dim_emb basis functions of each output channel.
        fused = (trunk_out * branch_out).view(
            trunk_out.shape[0], trunk_out.shape[1], -1, self.dim_out
        )
        return fused.sum(dim=2)

    def forward(
        self,
        x_query: Float[torch.Tensor, "batch nq dim_query"],
        x_kv: Float[torch.Tensor, "batch nkv dim_kv"],
        op_params: Float[torch.Tensor, "batch ..."],
        padding_kv: Shaped[torch.Tensor, "batch nkv ..."] | None = None,
    ) -> Float[torch.Tensor, "batch nq dim_out"]:
        return self.decode_queries(
            x_query, self.encode_geometry(x_kv, padding_kv), op_params
        )
