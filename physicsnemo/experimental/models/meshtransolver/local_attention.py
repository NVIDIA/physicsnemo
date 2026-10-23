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
r"""Neighborhood attention over static reference neighborhoods.

This is the local operator DeFormer can use in place of its MeshGraphNet
pre-/post-processors. Each node attends over a fixed set of ``K`` reference
neighbors (including itself as one slot) selected once on the initial mesh by
:func:`physicsnemo.nn.functional.neighbors.static_knn.static_knn`. Reference
edge geometry (displacement and distance) enters both the attention logits, as
a per-head bias, and the values, as an attention-weighted edge aggregate that is
projected after aggregation so no :math:`N \times K \times D` edge-value tensor
is formed. The fixed ``K`` makes the gather a dense batched operation instead of
an irregular edge scatter. Neighbor order is irrelevant; masked slots have no
influence. In the reference path keys and values are materialized per neighbor,
so peak memory per block scales as :math:`O(N K D)`; use checkpoint segments for
long rollouts. ``fused=True`` selects
:func:`~physicsnemo.experimental.models.meshtransolver.fused_neighborhood_attention.neighborhood_attention`,
which computes the same map while reading neighbor keys and values on the fly
and keeps the extra activation memory at :math:`O(N (H + D))`.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Bool, Float, Int
from torch.utils.checkpoint import checkpoint

from physicsnemo.nn import ConcreteDropout, get_activation

from .fused_neighborhood_attention import (
    Backend,
    neighborhood_attention,
    resolve_backend,
)


@dataclass(frozen=True)
class LocalNeighborhood:
    r"""Static reference neighborhood consumed by the attention processors.

    ``index`` holds ``K`` neighbor IDs per node, one of which is normally the
    node itself. Padded slots must repeat a valid ID and carry ``mask=False``.
    ``edge_features`` are reference-geometry features per slot.
    """

    index: Int[torch.Tensor, "num_nodes slots"]
    mask: Bool[torch.Tensor, "num_nodes slots"]
    edge_features: Float[torch.Tensor, "num_nodes slots edge_input_dim"]

    def validate(self, num_nodes: int, edge_input_dim: int) -> None:
        if (
            self.index.ndim != 2
            or self.index.shape[0] != num_nodes
            or self.index.dtype != torch.long
        ):
            raise ValueError(
                f"neighborhood index must be int64 [{num_nodes}, K]; got "
                f"{tuple(self.index.shape)} {self.index.dtype}"
            )
        if self.mask.shape != self.index.shape or self.mask.dtype != torch.bool:
            raise ValueError("neighborhood mask must be bool with the index shape")
        if self.edge_features.shape != (*self.index.shape, edge_input_dim):
            raise ValueError(
                "neighborhood edge_features must have shape "
                f"[{num_nodes}, K, {edge_input_dim}]; got "
                f"{tuple(self.edge_features.shape)}"
            )
        if (
            self.mask.device != self.index.device
            or self.edge_features.device != self.index.device
        ):
            raise ValueError("neighborhood tensors must share a device")
        if not torch.compiler.is_compiling():
            if self.index.numel() and (
                torch.any(self.index < 0) or torch.any(self.index >= num_nodes)
            ):
                raise ValueError("neighborhood index outside the node range")
            if self.index.numel() and not torch.all(self.mask.any(dim=1)):
                raise ValueError("every node needs at least one valid neighbor slot")

    def to(self, device: torch.device) -> "LocalNeighborhood":
        return LocalNeighborhood(
            self.index.to(device), self.mask.to(device), self.edge_features.to(device)
        )


class NeighborhoodAttentionBlock(nn.Module):
    r"""Pre-norm attention over a node's static neighbors, then a feed-forward net.

    Parameters
    ----------
    hidden_dim : int
        Node latent width :math:`D`.
    num_heads : int
        Attention heads :math:`H`; ``hidden_dim`` must be divisible by it.
    edge_dim : int
        Width of the encoded per-slot edge latent supplied by the processor.
    mlp_ratio : int, optional
        Feed-forward expansion ratio. Default is 2.
    activation : str, optional
        Feed-forward activation name. Default is ``"gelu"``.
    fused : bool, optional
        Use the fused neighborhood attention kernel instead of gathering keys
        and values per neighbor. Same math, lower memory; ``False`` keeps the
        validated reference path. Default is ``False``.
    fused_backend : {"auto", "triton", "torch"}, optional
        Kernel backend for the fused path. Default is ``"auto"`` (Triton on CUDA
        when available, otherwise the chunked PyTorch implementation).
    value_norm : bool, optional
        Project values from the pre-normalized latent (``True``, the original
        block) or from the raw residual stream (``False``), which lets the block
        form clean neighbor differences. ``False`` requires ``split_key_value``.
        Default is ``True``.
    difference_values : bool, optional
        Return :math:`\sum_j p_{ij} (v_j - v_i)` instead of :math:`\sum_j p_{ij}
        v_j`, i.e. attention-weighted neighbor differences. Default is ``False``.
    update : {"residual", "concat_mlp"}, optional
        ``"residual"`` adds the projected message to the stream and applies a
        feed-forward net (original block). ``"concat_mlp"`` concatenates the
        normalized latent, the aggregated values, the aggregated edge latent and
        the attention log-sum-exp and applies one MLP with an output LayerNorm,
        the MeshGraphNet node-update form. Default is ``"residual"``.
    bias_fp32 : bool, optional
        Compute the geometric logit bias in float32 even under autocast or with
        half-precision parameters (weight and bias are cast with the input), so
        the only route for geometry into the attention weights is not quantized
        to bfloat16. The bias is returned in float32 (float64 for float64
        parameters). Default is ``False`` (the bias follows the parameter and
        autocast dtype).
    split_key_value : bool, optional
        Use separate key and value projections instead of one fused ``[2D, D]``
        matrix (which Muon otherwise orthogonalizes jointly). Default ``False``.
    dropout : float, optional
        Dropout probability on the block's residual updates (the attention
        message and the feed-forward output, or the ``concat_mlp`` update).
        Default is 0.0.
    concrete_dropout : bool, optional
        Replace the fixed-rate dropout by a learned-rate
        :class:`~physicsnemo.nn.ConcreteDropout` shared by the block's update
        sites, initialised at ``max(dropout, 0.05)``. Default is ``False``.

    Forward
    -------
    x : torch.Tensor
        Node latent of shape :math:`(N, D)`.
    index : torch.Tensor
        Neighbor IDs of shape :math:`(N, K)`.
    mask : torch.Tensor
        Valid-slot mask of shape :math:`(N, K)`.
    edge_latent : torch.Tensor
        Encoded edge latent of shape :math:`(N, K, D_e)`.

    Outputs
    -------
    torch.Tensor
        Updated node latent of shape :math:`(N, D)`.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        edge_dim: int,
        mlp_ratio: int = 2,
        activation: str = "gelu",
        fused: bool = False,
        fused_backend: Backend = "auto",
        value_norm: bool = True,
        difference_values: bool = False,
        update: str = "residual",
        bias_fp32: bool = False,
        split_key_value: bool = False,
        dropout: float = 0.0,
        concrete_dropout: bool = False,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0 or num_heads <= 0 or hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be positive and divisible by num_heads")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if edge_dim <= 0:
            raise ValueError("edge_dim must be positive")
        if mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be positive")
        if fused_backend not in ("auto", "triton", "torch"):
            raise ValueError("fused_backend must be 'auto', 'triton' or 'torch'")
        if update not in ("residual", "concat_mlp"):
            raise ValueError("update must be 'residual' or 'concat_mlp'")
        if not value_norm and not split_key_value:
            raise ValueError("value_norm=False requires split_key_value=True")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.edge_dim = edge_dim
        self.fused = bool(fused)
        self.fused_backend = fused_backend
        self.value_norm = bool(value_norm)
        self.difference_values = bool(difference_values)
        self.update = update
        self.bias_fp32 = bool(bias_fp32)
        self.split_key_value = bool(split_key_value)
        # Dropout on the block's residual update(s): standard dropout, or a
        # learned-rate ConcreteDropout shared by the block's update sites.
        if concrete_dropout:
            self.dropout = ConcreteDropout(
                in_features=hidden_dim, init_p=max(float(dropout), 0.05)
            )
        elif dropout > 0:
            self.dropout = nn.Dropout(float(dropout))
        else:
            self.dropout = nn.Identity()
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.query = nn.Linear(hidden_dim, hidden_dim)
        if split_key_value:
            self.key = nn.Linear(hidden_dim, hidden_dim)
            self.value = nn.Linear(hidden_dim, hidden_dim)
        else:
            self.key_value = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.edge_bias = nn.Linear(edge_dim, num_heads)
        if update == "residual":
            # Aggregate the low-width edge latent with the attention weights
            # first, then project: equivalent to per-head edge values without an
            # N x K x D intermediate.
            self.edge_value = nn.Linear(num_heads * edge_dim, hidden_dim, bias=False)
            self.output = nn.Linear(hidden_dim, hidden_dim)
            self.ffn_norm = nn.LayerNorm(hidden_dim)
            self.ffn = nn.Sequential(
                nn.Linear(hidden_dim, mlp_ratio * hidden_dim),
                get_activation(activation),
                nn.Linear(mlp_ratio * hidden_dim, hidden_dim),
            )
        else:
            # Node update on [LN(x_i), message_i, edge_message_i, lse_i], the
            # self-plus-aggregate concatenation of a MeshGraphNet node block.
            update_in = 2 * hidden_dim + num_heads * edge_dim + num_heads
            self.update_mlp = nn.Sequential(
                nn.Linear(update_in, mlp_ratio * hidden_dim),
                get_activation(activation),
                nn.Linear(mlp_ratio * hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )

    def _projections(self, x):
        normalized = self.attention_norm(x)
        query = self.query(normalized)
        if self.split_key_value:
            key = self.key(normalized)
            value = self.value(normalized if self.value_norm else x)
        else:
            key, value = self.key_value(normalized).chunk(2, dim=-1)
        return normalized, query, key, value

    def _logit_bias(self, edge_latent):
        if self.bias_fp32:
            weight, bias = self.edge_bias.weight, self.edge_bias.bias
            # At least float32, also when the block itself was converted with
            # .bfloat16()/.half(): cast the weight and bias along with the
            # input so the matmul never sees mixed dtypes.
            compute = torch.promote_types(weight.dtype, torch.float32)
            with torch.autocast(device_type=edge_latent.device.type, enabled=False):
                return F.linear(
                    edge_latent.to(compute),
                    weight.to(compute),
                    None if bias is None else bias.to(compute),
                )
        return self.edge_bias(edge_latent)

    def forward(
        self,
        x: Float[torch.Tensor, "num_nodes hidden_dim"],
        index: Int[torch.Tensor, "num_nodes slots"],
        mask: Bool[torch.Tensor, "num_nodes slots"],
        edge_latent: Float[torch.Tensor, "num_nodes slots edge_dim"],
    ) -> Float[torch.Tensor, "num_nodes hidden_dim"]:
        num_nodes, slots = index.shape
        heads, head_dim = self.num_heads, self.head_dim
        normalized, query, key, value = self._projections(x)
        query = query.view(num_nodes, heads, head_dim)
        bias = self._logit_bias(edge_latent)
        if self.fused:
            # Same logits, mask, softmax and weighted sums as below, computed by
            # reading key[index] / value[index] on the fly (no [N, K, H, dh]).
            aggregated, edge_aggregate, lse = neighborhood_attention(
                query,
                key.view(num_nodes, heads, head_dim),
                value.view(num_nodes, heads, head_dim),
                bias,
                edge_latent,
                index,
                mask,
                head_dim**-0.5,
                backend=resolve_backend(self.fused_backend, x.device),
                return_lse=True,
            )
            aggregated = aggregated.reshape(num_nodes, self.hidden_dim)
            edge_aggregate = edge_aggregate.reshape(num_nodes, heads * self.edge_dim)
        else:
            flat = index.reshape(-1)
            gathered_key = key.index_select(0, flat).view(
                num_nodes, slots, heads, head_dim
            )
            gathered_value = value.index_select(0, flat).view(
                num_nodes, slots, heads, head_dim
            )
            # [N, H, K] logits, softmax and log-sum-exp in float32 regardless of
            # autocast (autocast would otherwise re-dispatch the einsum to bf16).
            compute = torch.promote_types(query.dtype, torch.float32)
            with torch.autocast(device_type=x.device.type, enabled=False):
                logits = torch.einsum(
                    "nhd,nkhd->nhk", query.to(compute), gathered_key.to(compute)
                )
                logits = logits * (head_dim**-0.5)
                logits = logits + bias.to(compute).permute(0, 2, 1)
                logits = logits.masked_fill(~mask.unsqueeze(1), float("-inf"))
                # A fully masked row gives lse = -inf here and +inf in the fused
                # kernel; validate() forbids such rows and the concat_mlp update
                # below zeroes a non-finite lse, so neither value is consumed.
                lse = torch.logsumexp(logits, dim=-1)
                weights = torch.nan_to_num(torch.softmax(logits, dim=-1), nan=0.0)
            weights = weights.to(gathered_value.dtype)
            aggregated = torch.einsum("nhk,nkhd->nhd", weights, gathered_value)
            aggregated = aggregated.reshape(num_nodes, self.hidden_dim)
            edge_aggregate = torch.einsum(
                "nhk,nke->nhe", weights, edge_latent.to(weights.dtype)
            ).reshape(num_nodes, heads * self.edge_dim)
        if self.difference_values:
            # sum_j p_ij v_j - v_i = sum_j p_ij (v_j - v_i) since the weights sum
            # to one over the valid slots.
            aggregated = aggregated - value.reshape(num_nodes, self.hidden_dim).to(
                aggregated.dtype
            )
        if self.update == "residual":
            x = x + self.dropout(
                self.output(aggregated) + self.edge_value(edge_aggregate)
            )
            return x + self.dropout(self.ffn(self.ffn_norm(x)))
        lse = torch.where(torch.isfinite(lse), lse, torch.zeros_like(lse)).detach()
        features = torch.cat(
            (
                normalized.to(x.dtype),
                aggregated.to(x.dtype),
                edge_aggregate.to(x.dtype),
                lse.to(x.dtype),
            ),
            dim=-1,
        )
        return x + self.dropout(self.update_mlp(features))


class NeighborhoodAttentionProcessor(nn.Module):
    r"""Stack of :class:`NeighborhoodAttentionBlock` layers with a shared edge encoder.

    Parameters
    ----------
    num_layers : int
        Number of attention blocks.
    hidden_dim : int
        Node latent width.
    num_heads : int
        Attention heads per block.
    edge_input_dim : int, optional
        Raw per-slot edge feature width (reference displacement and distance
        give four). Default is 4.
    edge_dim : int, optional
        Encoded edge latent width. Default is 32.
    mlp_ratio : int, optional
        Feed-forward expansion ratio. Default is 2.
    activation : str, optional
        Activation for the edge encoder and feed-forward nets. Default ``"gelu"``.
    checkpoint_segments : int, optional
        Number of contiguous block groups recomputed during backward; 0 disables
        activation checkpointing. Default is 0.
    checkpoint_offloading : bool, optional
        Offload checkpointed activations to host memory. Default is ``False``.
    fused_attention : bool, optional
        Build every block with the fused neighborhood attention kernel. Default
        is ``False`` (reference gather implementation).
    fused_backend : {"auto", "triton", "torch"}, optional
        Backend for the fused kernel. Default is ``"auto"``.
    value_norm, difference_values, update, bias_fp32, split_key_value, dropout, concrete_dropout
        Forwarded to every :class:`NeighborhoodAttentionBlock`; the defaults
        (``True``, ``False``, ``"residual"``, ``False``, ``False``, 0.0,
        ``False``) reproduce the original block.

    Forward
    -------
    x : torch.Tensor
        Node latent of shape :math:`(N, D)`.
    neighborhood : LocalNeighborhood
        Static reference neighborhood for these nodes.

    Outputs
    -------
    torch.Tensor
        Updated node latent of shape :math:`(N, D)`.
    """

    def __init__(
        self,
        num_layers: int,
        hidden_dim: int,
        num_heads: int,
        edge_input_dim: int = 4,
        edge_dim: int = 32,
        mlp_ratio: int = 2,
        activation: str = "gelu",
        checkpoint_segments: int = 0,
        checkpoint_offloading: bool = False,
        fused_attention: bool = False,
        fused_backend: Backend = "auto",
        value_norm: bool = True,
        difference_values: bool = False,
        update: str = "residual",
        bias_fp32: bool = False,
        split_key_value: bool = False,
        dropout: float = 0.0,
        concrete_dropout: bool = False,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if edge_input_dim <= 0:
            raise ValueError("edge_input_dim must be positive")
        if checkpoint_segments < 0 or checkpoint_segments > num_layers:
            raise ValueError("checkpoint_segments must be in [0, num_layers]")
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.edge_input_dim = edge_input_dim
        self.edge_dim = edge_dim
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_input_dim, edge_dim),
            get_activation(activation),
            nn.Linear(edge_dim, edge_dim),
            nn.LayerNorm(edge_dim),
        )
        self.blocks = nn.ModuleList(
            [
                NeighborhoodAttentionBlock(
                    hidden_dim,
                    num_heads,
                    edge_dim,
                    mlp_ratio,
                    activation,
                    fused=fused_attention,
                    fused_backend=fused_backend,
                    value_norm=value_norm,
                    difference_values=difference_values,
                    update=update,
                    bias_fp32=bias_fp32,
                    split_key_value=split_key_value,
                    dropout=dropout,
                    concrete_dropout=concrete_dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.checkpoint_segments = checkpoint_segments
        self.checkpoint_offloading = bool(checkpoint_offloading) and (
            checkpoint_segments > 0
        )
        if checkpoint_segments > 0:
            size, extra = divmod(num_layers, checkpoint_segments)
            bounds, start = [], 0
            for segment in range(checkpoint_segments):
                stop = start + size + (1 if segment < extra else 0)
                bounds.append((start, stop))
                start = stop
            self.segments = tuple(bounds)
        else:
            self.segments = ((0, num_layers),)

    def _run_segment(self, start: int, stop: int, x, index, mask, edge_latent):
        for block in self.blocks[start:stop]:
            x = block(x, index, mask, edge_latent)
        return x

    def forward(
        self,
        x: Float[torch.Tensor, "num_nodes hidden_dim"],
        neighborhood: LocalNeighborhood,
    ) -> Float[torch.Tensor, "num_nodes hidden_dim"]:
        if x.ndim != 2 or x.shape[1] != self.hidden_dim:
            raise ValueError(f"x must have shape [N, {self.hidden_dim}]")
        neighborhood.validate(x.shape[0], self.edge_input_dim)
        index, mask = neighborhood.index, neighborhood.mask
        edge_latent = self.edge_encoder(neighborhood.edge_features.to(x.dtype))
        use_checkpoint = (
            self.checkpoint_segments > 0 and self.training and torch.is_grad_enabled()
        )
        context = (
            torch.autograd.graph.save_on_cpu(pin_memory=True)
            if use_checkpoint and self.checkpoint_offloading
            else nullcontext()
        )
        with context:
            for start, stop in self.segments:
                if use_checkpoint:
                    x = checkpoint(
                        self._run_segment,
                        start,
                        stop,
                        x,
                        index,
                        mask,
                        edge_latent,
                        use_reentrant=False,
                    )
                else:
                    x = self._run_segment(start, stop, x, index, mask, edge_latent)
        return x
