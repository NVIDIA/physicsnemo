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
r"""kNN attention over static reference neighborhoods.

This is DeFormer's local operator, run before and after its FLARE backbone.
Each node attends over a fixed set of ``K`` reference
neighbors (including itself as one slot) selected once on the initial mesh,
along its edges by :func:`physicsnemo.experimental.nn.geodesic_knn.geodesic_knn`
or by straight-line distance with :func:`physicsnemo.nn.functional.knn`. Reference
edge geometry (displacement and distance) enters both the attention logits, as
a per-head bias, and the values, as an attention-weighted edge aggregate that is
projected after aggregation so no :math:`N \times K \times D` edge-value tensor
is formed. The fixed ``K`` makes the gather a dense batched operation instead of
an irregular edge scatter. Neighbor order is irrelevant; masked slots have no
influence. In the reference path keys and values are materialized per neighbor,
so peak memory per block scales as :math:`O(N K D)`; use checkpoint segments for
long rollouts. ``fused=True`` selects
:func:`~physicsnemo.experimental.models.deformer.fused_knn_attention.knn_attention`,
which computes the same map while reading neighbor keys and values on the fly;
the kernel's own extra activation memory is :math:`O(N (H + D))`, while the
block around it still holds the :math:`[N, K, E]` edge latent and the float32
:math:`[N, K, H]` logit bias. With checkpoint segments the processor's edge
encoder runs inside each segment, so those per-slot tensors are recomputed in
backward instead of being held for the whole processor.

Unlike NATTEN's neighborhood attention, which restricts attention to a sliding
window on a regular grid of pixels or voxels, a node's neighborhood here is an
explicit set of ``K`` mesh nodes selected once on the reference geometry: there
is no grid, window or dilation, and NATTEN's kernels are not used.
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

from .fused_knn_attention import (
    Backend,
    knn_attention,
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


class KNNAttentionBlock(nn.Module):
    r"""Pre-norm attention over a node's static neighbors with a MeshGraphNet-style update.

    Keys come from the normalized latent and values from the raw residual
    stream; the message is the attention-weighted neighbor difference
    :math:`\sum_j p_{ij} (v_j - v_i)`; the geometric logit bias is computed in
    float32; the node update concatenates the normalized latent, the message,
    the aggregated edge latent and the attention log-sum-exp and applies one
    MLP with an output LayerNorm, the MeshGraphNet node-update form.

    The value and edge-bias projections carry no bias: with the log-sum-exp
    detached the softmax is shift invariant per row, so an edge-bias bias would
    only shift the lse feature by a per-head constant and a value bias would
    cancel in the neighbor difference, and neither would receive a gradient.
    The key bias shifts the lse by a per-node amount and is kept.

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
        Use the fused kNN attention kernel instead of gathering keys
        and values per neighbor. Same math, lower memory; ``False`` keeps the
        validated reference path. Default is ``False``.
    fused_backend : {"auto", "triton", "torch"}, optional
        Kernel backend for the fused path. Default is ``"auto"`` (Triton on CUDA
        when available, otherwise the chunked PyTorch implementation).
    dropout : float, optional
        Dropout probability on the block's update. Default is 0.0.
    concrete_dropout : bool, optional
        Replace the fixed-rate dropout by a learned-rate
        :class:`~physicsnemo.nn.ConcreteDropout` on the block's update,
        initialised at ``max(dropout, 0.05)``. Default is ``False``.

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
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.edge_dim = edge_dim
        self.fused = bool(fused)
        self.fused_backend = fused_backend
        # Dropout on the block's update: standard dropout or a learned-rate
        # ConcreteDropout.
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
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.edge_bias = nn.Linear(edge_dim, num_heads, bias=False)
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
        key = self.key(normalized)
        # Values from the raw residual stream, so neighbor differences are clean.
        value = self.value(x)
        return normalized, query, key, value

    def _logit_bias(self, edge_latent):
        """The geometric logit bias in at least float32: the only route for
        geometry into the attention weights is never quantized to bfloat16."""
        weight, bias = self.edge_bias.weight, self.edge_bias.bias
        # Also when the block itself was converted with .bfloat16()/.half():
        # cast the weight and bias along with the input so the matmul never
        # sees mixed dtypes.
        compute = torch.promote_types(weight.dtype, torch.float32)
        with torch.autocast(device_type=edge_latent.device.type, enabled=False):
            return F.linear(
                edge_latent.to(compute),
                weight.to(compute),
                None if bias is None else bias.to(compute),
            )

    def forward(
        self,
        x: Float[torch.Tensor, "num_nodes hidden_dim"],
        index: Int[torch.Tensor, "num_nodes slots"],
        mask: Bool[torch.Tensor, "num_nodes slots"],
        edge_latent: Float[torch.Tensor, "num_nodes slots edge_dim"],
    ) -> Float[torch.Tensor, "num_nodes hidden_dim"]:
        if not torch.compiler.is_compiling():
            if x.ndim != 2 or x.shape[1] != self.hidden_dim:
                raise ValueError(
                    f"Expected x of shape (N, {self.hidden_dim}) but got tensor of "
                    f"shape {tuple(x.shape)}"
                )
            if (
                index.ndim != 2
                or index.shape[0] != x.shape[0]
                or index.dtype != torch.long
            ):
                raise ValueError(
                    f"Expected int64 index of shape ({x.shape[0]}, K) but got "
                    f"{index.dtype} tensor of shape {tuple(index.shape)}"
                )
            if mask.shape != index.shape or mask.dtype != torch.bool:
                raise ValueError(
                    f"Expected bool mask of shape {tuple(index.shape)} but got "
                    f"{mask.dtype} tensor of shape {tuple(mask.shape)}"
                )
            if edge_latent.shape != (*index.shape, self.edge_dim):
                raise ValueError(
                    f"Expected edge_latent of shape {(*index.shape, self.edge_dim)} "
                    f"but got tensor of shape {tuple(edge_latent.shape)}"
                )
        num_nodes, slots = index.shape
        heads, head_dim = self.num_heads, self.head_dim
        normalized, query, key, value = self._projections(x)
        query = query.view(num_nodes, heads, head_dim)
        bias = self._logit_bias(edge_latent)
        if self.fused:
            # Same logits, mask, softmax and weighted sums as below, computed by
            # reading key[index] / value[index] on the fly (no [N, K, H, dh]).
            aggregated, edge_aggregate, lse = knn_attention(
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
        # sum_j p_ij v_j - v_i = sum_j p_ij (v_j - v_i) since the weights sum
        # to one over the valid slots.
        aggregated = aggregated - value.reshape(num_nodes, self.hidden_dim).to(
            aggregated.dtype
        )
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


class KNNAttentionProcessor(nn.Module):
    r"""Stack of :class:`KNNAttentionBlock` layers with a shared edge encoder.

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
        Build every block with the fused kNN attention kernel. Default
        is ``False`` (reference gather implementation).
    fused_backend : {"auto", "triton", "torch"}, optional
        Backend for the fused kernel. Default is ``"auto"``.
    dropout, concrete_dropout
        Forwarded to every :class:`KNNAttentionBlock`.

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
                KNNAttentionBlock(
                    hidden_dim,
                    num_heads,
                    edge_dim,
                    mlp_ratio,
                    activation,
                    fused=fused_attention,
                    fused_backend=fused_backend,
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

    def _run_encoded_segment(
        self, start: int, stop: int, x, index, mask, edge_features
    ):
        # Encoding the edges inside the checkpointed segment means the
        # encoder's activations and the [N, K, E] latent are recomputed in
        # backward instead of being held until then.
        edge_latent = self.edge_encoder(edge_features.to(x.dtype))
        return self._run_segment(start, stop, x, index, mask, edge_latent)

    def forward(
        self,
        x: Float[torch.Tensor, "num_nodes hidden_dim"],
        neighborhood: LocalNeighborhood,
    ) -> Float[torch.Tensor, "num_nodes hidden_dim"]:
        if not torch.compiler.is_compiling():
            if x.ndim != 2 or x.shape[1] != self.hidden_dim:
                raise ValueError(
                    f"Expected x of shape (N, {self.hidden_dim}) but got tensor of "
                    f"shape {tuple(x.shape)}"
                )
            neighborhood.validate(x.shape[0], self.edge_input_dim)
        index, mask = neighborhood.index, neighborhood.mask
        use_checkpoint = (
            self.checkpoint_segments > 0 and self.training and torch.is_grad_enabled()
        )
        if not use_checkpoint:
            edge_latent = self.edge_encoder(neighborhood.edge_features.to(x.dtype))
            for start, stop in self.segments:
                x = self._run_segment(start, stop, x, index, mask, edge_latent)
            return x
        context = (
            torch.autograd.graph.save_on_cpu(pin_memory=True)
            if self.checkpoint_offloading
            else nullcontext()
        )
        with context:
            for start, stop in self.segments:
                x = checkpoint(
                    self._run_encoded_segment,
                    start,
                    stop,
                    x,
                    index,
                    mask,
                    neighborhood.edge_features,
                    use_reentrant=False,
                )
        return x
