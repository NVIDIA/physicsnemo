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

r"""GeoTransolver-specific FLARE++ attention backend."""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float

from physicsnemo.nn import ConcreteDropout, Mlp


class _FLAREPlusPlusMixer(nn.Module):
    r"""FLARE++ attention as used by GeoTransolver's ``"GALE_FPP"`` backend.

    Learned seeds synthesize routing queries from the input with one attention
    pass; the queries then gather from and scatter back to the tokens. This is
    the FLARE++ formulation GeoTransolver checkpoints were trained with; the
    standalone :class:`physicsnemo.nn.FLAREPlusPlus` mixer differs (normalized
    queries/keys, gated fixed queries, identity-initialized projections).
    """

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        n_global_queries: int = 64,
        use_te: bool = False,
        attn_scale: float | None = None,
    ) -> None:
        super().__init__()
        if use_te:
            raise ValueError(
                "FLAREPlusPlus does not support Transformer Engine; set use_te=False."
            )
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if heads <= 0:
            raise ValueError(f"heads must be positive, got {heads}")
        if dim_head <= 0:
            raise ValueError(f"dim_head must be positive, got {dim_head}")
        if n_global_queries <= 0:
            raise ValueError(
                f"n_global_queries must be positive, got {n_global_queries}"
            )
        if attn_scale is None:
            attn_scale = dim_head**-0.5
        if not math.isfinite(attn_scale) or attn_scale <= 0.0:
            raise ValueError(
                f"attn_scale must be a positive finite value, got {attn_scale}"
            )

        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.n_global_queries = n_global_queries
        self.use_te = False
        self.scale = float(attn_scale)
        inner_dim = heads * dim_head

        self.q_seed = nn.Parameter(torch.randn(1, heads, n_global_queries, dim_head))
        # A fused projection is mathematically identical to four independent
        # projections and executes them in one matrix multiplication. The
        # chunks are query-synthesis K/V followed by physical K/V.
        self.in_projection = nn.Linear(dim, 4 * inner_dim)
        self.out_linear = nn.Linear(inner_dim, dim)
        self.out_dropout = nn.Dropout(dropout)

    def _compute_attention(
        self, x: Float[torch.Tensor, "B N C"]
    ) -> tuple[
        Float[torch.Tensor, "B H N D"],
        Float[torch.Tensor, "B H N D"],
    ]:
        """Return FLARE++ head outputs and physical keys for backend reuse."""
        query_k, query_v, physical_k, physical_v = self.in_projection(x).chunk(
            4, dim=-1
        )
        query_k, query_v, physical_k, physical_v = (
            rearrange(tensor, "b n (h d) -> b h n d", h=self.heads, d=self.dim_head)
            for tensor in (query_k, query_v, physical_k, physical_v)
        )

        seeds = self.q_seed.to(dtype=x.dtype).expand(x.shape[0], -1, -1, -1)
        queries = F.scaled_dot_product_attention(
            seeds, query_k, query_v, scale=self.scale
        )
        routed_values = F.scaled_dot_product_attention(
            queries, physical_k, physical_v, scale=self.scale
        )
        output = F.scaled_dot_product_attention(
            physical_k, queries, routed_values, scale=self.scale
        )
        return output, physical_k

    def _project_output(
        self, output: Float[torch.Tensor, "B H N D"]
    ) -> Float[torch.Tensor, "B N C"]:
        """Merge attention heads and apply the output projection."""
        output = rearrange(output, "b h n d -> b n (h d)")
        return self.out_dropout(self.out_linear(output))

    def forward(self, x: Float[torch.Tensor, "B N C"]) -> Float[torch.Tensor, "B N C"]:
        r"""Apply FLARE++ dynamic routing to ``x``.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape :math:`(B, N, C)`.

        Returns
        -------
        torch.Tensor
            Output tensor of shape :math:`(B, N, C)`.
        """
        if not torch.compiler.is_compiling():
            if x.ndim != 3:
                raise ValueError(
                    f"Expected a 3D input tensor (B, N, C), got shape {tuple(x.shape)}"
                )
            # Exact token-sharded FLARE++ needs globally normalized encoders.
            # Ordinary DDP tensors do not expose ``redistribute`` and remain
            # fully supported.
            if hasattr(x, "redistribute"):
                raise NotImplementedError(
                    "FLAREPlusPlus does not yet support token-sharded inputs; "
                    "use replicated inputs with data parallelism."
                )

        output, _ = self._compute_attention(x)
        return self._project_output(output)


class _FLAREPlusPlusAttention(_FLAREPlusPlusMixer):
    r"""Add GeoTransolver context attention to the standalone FLARE++ mixer.

    Without context, this adapter has the same parameters and output as
    :class:`_FLAREPlusPlusMixer`. Geometry or global context is mixed
    in through a model-specific cross-attention path.
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        dropout: float,
        n_global_queries: int,
        context_dim: int,
        concrete_dropout: bool,
        state_mixing_mode: Literal["weighted", "concat_project"],
    ) -> None:
        if context_dim < 0:
            raise ValueError(f"context_dim must be non-negative, got {context_dim}")
        if state_mixing_mode not in ("weighted", "concat_project"):
            raise ValueError(
                f"Invalid state_mixing_mode: {state_mixing_mode!r}. "
                "Expected 'weighted' or 'concat_project'."
            )
        super().__init__(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
            n_global_queries=n_global_queries,
            use_te=False,
        )
        self.context_dim = context_dim
        self.state_mixing_mode = state_mixing_mode

        if context_dim > 0:
            self.cross_q = nn.Linear(dim_head, dim_head)
            self.context_kv = nn.Linear(context_dim, 2 * dim_head)
            if state_mixing_mode == "weighted":
                self.state_mixing = nn.Parameter(torch.tensor(0.0))
            else:
                self.concat_project = nn.Sequential(
                    nn.Linear(2 * dim_head, dim_head),
                    nn.GELU(),
                )

        if concrete_dropout:
            self.out_dropout = ConcreteDropout(
                in_features=dim,
                init_p=max(dropout, 0.05),
            )

    def forward(
        self,
        x: tuple[Float[torch.Tensor, "batch tokens channels"], ...],
        context: Float[torch.Tensor, "batch heads context_slices context_dim"]
        | None = None,
    ) -> list[Float[torch.Tensor, "batch tokens channels"]]:
        r"""Apply FLARE++ and optional GeoTransolver context attention."""
        if not torch.compiler.is_compiling():
            if not x:
                raise ValueError("Expected non-empty tuple of input tensors")
            for index, tensor in enumerate(x):
                if tensor.ndim != 3:
                    raise ValueError(
                        f"Expected 3D input tensor (B, N, C) at index {index}, "
                        f"got shape {tuple(tensor.shape)}"
                    )
                if hasattr(tensor, "redistribute"):
                    raise NotImplementedError(
                        "The GeoTransolver FLARE++ backend does not yet support "
                        "token-sharded inputs; use replicated inputs with data "
                        "parallelism."
                    )
            if context is not None and self.context_dim == 0:
                raise ValueError(
                    "Received context, but the FLARE++ backend was constructed "
                    "with context_dim=0"
                )

        attention_and_keys = [self._compute_attention(tensor) for tensor in x]
        outputs = [result[0] for result in attention_and_keys]

        if context is not None:
            context_k, context_v = self.context_kv(context).chunk(2, dim=-1)
            cross_outputs = [
                F.scaled_dot_product_attention(
                    self.cross_q(result[1]),
                    context_k,
                    context_v,
                    scale=self.scale,
                )
                for result in attention_and_keys
            ]
            if self.state_mixing_mode == "weighted":
                weight = torch.sigmoid(self.state_mixing)
                outputs = [
                    weight * self_output + (1.0 - weight) * cross_output
                    for self_output, cross_output in zip(
                        outputs, cross_outputs, strict=True
                    )
                ]
            else:
                outputs = [
                    self.concat_project(torch.cat((self_output, cross_output), dim=-1))
                    for self_output, cross_output in zip(
                        outputs, cross_outputs, strict=True
                    )
                ]

        return [self._project_output(output) for output in outputs]


class _FLAREPlusPlusBlock(nn.Module):
    r"""GeoTransolver residual block backed by FLARE++ attention."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        dropout: float,
        act: str,
        mlp_ratio: int,
        slice_num: int,
        context_dim: int,
        concrete_dropout: bool,
        state_mixing_mode: Literal["weighted", "concat_project"],
    ) -> None:
        super().__init__()
        dim_head = hidden_dim // num_heads
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.Attn = _FLAREPlusPlusAttention(
            dim=hidden_dim,
            heads=num_heads,
            dim_head=dim_head,
            dropout=dropout,
            n_global_queries=slice_num,
            context_dim=context_dim,
            concrete_dropout=concrete_dropout,
            state_mixing_mode=state_mixing_mode,
        )
        self.ln_mlp1 = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            Mlp(
                in_features=hidden_dim,
                hidden_features=hidden_dim * mlp_ratio,
                out_features=hidden_dim,
                act_layer=act,
                use_te=False,
            ),
        )

        if concrete_dropout:
            self.attn_dropout = ConcreteDropout(
                in_features=hidden_dim,
                init_p=max(dropout, 0.05),
            )
            self.ffn_dropout = ConcreteDropout(
                in_features=hidden_dim,
                init_p=max(dropout, 0.05),
            )
        else:
            self.attn_dropout = None
            self.ffn_dropout = None

    def forward(
        self,
        fx: tuple[Float[torch.Tensor, "batch tokens hidden_dim"], ...],
        global_context: Float[torch.Tensor, "batch heads context_slices context_dim"]
        | None,
    ) -> list[Float[torch.Tensor, "batch tokens hidden_dim"]]:
        r"""Apply pre-norm attention and feed-forward residuals."""
        if not torch.compiler.is_compiling():
            if not fx:
                raise ValueError("Expected non-empty tuple of input tensors")
            for index, tensor in enumerate(fx):
                if tensor.ndim != 3:
                    raise ValueError(
                        f"Expected 3D input tensor (B, N, C) at index {index}, "
                        f"got {tensor.ndim}D tensor with shape {tuple(tensor.shape)}"
                    )

        normed_inputs = tuple(self.ln_1(tensor) for tensor in fx)
        attention_outputs = self.Attn(normed_inputs, global_context)
        outputs = [
            attention_output + tensor
            for attention_output, tensor in zip(attention_outputs, fx, strict=True)
        ]
        if self.attn_dropout is not None:
            outputs = [self.attn_dropout(output) for output in outputs]

        outputs = [self.ln_mlp1(output) + output for output in outputs]
        if self.ffn_dropout is not None:
            outputs = [self.ffn_dropout(output) for output in outputs]
        return outputs
