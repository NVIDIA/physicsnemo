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

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float

from physicsnemo.nn import ConcreteDropout, FLAREPlusPlus, Mlp


class _FLAREPlusPlusAttention(FLAREPlusPlus):
    r"""Add GeoTransolver context attention to the standalone FLARE++ mixer.

    Without context, this adapter has the same parameters and output as
    :class:`physicsnemo.nn.FLAREPlusPlus`. Geometry or global context is mixed
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
