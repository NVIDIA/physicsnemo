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

"""AeroJEPA predictor head.

:class:`PrototypeTokenJEPAHead` maps a target-token coordinate set to
predicted target-token features, given context tokens plus a global
conditioning vector. It is the JEPA-side head that bridges the encoder
outputs to what the decoder consumes. Operating conditions enter the
model here via the ``cond`` argument; the encoders do not see them.

The matching JEPA-side losses (SIGReg, TokenLatentSIGReg, reconstruction
family) live in :mod:`physicsnemo.experimental.models.aerojepa.losses`.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .layers import (
    FourierPositionalEncoding,
    LocalPointTransformerBlock,
    LocalTokenCrossAttentionBlock,
    TokenSet,
    compute_batch_offset_step,
    flatten_batched_coords,
    flatten_padded_batch,
    unflatten_to_padded,
)


class PrototypeTokenJEPAHead(nn.Module):
    r"""Predict target-token features from context tokens and a conditioning vector.

    Pipeline per call:

    1. Embed each ``target_positions`` row through a Fourier positional
       encoding plus a linear projection, then add a projected
       conditioning vector built from ``cond``.
    2. Project context-token features through a separate linear.
    3. Run ``depth`` interleaved layers of self-attention (over the
       target queries) and cross-attention (queries → context tokens),
       both conditioned on the same projected ``cond``. The k-NN inside
       each block sees per-batch offset coordinates so it does not mix
       across batch items.
    4. ``LayerNorm`` + linear projection back to ``token_dim``.

    Accepts both unbatched (``context_tokens.features`` rank 2) and
    padded batched (rank 3) context inputs. When ``target_positions`` is
    rank 2 it is broadcast to match the context batch size. When
    ``cond`` is rank 1 it is treated as the single-sample case; when its
    leading dim is 1 it is broadcast over the batch.

    Parameters
    ----------
    token_dim : int
        Feature dimension of context and predicted target tokens.
    cond_dim : int
        Dimension of the conditioning vector. ``0`` disables
        conditioning entirely; the head still runs but the projected
        conditioning is replaced by a per-batch zero vector and the
        self/cross blocks ignore it.
    hidden_dim : int or None, optional
        Internal hidden dimension. Defaults to ``token_dim``.
    depth : int, optional
        Number of interleaved self+cross layers (clamped to at least 1).
        Default 4.
    num_heads : int, optional
        Attention heads per layer. Default 8.
    neighbor_k : int, optional
        Per-query neighborhood size in both self- and cross-attention.
        Default 24.
    knn_chunk_size : int, optional
        Chunk size for the inner k-NN. Default 128.
    query_pe_bands : int, optional
        Fourier positional-encoding bands for the target positions.
        Default 6.
    mlp_ratio : int, optional
        Hidden multiplier inside each block's ``ResidualMLP``. Default 4.
    dropout : float, optional
        Dropout used throughout. Default 0.0.
    """

    def __init__(
        self,
        *,
        token_dim: int,
        cond_dim: int,
        hidden_dim: int | None = None,
        depth: int = 4,
        num_heads: int = 8,
        neighbor_k: int = 24,
        knn_chunk_size: int = 128,
        query_pe_bands: int = 6,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.token_dim = int(token_dim)
        self.hidden_dim = (
            int(hidden_dim) if hidden_dim is not None else int(token_dim)
        )
        self.cond_dim = int(cond_dim)
        self.depth = int(max(1, depth))
        self.query_pos_enc = FourierPositionalEncoding(
            in_dim=3, num_bands=int(query_pe_bands), include_input=True
        )
        self.query_in = nn.Linear(self.query_pos_enc.out_dim, self.hidden_dim)
        self.context_in = nn.Linear(self.token_dim, self.hidden_dim)
        self.cond_proj = None
        if self.cond_dim > 0:
            self.cond_proj = nn.Sequential(
                nn.Linear(self.cond_dim, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
        self.self_blocks = nn.ModuleList(
            [
                LocalPointTransformerBlock(
                    dim=self.hidden_dim,
                    num_heads=int(num_heads),
                    neighbor_k=int(neighbor_k),
                    dilation=1,
                    mlp_ratio=int(mlp_ratio),
                    dropout=float(dropout),
                    knn_chunk_size=int(knn_chunk_size),
                    conditioning_dim=self.hidden_dim,
                )
                for _ in range(self.depth)
            ]
        )
        self.cross_blocks = nn.ModuleList(
            [
                LocalTokenCrossAttentionBlock(
                    dim=self.hidden_dim,
                    num_heads=int(num_heads),
                    neighbor_k=int(neighbor_k),
                    mlp_ratio=int(mlp_ratio),
                    dropout=float(dropout),
                    knn_chunk_size=int(knn_chunk_size),
                    conditioning_dim=self.hidden_dim,
                )
                for _ in range(self.depth)
            ]
        )
        self.out_norm = nn.LayerNorm(self.hidden_dim)
        self.out_proj = nn.Linear(self.hidden_dim, self.token_dim)

    def _prepare_inputs(
        self,
        *,
        context_tokens: TokenSet,
        target_positions: torch.Tensor,
        cond: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        batched_context = context_tokens.features.ndim == 3
        if not batched_context:
            context_features = context_tokens.features.unsqueeze(0)
            context_coords = context_tokens.coords.unsqueeze(0)
            context_mask = torch.ones(
                (1, int(context_tokens.features.shape[0])),
                device=context_tokens.features.device,
                dtype=torch.bool,
            )
        else:
            context_features = context_tokens.features
            context_coords = context_tokens.coords
            context_mask = context_tokens.mask
            if context_mask is None:
                context_mask = torch.ones(
                    context_features.shape[:2],
                    device=context_features.device,
                    dtype=torch.bool,
                )

        batch_size = int(context_features.shape[0])
        if self.cond_dim > 0:
            if cond is None:
                raise ValueError("cond must be provided when cond_dim > 0.")
            if cond.ndim == 1:
                cond = cond.unsqueeze(0)
            if cond.ndim != 2:
                raise ValueError(
                    f"cond must have shape [B, D] or [D], got {tuple(cond.shape)}"
                )
            if int(cond.shape[0]) != batch_size:
                if int(cond.shape[0]) == 1:
                    cond = cond.expand(batch_size, -1)
                else:
                    raise ValueError(
                        f"cond batch ({int(cond.shape[0])}) does not match "
                        f"context batch ({batch_size})."
                    )
            cond_embed = self.cond_proj(cond)
        else:
            if cond is not None:
                if cond.ndim == 1:
                    cond = cond.unsqueeze(0)
                if cond.ndim != 2:
                    raise ValueError(
                        f"cond must have shape [B, D] or [D], got {tuple(cond.shape)}"
                    )
                if int(cond.shape[0]) not in {1, batch_size}:
                    raise ValueError(
                        f"cond batch ({int(cond.shape[0])}) does not match "
                        f"context batch ({batch_size})."
                    )
            cond_embed = context_features.new_zeros((batch_size, self.hidden_dim))

        if target_positions.ndim == 2:
            target_positions = target_positions.unsqueeze(0).expand(
                batch_size, -1, -1
            )
        elif target_positions.ndim != 3:
            raise ValueError(
                "target_positions must have shape [T, 3] or [B, T, 3], got "
                f"{tuple(target_positions.shape)}"
            )
        elif int(target_positions.shape[0]) != batch_size:
            if int(target_positions.shape[0]) == 1:
                target_positions = target_positions.expand(batch_size, -1, -1)
            else:
                raise ValueError(
                    f"target_positions batch ({int(target_positions.shape[0])}) "
                    f"does not match context batch ({batch_size})."
                )

        target_mask = torch.ones(
            target_positions.shape[:2],
            device=target_positions.device,
            dtype=torch.bool,
        )
        return (
            context_features,
            context_coords,
            context_mask,
            target_positions,
            target_mask,
            cond_embed,
        )

    def forward(
        self,
        *,
        context_tokens: TokenSet,
        target_positions: torch.Tensor,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        (
            context_features,
            context_coords,
            context_mask,
            target_positions_b,
            target_mask,
            cond_embed,
        ) = self._prepare_inputs(
            context_tokens=context_tokens,
            target_positions=target_positions,
            cond=cond,
        )

        query = self.query_in(
            self.query_pos_enc(target_positions_b)
        ) + cond_embed.unsqueeze(1)
        context = self.context_in(context_features)

        offset_step = max(
            compute_batch_offset_step(target_positions_b, target_mask),
            compute_batch_offset_step(context_coords, context_mask),
        )
        _, flat_target_coords, target_batch_ids = flatten_batched_coords(
            target_positions_b,
            target_mask,
            offset_step=offset_step,
        )
        _, flat_context_coords, context_batch_ids = flatten_batched_coords(
            context_coords,
            context_mask,
            offset_step=offset_step,
        )
        flat_query = flatten_padded_batch(query, target_mask)
        flat_context = flatten_padded_batch(context, context_mask)
        flat_target_cond = flatten_padded_batch(
            cond_embed.unsqueeze(1).expand(
                -1, int(target_positions_b.shape[1]), -1
            ),
            target_mask,
        )
        flat_context_cond = flatten_padded_batch(
            cond_embed.unsqueeze(1).expand(-1, int(context.shape[1]), -1),
            context_mask,
        )

        for self_block, cross_block in zip(
            self.self_blocks, self.cross_blocks, strict=True
        ):
            flat_query = self_block(
                flat_query,
                flat_target_coords,
                cond=flat_target_cond,
                batch_ids=target_batch_ids,
            )
            flat_query = cross_block(
                flat_query,
                flat_target_coords,
                flat_context,
                flat_context_coords,
                cond=flat_target_cond,
                context_cond=flat_context_cond,
                query_batch_ids=target_batch_ids,
                context_batch_ids=context_batch_ids,
            )

        out = self.out_proj(
            self.out_norm(unflatten_to_padded(flat_query, target_mask))
        )
        if context_tokens.features.ndim == 2 and out.shape[0] == 1:
            return out[0]
        return out
