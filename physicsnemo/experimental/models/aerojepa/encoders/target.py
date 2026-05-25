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

"""Target encoder for AeroJEPA.

:class:`TargetTransformer` is the concrete :class:`BaseTargetEncoder`
that wires a :class:`PointTransformer` to the training-time
surface + volume input. The surface and volume halves are bundled into a
single padded point tensor (matching the layout the context branch sees
for whole-domain models) and handed to the inner encoder for a
self-attention pass.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from physicsnemo.experimental.nn.aerojepa import EncoderOutput, counts_to_mask

from .base import BaseTargetEncoder
from .point import PointTransformer


class TargetTransformer(BaseTargetEncoder):
    r"""Concrete target encoder for AeroJEPA.

    Builds an inner :class:`PointTransformer` (with conditioning disabled)
    and runs it on the concatenation of the surface and volume halves of
    the training-time input. Operating conditions are not passed through;
    they enter the model only at the predictor head.

    Parameters
    ----------
    point_input_dim : int
        Feature dimension of the per-point input.
    token_dim : int
        Token embedding dimension used throughout the encoder stack.
    max_point_tokens : int
        Maximum number of tokens after tokenization.
    tokenizer_strategy : str, optional
        Tokenizer center-selection strategy. Default
        ``"voxel_fps_cluster"``.
    tokenizer_deterministic_eval : bool, optional
        Forwarded to the tokenizer. Default ``True``.
    tokenizer_cluster_size : int or None, optional
        Forwarded to the tokenizer.
    tokenizer_knn_chunk_size : int, optional
        Chunk size for the tokenizer / attention k-NN lookups. Default
        128.
    tokenizer_voxel_size : float, Sequence[float] or None, optional
        Forwarded to the tokenizer.
    tokenizer_prototype_coords : torch.Tensor or None, optional
        Forwarded to the tokenizer.
    tokenizer_prototype_knn_k : int or None, optional
        Forwarded to the tokenizer.
    point_pos_pe_bands : int, optional
        Fourier positional-encoding bands. Default 6.
    num_heads : int, optional
        Attention heads per self-attention layer. Default 8.
    num_layers : int, optional
        Number of self-attention layers. Default 6.
    neighbor_k : int, optional
        Per-token neighborhood size in the self-attention blocks.
        Default 24.
    dilation_schedule : Sequence[int] or None, optional
        Per-layer dilation. ``None`` uses 1 everywhere.
    mlp_ratio : int, optional
        Hidden multiplier inside each block's ``ResidualMLP``. Default
        4.
    dropout : float, optional
        Dropout used throughout. Default 0.0.
    tokenizer_cluster_pooling : str, optional
        ``'mean'`` or ``'graph'``. Default ``'mean'``.
    tokenizer_graph_pool_hidden_dim : int or None, optional
        Hidden dim for the graph pool.
    tokenizer_graph_pool_layers : int, optional
        Message-passing layers in the graph pool. Default 2.
    """

    supports_batched_forward = True

    def __init__(
        self,
        *,
        point_input_dim: int,
        token_dim: int,
        max_point_tokens: int,
        tokenizer_strategy: str = "voxel_fps_cluster",
        tokenizer_deterministic_eval: bool = True,
        tokenizer_cluster_size: int | None = None,
        tokenizer_knn_chunk_size: int = 128,
        tokenizer_voxel_size: Sequence[float] | float | None = None,
        tokenizer_prototype_coords: torch.Tensor | None = None,
        tokenizer_prototype_knn_k: int | None = None,
        point_pos_pe_bands: int = 6,
        num_heads: int = 8,
        num_layers: int = 6,
        neighbor_k: int = 24,
        dilation_schedule: Sequence[int] | None = None,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
        tokenizer_cluster_pooling: str = "mean",
        tokenizer_graph_pool_hidden_dim: int | None = None,
        tokenizer_graph_pool_layers: int = 2,
    ):
        super().__init__()
        self.encoder = PointTransformer(
            point_input_dim=int(point_input_dim),
            token_dim=int(token_dim),
            max_point_tokens=int(max_point_tokens),
            tokenizer_strategy=str(tokenizer_strategy).lower(),
            tokenizer_deterministic_eval=bool(tokenizer_deterministic_eval),
            tokenizer_cluster_size=tokenizer_cluster_size,
            tokenizer_voxel_size=tokenizer_voxel_size,
            tokenizer_prototype_coords=tokenizer_prototype_coords,
            tokenizer_prototype_knn_k=tokenizer_prototype_knn_k,
            tokenizer_knn_chunk_size=int(tokenizer_knn_chunk_size),
            point_pos_pe_bands=int(point_pos_pe_bands),
            num_heads=int(num_heads),
            num_layers=int(num_layers),
            neighbor_k=int(neighbor_k),
            dilation_schedule=dilation_schedule,
            mlp_ratio=int(mlp_ratio),
            dropout=float(dropout),
            tokenizer_cluster_pooling=str(tokenizer_cluster_pooling).lower(),
            tokenizer_graph_pool_hidden_dim=tokenizer_graph_pool_hidden_dim,
            tokenizer_graph_pool_layers=int(tokenizer_graph_pool_layers),
            use_gen_conditioning=False,
            gen_conditioning_dim=None,
        )

    def _concat_inputs(
        self,
        *,
        surface_pos: torch.Tensor,
        surface_main_feat: torch.Tensor,
        volume_pos: torch.Tensor,
        volume_feat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if int(surface_main_feat.shape[-1]) != int(volume_feat.shape[-1]):
            raise ValueError(
                "Target encoder requires matching feature dims for "
                "surface_main_feat and volume_feat."
            )
        return (
            torch.cat([surface_pos, volume_pos], dim=0),
            torch.cat([surface_main_feat, volume_feat], dim=0),
        )

    def forward(
        self,
        *,
        surface_pos: torch.Tensor,
        surface_main_feat: torch.Tensor,
        volume_pos: torch.Tensor,
        volume_feat: torch.Tensor,
    ) -> EncoderOutput:
        point_positions, point_features = self._concat_inputs(
            surface_pos=surface_pos,
            surface_main_feat=surface_main_feat,
            volume_pos=volume_pos,
            volume_feat=volume_feat,
        )
        return self.encoder.encode_single(
            point_positions=point_positions,
            point_features=point_features,
            gen_params=None,
        )

    def forward_batched(
        self,
        *,
        surface_pos: torch.Tensor,
        surface_main_feat: torch.Tensor,
        surface_pos_n: torch.Tensor,
        volume_pos: torch.Tensor,
        volume_feat: torch.Tensor,
        volume_pos_n: torch.Tensor,
    ) -> EncoderOutput:
        r"""Encode a padded batched surface + volume input.

        Surface and volume halves are interleaved per batch item via
        :func:`counts_to_mask`: surface points are placed first, volume
        points immediately after, left-aligned. The combined per-batch
        count is forwarded to the inner encoder.

        Parameters
        ----------
        surface_pos, surface_main_feat : torch.Tensor
            Padded surface positions of shape ``(B, Ns, 3)`` and features
            of shape ``(B, Ns, F)``.
        surface_pos_n : torch.Tensor
            Per-batch valid surface-point counts of shape ``(B,)``.
        volume_pos, volume_feat : torch.Tensor
            Padded volumetric positions of shape ``(B, Nv, 3)`` and
            features of shape ``(B, Nv, F)``.
        volume_pos_n : torch.Tensor
            Per-batch valid volume-point counts of shape ``(B,)``.

        Returns
        -------
        EncoderOutput
            Padded target token set; see
            :meth:`PointTransformer.forward_batched`.
        """
        point_counts = surface_pos_n + volume_pos_n
        max_points = int(point_counts.max().item())
        point_positions = surface_pos.new_zeros(
            (int(surface_pos.shape[0]), max_points, int(surface_pos.shape[-1]))
        )
        point_features = surface_main_feat.new_zeros(
            (
                int(surface_main_feat.shape[0]),
                max_points,
                int(surface_main_feat.shape[-1]),
            )
        )
        surface_mask = counts_to_mask(
            surface_pos_n, max_len=int(surface_pos.shape[1])
        )
        volume_mask = counts_to_mask(
            volume_pos_n, max_len=int(volume_pos.shape[1])
        )

        batch_surface_ids = (
            torch.arange(
                int(surface_pos.shape[0]),
                device=surface_pos.device,
                dtype=torch.long,
            )
            .unsqueeze(1)
            .expand_as(surface_mask)[surface_mask]
        )
        batch_volume_ids = (
            torch.arange(
                int(volume_pos.shape[0]),
                device=volume_pos.device,
                dtype=torch.long,
            )
            .unsqueeze(1)
            .expand_as(volume_mask)[volume_mask]
        )

        surface_dst = (
            torch.arange(
                int(surface_pos.shape[1]),
                device=surface_pos.device,
                dtype=torch.long,
            )
            .unsqueeze(0)
            .expand_as(surface_mask)[surface_mask]
        )
        volume_local = (
            torch.arange(
                int(volume_pos.shape[1]),
                device=volume_pos.device,
                dtype=torch.long,
            )
            .unsqueeze(0)
            .expand_as(volume_mask)[volume_mask]
        )
        volume_dst = surface_pos_n[batch_volume_ids] + volume_local

        point_positions[batch_surface_ids, surface_dst] = surface_pos[surface_mask]
        point_features[batch_surface_ids, surface_dst] = surface_main_feat[
            surface_mask
        ]
        point_positions[batch_volume_ids, volume_dst] = volume_pos[volume_mask]
        point_features[batch_volume_ids, volume_dst] = volume_feat[volume_mask]
        return self.encoder.forward_batched(
            point_positions=point_positions,
            point_features=point_features,
            point_counts=point_counts,
            gen_params=None,
        )
