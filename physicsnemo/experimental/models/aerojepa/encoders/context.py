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

"""Context encoder for AeroJEPA.

:class:`ContextTransformer` is the concrete :class:`BaseContextEncoder`
implementation that consumes the geometry-side input (positions plus
per-point features, optionally including SDF / normals / n-dot-uinf
channels) and produces context tokens via a :class:`PointTransformer`.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from physicsnemo.experimental.nn.aerojepa import EncoderOutput

from .base import BaseContextEncoder
from .point import PointTransformer, build_geometry_features


class ContextTransformer(BaseContextEncoder):
    r"""Concrete context encoder built on a :class:`PointTransformer`.

    Constructor parameters mirror :class:`PointTransformer`'s tokenizer
    and attention-stack configuration. Three boolean flags
    (``use_sdf`` / ``use_solid_normals`` / ``use_solid_n_dot_uinf``)
    select which trailing channels of ``context_feat`` are appended to
    the per-point feature tensor by :func:`build_geometry_features`.

    Parameters
    ----------
    point_input_dim : int
        Feature dimension of the per-point tensor handed to the inner
        :class:`PointTransformer`. Must match the output of
        :func:`build_geometry_features` for the configured flags
        (i.e. ``3 + (1 if use_sdf) + (3 if use_solid_normals) + (1 if
        use_solid_n_dot_uinf)``).
    token_dim : int
        Token embedding dimension.
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
        Attention heads per layer. Default 8.
    num_layers : int, optional
        Number of attention layers. Default 6.
    neighbor_k : int, optional
        Per-token neighborhood size. Default 24.
    dilation_schedule : Sequence[int] or None, optional
        Per-layer dilation; ``None`` uses 1 everywhere.
    mlp_ratio : int, optional
        Hidden multiplier inside each block's ``ResidualMLP``. Default
        4.
    dropout : float, optional
        Dropout used throughout. Default 0.0.
    tokenizer_cluster_pooling : str, optional
        ``'mean'`` or ``'graph'``. Default ``'mean'``.
    tokenizer_graph_pool_hidden_dim : int or None, optional
        Hidden dim for the graph pool when ``'graph'`` is selected.
    tokenizer_graph_pool_layers : int, optional
        Message-passing layers in the graph pool. Default 2.
    use_sdf : bool, optional
        Append the SDF channel to per-point features. Default ``False``.
    use_solid_normals : bool, optional
        Append the three solid-normal channels. Default ``False``.
    use_solid_n_dot_uinf : bool, optional
        Append the solid-normal · u-inf channel. Default ``False``.
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
        use_sdf: bool = False,
        use_solid_normals: bool = False,
        use_solid_n_dot_uinf: bool = False,
    ):
        super().__init__()
        self.use_sdf = bool(use_sdf)
        self.use_solid_normals = bool(use_solid_normals)
        self.use_solid_n_dot_uinf = bool(use_solid_n_dot_uinf)
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

    def forward(
        self,
        *,
        context_pos: torch.Tensor,
        context_feat: torch.Tensor,
    ) -> EncoderOutput:
        point_features = build_geometry_features(
            context_pos=context_pos,
            context_feat=context_feat,
            use_sdf=self.use_sdf,
            use_solid_normals=self.use_solid_normals,
            use_solid_n_dot_uinf=self.use_solid_n_dot_uinf,
        )
        return self.encoder.encode_single(
            point_positions=context_pos,
            point_features=point_features,
            gen_params=None,
        )

    def forward_batched(
        self,
        *,
        context_pos: torch.Tensor,
        context_feat: torch.Tensor,
        context_pos_n: torch.Tensor,
    ) -> EncoderOutput:
        point_features = build_geometry_features(
            context_pos=context_pos,
            context_feat=context_feat,
            use_sdf=self.use_sdf,
            use_solid_normals=self.use_solid_normals,
            use_solid_n_dot_uinf=self.use_solid_n_dot_uinf,
        )
        return self.encoder.forward_batched(
            point_positions=context_pos,
            point_features=point_features,
            point_counts=context_pos_n,
            gen_params=None,
        )
