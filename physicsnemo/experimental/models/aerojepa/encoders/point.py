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

"""Point-cloud transformer building blocks for AeroJEPA encoders.

Three pieces share this module:

- :func:`build_geometry_features` — assemble the per-point feature tensor
  fed into the encoder, optionally appending SDF / solid-normal /
  n-dot-uinf channels carried in the trailing slots of the raw input
  features.
- :class:`PointClusterGraphPool` — learned local pooling from sampled
  points to tokenizer centroids; replaces the default mean-pool when the
  encoder is configured with ``tokenizer_cluster_pooling='graph'``.
- :class:`PointTransformer` — the main point-cloud encoder. Tokenizes the
  input via :class:`PointCloudTokenizer`, embeds tokens with a Fourier
  positional encoding, runs a stack of local point-transformer attention
  blocks (optionally conditioned), and emits an :class:`EncoderOutput`.

The class is used by both the context and target encoders downstream;
the context branch instantiates it with conditioning disabled, the
target branch with conditioning enabled.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from ..layers import (
    EncoderOutput,
    FourierPositionalEncoding,
    LocalPointTransformerBlock,
    PointCloudTokenizer,
    TokenSet,
    compute_batch_offset_step,
    flatten_batched_coords,
    flatten_padded_batch,
    masked_mean,
    unflatten_to_padded,
)


def build_geometry_features(
    *,
    context_pos: torch.Tensor,
    context_feat: torch.Tensor,
    use_sdf: bool = False,
    use_solid_normals: bool = False,
    use_solid_n_dot_uinf: bool = False,
) -> torch.Tensor:
    r"""Assemble per-point features from positions and selected feature channels.

    When none of the three boolean flags are set, returns ``context_pos``
    unchanged (the positions themselves act as the only per-point
    feature). Otherwise concatenates ``context_pos`` with the
    appropriate trailing slices of ``context_feat``: SDF channel (one
    column), solid normals (three columns), and the solid-normal · u-inf
    dot product (one column). The trailing-slot layout of
    ``context_feat`` is assumed to be
    ``[..., other_features, sdf, normals_xyz, n_dot_uinf]``, with each
    block present only when its corresponding flag is true.

    Parameters
    ----------
    context_pos : torch.Tensor
        Point positions of shape ``(*, 3)``.
    context_feat : torch.Tensor
        Per-point feature tensor of shape ``(*, F)``. Trailing channels
        carry the optional SDF / normals / n-dot data when the
        corresponding flags are set.
    use_sdf : bool, optional
        Append the SDF channel to the output. Default ``False``.
    use_solid_normals : bool, optional
        Append the three solid-normal channels. Default ``False``.
    use_solid_n_dot_uinf : bool, optional
        Append the solid-normal · u-inf channel. Default ``False``.

    Returns
    -------
    torch.Tensor
        Assembled features of shape ``(*, 3 + extras)``.

    Raises
    ------
    ValueError
        If ``context_feat`` does not have enough trailing channels for
        the requested features.
    """
    if not use_sdf and not use_solid_normals and not use_solid_n_dot_uinf:
        return context_pos

    extra_needed = (
        (1 if use_sdf else 0)
        + (3 if use_solid_normals else 0)
        + (1 if use_solid_n_dot_uinf else 0)
    )
    if int(context_feat.shape[-1]) < int(context_pos.shape[-1]) + extra_needed:
        raise ValueError(
            "context_feat does not contain enough channels for configured "
            "geometry features."
        )

    parts = [context_pos]
    if use_sdf:
        sdf_offset_from_end = (
            (3 if use_solid_normals else 0)
            + (1 if use_solid_n_dot_uinf else 0)
            + 1
        )
        sdf_start = -sdf_offset_from_end
        sdf_end = None if sdf_offset_from_end == 1 else -sdf_offset_from_end + 1
        parts.append(context_feat[..., sdf_start:sdf_end])
    if use_solid_normals:
        start = int(context_feat.shape[-1]) - (4 if use_solid_n_dot_uinf else 3)
        parts.append(context_feat[..., start : start + 3])
    if use_solid_n_dot_uinf:
        parts.append(context_feat[..., -1:])
    return torch.cat(parts, dim=-1)


class PointClusterGraphPool(nn.Module):
    r"""Learned local pooling from sampled points to tokenizer centroids.

    For each tokenizer center, gathers its k-nearest source points (via
    ``cluster_idx`` produced by
    :meth:`physicsnemo.experimental.models.aerojepa.layers.PointCloudTokenizer.tokenize_with_clusters`)
    and aggregates them through a stack of message-passing-style updates:
    per-edge message and gate MLPs computed from neighbor + center
    features plus the relative position and distance, gated mean
    aggregation, then a residual node update. Replaces the default
    mean-pool when the host encoder is configured with
    ``tokenizer_cluster_pooling='graph'``.

    Parameters
    ----------
    point_feature_dim : int
        Feature dimension of both source and token features.
    coord_dim : int, optional
        Coordinate dimension used in the edge feature. Default 3.
    hidden_dim : int
        Hidden dimension of the message / gate / update MLPs.
    num_layers : int
        Number of message-passing iterations. Clamped to at least 1.
    dropout : float
        Dropout used inside the message and update MLPs.

    Shape
    -----
    Forward inputs / outputs are flat (unbatched):

    - ``point_positions``: ``(N_src, 3)``
    - ``point_features``: ``(N_src, F)``
    - ``token_positions``: ``(N_tok, 3)``
    - ``token_features``: ``(N_tok, F)``
    - ``cluster_idx``: ``(N_tok, K)`` ``int64``
    - Output: ``(N_tok, F)``

    When ``token_positions`` is empty or ``cluster_idx`` is ``None`` the
    pool is a no-op and ``token_features`` is returned unchanged.
    """

    def __init__(
        self,
        *,
        point_feature_dim: int,
        coord_dim: int = 3,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.point_feature_dim = int(point_feature_dim)
        self.coord_dim = int(coord_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(max(1, num_layers))
        edge_in_dim = 2 * self.point_feature_dim + self.coord_dim + 1
        self.msg_mlps = nn.ModuleList()
        self.gate_mlps = nn.ModuleList()
        self.update_mlps = nn.ModuleList()
        for _ in range(self.num_layers):
            self.msg_mlps.append(
                nn.Sequential(
                    nn.Linear(edge_in_dim, self.hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(float(dropout)),
                    nn.Linear(self.hidden_dim, self.point_feature_dim),
                )
            )
            self.gate_mlps.append(
                nn.Sequential(
                    nn.Linear(edge_in_dim, self.hidden_dim),
                    nn.SiLU(),
                    nn.Linear(self.hidden_dim, 1),
                )
            )
            self.update_mlps.append(
                nn.Sequential(
                    nn.Linear(2 * self.point_feature_dim, self.hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(float(dropout)),
                    nn.Linear(self.hidden_dim, self.point_feature_dim),
                )
            )
        self.out_norm = nn.LayerNorm(self.point_feature_dim)

    def forward(
        self,
        *,
        point_positions: torch.Tensor,
        point_features: torch.Tensor,
        token_positions: torch.Tensor,
        token_features: torch.Tensor,
        cluster_idx: torch.Tensor,
    ) -> torch.Tensor:
        if int(token_positions.shape[0]) == 0 or cluster_idx is None:
            return token_features
        nbr_pos = point_positions[cluster_idx]
        nbr_feat = point_features[cluster_idx]
        token_feat = token_features
        for msg_mlp, gate_mlp, update_mlp in zip(
            self.msg_mlps, self.gate_mlps, self.update_mlps, strict=True
        ):
            token_feat_exp = token_feat.unsqueeze(1).expand(
                -1, int(cluster_idx.shape[1]), -1
            )
            rel = nbr_pos - token_positions.unsqueeze(1)
            dist = torch.linalg.norm(rel, dim=-1, keepdim=True)
            edge_input = torch.cat([nbr_feat, token_feat_exp, rel, dist], dim=-1)
            messages = msg_mlp(edge_input)
            gates = torch.sigmoid(gate_mlp(edge_input))
            pooled = (gates * messages).sum(dim=1) / gates.sum(dim=1).clamp_min(1e-6)
            token_feat = token_feat + update_mlp(
                torch.cat([token_feat, pooled], dim=-1)
            )
        return self.out_norm(token_feat)


class PointTransformer(nn.Module):
    r"""Point-cloud transformer encoder for AeroJEPA.

    Pipeline per call:

    1. Tokenize the input point set with a configured
       :class:`PointCloudTokenizer`. When cluster pooling is requested
       via ``tokenizer_cluster_pooling='graph'``, the token features go
       through a :class:`PointClusterGraphPool`; otherwise the
       tokenizer's own mean-pool is used.
    2. Embed each token with a linear projection of its features plus a
       Fourier-encoded positional embedding.
    3. Optionally add a per-call conditioning vector
       (``use_gen_conditioning=True``) projected from ``gen_params``.
    4. Run a stack of :class:`LocalPointTransformerBlock` blocks with
       configurable dilation per layer.
    5. LayerNorm, build the output ``TokenSet``, and emit an
       :class:`EncoderOutput`.

    Two entry points are exposed:

    - :meth:`encode_single` — unbatched, takes flat ``(N, *)`` inputs.
    - :meth:`forward_batched` — padded batched ``(B, N, *)`` inputs.
      Coordinates are flattened with a per-batch offset (via
      :func:`compute_batch_offset_step` and
      :func:`flatten_batched_coords`) so the k-NN inside each
      transformer block does not mix tokens across batch items.

    Conditioning is supported but optional: this class is composed by
    both the context branch (no conditioning) and the target branch
    (conditioning on operating conditions).

    Parameters
    ----------
    point_input_dim : int
        Feature dimension of the input point tensor (matches the output
        of :func:`build_geometry_features` for the configured flags).
    token_dim : int
        Token embedding dimension used throughout the attention stack.
    max_point_tokens : int
        Maximum number of tokens emitted by the tokenizer.
    tokenizer_strategy : str
        Tokenizer center-selection strategy; see
        :class:`PointCloudTokenizer` for the recognised values.
    tokenizer_deterministic_eval : bool
        Forwarded to the tokenizer.
    tokenizer_cluster_size : int or None
        Forwarded to the tokenizer.
    tokenizer_voxel_size : float, Sequence[float] or None
        Forwarded to the tokenizer.
    tokenizer_prototype_coords : torch.Tensor or None
        Forwarded to the tokenizer.
    tokenizer_prototype_knn_k : int or None
        Forwarded to the tokenizer.
    tokenizer_knn_chunk_size : int
        Chunk size for tokenizer / attention k-NN lookups.
    point_pos_pe_bands : int
        Number of Fourier bands for the positional encoding.
    num_heads : int
        Attention heads per ``LocalPointTransformerBlock``.
    num_layers : int
        Number of ``LocalPointTransformerBlock`` layers.
    neighbor_k : int
        Per-token neighborhood size inside the attention blocks.
    dilation_schedule : Sequence[int] or None
        Per-layer dilation. ``None`` uses dilation 1 everywhere. Shorter
        schedules are padded by repeating the last value.
    mlp_ratio : int
        Hidden multiplier for the ``ResidualMLP`` inside each block.
    dropout : float
        Dropout shared across the projections and attention blocks.
    tokenizer_cluster_pooling : str
        ``'mean'`` (default) for the tokenizer's built-in mean-pool, or
        ``'graph'`` for the learned :class:`PointClusterGraphPool`.
    tokenizer_graph_pool_hidden_dim : int or None
        Hidden dim for the graph pool. ``None`` uses
        ``max(4 * point_input_dim, token_dim // 2, 32)``.
    tokenizer_graph_pool_layers : int
        Number of message-passing layers in the graph pool.
    use_gen_conditioning : bool
        Whether to condition the embedding and attention blocks on
        ``gen_params``.
    gen_conditioning_dim : int or None
        Dimension of the conditioning vector; required when
        ``use_gen_conditioning=True``.

    Raises
    ------
    ValueError
        If ``tokenizer_cluster_pooling`` is unknown, if ``'graph'`` is
        requested with a non-cluster tokenizer strategy, or if
        ``use_gen_conditioning=True`` without a ``gen_conditioning_dim``.
    """

    supports_batched_forward = True

    def __init__(
        self,
        *,
        point_input_dim: int,
        token_dim: int,
        max_point_tokens: int,
        tokenizer_strategy: str,
        tokenizer_deterministic_eval: bool,
        tokenizer_cluster_size: int | None,
        tokenizer_voxel_size: Sequence[float] | float | None,
        tokenizer_prototype_coords: torch.Tensor | None,
        tokenizer_prototype_knn_k: int | None,
        tokenizer_knn_chunk_size: int,
        point_pos_pe_bands: int,
        num_heads: int,
        num_layers: int,
        neighbor_k: int,
        dilation_schedule: Sequence[int] | None,
        mlp_ratio: int,
        dropout: float,
        tokenizer_cluster_pooling: str,
        tokenizer_graph_pool_hidden_dim: int | None,
        tokenizer_graph_pool_layers: int,
        use_gen_conditioning: bool = False,
        gen_conditioning_dim: int | None,
    ):
        super().__init__()
        self.tokenizer = PointCloudTokenizer(
            max_point_tokens=int(max_point_tokens),
            strategy=str(tokenizer_strategy).lower(),
            deterministic_eval=bool(tokenizer_deterministic_eval),
            cluster_size=tokenizer_cluster_size,
            knn_chunk_size=int(tokenizer_knn_chunk_size),
            voxel_size=tokenizer_voxel_size,
            prototype_coords=tokenizer_prototype_coords,
            prototype_knn_k=tokenizer_prototype_knn_k,
        )
        self.tokenizer_cluster_pooling = str(tokenizer_cluster_pooling).lower()
        if self.tokenizer_cluster_pooling not in {"mean", "graph"}:
            raise ValueError(
                "tokenizer_cluster_pooling must be one of: 'mean', 'graph'."
            )
        if (
            self.tokenizer_cluster_pooling == "graph"
            and not self.tokenizer.uses_cluster_pooling
        ):
            raise ValueError(
                "tokenizer_cluster_pooling='graph' requires a cluster tokenizer strategy."
            )
        self.graph_pool = None
        if self.tokenizer_cluster_pooling == "graph":
            self.graph_pool = PointClusterGraphPool(
                point_feature_dim=int(point_input_dim),
                coord_dim=3,
                hidden_dim=(
                    int(tokenizer_graph_pool_hidden_dim)
                    if tokenizer_graph_pool_hidden_dim is not None
                    else max(int(point_input_dim) * 4, int(token_dim) // 2, 32)
                ),
                num_layers=int(tokenizer_graph_pool_layers),
                dropout=float(dropout),
            )
        self.feature_in = nn.Linear(int(point_input_dim), int(token_dim))
        self.pos_enc = FourierPositionalEncoding(
            in_dim=3, num_bands=int(point_pos_pe_bands), include_input=True
        )
        self.pos_proj = nn.Linear(self.pos_enc.out_dim, int(token_dim))
        self.use_gen_conditioning = bool(use_gen_conditioning)
        if self.use_gen_conditioning:
            if gen_conditioning_dim is None:
                raise ValueError(
                    "gen_conditioning_dim must be provided when use_gen_conditioning=True."
                )
            self.gen_proj = nn.Sequential(
                nn.Linear(int(gen_conditioning_dim), int(token_dim)),
                nn.SiLU(),
                nn.Linear(int(token_dim), int(token_dim)),
            )
        else:
            self.gen_proj = None

        dilations = (
            list(dilation_schedule)
            if dilation_schedule is not None
            else [1] * int(num_layers)
        )
        if len(dilations) < int(num_layers):
            dilations.extend([dilations[-1]] * (int(num_layers) - len(dilations)))
        self.blocks = nn.ModuleList(
            [
                LocalPointTransformerBlock(
                    dim=int(token_dim),
                    num_heads=int(num_heads),
                    neighbor_k=int(neighbor_k),
                    dilation=int(dilations[i]),
                    mlp_ratio=int(mlp_ratio),
                    dropout=float(dropout),
                    knn_chunk_size=int(tokenizer_knn_chunk_size),
                    conditioning_dim=(
                        int(token_dim) if self.use_gen_conditioning else None
                    ),
                )
                for i in range(int(num_layers))
            ]
        )
        self.out_norm = nn.LayerNorm(int(token_dim))

    def _compute_gen_embedding(
        self, gen_params: torch.Tensor | None
    ) -> torch.Tensor | None:
        if self.gen_proj is None:
            return None
        if gen_params is None:
            raise ValueError(
                "gen_params must be provided when use_gen_conditioning=True."
            )
        return self.gen_proj(gen_params)

    def _embed_tokens(
        self,
        *,
        token_coords: torch.Tensor,
        token_features: torch.Tensor,
        gen_params: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        x = self.feature_in(token_features) + self.pos_proj(self.pos_enc(token_coords))
        gen_embed = self._compute_gen_embedding(gen_params)
        if gen_embed is not None:
            x = x + gen_embed.unsqueeze(0)
        return x, gen_embed

    def _tokenize_single(
        self,
        *,
        point_positions: torch.Tensor,
        point_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.graph_pool is None:
            token_coords, token_features = self.tokenizer(
                point_positions=point_positions,
                point_features=point_features,
            )
            return token_coords, token_features
        token_coords, token_features, cluster_idx = (
            self.tokenizer.tokenize_with_clusters(
                point_positions=point_positions,
                point_features=point_features,
            )
        )
        if cluster_idx is None:
            return token_coords, token_features
        token_features = self.graph_pool(
            point_positions=point_positions,
            point_features=point_features,
            token_positions=token_coords,
            token_features=token_features,
            cluster_idx=cluster_idx,
        )
        return token_coords, token_features

    def _tokenize_batched(
        self,
        *,
        point_positions: torch.Tensor,
        point_features: torch.Tensor,
        point_counts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.graph_pool is None and not self.tokenizer.requires_per_sample_loop:
            return self.tokenizer.forward_batched(
                point_positions=point_positions,
                point_features=point_features,
                point_counts=point_counts,
            )

        token_positions_list = []
        token_features_list = []
        token_counts_list = []
        for i in range(int(point_positions.shape[0])):
            n_i = int(point_counts[i].item())
            pos_i, feat_i = self._tokenize_single(
                point_positions=point_positions[i, :n_i],
                point_features=point_features[i, :n_i],
            )
            token_positions_list.append(pos_i)
            token_features_list.append(feat_i)
            token_counts_list.append(int(pos_i.shape[0]))
        max_tokens = max(token_counts_list) if token_counts_list else 0
        token_positions = point_positions.new_zeros(
            (
                int(point_positions.shape[0]),
                max_tokens,
                int(point_positions.shape[-1]),
            )
        )
        token_features = point_features.new_zeros(
            (
                int(point_positions.shape[0]),
                max_tokens,
                int(point_features.shape[-1]),
            )
        )
        token_mask = torch.zeros(
            (int(point_positions.shape[0]), max_tokens),
            device=point_positions.device,
            dtype=torch.bool,
        )
        for i, (pos_i, feat_i) in enumerate(
            zip(token_positions_list, token_features_list, strict=True)
        ):
            count_i = int(pos_i.shape[0])
            if count_i <= 0:
                continue
            token_positions[i, :count_i] = pos_i
            token_features[i, :count_i] = feat_i
            token_mask[i, :count_i] = True
        token_counts = torch.tensor(
            token_counts_list, device=point_positions.device, dtype=torch.long
        )
        return token_positions, token_features, token_mask, token_counts

    def encode_single(
        self,
        *,
        point_positions: torch.Tensor,
        point_features: torch.Tensor,
        gen_params: torch.Tensor | None,
    ) -> EncoderOutput:
        r"""Encode a single unbatched point cloud.

        Parameters
        ----------
        point_positions : torch.Tensor
            Input positions of shape ``(N, 3)``.
        point_features : torch.Tensor
            Per-point input features of shape ``(N, point_input_dim)``.
        gen_params : torch.Tensor or None
            Operating conditions, required when the encoder was built
            with ``use_gen_conditioning=True``; otherwise ``None``.

        Returns
        -------
        EncoderOutput
            ``tokens`` is a :class:`TokenSet` carrying the encoded token
            features and coordinates; ``global_token`` is the masked mean
            of the token features; ``aux['point_token_count']`` is the
            number of tokens produced.
        """
        token_coords, token_features = self._tokenize_single(
            point_positions=point_positions,
            point_features=point_features,
        )
        x, gen_embed = self._embed_tokens(
            token_coords=token_coords,
            token_features=token_features,
            gen_params=gen_params,
        )
        for block in self.blocks:
            x = block(x, token_coords, cond=gen_embed)
        x = self.out_norm(x)
        global_token = masked_mean(x, None)
        tokens = TokenSet(features=x, coords=token_coords, global_token=global_token)
        return EncoderOutput(
            tokens=tokens,
            global_token=global_token,
            aux={"point_token_count": int(x.shape[0])},
        )

    def forward_batched(
        self,
        *,
        point_positions: torch.Tensor,
        point_features: torch.Tensor,
        point_counts: torch.Tensor,
        gen_params: torch.Tensor | None,
    ) -> EncoderOutput:
        r"""Encode a padded batched point cloud.

        Coordinates are flattened across the batch with a per-batch
        offset on the first coordinate axis so the k-NN inside each
        transformer block does not mix tokens across batch items.

        Parameters
        ----------
        point_positions : torch.Tensor
            Padded positions of shape ``(B, N, 3)``.
        point_features : torch.Tensor
            Padded features of shape ``(B, N, point_input_dim)``.
        point_counts : torch.Tensor
            Per-batch valid point counts of shape ``(B,)``.
        gen_params : torch.Tensor or None
            Per-batch conditioning, required when the encoder was built
            with ``use_gen_conditioning=True``.

        Returns
        -------
        EncoderOutput
            Padded ``TokenSet`` with mask and per-batch counts in
            ``aux['point_token_count']``.
        """
        (
            token_positions,
            token_features,
            token_mask,
            token_counts,
        ) = self._tokenize_batched(
            point_positions=point_positions,
            point_features=point_features,
            point_counts=point_counts,
        )
        x = self.feature_in(token_features) + self.pos_proj(self.pos_enc(token_positions))
        gen_embed = self._compute_gen_embedding(gen_params)
        if gen_embed is not None:
            x = x + gen_embed.unsqueeze(1)

        offset_step = compute_batch_offset_step(token_positions, token_mask)
        flat_coords, flat_offset_coords, batch_ids = flatten_batched_coords(
            token_positions,
            token_mask,
            offset_step=offset_step,
        )
        flat_x = flatten_padded_batch(x, token_mask)
        flat_cond = None
        if gen_embed is not None:
            flat_cond = flatten_padded_batch(
                gen_embed.unsqueeze(1).expand(-1, int(token_positions.shape[1]), -1),
                token_mask,
            )
        for block in self.blocks:
            flat_x = block(
                flat_x, flat_offset_coords, cond=flat_cond, batch_ids=batch_ids
            )
        flat_x = self.out_norm(flat_x)
        padded_x = unflatten_to_padded(flat_x, token_mask)
        global_token = masked_mean(padded_x, token_mask).squeeze(1)
        tokens = TokenSet(
            features=padded_x,
            coords=token_positions,
            mask=token_mask,
            global_token=global_token,
        )
        return EncoderOutput(
            tokens=tokens,
            global_token=global_token,
            aux={"point_token_count": token_counts},
        )
