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

"""Query-token field decoder for AeroJEPA.

:class:`QueryTokenDecoder` is the implicit field decoder that reads
predicted target tokens and emits field values at arbitrary query
positions. The decoder embeds each query (position + optional SDF +
optional conditioning) via a Fourier positional encoding plus a linear
projection, cross-attends from query embeddings to the target token
set, runs a trunk MLP, and produces per-query predictions via either a
single linear head or a velocity / pressure split (with an optional
SIREN-style pressure head and final refinement).

:class:`SineLayer` and :class:`SirenHead` are the small SIREN building
blocks used when the SIREN-style pressure head or final refinement are
enabled.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .layers import (
    FourierPositionalEncoding,
    LocalTokenCrossAttentionBlock,
    TokenSet,
    compute_batch_offset_step,
    counts_to_mask,
    flatten_batched_coords,
    flatten_padded_batch,
    unflatten_to_padded,
)


class SineLayer(nn.Module):
    r"""Linear layer with a SIREN-style sine activation.

    Computes ``sin(omega_0 * (W x + b))`` with the SIREN initialisation
    scheme: the first layer of a stack samples its weights uniformly
    from ``[-1/in_dim, 1/in_dim]``; subsequent layers use
    ``[-sqrt(6/in_dim)/omega_0, sqrt(6/in_dim)/omega_0]``.

    Parameters
    ----------
    in_dim : int
        Input dimension.
    out_dim : int
        Output dimension.
    omega_0 : float, optional
        Frequency multiplier inside the sine. Default 30.0.
    is_first : bool, optional
        Use the first-layer initialisation rule. Default ``False``.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        omega_0: float = 30.0,
        is_first: bool = False,
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.omega_0 = float(omega_0)
        self.is_first = bool(is_first)
        self.linear = nn.Linear(int(in_dim), int(out_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            if self.is_first:
                bound = 1.0 / max(self.in_dim, 1)
            else:
                bound = (6.0 / max(self.in_dim, 1)) ** 0.5 / max(self.omega_0, 1e-6)
            self.linear.weight.uniform_(-bound, bound)
            self.linear.bias.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * self.linear(x))


class SirenHead(nn.Module):
    r"""Small SIREN network: ``num_layers`` of :class:`SineLayer` + final ``Linear``.

    The hidden layers use the SIREN sine activation; the final layer is
    a plain ``nn.Linear`` initialised with the SIREN bound. Used by the
    decoder when ``pressure_head_style='siren'`` or
    ``final_refinement_style='siren'``.

    Parameters
    ----------
    in_dim : int
        Input dimension.
    hidden_dim : int
        Hidden dimension shared across the sine layers and feeding the
        final linear.
    out_dim : int
        Output dimension.
    num_layers : int
        Number of sine layers (clamped to at least 1).
    omega_0 : float, optional
        Frequency multiplier passed to every :class:`SineLayer`.
        Default 30.0.
    """

    def __init__(
        self,
        *,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int,
        omega_0: float = 30.0,
    ):
        super().__init__()
        hidden_dim = int(hidden_dim)
        num_layers = max(1, int(num_layers))
        layers = [
            SineLayer(int(in_dim), hidden_dim, omega_0=float(omega_0), is_first=True)
        ]
        for _ in range(max(0, num_layers - 1)):
            layers.append(
                SineLayer(
                    hidden_dim, hidden_dim, omega_0=float(omega_0), is_first=False
                )
            )
        self.hidden = nn.Sequential(*layers)
        self.out = nn.Linear(hidden_dim, int(out_dim))
        with torch.no_grad():
            bound = (6.0 / max(hidden_dim, 1)) ** 0.5 / max(float(omega_0), 1e-6)
            self.out.weight.uniform_(-bound, bound)
            self.out.bias.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.hidden(x))


class QueryTokenDecoder(nn.Module):
    r"""Implicit field decoder driven by cross-attention to target tokens.

    For each query position, builds a per-query embedding from a Fourier
    positional encoding of the position, an optional SDF channel, and an
    optional conditioning vector. The embedding is then refined by a
    stack of :class:`LocalTokenCrossAttentionBlock` blocks that
    cross-attend to the target token set, followed by a trunk MLP and an
    output head. Several optional behaviors:

    - **Wall-velocity gate** (``wall_velocity_gate_enabled``): multiplies
      the first three output channels by ``sigmoid(alpha * sdf)`` so
      velocity predictions vanish smoothly at the wall.
    - **Pressure split head** (``pressure_split_head_enabled``): splits
      the output into a velocity linear head (3 channels) and a separate
      pressure head (1 channel). The pressure head can be a plain MLP or
      a :class:`SirenHead` depending on ``pressure_head_style``.
    - **Final refinement** (``final_refinement_style='siren'``): adds a
      residual SIREN-style refinement of the trunk's hidden vector
      before the output head.
    - **Extra SDF features** (``extra_sdf_features_enabled``): appends
      ``[|sdf|, sign(sdf), 1/(|sdf|+eps)]`` to the per-query feature
      tensor in addition to the raw SDF channel.

    Long query lists are processed in chunks of ``query_chunk_size``.
    Both :meth:`forward` (single-cloud) and :meth:`forward_batched`
    (padded batched) return ``(pred, query_embeddings)``.

    Parameters
    ----------
    token_dim : int
        Feature dimension of the target tokens.
    query_dim : int, optional
        Dimension of the query coordinates. Default 3.
    hidden_dim : int, optional
        Trunk MLP hidden dimension. Default 256.
    num_layers : int, optional
        Number of trunk MLP linear-then-silu blocks. Default 4.
    out_dim : int, optional
        Output channel count (unused when ``pressure_split_head_enabled``
        because the split head fixes the layout to ``[u, v, w, p]``).
        Default 4.
    use_sdf : bool, optional
        Append the SDF channel to per-query features. Requires
        ``query_sdf`` at forward time. Default ``True``.
    cond_dim : int, optional
        Dimension of the per-query conditioning vector. ``0`` disables
        conditioning. Default 0.
    pe_num_bands : int, optional
        Fourier positional-encoding bands. Default 16.
    cross_attention_heads : int, optional
        Heads per cross-attention block. Default 8.
    cross_attention_layers : int, optional
        Number of cross-attention blocks. Default 2.
    cross_attention_k : int, optional
        Per-query neighborhood size into the target tokens. Default 32.
    attention_mlp_ratio : int, optional
        Hidden multiplier inside each cross-block's ``ResidualMLP``.
        Default 4.
    dropout : float, optional
        Dropout used throughout. Default 0.0.
    query_chunk_size : int, optional
        Chunk size for processing long query lists. Default 4096.
    wall_velocity_gate_enabled : bool, optional
        Enable the wall-velocity gate. Default ``False``.
    wall_velocity_gate_alpha : float, optional
        Gate sharpness. Default 20.0.
    pressure_split_head_enabled : bool, optional
        Use separate velocity and pressure heads. Default ``False``.
    pressure_head_hidden_dim : int or None, optional
        Hidden dim for the pressure head. Defaults to ``hidden_dim``.
    pressure_head_style : str, optional
        ``'mlp'`` or ``'siren'``. Default ``'mlp'``.
    pressure_head_siren_layers : int, optional
        SIREN layers when the pressure head is SIREN. Default 2.
    pressure_head_siren_omega0 : float, optional
        SIREN ``omega_0`` for the pressure head. Default 30.0.
    final_refinement_style : str, optional
        ``'none'`` or ``'siren'``. Default ``'none'``.
    final_refinement_siren_layers : int, optional
        SIREN layers when refinement is SIREN. Default 2.
    final_refinement_siren_omega0 : float, optional
        SIREN ``omega_0`` for refinement. Default 30.0.
    extra_sdf_features_enabled : bool, optional
        Append ``[|sdf|, sign(sdf), 1/(|sdf|+eps)]`` to per-query
        features. Default ``False``.
    extra_sdf_inv_eps : float, optional
        Epsilon for the ``1/(|sdf|+eps)`` channel. Default ``1e-3``.

    Raises
    ------
    ValueError
        If ``pressure_head_style`` or ``final_refinement_style`` is not
        recognised.
    """

    def __init__(
        self,
        *,
        token_dim: int,
        query_dim: int = 3,
        hidden_dim: int = 256,
        num_layers: int = 4,
        out_dim: int = 4,
        use_sdf: bool = True,
        cond_dim: int = 0,
        pe_num_bands: int = 16,
        cross_attention_heads: int = 8,
        cross_attention_layers: int = 2,
        cross_attention_k: int = 32,
        attention_mlp_ratio: int = 4,
        dropout: float = 0.0,
        query_chunk_size: int = 4096,
        wall_velocity_gate_enabled: bool = False,
        wall_velocity_gate_alpha: float = 20.0,
        pressure_split_head_enabled: bool = False,
        pressure_head_hidden_dim: int | None = None,
        pressure_head_style: str = "mlp",
        pressure_head_siren_layers: int = 2,
        pressure_head_siren_omega0: float = 30.0,
        final_refinement_style: str = "none",
        final_refinement_siren_layers: int = 2,
        final_refinement_siren_omega0: float = 30.0,
        extra_sdf_features_enabled: bool = False,
        extra_sdf_inv_eps: float = 1e-3,
    ):
        super().__init__()
        self.use_sdf = bool(use_sdf)
        self.cond_dim = int(cond_dim)
        self.query_chunk_size = int(query_chunk_size)
        self.wall_velocity_gate_enabled = bool(wall_velocity_gate_enabled)
        self.wall_velocity_gate_alpha = float(wall_velocity_gate_alpha)
        self.pressure_split_head_enabled = bool(pressure_split_head_enabled)
        self.pressure_head_hidden_dim = (
            int(pressure_head_hidden_dim)
            if pressure_head_hidden_dim is not None
            else int(hidden_dim)
        )
        self.pressure_head_style = str(pressure_head_style).lower()
        self.final_refinement_style = str(final_refinement_style).lower()
        if self.pressure_head_style not in {"mlp", "siren"}:
            raise ValueError("pressure_head_style must be one of: 'mlp', 'siren'.")
        if self.final_refinement_style not in {"none", "siren"}:
            raise ValueError(
                "final_refinement_style must be one of: 'none', 'siren'."
            )
        self.extra_sdf_features_enabled = bool(extra_sdf_features_enabled)
        self.extra_sdf_inv_eps = float(extra_sdf_inv_eps)

        self.pe = FourierPositionalEncoding(
            in_dim=int(query_dim),
            num_bands=int(pe_num_bands),
            include_input=True,
        )
        query_in_dim = self.pe.out_dim + (1 if self.use_sdf else 0) + self.cond_dim
        if self.extra_sdf_features_enabled:
            query_in_dim += 3
        self.query_in = nn.Linear(query_in_dim, int(token_dim))
        self.cross_blocks = nn.ModuleList(
            [
                LocalTokenCrossAttentionBlock(
                    dim=int(token_dim),
                    num_heads=int(cross_attention_heads),
                    neighbor_k=int(cross_attention_k),
                    mlp_ratio=int(attention_mlp_ratio),
                    dropout=float(dropout),
                    knn_chunk_size=max(1, min(int(query_chunk_size), 1024)),
                )
                for _ in range(int(cross_attention_layers))
            ]
        )
        self.trunk = nn.Sequential(
            nn.LayerNorm(int(token_dim)),
            nn.Linear(int(token_dim), int(hidden_dim)),
            nn.SiLU(),
            *[
                layer
                for _ in range(max(0, int(num_layers) - 1))
                for layer in (
                    nn.Linear(int(hidden_dim), int(hidden_dim)),
                    nn.SiLU(),
                )
            ],
        )
        if self.final_refinement_style == "siren":
            self.final_refinement = SirenHead(
                in_dim=int(hidden_dim),
                hidden_dim=int(hidden_dim),
                out_dim=int(hidden_dim),
                num_layers=int(final_refinement_siren_layers),
                omega_0=float(final_refinement_siren_omega0),
            )
        else:
            self.final_refinement = None
        if self.pressure_split_head_enabled:
            self.vel_head = nn.Linear(int(hidden_dim), 3)
            if self.pressure_head_style == "siren":
                self.p_head = SirenHead(
                    in_dim=int(hidden_dim),
                    hidden_dim=int(self.pressure_head_hidden_dim),
                    out_dim=1,
                    num_layers=int(pressure_head_siren_layers),
                    omega_0=float(pressure_head_siren_omega0),
                )
            else:
                self.p_head = nn.Sequential(
                    nn.Linear(int(hidden_dim), self.pressure_head_hidden_dim),
                    nn.SiLU(),
                    nn.Linear(self.pressure_head_hidden_dim, 1),
                )
            self.out = None
        else:
            self.out = nn.Linear(int(hidden_dim), int(out_dim))
            self.vel_head = None
            self.p_head = None

    def _build_query_inputs(
        self,
        *,
        query_pos: torch.Tensor,
        query_sdf: torch.Tensor | None,
        cond: torch.Tensor | None,
    ) -> torch.Tensor:
        parts = [self.pe(query_pos)]
        if self.use_sdf:
            if query_sdf is None:
                raise ValueError("query_sdf must be provided when use_sdf=True.")
            parts.append(query_sdf)
            if self.extra_sdf_features_enabled:
                sdf_abs = torch.abs(query_sdf)
                sdf_sign = torch.sign(query_sdf)
                sdf_inv_abs = 1.0 / (sdf_abs + max(self.extra_sdf_inv_eps, 1e-12))
                parts.append(torch.cat([sdf_abs, sdf_sign, sdf_inv_abs], dim=-1))
        if self.cond_dim > 0:
            if cond is None:
                raise ValueError("cond must be provided when cond_dim > 0.")
            parts.append(cond)
        return torch.cat(parts, dim=-1)

    def _apply_head(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.pressure_split_head_enabled:
            return torch.cat([self.vel_head(hidden), self.p_head(hidden)], dim=-1)
        return self.out(hidden)

    def _apply_wall_gate(
        self,
        pred: torch.Tensor,
        query_sdf: torch.Tensor | None,
    ) -> torch.Tensor:
        if not self.wall_velocity_gate_enabled:
            return pred
        if query_sdf is None:
            raise ValueError(
                "query_sdf must be provided when wall_velocity_gate_enabled=True."
            )
        gate = torch.sigmoid(self.wall_velocity_gate_alpha * query_sdf)
        return torch.cat([pred[:, :3] * gate, pred[:, 3:]], dim=-1)

    def _decode_chunk(
        self,
        *,
        query_pos: torch.Tensor,
        query_sdf: torch.Tensor | None,
        token_features: torch.Tensor,
        token_coords: torch.Tensor,
        cond: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query_tokens = self.query_in(
            self._build_query_inputs(
                query_pos=query_pos, query_sdf=query_sdf, cond=cond
            )
        )
        for block in self.cross_blocks:
            query_tokens = block(
                query_tokens,
                query_pos,
                token_features,
                token_coords,
            )
        hidden = self.trunk(query_tokens)
        if self.final_refinement is not None:
            hidden = hidden + self.final_refinement(hidden)
        pred = self._apply_head(hidden)
        pred = self._apply_wall_gate(pred, query_sdf)
        return pred, query_tokens

    def forward(
        self,
        *,
        query_pos: torch.Tensor,
        query_sdf: torch.Tensor | None,
        target_tokens: TokenSet,
        cond: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        token_features = target_tokens.features
        token_coords = target_tokens.coords
        if target_tokens.mask is not None:
            valid = target_tokens.mask
            token_features = token_features[valid]
            token_coords = token_coords[valid]
        if int(token_features.shape[0]) == 0:
            raise ValueError("target_tokens must contain at least one valid token.")

        preds = []
        query_embeddings = []
        for start in range(0, int(query_pos.shape[0]), self.query_chunk_size):
            end = min(start + self.query_chunk_size, int(query_pos.shape[0]))
            cond_chunk = None if cond is None else cond[start:end]
            pred_chunk, emb_chunk = self._decode_chunk(
                query_pos=query_pos[start:end],
                query_sdf=None if query_sdf is None else query_sdf[start:end],
                token_features=token_features,
                token_coords=token_coords,
                cond=cond_chunk,
            )
            preds.append(pred_chunk)
            query_embeddings.append(emb_chunk)
        return torch.cat(preds, dim=0), torch.cat(query_embeddings, dim=0)

    def forward_batched(
        self,
        *,
        query_pos: torch.Tensor,
        query_sdf: torch.Tensor | None,
        query_counts: torch.Tensor,
        target_tokens: TokenSet,
        cond: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Decode a padded batched query list against batched target tokens.

        Flattens both queries and target tokens with a per-batch offset on
        the first coordinate axis so the cross-attention's k-NN does not
        mix tokens across batch items. Queries are chunked for memory.

        Parameters
        ----------
        query_pos : torch.Tensor
            Padded query positions of shape ``(B, Nq, 3)``.
        query_sdf : torch.Tensor or None
            Padded SDF of shape ``(B, Nq, 1)``; required when
            ``use_sdf=True`` or when the wall-velocity gate is enabled.
        query_counts : torch.Tensor
            Per-batch valid query counts of shape ``(B,)``.
        target_tokens : TokenSet
            Batched target tokens with positions, features and an
            optional mask.
        cond : torch.Tensor or None
            Per-batch conditioning of shape ``(B, cond_dim)``. Required
            when ``cond_dim > 0``.

        Returns
        -------
        pred : torch.Tensor
            Padded predictions of shape ``(B, Nq, out_channels)``.
        query_embeddings : torch.Tensor
            Padded query embeddings of shape ``(B, Nq, token_dim)``.

        Raises
        ------
        ValueError
            If ``query_pos`` is not rank 3.
        """
        if query_pos.ndim != 3:
            raise ValueError(
                f"forward_batched expects rank-3 query_pos, got {tuple(query_pos.shape)}"
            )
        query_mask = counts_to_mask(query_counts, max_len=int(query_pos.shape[1]))
        target_mask = target_tokens.mask
        if target_mask is None:
            target_mask = torch.ones(
                target_tokens.features.shape[:2],
                device=target_tokens.features.device,
                dtype=torch.bool,
            )
        offset_step = max(
            compute_batch_offset_step(query_pos, query_mask),
            compute_batch_offset_step(target_tokens.coords, target_mask),
        )
        (
            flat_query_coords_raw,
            flat_query_coords,
            query_batch_ids,
        ) = flatten_batched_coords(
            query_pos,
            query_mask,
            offset_step=offset_step,
        )
        _, flat_token_coords, token_batch_ids = flatten_batched_coords(
            target_tokens.coords,
            target_mask,
            offset_step=offset_step,
        )
        flat_query_sdf = (
            None
            if query_sdf is None
            else flatten_padded_batch(query_sdf, query_mask)
        )
        flat_cond = None
        if cond is not None:
            flat_cond = flatten_padded_batch(
                cond.unsqueeze(1).expand(-1, int(query_pos.shape[1]), -1),
                query_mask,
            )
        flat_token_features = flatten_padded_batch(
            target_tokens.features, target_mask
        )

        preds = []
        query_embeddings = []
        total_queries = int(flat_query_coords.shape[0])
        for start in range(0, total_queries, self.query_chunk_size):
            end = min(start + self.query_chunk_size, total_queries)
            cond_chunk = None if flat_cond is None else flat_cond[start:end]
            query_tokens = self.query_in(
                self._build_query_inputs(
                    query_pos=flat_query_coords_raw[start:end],
                    query_sdf=(
                        None if flat_query_sdf is None else flat_query_sdf[start:end]
                    ),
                    cond=cond_chunk,
                )
            )
            for block in self.cross_blocks:
                query_tokens = block(
                    query_tokens,
                    flat_query_coords[start:end],
                    flat_token_features,
                    flat_token_coords,
                    query_batch_ids=query_batch_ids[start:end],
                    context_batch_ids=token_batch_ids,
                )
            hidden = self.trunk(query_tokens)
            if self.final_refinement is not None:
                hidden = hidden + self.final_refinement(hidden)
            pred_chunk = self._apply_head(hidden)
            pred_chunk = self._apply_wall_gate(
                pred_chunk,
                None if flat_query_sdf is None else flat_query_sdf[start:end],
            )
            preds.append(pred_chunk)
            query_embeddings.append(query_tokens)

        flat_pred = torch.cat(preds, dim=0)
        flat_query_embeddings = torch.cat(query_embeddings, dim=0)
        return (
            unflatten_to_padded(flat_pred, query_mask),
            unflatten_to_padded(flat_query_embeddings, query_mask),
        )
