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

"""Token-space wiring for AeroJEPA.

:class:`AeroJEPATrunk` owns the context encoder, target encoder, and
field decoder, and wires them together. At training time
:meth:`encode_context` runs both encoders side-by-side and returns the
context tokens, target tokens, and the decoder-side global conditioning
vector. :meth:`decode_queries` decodes a target token set at the supplied
query positions. :meth:`forward_single` / :meth:`forward_batch` chain
the two together.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .encoders.base import BaseContextEncoder, BaseTargetEncoder


class AeroJEPATrunk(nn.Module):
    r"""Owns the context encoder, target encoder, and decoder, and wires them together.

    Parameters
    ----------
    context_encoder : BaseContextEncoder
        Context-side encoder.
    target_encoder : BaseTargetEncoder
        Target-side encoder.
    decoder : torch.nn.Module
        Query-token decoder. Must expose ``forward`` /
        ``forward_batched`` returning ``(pred, query_embeddings)`` and
        carry the ``cond_dim`` / ``use_sdf`` / ``query_in`` attributes
        the mask head reads at construction time.
    include_geometry_global_in_decoder_cond : bool, optional
        Concatenate the context encoder's global token to the
        ``cond_global`` vector handed to the decoder. Default ``True``.
    mask_prediction_enabled : bool, optional
        Build a small per-query mask head that consumes the decoder's
        query embeddings (and optionally ``cond`` / ``query_sdf``) and
        predicts a binary mask logit. Default ``False``.
    mask_head_hidden_dim : int, optional
        Hidden dimension of the mask head MLP. Default 128.
    mask_head_use_cond : bool, optional
        Concatenate the decoder-side conditioning vector to the mask
        head input. Default ``True``.
    """

    def __init__(
        self,
        *,
        context_encoder: BaseContextEncoder,
        target_encoder: BaseTargetEncoder,
        decoder: nn.Module,
        include_geometry_global_in_decoder_cond: bool = True,
        mask_prediction_enabled: bool = False,
        mask_head_hidden_dim: int = 128,
        mask_head_use_cond: bool = True,
    ):
        super().__init__()
        self.context_encoder = context_encoder
        self.target_encoder = target_encoder
        self.decoder = decoder
        self.include_geometry_global_in_decoder_cond = bool(
            include_geometry_global_in_decoder_cond
        )
        self.mask_prediction_enabled = bool(mask_prediction_enabled)
        self.mask_head_use_cond = bool(mask_head_use_cond)

        if self.mask_prediction_enabled:
            token_dim = int(getattr(self.decoder.query_in, "out_features"))
            cond_dim = int(getattr(self.decoder, "cond_dim", 0))
            mask_in_dim = (
                token_dim
                + (cond_dim if self.mask_head_use_cond else 0)
                + (1 if getattr(self.decoder, "use_sdf", True) else 0)
            )
            self.mask_head = nn.Sequential(
                nn.Linear(mask_in_dim, int(mask_head_hidden_dim)),
                nn.SiLU(),
                nn.Linear(int(mask_head_hidden_dim), int(mask_head_hidden_dim)),
                nn.SiLU(),
                nn.Linear(int(mask_head_hidden_dim), 1),
            )
        else:
            self.mask_head = None

    @staticmethod
    def supports_batched_forward() -> bool:
        r"""Always ``True`` — the trunk supports padded batched inputs."""
        return True

    def _build_cond_global_single(
        self,
        *,
        gen_params: torch.Tensor,
        context_global: torch.Tensor | None,
    ) -> torch.Tensor:
        cond_global = gen_params.unsqueeze(0)
        if self.include_geometry_global_in_decoder_cond and context_global is not None:
            cond_global = torch.cat(
                [cond_global, context_global.reshape(1, -1)], dim=-1
            )
        return cond_global

    def _build_cond_global_batched(
        self,
        *,
        gen_params: torch.Tensor,
        context_global: torch.Tensor | None,
    ) -> torch.Tensor:
        cond_global = gen_params
        if self.include_geometry_global_in_decoder_cond and context_global is not None:
            cond_global = torch.cat([cond_global, context_global], dim=-1)
        return cond_global

    def encode_context(
        self,
        *,
        context_pos: torch.Tensor,
        context_feat: torch.Tensor,
        surface_main_feat: torch.Tensor | None = None,
        volume_pos: torch.Tensor,
        volume_feat: torch.Tensor,
        gen_params: torch.Tensor,
    ) -> dict:
        r"""Run the context and target encoders and assemble the decoder-side state.

        Parameters
        ----------
        context_pos, context_feat : torch.Tensor
            Context input. Used by the context encoder as
            ``context_pos`` / ``context_feat``, and re-used by the
            target encoder as its ``surface_pos`` (the surface half of
            the dual-input target).
        surface_main_feat : torch.Tensor, optional
            Surface main-feature tensor passed to the target encoder.
            When ``None``, ``context_feat`` is used as the fallback.
        volume_pos, volume_feat : torch.Tensor
            Volumetric inputs to the target encoder.
        gen_params : torch.Tensor
            Operating conditions. Not forwarded to the encoders;
            used to build the decoder-side ``cond_global``.

        Returns
        -------
        dict
            ``{"context_tokens": TokenSet, "target_tokens": TokenSet,
            "cond_global": torch.Tensor}``.
        """
        context_out = self.context_encoder(
            context_pos=context_pos,
            context_feat=context_feat,
        )
        surface_main = context_feat if surface_main_feat is None else surface_main_feat
        target_out = self.target_encoder(
            surface_pos=context_pos,
            surface_main_feat=surface_main,
            volume_pos=volume_pos,
            volume_feat=volume_feat,
        )
        context_global = context_out.global_token
        if context_global is None:
            context_global = context_out.tokens.global_token
        cond_global = self._build_cond_global_single(
            gen_params=gen_params, context_global=context_global
        )
        return {
            "context_tokens": context_out.tokens,
            "target_tokens": target_out.tokens,
            "cond_global": cond_global,
        }

    def decode_queries(
        self,
        *,
        context: dict,
        query_pos: torch.Tensor,
        query_sdf: torch.Tensor,
        return_mask_logits: bool = False,
    ):
        r"""Decode the field at the supplied query positions.

        Parameters
        ----------
        context : dict
            Dict produced by :meth:`encode_context` carrying
            ``target_tokens`` and ``cond_global``.
        query_pos : torch.Tensor
            Query positions of shape ``(Nq, 3)``.
        query_sdf : torch.Tensor
            Per-query SDF of shape ``(Nq, 1)``.
        return_mask_logits : bool, optional
            When ``True`` and the mask head was enabled, return
            ``(pred, mask_logits)``. Otherwise return ``pred`` only.

        Returns
        -------
        torch.Tensor or tuple
            ``pred`` of shape ``(Nq, out_channels)``; optionally paired
            with ``mask_logits`` of shape ``(Nq, 1)``.
        """
        cond_global = context["cond_global"]
        cond = cond_global.expand(int(query_pos.shape[0]), -1)
        pred, query_embeddings = self.decoder(
            query_pos=query_pos,
            query_sdf=query_sdf,
            target_tokens=context["target_tokens"],
            cond=cond,
        )
        if self.mask_head is None or not return_mask_logits:
            return pred
        mask_inputs = [query_embeddings]
        if self.mask_head_use_cond:
            mask_inputs.append(cond)
        if getattr(self.decoder, "use_sdf", True):
            mask_inputs.append(query_sdf)
        mask_logits = self.mask_head(torch.cat(mask_inputs, dim=-1))
        return pred, mask_logits

    def forward_single(
        self,
        *,
        context_pos: torch.Tensor,
        context_feat: torch.Tensor,
        surface_main_feat: torch.Tensor | None = None,
        volume_pos: torch.Tensor,
        volume_feat: torch.Tensor,
        query_pos: torch.Tensor,
        query_sdf: torch.Tensor,
        gen_params: torch.Tensor,
        return_mask_logits: bool = False,
    ):
        r"""End-to-end single-sample forward: encode then decode.

        Convenience wrapper that chains :meth:`encode_context` and
        :meth:`decode_queries`. See those methods for parameter
        semantics.
        """
        ctx = self.encode_context(
            context_pos=context_pos,
            context_feat=context_feat,
            surface_main_feat=surface_main_feat,
            volume_pos=volume_pos,
            volume_feat=volume_feat,
            gen_params=gen_params,
        )
        return self.decode_queries(
            context=ctx,
            query_pos=query_pos,
            query_sdf=query_sdf,
            return_mask_logits=return_mask_logits,
        )

    def forward_batch(
        self,
        *,
        context_pos: torch.Tensor,
        context_feat: torch.Tensor,
        surface_main_feat: torch.Tensor | None = None,
        context_pos_n: torch.Tensor,
        volume_pos: torch.Tensor,
        volume_feat: torch.Tensor,
        volume_pos_n: torch.Tensor,
        query_pos: torch.Tensor,
        query_sdf: torch.Tensor,
        query_pos_n: torch.Tensor,
        gen_params: torch.Tensor,
        return_mask_logits: bool = False,
    ):
        r"""End-to-end batched forward over padded inputs.

        Runs both encoders' ``forward_batched`` paths, builds the
        decoder-side ``cond_global``, then calls
        ``decoder.forward_batched`` with the batched target tokens and
        queries.

        Parameters
        ----------
        context_pos, context_feat : torch.Tensor
            Padded context input ``(B, N, *)``. Also doubles as the
            target encoder's surface half.
        surface_main_feat : torch.Tensor, optional
            Target encoder's surface main features. ``None`` falls back
            to ``context_feat``.
        context_pos_n : torch.Tensor
            Per-batch valid context-point counts of shape ``(B,)``.
            Also doubles as the target encoder's ``surface_pos_n``.
        volume_pos, volume_feat : torch.Tensor
            Padded volumetric inputs to the target encoder.
        volume_pos_n : torch.Tensor
            Per-batch valid volume-point counts of shape ``(B,)``.
        query_pos, query_sdf : torch.Tensor
            Padded query positions and SDF of shape ``(B, Nq, *)``.
        query_pos_n : torch.Tensor
            Per-batch valid query counts of shape ``(B,)``.
        gen_params : torch.Tensor
            Per-batch operating conditions ``(B, gen_dim)``.
        return_mask_logits : bool, optional
            When ``True`` and the mask head was enabled, return
            ``(pred, mask_logits)``.

        Returns
        -------
        torch.Tensor or tuple
            Padded predictions ``(B, Nq, out_channels)``; optionally
            paired with padded mask logits.
        """
        context_out = self.context_encoder.forward_batched(
            context_pos=context_pos,
            context_feat=context_feat,
            context_pos_n=context_pos_n,
        )
        surface_main = context_feat if surface_main_feat is None else surface_main_feat
        target_out = self.target_encoder.forward_batched(
            surface_pos=context_pos,
            surface_main_feat=surface_main,
            surface_pos_n=context_pos_n,
            volume_pos=volume_pos,
            volume_feat=volume_feat,
            volume_pos_n=volume_pos_n,
        )
        context_global = context_out.global_token
        if context_global is None:
            context_global = context_out.tokens.global_token
        cond_global = self._build_cond_global_batched(
            gen_params=gen_params, context_global=context_global
        )
        pred_batch, query_embeddings = self.decoder.forward_batched(
            query_pos=query_pos,
            query_sdf=query_sdf,
            query_counts=query_pos_n,
            target_tokens=target_out.tokens,
            cond=cond_global,
        )
        if self.mask_head is None or not return_mask_logits:
            return pred_batch
        cond = cond_global.unsqueeze(1).expand(-1, int(query_pos.shape[1]), -1)
        mask_inputs = [query_embeddings]
        if self.mask_head_use_cond:
            mask_inputs.append(cond)
        if getattr(self.decoder, "use_sdf", True):
            mask_inputs.append(query_sdf)
        return pred_batch, self.mask_head(torch.cat(mask_inputs, dim=-1))
