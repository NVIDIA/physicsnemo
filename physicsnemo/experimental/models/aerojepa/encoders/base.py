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

"""Abstract base classes for the AeroJEPA encoder family."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from physicsnemo.experimental.nn.aerojepa import EncoderOutput, TokenSet


class BaseContextEncoder(nn.Module, ABC):
    r"""Abstract base class for context encoders.

    A context encoder ingests the geometry-side input and produces an
    :class:`EncoderOutput` carrying context tokens plus an optional global
    token. ``context_pos`` bundles whatever positions the model is
    configured to ingest — for surface-only models this is the boundary
    only; for whole-domain models it is ``cat(boundary_pos,
    volumetric_sample_pos)`` and an SDF channel in ``context_feat``
    distinguishes the two halves.

    Subclasses must implement :meth:`forward`. The :meth:`forward_batched`
    path is optional — concrete encoders that support padded batched
    inputs override it and set ``supports_batched_forward = True`` on the
    class.
    """

    supports_batched_forward = False

    @abstractmethod
    def forward(
        self,
        *,
        context_pos: torch.Tensor,
        context_feat: torch.Tensor,
    ) -> EncoderOutput:
        raise NotImplementedError

    def forward_batched(
        self,
        *,
        context_pos: torch.Tensor,
        context_feat: torch.Tensor,
        context_pos_n: torch.Tensor,
    ) -> EncoderOutput:
        r"""Optional batched forward over a padded batch.

        Default implementation raises :class:`NotImplementedError`.
        Concrete encoders that support padded batched inputs override
        this and set ``supports_batched_forward = True`` on the class.

        Parameters
        ----------
        context_pos : torch.Tensor
            Padded context positions of shape ``(B, N, 3)``.
        context_feat : torch.Tensor
            Padded per-point features of shape ``(B, N, F)``.
        context_pos_n : torch.Tensor
            Per-batch valid point counts of shape ``(B,)``.

        Returns
        -------
        EncoderOutput
            Batched encoded tokens.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement forward_batched"
        )


class BaseTargetEncoder(nn.Module, ABC):
    r"""Abstract base class for target encoders.

    A target encoder runs at training time and produces the supervision
    target for the predictor head. Unlike the context encoder, it sees the
    surface and volume halves of the input *separately*: training-time
    subsampling for the two is intentionally decoupled (different point
    counts, different tokenization strategies are allowed). The encoder
    consumes the context tokens too — some implementations cross-attend
    to them when constructing target tokens.

    At inference, the target encoder is generally bypassed — only its
    tokenizer is reused (via the model's
    ``_build_target_token_coords`` helper) to recover the spatial
    coordinates of every target token.

    Subclasses must implement :meth:`forward`. The :meth:`forward_batched`
    path is optional.
    """

    supports_batched_forward = False

    @abstractmethod
    def forward(
        self,
        *,
        surface_pos: torch.Tensor,
        surface_main_feat: torch.Tensor,
        volume_pos: torch.Tensor,
        volume_feat: torch.Tensor,
        context_tokens: TokenSet,
    ) -> EncoderOutput:
        raise NotImplementedError

    def forward_batched(
        self,
        *,
        surface_pos: torch.Tensor,
        surface_main_feat: torch.Tensor,
        surface_pos_n: torch.Tensor,
        volume_pos: torch.Tensor,
        volume_feat: torch.Tensor,
        volume_pos_n: torch.Tensor,
        context_tokens: TokenSet,
    ) -> EncoderOutput:
        r"""Optional batched forward; see :class:`BaseTargetEncoder`.

        Parameters
        ----------
        surface_pos, surface_main_feat : torch.Tensor
            Padded surface positions and features.
        surface_pos_n : torch.Tensor
            Per-batch valid surface-point counts of shape ``(B,)``.
        volume_pos, volume_feat : torch.Tensor
            Padded volumetric positions and features.
        volume_pos_n : torch.Tensor
            Per-batch valid volume-point counts of shape ``(B,)``.
        context_tokens : TokenSet
            Context tokens forwarded from the context encoder.

        Returns
        -------
        EncoderOutput
            Batched encoded tokens.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement forward_batched"
        )
