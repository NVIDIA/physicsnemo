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
"""Observation cross-attention context container.

A single :class:`ObsContext` carries everything a video DiT block's observation
cross-attention needs -- the packed observation tokens plus the ragged packing
metadata that maps each pixel to its token slice -- consumed by
:class:`..pixel_cross_attention.PixelCrossAttention`. The packing itself is built
by :mod:`..pixel_cross_attention`.
"""

from __future__ import annotations

import dataclasses
from typing import Optional

import torch
from jaxtyping import Float, Int


@dataclasses.dataclass
class PixelGroupMap:
    """CSR map grouping non-empty pixels into shared ragged-attention programs.

    Parameters
    ----------
    program_ptr : torch.Tensor
        Int tensor of shape :math:`(\\text{num\\_programs} + 1,)`; program ``p``
        owns the pixels ``program_pixels[program_ptr[p]:program_ptr[p + 1]]``.
    program_pixels : torch.Tensor
        Int tensor of shape :math:`(\\text{num\\_grouped\\_pixels},)` listing the
        pixel ids assigned to each program.
    """

    program_ptr: Int[torch.Tensor, " num_programs_plus_one"]
    program_pixels: Int[torch.Tensor, " num_grouped_pixels"]

    def to(self, device=None, dtype=None, non_blocking: bool = True) -> "PixelGroupMap":
        # dtype is intentionally ignored: group-map tensors are integer indices.
        del dtype
        return PixelGroupMap(
            program_ptr=self.program_ptr.to(device=device, non_blocking=non_blocking),
            program_pixels=self.program_pixels.to(
                device=device, non_blocking=non_blocking
            ),
        )


@dataclasses.dataclass
class ObsContext:
    """Observation cross-attention context: packed tokens + ragged packing.

    The single container a block's observation sub-layer
    (:class:`..pixel_cross_attention.PixelCrossAttention`) consumes. It carries the
    raw per-observation arrays the tokenizer needs plus the ragged packing.
    Observations are sorted by flat pixel index so each pixel's tokens are
    contiguous in ``tokens``.

    Parameters
    ----------
    cu_seqlens_k : torch.Tensor
        Int prefix sums of shape :math:`(\\text{total\\_pixels} + 1,)`; pixel
        ``i`` attends to ``tokens[cu_seqlens_k[i]:cu_seqlens_k[i + 1]]``.
    max_seqlen_k : int
        Maximum per-pixel token count (kernel launch / autotune bucketing).
    values : torch.Tensor, optional
        Scalar observation values of shape :math:`(N_{obs},)`.
    float_metadata : torch.Tensor, optional
        Per-observation float metadata of shape :math:`(N_{obs}, M)`.
    obs_type : torch.Tensor, optional
        Observation-type ids of shape :math:`(N_{obs},)`.
    channel : torch.Tensor, optional
        Channel ids of shape :math:`(N_{obs},)`.
    platform : torch.Tensor, optional
        Platform ids of shape :math:`(N_{obs},)`.
    tokens : torch.Tensor, optional
        Packed observation tokens (the key/value source) of shape
        :math:`(N_{obs}, \\text{token\\_dim})`, concatenated over all pixels.
        Unset (``None``) until the observation tokenizer fills it.
    group_map : PixelGroupMap, optional
        Optional CSR map packing small pixels into shared kernel programs.
        ``None`` runs one program per pixel.
    """

    cu_seqlens_k: Int[torch.Tensor, " total_pixels_plus_one"]
    max_seqlen_k: int
    values: Optional[Float[torch.Tensor, " nobs"]] = None
    float_metadata: Optional[Float[torch.Tensor, "nobs meta_dim"]] = None
    obs_type: Optional[Int[torch.Tensor, " nobs"]] = None
    channel: Optional[Int[torch.Tensor, " nobs"]] = None
    platform: Optional[Int[torch.Tensor, " nobs"]] = None
    tokens: Optional[torch.Tensor] = None
    group_map: Optional[PixelGroupMap] = None

    def __post_init__(self) -> None:
        # Cheap, sync-free structural check at construction: the packing is a 1D
        # prefix sum over total_pixels + 1 entries. Per-element/value invariants
        # (token counts, pixel-id ranges) are the producer's responsibility.
        if self.cu_seqlens_k.ndim != 1:
            raise ValueError(
                "cu_seqlens_k must be 1D of shape (total_pixels + 1,); got shape "
                f"{tuple(self.cu_seqlens_k.shape)}"
            )

    def to(self, device=None, dtype=None, non_blocking: bool = True) -> "ObsContext":
        def move(t, cast):
            if t is None:
                return None
            return t.to(
                device=device,
                dtype=dtype if cast else None,
                non_blocking=non_blocking,
            )

        return ObsContext(
            cu_seqlens_k=self.cu_seqlens_k.to(device=device, non_blocking=non_blocking),
            max_seqlen_k=self.max_seqlen_k,
            values=move(self.values, cast=True),
            float_metadata=move(self.float_metadata, cast=True),
            obs_type=move(self.obs_type, cast=False),
            channel=move(self.channel, cast=False),
            platform=move(self.platform, cast=False),
            tokens=move(self.tokens, cast=True),
            group_map=(
                None
                if self.group_map is None
                else self.group_map.to(device=device, non_blocking=non_blocking)
            ),
        )
