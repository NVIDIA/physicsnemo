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
"""Packed observation inputs for ragged observation cross-attention.

A single :class:`ObsCrossAttention` bundle carries everything a video DiT block's
observation sub-layer needs -- the packed observation tokens plus the ragged
packing metadata that maps each pixel to its token slice -- consumed by
:class:`..pixel_cross_attention.PixelCrossAttention`.
"""

from __future__ import annotations

import dataclasses
from typing import Optional

import torch
from jaxtyping import Int


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
class ObsCrossAttention:
    """Packed observation tokens + ragged packing for observation cross-attention.

    Bundled into one object (rather than passing tokens and packing metadata
    separately) so a block's observation sub-layer takes a single argument.
    Observations are sorted by flat pixel index so each pixel's tokens are
    contiguous in ``tokens``.

    Parameters
    ----------
    tokens : torch.Tensor
        Packed observation tokens (the key/value source) of shape
        :math:`(N_{tokens}, \\text{token\\_dim})`, concatenated over all pixels.
    cu_seqlens_k : torch.Tensor
        Int prefix sums of shape :math:`(\\text{total\\_pixels} + 1,)`; pixel
        ``i`` attends to ``tokens[cu_seqlens_k[i]:cu_seqlens_k[i + 1]]``.
    max_seqlen_k : int
        Maximum per-pixel token count (kernel launch / autotune bucketing).
    group_map : PixelGroupMap, optional
        Optional CSR map packing small pixels into shared kernel programs.
        ``None`` runs one program per pixel.
    """

    tokens: torch.Tensor
    cu_seqlens_k: Int[torch.Tensor, " total_pixels_plus_one"]
    max_seqlen_k: int
    group_map: Optional[PixelGroupMap] = None

    def to(
        self, device=None, dtype=None, non_blocking: bool = True
    ) -> "ObsCrossAttention":
        return ObsCrossAttention(
            tokens=self.tokens.to(
                device=device, dtype=dtype, non_blocking=non_blocking
            ),
            cu_seqlens_k=self.cu_seqlens_k.to(device=device, non_blocking=non_blocking),
            max_seqlen_k=self.max_seqlen_k,
            group_map=(
                None
                if self.group_map is None
                else self.group_map.to(device=device, non_blocking=non_blocking)
            ),
        )
