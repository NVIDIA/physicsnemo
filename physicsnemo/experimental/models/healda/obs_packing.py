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
"""Packing metadata for ragged observation cross-attention.

Copied as-is from the healda data pipeline; describes the packed per-pixel
observation-token layout consumed by
:class:`..pixel_cross_attention.PixelCrossAttention`.
"""

import dataclasses
from typing import Optional

import torch


@dataclasses.dataclass
class PixelGroupMap:
    """CSR map for grouping non-empty pixels into shared attention programs."""

    program_ptr: torch.Tensor
    program_pixels: torch.Tensor

    def to(self, device=None, dtype=None, non_blocking=True):
        # dtype is intentionally ignored: group-map tensors are integer indices.
        del dtype
        return PixelGroupMap(
            program_ptr=self.program_ptr.to(device=device, non_blocking=non_blocking),
            program_pixels=self.program_pixels.to(
                device=device, non_blocking=non_blocking
            ),
        )


@dataclasses.dataclass
class AttentionPacking:
    """Precomputed packing metadata for backbone observation cross-attention.

    Observations are sorted by flat pixel index so each pixel's key/value tokens
    are contiguous; ``cu_seqlens_k`` holds the prefix sums into the packed token
    array. ``counts`` is the per-pixel observation count. The sort permutation is
    applied to the observation tensors in the data transform and is not retained
    here: this struct only describes the resulting packed layout.
    """

    counts: torch.Tensor
    cu_seqlens_k: torch.Tensor
    max_seqlen_k: int
    npix: int
    hpx_level: int
    pixel_order: str
    is_packed: bool = True
    # Optional CSR map pairing small pixels into shared kernel programs for the
    # ragged obs cross-attention. ``None`` -> one program per pixel.
    group_map: Optional[PixelGroupMap] = None

    def to(self, device=None, dtype=None, non_blocking=True):
        # dtype is intentionally ignored: packing tensors are integer indices.
        del dtype

        def _move(x):
            if x is None:
                return None
            return x.to(device=device, non_blocking=non_blocking)

        return AttentionPacking(
            counts=_move(self.counts),
            cu_seqlens_k=_move(self.cu_seqlens_k),
            max_seqlen_k=self.max_seqlen_k,
            npix=self.npix,
            hpx_level=self.hpx_level,
            pixel_order=self.pixel_order,
            is_packed=self.is_packed,
            group_map=(
                None
                if self.group_map is None
                else self.group_map.to(device=device, non_blocking=non_blocking)
            ),
        )
