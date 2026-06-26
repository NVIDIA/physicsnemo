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

A single :class:`ObsContext` carries everything a video DiT block's observation
sub-layer needs -- the packed observation tokens plus the ragged packing
metadata that maps each pixel to its token slice -- consumed by
:class:`..pixel_cross_attention.PixelCrossAttention`.
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


def build_pixel_group_map(
    cu_seqlens_k: Int[torch.Tensor, " total_pixels_plus_one"],
    thresh_mult: float = 2.0,
) -> PixelGroupMap:
    """Pack consecutive small pixels into shared kernel programs for obs attention.

    The ragged attention runs one kernel program per pixel; for the many tiny
    pixels the fixed per-program cost (``W_k`` / ``W_v`` load, prologue, launch
    latency) dominates the actual math. Pairing two small pixels into one program
    loads those weights once and cuts the program count.

    A pixel is "small" when its token count is below ``thresh_mult`` times the
    median of the non-empty counts (median-relative so it keeps grouping when the
    typical pixel is large). Empty pixels are dropped. Pure function of
    ``cu_seqlens_k``, so it is built once per batch and reused by every layer and
    both passes.

    Parameters
    ----------
    cu_seqlens_k : torch.Tensor
        Int prefix sums of shape :math:`(\\text{total\\_pixels} + 1,)`.
    thresh_mult : float, optional, default=2.0
        Small-pixel threshold as a multiple of the median non-empty count.

    Returns
    -------
    PixelGroupMap
        ``program_ptr`` of shape :math:`(\\text{num\\_programs} + 1,)` and
        ``program_pixels`` of shape :math:`(\\text{num\\_nonzero\\_pixels},)`,
        both int32 on the input device.
    """
    device = cu_seqlens_k.device
    counts = (cu_seqlens_k[1:] - cu_seqlens_k[:-1]).to(torch.int64)
    nonzero_pixels = torch.nonzero(counts > 0, as_tuple=False).flatten()
    if nonzero_pixels.numel() == 0:  # frame with no observations
        return PixelGroupMap(
            program_ptr=torch.zeros(1, dtype=torch.int32, device=device),
            program_pixels=torch.empty(0, dtype=torch.int32, device=device),
        )
    nonzero_counts = counts[nonzero_pixels].float()
    threshold = nonzero_counts.median() * thresh_mult
    is_small = nonzero_counts < threshold
    small_pixels = nonzero_pixels[is_small]
    large_pixels = nonzero_pixels[~is_small]

    # Large pixels stay solo; small pixels are taken two at a time, with a final
    # solo program if an odd one is left over.
    num_pairs = small_pixels.numel() // 2
    has_leftover = small_pixels.numel() % 2 == 1
    program_sizes = torch.cat(
        [
            torch.ones(large_pixels.numel(), dtype=torch.int64, device=device),
            torch.full((num_pairs,), 2, dtype=torch.int64, device=device),
            torch.ones(int(has_leftover), dtype=torch.int64, device=device),
        ]
    )
    program_ptr = torch.zeros(
        program_sizes.numel() + 1, dtype=torch.int32, device=device
    )
    program_ptr[1:] = torch.cumsum(program_sizes, 0).to(torch.int32)
    program_pixels = torch.cat(
        [large_pixels.to(torch.int32), small_pixels.to(torch.int32)]
    )
    return PixelGroupMap(
        program_ptr=program_ptr.contiguous(),
        program_pixels=program_pixels.contiguous(),
    )
