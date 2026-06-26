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

"""Packing utilities for ragged pixel/observation cross-attention.

Companion preprocessing helpers for
:class:`..pixel_cross_attention.PixelCrossAttention`: sort observations into
per-pixel contiguous groups (:func:`sort_and_pack`), turn per-pixel counts into
the ``cu_seqlens_k`` prefix sums the kernel addresses by
(:func:`counts_to_cu_seqlens`), and pack small pixels into shared kernel programs
(:func:`build_pixel_group_map`). They operate on plain index/count tensors, so
they are grid- and observation-layout agnostic. The data pipeline calls them to
populate :class:`..obs_context.ObsContext`; the model never does.
"""

from __future__ import annotations

from typing import Tuple

import torch
from jaxtyping import Int

from physicsnemo.core.version_check import OptionalImport
from physicsnemo.experimental.models.healda.obs_context import PixelGroupMap

triton = OptionalImport("triton")

if triton.available:
    tl = triton.language

    @triton.jit
    def _counting_sort_scatter(
        keys_ptr,
        sorted_order_ptr,
        bucket_offsets_ptr,
        N,
        BLOCK: tl.constexpr,
    ):
        # Counting sort over N items keyed by a bounded integer: each item
        # atomically claims the next free slot in its key's bucket and writes its
        # source index there, producing a key-grouped permutation in one pass.
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        key = tl.load(keys_ptr + offs, mask=mask).to(tl.int64)
        pos = tl.atomic_add(bucket_offsets_ptr + key, 1, mask=mask)
        tl.store(sorted_order_ptr + pos.to(tl.int32), offs.to(tl.int32), mask=mask)


def counting_sort_and_pack(
    flat_idx: Int[torch.Tensor, " nobs"], total_pixels: int
) -> Tuple[Int[torch.Tensor, " nobs"], Int[torch.Tensor, " total_pixels"]]:
    r"""Sort observations by flat pixel index with a Triton counting sort (CUDA only).

    For bounded integer keys a counting sort is :math:`O(N)` in a single
    atomic-scatter pass, faster than ``argsort``'s multi-pass radix sort.
    Within-bucket order is non-deterministic, which is fine for attention
    (permutation-invariant over a pixel's tokens).

    Parameters
    ----------
    flat_idx : torch.Tensor
        Int per-observation flat pixel indices of shape :math:`(N_{obs},)`, each
        in :math:`[0, \text{total\_pixels})`.
    total_pixels : int
        Number of pixel buckets (:math:`B \cdot T \cdot X`).

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        ``sorted_order`` (int32 permutation of shape :math:`(N_{obs},)`) and
        ``counts`` (int64 per-pixel counts of shape
        :math:`(\text{total\_pixels},)`).
    """
    n = flat_idx.shape[0]
    device = flat_idx.device
    counts = torch.bincount(flat_idx.long(), minlength=total_pixels)
    bucket_offsets = torch.zeros(total_pixels, dtype=torch.int64, device=device)
    bucket_offsets[1:] = counts[:-1].cumsum(0)
    sorted_order = torch.empty(n, dtype=torch.int32, device=device)
    block = 1024
    grid = ((n + block - 1) // block,)
    _counting_sort_scatter[grid](flat_idx, sorted_order, bucket_offsets, n, BLOCK=block)
    return sorted_order, counts


def sort_and_pack(
    flat_idx: Int[torch.Tensor, " nobs"], total_pixels: int
) -> Tuple[Int[torch.Tensor, " nobs"], Int[torch.Tensor, " total_pixels"]]:
    r"""Sort observations by flat pixel index into per-pixel contiguous groups.

    Uses the Triton counting sort (:func:`counting_sort_and_pack`) when triton is
    available and ``flat_idx`` is on CUDA, else ``argsort``.

    Parameters
    ----------
    flat_idx : torch.Tensor
        Int per-observation flat pixel indices of shape :math:`(N_{obs},)`, each
        in :math:`[0, \text{total\_pixels})`.
    total_pixels : int
        Number of pixel buckets (:math:`B \cdot T \cdot X`).

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        ``sorted_order`` (int32 permutation) that reorders the per-observation
        tensors so each pixel's tokens are contiguous, and ``counts`` (int64
        per-pixel counts) that :func:`counts_to_cu_seqlens` turns into
        ``cu_seqlens_k``.
    """
    if triton.available and flat_idx.is_cuda:
        return counting_sort_and_pack(flat_idx, total_pixels)
    counts = torch.bincount(flat_idx.long(), minlength=total_pixels)
    sorted_order = flat_idx.argsort().int()
    return sorted_order, counts


def counts_to_cu_seqlens(
    counts: Int[torch.Tensor, " total_pixels"],
) -> Int[torch.Tensor, " total_pixels_plus_one"]:
    r"""Prefix-sum per-pixel ``counts`` into ``cu_seqlens_k``.

    Parameters
    ----------
    counts : torch.Tensor
        Int per-pixel token counts of shape :math:`(\text{total\_pixels},)`.

    Returns
    -------
    torch.Tensor
        Int32 prefix sums of shape :math:`(\text{total\_pixels} + 1,)` with a
        leading zero; pixel :math:`i` owns tokens
        :math:`[\text{cu\_seqlens\_k}[i], \text{cu\_seqlens\_k}[i + 1])`.
    """
    cu_seqlens_k = torch.zeros(
        counts.shape[0] + 1, dtype=torch.int32, device=counts.device
    )
    cu_seqlens_k[1:] = counts.cumsum(0).to(torch.int32)
    return cu_seqlens_k


def build_pixel_group_map(
    cu_seqlens_k: Int[torch.Tensor, " total_pixels_plus_one"],
    thresh_mult: float = 2.0,
) -> PixelGroupMap:
    r"""Pack consecutive small pixels into shared ragged-attention kernel programs.

    The ragged attention runs one kernel program per pixel; for the many tiny
    pixels the fixed per-program cost (``W_k`` / ``W_v`` load, prologue, launch
    latency) dominates the actual math. Pairing two small pixels into one program
    loads those weights once and cuts the program count.

    A pixel is "small" when its token count is below ``thresh_mult`` times the
    median of the non-empty counts (median-relative, so it keeps grouping when the
    typical pixel is large). Empty pixels are dropped. A pure function of
    ``cu_seqlens_k``, so it is built once per batch and reused by every layer and
    both passes.

    Parameters
    ----------
    cu_seqlens_k : torch.Tensor
        Int prefix sums of shape :math:`(\text{total\_pixels} + 1,)`, as produced
        by :func:`counts_to_cu_seqlens`.
    thresh_mult : float, optional, default=2.0
        Small-pixel threshold as a multiple of the median non-empty count.

    Returns
    -------
    PixelGroupMap
        ``program_ptr`` of shape :math:`(\text{num\_programs} + 1,)` and
        ``program_pixels`` of shape :math:`(\text{num\_nonzero\_pixels},)`, both
        int32 on the input device; program :math:`p` owns pixels
        ``program_pixels[program_ptr[p]:program_ptr[p + 1]]``.

    Examples
    --------
    For counts ``[5, 0, 3, 4, 200]`` (non-empty median 4, threshold 8): large
    ``[4]``, small ``[0, 2, 3]``. Large pixels go first, each solo; small pixels
    are then paired (an odd one left solo), giving programs
    ``[[4], [0, 2], [3]]`` -- ``program_ptr = [0, 1, 3, 4]`` and
    ``program_pixels = [4, 0, 2, 3]``. Pixel 1 is empty and dropped.
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
