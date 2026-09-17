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
"""Observation cross-attention context and packing utilities.

Build an :class:`ObsContext` with :func:`prepare_obs_context`, which sorts
observations by pixel, builds the ragged prefix sums, and optionally attaches a
:class:`PixelGroupMap` for the fused Triton kernel.
"""

from __future__ import annotations

import dataclasses
from typing import Optional, Tuple

import torch
from jaxtyping import Float, Int

from physicsnemo.core.version_check import OptionalImport

triton = OptionalImport("triton")


def _to_device_cast_float(t, device, dtype, non_blocking):
    # Move to device; cast float payloads only. Integer id/index tensors keep original dtype.
    cast = dtype if (dtype is not None and t.is_floating_point()) else None
    return t.to(device=device, dtype=cast, non_blocking=non_blocking)


@dataclasses.dataclass
class PixelGroupMap:
    r"""CSR map grouping non-empty pixels into shared ragged-attention kernel programs.

    Built by :func:`build_pixel_group_map`; see it for a worked example.

    Attributes
    ----------
    program_ptr : torch.Tensor
        Int prefix sums of shape :math:`(\text{num\_programs} + 1,)`; program
        :math:`p` owns pixels
        :math:`\text{program\_pixels}[\text{program\_ptr}[p]:\text{program\_ptr}[p + 1]]`.
    program_pixels : torch.Tensor
        Int pixel ids of shape :math:`(\text{num\_grouped\_pixels},)`, ordered by program.
    """

    program_ptr: Int[torch.Tensor, " num_programs_plus_one"]
    program_pixels: Int[torch.Tensor, " num_grouped_pixels"]

    def to(self, device=None, dtype=None, non_blocking: bool = True) -> "PixelGroupMap":
        # torch dispatches .to's first positional by type
        if isinstance(device, torch.dtype):
            device, dtype = None, device

        def move(t):
            return _to_device_cast_float(t, device, dtype, non_blocking)

        return PixelGroupMap(
            program_ptr=move(self.program_ptr),
            program_pixels=move(self.program_pixels),
        )


@dataclasses.dataclass
class ObsContext:
    r"""Raw observations plus ragged pixel packing, sorted by flat pixel index.

    Produced by :func:`prepare_obs_context` and consumed by
    :meth:`~physicsnemo.experimental.models.healda.video_healda.VideoHealDA.forward`:
    the raw obs fields feed :class:`~physicsnemo.experimental.models.healda.obs_tokenizer.ObsTokenizerFiLM`,
    and the ragged packing feeds the observation cross-attention. Observations are
    sorted by flat pixel index so each pixel's observations are contiguous.

    :math:`N_{obs}` is the total observation count across all batch elements and
    time frames; :math:`\text{total\_pixels} = B \cdot T \cdot X'` over the
    backbone grid.

    Attributes
    ----------
    obs : torch.Tensor
        Scalar observation measurement values of shape :math:`(N_{obs},)`.
    float_metadata : torch.Tensor
        Per-observation float metadata features of shape :math:`(N_{obs}, M_{float})`.
    obs_type : torch.Tensor
        Observation-type ids of shape :math:`(N_{obs},)`.
    channel : torch.Tensor
        Instrument channel ids of shape :math:`(N_{obs},)`.
    platform : torch.Tensor
        Platform/satellite ids of shape :math:`(N_{obs},)`.
    cu_seqlens_k : torch.Tensor
        Int prefix sums of shape :math:`(\text{total\_pixels} + 1,)` with a
        leading zero; pixel :math:`i` owns observations
        :math:`[\text{cu\_seqlens\_k}[i], \text{cu\_seqlens\_k}[i + 1])`.
    max_seqlen_k : int
        Maximum per-pixel observation count.
    group_map : :class:`PixelGroupMap` or None, optional, default=None
        Groups small pixels into shared ragged-attention kernel programs; ``None``
        disables grouping.
    """

    obs: Float[torch.Tensor, " nobs"]
    float_metadata: Float[torch.Tensor, "nobs meta_dim"]
    obs_type: Int[torch.Tensor, " nobs"]
    channel: Int[torch.Tensor, " nobs"]
    platform: Int[torch.Tensor, " nobs"]
    cu_seqlens_k: Int[torch.Tensor, " total_pixels_plus_one"]
    max_seqlen_k: int
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
        # torch dispatches .to's first positional by type
        if isinstance(device, torch.dtype):
            device, dtype = None, device

        def move(t):
            return _to_device_cast_float(t, device, dtype, non_blocking)

        return ObsContext(
            obs=move(self.obs),
            float_metadata=move(self.float_metadata),
            obs_type=move(self.obs_type),
            channel=move(self.channel),
            platform=move(self.platform),
            cu_seqlens_k=move(self.cu_seqlens_k),
            max_seqlen_k=self.max_seqlen_k,
            group_map=(
                None
                if self.group_map is None
                else self.group_map.to(
                    device=device, dtype=dtype, non_blocking=non_blocking
                )
            ),
        )


# ---------------------------------------------------------------------------
# Packing utilities
#
# Preprocess ragged observations into the sorted/grouped layout the kernel
# consumes: sort_and_pack -> counts_to_cu_seqlens -> build_pixel_group_map.
# They operate on plain index/count tensors, so they are grid- and
# observation-layout agnostic.
# ---------------------------------------------------------------------------


def sort_and_pack(
    flat_idx: Int[torch.Tensor, " nobs"], total_pixels: int
) -> Tuple[Int[torch.Tensor, " nobs"], Int[torch.Tensor, " total_pixels"]]:
    r"""Sort observations by flat pixel index into per-pixel contiguous groups.

    Uses the Triton counting sort (:func:`~physicsnemo.experimental.models.healda.kernels.pixel_attention.counting_sort_and_pack`)
    when triton is available and ``flat_idx`` is on CUDA, else ``argsort``.

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
        from .kernels.pixel_attention import counting_sort_and_pack

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
    ``cu_seqlens_k``, so it is built once per batch and reused by every layer.

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


def prepare_obs_context(
    obs: Float[torch.Tensor, " nobs"],
    float_metadata: Float[torch.Tensor, "nobs meta_dim"],
    obs_type: Int[torch.Tensor, " nobs"],
    channel: Int[torch.Tensor, " nobs"],
    platform: Int[torch.Tensor, " nobs"],
    flat_idx: Int[torch.Tensor, " nobs"],
    total_pixels: int,
    *,
    build_group_map: bool = True,
    group_thresh_mult: float = 2.0,
) -> ObsContext:
    r"""Sort observations by pixel and build an :class:`ObsContext`.

    Parameters
    ----------
    obs : torch.Tensor
        Observation values of shape :math:`(N_{obs},)`.
    float_metadata : torch.Tensor
        Per-observation float metadata of shape :math:`(N_{obs}, M_{float})`.
    obs_type : torch.Tensor
        Observation-type ids of shape :math:`(N_{obs},)`.
    channel : torch.Tensor
        Channel ids of shape :math:`(N_{obs},)`.
    platform : torch.Tensor
        Platform ids of shape :math:`(N_{obs},)`.
    flat_idx : torch.Tensor
        Flat pixel ids of shape :math:`(N_{obs},)` over :math:`B \cdot T \cdot X`.
    total_pixels : int
        Number of pixel buckets :math:`B \cdot T \cdot X`.
    build_group_map : bool, optional, default=True
        If ``True``, attach a :class:`PixelGroupMap` for the fused Triton kernel.
    group_thresh_mult : float, optional, default=2.0
        Small-pixel grouping threshold passed to :func:`build_pixel_group_map`.

    Returns
    -------
    ObsContext
        Packed observation context with per-observation tensors sorted by
        ``flat_idx`` and prefix sums over ``total_pixels``.
    """
    if obs.ndim != 1:
        raise ValueError(
            f"Expected obs of shape (nobs,), got {obs.ndim}D tensor with shape "
            f"{tuple(obs.shape)}"
        )
    nobs = obs.shape[0]
    if float_metadata.ndim != 2 or float_metadata.shape[0] != nobs:
        raise ValueError(
            f"Expected float_metadata of shape ({nobs}, meta_dim), got tensor "
            f"with shape {tuple(float_metadata.shape)}"
        )
    for name, tensor in (
        ("obs_type", obs_type),
        ("channel", channel),
        ("platform", platform),
        ("flat_idx", flat_idx),
    ):
        if tensor.ndim != 1 or tensor.shape[0] != nobs:
            raise ValueError(
                f"Expected {name} of shape ({nobs},), got tensor with shape "
                f"{tuple(tensor.shape)}"
            )

    if nobs == 0:
        counts = torch.zeros(total_pixels, dtype=torch.int64, device=obs.device)
        order = torch.empty(0, dtype=torch.long, device=obs.device)
    else:
        sorted_order, counts = sort_and_pack(flat_idx, total_pixels)
        order = sorted_order.long()

    cu_seqlens_k = counts_to_cu_seqlens(counts)
    group_map = (
        build_pixel_group_map(cu_seqlens_k, thresh_mult=group_thresh_mult)
        if build_group_map
        else None
    )
    return ObsContext(
        obs=obs[order],
        float_metadata=float_metadata[order],
        obs_type=obs_type[order],
        channel=channel[order],
        platform=platform[order],
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_k=int(counts.max()) if nobs > 0 else 0,
        group_map=group_map,
    )
