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

"""Automatic inactive-cell mask detection for reservoir grids.

Detection priority:

1. **Explicit channel** — caller supplies a channel index.
2. **ACTNUM** — binary {0, 1} channel, static across time, whose zeros
   coincide with output-inactive cells.
3. **Non-zero fallback** — any channel with a static zero pattern
   matching the output's inactive cells.
4. **No mask** — all cells treated as active.
"""

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor


@dataclass
class MaskResult:
    """Outcome of mask detection.

    Attributes
    ----------
    channel : int or None
        Input channel index used as the mask source.
    per_sample : bool
        ``True`` when the mask varies across samples (built per-batch).
    static_mask : Tensor or None
        ``(*spatial)`` boolean mask when the mask is the same for all
        samples; ``None`` when *per_sample* is ``True`` or no mask.
    method : str
        Human-readable label: ``'config'``, ``'actnum'``,
        ``'nonzero'``, or ``'none'``.
    n_active : int
        Number of active cells in sample 0 (for logging).
    n_total : int
        Total number of spatial cells (for logging).
    """

    channel: Optional[int]
    per_sample: bool
    static_mask: Optional[Tensor]
    method: str
    n_active: int
    n_total: int


def detect_mask(
    input_data: Tensor,
    output_data: Tensor,
    config_channel: Optional[int] = None,
) -> MaskResult:
    """Detect the best inactive-cell mask for a reservoir dataset.

    Parameters
    ----------
    input_data : Tensor
        Full input tensor ``(N, *spatial, T, C)``.
    output_data : Tensor
        Full output tensor ``(N, *spatial, T)``.
    config_channel : int or None
        Explicit mask channel from config (highest priority).

    Returns
    -------
    MaskResult
    """
    if config_channel is not None:
        return _from_explicit_channel(input_data, config_channel)

    result = _find_actnum_channel(input_data, output_data)
    if result is not None:
        return result

    result = _find_nonzero_channel(input_data, output_data)
    if result is not None:
        return result

    return MaskResult(
        channel=None,
        per_sample=False,
        static_mask=None,
        method="none",
        n_active=0,
        n_total=0,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_N_CROSS_CHECK = 3


def _output_inactive(output_data: Tensor) -> Tensor:
    """Boolean mask of cells that are zero across all timesteps in sample 0."""
    return output_data[0].abs().sum(dim=-1) == 0


def _check_consistency(input_data: Tensor, reference: Tensor, channel: int) -> bool:
    """True if the mask is identical across the first few samples."""
    n = min(input_data.shape[0], _N_CROSS_CHECK)
    return all(
        torch.equal(reference, input_data[si][..., 0, channel]) for si in range(1, n)
    )


def _build_result(
    input_data: Tensor,
    channel: int,
    consistent: bool,
    method: str,
) -> MaskResult:
    mask_s0 = input_data[0][..., 0, channel] != 0
    static = mask_s0 if consistent else None
    return MaskResult(
        channel=channel,
        per_sample=not consistent,
        static_mask=static,
        method=method,
        n_active=int(mask_s0.sum().item()),
        n_total=int(mask_s0.numel()),
    )


def _from_explicit_channel(input_data: Tensor, channel: int) -> MaskResult:
    col = input_data[0][..., 0, channel]
    n = min(input_data.shape[0], _N_CROSS_CHECK)
    consistent = all(
        torch.equal(col != 0, input_data[si][..., 0, channel] != 0)
        for si in range(1, n)
    )
    return _build_result(input_data, channel, consistent, "config")


def _find_actnum_channel(
    input_data: Tensor, output_data: Tensor
) -> Optional[MaskResult]:
    """Binary {0,1} channel, static across time, zeros subset of inactive."""
    s0 = input_data[0]
    out_inactive = _output_inactive(output_data)
    candidates = []
    for ch in range(s0.shape[-1]):
        col = s0[..., 0, ch]
        vals = col.unique()
        if not (vals.numel() <= 2 and all(v in (0.0, 1.0) for v in vals.tolist())):
            continue
        if not torch.equal(s0[..., 0, ch], s0[..., -1, ch]):
            continue
        zeros = col == 0
        if (zeros & ~out_inactive).any():
            continue
        consistent = _check_consistency(input_data, col, ch)
        candidates.append((ch, zeros.sum().item(), consistent))

    if not candidates:
        return None
    best_ch, _, best_consistent = max(candidates, key=lambda x: x[1])
    return _build_result(input_data, best_ch, best_consistent, "actnum")


def _find_nonzero_channel(
    input_data: Tensor, output_data: Tensor
) -> Optional[MaskResult]:
    """Channel with a static zero pattern matching output-inactive cells."""
    s0 = input_data[0]
    out_inactive = _output_inactive(output_data)
    candidates = []
    for ch in range(s0.shape[-1]):
        zeros_t0 = s0[..., 0, ch] == 0
        n_zeros = zeros_t0.sum().item()
        if n_zeros == 0 or n_zeros == zeros_t0.numel():
            continue
        if not torch.equal(zeros_t0, s0[..., -1, ch] == 0):
            continue
        if (zeros_t0 & ~out_inactive).any():
            continue
        n = min(input_data.shape[0], _N_CROSS_CHECK)
        consistent = all(
            torch.equal(zeros_t0, input_data[si][..., 0, ch] == 0) for si in range(1, n)
        )
        candidates.append((ch, n_zeros, consistent))

    if not candidates:
        return None
    best_ch, _, best_consistent = max(candidates, key=lambda x: x[1])
    return _build_result(input_data, best_ch, best_consistent, "nonzero")
