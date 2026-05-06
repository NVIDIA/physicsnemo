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

"""Segmented (ragged) tensor utilities for spatial data structures."""

import torch
from jaxtyping import Int


def _ragged_arange(
    starts: Int[torch.Tensor, " n_segments"],
    counts: Int[torch.Tensor, " n_segments"],
    total: int | torch.SymInt | None = None,
) -> tuple[Int[torch.Tensor, " n_total"], Int[torch.Tensor, " n_total"]]:
    r"""Expand segment descriptors ``(start, count)`` into flat index arrays.

    Given :math:`N` segments where segment ``i`` spans positions
    ``[starts[i], starts[i] + counts[i])``, produces two flat tensors of
    length ``sum(counts)``:

    - ``positions[k]``: the absolute index for element ``k``.
    - ``seg_ids[k]``: the segment (``0..N-1``) that element ``k`` belongs to.

    Conceptually, this concatenates ``arange(s, s+c)`` for each ``(s, c)``
    pair, along with the corresponding segment labels.

    The implementation uses ``searchsorted`` rather than
    ``repeat_interleave``, so it is fully traceable by ``torch.compile``.

    Parameters
    ----------
    starts : torch.Tensor
        Start offset per segment, shape ``(N,)``, int64.
    counts : torch.Tensor
        Element count per segment, shape ``(N,)``, int64.
        Entries may be zero (those segments produce no output elements).
    total : int | torch.SymInt | None, optional
        Pre-computed ``counts.sum()``.  When available from a tensor shape
        (e.g. ``some_tensor.shape[0]``), passing it avoids an internal
        ``.item()`` call and the associated ``torch.compile`` graph break.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(positions, seg_ids)`` each with shape ``(sum(counts),)``.
    """
    device = starts.device
    if total is None:
        total = counts.sum()

    # Exclusive prefix sum: flat-space start of each segment.
    seg_start_flat = counts.cumsum(0) - counts  # (N,)

    # For each flat position, find the owning segment via binary search.
    # searchsorted(right=True) returns the index *after* the last matching
    # entry, so subtracting 1 gives the last segment whose flat start is
    # <= the query position.  This correctly skips zero-count segments
    # (which share a seg_start_flat value with the next non-zero segment).
    flat_idx = torch.arange(total, dtype=torch.long, device=device)
    seg_ids = torch.searchsorted(seg_start_flat, flat_idx, right=True) - 1

    # Within-segment offsets: [0, 1, ..., c0-1, 0, 1, ..., c1-1, ...]
    intra_offset = flat_idx - seg_start_flat[seg_ids]
    positions = starts[seg_ids] + intra_offset

    return positions, seg_ids
