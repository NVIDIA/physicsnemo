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


def _ragged_arange(
    starts: torch.Tensor,
    counts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Expand segment descriptors ``(start, count)`` into flat index arrays.

    Given *N* segments where segment *i* spans positions
    ``[starts[i], starts[i] + counts[i])``, produces two flat tensors of
    length ``sum(counts)``:

    - ``positions[k]``: the absolute index for element *k*
    - ``seg_ids[k]``: the segment (``0..N-1``) that element *k* belongs to

    Conceptually, this concatenates ``arange(s, s+c)`` for each ``(s, c)``
    pair, along with the corresponding segment labels.

    Parameters
    ----------
    starts : torch.Tensor
        Start offset per segment, shape ``(N,)``, int64.
    counts : torch.Tensor
        Element count per segment, shape ``(N,)``, int64.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(positions, seg_ids)`` each with shape ``(sum(counts),)``.
    """
    total = int(counts.sum())
    device = starts.device
    n_segments = starts.shape[0]

    seg_ids = torch.repeat_interleave(
        torch.arange(n_segments, dtype=torch.long, device=device),
        counts,
    )
    # Within-segment offsets: [0, 1, ..., c0-1, 0, 1, ..., c1-1, ...]
    cum = counts.cumsum(0)
    offsets = torch.arange(total, dtype=torch.long, device=device)
    offsets = offsets - torch.repeat_interleave(cum - counts, counts)

    positions = torch.repeat_interleave(starts, counts) + offsets

    return positions, seg_ids
