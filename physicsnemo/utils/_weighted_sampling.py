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

"""Shared weighted sampling operations."""

import torch

_WEIGHTED_SAMPLE_CHUNK_SIZE = 1 << 22


def weighted_sample_without_replacement(
    weights: torch.Tensor,
    count: int,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample indices by an uncapped, chunked exponential race.

    Each chunk contributes its local ``count`` smallest keys. The global
    ``count`` smallest keys must be in that union, so chunking reduces temporary
    memory without changing the weighted-without-replacement distribution.

    Unlike :func:`torch.multinomial`, the number of input categories is not
    limited to :math:`2^{24}`.

    Parameters
    ----------
    weights : torch.Tensor
        One-dimensional, floating-point sampling weights.
    count : int
        Number of unique indices to sample.
    generator : torch.Generator, optional
        Generator used for the exponential draws.

    Returns
    -------
    torch.Tensor
        Sampled indices with shape ``(count,)``.
    """
    if weights.ndim != 1:
        raise ValueError(
            f"weights must be 1D, got {weights.ndim}D with {weights.shape=}."
        )
    if not weights.is_floating_point():
        raise TypeError(f"weights must be floating point, got {weights.dtype=}.")
    if count < 0 or count > weights.shape[0]:
        raise ValueError(
            f"count must be between 0 and the number of weights, got "
            f"{count=} and {weights.shape[0]=}."
        )
    if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
        raise RuntimeError("weights must contain only finite, non-negative values.")
    if weights.numel() and not bool((weights > 0).any()):
        raise RuntimeError("weights must contain at least one positive value.")
    if count == weights.shape[0]:
        return torch.arange(weights.shape[0], device=weights.device)

    candidate_keys = []
    candidate_indices = []
    tiny = torch.finfo(weights.dtype).tiny
    for start in range(0, weights.shape[0], _WEIGHTED_SAMPLE_CHUNK_SIZE):
        stop = min(start + _WEIGHTED_SAMPLE_CHUNK_SIZE, weights.shape[0])
        chunk_weights = weights[start:stop].clamp_min(tiny)
        keys = torch.empty_like(chunk_weights).exponential_(generator=generator)
        keys.div_(chunk_weights)
        local_count = min(count, stop - start)
        local_keys, local_indices = torch.topk(
            keys,
            local_count,
            largest=False,
            sorted=False,
        )
        candidate_keys.append(local_keys)
        candidate_indices.append(local_indices + start)

    if len(candidate_keys) == 1:
        return candidate_indices[0]

    keys = torch.cat(candidate_keys)
    indices = torch.cat(candidate_indices)
    selected = torch.topk(keys, count, largest=False, sorted=False).indices
    return indices[selected]
