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

import torch

from .utils import flatten_logits, validate_inputs


def _segment_ids_from_offsets(offsets: torch.Tensor) -> torch.Tensor:
    counts = offsets[1:] - offsets[:-1]
    segment_ids = torch.arange(
        int(offsets.shape[0]) - 1,
        device=offsets.device,
        dtype=torch.int64,
    )
    return torch.repeat_interleave(segment_ids, counts)


def segment_softmax(logits: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """Pure-PyTorch segmented softmax using ``torch.segment_reduce``."""
    validate_inputs(logits, offsets)
    if int(logits.shape[0]) == 0 or int(logits.numel()) == 0:
        return logits.clone()

    offsets = offsets.to(dtype=torch.int64).contiguous()
    flat, original_shape = flatten_logits(logits)
    segment_ids = _segment_ids_from_offsets(offsets)

    max_per_segment = torch.segment_reduce(
        flat,
        "max",
        offsets=offsets,
        axis=0,
    )
    shifted = flat - max_per_segment.index_select(0, segment_ids)
    exp_shifted = shifted.exp()
    sum_per_segment = torch.segment_reduce(
        exp_shifted,
        "sum",
        offsets=offsets,
        axis=0,
    )
    out = exp_shifted / sum_per_segment.index_select(0, segment_ids)
    return out.reshape(original_shape)
