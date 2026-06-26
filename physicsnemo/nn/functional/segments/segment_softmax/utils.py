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


def validate_inputs(logits: torch.Tensor, offsets: torch.Tensor) -> None:
    """Validate common ``segment_softmax`` inputs."""
    if logits.ndim < 1:
        raise ValueError(f"logits must have rank >= 1, got rank {logits.ndim}")
    if not torch.is_floating_point(logits):
        raise ValueError(f"logits must be floating point, got dtype {logits.dtype}")
    if offsets.ndim != 1:
        raise ValueError(f"offsets must be a rank-1 tensor, got rank {offsets.ndim}")
    if offsets.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"offsets must have dtype int32 or int64, got {offsets.dtype}")
    if offsets.device != logits.device:
        raise ValueError(
            "logits and offsets must be on the same device, got "
            f"{logits.device} and {offsets.device}"
        )
    if int(offsets.shape[0]) < 1:
        raise ValueError("offsets must contain at least one element")

    if torch.compiler.is_compiling():
        return

    if int(offsets[0].item()) != 0:
        raise ValueError(f"offsets must start at 0, got {int(offsets[0].item())}")
    last_offset = int(offsets[-1].item())
    if last_offset != int(logits.shape[0]):
        raise ValueError(
            "offsets[-1] must equal logits.shape[0], got "
            f"{last_offset} and {int(logits.shape[0])}"
        )
    if bool(torch.any(offsets[1:] < offsets[:-1]).item()):
        raise ValueError("offsets must be monotonically nondecreasing")


def flatten_logits(logits: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
    """Flatten trailing logit dimensions to ``(num_entries, channels)``."""
    original_shape = tuple(int(dim) for dim in logits.shape)
    num_entries = original_shape[0]
    channels = 1
    for dim in original_shape[1:]:
        channels *= dim
    flat = logits.reshape(num_entries, channels)
    return flat, original_shape
