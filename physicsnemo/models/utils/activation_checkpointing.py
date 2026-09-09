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

r"""Shared activation-checkpointing policy and backend helpers."""

from collections.abc import Callable
from typing import Any

import torch
from torch.utils.checkpoint import checkpoint as activation_checkpoint


def resolve_checkpointing_ratio(
    activation_checkpointing: bool, checkpointing_ratio: float
) -> float:
    r"""Validate checkpointing controls and return the effective block ratio."""
    if not isinstance(activation_checkpointing, bool):
        raise TypeError(
            "activation_checkpointing must be bool, got "
            f"{type(activation_checkpointing).__name__}"
        )
    # ``bool`` is a subclass of ``int`` but is not a meaningful ratio here.
    if isinstance(checkpointing_ratio, bool) or not isinstance(
        checkpointing_ratio, (int, float)
    ):
        raise TypeError(
            "checkpointing_ratio must be numeric, got "
            f"{type(checkpointing_ratio).__name__}"
        )

    ratio = float(checkpointing_ratio)
    if not 0.0 <= ratio <= 1.0:
        raise ValueError(f"checkpointing_ratio must be in [0, 1], got {ratio}")
    return ratio if activation_checkpointing else 0.0


def should_checkpoint_interleaved_block(
    block_idx: int,
    block_count: int,
    ratio: float,
    *,
    training: bool,
) -> bool:
    r"""Return whether a block belongs to an evenly interleaved checkpoint set."""
    if not training or not torch.is_grad_enabled() or ratio <= 0.0:
        return False
    if ratio >= 1.0:
        return True

    checkpoint_count = round(ratio * block_count)
    if checkpoint_count <= 0:
        return False

    # Distribute the selected blocks across the stack instead of checkpointing
    # one contiguous prefix. This Bresenham-style rule selects exactly
    # ``checkpoint_count`` blocks with near-uniform spacing, starting at block 0.
    return (block_idx * checkpoint_count) % block_count < checkpoint_count


def run_checkpoint(
    function: Callable[..., Any],
    *inputs: Any,
    use_te: bool,
    te_module: Any,
) -> Any:
    r"""Call the backend-appropriate non-reentrant checkpoint wrapper.

    Non-reentrant checkpointing supports nested inputs and outputs, including
    optional values passed positionally. This lets callers checkpoint the
    canonical model function instead of duplicating it to flatten arguments.
    """
    if use_te:
        return te_module.checkpoint(function, *inputs, use_reentrant=False)
    return activation_checkpoint(function, *inputs, use_reentrant=False)
