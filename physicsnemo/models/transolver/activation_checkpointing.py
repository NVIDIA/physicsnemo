# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Activation-checkpointing helpers for Transolver."""

from typing import Any, Callable

import torch
from torch.utils.checkpoint import checkpoint as activation_checkpoint


def parse_checkpointing_param(activation_checkpointing: bool | float) -> float:
    r"""Parse an activation-checkpointing argument into a ratio in ``[0, 1]``."""
    # ``bool`` is a subclass of ``int``, so handle it before numbers.
    if isinstance(activation_checkpointing, bool):
        return 1.0 if activation_checkpointing else 0.0
    if not isinstance(activation_checkpointing, (int, float)):
        raise TypeError(
            "activation_checkpointing must be bool or numeric, got "
            f"{type(activation_checkpointing).__name__}"
        )

    ratio = float(activation_checkpointing)
    if not 0.0 <= ratio <= 1.0:
        raise ValueError(
            f"activation_checkpointing must be bool or a float in [0, 1], got {ratio}"
        )
    return ratio


def should_checkpoint_block(
    block_idx: int,
    block_count: int,
    ratio: float,
    *,
    training: bool,
) -> bool:
    r"""Return whether a leading Transolver block should be checkpointed."""
    if not training or not torch.is_grad_enabled() or ratio <= 0.0:
        return False
    if ratio >= 1.0:
        return True
    return block_idx < round(ratio * block_count)


def checkpoint_block(
    block: Callable[[torch.Tensor], torch.Tensor],
    input_tensor: torch.Tensor,
    *,
    use_te: bool,
    te_module: Any,
) -> torch.Tensor:
    r"""Checkpoint a block with the backend-appropriate implementation."""
    if use_te:
        return te_module.checkpoint(block, input_tensor, use_reentrant=False)
    return activation_checkpoint(block, input_tensor, use_reentrant=False)
