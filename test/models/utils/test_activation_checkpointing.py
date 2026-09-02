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

import pytest
import torch

from physicsnemo.models.utils.activation_checkpointing import (
    resolve_checkpointing_ratio,
    should_checkpoint_interleaved_block,
)


def _checkpoint_mask(block_count: int, ratio: float) -> list[bool]:
    return [
        should_checkpoint_interleaved_block(
            block_idx,
            block_count,
            ratio,
            training=True,
        )
        for block_idx in range(block_count)
    ]


@pytest.mark.parametrize(
    "block_count,ratio,expected",
    [
        (5, 0.0, [False] * 5),
        (5, 0.09, [False] * 5),
        (3, 0.5, [True, False, True]),
        (5, 0.4, [True, False, False, True, False]),
        (5, 0.6, [True, False, True, False, True]),
        (5, 1.0, [True] * 5),
    ],
)
def test_interleaved_block_selector(block_count, ratio, expected):
    assert _checkpoint_mask(block_count, ratio) == expected
    assert sum(expected) == round(ratio * block_count)


def test_interleaved_block_selector_is_training_only():
    assert not should_checkpoint_interleaved_block(0, 5, 1.0, training=False)
    with torch.no_grad():
        assert not should_checkpoint_interleaved_block(0, 5, 1.0, training=True)


@pytest.mark.parametrize(
    "enabled,ratio,error,error_match",
    [
        (1, 1.0, TypeError, "activation_checkpointing"),
        (True, -0.1, ValueError, "checkpointing_ratio"),
        (True, 1.1, ValueError, "checkpointing_ratio"),
        (True, True, TypeError, "checkpointing_ratio"),
        (True, float("nan"), ValueError, "checkpointing_ratio"),
    ],
)
def test_checkpointing_ratio_rejects_invalid_values(enabled, ratio, error, error_match):
    with pytest.raises(error, match=error_match):
        resolve_checkpointing_ratio(enabled, ratio)
