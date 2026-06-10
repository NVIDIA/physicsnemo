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

"""Tests for ``TokenSet``-coupled batching helpers."""

import pytest
import torch

from physicsnemo.experimental.models.aerojepa.layers import (
    TokenSet,
    pad_token_sets,
    trim_batched_tokens,
)

# ---------------------------------------------------------------------------
# trim_batched_tokens / pad_token_sets
# ---------------------------------------------------------------------------


def test_pad_token_sets_packs_lists(device):
    """``pad_token_sets`` builds a padded batched ``TokenSet`` plus mask."""
    ts1 = TokenSet(
        features=torch.randn(3, 4, device=device),
        coords=torch.randn(3, 3, device=device),
    )
    ts2 = TokenSet(
        features=torch.randn(5, 4, device=device),
        coords=torch.randn(5, 3, device=device),
    )
    packed = pad_token_sets([ts1, ts2])
    assert packed.is_batched is True
    assert packed.features.shape == (2, 5, 4)
    assert packed.mask.sum().item() == 8  # 3 + 5
    assert packed.global_token.shape == (2, 4)


def test_pad_token_sets_empty_raises():
    """Empty iterable is rejected."""
    with pytest.raises(ValueError, match="at least one TokenSet"):
        pad_token_sets([])


def test_trim_batched_tokens(device):
    """Slice one batch element to a shorter length."""
    ts = TokenSet(
        features=torch.randn(2, 6, 4, device=device),
        coords=torch.randn(2, 6, 3, device=device),
        mask=torch.ones(2, 6, device=device, dtype=torch.bool),
        global_token=torch.randn(2, 4, device=device),
    )
    out = trim_batched_tokens(ts, index=1, count=3)
    assert out.is_batched is False
    assert out.features.shape == (3, 4)
    assert out.coords.shape == (3, 3)
    assert out.mask.shape == (3,)
    assert out.global_token.shape == (1, 4)


def test_trim_batched_tokens_requires_batched():
    """Unbatched input is rejected."""
    ts = TokenSet(features=torch.zeros(3, 4), coords=torch.zeros(3, 3))
    with pytest.raises(ValueError, match="batched TokenSet"):
        trim_batched_tokens(ts, index=0, count=2)
