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

"""Tests for the AeroJEPA loss masking helpers."""

import pytest
import torch

from physicsnemo.experimental.models.aerojepa.losses import (
    flatten_valid_token_features,
    reshape_token_features_for_sigreg,
)

# ---------------------------------------------------------------------------
# flatten_valid_token_features
# ---------------------------------------------------------------------------


def test_flatten_rank2_passthrough(device):
    """Rank-2 input is returned unchanged (same object identity)."""
    x = torch.randn(10, 8, device=device)
    out = flatten_valid_token_features(x)
    assert out is x


def test_flatten_rank3_no_mask(device):
    """Rank-3 input without a mask collapses to ``(B * N, D)``."""
    x = torch.randn(2, 5, 8, device=device)
    out = flatten_valid_token_features(x)
    assert out.shape == (10, 8)


def test_flatten_rank3_with_mask(device):
    """Rank-3 input with a mask returns only the masked-True rows."""
    x = torch.randn(2, 5, 8, device=device)
    mask = torch.tensor(
        [[True, True, True, True, False], [True, True, True, False, False]],
        device=device,
    )
    out = flatten_valid_token_features(x, mask)
    assert out.shape == (7, 8)  # 4 + 3 valid rows


def test_flatten_bad_mask_shape_raises():
    """A mask whose shape disagrees with ``features.shape[:2]`` is rejected."""
    x = torch.zeros(2, 5, 8)
    with pytest.raises(ValueError, match=r"mask must match features.shape"):
        flatten_valid_token_features(x, torch.zeros(3, 5, dtype=torch.bool))


def test_flatten_bad_rank_raises():
    """Rank-4 input is rejected with a clear message."""
    with pytest.raises(ValueError, match=r"rank-2 or rank-3"):
        flatten_valid_token_features(torch.zeros(1, 2, 3, 4))


# ---------------------------------------------------------------------------
# reshape_token_features_for_sigreg
# ---------------------------------------------------------------------------


def test_reshape_adds_leading_t_axis(device):
    """A nonempty flatten becomes ``(1, M, D)``."""
    x = torch.randn(2, 5, 8, device=device)
    out = reshape_token_features_for_sigreg(x)
    assert out.shape == (1, 10, 8)


def test_reshape_with_mask(device):
    """Masking applies before the unsqueeze."""
    x = torch.randn(2, 5, 8, device=device)
    mask = torch.tensor(
        [[True, True, True, True, False], [True, True, True, False, False]],
        device=device,
    )
    out = reshape_token_features_for_sigreg(x, mask)
    assert out.shape == (1, 7, 8)


def test_reshape_all_false_mask_returns_empty_placeholder(device):
    """An all-False mask returns a ``(1, 0, D)`` placeholder, not an error."""
    x = torch.randn(2, 5, 8, device=device)
    mask = torch.zeros(2, 5, dtype=torch.bool, device=device)
    out = reshape_token_features_for_sigreg(x, mask)
    assert out.shape == (1, 0, 8)
