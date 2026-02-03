# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
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
import torch

from physicsnemo.experimental.models.healda import (
    HPXPatchDetokenizer,
    HPXPatchTokenizer,
)


def test_hpx_patch_tokenizer():
    """Test HPXPatchTokenizer forward pass."""
    in_channels = 5
    hidden_size = 64
    level_fine = 6
    level_coarse = 4

    tokenizer = HPXPatchTokenizer(
        in_channels=in_channels,
        hidden_size=hidden_size,
        level_fine=level_fine,
        level_coarse=level_coarse,
    )

    b, t = 2, 3
    npix = 12 * 4**level_fine
    x = torch.randn(b, in_channels, t, npix)

    second_of_day = torch.randint(0, 86400, (b, t))
    day_of_year = torch.randint(0, 365, (b, t))

    out = tokenizer(x, second_of_day=second_of_day, day_of_year=day_of_year)

    # Output should be (B, L, D) where L = T * npix_coarse
    expected_npix_coarse = 12 * 4**level_coarse
    expected_L = t * expected_npix_coarse
    assert out.shape == (b, expected_L, hidden_size)

    # Verify output is finite
    assert torch.isfinite(out).all(), "Output contains non-finite values"


def test_hpx_patch_tokenizer_no_calendar():
    """Test HPXPatchTokenizer without calendar embedding."""
    tokenizer = HPXPatchTokenizer(
        in_channels=5,
        hidden_size=64,
        level_fine=6,
        level_coarse=4,
    )

    b, t = 2, 1
    npix = 12 * 4**6
    x = torch.randn(b, 5, t, npix)

    # No calendar embedding
    out = tokenizer(x)

    expected_L = t * 12 * 4**4
    assert out.shape == (b, expected_L, 64)


def test_hpx_patch_detokenizer():
    """Test HPXPatchDetokenizer forward pass."""
    hidden_size = 64
    out_channels = 5
    level_coarse = 4
    level_fine = 6
    time_length = 3

    detokenizer = HPXPatchDetokenizer(
        hidden_size=hidden_size,
        out_channels=out_channels,
        level_coarse=level_coarse,
        level_fine=level_fine,
        time_length=time_length,
    )

    b = 2
    npix_coarse = 12 * 4**level_coarse
    L = time_length * npix_coarse
    x = torch.randn(b, L, hidden_size)
    c = torch.randn(b, hidden_size)  # Conditioning

    out = detokenizer(x, c)

    # Output should be (B, C, T, npix_fine)
    npix_fine = 12 * 4**level_fine
    assert out.shape == (b, out_channels, time_length, npix_fine)
    assert torch.isfinite(out).all()


def test_hpx_patch_detokenizer_vit_mode():
    """Test HPXPatchDetokenizer in VIT mode (c=zeros)."""
    hidden_size = 64
    out_channels = 5
    level_coarse = 4
    level_fine = 6
    time_length = 3

    detokenizer = HPXPatchDetokenizer(
        hidden_size=hidden_size,
        out_channels=out_channels,
        level_coarse=level_coarse,
        level_fine=level_fine,
        time_length=time_length,
    )
    # Zero-initialize (as in DiT)
    detokenizer.initialize_weights()

    b = 2
    npix_coarse = 12 * 4**level_coarse
    L = time_length * npix_coarse
    x = torch.randn(b, L, hidden_size)

    # VIT mode: pass zeros for c
    c = torch.zeros(b, hidden_size)
    out = detokenizer(x, c)

    npix_fine = 12 * 4**level_fine
    assert out.shape == (b, out_channels, time_length, npix_fine)
    assert torch.isfinite(out).all()


def test_tokenizer_detokenizer_roundtrip():
    """Test that tokenizer -> detokenizer roundtrip produces correct shapes."""
    in_channels = 5
    hidden_size = 64
    level_fine = 6
    level_coarse = 4
    time_length = 2

    tokenizer = HPXPatchTokenizer(
        in_channels=in_channels,
        hidden_size=hidden_size,
        level_fine=level_fine,
        level_coarse=level_coarse,
    )

    detokenizer = HPXPatchDetokenizer(
        hidden_size=hidden_size,
        out_channels=in_channels,
        level_coarse=level_coarse,
        level_fine=level_fine,
        time_length=time_length,
    )

    b = 2
    npix_fine = 12 * 4**level_fine
    x = torch.randn(b, in_channels, time_length, npix_fine)
    second_of_day = torch.randint(0, 86400, (b, time_length))
    day_of_year = torch.randint(0, 365, (b, time_length))
    c = torch.randn(b, hidden_size)

    # Tokenize
    tokens = tokenizer(x, second_of_day=second_of_day, day_of_year=day_of_year)
    npix_coarse = 12 * 4**level_coarse
    assert tokens.shape == (b, time_length * npix_coarse, hidden_size)

    # Detokenize
    out = detokenizer(tokens, c)
    assert out.shape == x.shape


def test_tokenizer_backward():
    """Test HPXPatchTokenizer backward pass."""
    tokenizer = HPXPatchTokenizer(
        in_channels=5,
        hidden_size=64,
        level_fine=6,
        level_coarse=4,
    )

    b, t = 2, 1
    npix = 12 * 4**6
    x = torch.randn(b, 5, t, npix, requires_grad=True)
    second_of_day = torch.randint(0, 86400, (b, t))
    day_of_year = torch.randint(0, 365, (b, t))

    out = tokenizer(x, second_of_day=second_of_day, day_of_year=day_of_year)
    out.sum().backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape
