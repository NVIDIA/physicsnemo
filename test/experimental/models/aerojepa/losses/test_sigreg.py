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

"""Tests for the SIGReg latent regularizer and its token wrapper."""

import pytest
import torch

from physicsnemo.experimental.models.aerojepa.losses import SIGReg, TokenLatentSIGReg

# ---------------------------------------------------------------------------
# SIGReg
# ---------------------------------------------------------------------------


def test_sigreg_buffers_no_parameters(device):
    """SIGReg owns three non-learnable buffers and no parameters."""
    sr = SIGReg(knots=17, num_proj=32).to(device)
    buffer_names = {name for name, _ in sr.named_buffers()}
    assert {"t", "phi", "weights"} <= buffer_names
    assert sum(p.numel() for p in sr.parameters()) == 0


def test_sigreg_knots_lower_bound_raises():
    """``knots < 2`` is rejected at construction."""
    with pytest.raises(ValueError, match=r"knots >= 2"):
        SIGReg(knots=1)


def test_sigreg_forward_returns_scalar(device):
    """Forward on rank-3 input returns a scalar tensor."""
    sr = SIGReg(knots=17, num_proj=32).to(device)
    out = sr(torch.randn(1, 100, 16, device=device))
    assert out.ndim == 0
    assert torch.isfinite(out)


def test_sigreg_bad_rank_raises(device):
    """A non-rank-3 input is rejected with a shape message."""
    sr = SIGReg(num_proj=8).to(device)
    with pytest.raises(ValueError, match=r"\[T, B, D\]"):
        sr(torch.randn(5, 3, device=device))


def test_sigreg_empty_batch_returns_zero(device):
    """An empty B or D axis returns a zero scalar with the input dtype."""
    sr = SIGReg(num_proj=8).to(device)
    out = sr(torch.zeros(1, 0, 16, device=device))
    assert out.ndim == 0
    assert float(out) == 0.0


def test_sigreg_default_num_proj():
    """Default ``num_proj`` is 1024 and is stored as a Python int."""
    sr = SIGReg()
    assert sr.num_proj == 1024


# ---------------------------------------------------------------------------
# TokenLatentSIGReg
# ---------------------------------------------------------------------------


def test_token_latent_sigreg_wraps_sigreg(device):
    """The wrapper exposes an inner ``regularizer`` attribute of type SIGReg."""
    tl = TokenLatentSIGReg(knots=17, num_proj=8).to(device)
    assert isinstance(tl.regularizer, SIGReg)


def test_token_latent_sigreg_unbatched_forward(device):
    """Rank-2 features without a mask produce a scalar regularization."""
    tl = TokenLatentSIGReg(num_proj=8).to(device)
    out = tl(torch.randn(50, 16, device=device))
    assert out.ndim == 0
    assert torch.isfinite(out)


def test_token_latent_sigreg_batched_with_mask(device):
    """Rank-3 features with a partial mask produce a scalar regularization."""
    tl = TokenLatentSIGReg(num_proj=8).to(device)
    feats = torch.randn(2, 30, 16, device=device)
    mask = torch.cat(
        [
            torch.ones(2, 25, dtype=torch.bool, device=device),
            torch.zeros(2, 5, dtype=torch.bool, device=device),
        ],
        dim=1,
    )
    out = tl(feats, mask)
    assert out.ndim == 0
    assert torch.isfinite(out)


def test_token_latent_sigreg_all_false_mask_short_circuits(device):
    """An all-False mask short-circuits to a zero scalar without calling SIGReg."""
    tl = TokenLatentSIGReg(num_proj=8).to(device)
    feats = torch.randn(2, 30, 16, device=device)
    mask = torch.zeros(2, 30, dtype=torch.bool, device=device)
    out = tl(feats, mask)
    assert out.ndim == 0
    assert float(out) == 0.0
