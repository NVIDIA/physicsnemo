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

"""Tests for the AeroJEPA reconstruction loss family."""

import pytest
import torch

from physicsnemo.experimental.models.aerojepa.losses import (
    MSELoss,
    RelativeL2Loss,
    RelativeL2MSELoss,
    RelativeMSELoss,
    mse_loss,
    relative_l2_loss,
    relative_l2_mse_loss,
    relative_mse_loss,
)

# ---------------------------------------------------------------------------
# Plain MSE
# ---------------------------------------------------------------------------


def test_mse_function_module_agree(device):
    """The ``MSELoss`` module wraps ``mse_loss`` exactly."""
    pred = torch.randn(8, 3, device=device)
    target = torch.randn(8, 3, device=device)
    func = mse_loss(pred, target)
    mod = MSELoss().to(device)(pred, target)
    assert torch.allclose(func, mod)


def test_mse_zero_at_identity(device):
    """``pred == target`` yields exact zero MSE."""
    x = torch.randn(8, 3, device=device)
    assert float(mse_loss(x, x)) == 0.0


def test_mse_channel_weights_persistent_buffer():
    """Supplied ``channel_weights`` persists in the Module's ``state_dict``."""
    m = MSELoss(channel_weights=[1.0, 2.0, 3.0])
    sd = m.state_dict()
    assert "channel_weights" in sd
    assert sd["channel_weights"].shape == (3,)


def test_mse_no_channel_weights_omits_from_state_dict():
    """No ``channel_weights`` means no entry in ``state_dict``."""
    m = MSELoss()
    assert "channel_weights" not in m.state_dict()


def test_mse_channel_weights_function_module_agree(device):
    """Channel-weighted function and channel-weighted Module agree."""
    pred = torch.randn(8, 3, device=device)
    target = torch.randn(8, 3, device=device)
    cw = torch.tensor([1.0, 2.0, 3.0], device=device)
    func = mse_loss(pred, target, channel_weights=cw)
    mod = MSELoss(channel_weights=[1.0, 2.0, 3.0]).to(device)(pred, target)
    assert torch.allclose(func, mod)


def test_mse_mask_drops_invalid_rows(device):
    """Masked-out rows contribute neither to the numerator nor the denominator."""
    pred = torch.randn(8, 3, device=device)
    target = torch.randn(8, 3, device=device)
    full_mask = torch.ones(8, dtype=torch.bool, device=device)
    base = mse_loss(pred, target, mask=full_mask)
    no_mask = mse_loss(pred, target)
    assert torch.allclose(base, no_mask)


def test_mse_shape_mismatch_raises():
    """Disagreeing shapes are rejected with a clear error."""
    with pytest.raises(ValueError, match=r"shapes must match"):
        mse_loss(torch.zeros(3, 4), torch.zeros(3, 5))


# ---------------------------------------------------------------------------
# Per-channel relative L2
# ---------------------------------------------------------------------------


def test_relative_l2_function_module_agree(device):
    """``RelativeL2Loss`` wraps ``relative_l2_loss`` exactly."""
    pred = torch.randn(8, 3, device=device)
    target = torch.randn(8, 3, device=device)
    func = relative_l2_loss(pred, target)
    mod = RelativeL2Loss().to(device)(pred, target)
    assert torch.allclose(func, mod)


def test_relative_l2_zero_at_identity(device):
    """``pred == target`` yields exact zero (modulo eps)."""
    x = torch.randn(8, 3, device=device)
    assert float(relative_l2_loss(x, x)) == 0.0


def test_relative_l2_batched(device):
    """Rank-3 inputs produce a scalar by averaging over batch and channels."""
    pred = torch.randn(4, 10, 3, device=device)
    target = torch.randn(4, 10, 3, device=device)
    out = relative_l2_loss(pred, target)
    assert out.ndim == 0


def test_relative_l2_low_rank_raises():
    """Rank-1 inputs are rejected."""
    with pytest.raises(ValueError, match=r"ndim >= 2"):
        relative_l2_loss(torch.zeros(5), torch.zeros(5))


# ---------------------------------------------------------------------------
# Relative MSE with selectable mode
# ---------------------------------------------------------------------------


def test_relative_mse_function_module_agree(device):
    """``RelativeMSELoss`` wraps ``relative_mse_loss`` exactly in both modes."""
    pred = torch.randn(8, 3, device=device)
    target = torch.randn(8, 3, device=device)
    for mode in ("pointwise", "channel_max"):
        func = relative_mse_loss(pred, target, mode=mode)
        mod = RelativeMSELoss(mode=mode).to(device)(pred, target)
        assert torch.allclose(func, mod), f"mode={mode} disagrees"


def test_relative_mse_modes_differ(device):
    """``pointwise`` and ``channel_max`` produce different values on random inputs."""
    pred = torch.randn(20, 3, device=device)
    target = torch.randn(20, 3, device=device)
    a = relative_mse_loss(pred, target, mode="pointwise")
    b = relative_mse_loss(pred, target, mode="channel_max")
    assert not torch.allclose(a, b)


def test_relative_mse_unknown_mode_raises():
    """An unrecognised mode is rejected."""
    with pytest.raises(ValueError, match=r"pointwise.*channel_max"):
        relative_mse_loss(torch.zeros(8, 3), torch.zeros(8, 3), mode="bogus")


def test_relative_mse_channel_max_requires_rank_2_plus(device):
    """``channel_max`` mode rejects rank-1 inputs."""
    with pytest.raises(ValueError, match=r"ndim >= 2"):
        relative_mse_loss(
            torch.zeros(5, device=device),
            torch.zeros(5, device=device),
            mode="channel_max",
        )


# ---------------------------------------------------------------------------
# Hybrid: relative L2 + MSE
# ---------------------------------------------------------------------------


def test_hybrid_function_module_agree(device):
    """``RelativeL2MSELoss`` wraps ``relative_l2_mse_loss`` exactly."""
    pred = torch.randn(8, 3, device=device)
    target = torch.randn(8, 3, device=device)
    func = relative_l2_mse_loss(pred, target)
    mod = RelativeL2MSELoss().to(device)(pred, target)
    assert torch.allclose(func, mod)


def test_hybrid_degenerates_to_l2_when_mse_weight_zero(device):
    """``mse_weight=0`` makes the hybrid equal a scaled relative-L2 loss."""
    pred = torch.randn(8, 3, device=device)
    target = torch.randn(8, 3, device=device)
    hybrid = relative_l2_mse_loss(pred, target, relative_l2_weight=1.0, mse_weight=0.0)
    pure = relative_l2_loss(pred, target)
    assert torch.allclose(hybrid, pure)


def test_hybrid_degenerates_to_mse_when_l2_weight_zero(device):
    """``relative_l2_weight=0`` makes the hybrid equal a scaled MSE loss."""
    pred = torch.randn(8, 3, device=device)
    target = torch.randn(8, 3, device=device)
    hybrid = relative_l2_mse_loss(pred, target, relative_l2_weight=0.0, mse_weight=1.0)
    pure = mse_loss(pred, target)
    assert torch.allclose(hybrid, pure)


def test_hybrid_default_weights():
    """Default constructor uses ``relative_l2_weight=1.0`` and ``mse_weight=0.1``."""
    m = RelativeL2MSELoss()
    assert m.relative_l2_weight == 1.0
    assert m.mse_weight == 0.1
