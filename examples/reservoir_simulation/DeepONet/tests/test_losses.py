# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
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

"""Unit tests for loss functions."""

import sys
from pathlib import Path

import pytest
import torch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from losses import SimpleRelativeL2Loss, UnifiedLoss, get_loss_function


class TestSimpleRelativeL2Loss:
    """Tests for SimpleRelativeL2Loss."""

    @pytest.fixture
    def device(self):
        """Return available device."""
        return "cuda" if torch.cuda.is_available() else "cpu"

    def test_zero_loss_for_identical_tensors(self, device):
        """Test that loss is zero when prediction equals target."""
        loss_fn = SimpleRelativeL2Loss()

        target = torch.randn(4, 96, 200, 24).to(device)
        pred = target.clone()

        loss = loss_fn(pred, target)

        assert torch.isclose(loss, torch.tensor(0.0).to(device), atol=1e-6)

    def test_positive_loss_for_different_tensors(self, device):
        """Test that loss is positive when prediction differs from target."""
        loss_fn = SimpleRelativeL2Loss()

        target = torch.randn(4, 96, 200, 24).to(device)
        pred = target + torch.randn_like(target) * 0.1

        loss = loss_fn(pred, target)

        assert loss > 0

    def test_relative_scaling(self, device):
        """Test that relative L2 loss is scale-invariant."""
        loss_fn = SimpleRelativeL2Loss()

        target = torch.randn(2, 32, 32, 8).to(device)
        pred = target + torch.randn_like(target) * 0.1

        loss1 = loss_fn(pred, target)
        loss2 = loss_fn(pred * 10, target * 10)  # Scale both by 10

        # Losses should be approximately equal (scale-invariant)
        assert torch.isclose(loss1, loss2, rtol=1e-4)

    def test_batch_averaging(self, device):
        """Test that loss is averaged across batch."""
        loss_fn = SimpleRelativeL2Loss()

        # Create batch where first sample has zero error, second has error
        target = torch.ones(2, 16, 16, 4).to(device)
        pred = target.clone()
        pred[1] = pred[1] + 0.5  # Add error to second sample

        loss = loss_fn(pred, target)

        # Loss should be average of 0 and some positive value
        assert loss > 0

        # Compare with single-sample loss
        loss_single = loss_fn(pred[1:2], target[1:2])
        # Batch loss should be less than single sample with error
        assert loss < loss_single

    def test_inputs_parameter_ignored(self, device):
        """Test that inputs parameter is ignored (for interface compatibility)."""
        loss_fn = SimpleRelativeL2Loss()

        target = torch.randn(2, 16, 16, 4).to(device)
        pred = target + torch.randn_like(target) * 0.1
        inputs = torch.randn(2, 16, 16, 4, 12).to(device)

        loss_without_inputs = loss_fn(pred, target)
        loss_with_inputs = loss_fn(pred, target, inputs)

        assert torch.isclose(loss_without_inputs, loss_with_inputs)


class TestUnifiedLoss:
    """Tests for UnifiedLoss."""

    @pytest.fixture
    def device(self):
        """Return available device."""
        return "cuda" if torch.cuda.is_available() else "cpu"

    @pytest.mark.parametrize("base_loss_type", ["mse", "l1", "relative_l2"])
    def test_base_loss_types(self, device, base_loss_type):
        """Test different base loss types."""
        loss_fn = UnifiedLoss(
            base_loss_type=base_loss_type,
            use_mask=False,
            use_derivative=False,
        )

        target = torch.randn(2, 32, 32, 8).to(device)
        pred = target + torch.randn_like(target) * 0.1

        loss = loss_fn(pred, target)

        assert loss > 0
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_mse_loss_value(self, device):
        """Test that MSE loss matches expected value."""
        loss_fn = UnifiedLoss(base_loss_type="mse", use_mask=False)

        target = torch.zeros(1, 4, 4, 2).to(device)
        pred = torch.ones(1, 4, 4, 2).to(device)

        loss = loss_fn(pred, target)

        # MSE of (1-0)^2 = 1 for all elements
        assert torch.isclose(loss, torch.tensor(1.0).to(device))

    def test_l1_loss_value(self, device):
        """Test that L1 loss matches expected value."""
        loss_fn = UnifiedLoss(base_loss_type="l1", use_mask=False)

        target = torch.zeros(1, 4, 4, 2).to(device)
        pred = torch.ones(1, 4, 4, 2).to(device) * 2

        loss = loss_fn(pred, target)

        # L1 of |2-0| = 2 for all elements
        assert torch.isclose(loss, torch.tensor(2.0).to(device))

    def test_invalid_base_loss_type(self):
        """Test that invalid base loss type raises error."""
        with pytest.raises(ValueError, match="base_loss_type must be"):
            UnifiedLoss(base_loss_type="invalid")

    def test_invalid_reduction(self):
        """Test that invalid reduction raises error."""
        with pytest.raises(ValueError, match="reduction must be"):
            UnifiedLoss(reduction="invalid")

    def test_mask_requires_inputs(self, device):
        """Test that masking requires inputs."""
        loss_fn = UnifiedLoss(use_mask=True, use_derivative=False)

        target = torch.randn(2, 32, 32, 8).to(device)
        pred = torch.randn(2, 32, 32, 8).to(device)

        with pytest.raises(ValueError, match="inputs must be provided"):
            loss_fn(pred, target, inputs=None)

    def test_derivative_requires_inputs(self, device):
        """Test that derivative loss requires inputs."""
        loss_fn = UnifiedLoss(use_mask=False, use_derivative=True)

        target = torch.randn(2, 32, 32, 8).to(device)
        pred = torch.randn(2, 32, 32, 8).to(device)

        with pytest.raises(ValueError, match="inputs must be provided"):
            loss_fn(pred, target, inputs=None)

    def test_masking(self, device):
        """Test loss with masking enabled."""
        loss_fn = UnifiedLoss(
            base_loss_type="mse",
            use_mask=True,
            use_derivative=False,
        )

        # Create inputs with some zero regions (inactive)
        B, H, W, T, C = 2, 32, 64, 8, 12
        inputs = torch.randn(B, H, W, T, C).to(device)
        inputs[:, :16, :, :, 0] = 0  # First half of height is inactive

        target = torch.randn(B, H, W, T).to(device)
        pred = torch.randn(B, H, W, T).to(device)

        loss = loss_fn(pred, target, inputs)

        assert not torch.isnan(loss)
        assert loss > 0

    def test_derivative_loss(self, device):
        """Test loss with derivative term enabled."""
        loss_fn = UnifiedLoss(
            base_loss_type="mse",
            use_mask=False,
            use_derivative=True,
            derivative_weight=0.5,
        )

        # Create inputs with grid coordinates in channel -3
        B, H, W, T, C = 2, 32, 64, 8, 12
        inputs = torch.randn(B, H, W, T, C).to(device)
        # Set grid_x channel (channel -3) with increasing values
        grid_x = torch.linspace(0, 1, W).to(device)
        inputs[..., -3] = grid_x.view(1, 1, W, 1).expand(B, H, W, T)

        target = torch.randn(B, H, W, T).to(device)
        pred = torch.randn(B, H, W, T).to(device)

        loss = loss_fn(pred, target, inputs)

        assert not torch.isnan(loss)
        assert loss > 0

    @pytest.mark.parametrize("derivative_dim", ["dx", "dz", ["dx", "dz"]])
    def test_derivative_dimensions(self, device, derivative_dim):
        """Test different derivative dimension configurations."""
        loss_fn = UnifiedLoss(
            base_loss_type="mse",
            use_mask=False,
            use_derivative=True,
            derivative_dim=derivative_dim,
        )

        assert loss_fn.derivative_dims == (
            [derivative_dim] if isinstance(derivative_dim, str) else derivative_dim
        )

    def test_gradient_flow(self, device):
        """Test that gradients flow through loss computation."""
        loss_fn = UnifiedLoss(base_loss_type="mse", use_mask=False)

        # Create tensors on device directly to ensure they're leaf tensors
        target = torch.randn(2, 16, 16, 4, device=device)
        pred = torch.randn(2, 16, 16, 4, requires_grad=True, device=device)

        loss = loss_fn(pred, target)
        loss.backward()

        assert pred.grad is not None
        assert pred.grad.shape == pred.shape


class TestGetLossFunction:
    """Tests for loss function factory."""

    def test_simple_relative_l2(self):
        """Test creating SimpleRelativeL2Loss via factory."""
        config = {"base_loss_type": "simple_relative_l2"}
        loss_fn = get_loss_function(config)

        assert isinstance(loss_fn, SimpleRelativeL2Loss)

    @pytest.mark.parametrize("loss_type", ["mse", "l1", "relative_l2"])
    def test_unified_loss_types(self, loss_type):
        """Test creating UnifiedLoss via factory."""
        config = {"base_loss_type": loss_type}
        loss_fn = get_loss_function(config)

        assert isinstance(loss_fn, UnifiedLoss)
        assert loss_fn.base_loss_type == loss_type

    def test_factory_with_all_options(self):
        """Test factory with all configuration options."""
        config = {
            "base_loss_type": "relative_l2",
            "use_mask": True,
            "use_derivative": True,
            "derivative_weight": 0.3,
            "derivative_dim": ["dx", "dz"],
            "eps": 1e-8,
            "reduction": "sum",
        }
        loss_fn = get_loss_function(config)

        assert isinstance(loss_fn, UnifiedLoss)
        assert loss_fn.use_mask is True
        assert loss_fn.use_derivative is True
        assert loss_fn.derivative_weight == 0.3
        assert loss_fn.derivative_dims == ["dx", "dz"]
        assert loss_fn.eps == 1e-8
        assert loss_fn.reduction == "sum"

    def test_factory_defaults(self):
        """Test factory uses correct defaults."""
        config = {}
        loss_fn = get_loss_function(config)

        assert isinstance(loss_fn, UnifiedLoss)
        assert loss_fn.base_loss_type == "relative_l2"
        assert loss_fn.use_mask is False
        assert loss_fn.use_derivative is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
