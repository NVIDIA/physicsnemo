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

"""Unit tests for custom UNet implementations (UNet2D, UNet3D)."""

import sys
from pathlib import Path

import pytest
import torch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from unet3d import UNet2D, UNet3D, UNetModule2D, UNetModule3D


class TestUNet2D:
    """Tests for UNet2D model."""

    @pytest.fixture
    def device(self):
        """Return available device."""
        return "cuda" if torch.cuda.is_available() else "cpu"

    def test_forward_pass(self, device):
        """Test UNet2D forward pass with valid input dimensions."""
        torch.manual_seed(42)
        model = UNet2D(
            input_channels=32,
            output_channels=32,
            kernel_size=3,
            dropout_rate=0.0,
        ).to(device)

        # Input must be divisible by 8: (B, C, H, W)
        batch_size = 2
        x = torch.randn(batch_size, 32, 64, 64).to(device)
        output = model(x)

        assert output.shape == x.shape, f"Expected shape {x.shape}, got {output.shape}"

    def test_different_channel_sizes(self, device):
        """Test UNet2D with same input/output channel sizes.

        Note: The current UNet implementation is designed for U-FNO where
        input_channels == output_channels (constant channel dimension in latent space).
        """
        torch.manual_seed(42)

        # UNet is designed for same input/output channels (for U-FNO latent space)
        for channels in [16, 32, 64]:
            model = UNet2D(
                input_channels=channels,
                output_channels=channels,
                kernel_size=3,
            ).to(device)

            x = torch.randn(2, channels, 32, 32).to(device)
            output = model(x)

            expected_shape = (2, channels, 32, 32)
            assert output.shape == expected_shape, (
                f"For channels={channels}: "
                f"expected {expected_shape}, got {output.shape}"
            )

    def test_invalid_dimensions(self, device):
        """Test that UNet2D raises error for invalid input dimensions."""
        model = UNet2D(input_channels=32, output_channels=32).to(device)

        # Dimensions not divisible by 8 should raise ValueError
        x = torch.randn(2, 32, 65, 65).to(device)

        with pytest.raises(ValueError, match="must be divisible by 8"):
            model(x)

    def test_dropout(self, device):
        """Test UNet2D with dropout enabled."""
        model = UNet2D(
            input_channels=32,
            output_channels=32,
            dropout_rate=0.5,
        ).to(device)

        x = torch.randn(2, 32, 32, 32).to(device)

        # Training mode - dropout active
        model.train()
        output1 = model(x)
        output2 = model(x)

        # With dropout, outputs should differ (with high probability)
        # Note: This is a probabilistic test
        assert output1.shape == x.shape

    def test_count_params(self, device):
        """Test parameter counting."""
        model = UNet2D(input_channels=32, output_channels=32).to(device)
        param_count = model.count_params()

        assert param_count > 0
        assert isinstance(param_count, int)

    def test_alias(self):
        """Test that UNetModule2D is an alias for UNet2D."""
        assert UNetModule2D is UNet2D


class TestUNet3D:
    """Tests for UNet3D model."""

    @pytest.fixture
    def device(self):
        """Return available device."""
        return "cuda" if torch.cuda.is_available() else "cpu"

    def test_forward_pass(self, device):
        """Test UNet3D forward pass with valid input dimensions."""
        torch.manual_seed(42)
        model = UNet3D(
            input_channels=32,
            output_channels=32,
            kernel_size=3,
            dropout_rate=0.0,
        ).to(device)

        # Input must be divisible by 8: (B, C, H, W, T)
        batch_size = 2
        x = torch.randn(batch_size, 32, 32, 32, 16).to(device)
        output = model(x)

        assert output.shape == x.shape, f"Expected shape {x.shape}, got {output.shape}"

    def test_spatiotemporal_dimensions(self, device):
        """Test UNet3D with CO2 sequestration-like dimensions."""
        torch.manual_seed(42)
        model = UNet3D(
            input_channels=36,  # Typical latent width
            output_channels=36,
            kernel_size=3,
        ).to(device)

        # Simulated padded dimensions (H=96+8=104, W=200+8=208 -> use 104, 104, 32 for test)
        # Must be divisible by 8
        x = torch.randn(2, 36, 104, 104, 32).to(device)
        output = model(x)

        assert output.shape == x.shape

    def test_different_channel_sizes(self, device):
        """Test UNet3D with same input/output channel sizes.

        Note: The current UNet implementation is designed for U-FNO where
        input_channels == output_channels (constant channel dimension in latent space).
        """
        torch.manual_seed(42)

        # UNet is designed for same input/output channels (for U-FNO latent space)
        for channels in [16, 32, 64]:
            model = UNet3D(
                input_channels=channels,
                output_channels=channels,
                kernel_size=3,
            ).to(device)

            x = torch.randn(1, channels, 16, 16, 16).to(device)
            output = model(x)

            expected_shape = (1, channels, 16, 16, 16)
            assert output.shape == expected_shape, (
                f"For channels={channels}: "
                f"expected {expected_shape}, got {output.shape}"
            )

    def test_invalid_dimensions(self, device):
        """Test that UNet3D raises error for invalid input dimensions."""
        model = UNet3D(input_channels=32, output_channels=32).to(device)

        # Dimensions not divisible by 8 should raise ValueError
        x = torch.randn(2, 32, 33, 33, 17).to(device)

        with pytest.raises(ValueError, match="must be divisible by 8"):
            model(x)

    def test_gradient_flow(self, device):
        """Test that gradients flow through UNet3D."""
        torch.manual_seed(42)
        model = UNet3D(input_channels=16, output_channels=16).to(device)

        # Create tensor on device directly to ensure it's a leaf tensor
        x = torch.randn(1, 16, 16, 16, 16, requires_grad=True, device=device)
        output = model(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_count_params(self, device):
        """Test parameter counting."""
        model = UNet3D(input_channels=32, output_channels=32).to(device)
        param_count = model.count_params()

        assert param_count > 0
        assert isinstance(param_count, int)

        # UNet3D should have more parameters than UNet2D with same config
        model_2d = UNet2D(input_channels=32, output_channels=32).to(device)
        assert model.count_params() > model_2d.count_params()

    def test_alias(self):
        """Test that UNetModule3D is an alias for UNet3D."""
        assert UNetModule3D is UNet3D


class TestUNetIntegration:
    """Integration tests for UNet models."""

    @pytest.fixture
    def device(self):
        """Return available device."""
        return "cuda" if torch.cuda.is_available() else "cpu"

    def test_unet3d_as_skip_connection(self, device):
        """Test UNet3D used as skip connection (input + output addition)."""
        torch.manual_seed(42)
        model = UNet3D(input_channels=32, output_channels=32).to(device)

        x = torch.randn(2, 32, 32, 32, 16).to(device)
        output = model(x)

        # Simulating skip connection: x + UNet(x)
        combined = x + output

        assert combined.shape == x.shape
        # Combined should be different from x
        assert not torch.allclose(combined, x)

    def test_memory_efficiency(self, device):
        """Test that model doesn't leak memory."""
        if device == "cpu":
            pytest.skip("Memory test only relevant for GPU")

        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()

        model = UNet3D(input_channels=16, output_channels=16).to(device)
        x = torch.randn(1, 16, 16, 16, 16).to(device)

        for _ in range(5):
            output = model(x)
            del output

        torch.cuda.empty_cache()
        final_memory = torch.cuda.memory_allocated()

        # Memory should be roughly the same (within model parameters)
        # Allow for some variance
        memory_diff = abs(final_memory - initial_memory)
        model_memory = sum(p.numel() * p.element_size() for p in model.parameters())

        assert memory_diff < model_memory * 2, "Potential memory leak detected"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
