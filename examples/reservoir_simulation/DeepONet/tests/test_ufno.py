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

"""Unit tests for U-FNO model.

Note: These tests require physicsnemo to be installed. If physicsnemo is not
available, all tests in this module will be skipped.
"""

import sys
from pathlib import Path

import pytest
import torch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Check if physicsnemo is available
try:
    import physicsnemo

    PHYSICSNEMO_AVAILABLE = True
except ImportError:
    PHYSICSNEMO_AVAILABLE = False

# Skip entire module if physicsnemo is not installed
if not PHYSICSNEMO_AVAILABLE:
    pytest.skip(
        "physicsnemo not installed - skipping U-FNO tests", allow_module_level=True
    )

from ufno import UFNO, UFNONet


class TestUFNO:
    """Tests for UFNO model."""

    @pytest.fixture
    def device(self):
        """Return available device."""
        return "cuda" if torch.cuda.is_available() else "cpu"

    @pytest.fixture
    def small_model_params(self):
        """Return parameters for a small test model."""
        return {
            "in_channels": 12,
            "out_channels": 1,
            "width": 16,
            "modes1": 4,
            "modes2": 4,
            "modes3": 4,
            "num_fno_layers": 1,
            "num_unet_layers": 1,
            "num_conv_layers": 0,
            "unet_type": "custom",
        }

    def test_forward_pass_ufno(self, device, small_model_params):
        """Test U-FNO (FNO + UNet) forward pass."""
        torch.manual_seed(42)
        model = UFNO(**small_model_params).to(device)

        # Input shape: (B, H, W, T, C) - must be divisible by 8 for UNet
        batch_size = 2
        x = torch.randn(batch_size, 32, 32, 16, 12).to(device)
        output = model(x)

        expected_shape = (batch_size, 32, 32, 16, 1)
        assert output.shape == expected_shape, (
            f"Expected shape {expected_shape}, got {output.shape}"
        )

    def test_forward_pass_convfno(self, device, small_model_params):
        """Test Conv-FNO (FNO + Conv) forward pass."""
        torch.manual_seed(42)
        params = small_model_params.copy()
        params["num_unet_layers"] = 0
        params["num_conv_layers"] = 2

        model = UFNO(**params).to(device)

        batch_size = 2
        x = torch.randn(batch_size, 32, 32, 16, 12).to(device)
        output = model(x)

        expected_shape = (batch_size, 32, 32, 16, 1)
        assert output.shape == expected_shape

    def test_forward_pass_pure_fno(self, device, small_model_params):
        """Test pure FNO (no UNet or Conv) forward pass."""
        torch.manual_seed(42)
        params = small_model_params.copy()
        params["num_unet_layers"] = 0
        params["num_conv_layers"] = 0
        params["num_fno_layers"] = 3

        model = UFNO(**params).to(device)

        batch_size = 2
        x = torch.randn(batch_size, 32, 32, 16, 12).to(device)
        output = model(x)

        expected_shape = (batch_size, 32, 32, 16, 1)
        assert output.shape == expected_shape

    @pytest.mark.parametrize("unet_type", ["custom", "physicsnemo"])
    def test_unet_types(self, device, small_model_params, unet_type):
        """Test different UNet types."""
        torch.manual_seed(42)
        params = small_model_params.copy()
        params["unet_type"] = unet_type

        model = UFNO(**params).to(device)

        x = torch.randn(2, 32, 32, 16, 12).to(device)
        output = model(x)

        assert output.shape == (2, 32, 32, 16, 1)

    def test_invalid_unet_type(self, small_model_params):
        """Test that invalid UNet type raises error."""
        params = small_model_params.copy()
        params["unet_type"] = "invalid"

        with pytest.raises(ValueError, match="Unknown unet_type"):
            UFNO(**params)

    @pytest.mark.parametrize("lifting_type", ["mlp", "conv"])
    def test_lifting_types(self, device, small_model_params, lifting_type):
        """Test different lifting network types."""
        torch.manual_seed(42)
        params = small_model_params.copy()
        params["lifting_type"] = lifting_type

        model = UFNO(**params).to(device)

        x = torch.randn(2, 32, 32, 16, 12).to(device)
        output = model(x)

        assert output.shape == (2, 32, 32, 16, 1)

    @pytest.mark.parametrize("decoder_type", ["mlp", "conv"])
    def test_decoder_types(self, device, small_model_params, decoder_type):
        """Test different decoder network types."""
        torch.manual_seed(42)
        params = small_model_params.copy()
        params["decoder_type"] = decoder_type

        model = UFNO(**params).to(device)

        x = torch.randn(2, 32, 32, 16, 12).to(device)
        output = model(x)

        assert output.shape == (2, 32, 32, 16, 1)

    def test_multi_layer_lifting(self, device, small_model_params):
        """Test multi-layer lifting network."""
        torch.manual_seed(42)
        params = small_model_params.copy()
        params["lifting_layers"] = 3
        params["lifting_width"] = 2

        model = UFNO(**params).to(device)

        x = torch.randn(2, 32, 32, 16, 12).to(device)
        output = model(x)

        assert output.shape == (2, 32, 32, 16, 1)

    def test_multi_layer_decoder(self, device, small_model_params):
        """Test multi-layer decoder network."""
        torch.manual_seed(42)
        params = small_model_params.copy()
        params["decoder_layers"] = 2
        params["decoder_width"] = 64

        model = UFNO(**params).to(device)

        x = torch.randn(2, 32, 32, 16, 12).to(device)
        output = model(x)

        assert output.shape == (2, 32, 32, 16, 1)

    @pytest.mark.parametrize("activation_fn", ["relu", "gelu", "silu"])
    def test_activation_functions(self, device, small_model_params, activation_fn):
        """Test different activation functions."""
        torch.manual_seed(42)
        params = small_model_params.copy()
        params["activation_fn"] = activation_fn

        model = UFNO(**params).to(device)

        x = torch.randn(2, 32, 32, 16, 12).to(device)
        output = model(x)

        assert output.shape == (2, 32, 32, 16, 1)

    def test_gradient_flow(self, device, small_model_params):
        """Test that gradients flow through the model."""
        torch.manual_seed(42)
        model = UFNO(**small_model_params).to(device)

        x = torch.randn(1, 32, 32, 16, 12, requires_grad=True).to(device)
        output = model(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape

        # Check that all parameters have gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_count_params(self, device, small_model_params):
        """Test parameter counting."""
        model = UFNO(**small_model_params).to(device)
        param_count = model.count_params()

        assert param_count > 0
        assert isinstance(param_count, int)

        # Manual count should match
        manual_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert param_count == manual_count

    def test_different_channel_sizes(self, device):
        """Test with different input/output channel configurations."""
        torch.manual_seed(42)

        configs = [
            (12, 1),  # Standard CO2 config
            (8, 1),  # Fewer inputs
            (12, 3),  # Multi-output
        ]

        for in_ch, out_ch in configs:
            model = UFNO(
                in_channels=in_ch,
                out_channels=out_ch,
                width=16,
                modes1=4,
                modes2=4,
                modes3=4,
                num_fno_layers=1,
                num_unet_layers=0,
                num_conv_layers=0,
            ).to(device)

            x = torch.randn(1, 32, 32, 16, in_ch).to(device)
            output = model(x)

            expected_shape = (1, 32, 32, 16, out_ch)
            assert output.shape == expected_shape, (
                f"For in_ch={in_ch}, out_ch={out_ch}: "
                f"expected {expected_shape}, got {output.shape}"
            )


class TestUFNONet:
    """Tests for UFNONet wrapper."""

    @pytest.fixture
    def device(self):
        """Return available device."""
        return "cuda" if torch.cuda.is_available() else "cpu"

    def test_forward_with_padding(self, device):
        """Test UFNONet handles padding correctly."""
        torch.manual_seed(42)
        model = UFNONet(
            modes1=4,
            modes2=4,
            modes3=4,
            width=16,
            in_channels=12,
            out_channels=1,
            num_fno_layers=1,
            num_unet_layers=1,
            padding=8,
            unet_type="custom",
        ).to(device)

        # Input shape: (B, H, W, T, C)
        x = torch.randn(2, 32, 40, 16, 12).to(device)
        output = model(x)

        # Output should match input spatial dims without channel dim
        expected_shape = (2, 32, 40, 16)
        assert output.shape == expected_shape, (
            f"Expected shape {expected_shape}, got {output.shape}"
        )

    def test_padding_adjustment(self, device):
        """Test that padding is adjusted to be divisible by 8."""
        # Padding not divisible by 8
        model = UFNONet(
            modes1=4,
            modes2=4,
            modes3=4,
            width=16,
            padding=10,  # Will be adjusted to 16
            unet_type="custom",
            num_unet_layers=0,
        ).to(device)

        assert model.padding == 16  # Adjusted to next multiple of 8

    def test_count_params(self, device):
        """Test parameter counting via wrapper."""
        model = UFNONet(
            modes1=4,
            modes2=4,
            modes3=4,
            width=16,
            num_fno_layers=1,
            num_unet_layers=0,
            unet_type="custom",
        ).to(device)

        param_count = model.count_params()
        assert param_count > 0


class TestUFNOIntegration:
    """Integration tests for UFNO model."""

    @pytest.fixture
    def device(self):
        """Return available device."""
        return "cuda" if torch.cuda.is_available() else "cpu"

    def test_training_step(self, device):
        """Test a simulated training step."""
        torch.manual_seed(42)
        model = UFNO(
            in_channels=12,
            out_channels=1,
            width=16,
            modes1=4,
            modes2=4,
            modes3=4,
            num_fno_layers=1,
            num_unet_layers=1,
            unet_type="custom",
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.MSELoss()

        # Simulated batch
        x = torch.randn(2, 32, 32, 16, 12).to(device)
        target = torch.randn(2, 32, 32, 16, 1).to(device)

        # Training step
        model.train()
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        assert not torch.isnan(loss)
        assert loss > 0

    def test_eval_mode(self, device):
        """Test model evaluation mode."""
        torch.manual_seed(42)
        model = UFNO(
            in_channels=12,
            out_channels=1,
            width=16,
            modes1=4,
            modes2=4,
            modes3=4,
            num_fno_layers=1,
            num_unet_layers=1,
            unet_type="custom",
            unet_dropout=0.5,  # Enable dropout
        ).to(device)

        x = torch.randn(2, 32, 32, 16, 12).to(device)

        # In eval mode, outputs should be deterministic
        model.eval()
        with torch.no_grad():
            output1 = model(x)
            output2 = model(x)

        assert torch.allclose(output1, output2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
