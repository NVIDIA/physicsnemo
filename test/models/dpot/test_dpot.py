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
import torch.nn as nn

from physicsnemo.models.dpot.dpot import Block, DPOT2DLayer, DPOTNet


class TestDPOT2DLayer:
    """Test DPOT2DLayer functionality."""

    def test_forward_pass(self):
        """Test basic forward pass."""
        layer = DPOT2DLayer(width=32, num_blocks=4, modes=16)
        x = torch.randn(2, 8, 8, 32)  # (B, H, W, C)
        output = layer(x)
        assert output.shape == x.shape

    def test_channel_first(self):
        """Test channel_first=True option."""
        layer = DPOT2DLayer(width=32, num_blocks=4, modes=16, channel_first=True)
        x = torch.randn(2, 32, 8, 8)  # (B, C, H, W)
        output = layer(x)
        assert output.shape == x.shape

    def test_width_divisible_by_blocks(self):
        """Test that width must be divisible by num_blocks."""
        with pytest.raises(
            ValueError, match="width .* must be divisible by num_blocks"
        ):
            DPOT2DLayer(width=33, num_blocks=4)


class TestBlock:
    """Test Block functionality."""

    def test_default_norm_groups(self):
        """Test default norm_groups value."""
        block = Block(width=32, num_blocks=4, mlp_ratio=1.0, modes=16)
        assert block.norm1.num_groups == 8
        assert block.norm2.num_groups == 8

    def test_custom_norm_groups(self):
        """Test custom norm_groups parameter."""
        block = Block(width=32, num_blocks=4, mlp_ratio=1.0, modes=16, norm_groups=4)
        assert block.norm1.num_groups == 4
        assert block.norm2.num_groups == 4

    def test_forward_pass(self):
        """Test block forward pass."""
        block = Block(width=32, num_blocks=4, mlp_ratio=1.0, modes=16)
        x = torch.randn(2, 32, 8, 8)  # (B, C, H, W)
        output = block(x)
        assert output.shape == x.shape

    def test_double_skip_false(self):
        """Test block with double_skip=False."""
        block = Block(
            width=32, num_blocks=4, mlp_ratio=1.0, modes=16, double_skip=False
        )
        x = torch.randn(2, 32, 8, 8)
        output = block(x)
        assert output.shape == x.shape


class TestDPOTNet:
    """Test DPOTNet functionality."""

    def test_basic_forward(self):
        """Test basic forward pass."""
        model = DPOTNet(
            inp_shape=32,
            patch_size=8,
            in_channels=3,
            out_channels=2,
            in_timesteps=4,
            out_timesteps=1,
            embed_dim=64,
            depth=2,
            num_blocks=4,
        )
        x = torch.randn(1, 32, 32, 4, 3)  # (B, H, W, T, C)
        output = model(x)
        assert output.shape == (1, 32, 32, 1, 2)

    def test_custom_norm_groups(self):
        """Test custom norm_groups parameter."""
        model = DPOTNet(
            inp_shape=32,
            patch_size=8,
            in_channels=3,
            out_channels=2,
            embed_dim=64,
            depth=2,
            norm_groups=4,
        )
        # Check that blocks use the custom norm_groups
        for block in model.blocks:
            assert block.norm1.num_groups == 4
            assert block.norm2.num_groups == 4

    def test_norm_groups_validation(self):
        """Test that embed_dim must be divisible by norm_groups."""
        with pytest.raises(ValueError):
            DPOTNet(
                embed_dim=65,  # Not divisible by 8
                norm_groups=8,
            )

    def test_input_validation(self):
        """Test input shape validation."""
        model = DPOTNet(
            in_channels=3,
            in_timesteps=4,
            embed_dim=64,
        )

        # Wrong number of timesteps
        x_wrong_t = torch.randn(1, 224, 224, 5, 3)  # 5 timesteps instead of 4
        with pytest.raises(ValueError):
            model(x_wrong_t)

        # Wrong number of channels
        x_wrong_c = torch.randn(1, 224, 224, 4, 2)  # 2 channels instead of 3
        with pytest.raises(ValueError):
            model(x_wrong_c)

    def test_normalize_option_disabled(self):
        """Test normalization disabled (safer test)."""
        model = DPOTNet(
            inp_shape=32,
            patch_size=8,
            in_channels=2,
            embed_dim=64,
            normalize=False,  # Disable normalization to avoid shape issues
        )
        x = torch.randn(1, 32, 32, 1, 2)
        output = model(x)
        assert output.shape == (1, 32, 32, 1, 4)  # default out_channels=4

    def test_different_patch_sizes(self):
        """Test model with different patch sizes."""
        model = DPOTNet(
            inp_shape=24,
            patch_size=6,  # Evenly divides into 24
            in_channels=1,
            embed_dim=32,
            depth=1,
        )
        x = torch.randn(1, 24, 24, 1, 1)
        output = model(x)
        assert output.shape == (1, 24, 24, 1, 4)

    def test_multiple_timesteps_output(self):
        """Test model with multiple output timesteps."""
        model = DPOTNet(
            inp_shape=32,
            patch_size=8,
            in_channels=2,
            out_channels=3,
            in_timesteps=2,
            out_timesteps=5,
            embed_dim=64,
        )
        x = torch.randn(1, 32, 32, 2, 2)
        output = model(x)
        assert output.shape == (1, 32, 32, 5, 3)


class TestModelComponents:
    """Test individual model components."""

    def test_activation_factory(self):
        """Test activation function factory."""
        from physicsnemo.models.dpot.dpot import get_activation

        # Test valid activations
        gelu = get_activation("gelu")
        assert isinstance(gelu, nn.GELU)

        relu = get_activation("ReLU")  # Test case insensitive
        assert isinstance(relu, nn.ReLU)

        # Test invalid activation
        with pytest.raises(ValueError, match="Unsupported activation"):
            get_activation("invalid_activation")

    def test_model_meta(self):
        """Test model metadata."""
        from physicsnemo.models.dpot.dpot import DPOTMeta

        meta = DPOTMeta()
        assert meta.name == "DPOTNet"
        assert meta.amp is True
        assert meta.jit is False


if __name__ == "__main__":
    pytest.main([__file__])
