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

from physicsnemo.models.dpot.dpot3d import AFNO3DLayer, Block3D, DPOTNet3D


class TestAFNO3DLayer:
    """Test AFNO3DLayer functionality."""

    def test_forward_pass_channel_first(self):
        """Test basic forward pass with channel_first=True."""
        layer = AFNO3DLayer(
            width=32, num_blocks=4, modes=8, temporal_modes=4, channel_first=True
        )
        x = torch.randn(2, 32, 16, 16, 8)  # (B, C, X, Y, Z)
        output = layer(x)
        assert output.shape == x.shape

    def test_forward_pass_channel_last(self):
        """Test basic forward pass with channel_last (default)."""
        layer = AFNO3DLayer(
            width=32, num_blocks=4, modes=8, temporal_modes=4, channel_first=False
        )
        x = torch.randn(2, 16, 16, 8, 32)  # (B, X, Y, Z, C)
        output = layer(x)
        assert output.shape == x.shape

    def test_width_validation(self):
        """Test that width must be divisible by num_blocks."""
        with pytest.raises(ValueError, match="width must be divisible by num_blocks"):
            AFNO3DLayer(width=33, num_blocks=4)  # 33 not divisible by 4


class TestBlock3D:
    """Test Block3D functionality."""

    def test_forward_pass(self):
        """Test basic forward pass."""
        block = Block3D(
            width=32, num_blocks=4, mlp_ratio=1.0, modes=8, temporal_modes=4
        )
        x = torch.randn(2, 32, 8, 8, 6)  # (B, C, X, Y, Z)
        output = block(x)
        assert output.shape == x.shape

    def test_no_double_skip(self):
        """Test disabling double skip."""
        block = Block3D(
            width=32,
            num_blocks=4,
            mlp_ratio=1.0,
            modes=8,
            temporal_modes=4,
            double_skip=False,
        )
        x = torch.randn(2, 32, 8, 8, 6)
        output = block(x)
        assert output.shape == x.shape


class TestDPOTNet3D:
    """Test DPOTNet3D functionality."""

    def test_basic_forward(self):
        """Test basic forward pass."""
        model = DPOTNet3D(
            inp_shape=24,
            patch_size=8,
            in_channels=3,
            out_channels=3,
            embed_dim=64,
            depth=1,
        )
        x = torch.randn(1, 24, 24, 24, 1, 3)
        output = model(x)
        assert output.shape == (1, 24, 24, 24, 1, 3)

    def test_different_timesteps(self):
        """Test with different input/output timesteps."""
        model = DPOTNet3D(
            inp_shape=16,
            patch_size=8,
            in_channels=2,
            out_channels=4,
            in_timesteps=2,
            out_timesteps=3,
            embed_dim=64,
            depth=1,
        )
        x = torch.randn(1, 16, 16, 16, 2, 2)
        output = model(x)
        assert output.shape == (1, 16, 16, 16, 3, 4)

    def test_cube_shape_validation(self):
        """Test that inp_shape dimensions must be compatible with patch_size."""
        # This should work (24 % 8 = 3, no remainder)
        model = DPOTNet3D(inp_shape=24, patch_size=8)
        assert model is not None

        # This should also work (16 % 8 = 2, no remainder)
        model2 = DPOTNet3D(inp_shape=16, patch_size=8)
        assert model2 is not None

    def test_input_timestep_validation(self):
        """Test input timestep validation."""
        model = DPOTNet3D(
            inp_shape=16,
            patch_size=8,
            in_timesteps=2,
            embed_dim=64,
            depth=1,
        )
        # Correct input
        x_good = torch.randn(1, 16, 16, 16, 2, 1)
        output = model(x_good)
        assert output.shape[4] == 1  # out_timesteps

        # Wrong timesteps
        x_bad = torch.randn(1, 16, 16, 16, 3, 1)  # 3 instead of 2
        with pytest.raises(ValueError, match="Input timesteps/channels mismatch"):
            model(x_bad)

    def test_input_channel_validation(self):
        """Test input channel validation."""
        model = DPOTNet3D(
            inp_shape=16,
            patch_size=8,
            in_channels=3,
            embed_dim=64,
            depth=1,
        )
        # Wrong channels
        x_bad = torch.randn(1, 16, 16, 16, 1, 2)  # 2 instead of 3
        with pytest.raises(ValueError, match="Input timesteps/channels mismatch"):
            model(x_bad)

    def test_normalize_same_channels(self):
        """Test normalization with same input/output channels (should denormalize)."""
        model = DPOTNet3D(
            inp_shape=16,
            patch_size=8,
            in_channels=2,
            out_channels=2,  # Same as input
            embed_dim=64,
            normalize=True,
            depth=1,
        )
        x = torch.randn(1, 16, 16, 16, 1, 2)
        output = model(x)
        assert output.shape == (1, 16, 16, 16, 1, 2)


if __name__ == "__main__":
    pytest.main([__file__])
