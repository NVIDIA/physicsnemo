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

"""Unit tests for PhysicsNeMo UNet wrappers."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.physicsnemo_unet import PhysicsNemoUNet2D, PhysicsNemoUNet3D, StandaloneUNet


class TestPhysicsNemoUNet2D:
    """Tests for 2D UNet wrapper (adds/removes dummy T dim)."""

    def test_output_shape(self):
        """Verify 2D UNet output shape matches input spatial dimensions."""
        unet = PhysicsNemoUNet2D(
            in_channels=32, out_channels=32, kernel_size=3, model_depth=1
        )
        x = torch.randn(2, 32, 16, 24)
        out = unet(x)
        assert out.shape == (2, 32, 16, 24)

    def test_preserves_spatial_dims(self):
        """Verify 2D UNet preserves spatial H and W dimensions."""
        unet = PhysicsNemoUNet2D(in_channels=16, out_channels=16, model_depth=1)
        x = torch.randn(1, 16, 32, 48)
        assert unet(x).shape[2:] == x.shape[2:]

    def test_different_in_out_channels(self):
        """Verify 2D UNet handles different input and output channel counts."""
        unet = PhysicsNemoUNet2D(in_channels=8, out_channels=16, model_depth=1)
        x = torch.randn(1, 8, 16, 16)
        assert unet(x).shape == (1, 16, 16, 16)

    def test_gradient_flow(self):
        """Verify gradients propagate through the 2D UNet."""
        unet = PhysicsNemoUNet2D(in_channels=16, out_channels=16, model_depth=1)
        x = torch.randn(1, 16, 16, 24, requires_grad=True)
        unet(x).sum().backward()
        assert x.grad is not None


class TestPhysicsNemoUNet3D:
    """Tests for 3D UNet wrapper (passthrough)."""

    def test_output_shape(self):
        """Verify 3D UNet output shape matches input spatial dimensions."""
        unet = PhysicsNemoUNet3D(in_channels=32, out_channels=32, kernel_size=3)
        x = torch.randn(2, 32, 8, 16, 8)
        out = unet(x)
        assert out.shape == (2, 32, 8, 16, 8)

    def test_different_channels(self):
        """Verify 3D UNet handles different input and output channel counts."""
        unet = PhysicsNemoUNet3D(in_channels=8, out_channels=4)
        x = torch.randn(1, 8, 8, 16, 8)
        assert unet(x).shape == (1, 4, 8, 16, 8)


class TestStandaloneUNet:
    """Tests for StandaloneUNet (channel-last convention)."""

    def test_output_shape(self):
        """Verify StandaloneUNet output shape with channel-last convention."""
        unet = StandaloneUNet(in_channels=12, out_channels=1, unet_type="physicsnemo")
        x = torch.randn(2, 16, 24, 8, 12)  # (B, H, W, T, C)
        out = unet(x)
        assert out.shape == (2, 16, 24, 8)  # (B, H, W, T)

    def test_gradient_flow(self):
        """Verify gradients propagate through the StandaloneUNet."""
        unet = StandaloneUNet(in_channels=5, out_channels=1, unet_type="physicsnemo")
        x = torch.randn(1, 16, 16, 8, 5, requires_grad=True)
        unet(x).sum().backward()
        assert x.grad is not None

    def test_count_params(self):
        """Verify count_params returns a positive parameter count."""
        unet = StandaloneUNet(in_channels=12, out_channels=1, unet_type="physicsnemo")
        assert unet.count_params() > 0

    def test_custom_unet_type_raises(self):
        """Verify ValueError is raised for the unsupported 'custom' unet_type."""
        with pytest.raises(ValueError, match="Custom UNet3D"):
            StandaloneUNet(in_channels=12, out_channels=1, unet_type="custom")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
