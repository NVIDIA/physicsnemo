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

"""Tests for PhysicsAttention modules.

This module tests the physics-informed attention mechanisms used in Transolver:
- PhysicsAttentionIrregularMesh: For unstructured mesh data
- PhysicsAttentionStructuredMesh2D: For 2D image-like data
- PhysicsAttentionStructuredMesh3D: For 3D volumetric data
"""

import pytest
import torch

from physicsnemo.nn.module.physics_attention import (
    PhysicsAttentionIrregularMesh,
    PhysicsAttentionStructuredMesh2D,
    PhysicsAttentionStructuredMesh3D,
)

# =============================================================================
# PhysicsAttentionIrregularMesh Tests
# =============================================================================


class TestPhysicsAttentionIrregularMesh:
    """Tests for PhysicsAttentionIrregularMesh module."""

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    @pytest.mark.parametrize("num_tokens", [100, 500, 1000])
    @pytest.mark.parametrize("dim", [64, 128, 256])
    def test_output_shape(self, device, batch_size, num_tokens, dim):
        """Test that output shape matches input shape."""
        attn = PhysicsAttentionIrregularMesh(
            dim=dim,
            heads=4,
            dim_head=32,
            dropout=0.0,
            slice_num=16,
            use_te=False,
            plus=False,
        ).to(device)

        x = torch.randn(batch_size, num_tokens, dim, device=device)
        output = attn(x)

        assert output.shape == x.shape

    @pytest.mark.parametrize("heads", [1, 4, 8])
    @pytest.mark.parametrize("dim_head", [16, 32, 64])
    def test_various_head_configs(self, device, heads, dim_head):
        """Test with various head configurations."""
        dim = 128
        attn = PhysicsAttentionIrregularMesh(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            dropout=0.0,
            slice_num=16,
            use_te=False,
            plus=False,
        ).to(device)

        x = torch.randn(2, 100, dim, device=device)
        output = attn(x)

        assert output.shape == x.shape

    @pytest.mark.parametrize("slice_num", [4, 16, 32, 64])
    def test_various_slice_nums(self, device, slice_num):
        """Test with various numbers of physics slices."""
        dim = 128
        attn = PhysicsAttentionIrregularMesh(
            dim=dim,
            heads=4,
            dim_head=32,
            dropout=0.0,
            slice_num=slice_num,
            use_te=False,
            plus=False,
        ).to(device)

        x = torch.randn(2, 100, dim, device=device)
        output = attn(x)

        assert output.shape == x.shape

    def test_plus_variant(self, device):
        """Test the Transolver++ variant."""
        dim = 128
        attn = PhysicsAttentionIrregularMesh(
            dim=dim,
            heads=4,
            dim_head=32,
            dropout=0.0,
            slice_num=16,
            use_te=False,
            plus=True,  # Enable Transolver++
        ).to(device)

        x = torch.randn(2, 100, dim, device=device)
        output = attn(x)

        assert output.shape == x.shape

    def test_dropout_training_mode(self, device):
        """Test that dropout is applied during training."""
        dim = 128
        attn = PhysicsAttentionIrregularMesh(
            dim=dim,
            heads=4,
            dim_head=32,
            dropout=0.5,
            slice_num=16,
            use_te=False,
            plus=False,
        ).to(device)
        attn.train()

        x = torch.randn(2, 100, dim, device=device)

        # Run multiple times - outputs should differ due to dropout
        output1 = attn(x)
        output2 = attn(x)

        # With high dropout, outputs should differ
        assert not torch.allclose(output1, output2)

    def test_dropout_eval_mode(self, device):
        """Test that dropout is not applied during evaluation."""
        dim = 128
        attn = PhysicsAttentionIrregularMesh(
            dim=dim,
            heads=4,
            dim_head=32,
            dropout=0.5,
            slice_num=16,
            use_te=False,
            plus=False,
        ).to(device)
        attn.eval()

        x = torch.randn(2, 100, dim, device=device)

        with torch.no_grad():
            output1 = attn(x)
            output2 = attn(x)

        assert torch.allclose(output1, output2)

    def test_gradient_flow(self, device):
        """Test that gradients flow through the module."""
        dim = 128
        attn = PhysicsAttentionIrregularMesh(
            dim=dim,
            heads=4,
            dim_head=32,
            dropout=0.0,
            slice_num=16,
            use_te=False,
            plus=False,
        ).to(device)

        x = torch.randn(2, 100, dim, device=device, requires_grad=True)
        output = attn(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape
        assert not torch.any(torch.isnan(x.grad))

    def test_gradient_flow_plus_variant(self, device):
        """Test gradient flow in Transolver++ variant."""
        dim = 128
        attn = PhysicsAttentionIrregularMesh(
            dim=dim,
            heads=4,
            dim_head=32,
            dropout=0.0,
            slice_num=16,
            use_te=False,
            plus=True,
        ).to(device)

        x = torch.randn(2, 100, dim, device=device, requires_grad=True)
        output = attn(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.any(torch.isnan(x.grad))

    def test_output_finite(self, device):
        """Test that output contains no NaN or Inf values."""
        dim = 128
        attn = PhysicsAttentionIrregularMesh(
            dim=dim,
            heads=4,
            dim_head=32,
            dropout=0.0,
            slice_num=16,
            use_te=False,
            plus=False,
        ).to(device)

        x = torch.randn(2, 100, dim, device=device)
        output = attn(x)

        assert torch.all(torch.isfinite(output))

    def test_invalid_input_dims(self, device):
        """Test that invalid input dimensions raise an error."""
        dim = 128
        attn = PhysicsAttentionIrregularMesh(
            dim=dim,
            heads=4,
            dim_head=32,
            dropout=0.0,
            slice_num=16,
            use_te=False,
            plus=False,
        ).to(device)

        # 2D input should raise error
        x_2d = torch.randn(100, dim, device=device)
        with pytest.raises(ValueError, match="Expected 3D input"):
            attn(x_2d)

        # 4D input should raise error
        x_4d = torch.randn(2, 10, 10, dim, device=device)
        with pytest.raises(ValueError, match="Expected 3D input"):
            attn(x_4d)


# =============================================================================
# PhysicsAttentionStructuredMesh2D Tests
# =============================================================================


class TestPhysicsAttentionStructuredMesh2D:
    """Tests for PhysicsAttentionStructuredMesh2D module."""

    @pytest.mark.parametrize("spatial_shape", [(16, 16), (32, 32), (64, 64)])
    def test_output_shape(self, device, spatial_shape):
        """Test that output shape matches input shape for various spatial sizes."""
        dim = 128
        h, w = spatial_shape
        attn = PhysicsAttentionStructuredMesh2D(
            dim=dim,
            spatial_shape=spatial_shape,
            heads=4,
            dim_head=32,
            dropout=0.0,
            slice_num=16,
            use_te=False,
            plus=False,
        ).to(device)

        x = torch.randn(2, h * w, dim, device=device)
        output = attn(x)

        assert output.shape == x.shape

    @pytest.mark.parametrize("spatial_shape", [(16, 32), (32, 16), (24, 48)])
    def test_non_square_spatial_shapes(self, device, spatial_shape):
        """Test with non-square spatial dimensions."""
        dim = 128
        h, w = spatial_shape
        attn = PhysicsAttentionStructuredMesh2D(
            dim=dim,
            spatial_shape=spatial_shape,
            heads=4,
            dim_head=32,
            dropout=0.0,
            slice_num=16,
            use_te=False,
            plus=False,
        ).to(device)

        x = torch.randn(2, h * w, dim, device=device)
        output = attn(x)

        assert output.shape == x.shape

    @pytest.mark.parametrize("kernel", [1, 3, 5, 7])
    def test_various_kernel_sizes(self, device, kernel):
        """Test with various convolution kernel sizes."""
        dim = 128
        spatial_shape = (32, 32)
        h, w = spatial_shape
        attn = PhysicsAttentionStructuredMesh2D(
            dim=dim,
            spatial_shape=spatial_shape,
            heads=4,
            dim_head=32,
            dropout=0.0,
            slice_num=16,
            kernel=kernel,
            use_te=False,
            plus=False,
        ).to(device)

        x = torch.randn(2, h * w, dim, device=device)
        output = attn(x)

        assert output.shape == x.shape

    def test_plus_variant(self, device):
        """Test the Transolver++ variant for 2D structured mesh."""
        dim = 128
        spatial_shape = (32, 32)
        h, w = spatial_shape
        attn = PhysicsAttentionStructuredMesh2D(
            dim=dim,
            spatial_shape=spatial_shape,
            heads=4,
            dim_head=32,
            dropout=0.0,
            slice_num=16,
            use_te=False,
            plus=True,
        ).to(device)

        x = torch.randn(2, h * w, dim, device=device)
        output = attn(x)

        assert output.shape == x.shape

    def test_gradient_flow(self, device):
        """Test that gradients flow through the 2D attention module."""
        dim = 128
        spatial_shape = (32, 32)
        h, w = spatial_shape
        attn = PhysicsAttentionStructuredMesh2D(
            dim=dim,
            spatial_shape=spatial_shape,
            heads=4,
            dim_head=32,
            dropout=0.0,
            slice_num=16,
            use_te=False,
            plus=False,
        ).to(device)

        x = torch.randn(2, h * w, dim, device=device, requires_grad=True)
        output = attn(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape
        assert not torch.any(torch.isnan(x.grad))

    def test_output_finite(self, device):
        """Test that output contains no NaN or Inf values."""
        dim = 128
        spatial_shape = (32, 32)
        h, w = spatial_shape
        attn = PhysicsAttentionStructuredMesh2D(
            dim=dim,
            spatial_shape=spatial_shape,
            heads=4,
            dim_head=32,
            dropout=0.0,
            slice_num=16,
            use_te=False,
            plus=False,
        ).to(device)

        x = torch.randn(2, h * w, dim, device=device)
        output = attn(x)

        assert torch.all(torch.isfinite(output))

    def test_uses_2d_convolution(self, device):
        """Test that the module uses 2D convolution layers."""
        dim = 128
        spatial_shape = (32, 32)
        attn = PhysicsAttentionStructuredMesh2D(
            dim=dim,
            spatial_shape=spatial_shape,
            heads=4,
            dim_head=32,
            dropout=0.0,
            slice_num=16,
            use_te=False,
            plus=False,
        ).to(device)

        # Verify that in_project_x is a Conv2d
        assert isinstance(attn.in_project_x, torch.nn.Conv2d)


# =============================================================================
# PhysicsAttentionStructuredMesh3D Tests
# =============================================================================


class TestPhysicsAttentionStructuredMesh3D:
    """Tests for PhysicsAttentionStructuredMesh3D module."""

    @pytest.mark.parametrize("spatial_shape", [(8, 8, 8), (16, 16, 16), (8, 16, 8)])
    def test_output_shape(self, device, spatial_shape):
        """Test that output shape matches input shape for various 3D spatial sizes."""
        dim = 128
        d, h, w = spatial_shape
        attn = PhysicsAttentionStructuredMesh3D(
            dim=dim,
            spatial_shape=spatial_shape,
            heads=4,
            dim_head=32,
            dropout=0.0,
            slice_num=16,
            use_te=False,
            plus=False,
        ).to(device)

        x = torch.randn(2, d * h * w, dim, device=device)
        output = attn(x)

        assert output.shape == x.shape

    @pytest.mark.parametrize("kernel", [1, 3, 5])
    def test_various_kernel_sizes(self, device, kernel):
        """Test with various 3D convolution kernel sizes."""
        dim = 128
        spatial_shape = (8, 8, 8)
        d, h, w = spatial_shape
        attn = PhysicsAttentionStructuredMesh3D(
            dim=dim,
            spatial_shape=spatial_shape,
            heads=4,
            dim_head=32,
            dropout=0.0,
            slice_num=16,
            kernel=kernel,
            use_te=False,
            plus=False,
        ).to(device)

        x = torch.randn(2, d * h * w, dim, device=device)
        output = attn(x)

        assert output.shape == x.shape

    def test_plus_variant(self, device):
        """Test the Transolver++ variant for 3D structured mesh."""
        dim = 128
        spatial_shape = (8, 8, 8)
        d, h, w = spatial_shape
        attn = PhysicsAttentionStructuredMesh3D(
            dim=dim,
            spatial_shape=spatial_shape,
            heads=4,
            dim_head=32,
            dropout=0.0,
            slice_num=16,
            use_te=False,
            plus=True,
        ).to(device)

        x = torch.randn(2, d * h * w, dim, device=device)
        output = attn(x)

        assert output.shape == x.shape

    def test_gradient_flow(self, device):
        """Test that gradients flow through the 3D attention module."""
        dim = 128
        spatial_shape = (8, 8, 8)
        d, h, w = spatial_shape
        attn = PhysicsAttentionStructuredMesh3D(
            dim=dim,
            spatial_shape=spatial_shape,
            heads=4,
            dim_head=32,
            dropout=0.0,
            slice_num=16,
            use_te=False,
            plus=False,
        ).to(device)

        x = torch.randn(2, d * h * w, dim, device=device, requires_grad=True)
        output = attn(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape
        assert not torch.any(torch.isnan(x.grad))

    def test_output_finite(self, device):
        """Test that output contains no NaN or Inf values."""
        dim = 128
        spatial_shape = (8, 8, 8)
        d, h, w = spatial_shape
        attn = PhysicsAttentionStructuredMesh3D(
            dim=dim,
            spatial_shape=spatial_shape,
            heads=4,
            dim_head=32,
            dropout=0.0,
            slice_num=16,
            use_te=False,
            plus=False,
        ).to(device)

        x = torch.randn(2, d * h * w, dim, device=device)
        output = attn(x)

        assert torch.all(torch.isfinite(output))

    def test_uses_3d_convolution(self, device):
        """Test that the module uses 3D convolution layers."""
        dim = 128
        spatial_shape = (8, 8, 8)
        attn = PhysicsAttentionStructuredMesh3D(
            dim=dim,
            spatial_shape=spatial_shape,
            heads=4,
            dim_head=32,
            dropout=0.0,
            slice_num=16,
            use_te=False,
            plus=False,
        ).to(device)

        # Verify that in_project_x is a Conv3d
        assert isinstance(attn.in_project_x, torch.nn.Conv3d)


# =============================================================================
# Cross-Module Comparison Tests
# =============================================================================


class TestPhysicsAttentionComparison:
    """Tests comparing different PhysicsAttention variants."""

    def test_irregular_vs_structured_different_outputs(self, device):
        """Test that irregular and structured variants produce different outputs."""
        dim = 128
        spatial_shape = (10, 10)
        h, w = spatial_shape
        num_tokens = h * w

        # Same base configuration
        attn_irregular = PhysicsAttentionIrregularMesh(
            dim=dim,
            heads=4,
            dim_head=32,
            dropout=0.0,
            slice_num=16,
            use_te=False,
            plus=False,
        ).to(device)
        attn_structured = PhysicsAttentionStructuredMesh2D(
            dim=dim,
            spatial_shape=spatial_shape,
            heads=4,
            dim_head=32,
            dropout=0.0,
            slice_num=16,
            use_te=False,
            plus=False,
        ).to(device)

        x = torch.randn(2, num_tokens, dim, device=device)

        output_irregular = attn_irregular(x)
        output_structured = attn_structured(x)

        # Outputs should differ due to different projection methods
        assert not torch.allclose(output_irregular, output_structured)

    def test_standard_vs_plus_different_outputs(self, device):
        """Test that standard and plus variants produce different outputs."""
        dim = 128

        attn_standard = PhysicsAttentionIrregularMesh(
            dim=dim,
            heads=4,
            dim_head=32,
            dropout=0.0,
            slice_num=16,
            use_te=False,
            plus=False,
        ).to(device)

        attn_plus = PhysicsAttentionIrregularMesh(
            dim=dim,
            heads=4,
            dim_head=32,
            dropout=0.0,
            slice_num=16,
            use_te=False,
            plus=True,
        ).to(device)

        x = torch.randn(2, 100, dim, device=device)

        output_standard = attn_standard(x)
        output_plus = attn_plus(x)

        # Outputs should differ due to different mechanisms
        assert not torch.allclose(output_standard, output_plus)


# =============================================================================
# Module Attribute Tests
# =============================================================================


class TestPhysicsAttentionAttributes:
    """Tests for module attributes and initialization."""

    def test_temperature_parameter_exists(self, device):
        """Test that temperature is a learnable parameter."""
        attn = PhysicsAttentionIrregularMesh(
            dim=128,
            heads=4,
            dim_head=32,
            dropout=0.0,
            slice_num=16,
            use_te=False,
            plus=False,
        ).to(device)

        assert hasattr(attn, "temperature")
        assert isinstance(attn.temperature, torch.nn.Parameter)

    def test_slice_projection_orthogonal_init(self, device):
        """Test that slice projection weights are orthogonally initialized."""
        attn = PhysicsAttentionIrregularMesh(
            dim=128,
            heads=4,
            dim_head=32,
            dropout=0.0,
            slice_num=16,
            use_te=False,
            plus=False,
        ).to(device)

        # Check that in_project_slice weights are somewhat orthogonal
        weight = attn.in_project_slice.weight
        # For orthogonal matrices, W @ W^T should be close to identity
        product = torch.mm(weight, weight.t())
        # The diagonal should be close to 1 (columns have unit norm)
        diag = torch.diag(product)
        assert torch.allclose(diag, torch.ones_like(diag), atol=0.1)

    def test_plus_has_temperature_projection(self, device):
        """Test that plus variant has temperature projection layer."""
        attn_plus = PhysicsAttentionIrregularMesh(
            dim=128,
            heads=4,
            dim_head=32,
            dropout=0.0,
            slice_num=16,
            use_te=False,
            plus=True,
        ).to(device)

        assert hasattr(attn_plus, "proj_temperature")

        attn_standard = PhysicsAttentionIrregularMesh(
            dim=128,
            heads=4,
            dim_head=32,
            dropout=0.0,
            slice_num=16,
            use_te=False,
            plus=False,
        ).to(device)

        assert not hasattr(attn_standard, "proj_temperature")

    def test_parameter_count_reasonable(self, device):
        """Test that parameter count is within reasonable bounds."""
        dim = 128
        heads = 4
        dim_head = 32
        slice_num = 16

        attn = PhysicsAttentionIrregularMesh(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            dropout=0.0,
            slice_num=slice_num,
            use_te=False,
            plus=False,
        ).to(device)

        total_params = sum(p.numel() for p in attn.parameters())

        # Should have parameters, but not an unreasonable amount
        assert total_params > 0
        assert total_params < 10_000_000  # Less than 10M parameters


# =============================================================================
# Memory and Performance Tests
# =============================================================================


class TestPhysicsAttentionMemory:
    """Tests for memory efficiency."""

    def test_no_memory_leak_forward(self, device):
        """Test that forward pass doesn't leak memory."""
        if device == "cpu":
            pytest.skip("Memory leak test only meaningful on CUDA")

        dim = 128
        attn = PhysicsAttentionIrregularMesh(
            dim=dim,
            heads=4,
            dim_head=32,
            dropout=0.0,
            slice_num=16,
            use_te=False,
            plus=False,
        ).to(device)

        x = torch.randn(2, 1000, dim, device=device)

        # Warm up
        _ = attn(x)
        torch.cuda.synchronize()

        # Get initial memory
        torch.cuda.reset_peak_memory_stats()
        initial_memory = torch.cuda.memory_allocated()

        # Run multiple forward passes
        for _ in range(10):
            output = attn(x)
            del output

        torch.cuda.synchronize()
        final_memory = torch.cuda.memory_allocated()

        # Memory should not grow significantly
        memory_growth = final_memory - initial_memory
        assert memory_growth < 1_000_000  # Less than 1MB growth
