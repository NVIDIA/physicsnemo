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

"""Tests for procedural noise generation functions."""

import pytest
import torch

from physicsnemo.mesh.primitives.procedural.noise import (
    perlin_noise_1d,
    perlin_noise_2d,
    perlin_noise_3d,
    perlin_noise_nd,
)


class TestPerlinNoiseND:
    """Test suite for dimension-agnostic Perlin noise."""

    @pytest.mark.parametrize("n_dims", [1, 2, 3, 4, 5])
    def test_dimension_agnostic(self, n_dims):
        """Test that Perlin noise works for any number of dimensions."""
        n_points = 50
        points = torch.randn(n_points, n_dims)

        noise = perlin_noise_nd(points, scale=1.0, seed=42)

        ### Verify output shape
        assert noise.shape == (n_points,)
        assert noise.dtype == points.dtype
        assert noise.device == points.device

        ### Verify output range is reasonable (approximately [-1, 1])
        assert noise.min() >= -2.0  # Allow some margin
        assert noise.max() <= 2.0

    @pytest.mark.parametrize("seed", [0, 42, 123, 999])
    def test_reproducibility(self, seed):
        """Test that same seed produces same output."""
        points = torch.randn(100, 3)

        noise1 = perlin_noise_nd(points, scale=1.0, seed=seed)
        noise2 = perlin_noise_nd(points, scale=1.0, seed=seed)

        assert torch.allclose(noise1, noise2)

    def test_different_seeds_produce_different_output(self):
        """Test that different seeds produce different noise patterns."""
        points = torch.randn(100, 3)

        noise1 = perlin_noise_nd(points, scale=1.0, seed=42)
        noise2 = perlin_noise_nd(points, scale=1.0, seed=123)

        # Noise should be different
        assert not torch.allclose(noise1, noise2)
        # But should have similar statistics
        assert abs(noise1.mean() - noise2.mean()) < 0.5
        assert abs(noise1.std() - noise2.std()) < 0.5

    @pytest.mark.parametrize("scale", [0.1, 0.5, 1.0, 2.0, 5.0])
    def test_scale_parameter(self, scale):
        """Test that scale parameter affects noise frequency."""
        points = torch.randn(100, 3)

        noise = perlin_noise_nd(points, scale=scale, seed=42)

        ### Verify output shape and type
        assert noise.shape == (100,)

        ### Verify noise is non-constant (has variation)
        assert noise.std() > 0.01

    def test_smoothness(self):
        """Test that neighboring points have similar noise values."""
        # Create points on a regular grid
        x = torch.linspace(0, 5, 50)
        points = torch.stack([x, torch.zeros_like(x)], dim=1)

        noise = perlin_noise_nd(points, scale=1.0, seed=42)

        # Adjacent points should have similar values (noise is smooth)
        differences = torch.abs(noise[1:] - noise[:-1])
        max_diff = differences.max().item()

        # Smooth noise shouldn't have large jumps between adjacent points
        assert max_diff < 1.0  # Conservative bound

    @pytest.mark.cuda
    def test_gpu_compatibility(self):
        """Test that noise works on GPU."""
        points_cpu = torch.randn(100, 3)
        points_gpu = points_cpu.cuda()

        noise_gpu = perlin_noise_nd(points_gpu, scale=1.0, seed=42)

        ### Verify output is on GPU
        assert noise_gpu.device.type == "cuda"
        assert noise_gpu.shape == (100,)

    @pytest.mark.cuda
    def test_cpu_gpu_consistency(self):
        """Test that CPU and GPU produce similar statistical properties.

        Note: Exact values differ because torch.manual_seed() uses different RNG
        implementations on CPU vs GPU, but statistical properties should be similar.
        """
        points_cpu = torch.randn(1000, 3)
        points_gpu = points_cpu.cuda()

        noise_cpu = perlin_noise_nd(points_cpu, scale=1.0, seed=42)
        noise_gpu = perlin_noise_nd(points_gpu, scale=1.0, seed=42)

        ### Statistical properties should be similar
        assert abs(noise_cpu.mean() - noise_gpu.cpu().mean()) < 0.2
        assert abs(noise_cpu.std() - noise_gpu.cpu().std()) < 0.2
        assert abs(noise_cpu.min() - noise_gpu.cpu().min()) < 1.0
        assert abs(noise_cpu.max() - noise_gpu.cpu().max()) < 1.0

    def test_batch_processing(self):
        """Test that noise handles large batches efficiently."""
        # Large batch
        points = torch.randn(10000, 3)

        noise = perlin_noise_nd(points, scale=1.0, seed=42)

        assert noise.shape == (10000,)
        assert not torch.isnan(noise).any()
        assert not torch.isinf(noise).any()

    def test_zero_scale(self):
        """Test behavior with zero scale."""
        points = torch.randn(50, 3)

        noise = perlin_noise_nd(points, scale=0.0, seed=42)

        # With zero scale, all points map to same lattice position
        # So noise should be constant (or nearly so)
        assert noise.std() < 0.01

    def test_very_large_scale(self):
        """Test that very large scales don't cause issues."""
        points = torch.randn(50, 3)

        noise = perlin_noise_nd(points, scale=100.0, seed=42)

        assert noise.shape == (50,)
        assert not torch.isnan(noise).any()

    @pytest.mark.parametrize("n_dims", [1, 2, 3, 4])
    def test_origin_is_deterministic(self, n_dims):
        """Test that noise at origin is reproducible."""
        origin = torch.zeros(1, n_dims)

        noise1 = perlin_noise_nd(origin, scale=1.0, seed=42)
        noise2 = perlin_noise_nd(origin, scale=1.0, seed=42)

        assert torch.allclose(noise1, noise2)

    def test_negative_coordinates(self):
        """Test that noise works with negative coordinates."""
        points = torch.tensor([[-1.0, -1.0, -1.0], [-2.5, 0.5, 1.0], [0.0, -5.0, 2.0]])

        noise = perlin_noise_nd(points, scale=1.0, seed=42)

        assert noise.shape == (3,)
        assert not torch.isnan(noise).any()

    def test_different_dtypes(self):
        """Test that noise works with different float dtypes."""
        points_float32 = torch.randn(50, 3, dtype=torch.float32)
        points_float64 = points_float32.double()

        noise_32 = perlin_noise_nd(points_float32, scale=1.0, seed=42)
        noise_64 = perlin_noise_nd(points_float64, scale=1.0, seed=42)

        ### Output dtype should match input
        assert noise_32.dtype == torch.float32
        assert noise_64.dtype == torch.float64

        ### Values should be similar (allowing for precision differences)
        assert torch.allclose(noise_32, noise_64.float(), atol=1e-6)


class TestPerlinNoiseConvenienceWrappers:
    """Test convenience wrappers for specific dimensions."""

    def test_perlin_noise_1d(self):
        """Test 1D convenience wrapper."""
        points = torch.randn(50, 1)
        noise = perlin_noise_1d(points, scale=1.0, seed=42)

        assert noise.shape == (50,)

    def test_perlin_noise_2d(self):
        """Test 2D convenience wrapper."""
        points = torch.randn(50, 2)
        noise = perlin_noise_2d(points, scale=1.0, seed=42)

        assert noise.shape == (50,)

    def test_perlin_noise_3d(self):
        """Test 3D convenience wrapper."""
        points = torch.randn(50, 3)
        noise = perlin_noise_3d(points, scale=1.0, seed=42)

        assert noise.shape == (50,)

    def test_wrappers_match_general_function(self):
        """Test that wrappers produce same results as general function."""
        points_1d = torch.randn(20, 1)
        points_2d = torch.randn(20, 2)
        points_3d = torch.randn(20, 3)

        # Wrappers should match general function
        assert torch.allclose(
            perlin_noise_1d(points_1d, scale=1.0, seed=42),
            perlin_noise_nd(points_1d, scale=1.0, seed=42),
        )
        assert torch.allclose(
            perlin_noise_2d(points_2d, scale=1.0, seed=42),
            perlin_noise_nd(points_2d, scale=1.0, seed=42),
        )
        assert torch.allclose(
            perlin_noise_3d(points_3d, scale=1.0, seed=42),
            perlin_noise_nd(points_3d, scale=1.0, seed=42),
        )


class TestPerlinNoiseWithMesh:
    """Test Perlin noise integration with Mesh objects."""

    def test_noise_on_mesh_cell_data(self):
        """Test applying noise to mesh cell centroids."""
        from physicsnemo.mesh import Mesh

        # Create simple triangle mesh
        points = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]
        )
        cells = torch.tensor([[0, 1, 2], [1, 3, 2]])
        mesh = Mesh(points=points, cells=cells)

        # Generate noise at cell centroids
        centroids = mesh.cell_centroids
        noise = perlin_noise_nd(centroids, scale=1.0, seed=42)

        ### Add to cell data
        mesh.cell_data["noise"] = noise

        assert "noise" in mesh.cell_data
        assert mesh.cell_data["noise"].shape == (mesh.n_cells,)

    def test_noise_on_mesh_point_data(self):
        """Test applying noise to mesh vertices."""
        from physicsnemo.mesh import Mesh

        # Create simple triangle mesh
        points = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]
        )
        cells = torch.tensor([[0, 1, 2], [1, 3, 2]])
        mesh = Mesh(points=points, cells=cells)

        # Generate noise at vertices
        noise = perlin_noise_nd(mesh.points, scale=1.0, seed=42)

        ### Add to point data
        mesh.point_data["noise"] = noise

        assert "noise" in mesh.point_data
        assert mesh.point_data["noise"].shape == (mesh.n_points,)
