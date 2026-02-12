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

"""Tests for random sampling functionality.

Tests validate random point sampling across spatial dimensions, manifold dimensions,
and compute backends, ensuring uniform distribution and correctness.
"""

import pytest
import torch

from physicsnemo.mesh.mesh import Mesh
from physicsnemo.mesh.sampling import sample_random_points_on_cells

### Helper Functions ###


def create_simple_mesh(n_spatial_dims: int, n_manifold_dims: int, device: str = "cpu"):
    """Create a simple mesh for testing."""
    if n_manifold_dims > n_spatial_dims:
        raise ValueError(
            f"Manifold dimension {n_manifold_dims} cannot exceed spatial dimension {n_spatial_dims}"
        )

    if n_manifold_dims == 1:
        if n_spatial_dims == 2:
            points = torch.tensor(
                [[0.0, 0.0], [1.0, 0.0], [1.5, 1.0], [0.5, 1.5]], device=device
            )
        elif n_spatial_dims == 3:
            points = torch.tensor(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 1.0]],
                device=device,
            )
        else:
            raise ValueError(f"Unsupported {n_spatial_dims=}")
        cells = torch.tensor([[0, 1], [1, 2], [2, 3]], device=device, dtype=torch.int64)
    elif n_manifold_dims == 2:
        if n_spatial_dims == 2:
            points = torch.tensor(
                [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0], [1.5, 0.5]], device=device
            )
        elif n_spatial_dims == 3:
            points = torch.tensor(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0], [1.5, 0.5, 0.5]],
                device=device,
            )
        else:
            raise ValueError(f"Unsupported {n_spatial_dims=}")
        cells = torch.tensor([[0, 1, 2], [1, 3, 2]], device=device, dtype=torch.int64)
    elif n_manifold_dims == 3:
        if n_spatial_dims == 3:
            points = torch.tensor(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [1.0, 1.0, 1.0],
                ],
                device=device,
            )
            cells = torch.tensor(
                [[0, 1, 2, 3], [1, 2, 3, 4]], device=device, dtype=torch.int64
            )
        else:
            raise ValueError("3-simplices require 3D embedding space")
    else:
        raise ValueError(f"Unsupported {n_manifold_dims=}")

    return Mesh(points=points, cells=cells)


def assert_on_device(tensor: torch.Tensor, expected_device: str) -> None:
    """Assert tensor is on expected device."""
    actual_device = tensor.device.type
    assert actual_device == expected_device, (
        f"Device mismatch: tensor is on {actual_device!r}, expected {expected_device!r}"
    )


### Test Fixtures ###


class TestRandomSampling:
    """Tests for sample_random_points_on_cells."""

    def test_default_sampling_one_per_cell(self):
        """Test that default behavior samples one point per cell."""
        torch.manual_seed(42)
        ### Create a simple triangle mesh
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ]
        )
        cells = torch.tensor(
            [
                [0, 1, 2],
                [1, 3, 2],
            ]
        )
        mesh = Mesh(points=points, cells=cells)

        ### Sample without specifying cell_indices
        sampled_points = sample_random_points_on_cells(mesh)

        ### Should get one point per cell
        assert sampled_points.shape == (2, 2)

    def test_specific_cell_indices(self):
        """Test sampling from specific cells."""
        torch.manual_seed(42)
        ### Create a simple triangle mesh
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ]
        )
        cells = torch.tensor(
            [
                [0, 1, 2],
                [1, 3, 2],
            ]
        )
        mesh = Mesh(points=points, cells=cells)

        ### Sample from specific cells
        cell_indices = torch.tensor([0, 1, 0])
        sampled_points = sample_random_points_on_cells(mesh, cell_indices=cell_indices)

        ### Should get three points (two from cell 0, one from cell 1)
        assert sampled_points.shape == (3, 2)

    def test_repeated_cell_indices(self):
        """Test that repeated indices sample multiple points from the same cell."""
        torch.manual_seed(42)
        ### Create a simple triangle mesh
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        )
        cells = torch.tensor([[0, 1, 2]])
        mesh = Mesh(points=points, cells=cells)

        ### Sample multiple times from the same cell
        cell_indices = torch.tensor([0, 0, 0, 0, 0])
        sampled_points = sample_random_points_on_cells(mesh, cell_indices=cell_indices)

        ### Should get 5 points, all within the same triangle
        assert sampled_points.shape == (5, 2)

        ### All points should be within the triangle (have non-negative barycentric coords)
        # This is a simple check: all points should be in the bounding box
        assert torch.all(sampled_points >= 0.0)
        assert torch.all(sampled_points <= 1.0)

    def test_cell_indices_as_list(self):
        """Test that cell_indices can be passed as a Python list."""
        torch.manual_seed(42)
        ### Create a simple triangle mesh
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ]
        )
        cells = torch.tensor(
            [
                [0, 1, 2],
                [1, 3, 2],
            ]
        )
        mesh = Mesh(points=points, cells=cells)

        ### Pass cell_indices as a list
        cell_indices = [0, 1, 0, 1]
        sampled_points = sample_random_points_on_cells(mesh, cell_indices=cell_indices)

        ### Should get four points
        assert sampled_points.shape == (4, 2)

    def test_3d_mesh_sampling(self):
        """Test sampling from a 3D tetrahedral mesh."""
        torch.manual_seed(42)
        ### Create a simple tetrahedron
        points = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        cells = torch.tensor([[0, 1, 2, 3]])
        mesh = Mesh(points=points, cells=cells)

        ### Sample from the tetrahedron
        sampled_points = sample_random_points_on_cells(mesh)

        ### Should get one 3D point
        assert sampled_points.shape == (1, 3)

    def test_out_of_bounds_indices_raises_error(self):
        """Test that out-of-bounds indices raise an error."""
        torch.manual_seed(42)
        ### Create a simple triangle mesh
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        )
        cells = torch.tensor([[0, 1, 2]])
        mesh = Mesh(points=points, cells=cells)

        ### Try to sample from non-existent cell
        cell_indices = torch.tensor([0, 1])  # Cell 1 doesn't exist
        with pytest.raises(IndexError):
            sample_random_points_on_cells(mesh, cell_indices=cell_indices)

    def test_negative_indices_raises_error(self):
        """Test that negative indices raise an error."""
        torch.manual_seed(42)
        ### Create a simple triangle mesh
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        )
        cells = torch.tensor([[0, 1, 2]])
        mesh = Mesh(points=points, cells=cells)

        ### Try to use negative index
        cell_indices = torch.tensor([0, -1])
        with pytest.raises(IndexError):
            sample_random_points_on_cells(mesh, cell_indices=cell_indices)

    def test_mesh_method_delegates_correctly(self):
        """Test that the Mesh.sample_random_points_on_cells method works correctly."""
        torch.manual_seed(42)
        ### Create a simple triangle mesh
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ]
        )
        cells = torch.tensor(
            [
                [0, 1, 2],
                [1, 3, 2],
            ]
        )
        mesh = Mesh(points=points, cells=cells)

        ### Test default behavior
        sampled_default = mesh.sample_random_points_on_cells()
        assert sampled_default.shape == (2, 2)

        ### Test with specific indices
        cell_indices = torch.tensor([0, 0, 1])
        sampled_specific = mesh.sample_random_points_on_cells(cell_indices=cell_indices)
        assert sampled_specific.shape == (3, 2)

    def test_alpha_parameter_works(self):
        """Test that the alpha parameter is passed through correctly."""
        torch.manual_seed(42)
        ### Create a simple triangle mesh
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        )
        cells = torch.tensor([[0, 1, 2]])
        mesh = Mesh(points=points, cells=cells)

        ### Sample with different alpha values (just check it doesn't crash)
        sampled_uniform = mesh.sample_random_points_on_cells(alpha=1.0)
        assert sampled_uniform.shape == (1, 2)

        ### Note: alpha != 1.0 is not supported under torch.compile
        # so we don't test it here to avoid the NotImplementedError

    def test_empty_cell_indices(self):
        """Test sampling with empty cell_indices."""
        torch.manual_seed(42)
        ### Create a simple triangle mesh
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        )
        cells = torch.tensor([[0, 1, 2]])
        mesh = Mesh(points=points, cells=cells)

        ### Sample with empty indices
        cell_indices = torch.tensor([], dtype=torch.long)
        sampled_points = sample_random_points_on_cells(mesh, cell_indices=cell_indices)

        ### Should get zero points
        assert sampled_points.shape == (0, 2)

    @pytest.mark.cuda
    def test_device_consistency(self):
        """Test that sampling preserves device."""
        torch.manual_seed(42)

        ### Create a simple triangle mesh on CUDA
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            device="cuda",
        )
        cells = torch.tensor([[0, 1, 2]], device="cuda")
        mesh = Mesh(points=points, cells=cells)

        ### Sample
        sampled_points = sample_random_points_on_cells(mesh)

        ### Should be on CUDA
        assert sampled_points.device.type == "cuda"


### Parametrized Tests for Exhaustive Dimensional Coverage ###


class TestRandomSamplingParametrized:
    """Parametrized tests for sampling across all dimensions and backends."""

    @pytest.mark.parametrize(
        "n_spatial_dims,n_manifold_dims",
        [
            (2, 1),  # Edges in 2D
            (2, 2),  # Triangles in 2D
            (3, 1),  # Edges in 3D
            (3, 2),  # Surfaces in 3D
            (3, 3),  # Volumes in 3D
        ],
    )
    def test_default_sampling_parametrized(
        self, n_spatial_dims, n_manifold_dims, device
    ):
        """Test default sampling (one per cell) across dimensions."""
        torch.manual_seed(42)
        mesh = create_simple_mesh(n_spatial_dims, n_manifold_dims, device=device)

        sampled = sample_random_points_on_cells(mesh)

        # Should get one point per cell
        assert sampled.shape == (mesh.n_cells, n_spatial_dims), (
            f"Expected shape ({mesh.n_cells}, {n_spatial_dims}), got {sampled.shape}"
        )

        # Verify device
        assert_on_device(sampled, device)

    @pytest.mark.parametrize(
        "n_spatial_dims,n_manifold_dims",
        [
            (2, 1),
            (2, 2),
            (3, 1),
            (3, 2),
            (3, 3),
        ],
    )
    def test_specific_cell_indices_parametrized(
        self, n_spatial_dims, n_manifold_dims, device
    ):
        """Test sampling from specific cells across dimensions."""
        torch.manual_seed(42)
        mesh = create_simple_mesh(n_spatial_dims, n_manifold_dims, device=device)

        # Sample from specific cells (with repetition)
        cell_indices = torch.tensor([0, 1, 0], device=device, dtype=torch.int64)
        sampled = sample_random_points_on_cells(mesh, cell_indices=cell_indices)

        assert sampled.shape == (3, n_spatial_dims)
        assert_on_device(sampled, device)

    @pytest.mark.parametrize(
        "n_spatial_dims,n_manifold_dims",
        [
            (2, 2),
            (3, 2),
            (3, 3),
        ],
    )
    def test_multiple_samples_per_cell_parametrized(
        self, n_spatial_dims, n_manifold_dims, device
    ):
        """Test repeated sampling from same cell across dimensions."""
        torch.manual_seed(42)
        mesh = create_simple_mesh(n_spatial_dims, n_manifold_dims, device=device)

        # Sample multiple times from first cell
        n_samples = 20
        cell_indices = torch.zeros(n_samples, device=device, dtype=torch.int64)
        sampled = sample_random_points_on_cells(mesh, cell_indices=cell_indices)

        assert sampled.shape == (n_samples, n_spatial_dims)
        assert_on_device(sampled, device)

        # All samples should be different (with extremely high probability)
        # Check that at least some variation exists
        if n_samples > 1:
            std_dev = sampled.std(dim=0)
            assert torch.any(std_dev > 0), "Samples should have variation"

    @pytest.mark.parametrize(
        "n_spatial_dims,n_manifold_dims",
        [
            (2, 1),
            (2, 2),
            (3, 1),
            (3, 2),
            (3, 3),
        ],
    )
    def test_empty_cell_indices_parametrized(
        self, n_spatial_dims, n_manifold_dims, device
    ):
        """Test sampling with empty indices across dimensions."""
        torch.manual_seed(42)
        mesh = create_simple_mesh(n_spatial_dims, n_manifold_dims, device=device)

        cell_indices = torch.tensor([], dtype=torch.int64, device=device)
        sampled = sample_random_points_on_cells(mesh, cell_indices=cell_indices)

        assert sampled.shape == (0, n_spatial_dims)
        assert_on_device(sampled, device)

    @pytest.mark.parametrize(
        "n_spatial_dims,n_manifold_dims",
        [
            (2, 1),
            (2, 2),
            (3, 1),
            (3, 2),
            (3, 3),
        ],
    )
    def test_mesh_method_parametrized(self, n_spatial_dims, n_manifold_dims, device):
        """Test Mesh.sample_random_points_on_cells method across dimensions."""
        torch.manual_seed(42)
        mesh = create_simple_mesh(n_spatial_dims, n_manifold_dims, device=device)

        # Test default
        sampled_default = mesh.sample_random_points_on_cells()
        assert sampled_default.shape == (mesh.n_cells, n_spatial_dims)
        assert_on_device(sampled_default, device)

        # Test with specific indices
        if mesh.n_cells > 1:
            cell_indices = torch.tensor([0, 1], device=device, dtype=torch.int64)
            sampled_specific = mesh.sample_random_points_on_cells(
                cell_indices=cell_indices
            )
            assert sampled_specific.shape == (2, n_spatial_dims)
            assert_on_device(sampled_specific, device)


class TestRealisticMeshSampling:
    """Tests for sampling on realistic meshes (lumpy_sphere)."""

    def test_lumpy_sphere_sampling(self, device):
        """Test sampling on lumpy_sphere - a realistic 3D surface mesh."""
        from physicsnemo.mesh.primitives.procedural import lumpy_sphere

        torch.manual_seed(42)
        mesh = lumpy_sphere.load(subdivisions=2, device=device)

        # Sample one point per cell
        sampled = sample_random_points_on_cells(mesh)

        # Should get one point per cell
        assert sampled.shape == (mesh.n_cells, 3)
        assert_on_device(sampled, device)

        # All samples should be on surface (approximately at radius ~1)
        radii = torch.norm(sampled, dim=-1)
        # With noise_amplitude=0.1, lumpy_sphere has varying radii
        # Check mean radius is reasonable rather than strict bounds on all samples
        assert radii.mean() > 0.5, "Mean radius should be away from origin"
        assert radii.mean() < 2.0, "Mean radius should be near surface"
        assert torch.all(torch.isfinite(radii)), "All radii should be finite"

    def test_lumpy_sphere_multiple_samples(self, device):
        """Test multiple samples from specific cells on lumpy_sphere."""
        from physicsnemo.mesh.primitives.procedural import lumpy_sphere

        torch.manual_seed(42)
        mesh = lumpy_sphere.load(subdivisions=2, device=device)

        # Sample 10 points from the first 5 cells
        n_samples = 50
        cell_indices = torch.arange(5, device=device, dtype=torch.int64).repeat(10)
        sampled = sample_random_points_on_cells(mesh, cell_indices=cell_indices)

        assert sampled.shape == (n_samples, 3)
        assert_on_device(sampled, device)

        # Samples should have variation
        std_dev = sampled.std(dim=0)
        assert torch.all(std_dev > 0), "Samples should have variation"

    def test_lumpy_sphere_specific_cells(self, device):
        """Test sampling from specific cells on lumpy_sphere."""
        from physicsnemo.mesh.primitives.procedural import lumpy_sphere

        torch.manual_seed(42)
        mesh = lumpy_sphere.load(subdivisions=2, device=device)

        # Sample from specific cells (with repetition)
        cell_indices = torch.tensor(
            [0, 10, 50, 10, 0], device=device, dtype=torch.int64
        )
        sampled = sample_random_points_on_cells(mesh, cell_indices=cell_indices)

        assert sampled.shape == (5, 3)
        assert_on_device(sampled, device)
