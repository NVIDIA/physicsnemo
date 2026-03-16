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

"""Tests for Mesh.merge() class method.

Tests validate merging multiple meshes into a single mesh, including:
- Basic merging of meshes with consistent dimensions
- Cell index remapping after merge
- Point and cell data concatenation
- Global data stacking
- Error handling for invalid inputs
- Device preservation
"""

import pytest
import torch

from physicsnemo.mesh import Mesh

### Helper Functions ###


def create_simple_mesh(
    n_points: int,
    n_cells: int,
    n_spatial_dims: int,
    n_manifold_dims: int,
    device: str = "cpu",
    add_point_data: bool = False,
    add_cell_data: bool = False,
    add_global_data: bool = False,
) -> Mesh:
    """Create a simple mesh for testing."""
    torch.manual_seed(42)
    points = torch.randn(n_points, n_spatial_dims, device=device)
    cells = torch.randint(0, n_points, (n_cells, n_manifold_dims + 1), device=device)

    point_data = {}
    cell_data = {}
    global_data = {}

    if add_point_data:
        point_data["temperature"] = torch.randn(n_points, device=device)

    if add_cell_data:
        cell_data["pressure"] = torch.randn(n_cells, device=device)

    if add_global_data:
        global_data["time"] = torch.tensor(1.0, device=device)

    return Mesh(
        points=points,
        cells=cells,
        point_data=point_data,
        cell_data=cell_data,
        global_data=global_data,
    )


### Test Classes ###


class TestMergeBasic:
    """Tests for basic merge functionality."""

    def test_merge_two_meshes(self):
        """Test merging two meshes with consistent dimensions."""
        mesh1 = create_simple_mesh(
            n_points=10, n_cells=5, n_spatial_dims=3, n_manifold_dims=2
        )
        mesh2 = create_simple_mesh(
            n_points=15, n_cells=8, n_spatial_dims=3, n_manifold_dims=2
        )

        merged = Mesh.merge([mesh1, mesh2])

        assert merged.n_points == 10 + 15
        assert merged.n_cells == 5 + 8
        assert merged.n_spatial_dims == 3
        assert merged.n_manifold_dims == 2

    def test_merge_three_meshes(self):
        """Test merging three meshes."""
        mesh1 = create_simple_mesh(
            n_points=5, n_cells=3, n_spatial_dims=2, n_manifold_dims=1
        )
        mesh2 = create_simple_mesh(
            n_points=7, n_cells=4, n_spatial_dims=2, n_manifold_dims=1
        )
        mesh3 = create_simple_mesh(
            n_points=9, n_cells=6, n_spatial_dims=2, n_manifold_dims=1
        )

        merged = Mesh.merge([mesh1, mesh2, mesh3])

        assert merged.n_points == 5 + 7 + 9
        assert merged.n_cells == 3 + 4 + 6

    def test_merge_single_mesh_returns_clone(self):
        """Test that merging a single mesh returns an equal but distinct copy."""
        mesh = create_simple_mesh(
            n_points=10, n_cells=5, n_spatial_dims=3, n_manifold_dims=2
        )

        merged = Mesh.merge([mesh])

        # Should be equal but not the same object (no aliasing)
        assert merged is not mesh
        assert torch.equal(merged.points, mesh.points)
        assert torch.equal(merged.cells, mesh.cells)

    def test_merge_preserves_points(self):
        """Test that merged points are correctly concatenated."""
        points1 = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
        cells1 = torch.tensor([[0, 1]])
        mesh1 = Mesh(points=points1, cells=cells1)

        points2 = torch.tensor([[2.0, 0.0], [3.0, 0.0]])
        cells2 = torch.tensor([[0, 1]])
        mesh2 = Mesh(points=points2, cells=cells2)

        merged = Mesh.merge([mesh1, mesh2])

        expected_points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        assert torch.allclose(merged.points, expected_points)


class TestMergeCellIndexRemapping:
    """Tests for cell index remapping after merge."""

    def test_cell_indices_remapped(self):
        """Test that cell indices are correctly remapped after merge."""
        points1 = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]])
        cells1 = torch.tensor([[0, 1, 2]])
        mesh1 = Mesh(points=points1, cells=cells1)

        points2 = torch.tensor([[2.0, 0.0], [3.0, 0.0], [2.5, 1.0]])
        cells2 = torch.tensor([[0, 1, 2]])
        mesh2 = Mesh(points=points2, cells=cells2)

        merged = Mesh.merge([mesh1, mesh2])

        # First cell should remain unchanged
        assert merged.cells[0].tolist() == [0, 1, 2]
        # Second cell should be offset by 3 (number of points in mesh1)
        assert merged.cells[1].tolist() == [3, 4, 5]

    def test_cell_indices_multiple_meshes(self):
        """Test cell index remapping with multiple meshes."""
        points1 = torch.tensor([[0.0, 0.0]])
        cells1 = torch.tensor([[0]])
        mesh1 = Mesh(points=points1, cells=cells1)

        points2 = torch.tensor([[1.0, 0.0], [2.0, 0.0]])
        cells2 = torch.tensor([[0], [1]])
        mesh2 = Mesh(points=points2, cells=cells2)

        points3 = torch.tensor([[3.0, 0.0], [4.0, 0.0], [5.0, 0.0]])
        cells3 = torch.tensor([[0], [1], [2]])
        mesh3 = Mesh(points=points3, cells=cells3)

        merged = Mesh.merge([mesh1, mesh2, mesh3])

        # Mesh1 cells: offset = 0
        assert merged.cells[0].tolist() == [0]
        # Mesh2 cells: offset = 1
        assert merged.cells[1].tolist() == [1]
        assert merged.cells[2].tolist() == [2]
        # Mesh3 cells: offset = 1 + 2 = 3
        assert merged.cells[3].tolist() == [3]
        assert merged.cells[4].tolist() == [4]
        assert merged.cells[5].tolist() == [5]


class TestMergeWithData:
    """Tests for merging meshes with attached data."""

    def test_merge_with_point_data(self):
        """Test that point_data is correctly concatenated."""
        mesh1 = create_simple_mesh(
            n_points=5,
            n_cells=3,
            n_spatial_dims=2,
            n_manifold_dims=1,
            add_point_data=True,
        )
        mesh2 = create_simple_mesh(
            n_points=7,
            n_cells=4,
            n_spatial_dims=2,
            n_manifold_dims=1,
            add_point_data=True,
        )

        merged = Mesh.merge([mesh1, mesh2])

        assert "temperature" in merged.point_data
        assert merged.point_data["temperature"].shape[0] == 5 + 7

    def test_merge_with_cell_data(self):
        """Test that cell_data is correctly concatenated."""
        mesh1 = create_simple_mesh(
            n_points=5,
            n_cells=3,
            n_spatial_dims=2,
            n_manifold_dims=1,
            add_cell_data=True,
        )
        mesh2 = create_simple_mesh(
            n_points=7,
            n_cells=4,
            n_spatial_dims=2,
            n_manifold_dims=1,
            add_cell_data=True,
        )

        merged = Mesh.merge([mesh1, mesh2])

        assert "pressure" in merged.cell_data
        assert merged.cell_data["pressure"].shape[0] == 3 + 4

    def test_merge_with_global_data_stacking(self):
        """Test that global_data is stacked along a new dimension."""
        points1 = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
        cells1 = torch.tensor([[0, 1]])
        mesh1 = Mesh(
            points=points1, cells=cells1, global_data={"time": torch.tensor(1.0)}
        )

        points2 = torch.tensor([[2.0, 0.0], [3.0, 0.0]])
        cells2 = torch.tensor([[0, 1]])
        mesh2 = Mesh(
            points=points2, cells=cells2, global_data={"time": torch.tensor(2.0)}
        )

        merged = Mesh.merge([mesh1, mesh2], global_data_strategy="stack")

        assert "time" in merged.global_data.keys()
        # Global data should be stacked - time should have shape [2]
        assert merged.global_data["time"].shape == torch.Size([2])

    def test_merge_with_vector_data(self):
        """Test merging with vector-valued data fields."""
        points1 = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]])
        cells1 = torch.tensor([[0, 1, 2]])
        mesh1 = Mesh(
            points=points1,
            cells=cells1,
            point_data={
                "velocity": torch.tensor(
                    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
                )
            },
        )

        points2 = torch.tensor([[2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [2.5, 1.0, 0.0]])
        cells2 = torch.tensor([[0, 1, 2]])  # Same manifold dimension (triangles)
        mesh2 = Mesh(
            points=points2,
            cells=cells2,
            point_data={
                "velocity": torch.tensor(
                    [[1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0]]
                )
            },
        )

        merged = Mesh.merge([mesh1, mesh2])

        assert merged.point_data["velocity"].shape == (6, 3)


class TestMergeErrors:
    """Tests for error handling in merge."""

    def test_merge_empty_list_raises(self):
        """Test that merging empty list raises ValueError."""
        with pytest.raises(ValueError, match="At least one Mesh must be provided"):
            Mesh.merge([])

    def test_merge_non_mesh_raises(self):
        """Test that merging non-Mesh objects raises TypeError."""
        mesh = create_simple_mesh(
            n_points=5, n_cells=3, n_spatial_dims=2, n_manifold_dims=1
        )

        with pytest.raises(TypeError, match="All objects must be Mesh types"):
            Mesh.merge([mesh, "not a mesh"])

    def test_merge_mismatched_spatial_dims_raises(self):
        """Test that merging meshes with different spatial dimensions raises ValueError."""
        mesh1 = create_simple_mesh(
            n_points=5, n_cells=3, n_spatial_dims=2, n_manifold_dims=1
        )
        mesh2 = create_simple_mesh(
            n_points=5, n_cells=3, n_spatial_dims=3, n_manifold_dims=1
        )

        with pytest.raises(ValueError, match="spatial dimensions"):
            Mesh.merge([mesh1, mesh2])

    def test_merge_mismatched_manifold_dims_raises(self):
        """Test that merging meshes with different manifold dimensions raises ValueError."""
        mesh1 = create_simple_mesh(
            n_points=5, n_cells=3, n_spatial_dims=3, n_manifold_dims=1
        )
        mesh2 = create_simple_mesh(
            n_points=5, n_cells=3, n_spatial_dims=3, n_manifold_dims=2
        )

        with pytest.raises(ValueError, match="manifold dimensions"):
            Mesh.merge([mesh1, mesh2])

    def test_merge_mismatched_cell_data_keys_raises(self):
        """Test that merging meshes with different cell_data keys raises ValueError."""
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
        cells = torch.tensor([[0, 1]])

        mesh1 = Mesh(
            points=points, cells=cells, cell_data={"pressure": torch.tensor([1.0])}
        )
        mesh2 = Mesh(
            points=points.clone(),
            cells=cells.clone(),
            cell_data={"temperature": torch.tensor([2.0])},
        )

        with pytest.raises(ValueError, match="same cell_data keys"):
            Mesh.merge([mesh1, mesh2])

    def test_merge_invalid_global_data_strategy_raises(self):
        """Test that invalid global_data_strategy raises ValueError."""
        mesh1 = create_simple_mesh(
            n_points=5, n_cells=3, n_spatial_dims=2, n_manifold_dims=1
        )
        mesh2 = create_simple_mesh(
            n_points=5, n_cells=3, n_spatial_dims=2, n_manifold_dims=1
        )

        with pytest.raises(ValueError, match="Invalid global_data_strategy"):
            Mesh.merge([mesh1, mesh2], global_data_strategy="invalid")


class TestMergeDeviceHandling:
    """Tests for device handling in merge."""

    def test_merge_preserves_cpu_device(self):
        """Test that merging CPU meshes produces CPU result."""
        mesh1 = create_simple_mesh(
            n_points=5, n_cells=3, n_spatial_dims=2, n_manifold_dims=1, device="cpu"
        )
        mesh2 = create_simple_mesh(
            n_points=7, n_cells=4, n_spatial_dims=2, n_manifold_dims=1, device="cpu"
        )

        merged = Mesh.merge([mesh1, mesh2])

        assert merged.points.device.type == "cpu"
        assert merged.cells.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_merge_preserves_cuda_device(self):
        """Test that merging CUDA meshes produces CUDA result."""
        mesh1 = create_simple_mesh(
            n_points=5, n_cells=3, n_spatial_dims=2, n_manifold_dims=1, device="cuda"
        )
        mesh2 = create_simple_mesh(
            n_points=7, n_cells=4, n_spatial_dims=2, n_manifold_dims=1, device="cuda"
        )

        merged = Mesh.merge([mesh1, mesh2])

        assert merged.points.device.type == "cuda"
        assert merged.cells.device.type == "cuda"


class TestMergeParametrized:
    """Parametrized tests for merge across dimensions."""

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
    def test_merge_various_dimensions(self, n_spatial_dims, n_manifold_dims, device):
        """Test merge across various dimension combinations."""
        mesh1 = create_simple_mesh(
            n_points=10,
            n_cells=5,
            n_spatial_dims=n_spatial_dims,
            n_manifold_dims=n_manifold_dims,
            device=device,
        )
        mesh2 = create_simple_mesh(
            n_points=15,
            n_cells=8,
            n_spatial_dims=n_spatial_dims,
            n_manifold_dims=n_manifold_dims,
            device=device,
        )

        merged = Mesh.merge([mesh1, mesh2])

        assert merged.n_points == 25
        assert merged.n_cells == 13
        assert merged.n_spatial_dims == n_spatial_dims
        assert merged.n_manifold_dims == n_manifold_dims
        assert merged.points.device.type == device

    @pytest.mark.parametrize("n_meshes", [2, 3, 5, 10])
    def test_merge_many_meshes(self, n_meshes):
        """Test merging varying numbers of meshes."""
        meshes = [
            create_simple_mesh(
                n_points=10, n_cells=5, n_spatial_dims=2, n_manifold_dims=1
            )
            for _ in range(n_meshes)
        ]

        merged = Mesh.merge(meshes)

        assert merged.n_points == 10 * n_meshes
        assert merged.n_cells == 5 * n_meshes
