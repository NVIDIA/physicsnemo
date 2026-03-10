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

"""Tests for Mesh.__init__() validation error paths.

Tests validate that the Mesh constructor correctly validates inputs and raises
appropriate errors for:
- Non-2D points tensor
- Non-2D cells tensor
- manifold_dims > spatial_dims
- Floating-point cells dtype
"""

import pytest
import torch
from tensordict import TensorDict

from physicsnemo.mesh import Mesh


class TestPointsValidation:
    """Tests for points tensor validation."""

    def test_valid_points_2d(self):
        """Test that valid 2D points are accepted."""
        points = torch.randn(10, 3)
        cells = torch.randint(0, 10, (5, 3))
        mesh = Mesh(points=points, cells=cells)
        assert mesh.n_points == 10
        assert mesh.n_spatial_dims == 3

    def test_points_1d_raises(self):
        """Test that 1D points tensor raises ValueError."""
        points = torch.randn(10)  # 1D tensor
        cells = torch.randint(0, 10, (5, 3))

        with pytest.raises(ValueError, match=r"`points` must have shape.*got.*shape"):
            Mesh(points=points, cells=cells)

    def test_points_3d_raises(self):
        """Test that 3D points tensor raises ValueError."""
        points = torch.randn(10, 3, 2)  # 3D tensor
        cells = torch.randint(0, 10, (5, 3))

        with pytest.raises(ValueError, match=r"`points` must have shape.*got.*shape"):
            Mesh(points=points, cells=cells)

    def test_points_0d_raises(self):
        """Test that 0D (scalar) points tensor causes an error during construction."""
        points = torch.tensor(1.0)  # 0D tensor
        cells = torch.randint(0, 10, (5, 3))

        # This will raise an IndexError during n_points property access before validation,
        # which is acceptable behavior for invalid input
        with pytest.raises((ValueError, IndexError)):
            Mesh(points=points, cells=cells)

    def test_points_4d_raises(self):
        """Test that 4D points tensor raises ValueError."""
        points = torch.randn(2, 10, 3, 4)  # 4D tensor
        cells = torch.randint(0, 10, (5, 3))

        with pytest.raises(ValueError, match=r"`points` must have shape.*got.*shape"):
            Mesh(points=points, cells=cells)


class TestCellsValidation:
    """Tests for cells tensor validation."""

    def test_valid_cells_2d(self):
        """Test that valid 2D cells are accepted."""
        points = torch.randn(10, 3)
        cells = torch.randint(0, 10, (5, 3))
        mesh = Mesh(points=points, cells=cells)
        assert mesh.n_cells == 5
        assert mesh.n_manifold_dims == 2

    def test_cells_1d_raises(self):
        """Test that 1D cells tensor raises ValueError."""
        points = torch.randn(10, 3)
        cells = torch.randint(0, 10, (5,))  # 1D tensor

        with pytest.raises(ValueError, match=r"`cells` must have shape.*got.*shape"):
            Mesh(points=points, cells=cells)

    def test_cells_3d_raises(self):
        """Test that 3D cells tensor raises ValueError."""
        points = torch.randn(10, 3)
        cells = torch.randint(0, 10, (5, 3, 2))  # 3D tensor

        with pytest.raises(ValueError, match=r"`cells` must have shape.*got.*shape"):
            Mesh(points=points, cells=cells)

    def test_cells_0d_raises(self):
        """Test that 0D (scalar) cells tensor causes an error during construction."""
        points = torch.randn(10, 3)
        cells = torch.tensor(0)  # 0D tensor

        # This will raise an IndexError during n_cells property access before validation,
        # which is acceptable behavior for invalid input
        with pytest.raises((ValueError, IndexError)):
            Mesh(points=points, cells=cells)


class TestCellsDtypeValidation:
    """Tests for cells dtype validation."""

    def test_cells_int64_valid(self):
        """Test that int64 cells are accepted."""
        points = torch.randn(10, 3)
        cells = torch.randint(0, 10, (5, 3), dtype=torch.int64)
        mesh = Mesh(points=points, cells=cells)
        assert mesh.cells.dtype == torch.int64

    def test_cells_int32_valid(self):
        """Test that int32 cells are accepted."""
        points = torch.randn(10, 3)
        cells = torch.randint(0, 10, (5, 3), dtype=torch.int32)
        mesh = Mesh(points=points, cells=cells)
        assert mesh.cells.dtype == torch.int32

    def test_cells_int16_valid(self):
        """Test that int16 cells are accepted."""
        points = torch.randn(10, 3)
        cells = torch.randint(0, 10, (5, 3), dtype=torch.int16)
        mesh = Mesh(points=points, cells=cells)
        assert mesh.cells.dtype == torch.int16

    def test_cells_float32_raises(self):
        """Test that float32 cells raise TypeError."""
        points = torch.randn(10, 3)
        cells = torch.randint(0, 10, (5, 3)).float()  # float32

        with pytest.raises(TypeError, match=r"`cells` must have an int-like dtype"):
            Mesh(points=points, cells=cells)

    def test_cells_float64_raises(self):
        """Test that float64 cells raise TypeError."""
        points = torch.randn(10, 3)
        cells = torch.randint(0, 10, (5, 3)).double()  # float64

        with pytest.raises(TypeError, match=r"`cells` must have an int-like dtype"):
            Mesh(points=points, cells=cells)

    def test_cells_float16_raises(self):
        """Test that float16 cells raise TypeError."""
        points = torch.randn(10, 3)
        cells = torch.randint(0, 10, (5, 3)).half()  # float16

        with pytest.raises(TypeError, match=r"`cells` must have an int-like dtype"):
            Mesh(points=points, cells=cells)


class TestDimensionValidation:
    """Tests for dimension relationship validation."""

    def test_manifold_less_than_spatial_valid(self):
        """Test that manifold_dims < spatial_dims is valid."""
        points = torch.randn(10, 3)  # 3D spatial
        cells = torch.randint(0, 10, (5, 3))  # 2-manifold (triangles)
        mesh = Mesh(points=points, cells=cells)
        assert mesh.n_manifold_dims == 2
        assert mesh.n_spatial_dims == 3

    def test_manifold_equals_spatial_valid(self):
        """Test that manifold_dims == spatial_dims is valid."""
        points = torch.randn(10, 3)  # 3D spatial
        cells = torch.randint(0, 10, (5, 4))  # 3-manifold (tetrahedra)
        mesh = Mesh(points=points, cells=cells)
        assert mesh.n_manifold_dims == 3
        assert mesh.n_spatial_dims == 3

    def test_manifold_greater_than_spatial_raises(self):
        """Test that manifold_dims > spatial_dims raises ValueError."""
        points = torch.randn(10, 2)  # 2D spatial
        cells = torch.randint(
            0, 10, (5, 4)
        )  # 3-manifold (would need at least 3D space)

        with pytest.raises(
            ValueError, match=r"`n_manifold_dims` must be <= `n_spatial_dims`"
        ):
            Mesh(points=points, cells=cells)

    def test_0_manifold_in_0_spatial_invalid(self):
        """Test that 0D points with 0-simplex cells requires validation."""
        # 0D points (empty spatial dimension)
        points = torch.randn(10, 0)
        # 0-simplex cells (single vertex per cell) -> n_manifold_dims = 0
        cells = torch.randint(0, 10, (5, 1))

        # This should be valid: 0-manifold in 0D space
        mesh = Mesh(points=points, cells=cells)
        assert mesh.n_manifold_dims == 0
        assert mesh.n_spatial_dims == 0

    def test_edges_in_2d_valid(self):
        """Test that edges (1-simplices) in 2D space are valid."""
        points = torch.randn(10, 2)  # 2D spatial
        cells = torch.randint(0, 10, (5, 2))  # 1-manifold (edges)
        mesh = Mesh(points=points, cells=cells)
        assert mesh.n_manifold_dims == 1
        assert mesh.n_spatial_dims == 2
        assert mesh.codimension == 1


class TestDataInputValidation:
    """Tests for point_data, cell_data, and global_data input handling."""

    def test_point_data_as_dict(self):
        """Test that point_data can be provided as a dict."""
        points = torch.randn(10, 3)
        cells = torch.randint(0, 10, (5, 3))
        point_data = {"temperature": torch.randn(10)}

        mesh = Mesh(points=points, cells=cells, point_data=point_data)

        assert "temperature" in mesh.point_data
        assert mesh.point_data["temperature"].shape == (10,)

    def test_point_data_as_tensordict(self):
        """Test that point_data can be provided as a TensorDict."""
        points = torch.randn(10, 3)
        cells = torch.randint(0, 10, (5, 3))
        point_data = TensorDict({"temperature": torch.randn(10)}, batch_size=[10])

        mesh = Mesh(points=points, cells=cells, point_data=point_data)

        assert "temperature" in mesh.point_data
        assert mesh.point_data["temperature"].shape == (10,)

    def test_cell_data_as_dict(self):
        """Test that cell_data can be provided as a dict."""
        points = torch.randn(10, 3)
        cells = torch.randint(0, 10, (5, 3))
        cell_data = {"pressure": torch.randn(5)}

        mesh = Mesh(points=points, cells=cells, cell_data=cell_data)

        assert "pressure" in mesh.cell_data
        assert mesh.cell_data["pressure"].shape == (5,)

    def test_global_data_as_dict(self):
        """Test that global_data can be provided as a dict."""
        points = torch.randn(10, 3)
        cells = torch.randint(0, 10, (5, 3))
        global_data = {"time": torch.tensor(1.0)}

        mesh = Mesh(points=points, cells=cells, global_data=global_data)

        assert "time" in mesh.global_data
        assert mesh.global_data["time"].shape == ()

    def test_none_data_creates_empty_tensordicts(self):
        """Test that None data creates empty TensorDicts."""
        points = torch.randn(10, 3)
        cells = torch.randint(0, 10, (5, 3))

        mesh = Mesh(points=points, cells=cells)

        assert isinstance(mesh.point_data, TensorDict)
        assert isinstance(mesh.cell_data, TensorDict)
        assert isinstance(mesh.global_data, TensorDict)
        assert len(mesh.point_data.keys()) == 0
        assert len(mesh.cell_data.keys()) == 0
        assert len(mesh.global_data.keys()) == 0


class TestValidMeshCreation:
    """Tests for valid mesh creation scenarios."""

    def test_empty_mesh_valid(self):
        """Test that empty mesh (0 points, 0 cells) can be created."""
        points = torch.zeros((0, 3))
        cells = torch.zeros((0, 3), dtype=torch.long)

        mesh = Mesh(points=points, cells=cells)

        assert mesh.n_points == 0
        assert mesh.n_cells == 0

    def test_minimal_mesh_valid(self):
        """Test that minimal mesh (1 point, 1 cell) can be created."""
        points = torch.tensor([[0.0, 0.0, 0.0]])
        cells = torch.tensor([[0]])

        mesh = Mesh(points=points, cells=cells)

        assert mesh.n_points == 1
        assert mesh.n_cells == 1
        assert mesh.n_manifold_dims == 0
        assert mesh.n_spatial_dims == 3

    def test_point_cloud_no_cells(self):
        """Test that omitting cells creates a valid point-cloud mesh."""
        points = torch.randn(50, 3)
        mesh = Mesh(points=points)
        assert mesh.n_points == 50
        assert mesh.n_cells == 0
        assert mesh.n_manifold_dims == 0
        assert mesh.cells.shape == (0, 1)
        assert mesh.cells.dtype == torch.long
        assert mesh.cells.device == points.device

    def test_large_spatial_dims_valid(self):
        """Test that high-dimensional spatial embedding is valid."""
        points = torch.randn(10, 10)  # 10D spatial embedding
        cells = torch.randint(0, 10, (5, 3))  # 2-manifold

        mesh = Mesh(points=points, cells=cells)

        assert mesh.n_spatial_dims == 10
        assert mesh.n_manifold_dims == 2
        assert mesh.codimension == 8


class TestParametrized:
    """Parametrized tests for initialization."""

    @pytest.mark.parametrize(
        "n_points,n_cells,n_spatial_dims,n_manifold_dims",
        [
            (10, 5, 2, 1),
            (10, 5, 2, 2),
            (10, 5, 3, 1),
            (10, 5, 3, 2),
            (10, 5, 3, 3),
            (100, 50, 3, 2),
            (1, 1, 3, 0),
        ],
    )
    def test_valid_combinations(
        self, n_points, n_cells, n_spatial_dims, n_manifold_dims
    ):
        """Test various valid dimension combinations."""
        torch.manual_seed(42)
        points = torch.randn(n_points, n_spatial_dims)
        cells = torch.randint(0, max(1, n_points), (n_cells, n_manifold_dims + 1))

        mesh = Mesh(points=points, cells=cells)

        assert mesh.n_points == n_points
        assert mesh.n_cells == n_cells
        assert mesh.n_spatial_dims == n_spatial_dims
        assert mesh.n_manifold_dims == n_manifold_dims
