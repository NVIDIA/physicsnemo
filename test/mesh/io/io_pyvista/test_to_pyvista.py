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

"""Tests for physicsnemo.mesh.io module - to_pyvista conversion."""

import numpy as np
import pytest
import torch

from physicsnemo.mesh.mesh import Mesh

pv = pytest.importorskip("pyvista")

from physicsnemo.mesh.io.io_pyvista import to_pyvista  # noqa: E402


class TestToPyvista:
    """Tests for converting physicsnemo.mesh Mesh back to PyVista."""

    def test_2d_mesh_to_polydata(self):
        """Test converting 2D mesh to PolyData."""
        # Create a simple triangular mesh
        points = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 1.0, 0.0],
                [1.5, 1.0, 0.0],
            ]
        )
        cells = torch.tensor([[0, 1, 2], [1, 3, 2]], dtype=torch.long)

        mesh = Mesh(points=points, cells=cells)
        pv_mesh = to_pyvista(mesh)

        # Verify it's PolyData
        assert isinstance(pv_mesh, pv.PolyData)
        assert pv_mesh.n_points == 4
        assert pv_mesh.n_cells == 2
        assert pv_mesh.is_all_triangles

        # Verify points match
        assert np.allclose(pv_mesh.points, points.numpy())

    def test_3d_mesh_to_unstructured_grid(self):
        """Test converting 3D mesh to UnstructuredGrid."""
        # Create a simple tetrahedral mesh
        points = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        cells = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)

        mesh = Mesh(points=points, cells=cells)
        pv_mesh = to_pyvista(mesh)

        # Verify it's UnstructuredGrid
        assert isinstance(pv_mesh, pv.UnstructuredGrid)
        assert pv_mesh.n_points == 4
        assert pv_mesh.n_cells == 1
        assert list(pv_mesh.cells_dict.keys()) == [pv.CellType.TETRA]

        # Verify connectivity
        assert np.array_equal(pv_mesh.cells_dict[pv.CellType.TETRA], cells.numpy())

    def test_1d_mesh_to_polydata(self):
        """Test converting 1D mesh to PolyData with lines."""
        # Create a simple line mesh
        points = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ]
        )
        cells = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)

        mesh = Mesh(points=points, cells=cells)
        pv_mesh = to_pyvista(mesh)

        # Verify it's PolyData
        assert isinstance(pv_mesh, pv.PolyData)
        assert pv_mesh.n_points == 3
        assert pv_mesh.n_lines == 2

    def test_0d_mesh_to_pointset(self):
        """Test converting 0D mesh to PointSet."""
        points = torch.from_numpy(np.random.rand(50, 3).astype(np.float32))
        cells = torch.empty((0, 1), dtype=torch.long)

        mesh = Mesh(points=points, cells=cells)
        pv_mesh = to_pyvista(mesh)

        # Verify it's PointSet
        assert isinstance(pv_mesh, pv.PointSet)
        assert pv_mesh.n_points == 50
        assert np.allclose(pv_mesh.points, points.numpy())

    def test_data_preservation_to_pyvista(self):
        """Test that point_data, cell_data, and global_data are preserved."""
        # Create a mesh with data
        points = torch.rand(10, 3)
        cells = torch.tensor([[0, 1, 2], [2, 3, 4]], dtype=torch.long)

        mesh = Mesh(points=points, cells=cells)

        # Add data to the mesh
        mesh.point_data["temperature"] = torch.rand(10)
        mesh.point_data["velocity"] = torch.rand(10, 3)
        mesh.cell_data["pressure"] = torch.rand(2)
        mesh.global_data["time"] = torch.tensor([1.5])

        pv_mesh = to_pyvista(mesh)

        # Verify data is preserved
        assert "temperature" in pv_mesh.point_data
        assert "velocity" in pv_mesh.point_data
        assert "pressure" in pv_mesh.cell_data
        assert "time" in pv_mesh.field_data

        # Verify values match
        assert np.allclose(
            pv_mesh.point_data["temperature"], mesh.point_data["temperature"].numpy()
        )
        assert np.allclose(
            pv_mesh.cell_data["pressure"], mesh.cell_data["pressure"].numpy()
        )


class TestHighRankTensorFlattening:
    """Tests for high-rank tensor flattening in to_pyvista conversion.

    VTK only supports arrays with dimensionality <= 2. Higher-rank tensors
    (e.g., stress tensors with shape (n, 3, 3)) must be flattened to
    (n, 9) for VTK compatibility.
    """

    def test_rank2_tensor_flattened(self):
        """Test that rank-2 tensors are flattened correctly."""
        points = torch.rand(10, 3)
        cells = torch.tensor([[0, 1, 2], [2, 3, 4]], dtype=torch.long)
        mesh = Mesh(points=points, cells=cells)

        # Add rank-2 tensor (3x3 stress tensor per cell)
        stress_data = torch.rand(2, 3, 3)
        mesh.cell_data["stress"] = stress_data

        pv_mesh = to_pyvista(mesh)

        # Verify key is preserved
        assert "stress" in pv_mesh.cell_data

        # Verify shape is flattened correctly
        assert pv_mesh.cell_data["stress"].shape == (2, 9)

        # Verify values are preserved (raveled)
        expected = stress_data.numpy().reshape(2, 9)
        assert np.allclose(pv_mesh.cell_data["stress"], expected)

    def test_rank3_tensor_flattened(self):
        """Test that rank-3 tensors are flattened correctly."""
        points = torch.rand(10, 3)
        cells = torch.tensor([[0, 1, 2]], dtype=torch.long)
        mesh = Mesh(points=points, cells=cells)

        # Add rank-3 tensor (2x3x4 tensor per cell)
        tensor_data = torch.rand(1, 2, 3, 4)
        mesh.cell_data["elasticity"] = tensor_data

        pv_mesh = to_pyvista(mesh)

        # Verify key and shape
        assert "elasticity" in pv_mesh.cell_data
        assert pv_mesh.cell_data["elasticity"].shape == (1, 24)

        # Verify values
        expected = tensor_data.numpy().reshape(1, 24)
        assert np.allclose(pv_mesh.cell_data["elasticity"], expected)

    def test_point_data_high_rank_flattened(self):
        """Test that high-rank point_data is also flattened."""
        points = torch.rand(5, 3)
        cells = torch.tensor([[0, 1, 2]], dtype=torch.long)
        mesh = Mesh(points=points, cells=cells)

        # Add rank-2 tensor to point_data
        jacobian_data = torch.rand(5, 2, 2)
        mesh.point_data["jacobian"] = jacobian_data

        pv_mesh = to_pyvista(mesh)

        assert "jacobian" in pv_mesh.point_data
        assert pv_mesh.point_data["jacobian"].shape == (5, 4)

    def test_global_data_high_rank_flattened(self):
        """Test that high-rank global_data is also flattened."""
        points = torch.rand(5, 3)
        cells = torch.tensor([[0, 1, 2]], dtype=torch.long)
        mesh = Mesh(points=points, cells=cells)

        # Add rank-2 tensor to global_data (single 3x3 matrix)
        transform_data = torch.rand(1, 3, 3)
        mesh.global_data["transform"] = transform_data

        pv_mesh = to_pyvista(mesh)

        assert "transform" in pv_mesh.field_data
        assert pv_mesh.field_data["transform"].shape == (1, 9)

    def test_low_rank_tensors_unchanged(self):
        """Test that scalars and vectors are not modified."""
        points = torch.rand(10, 3)
        cells = torch.tensor([[0, 1, 2], [2, 3, 4]], dtype=torch.long)
        mesh = Mesh(points=points, cells=cells)

        # Add scalar and vector data (should not be flattened)
        mesh.point_data["temperature"] = torch.rand(10)
        mesh.point_data["velocity"] = torch.rand(10, 3)
        mesh.cell_data["pressure"] = torch.rand(2)

        pv_mesh = to_pyvista(mesh)

        # Keys should be unchanged (no shape suffix)
        assert "temperature" in pv_mesh.point_data
        assert "velocity" in pv_mesh.point_data
        assert "pressure" in pv_mesh.cell_data

        # Shapes should be unchanged
        assert pv_mesh.point_data["temperature"].shape == (10,)
        assert pv_mesh.point_data["velocity"].shape == (10, 3)
        assert pv_mesh.cell_data["pressure"].shape == (2,)

    def test_mixed_rank_tensors(self):
        """Test mesh with both low-rank and high-rank tensors."""
        points = torch.rand(10, 3)
        cells = torch.tensor([[0, 1, 2], [2, 3, 4]], dtype=torch.long)
        mesh = Mesh(points=points, cells=cells)

        # Mix of ranks
        mesh.point_data["scalar"] = torch.rand(10)
        mesh.point_data["vector"] = torch.rand(10, 3)
        mesh.point_data["matrix"] = torch.rand(10, 3, 3)  # High-rank

        pv_mesh = to_pyvista(mesh)

        # Low-rank unchanged
        assert "scalar" in pv_mesh.point_data
        assert "vector" in pv_mesh.point_data
        assert pv_mesh.point_data["scalar"].shape == (10,)
        assert pv_mesh.point_data["vector"].shape == (10, 3)

        # High-rank flattened (key unchanged)
        assert "matrix" in pv_mesh.point_data
        assert pv_mesh.point_data["matrix"].shape == (10, 9)
