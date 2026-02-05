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
