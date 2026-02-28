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

"""Tests for physicsnemo.mesh.io module - 2D mesh conversion."""

import numpy as np
import pytest
import torch

pv = pytest.importorskip("pyvista")

from physicsnemo.mesh.io.io_pyvista import from_pyvista  # noqa: E402


class TestFromPyvista2D:
    """Tests for converting 2D (surface) meshes."""

    def test_airplane_mesh_auto_detection(self):
        """Test automatic detection of 2D manifold from airplane mesh."""
        pv_mesh = pv.examples.load_airplane()

        mesh = from_pyvista(pv_mesh, manifold_dim="auto")

        assert mesh.n_manifold_dims == 2
        assert mesh.n_spatial_dims == 3
        assert mesh.cells.shape[1] == 3  # Triangular cells
        assert mesh.n_points == pv_mesh.n_points
        assert mesh.n_cells == pv_mesh.n_cells
        assert mesh.points.dtype == torch.float32
        assert mesh.cells.dtype == torch.long

    def test_airplane_mesh_explicit_dim(self):
        """Test explicit manifold_dim specification."""
        pv_mesh = pv.examples.load_airplane()

        mesh = from_pyvista(pv_mesh, manifold_dim=2)

        assert mesh.n_manifold_dims == 2
        assert mesh.n_spatial_dims == 3

    def test_sphere_mesh(self):
        """Test conversion of sphere mesh."""
        pv_mesh = pv.Sphere(radius=1.0, theta_resolution=10, phi_resolution=10)

        mesh = from_pyvista(pv_mesh, manifold_dim="auto")

        assert mesh.n_manifold_dims == 2
        assert mesh.cells.shape[1] == 3

    def test_automatic_triangulation(self):
        """Test that non-triangular meshes are automatically triangulated."""
        # Create a plane with quad cells
        pv_mesh = pv.Plane(i_resolution=2, j_resolution=2)
        assert not pv_mesh.is_all_triangles

        mesh = from_pyvista(pv_mesh, manifold_dim="auto")

        # Should be automatically triangulated
        assert mesh.cells.shape[1] == 3
        assert mesh.n_manifold_dims == 2


class TestFromPyvistaUnstructuredGrid2D:
    """Tests for converting 2D UnstructuredGrid meshes.

    pv.read() on .vtu files returns UnstructuredGrid even when all cells
    are 2D, so from_pyvista must handle this case correctly.
    """

    @staticmethod
    def _make_triangle_ugrid() -> pv.UnstructuredGrid:
        """Build an UnstructuredGrid with two triangles sharing an edge."""
        points = np.array(
            [[0, 0, 0], [1, 0, 0], [0.5, 1, 0], [1.5, 1, 0]],
            dtype=np.float32,
        )
        cells = np.array([3, 0, 1, 2, 3, 1, 3, 2])
        celltypes = np.array([pv.CellType.TRIANGLE, pv.CellType.TRIANGLE])
        return pv.UnstructuredGrid(cells, celltypes, points)

    @staticmethod
    def _make_quad_ugrid() -> pv.UnstructuredGrid:
        """Build an UnstructuredGrid with a single quad cell."""
        points = np.array(
            [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
            dtype=np.float32,
        )
        cells = np.array([4, 0, 1, 2, 3])
        celltypes = np.array([pv.CellType.QUAD])
        return pv.UnstructuredGrid(cells, celltypes, points)

    def test_all_triangles_auto_detection(self):
        """Test auto-detection for an UnstructuredGrid containing only triangles."""
        pv_mesh = self._make_triangle_ugrid()

        mesh = from_pyvista(pv_mesh, manifold_dim="auto")

        assert mesh.n_manifold_dims == 2
        assert mesh.n_points == 4
        assert mesh.n_cells == 2
        assert mesh.cells.shape == (2, 3)
        expected = torch.tensor([[0, 1, 2], [1, 3, 2]], dtype=torch.long)
        assert torch.equal(mesh.cells, expected)

    def test_all_triangles_explicit_dim(self):
        """Test explicit manifold_dim=2 with an all-triangle UnstructuredGrid."""
        pv_mesh = self._make_triangle_ugrid()

        mesh = from_pyvista(pv_mesh, manifold_dim=2)

        assert mesh.n_manifold_dims == 2
        assert mesh.cells.shape == (2, 3)

    def test_quad_auto_triangulation(self):
        """Test that quad cells in an UnstructuredGrid are triangulated."""
        pv_mesh = self._make_quad_ugrid()
        assert set(pv_mesh.cells_dict.keys()) == {pv.CellType.QUAD}

        mesh = from_pyvista(pv_mesh, manifold_dim="auto")

        assert mesh.n_manifold_dims == 2
        assert mesh.cells.shape[1] == 3
        assert mesh.n_cells >= 2  # one quad -> at least 2 triangles

    def test_point_data_preserved(self):
        """Test that point data survives conversion from UnstructuredGrid."""
        pv_mesh = self._make_triangle_ugrid()
        pv_mesh.point_data["temperature"] = np.array(
            [300.0, 310.0, 305.0, 315.0], dtype=np.float32
        )

        mesh = from_pyvista(pv_mesh, manifold_dim="auto")

        assert "temperature" in mesh.point_data
        expected = torch.tensor([300.0, 310.0, 305.0, 315.0], dtype=torch.float32)
        assert torch.allclose(mesh.point_data["temperature"], expected)
