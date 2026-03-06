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

"""Tests for physicsnemo.mesh.io module - error handling.

Tests validate error paths in from_pyvista and to_pyvista functions, including:
- Invalid manifold_dim values
- Mixed geometry types (lines and faces)
- Missing cells_dict for 3D meshes
- Non-tetra cells after tessellation
- Unsupported manifold dimensions in to_pyvista
"""

import numpy as np
import pytest
import torch

from physicsnemo.mesh import Mesh

pv = pytest.importorskip("pyvista")

from physicsnemo.mesh.io.io_pyvista import from_pyvista, to_pyvista  # noqa: E402


class TestFromPyvistaInvalidManifoldDim:
    """Tests for invalid manifold_dim handling."""

    def test_invalid_manifold_dim_4(self):
        """Test that manifold_dim=4 raises ValueError."""
        pv_mesh = pv.Sphere()

        with pytest.raises(ValueError, match="Invalid manifold_dim"):
            from_pyvista(pv_mesh, manifold_dim=4)

    def test_invalid_manifold_dim_negative(self):
        """Test that negative manifold_dim raises ValueError."""
        pv_mesh = pv.Sphere()

        with pytest.raises(ValueError, match="Invalid manifold_dim"):
            from_pyvista(pv_mesh, manifold_dim=-1)

    def test_invalid_manifold_dim_5(self):
        """Test that manifold_dim=5 raises ValueError."""
        pv_mesh = pv.Sphere()

        with pytest.raises(ValueError, match="Invalid manifold_dim"):
            from_pyvista(pv_mesh, manifold_dim=5)

    def test_invalid_manifold_dim_100(self):
        """Test that manifold_dim=100 raises ValueError."""
        pv_mesh = pv.Sphere()

        with pytest.raises(ValueError, match="Invalid manifold_dim"):
            from_pyvista(pv_mesh, manifold_dim=100)


class TestFromPyvistaMixedGeometry:
    """Tests for mixed geometry type detection."""

    def test_mixed_lines_and_faces_raises(self):
        """Test that mesh with both lines and faces raises ValueError."""
        # Create a PolyData with both lines and faces
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 1.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )

        # Create faces (triangle)
        faces = np.array([3, 0, 1, 2])  # One triangle

        # Create lines
        lines = np.array([2, 3, 4])  # One line segment

        pv_mesh = pv.PolyData(points, faces=faces, lines=lines)

        # Should raise because we have both lines and faces
        with pytest.raises(
            ValueError, match="Cannot automatically determine manifold dimension"
        ):
            from_pyvista(pv_mesh, manifold_dim="auto")

    def test_mixed_geometry_explicit_dim_works(self):
        """Test that explicit manifold_dim bypasses mixed geometry check."""
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 1.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )

        faces = np.array([3, 0, 1, 2])
        lines = np.array([2, 3, 4])

        pv_mesh = pv.PolyData(points, faces=faces, lines=lines)

        # With explicit manifold_dim=2, should work (uses faces)
        mesh = from_pyvista(pv_mesh, manifold_dim=2)

        assert mesh.n_manifold_dims == 2


class TestFromPyvista3DErrors:
    """Tests for 3D mesh conversion error handling."""

    def test_polydata_3d_no_cells_dict_raises(self):
        """Test that PolyData without cells_dict for 3D raises ValueError."""
        # Create a simple PolyData (surface mesh)
        pv_mesh = pv.Sphere()

        # Trying to convert as 3D should fail because PolyData doesn't have cells_dict
        # (it's a surface, not a volume)
        with pytest.raises(ValueError, match="cells_dict"):
            from_pyvista(pv_mesh, manifold_dim=3)


class TestFromPyvistaEmptyMeshes:
    """Tests for empty mesh handling."""

    def test_empty_points_mesh(self):
        """Test conversion of mesh with no points."""
        points = np.empty((0, 3), dtype=np.float32)
        pv_mesh = pv.PolyData(points)

        mesh = from_pyvista(pv_mesh, manifold_dim="auto")

        assert mesh.n_points == 0
        assert mesh.n_cells == 0
        assert mesh.n_manifold_dims == 0

    def test_points_only_mesh(self):
        """Test conversion of point cloud (no cells)."""
        points = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32
        )
        pv_mesh = pv.PointSet(points)

        mesh = from_pyvista(pv_mesh, manifold_dim="auto")

        assert mesh.n_points == 3
        assert mesh.n_cells == 0
        assert mesh.n_manifold_dims == 0


class TestFromPyvistaAutoDetection:
    """Tests for automatic manifold dimension detection."""

    def test_auto_detect_surface_mesh(self):
        """Test auto detection for surface mesh (2D)."""
        pv_mesh = pv.Sphere()

        mesh = from_pyvista(pv_mesh, manifold_dim="auto")

        assert mesh.n_manifold_dims == 2

    def test_auto_detect_line_mesh(self):
        """Test auto detection for line mesh (1D)."""
        points = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32
        )
        lines = np.array([2, 0, 1, 2, 1, 2])  # Two line segments
        pv_mesh = pv.PolyData(points, lines=lines)

        mesh = from_pyvista(pv_mesh, manifold_dim="auto")

        assert mesh.n_manifold_dims == 1

    def test_auto_detect_volume_mesh(self):
        """Test auto detection for volume mesh (3D)."""
        # Create a simple tetrahedron
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 1.0, 0.0],
                [0.5, 0.5, 1.0],
            ],
            dtype=np.float32,
        )
        cells = np.array([4, 0, 1, 2, 3])  # One tetrahedron
        celltypes = np.array([pv.CellType.TETRA])
        pv_mesh = pv.UnstructuredGrid(cells, celltypes, points)

        mesh = from_pyvista(pv_mesh, manifold_dim="auto")

        assert mesh.n_manifold_dims == 3


class TestToPyvistaErrors:
    """Tests for error handling in to_pyvista."""

    def test_unsupported_manifold_dims_raises(self):
        """Test that unsupported manifold dimensions raise ValueError."""
        # Create a mesh with 4 manifold dims (not supported by PyVista)
        points = torch.randn(10, 5)  # 5D spatial
        cells = torch.randint(0, 10, (5, 5))  # 4-manifold (5 vertices per cell)
        mesh = Mesh(points=points, cells=cells)

        assert mesh.n_manifold_dims == 4

        with pytest.raises(ValueError, match="Unsupported"):
            to_pyvista(mesh)


class TestToPyvistaValidCases:
    """Tests for valid to_pyvista conversions."""

    def test_to_pyvista_0d(self):
        """Test conversion of 0D mesh (point cloud)."""
        points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        cells = torch.zeros((0, 1), dtype=torch.long)
        mesh = Mesh(points=points, cells=cells)

        pv_mesh = to_pyvista(mesh)

        assert isinstance(pv_mesh, pv.PointSet)
        assert pv_mesh.n_points == 3

    def test_to_pyvista_1d(self):
        """Test conversion of 1D mesh (lines)."""
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        cells = torch.tensor([[0, 1], [1, 2]])
        mesh = Mesh(points=points, cells=cells)

        pv_mesh = to_pyvista(mesh)

        assert isinstance(pv_mesh, pv.PolyData)
        assert pv_mesh.n_points == 3

    def test_to_pyvista_2d(self):
        """Test conversion of 2D mesh (triangles)."""
        points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]])
        cells = torch.tensor([[0, 1, 2]])
        mesh = Mesh(points=points, cells=cells)

        pv_mesh = to_pyvista(mesh)

        assert isinstance(pv_mesh, pv.PolyData)
        assert pv_mesh.n_cells == 1

    def test_to_pyvista_3d(self):
        """Test conversion of 3D mesh (tetrahedra)."""
        points = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 1.0, 0.0],
                [0.5, 0.5, 1.0],
            ]
        )
        cells = torch.tensor([[0, 1, 2, 3]])
        mesh = Mesh(points=points, cells=cells)

        pv_mesh = to_pyvista(mesh)

        assert isinstance(pv_mesh, pv.UnstructuredGrid)
        assert pv_mesh.n_cells == 1


class TestToPyvistaEmptyMeshes:
    """Tests for empty mesh handling in to_pyvista."""

    def test_to_pyvista_empty_0d(self):
        """Test conversion of empty 0D mesh."""
        points = torch.zeros((0, 3))
        cells = torch.zeros((0, 1), dtype=torch.long)
        mesh = Mesh(points=points, cells=cells)

        pv_mesh = to_pyvista(mesh)

        assert pv_mesh.n_points == 0

    def test_to_pyvista_empty_2d(self):
        """Test conversion of mesh with points but no cells."""
        points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        cells = torch.zeros((0, 3), dtype=torch.long)
        mesh = Mesh(points=points, cells=cells)

        pv_mesh = to_pyvista(mesh)

        assert pv_mesh.n_points == 2
        # PyVista PolyData with empty faces may still have cells (as vertices)
        # The important thing is that faces are empty
        assert isinstance(pv_mesh, pv.PolyData)

    def test_to_pyvista_empty_3d(self):
        """Test conversion of empty 3D mesh."""
        points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        cells = torch.zeros((0, 4), dtype=torch.long)
        mesh = Mesh(points=points, cells=cells)

        pv_mesh = to_pyvista(mesh)

        assert pv_mesh.n_points == 2
        assert pv_mesh.n_cells == 0


class TestToPyvistaSpatialPadding:
    """Tests for 2D to 3D spatial dimension padding."""

    def test_2d_points_padded_to_3d(self):
        """Test that 2D points are padded to 3D for PyVista."""
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]])
        cells = torch.tensor([[0, 1, 2]])
        mesh = Mesh(points=points, cells=cells)

        pv_mesh = to_pyvista(mesh)

        # PyVista always has 3D points
        assert pv_mesh.points.shape[1] == 3
        # Z coordinates should be 0
        assert np.allclose(pv_mesh.points[:, 2], 0.0)

    def test_1d_points_padded_to_3d(self):
        """Test that 1D points are padded to 3D for PyVista."""
        points = torch.tensor([[0.0], [1.0], [2.0]])
        cells = torch.tensor([[0, 1], [1, 2]])
        mesh = Mesh(points=points, cells=cells)

        pv_mesh = to_pyvista(mesh)

        # PyVista always has 3D points
        assert pv_mesh.points.shape[1] == 3
        # Y and Z coordinates should be 0
        assert np.allclose(pv_mesh.points[:, 1], 0.0)
        assert np.allclose(pv_mesh.points[:, 2], 0.0)
