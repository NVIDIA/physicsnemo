"""Tests for physicsnemo.mesh.io module - 0D mesh conversion."""

import numpy as np
import pyvista as pv
import torch

from physicsnemo.mesh.io import from_pyvista


class TestFromPyvista0D:
    """Tests for converting 0D (point cloud) meshes."""

    def test_pointset_auto_detection(self):
        """Test automatic detection of 0D manifold from PointSet."""
        np.random.seed(0)
        points = np.random.rand(100, 3).astype(np.float32)
        pv_mesh = pv.PointSet(points)

        # Verify it's just points (no connectivity)
        assert pv_mesh.n_points == 100

        mesh = from_pyvista(pv_mesh, manifold_dim="auto")

        assert mesh.n_manifold_dims == 0
        assert mesh.n_spatial_dims == 3
        assert mesh.n_points == 100
        assert mesh.n_cells == 0
        assert mesh.cells.shape == (0, 1)

        # Verify points are preserved correctly
        assert torch.allclose(mesh.points, torch.from_numpy(points).float(), atol=1e-6)

    def test_pointset_explicit_dim(self):
        """Test explicit manifold_dim specification for point cloud."""
        np.random.seed(0)
        points = np.random.rand(50, 3).astype(np.float32)
        pv_mesh = pv.PointSet(points)

        mesh = from_pyvista(pv_mesh, manifold_dim=0)

        assert mesh.n_manifold_dims == 0
        assert mesh.n_points == 50
        assert mesh.cells.shape == (0, 1)

    def test_polydata_points_only(self):
        """Test PolyData with only points (no lines or cells).

        PolyData can represent point clouds using vertex cells.
        """
        np.random.seed(0)
        points = np.random.rand(25, 3).astype(np.float32)
        pv_mesh = pv.PolyData(points)

        # Verify it has vertex cells but no lines or polygon cells
        assert pv_mesh.n_verts == 25
        assert pv_mesh.n_lines == 0

        mesh = from_pyvista(pv_mesh, manifold_dim="auto")

        assert mesh.n_manifold_dims == 0
        assert mesh.n_points == 25
        assert mesh.n_cells == 0
        assert mesh.cells.shape == (0, 1)
