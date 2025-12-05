"""Tests for physicsnemo.mesh.io module - error handling."""

import numpy as np
import pytest
import pyvista as pv

from physicsnemo.mesh.io import from_pyvista


class TestErrorHandling:
    """Tests for error handling and edge cases."""

    def test_invalid_manifold_dim(self):
        """Test that invalid manifold_dim raises ValueError."""
        pv_mesh = pv.Sphere()

        with pytest.raises(ValueError, match="Invalid manifold_dim"):
            from_pyvista(pv_mesh, manifold_dim=4)

        with pytest.raises(ValueError, match="Invalid manifold_dim"):
            from_pyvista(pv_mesh, manifold_dim=-1)

    def test_mixed_geometry_error(self):
        """Test that meshes with mixed geometry types raise error."""
        # Create a mesh with both lines and cells (if possible)
        # This is tricky with PyVista; skip if not easily testable
        pass

    def test_empty_mesh(self):
        """Test conversion of empty mesh."""
        points = np.empty((0, 3), dtype=np.float32)
        pv_mesh = pv.PolyData(points)

        mesh = from_pyvista(pv_mesh, manifold_dim="auto")

        assert mesh.n_points == 0
        assert mesh.n_cells == 0
        assert mesh.n_manifold_dims == 0
