"""Tests for physicsnemo.mesh.io module - 3D mesh conversion."""

import numpy as np
import pyvista as pv
import torch

from physicsnemo.mesh.io import from_pyvista


class TestFromPyvista3D:
    """Tests for converting 3D (volume) meshes."""

    def test_tetbeam_mesh_auto_detection(self):
        """Test automatic detection of 3D manifold from tetbeam mesh."""
        pv_mesh = pv.examples.load_tetbeam()

        # Verify it's all tetrahedral cells
        assert list(pv_mesh.cells_dict.keys()) == [pv.CellType.TETRA]

        mesh = from_pyvista(pv_mesh, manifold_dim="auto")

        assert mesh.n_manifold_dims == 3
        assert mesh.n_spatial_dims == 3
        assert mesh.cells.shape[1] == 4  # Tetrahedral cells
        assert mesh.n_points == pv_mesh.n_points
        assert mesh.n_cells == pv_mesh.n_cells

    def test_tetbeam_mesh_explicit_dim(self):
        """Test explicit manifold_dim specification for 3D mesh."""
        pv_mesh = pv.examples.load_tetbeam()

        mesh = from_pyvista(pv_mesh, manifold_dim=3)

        assert mesh.n_manifold_dims == 3
        assert mesh.cells.shape[1] == 4

    def test_hexbeam_mesh_tessellation(self):
        """Test automatic tessellation of hexahedral mesh to tetrahedral.

        The hexbeam mesh contains hexahedral cells which must be converted
        to tetrahedral cells for our simplex-based mesh representation.
        """
        pv_mesh = pv.examples.load_hexbeam()

        # Verify it contains hexahedral cells (not tetrahedral)
        assert pv.CellType.HEXAHEDRON in pv_mesh.cells_dict
        assert pv.CellType.TETRA not in pv_mesh.cells_dict
        original_n_points = pv_mesh.n_points

        # Convert - should automatically tessellate
        mesh = from_pyvista(pv_mesh, manifold_dim="auto")

        assert mesh.n_manifold_dims == 3
        assert mesh.n_spatial_dims == 3
        assert mesh.cells.shape[1] == 4  # Tetrahedral cells after tessellation
        # Tessellation may add points at cell centers
        assert mesh.n_points >= original_n_points
        # Each hexahedron is tessellated into at least 5 tetrahedra
        assert mesh.n_cells >= 5 * pv_mesh.n_cells

    def test_simple_tetrahedron(self):
        """Test conversion of a single tetrahedron."""
        points = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )
        cells = np.array([4, 0, 1, 2, 3])
        celltypes = np.array([pv.CellType.TETRA])

        pv_mesh = pv.UnstructuredGrid(cells, celltypes, points)
        mesh = from_pyvista(pv_mesh, manifold_dim="auto")

        assert mesh.n_manifold_dims == 3
        assert mesh.n_points == 4
        assert mesh.n_cells == 1
        assert mesh.cells.shape == (1, 4)

        # Verify the face connectivity is correct
        expected_cells = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
        assert torch.equal(mesh.cells, expected_cells)
