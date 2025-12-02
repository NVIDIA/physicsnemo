"""Tests for physicsnemo.mesh.io module - mesh equivalence."""

import pyvista as pv
import torch

from physicsnemo.mesh.io import from_pyvista
from physicsnemo.mesh.mesh import Mesh


class TestMeshEquivalence:
    """Tests that converted meshes are equivalent to direct construction."""

    def test_airplane_equivalence(self):
        """Test that from_pyvista produces same result as direct construction."""
        pv_mesh = pv.examples.load_airplane()

        # Using from_pyvista
        mesh_from_pv = from_pyvista(pv_mesh)

        # Direct construction (as in primitives.py)
        mesh_direct = Mesh(
            points=pv_mesh.points,
            cells=pv_mesh.regular_faces,
            point_data=pv_mesh.point_data,
            cell_data=pv_mesh.cell_data,
            global_data=pv_mesh.field_data,
        )

        assert torch.equal(mesh_from_pv.points, mesh_direct.points)
        assert torch.equal(mesh_from_pv.cells, mesh_direct.cells)

    def test_tetbeam_equivalence(self):
        """Test that from_pyvista produces same result as direct construction for tetbeam."""
        pv_mesh = pv.examples.load_tetbeam()

        # Using from_pyvista
        mesh_from_pv = from_pyvista(pv_mesh)

        # Direct construction (as in primitives.py)
        mesh_direct = Mesh(
            points=pv_mesh.points,
            cells=pv_mesh.cells_dict[pv.CellType.TETRA],
            point_data=pv_mesh.point_data,
            cell_data=pv_mesh.cell_data,
            global_data=pv_mesh.field_data,
        )

        assert torch.equal(mesh_from_pv.points, mesh_direct.points)
        assert torch.equal(mesh_from_pv.cells, mesh_direct.cells)
