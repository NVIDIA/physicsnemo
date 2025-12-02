"""Hexahedral beam volume mesh from PyVista examples (tessellated to tets).

Dimensional: 3D manifold in 3D space.
"""

import pyvista as pv

from physicsnemo.mesh.io import from_pyvista
from physicsnemo.mesh.mesh import Mesh


def load(device: str = "cpu") -> Mesh:
    """Load hexahedral beam mesh from PyVista examples.

    The hexahedral cells are automatically tessellated into
    tetrahedra during conversion.

    Args:
        device: Compute device ('cpu' or 'cuda')

    Returns:
        Mesh with n_manifold_dims=3, n_spatial_dims=3
    """
    pv_mesh = pv.examples.load_hexbeam()
    mesh = from_pyvista(pv_mesh, manifold_dim=3)

    # Move to specified device
    if device != str(mesh.points.device):
        mesh = mesh.to(device)

    return mesh
