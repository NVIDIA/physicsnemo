"""Tetrahedral beam volume from PyVista.

Dimensional: 3D manifold in 3D space.
"""

import pyvista as pv

from physicsnemo.mesh.io import from_pyvista
from physicsnemo.mesh.mesh import Mesh


def load(device: str = "cpu") -> Mesh:
    """Load tetrahedral beam volume from PyVista examples.

    This is a classic test case for 3D finite element analysis.

    Parameters
    ----------
    device : str
        Compute device ('cpu' or 'cuda').

    Returns
    -------
    Mesh
        Mesh with n_manifold_dims=3, n_spatial_dims=3.
    """
    pv_mesh = pv.examples.load_tetbeam()
    mesh = from_pyvista(pv_mesh, manifold_dim=3)

    # Move to specified device
    if device != str(mesh.points.device):
        mesh = mesh.to(device)

    return mesh
