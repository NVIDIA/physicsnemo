"""Cow surface mesh from PyVista examples.

Dimensional: 2D manifold in 3D space.
"""

import pyvista as pv

from physicsnemo.mesh.io import from_pyvista
from physicsnemo.mesh.mesh import Mesh


def load(device: str = "cpu") -> Mesh:
    """Load cow surface mesh from PyVista examples.

    The cow mesh is a classic test case that originally contains
    both triangular and quadrilateral cells, automatically
    triangulated during conversion.

    PyVista caches the downloaded file automatically.

    Args:
        device: Compute device ('cpu' or 'cuda')

    Returns:
        Mesh with n_manifold_dims=2, n_spatial_dims=3
    """
    pv_mesh = pv.examples.download_cow()
    mesh = from_pyvista(pv_mesh, manifold_dim=2)

    # Move to specified device
    if device != str(mesh.points.device):
        mesh = mesh.to(device)

    return mesh
