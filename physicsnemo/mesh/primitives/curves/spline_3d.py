"""3D spline curve from PyVista.

Dimensional: 1D manifold in 3D space.
"""

import pyvista as pv

from physicsnemo.mesh.io import from_pyvista
from physicsnemo.mesh.mesh import Mesh


def load(device: torch.device | str = "cpu") -> Mesh:
    """Load a 3D spline curve from PyVista examples.

    This loads the built-in PyVista spline example, which is a smooth
    3D curve useful for testing 1D operations in 3D space.

    Parameters
    ----------
    device : str
        Compute device ('cpu' or 'cuda').

    Returns
    -------
    Mesh
        Mesh with n_manifold_dims=1, n_spatial_dims=3.
    """
    pv_mesh = pv.examples.load_spline()
    mesh = from_pyvista(pv_mesh, manifold_dim=1)

    # Move to specified device
    if device != str(mesh.points.device):
        mesh = mesh.to(device)

    return mesh
