"""Tetrahedral sphere volume mesh in 3D space.

Dimensional: 3D manifold in 3D space.
"""

import pyvista as pv

from physicsnemo.mesh.io import from_pyvista
from physicsnemo.mesh.mesh import Mesh


def load(radius: float = 1.0, resolution: int = 20, device: str = "cpu") -> Mesh:
    """Create a tetrahedral volume mesh of a sphere.

    Parameters
    ----------
    radius : float
        Radius of the sphere.
    resolution : int
        Resolution of the initial surface mesh.
    device : str
        Compute device ('cpu' or 'cuda').

    Returns
    -------
    Mesh
        Mesh with n_manifold_dims=3, n_spatial_dims=3.
    """
    # Create surface sphere
    surface = pv.Sphere(
        radius=radius, theta_resolution=resolution, phi_resolution=resolution
    )

    # Fill with tetrahedra using Delaunay 3D
    volume = surface.delaunay_3d()

    mesh = from_pyvista(volume, manifold_dim=3)

    # Move to specified device
    if device != str(mesh.points.device):
        mesh = mesh.to(device)

    return mesh
