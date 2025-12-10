"""Filled disk (circle) triangulated in 2D space.

Dimensional: 2D manifold in 2D space.
"""

import pyvista as pv

from physicsnemo.mesh.io import from_pyvista
from physicsnemo.mesh.mesh import Mesh


def load(
    radius: float = 1.0,
    n_radial: int = 10,
    n_angular: int = 32,
    device: str = "cpu",
) -> Mesh:
    """Create a filled disk (circle) triangulated in 2D space.

    Parameters
    ----------
    radius : float
        Radius of the disk.
    n_radial : int
        Number of points in radial direction.
    n_angular : int
        Number of points around the circumference.
    device : str
        Compute device ('cpu' or 'cuda').

    Returns
    -------
    Mesh
        Mesh with n_manifold_dims=2, n_spatial_dims=2.
    """
    # Use PyVista to create a disk
    pv_disk = pv.Disc(
        center=(0.0, 0.0, 0.0),
        inner=0.0,
        outer=radius,
        r_res=n_radial,
        c_res=n_angular,
    )

    # Extract only x and y coordinates (discard z)
    mesh = from_pyvista(pv_disk, manifold_dim=2)

    # Project to 2D by removing z coordinate
    points_2d = mesh.points[:, :2]
    mesh = Mesh(points=points_2d, cells=mesh.cells)

    # Move to specified device
    if device != str(mesh.points.device):
        mesh = mesh.to(device)

    return mesh
