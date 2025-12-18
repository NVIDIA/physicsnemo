"""Cube surface triangulated in 3D space.

Dimensional: 2D manifold in 3D space (closed, no boundary).
"""

import pyvista as pv
import torch

from physicsnemo.mesh.io import from_pyvista
from physicsnemo.mesh.mesh import Mesh


def load(size: float = 1.0, device: torch.device | str = "cpu") -> Mesh:
    """Create a cube surface triangulated in 3D space.

    Parameters
    ----------
    size : float
        Side length of the cube.
    device : str
        Compute device ('cpu' or 'cuda').

    Returns
    -------
    Mesh
        Mesh with n_manifold_dims=2, n_spatial_dims=3.
    """
    # Create cube with PyVista (automatically triangulated)
    pv_cube = pv.Cube(x_length=size, y_length=size, z_length=size)

    mesh = from_pyvista(pv_cube, manifold_dim=2)

    # Move to specified device
    if device != str(mesh.points.device):
        mesh = mesh.to(device)

    return mesh
