"""Tetrahedral cube volume mesh in 3D space.

Dimensional: 3D manifold in 3D space.
"""

import pyvista as pv
import torch

from physicsnemo.mesh.io import from_pyvista
from physicsnemo.mesh.mesh import Mesh


def load(size: float = 1.0, n_subdivisions: int = 5, device: torch.device | str = "cpu") -> Mesh:
    """Create a tetrahedral volume mesh of a cube.

    Parameters
    ----------
    size : float
        Side length of the cube.
    n_subdivisions : int
        Number of subdivisions per edge.
    device : str
        Compute device ('cpu' or 'cuda').

    Returns
    -------
    Mesh
        Mesh with n_manifold_dims=3, n_spatial_dims=3.
    """
    if n_subdivisions < 1:
        raise ValueError(f"n_subdivisions must be at least 1, got {n_subdivisions=}")

    # Create a structured grid
    n = n_subdivisions + 1

    # Create PyVista structured grid
    grid = pv.ImageData(
        dimensions=(n, n, n),
        spacing=(size / n_subdivisions, size / n_subdivisions, size / n_subdivisions),
        origin=(-size / 2, -size / 2, -size / 2),
    )

    # Tessellate to tetrahedra
    tet_grid = grid.tessellate()

    mesh = from_pyvista(tet_grid, manifold_dim=3)

    # Move to specified device
    if device != str(mesh.points.device):
        mesh = mesh.to(device)

    return mesh
