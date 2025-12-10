"""Three points in 2D space.

Dimensional: 0D manifold in 2D space.
"""

import torch

from physicsnemo.mesh.mesh import Mesh


def load(device: str = "cpu") -> Mesh:
    """Create a mesh with three points in 2D space.

    Parameters
    ----------
    device : str
        Compute device ('cpu' or 'cuda').

    Returns
    -------
    Mesh
        Mesh with n_manifold_dims=0, n_spatial_dims=2, n_cells=3.
    """
    points = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]], dtype=torch.float32, device=device
    )
    cells = torch.tensor([[0], [1], [2]], dtype=torch.int64, device=device)
    return Mesh(points=points, cells=cells)
