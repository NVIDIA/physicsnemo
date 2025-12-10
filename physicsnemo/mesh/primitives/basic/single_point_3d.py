"""Single point in 3D space.

Dimensional: 0D manifold in 3D space.
"""

import torch

from physicsnemo.mesh.mesh import Mesh


def load(device: str = "cpu") -> Mesh:
    """Create a mesh with a single point in 3D space.

    Parameters
    ----------
    device : str
        Compute device ('cpu' or 'cuda').

    Returns
    -------
    Mesh
        Mesh with n_manifold_dims=0, n_spatial_dims=3, n_cells=1.
    """
    points = torch.tensor([[0.5, 0.5, 0.5]], dtype=torch.float32, device=device)
    cells = torch.tensor([[0]], dtype=torch.int64, device=device)
    return Mesh(points=points, cells=cells)
