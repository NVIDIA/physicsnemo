"""Three connected edges forming a polyline in 3D space.

Dimensional: 1D manifold in 3D space.
"""

import torch

from physicsnemo.mesh.mesh import Mesh


def load(device: torch.device | str = "cpu") -> Mesh:
    """Create a polyline with three edges in 3D space.

    Parameters
    ----------
    device : str
        Compute device ('cpu' or 'cuda').

    Returns
    -------
    Mesh
        Mesh with n_manifold_dims=1, n_spatial_dims=3, n_cells=3.
    """
    points = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 1.0]],
        dtype=torch.float32,
        device=device,
    )
    cells = torch.tensor([[0, 1], [1, 2], [2, 3]], dtype=torch.int64, device=device)
    return Mesh(points=points, cells=cells)
