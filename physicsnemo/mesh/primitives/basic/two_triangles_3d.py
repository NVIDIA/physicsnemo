"""Two triangles sharing an edge in 3D space.

Dimensional: 2D manifold in 3D space.
"""

import torch

from physicsnemo.mesh.mesh import Mesh


def load(device: str = "cpu") -> Mesh:
    """Create a mesh with two triangles in 3D space.

    The triangles share a common edge, forming a simple surface patch.

    Parameters
    ----------
    device : str
        Compute device ('cpu' or 'cuda').

    Returns
    -------
    Mesh
        Mesh with n_manifold_dims=2, n_spatial_dims=3, n_cells=2.
    """
    points = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0], [1.5, 0.5, 0.5]],
        dtype=torch.float32,
        device=device,
    )
    cells = torch.tensor([[0, 1, 2], [1, 3, 2]], dtype=torch.int64, device=device)
    return Mesh(points=points, cells=cells)
