"""Two tetrahedra sharing a triangular face in 3D space.

Dimensional: 3D manifold in 3D space.
"""

import torch

from physicsnemo.mesh.mesh import Mesh


def load(device: str = "cpu") -> Mesh:
    """Create a mesh with two tetrahedra in 3D space.

    The tetrahedra share a common triangular face, forming a simple
    volumetric region.

    Parameters
    ----------
    device : str
        Compute device ('cpu' or 'cuda').

    Returns
    -------
    Mesh
        Mesh with n_manifold_dims=3, n_spatial_dims=3, n_cells=2.
    """
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=torch.float32,
        device=device,
    )
    cells = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.int64, device=device)
    return Mesh(points=points, cells=cells)
