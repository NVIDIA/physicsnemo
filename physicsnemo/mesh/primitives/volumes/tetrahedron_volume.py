"""Single tetrahedron volume in 3D space.

Dimensional: 3D manifold in 3D space.
"""

import torch

from physicsnemo.mesh.mesh import Mesh


def load(side_length: float = 1.0, device: str = "cpu") -> Mesh:
    """Create a single regular tetrahedron volume.

    Args:
        side_length: Length of each edge
        device: Compute device ('cpu' or 'cuda')

    Returns:
        Mesh with n_manifold_dims=3, n_spatial_dims=3, n_cells=1
    """
    # Regular tetrahedron vertices
    vertices = [
        [0.0, 0.0, 0.0],
        [side_length, 0.0, 0.0],
        [side_length / 2, side_length * (3**0.5) / 2, 0.0],
        [
            side_length / 2,
            side_length * (3**0.5) / 6,
            side_length * ((2 / 3) ** 0.5),
        ],
    ]

    points = torch.tensor(vertices, dtype=torch.float32, device=device)
    cells = torch.tensor([[0, 1, 2, 3]], dtype=torch.int64, device=device)

    return Mesh(points=points, cells=cells)
