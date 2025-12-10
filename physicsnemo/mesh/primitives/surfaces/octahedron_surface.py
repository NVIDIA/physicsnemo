"""Regular octahedron surface in 3D space.

Dimensional: 2D manifold in 3D space (closed, no boundary).
"""

import torch

from physicsnemo.mesh.mesh import Mesh


def load(size: float = 1.0, device: str = "cpu") -> Mesh:
    """Create a regular octahedron surface in 3D space.

    Parameters
    ----------
    size : float
        Distance from center to vertex.
    device : str
        Compute device ('cpu' or 'cuda').

    Returns
    -------
    Mesh
        Mesh with n_manifold_dims=2, n_spatial_dims=3.
    """
    # 6 vertices (axis-aligned)
    vertices = [
        [size, 0, 0],
        [-size, 0, 0],
        [0, size, 0],
        [0, -size, 0],
        [0, 0, size],
        [0, 0, -size],
    ]

    # 8 triangular faces
    faces = [
        [0, 2, 4],
        [2, 1, 4],
        [1, 3, 4],
        [3, 0, 4],
        [0, 5, 2],
        [2, 5, 1],
        [1, 5, 3],
        [3, 5, 0],
    ]

    points = torch.tensor(vertices, dtype=torch.float32, device=device)
    cells = torch.tensor(faces, dtype=torch.int64, device=device)

    return Mesh(points=points, cells=cells)
