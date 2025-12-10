"""Unit square triangulated in 2D space.

Dimensional: 2D manifold in 2D space.
"""

import torch

from physicsnemo.mesh.mesh import Mesh


def load(n_subdivisions: int = 1, device: str = "cpu") -> Mesh:
    """Create a triangulated unit square in 2D space.

    Parameters
    ----------
    n_subdivisions : int
        Number of subdivision levels (0 = 2 triangles).
    device : str
        Compute device ('cpu' or 'cuda').

    Returns
    -------
    Mesh
        Mesh with n_manifold_dims=2, n_spatial_dims=2.
    """
    if n_subdivisions < 0:
        raise ValueError(f"n_subdivisions must be non-negative, got {n_subdivisions=}")

    n = 2**n_subdivisions + 1

    # Create grid of points
    x = torch.linspace(0.0, 1.0, n, device=device)
    y = torch.linspace(0.0, 1.0, n, device=device)
    xx, yy = torch.meshgrid(x, y, indexing="ij")

    points = torch.stack([xx.flatten(), yy.flatten()], dim=1)

    # Create triangular cells
    cells = []
    for i in range(n - 1):
        for j in range(n - 1):
            idx = i * n + j
            # Two triangles per quad
            cells.append([idx, idx + 1, idx + n])
            cells.append([idx + 1, idx + n + 1, idx + n])

    cells = torch.tensor(cells, dtype=torch.int64, device=device)
    return Mesh(points=points, cells=cells)
