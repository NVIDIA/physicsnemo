"""Rectangle triangulated in 2D space.

Dimensional: 2D manifold in 2D space.
"""

import torch

from physicsnemo.mesh.mesh import Mesh


def load(
    width: float = 2.0,
    height: float = 1.0,
    n_x: int = 10,
    n_y: int = 5,
    device: str = "cpu",
) -> Mesh:
    """Create a triangulated rectangle in 2D space.

    Args:
        width: Width of the rectangle
        height: Height of the rectangle
        n_x: Number of points in x-direction
        n_y: Number of points in y-direction
        device: Compute device ('cpu' or 'cuda')

    Returns:
        Mesh with n_manifold_dims=2, n_spatial_dims=2
    """
    if n_x < 2:
        raise ValueError(f"n_x must be at least 2, got {n_x=}")
    if n_y < 2:
        raise ValueError(f"n_y must be at least 2, got {n_y=}")

    # Create grid of points
    x = torch.linspace(0.0, width, n_x, device=device)
    y = torch.linspace(0.0, height, n_y, device=device)
    xx, yy = torch.meshgrid(x, y, indexing="ij")

    points = torch.stack([xx.flatten(), yy.flatten()], dim=1)

    # Create triangular cells
    cells = []
    for i in range(n_x - 1):
        for j in range(n_y - 1):
            idx = i * n_y + j
            # Two triangles per quad
            cells.append([idx, idx + 1, idx + n_y])
            cells.append([idx + 1, idx + n_y + 1, idx + n_y])

    cells = torch.tensor(cells, dtype=torch.int64, device=device)
    return Mesh(points=points, cells=cells)
