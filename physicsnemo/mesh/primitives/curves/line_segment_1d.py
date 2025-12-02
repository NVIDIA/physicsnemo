"""Single line segment in 1D space.

Dimensional: 1D manifold in 1D space.
"""

import torch

from physicsnemo.mesh.mesh import Mesh


def load(length: float = 1.0, n_points: int = 2, device: str = "cpu") -> Mesh:
    """Create a line segment in 1D space.

    Args:
        length: Length of the line segment
        n_points: Number of points (minimum 2)
        device: Compute device ('cpu' or 'cuda')

    Returns:
        Mesh with n_manifold_dims=1, n_spatial_dims=1, n_cells=n_points-1
    """
    if n_points < 2:
        raise ValueError(f"n_points must be at least 2, got {n_points=}")

    # Create evenly spaced points along the line
    points = torch.linspace(0.0, length, n_points, device=device).unsqueeze(1)

    # Create edge cells connecting consecutive points
    cells = torch.stack(
        [
            torch.arange(n_points - 1, device=device),
            torch.arange(1, n_points, device=device),
        ],
        dim=1,
    )

    return Mesh(points=points, cells=cells)
