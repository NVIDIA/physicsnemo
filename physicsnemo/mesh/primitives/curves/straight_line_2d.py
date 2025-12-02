"""Straight line segment in 2D space.

Dimensional: 1D manifold in 2D space.
"""

import torch

from physicsnemo.mesh.mesh import Mesh


def load(
    start: tuple[float, float] = (0.0, 0.0),
    end: tuple[float, float] = (1.0, 1.0),
    n_points: int = 10,
    device: str = "cpu",
) -> Mesh:
    """Create a straight line segment in 2D space.

    Args:
        start: Starting point (x, y)
        end: Ending point (x, y)
        n_points: Number of points along the line
        device: Compute device ('cpu' or 'cuda')

    Returns:
        Mesh with n_manifold_dims=1, n_spatial_dims=2, n_cells=n_points-1
    """
    if n_points < 2:
        raise ValueError(f"n_points must be at least 2, got {n_points=}")

    # Interpolate between start and end
    t = torch.linspace(0.0, 1.0, n_points, device=device).unsqueeze(1)
    start_t = torch.tensor(start, dtype=torch.float32, device=device)
    end_t = torch.tensor(end, dtype=torch.float32, device=device)
    points = start_t * (1 - t) + end_t * t

    # Create edge cells
    cells = torch.stack(
        [
            torch.arange(n_points - 1, device=device),
            torch.arange(1, n_points, device=device),
        ],
        dim=1,
    )

    return Mesh(points=points, cells=cells)
