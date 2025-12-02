"""Circular arc in 2D space.

Dimensional: 1D manifold in 2D space.
"""

import torch

from physicsnemo.mesh.mesh import Mesh


def load(
    radius: float = 1.0,
    start_angle: float = 0.0,
    end_angle: float = float(torch.pi / 2),
    n_points: int = 20,
    device: str = "cpu",
) -> Mesh:
    """Create a circular arc in 2D space.

    Args:
        radius: Radius of the arc
        start_angle: Starting angle in radians
        end_angle: Ending angle in radians
        n_points: Number of points along the arc
        device: Compute device ('cpu' or 'cuda')

    Returns:
        Mesh with n_manifold_dims=1, n_spatial_dims=2, n_cells=n_points-1
    """
    if n_points < 2:
        raise ValueError(f"n_points must be at least 2, got {n_points=}")

    theta = torch.linspace(start_angle, end_angle, n_points, device=device)

    points = torch.stack(
        [radius * torch.cos(theta), radius * torch.sin(theta)],
        dim=1,
    )

    # Create edge cells
    cells = torch.stack(
        [
            torch.arange(n_points - 1, device=device),
            torch.arange(1, n_points, device=device),
        ],
        dim=1,
    )

    return Mesh(points=points, cells=cells)
