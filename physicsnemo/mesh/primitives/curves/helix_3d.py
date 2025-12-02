"""Helical curve in 3D space.

Dimensional: 1D manifold in 3D space.
"""

import torch

from physicsnemo.mesh.mesh import Mesh


def load(
    radius: float = 1.0,
    height: float = 5.0,
    n_turns: float = 3.0,
    n_points: int = 100,
    device: str = "cpu",
) -> Mesh:
    """Create a helical curve in 3D space.

    Args:
        radius: Radius of the helix
        height: Total height of the helix
        n_turns: Number of complete turns
        n_points: Number of points along the helix
        device: Compute device ('cpu' or 'cuda')

    Returns:
        Mesh with n_manifold_dims=1, n_spatial_dims=3, n_cells=n_points-1
    """
    if n_points < 2:
        raise ValueError(f"n_points must be at least 2, got {n_points=}")

    theta = torch.linspace(0, 2 * torch.pi * n_turns, n_points, device=device)
    z = torch.linspace(0, height, n_points, device=device)

    points = torch.stack(
        [
            radius * torch.cos(theta),
            radius * torch.sin(theta),
            z,
        ],
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
