"""Trefoil knot curve in 3D space.

Dimensional: 1D manifold in 3D space (closed, knotted).
"""

import torch

from physicsnemo.mesh.mesh import Mesh


def load(scale: float = 1.0, n_points: int = 100, device: str = "cpu") -> Mesh:
    """Create a trefoil knot curve in 3D space.

    The trefoil knot is the simplest nontrivial knot.
    Parametric equations:
        x = sin(t) + 2*sin(2*t)
        y = cos(t) - 2*cos(2*t)
        z = -sin(3*t)

    Args:
        scale: Overall scale factor
        n_points: Number of points around the knot
        device: Compute device ('cpu' or 'cuda')

    Returns:
        Mesh with n_manifold_dims=1, n_spatial_dims=3, n_cells=n_points
    """
    if n_points < 3:
        raise ValueError(f"n_points must be at least 3, got {n_points=}")

    t = torch.linspace(0, 2 * torch.pi, n_points + 1, device=device)[:-1]

    x = scale * (torch.sin(t) + 2 * torch.sin(2 * t))
    y = scale * (torch.cos(t) - 2 * torch.cos(2 * t))
    z = scale * (-torch.sin(3 * t))

    points = torch.stack([x, y, z], dim=1)

    # Create edge cells, including wrap-around edge
    cells = torch.stack(
        [
            torch.arange(n_points, device=device),
            torch.cat(
                [
                    torch.arange(1, n_points, device=device),
                    torch.tensor([0], device=device),
                ]
            ),
        ],
        dim=1,
    )

    return Mesh(points=points, cells=cells)
