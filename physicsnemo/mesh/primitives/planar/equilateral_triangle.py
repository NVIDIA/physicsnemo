"""Equilateral triangle triangulated in 2D space.

Dimensional: 2D manifold in 2D space.
"""

import torch

from physicsnemo.mesh.mesh import Mesh


def load(
    side_length: float = 1.0, n_subdivisions: int = 0, device: str = "cpu"
) -> Mesh:
    """Create an equilateral triangle in 2D space.

    Args:
        side_length: Length of each side
        n_subdivisions: Number of subdivision levels
        device: Compute device ('cpu' or 'cuda')

    Returns:
        Mesh with n_manifold_dims=2, n_spatial_dims=2
    """
    if n_subdivisions < 0:
        raise ValueError(f"n_subdivisions must be non-negative, got {n_subdivisions=}")

    # Create vertices of equilateral triangle
    height = side_length * (3**0.5) / 2
    points = torch.tensor(
        [[0.0, 0.0], [side_length, 0.0], [side_length / 2, height]],
        dtype=torch.float32,
        device=device,
    )
    cells = torch.tensor([[0, 1, 2]], dtype=torch.int64, device=device)

    mesh = Mesh(points=points, cells=cells)

    # Apply subdivisions if requested
    if n_subdivisions > 0:
        mesh = mesh.subdivide(levels=n_subdivisions, filter="linear")

    return mesh
