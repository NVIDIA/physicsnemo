"""Perturbed structured grid in 2D space.

Dimensional: 2D manifold in 2D space (irregular).
"""

import torch

from physicsnemo.mesh.mesh import Mesh
from physicsnemo.mesh.primitives.planar import structured_grid


def load(
    n_x: int = 11,
    n_y: int = 11,
    perturbation_scale: float = 0.05,
    seed: int = 0,
    device: str = "cpu",
) -> Mesh:
    """Create a perturbed structured grid in 2D space.

    Parameters
    ----------
    n_x : int
        Number of points in x-direction.
    n_y : int
        Number of points in y-direction.
    perturbation_scale : float
        Amount of random perturbation.
    seed : int
        Random seed for reproducibility.
    device : str
        Compute device ('cpu' or 'cuda').

    Returns
    -------
    Mesh
        Mesh with n_manifold_dims=2, n_spatial_dims=2.
    """
    # Create base structured grid
    mesh = structured_grid.load(
        x_min=0.0,
        x_max=1.0,
        y_min=0.0,
        y_max=1.0,
        n_x=n_x,
        n_y=n_y,
        device=device,
    )

    # Add perturbation to interior points (not boundary)
    generator = torch.Generator(device=device).manual_seed(seed)

    # Identify boundary points
    x_coords = mesh.points[:, 0]
    y_coords = mesh.points[:, 1]
    is_boundary = (
        (torch.abs(x_coords) < 1e-6)
        | (torch.abs(x_coords - 1.0) < 1e-6)
        | (torch.abs(y_coords) < 1e-6)
        | (torch.abs(y_coords - 1.0) < 1e-6)
    )

    # Generate perturbation
    perturbation = (
        torch.randn(
            mesh.points.shape,
            dtype=mesh.points.dtype,
            device=device,
            generator=generator,
        )
        * perturbation_scale
    )

    # Zero out perturbation for boundary points
    perturbation[is_boundary] = 0.0

    # Apply perturbation
    perturbed_points = mesh.points + perturbation

    return Mesh(
        points=perturbed_points,
        cells=mesh.cells,
        point_data=mesh.point_data,
        cell_data=mesh.cell_data,
        global_data=mesh.global_data,
    )
