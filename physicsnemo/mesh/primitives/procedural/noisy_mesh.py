"""Add Gaussian noise to any mesh.

This is a generic utility for creating perturbed versions of meshes.
"""

import torch

from physicsnemo.mesh.mesh import Mesh


def load(
    base_mesh: Mesh,
    noise_scale: float = 0.1,
    seed: int = 0,
) -> Mesh:
    """Add Gaussian noise to mesh vertex positions.

    Parameters
    ----------
    base_mesh : Mesh
        Input mesh to perturb.
    noise_scale : float
        Standard deviation of Gaussian noise.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    Mesh
        Mesh with same connectivity but perturbed vertex positions.
    """
    generator = torch.Generator(device=base_mesh.points.device).manual_seed(seed)

    # Generate noise with same shape as points
    noise = torch.randn(
        base_mesh.points.shape,
        dtype=base_mesh.points.dtype,
        device=base_mesh.points.device,
        generator=generator,
    )

    # Add scaled noise to points
    noisy_points = base_mesh.points + noise_scale * noise

    return Mesh(
        points=noisy_points,
        cells=base_mesh.cells,
        point_data=base_mesh.point_data,
        cell_data=base_mesh.cell_data,
        global_data=base_mesh.global_data,
    )
