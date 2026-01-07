"""Lumpy sphere with radial noise in 3D space.

Dimensional: 2D manifold in 3D space (closed, no boundary, irregular).
"""

import torch

from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral
from physicsnemo.mesh.mesh import Mesh


def load(
    radius: float = 1.0,
    subdivisions: int = 3,
    noise_amplitude: float = 0.1,
    seed: int = 0,
    device: str = "cpu",
) -> Mesh:
    """Create a lumpy sphere by adding radial noise to a sphere.

    Args:
        radius: Base radius of the sphere
        subdivisions: Number of subdivision levels
        noise_amplitude: Amplitude of radial noise
        seed: Random seed for reproducibility
        device: Compute device ('cpu' or 'cuda')

    Returns:
        Mesh with n_manifold_dims=2, n_spatial_dims=3
    """
    # Create base sphere
    mesh = sphere_icosahedral.load(
        radius=radius, subdivisions=subdivisions, device=device
    )

    # Add radial noise
    generator = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn(mesh.n_points, 1, generator=generator, device=device)

    # Compute radial direction for each point
    radial_dirs = mesh.points / torch.norm(mesh.points, dim=-1, keepdim=True)

    # Add noise in radial direction
    noisy_points = mesh.points + noise_amplitude * noise * radial_dirs

    return Mesh(
        points=noisy_points,
        cells=mesh.cells,
        point_data=mesh.point_data,
        cell_data=mesh.cell_data,
        global_data=mesh.global_data,
    )
