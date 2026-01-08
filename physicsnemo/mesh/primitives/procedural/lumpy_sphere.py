"""Lumpy sphere with radial noise in 3D space.

Dimensional: 2D manifold in 3D space (closed, no boundary, irregular).
"""

import torch

from physicsnemo.mesh.mesh import Mesh
from physicsnemo.mesh.primitives.surfaces import icosahedron_surface


def load(
    radius: float = 1.0,
    subdivisions: int = 3,
    noise_amplitude: float = 0.5,
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
    mesh = icosahedron_surface.load(radius=radius, device=device)
    generator = torch.Generator(device=device).manual_seed(seed)
    noise = noise_amplitude * torch.randn(
        mesh.n_points, 1, generator=generator, device=device
    )
    mesh.points = mesh.points * noise.exp()

    return mesh.subdivide(subdivisions, "loop")
