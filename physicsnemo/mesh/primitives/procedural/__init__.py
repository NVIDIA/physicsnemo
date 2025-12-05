"""Procedurally generated mesh variations.

Includes functions for adding noise, perturbations, and other procedural
modifications to meshes, plus standalone noise generation functions.
"""

from physicsnemo.mesh.primitives.procedural import (
    noisy_mesh,
    perturbed_grid,
)
from physicsnemo.mesh.primitives.procedural.noise import (
    perlin_noise_1d,
    perlin_noise_2d,
    perlin_noise_3d,
    perlin_noise_nd,
)

__all__ = [
    "lumpy_sphere",
    "noisy_mesh",
    "perturbed_grid",
    "perlin_noise_nd",
    "perlin_noise_1d",
    "perlin_noise_2d",
    "perlin_noise_3d",
]
