"""Sampling operations for meshes.

This module provides functions for sampling points on meshes, including:
- Random uniform point sampling on cells using Dirichlet distributions
- Spatial data sampling at query points with interpolation
- Hierarchical BVH-accelerated sampling for large meshes
"""

from physicsnemo.mesh.sampling.random_point_sampling import (
    sample_random_points_on_cells,
)
from physicsnemo.mesh.sampling.sample_data import (
    sample_data_at_points,
    find_containing_cells,
    find_all_containing_cells,
    find_nearest_cells,
    compute_barycentric_coordinates,
)
from physicsnemo.mesh.sampling import sample_data_hierarchical

__all__ = [
    "sample_random_points_on_cells",
    "sample_data_at_points",
    "find_containing_cells",
    "find_all_containing_cells",
    "find_nearest_cells",
    "compute_barycentric_coordinates",
    "sample_data_hierarchical",
]
