"""Curvature computation for simplicial meshes.

This module provides discrete differential geometry tools for computing
intrinsic and extrinsic curvatures on n-dimensional simplicial manifolds.

Gaussian Curvature (Intrinsic):
- Angle defect method: K = (full_angle - Σ angles) / voronoi_area
- Works for any codimension (intrinsic property)
- Measures intrinsic geometry (Theorema Egregium)

Mean Curvature (Extrinsic):
- Cotangent Laplacian method: H = ||L @ points|| / (2 * voronoi_area)
- Requires codimension-1 (needs normal vectors)
- Measures extrinsic bending

Example:
    >>> from physicsnemo.mesh.curvature import gaussian_curvature_vertices, mean_curvature_vertices
    >>>
    >>> # Compute Gaussian curvature
    >>> K = gaussian_curvature_vertices(mesh)
    >>>
    >>> # Compute mean curvature (codimension-1 only)
    >>> H = mean_curvature_vertices(mesh)
    >>>
    >>> # Or use Mesh properties:
    >>> K = mesh.gaussian_curvature_vertices
    >>> H = mesh.mean_curvature_vertices
"""

from physicsnemo.mesh.curvature.gaussian import (
    gaussian_curvature_cells,
    gaussian_curvature_vertices,
)
from physicsnemo.mesh.curvature.mean import mean_curvature_vertices

__all__ = [
    "gaussian_curvature_vertices",
    "gaussian_curvature_cells",
    "mean_curvature_vertices",
]
