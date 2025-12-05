"""Example meshes for physicsnemo.mesh.

This module provides a comprehensive collection of canonical meshes organized by
category and dimensional configuration. All meshes are generated at runtime and
can be used for tutorials, testing, and experimentation.

Categories:
    - basic: Minimal test meshes (single cells, few cells)
    - curves: 1D manifolds in 1D, 2D, and 3D spaces
    - planar: 2D manifolds in 2D space (triangle meshes)
    - surfaces: 2D manifolds in 3D space (surface meshes)
    - volumes: 3D manifolds in 3D space (tetrahedral volumes)
    - procedural: Procedurally generated mesh variations
    - pyvista_datasets: Wrappers for PyVista example datasets

Usage:
    >>> from physicsnemo.mesh import examples
    >>> mesh = examples.surfaces.sphere_icosahedral.load(radius=1.0, subdivisions=2)
    >>> mesh = examples.pyvista_datasets.bunny.load()
"""

from physicsnemo.mesh.primitives import (
    basic,
    curves,
    planar,
    procedural,
    pyvista_datasets,
    surfaces,
    volumes,
)

__all__ = [
    "basic",
    "curves",
    "planar",
    "surfaces",
    "volumes",
    "procedural",
    "pyvista_datasets",
]
