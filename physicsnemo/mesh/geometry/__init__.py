"""Geometric primitives and computations for simplicial meshes.

This module contains fundamental geometric operations that are shared across
the codebase, including:
- Dual mesh (Voronoi/circumcentric) computations
- Circumcenter calculations
- Support volume computations (for DEC)
- Geometric utility functions

These are used by both DEC operators (calculus module) and differential
geometry computations (curvature module).
"""

from physicsnemo.mesh.geometry.dual_meshes import compute_dual_volumes_0
