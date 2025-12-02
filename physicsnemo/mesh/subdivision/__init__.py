"""Mesh subdivision algorithms for simplicial meshes.

This module provides subdivision schemes for refining simplicial meshes:
- Linear: Simple midpoint subdivision (interpolating)
- Butterfly: Weighted stencil subdivision for smooth surfaces (interpolating)
- Loop: Valence-based subdivision with vertex repositioning (approximating)

All schemes work by:
1. Extracting edges from the mesh
2. Adding new vertices (at or near edge midpoints)
3. Splitting each n-simplex into 2^n child simplices
4. Interpolating/propagating data to new mesh

Example:
    >>> from physicsnemo.mesh.subdivision import subdivide_linear
    >>> subdivided = subdivide_linear(mesh)
    >>> # Or use the Mesh method:
    >>> subdivided = mesh.subdivide(levels=2, filter="loop")
"""

from physicsnemo.mesh.subdivision.butterfly import subdivide_butterfly
from physicsnemo.mesh.subdivision.linear import subdivide_linear
from physicsnemo.mesh.subdivision.loop import subdivide_loop

__all__ = [
    "subdivide_linear",
    "subdivide_butterfly",
    "subdivide_loop",
]
