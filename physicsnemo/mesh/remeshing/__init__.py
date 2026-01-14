"""Uniform mesh remeshing via clustering.

This module provides dimension-agnostic uniform remeshing based on the ACVD
(Approximate Centroidal Voronoi Diagram) clustering algorithm. It works for
arbitrary n-dimensional simplicial manifolds.

The algorithm:
1. Weights vertices by incident cell areas
2. Initializes clusters via area-based region growing
3. Removes spatially isolated cluster regions
4. Reconstructs a simplified mesh from cluster adjacency

The output mesh has approximately uniform cell distribution with ~0.5% non-manifold
edges (multiple faces sharing an edge), which is inherent to the face-mapping approach.

Current limitations:
- Energy minimization is disabled (made topology worse; needs investigation)
- Small percentage (~0.5-1%) of edges may be non-manifold with moderate cluster counts
- Higher cluster counts relative to mesh resolution produce better manifold quality

Example:
    >>> # Remesh a triangle mesh to ~1000 triangles
    >>> remeshed = remesh(mesh, n_clusters=1000)
    >>>
    >>> # With custom vertex weights for adaptive remeshing
    >>> weights = torch.ones(mesh.n_points)
    >>> weights[important_region] = 10.0
    >>> remeshed = remesh(mesh, n_clusters=500, weights=weights)
"""

from physicsnemo.mesh.remeshing._remeshing import remesh

__all__ = ["remesh"]
