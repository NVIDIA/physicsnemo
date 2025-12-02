"""Boundary detection and facet extraction for simplicial meshes.

This module provides:
1. Boundary detection: identify vertices, edges, and cells on mesh boundaries
2. Facet extraction: extract lower-dimensional simplices from cells
3. Boundary mesh extraction: extract the watertight boundary surface
4. Topology checking: validate watertight and manifold properties
5. Mesh cleaning: repair common mesh issues
"""

from physicsnemo.mesh.boundaries._detection import (
    get_boundary_vertices,
    get_boundary_cells,
    get_boundary_edges,
)
from physicsnemo.mesh.boundaries._facet_extraction import (
    extract_candidate_facets,
    categorize_facets_by_count,
    deduplicate_and_aggregate_facets,
    extract_facet_mesh_data,
    compute_aggregation_weights,
)
from physicsnemo.mesh.boundaries._boundary_extraction import (
    extract_boundary_mesh_data,
)
from physicsnemo.mesh.boundaries._topology import (
    is_watertight,
    is_manifold,
)
from physicsnemo.mesh.boundaries._cleaning import (
    merge_duplicate_points,
    remove_duplicate_cells,
    remove_unused_points,
    clean_mesh,
)

__all__ = [
    # Boundary detection
    "get_boundary_vertices",
    "get_boundary_cells",
    "get_boundary_edges",
    # Facet extraction
    "extract_candidate_facets",
    "categorize_facets_by_count",
    "deduplicate_and_aggregate_facets",
    "extract_facet_mesh_data",
    "compute_aggregation_weights",
    # Boundary mesh extraction
    "extract_boundary_mesh_data",
    # Topology checking
    "is_watertight",
    "is_manifold",
    # Mesh cleaning
    "merge_duplicate_points",
    "remove_duplicate_cells",
    "remove_unused_points",
    "clean_mesh",
]
