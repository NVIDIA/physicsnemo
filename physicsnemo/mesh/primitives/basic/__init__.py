"""Minimal test meshes for basic validation.

These meshes contain single cells or a few cells and are useful for unit testing
and validating basic mesh operations.
"""

from physicsnemo.mesh.primitives.basic import (
    single_edge_2d,
    single_edge_3d,
    single_point_2d,
    single_point_3d,
    single_tetrahedron,
    single_triangle_2d,
    single_triangle_3d,
    three_edges_2d,
    three_edges_3d,
    three_points_2d,
    three_points_3d,
    two_tetrahedra,
    two_triangles_2d,
    two_triangles_3d,
)

__all__ = [
    "single_point_2d",
    "single_point_3d",
    "three_points_2d",
    "three_points_3d",
    "single_edge_2d",
    "single_edge_3d",
    "three_edges_2d",
    "three_edges_3d",
    "single_triangle_2d",
    "single_triangle_3d",
    "two_triangles_2d",
    "two_triangles_3d",
    "single_tetrahedron",
    "two_tetrahedra",
]
