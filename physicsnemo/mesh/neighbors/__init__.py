"""Neighbor and adjacency computation for simplicial meshes.

This module provides GPU-compatible functions for computing various adjacency
relationships in simplicial meshes, including point-to-cells, point-to-points,
and cell-to-cells adjacency.

All adjacency relationships are returned as Adjacency tensorclass objects using
offset-indices encoding for efficient representation of ragged arrays.
"""

from physicsnemo.mesh.neighbors._adjacency import Adjacency, build_adjacency_from_pairs
from physicsnemo.mesh.neighbors._cell_neighbors import (
    get_cell_to_cells_adjacency,
    get_cells_to_points_adjacency,
)
from physicsnemo.mesh.neighbors._point_neighbors import (
    get_point_to_cells_adjacency,
    get_point_to_points_adjacency,
)

__all__ = [
    "Adjacency",
    "build_adjacency_from_pairs",
    "get_point_to_cells_adjacency",
    "get_point_to_points_adjacency",
    "get_cell_to_cells_adjacency",
    "get_cells_to_points_adjacency",
]
