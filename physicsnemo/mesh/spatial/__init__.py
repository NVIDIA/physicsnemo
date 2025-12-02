"""Spatial acceleration structures for efficient queries on large meshes.

This module provides data structures and algorithms for fast spatial queries:
- BVH (Bounding Volume Hierarchy) for point-in-cell queries
"""

from physicsnemo.mesh.spatial.bvh import BVH

__all__ = ["BVH"]
