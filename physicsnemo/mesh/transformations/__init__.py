"""Geometric transformations for simplicial meshes.

This module provides linear and affine transformations with intelligent cache handling.
"""

from physicsnemo.mesh.transformations.geometric import (
    rotate,
    scale,
    transform,
    translate,
)

__all__ = [
    "transform",
    "translate",
    "rotate",
    "scale",
]
