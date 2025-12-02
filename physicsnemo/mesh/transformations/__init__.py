"""Geometric transformations for simplicial meshes.

This module provides linear and affine transformations with intelligent cache handling.
"""

from physicsnemo.mesh.transformations.geometric import (
    transform,
    translate,
    rotate,
    scale,
)

__all__ = [
    "transform",
    "translate",
    "rotate",
    "scale",
]
