"""PyVista example dataset wrappers.

These functions wrap PyVista's built-in example datasets, converting them
to physicsnemo.mesh format. PyVista handles caching of downloaded datasets automatically.
"""

from physicsnemo.mesh.primitives.pyvista_datasets import (
    airplane,
    ant,
    bunny,
    cow,
    globe,
    hexbeam,
    tetbeam,
)

__all__ = [
    # Surface meshes
    "airplane",
    "bunny",
    "ant",
    "cow",
    "globe",
    # Volume meshes
    "tetbeam",
    "hexbeam",
]
