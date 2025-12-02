"""2D manifolds in 2D space (planar triangle meshes).

Includes squares, circles, polygons, and other 2D shapes triangulated
in the plane.
"""

from physicsnemo.mesh.primitives.planar import (
    annulus_2d,
    circle_2d,
    equilateral_triangle,
    l_shape,
    rectangle,
    regular_polygon,
    structured_grid,
    unit_square,
)

__all__ = [
    "unit_square",
    "rectangle",
    "equilateral_triangle",
    "regular_polygon",
    "circle_2d",
    "annulus_2d",
    "l_shape",
    "structured_grid",
]
