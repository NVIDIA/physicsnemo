"""1D manifolds (curves) in various dimensional spaces.

Includes line segments, circular arcs, parametric curves, and knots in
1D, 2D, and 3D embedding spaces.
"""

from physicsnemo.mesh.primitives.curves import (
    circle_2d,
    circle_3d,
    circular_arc_2d,
    ellipse_2d,
    helix_3d,
    line_segment_1d,
    line_segments_1d,
    polyline_2d,
    spiral_2d,
    spline_3d,
    straight_line_2d,
    straight_line_3d,
    trefoil_knot_3d,
)

__all__ = [
    # 1D → 1D
    "line_segment_1d",
    "line_segments_1d",
    # 1D → 2D
    "straight_line_2d",
    "circular_arc_2d",
    "circle_2d",
    "ellipse_2d",
    "polyline_2d",
    "spiral_2d",
    # 1D → 3D
    "straight_line_3d",
    "helix_3d",
    "circle_3d",
    "trefoil_knot_3d",
    "spline_3d",
]
