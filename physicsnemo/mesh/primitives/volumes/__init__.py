"""3D manifolds in 3D space (volume meshes).

Includes tetrahedral meshes of cubes, spheres, cylinders, and other
3D solid shapes.
"""

from physicsnemo.mesh.primitives.volumes import (
    beam_volume,
    cube_volume,
    cylinder_volume,
    sphere_volume,
    tetrahedron_volume,
)

__all__ = [
    "cube_volume",
    "sphere_volume",
    "cylinder_volume",
    "tetrahedron_volume",
    "beam_volume",
]
