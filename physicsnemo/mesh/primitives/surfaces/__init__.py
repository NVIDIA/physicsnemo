"""2D manifolds in 3D space (surface meshes).

Includes spheres, cylinders, tori, platonic solids, and various parametric
surfaces embedded in 3D space.
"""

from physicsnemo.mesh.primitives.surfaces import (
    cone,
    cube_surface,
    cylinder,
    cylinder_open,
    disk,
    hemisphere,
    icosahedron_surface,
    mobius_strip,
    octahedron_surface,
    plane,
    sphere_uv,
    tetrahedron_surface,
    torus,
)

__all__ = [
    # Spheres
    "sphere_uv",
    # Cylinders
    "cylinder",
    "cylinder_open",
    # Other shapes
    "torus",
    "plane",
    "cone",
    "disk",
    "hemisphere",
    # Platonic solids
    "cube_surface",
    "tetrahedron_surface",
    "octahedron_surface",
    "icosahedron_surface",
    # Special surfaces
    "mobius_strip",
]
