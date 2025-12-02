"""Icosahedral sphere using subdivision in 3D space.

Dimensional: 2D manifold in 3D space (closed, no boundary).
"""

import torch

from physicsnemo.mesh.mesh import Mesh


def load(radius: float = 1.0, subdivisions: int = 2, device: str = "cpu") -> Mesh:
    """Create a sphere using icosahedron subdivision.

    Starts with a regular icosahedron and subdivides it, projecting
    vertices to the sphere surface after each subdivision.

    Args:
        radius: Radius of the sphere
        subdivisions: Number of subdivision levels (0 = icosahedron)
        device: Compute device ('cpu' or 'cuda')

    Returns:
        Mesh with n_manifold_dims=2, n_spatial_dims=3
    """
    ### Start with icosahedron
    phi = (1.0 + (5.0**0.5)) / 2.0  # Golden ratio

    # 12 vertices of icosahedron
    vertices = [
        [-1, phi, 0],
        [1, phi, 0],
        [-1, -phi, 0],
        [1, -phi, 0],
        [0, -1, phi],
        [0, 1, phi],
        [0, -1, -phi],
        [0, 1, -phi],
        [phi, 0, -1],
        [phi, 0, 1],
        [-phi, 0, -1],
        [-phi, 0, 1],
    ]

    # Normalize to unit sphere
    points = torch.tensor(vertices, dtype=torch.float32, device=device)
    points = points / torch.norm(points, dim=-1, keepdim=True)

    # 20 triangular faces of icosahedron
    faces = [
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ]

    cells = torch.tensor(faces, dtype=torch.int64, device=device)
    mesh = Mesh(points=points, cells=cells)

    ### Subdivide and project to sphere
    for _ in range(subdivisions):
        mesh = mesh.subdivide(levels=1, filter="linear")
        # Project all points to sphere surface
        mesh = Mesh(
            points=mesh.points / torch.norm(mesh.points, dim=-1, keepdim=True),
            cells=mesh.cells,
            point_data=mesh.point_data,
            cell_data=mesh.cell_data,
            global_data=mesh.global_data,
        )

    ### Scale to desired radius
    if radius != 1.0:
        mesh = Mesh(
            points=mesh.points * radius,
            cells=mesh.cells,
            point_data=mesh.point_data,
            cell_data=mesh.cell_data,
            global_data=mesh.global_data,
        )

    return mesh
