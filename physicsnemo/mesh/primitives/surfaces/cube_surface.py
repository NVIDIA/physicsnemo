"""Cube surface triangulated in 3D space.

Dimensional: 2D manifold in 3D space (closed, no boundary).
"""

import torch

from physicsnemo.mesh.mesh import Mesh


def load(size: float = 1.0, device: torch.device | str = "cpu") -> Mesh:
    """Create a cube surface triangulated in 3D space.

    The cube is centered at the origin with vertices at (±size/2, ±size/2, ±size/2).
    Each face is split into 2 triangles with consistent outward-facing normals
    (counter-clockwise winding when viewed from outside).

    Parameters
    ----------
    size : float
        Side length of the cube.
    device : torch.device or str
        Compute device ('cpu' or 'cuda').

    Returns
    -------
    Mesh
        Mesh with n_manifold_dims=2, n_spatial_dims=3, 8 vertices, 12 triangles.

    Examples
    --------
    >>> from physicsnemo.mesh.primitives.surfaces import cube_surface
    >>> mesh = cube_surface.load()
    >>> mesh.n_points, mesh.n_cells
    (8, 12)
    >>> mesh.n_manifold_dims, mesh.n_spatial_dims
    (2, 3)
    """
    s = size / 2

    # 8 vertices of the cube
    # Vertex ordering: binary encoding of (x+, y+, z+)
    #   0: (-, -, -)    4: (-, -, +)
    #   1: (+, -, -)    5: (+, -, +)
    #   2: (-, +, -)    6: (-, +, +)
    #   3: (+, +, -)    7: (+, +, +)
    points = torch.tensor(
        [
            [-s, -s, -s],  # 0
            [+s, -s, -s],  # 1
            [-s, +s, -s],  # 2
            [+s, +s, -s],  # 3
            [-s, -s, +s],  # 4
            [+s, -s, +s],  # 5
            [-s, +s, +s],  # 6
            [+s, +s, +s],  # 7
        ],
        dtype=torch.float32,
        device=device,
    )

    # 12 triangles (2 per face, CCW winding when viewed from outside)
    # fmt: off
    cells = torch.tensor(
        [
            # -Z face (z = -s): vertices 0, 1, 2, 3; normal points -Z
            [0, 2, 1],
            [1, 2, 3],
            # +Z face (z = +s): vertices 4, 5, 6, 7; normal points +Z
            [4, 5, 6],
            [5, 7, 6],
            # -Y face (y = -s): vertices 0, 1, 4, 5; normal points -Y
            [0, 1, 4],
            [1, 5, 4],
            # +Y face (y = +s): vertices 2, 3, 6, 7; normal points +Y
            [2, 6, 3],
            [3, 6, 7],
            # -X face (x = -s): vertices 0, 2, 4, 6; normal points -X
            [0, 4, 2],
            [2, 4, 6],
            # +X face (x = +s): vertices 1, 3, 5, 7; normal points +X
            [1, 3, 5],
            [3, 7, 5],
        ],
        dtype=torch.int64,
        device=device,
    )
    # fmt: on

    return Mesh(points=points, cells=cells)
