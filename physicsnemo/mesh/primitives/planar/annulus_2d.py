"""Annulus (ring) triangulated in 2D space.

Dimensional: 2D manifold in 2D space.
"""

import pyvista as pv

from physicsnemo.mesh.io import from_pyvista
from physicsnemo.mesh.mesh import Mesh


def load(
    inner_radius: float = 0.5,
    outer_radius: float = 1.0,
    n_radial: int = 5,
    n_angular: int = 32,
    device: str = "cpu",
) -> Mesh:
    """Create an annulus (ring) triangulated in 2D space.

    Args:
        inner_radius: Inner radius of the annulus
        outer_radius: Outer radius of the annulus
        n_radial: Number of points in radial direction
        n_angular: Number of points around the circumference
        device: Compute device ('cpu' or 'cuda')

    Returns:
        Mesh with n_manifold_dims=2, n_spatial_dims=2
    """
    if inner_radius >= outer_radius:
        raise ValueError(
            f"inner_radius must be < outer_radius, got {inner_radius=}, {outer_radius=}"
        )

    # Use PyVista to create a disk with a hole
    pv_disk = pv.Disc(
        center=(0.0, 0.0, 0.0),
        inner=inner_radius,
        outer=outer_radius,
        r_res=n_radial,
        c_res=n_angular,
    )

    # Extract only x and y coordinates (discard z)
    mesh = from_pyvista(pv_disk, manifold_dim=2)

    # Project to 2D by removing z coordinate
    points_2d = mesh.points[:, :2]
    mesh = Mesh(points=points_2d, cells=mesh.cells)

    # Move to specified device
    if device != str(mesh.points.device):
        mesh = mesh.to(device)

    return mesh
