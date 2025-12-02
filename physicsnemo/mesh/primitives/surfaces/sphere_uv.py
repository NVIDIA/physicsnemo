"""UV sphere using latitude/longitude parameterization in 3D space.

Dimensional: 2D manifold in 3D space (closed, no boundary).
"""

import pyvista as pv

from physicsnemo.mesh.io import from_pyvista
from physicsnemo.mesh.mesh import Mesh


def load(
    radius: float = 1.0,
    theta_resolution: int = 30,
    phi_resolution: int = 30,
    device: str = "cpu",
) -> Mesh:
    """Create a UV sphere using latitude/longitude parameterization.

    Args:
        radius: Radius of the sphere
        theta_resolution: Number of points around the equator
        phi_resolution: Number of points from pole to pole
        device: Compute device ('cpu' or 'cuda')

    Returns:
        Mesh with n_manifold_dims=2, n_spatial_dims=3
    """
    pv_sphere = pv.Sphere(
        radius=radius,
        theta_resolution=theta_resolution,
        phi_resolution=phi_resolution,
    )

    mesh = from_pyvista(pv_sphere, manifold_dim=2)

    # Move to specified device
    if device != str(mesh.points.device):
        mesh = mesh.to(device)

    return mesh
