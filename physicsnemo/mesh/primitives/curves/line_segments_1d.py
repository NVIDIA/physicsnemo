"""Multiple disconnected line segments in 1D space.

Dimensional: 1D manifold in 1D space.
"""

import torch

from physicsnemo.mesh.mesh import Mesh


def load(n_segments: int = 3, gap: float = 0.2, device: str = "cpu") -> Mesh:
    """Create multiple disconnected line segments in 1D space.

    Parameters
    ----------
    n_segments : int
        Number of disconnected segments.
    gap : float
        Gap between segments.
    device : str
        Compute device ('cpu' or 'cuda').

    Returns
    -------
    Mesh
        Mesh with n_manifold_dims=1, n_spatial_dims=1, n_cells=n_segments.
    """
    if n_segments < 1:
        raise ValueError(f"n_segments must be at least 1, got {n_segments=}")

    points = []
    cells = []

    for i in range(n_segments):
        start = i * (1.0 + gap)
        end = start + 1.0
        points.extend([[start], [end]])
        cells.append([2 * i, 2 * i + 1])

    points = torch.tensor(points, dtype=torch.float32, device=device)
    cells = torch.tensor(cells, dtype=torch.int64, device=device)

    return Mesh(points=points, cells=cells)
