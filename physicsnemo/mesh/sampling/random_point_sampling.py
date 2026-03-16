# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Random sampling of points on mesh cells."""

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from physicsnemo.mesh.mesh import Mesh


def sample_random_points_on_cells(
    mesh: "Mesh",
    cell_indices: Sequence[int] | torch.Tensor | None = None,
    alpha: float = 1.0,
) -> torch.Tensor:
    """Sample random points uniformly distributed on specified cells of the mesh.

    Uses a Dirichlet distribution to generate barycentric coordinates, which are
    then used to compute random points as weighted combinations of cell vertices.
    The concentration parameter alpha controls the distribution of samples within
    each cell (simplex).

    Parameters
    ----------
    mesh : Mesh
        The mesh to sample from.
    cell_indices : Sequence[int] | torch.Tensor | None
        Indices of cells to sample from. Can be a Sequence or tensor.
        Allows repeated indices to sample multiple points from the same cell.
        If None, samples one point from each cell (equivalent to arange(n_cells)).
        Shape: (n_samples,) where n_samples is the number of points to sample.
    alpha : float
        Concentration parameter for the Dirichlet distribution. Controls how
        samples are distributed within each cell:
        - alpha = 1.0: Uniform distribution over the simplex (default)
        - alpha > 1.0: Concentrates samples toward the center of each cell
        - alpha < 1.0: Concentrates samples toward vertices and edges

    Returns
    -------
    torch.Tensor
        Random points on cells, shape (n_samples, n_spatial_dims). Each point lies
        within its corresponding cell. If cell_indices is None, n_samples = n_cells.

    Raises
    ------
    NotImplementedError
        If alpha != 1.0 and torch.compile is being used.
        This is due to a PyTorch limitation with Gamma distributions under torch.compile.
    IndexError
        If any cell_indices are out of bounds.

    Examples
    --------
    >>> from physicsnemo.mesh.primitives.basic import two_triangles_2d
    >>> mesh = two_triangles_2d.load()
    >>> # Sample one point from each cell uniformly
    >>> points = sample_random_points_on_cells(mesh)
    >>> assert points.shape == (mesh.n_cells, mesh.n_spatial_dims)
    >>> # Sample with concentration toward cell centers
    >>> points = sample_random_points_on_cells(mesh, alpha=3.0)
    """
    ### Handle default case: sample one point from each cell
    if cell_indices is None:
        cell_indices = torch.arange(
            mesh.n_cells,
            device=mesh.points.device,
            dtype=torch.long,
        )
    else:
        # Convert to tensor if needed (as_tensor avoids copy if already a tensor)
        cell_indices = torch.as_tensor(
            cell_indices,
            device=mesh.points.device,
            dtype=torch.long,
        )

    ### Validate indices
    if not torch.compiler.is_compiling():
        if len(cell_indices) > 0:
            if cell_indices.min() < 0:
                raise IndexError(
                    f"cell_indices contains negative values: {cell_indices.min()=}"
                )
            if cell_indices.max() >= mesh.n_cells:
                raise IndexError(
                    f"cell_indices contains out-of-bounds values: "
                    f"{cell_indices.max()=} >= {mesh.n_cells=}"
                )

    n_samples = len(cell_indices)

    ### Sample from Gamma(alpha, 1) distribution and normalize to get Dirichlet
    # When alpha=1, Gamma(1,1) is equivalent to Exponential(1), which is more efficient
    if alpha == 1.0:
        distribution = torch.distributions.Exponential(
            rate=torch.ones((), device=mesh.points.device),
        )
    else:
        if torch.compiler.is_compiling():
            raise NotImplementedError(
                f"alpha={alpha!r} is not supported under torch.compile.\n"
                f"PyTorch does not yet support sampling from a Gamma distribution\n"
                f"when using torch.compile. Use alpha=1.0 (uniform distribution) instead, or disable torch.compile.\n"
                f"See https://github.com/pytorch/pytorch/issues/165751."
            )
        _rate = torch.ones((), device=mesh.points.device)
        distribution = torch.distributions.Gamma(
            concentration=torch.full((), alpha, device=mesh.points.device),
            rate=_rate,
        )

    raw_barycentric_coords = distribution.sample((n_samples, mesh.n_manifold_dims + 1))

    ### Normalize so they sum to 1
    barycentric_coords = F.normalize(raw_barycentric_coords, p=1, dim=-1)

    ### Compute weighted combination of cell vertices
    # Get the vertices for the selected cells: (n_samples, n_manifold_dims + 1, n_spatial_dims)
    selected_cell_vertices = mesh.points[mesh.cells[cell_indices]]

    # Compute weighted sum: (n_samples, n_spatial_dims)
    return (barycentric_coords.unsqueeze(-1) * selected_cell_vertices).sum(dim=1)
