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
from jaxtyping import Float, Int

if TYPE_CHECKING:
    from physicsnemo.mesh.mesh import Mesh


def sample_random_points_on_cells(
    mesh: "Mesh",
    cell_indices: Sequence[int] | Int[torch.Tensor, " n_samples"] | None = None,
    alpha: float = 1.0,
    generator: torch.Generator | None = None,
) -> Float[torch.Tensor, "n_samples n_spatial_dims"]:
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
    generator : torch.Generator, optional
        Random generator used when ``alpha=1``. This enables reproducible
        sampling without modifying global PyTorch random state.

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
    # Gamma(1, 1) is Exponential(1). The inverse-CDF form accepts the caller's
    # generator and draws directly in the mesh dtype.
    dtype = mesh.points.dtype
    if alpha == 1.0:
        uniform = torch.rand(
            (n_samples, mesh.n_manifold_dims + 1),
            dtype=dtype,
            device=mesh.points.device,
            generator=generator,
        )
        raw_barycentric_coords = -torch.log(
            uniform.clamp_min(torch.finfo(uniform.dtype).tiny)
        )
    else:
        if generator is not None:
            raise ValueError("generator is only supported when alpha=1.0.")
        if torch.compiler.is_compiling():
            raise NotImplementedError(
                f"alpha={alpha!r} is not supported under torch.compile.\n"
                f"PyTorch does not yet support sampling from a Gamma distribution\n"
                f"when using torch.compile. Use alpha=1.0 (uniform distribution) instead, or disable torch.compile.\n"
                f"See https://github.com/pytorch/pytorch/issues/165751."
            )
        _rate = torch.ones((), device=mesh.points.device, dtype=dtype)
        distribution = torch.distributions.Gamma(
            concentration=torch.full((), alpha, device=mesh.points.device, dtype=dtype),
            rate=_rate,
        )
        raw_barycentric_coords = distribution.sample(
            (n_samples, mesh.n_manifold_dims + 1)
        )

    ### Normalize so they sum to 1
    barycentric_coords = F.normalize(raw_barycentric_coords, p=1, dim=-1)

    ### Compute weighted combination of cell vertices
    # Get the vertices for the selected cells: (n_samples, n_manifold_dims + 1, n_spatial_dims)
    selected_cell_vertices = mesh.points[mesh.cells[cell_indices]]

    # Compute weighted sum: (n_samples, n_spatial_dims)
    return (barycentric_coords.unsqueeze(-1) * selected_cell_vertices).sum(dim=1)


def sample_random_points(
    mesh: "Mesh",
    num_samples: int,
    *,
    generator: torch.Generator | None = None,
) -> Float[torch.Tensor, "num_samples n_spatial_dims"]:
    """Sample uniformly over the measure of an entire simplicial mesh.

    Cells are selected in proportion to their length, area, or volume, then a
    point is sampled uniformly inside each selected simplex. For a triangle
    surface mesh this produces area-uniform point clouds suitable for geometric
    neural-network inputs.

    Parameters
    ----------
    mesh : Mesh
        Non-empty simplicial mesh.
    num_samples : int
        Number of points to return.
    generator : torch.Generator, optional
        Random generator for reproducible cell and barycentric sampling.

    Returns
    -------
    Float[torch.Tensor, "num_samples n_spatial_dims"]
        Sampled point coordinates. Each row lies within one mesh cell.

    Raises
    ------
    ValueError
        If ``num_samples`` is not positive, the mesh has no cells, or the cell
        measures are non-finite or sum to zero.

    Examples
    --------
    >>> from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral
    >>> mesh = sphere_icosahedral.load(subdivisions=2)
    >>> points = sample_random_points(mesh, 1000)
    >>> points.shape
    torch.Size([1000, 3])
    """
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if mesh.n_cells == 0:
        raise ValueError("mesh must contain at least one cell")
    measures = mesh.cell_areas
    if not torch.compiler.is_compiling():
        if not bool(torch.isfinite(measures).all()):
            raise ValueError("mesh cell measures must be finite")
        if float(measures.sum()) <= 0.0:
            raise ValueError("mesh must contain positive-measure cells")
    ### Select cells with probability proportional to their measure.
    # Inverse-CDF sampling (cumulative measure + searchsorted) instead of
    # torch.multinomial, whose 2**24-category cap rejects meshes that large
    # CFD/automotive cases routinely exceed. The cumulative sum runs in
    # float64: a float32 running total stops resolving individual cell
    # measures once it exceeds them by ~2**24, silently zeroing the selection
    # probability of late-indexed cells (~8% of a 20M-cell mesh).
    cumulative_measure = torch.cumsum(measures, dim=0, dtype=torch.float64)
    thresholds = cumulative_measure[-1] * torch.rand(
        num_samples,
        dtype=torch.float64,
        device=measures.device,
        generator=generator,
    )
    # right=True keeps zero-measure cells unselectable (their zero-width CDF
    # interval cannot strictly contain a threshold); the clamp covers a
    # threshold that rounds up to exactly the total measure.
    cell_indices = torch.searchsorted(
        cumulative_measure, thresholds, right=True
    ).clamp_max_(mesh.n_cells - 1)
    return sample_random_points_on_cells(
        mesh,
        cell_indices,
        generator=generator,
    )
