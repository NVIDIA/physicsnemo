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

"""Discrete restricted Voronoi partition of mesh cells.

Given a mesh with M cells and N seed points on the surface, assigns each cell
to its nearest seed (by Euclidean centroid distance) and accumulates per-cluster
area, normal, and centroid.  This approximates the restricted Voronoi diagram
on the surface by grouping whole cells rather than splitting them - exact in
the limit M/N -> infinity on a fine mesh.

This is complementary to :func:`~physicsnemo.mesh.remeshing.remesh` (ACVD),
which creates *new* mesh topology via iterative Centroidal Voronoi Tessellation.
``partition_cells`` preserves the original cells and produces aggregate
properties - no topology reconstruction and no external dependencies.

``partition_cells`` is also a natural building block for a pure-PyTorch CVT
(Lloyd's algorithm iterates: partition -> move seeds to cluster centroids ->
repeat).
"""

from typing import NamedTuple

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int

from physicsnemo.mesh.mesh import Mesh
from physicsnemo.mesh.utilities._scatter_ops import scatter_aggregate
from physicsnemo.nn.functional.neighbors import knn


class CellPartition(NamedTuple):
    """Result of partitioning mesh cells by nearest seed point.

    Parameters
    ----------
    assignments : Int[torch.Tensor, " n_cells"]
        Index of the nearest seed for each original cell.
    cluster_areas : Float[torch.Tensor, " n_seeds"]
        Total cell area assigned to each seed.
        Sums to the total surface area of the original mesh by construction.
    cluster_normals : Float[torch.Tensor, "n_seeds n_spatial_dims"]
        Area-weighted average unit normal per cluster.
    cluster_centroids : Float[torch.Tensor, "n_seeds n_spatial_dims"]
        Area-weighted centroid per cluster.
        For a well-centered seed this is close to the seed itself; the
        difference measures how far the seed is from its Voronoi centroid
        (exactly the quantity that Lloyd's algorithm drives to zero).
    """

    assignments: Int[torch.Tensor, " n_cells"]
    cluster_areas: Float[torch.Tensor, " n_seeds"]
    cluster_normals: Float[torch.Tensor, "n_seeds n_spatial_dims"]
    cluster_centroids: Float[torch.Tensor, "n_seeds n_spatial_dims"]


def partition_cells(
    mesh: Mesh,
    seeds: Float[torch.Tensor, "n_seeds n_spatial_dims"],
) -> CellPartition:
    """Partition mesh cells into Voronoi regions around seed points.

    Each cell is assigned to the seed whose Euclidean distance to the cell
    centroid is smallest.  Aggregate geometric properties (area, normal,
    centroid) are then accumulated per cluster via ``scatter_add``.

    This is the discrete analog of a restricted Voronoi diagram: the original
    cells discretize the continuous surface, and their nearest-seed assignment
    approximates the Voronoi partition.  The approximation is exact when the
    original mesh is infinitely fine relative to the seed spacing.

    The nearest-neighbor search uses :func:`~physicsnemo.nn.functional.neighbors.knn`
    which auto-dispatches to the optimal backend (cuML on GPU, scipy KDTree
    on CPU) for O(M log N) query complexity.

    Parameters
    ----------
    mesh : Mesh
        Source mesh whose cells will be partitioned.  For codimension-1
        meshes (surfaces), cluster normals are computed from cell normals.
        For other meshes, cluster normals are zero vectors.
    seeds : Float[torch.Tensor, "n_seeds n_spatial_dims"]
        Seed point positions.  ``n_spatial_dims`` must match
        ``mesh.n_spatial_dims``.

    Returns
    -------
    CellPartition
        Named tuple with ``assignments``, ``cluster_areas``,
        ``cluster_normals``, and ``cluster_centroids``.

    Raises
    ------
    ValueError
        If ``seeds`` and ``mesh`` have different devices or dtypes.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.mesh import Mesh
    >>> from physicsnemo.mesh.remeshing import partition_cells
    >>> # Two triangles in 3D (codimension-1 surface)
    >>> pts = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [1., 1., 0.]])
    >>> cells = torch.tensor([[0, 1, 2], [1, 3, 2]])
    >>> mesh = Mesh(points=pts, cells=cells)
    >>> seeds = torch.tensor([[0.25, 0.25, 0.0], [0.75, 0.75, 0.0]])
    >>> result = partition_cells(mesh, seeds)
    >>> result.assignments.tolist()
    [0, 1]
    >>> abs(result.cluster_areas.sum().item() - mesh.cell_areas.sum().item()) < 1e-10
    True

    Notes
    -----
    - Uses Euclidean (ambient-space) distance, not geodesic distance.  For
      smooth surfaces where inter-seed spacing is small relative to the radius
      of curvature, this is an excellent approximation.
    - Every original cell is assigned to exactly one cluster, so
      ``cluster_areas.sum() == mesh.cell_areas.sum()`` by construction.
    - If a cluster receives no cells (possible when seeds outnumber cells or
      cluster heavily), its area is 0, its normal is the zero vector, and its
      centroid falls back to the seed position.
    """
    n_seeds = len(seeds)

    ### Validate that seeds and mesh share the same device and dtype
    if seeds.device != mesh.points.device:
        raise ValueError(
            f"`seeds` and `mesh` must be on the same device, "
            f"got {seeds.device=} and {mesh.points.device=}"
        )
    if seeds.dtype != mesh.points.dtype:
        raise ValueError(
            f"`seeds` and `mesh` must have the same dtype, "
            f"got {seeds.dtype=} and {mesh.points.dtype=}"
        )

    device = seeds.device
    dtype = mesh.points.dtype

    ### Read source geometry (cached on Mesh)
    n_dims = mesh.n_spatial_dims
    cell_centroids = mesh.cell_centroids  # (M, D)
    cell_areas = mesh.cell_areas  # (M,)
    has_normals = mesh.codimension == 1

    ### Assign each cell to its nearest seed via kNN search (k=1).
    # Auto-dispatches: cuML on GPU, scipy KDTree on CPU.
    assignments, _ = knn(seeds, cell_centroids, k=1)
    assignments = assignments.squeeze(1)

    ### Accumulate areas per cluster
    cluster_areas = scatter_aggregate(
        cell_areas,
        assignments,
        n_seeds,
        aggregation="sum",
    )

    ### Accumulate area-weighted normals, then normalize to unit length
    if has_normals:
        cluster_normals = scatter_aggregate(
            mesh.cell_normals,
            assignments,
            n_seeds,
            weights=cell_areas,
            aggregation="sum",
        )
        cluster_normals = F.normalize(cluster_normals, dim=-1)
    else:
        cluster_normals = torch.zeros(n_seeds, n_dims, dtype=dtype, device=device)

    ### Accumulate area-weighted centroids (weighted mean, with seed fallback)
    cluster_centroids = scatter_aggregate(
        cell_centroids,
        assignments,
        n_seeds,
        weights=cell_areas,
        aggregation="mean",
    )
    cluster_centroids[cluster_areas == 0] = seeds[cluster_areas == 0]

    return CellPartition(
        assignments=assignments,
        cluster_areas=cluster_areas,
        cluster_normals=cluster_normals,
        cluster_centroids=cluster_centroids,
    )
