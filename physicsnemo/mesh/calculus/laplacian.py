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

r"""Laplacian operators for fields on simplicial meshes.

This module provides two complementary discretizations:

- :func:`compute_laplacian_points_dec` is an intrinsic cotangent
  Laplace--Beltrami operator based on Discrete Exterior Calculus (DEC).
- :func:`compute_laplacian_points_lsq` and
  :func:`compute_laplacian_cells_lsq` are extrinsic double least-squares
  operators: they estimate a gradient in ambient coordinates, then estimate
  its divergence.

The DEC implementation uses the analyst's sign convention,

.. math::

    \Delta f(v_0) = \frac{1}{|{\star}v_0|}
        \sum_{\text{edges from } v_0}
            \frac{|{\star}e|}{|e|} \, \bigl(f(v) - f(v_0)\bigr),

which is positive for locally convex functions (e.g. :math:`\Delta(x^2) = 2`).

For functions (0-forms), the cotangent formula is intrinsic to the manifold
and reduces to the standard Laplacian on a flat domain.  The LSQ formulation
instead differentiates in the embedding space, which is useful for general
point and cell fields but is not an intrinsic surface Laplacian.
"""

from typing import TYPE_CHECKING, Literal

import torch
from jaxtyping import Float, Int

if TYPE_CHECKING:
    from physicsnemo.mesh.mesh import Mesh


def _apply_cotan_laplacian_operator(
    n_points: int,
    edges: Int[torch.Tensor, "n_edges 2"],
    cotan_weights: Float[torch.Tensor, " n_edges"],
    data: Float[torch.Tensor, "n_points ..."],
) -> Float[torch.Tensor, "n_points ..."]:
    r"""Apply cotangent Laplacian operator to data via scatter-add.

    For data :math:`f` indexed by vertex with neighborhood :math:`N(i)`,
    computes

    .. math::

        (L f)_i = \sum_{j \in N(i)} w_{ij} \, (f_j - f_i).

    This is the core scatter-add pattern shared by all cotangent Laplacian
    computations. Used by :func:`compute_laplacian_points_dec` for scalar
    fields and by ``compute_laplacian_at_points`` in the curvature module for
    point coordinates.

    Parameters
    ----------
    n_points : int
        Number of points (vertices).
    edges : Int[torch.Tensor, "n_edges 2"]
        Edge connectivity.
    cotan_weights : Float[torch.Tensor, " n_edges"]
        Cotangent weight for each edge.
    data : Float[torch.Tensor, "n_points ..."]
        Data at points.

    Returns
    -------
    Float[torch.Tensor, "n_points ..."]
        Laplacian applied to data, same shape as ``data``.

    Examples
    --------
    >>> import torch
    >>> # For scalar field
    >>> n_points, edges = 4, torch.tensor([[0, 1], [1, 2], [0, 2]])
    >>> weights = torch.ones(3)
    >>> scalar_field = torch.randn(4)
    >>> laplacian = _apply_cotan_laplacian_operator(n_points, edges, weights, scalar_field)
    """
    ### Initialize output with same shape as data
    device = data.device
    if data.ndim == 1:
        laplacian = torch.zeros(n_points, dtype=data.dtype, device=device)
    else:
        laplacian = torch.zeros_like(data)

    ### Extract vertex indices
    v0_indices = edges[:, 0]  # (n_edges,)
    v1_indices = edges[:, 1]  # (n_edges,)

    ### Compute weighted differences
    if data.ndim == 1:
        # Scalar case
        contrib_v0 = cotan_weights * (data[v1_indices] - data[v0_indices])
        contrib_v1 = cotan_weights * (data[v0_indices] - data[v1_indices])
        laplacian.scatter_add_(0, v0_indices, contrib_v0)
        laplacian.scatter_add_(0, v1_indices, contrib_v1)
    else:
        # Multi-dimensional case (vectors, tensors)
        # Broadcast weights to match data dimensions
        weights_expanded = cotan_weights.view(-1, *([1] * (data.ndim - 1)))
        contrib_v0 = weights_expanded * (data[v1_indices] - data[v0_indices])
        contrib_v1 = weights_expanded * (data[v0_indices] - data[v1_indices])

        # Flatten for scatter_add
        laplacian_flat = laplacian.reshape(n_points, -1)
        contrib_v0_flat = contrib_v0.reshape(len(edges), -1)
        contrib_v1_flat = contrib_v1.reshape(len(edges), -1)

        v0_expanded = v0_indices.unsqueeze(-1).expand(-1, contrib_v0_flat.shape[1])
        v1_expanded = v1_indices.unsqueeze(-1).expand(-1, contrib_v1_flat.shape[1])

        laplacian_flat.scatter_add_(0, v0_expanded, contrib_v0_flat)
        laplacian_flat.scatter_add_(0, v1_expanded, contrib_v1_flat)

        laplacian = laplacian_flat.reshape(laplacian.shape)

    return laplacian


def compute_laplacian_points_dec(
    mesh: "Mesh",
    point_values: Float[torch.Tensor, "n_points ..."],
    implementation: Literal["warp", "torch"] | None = "torch",
) -> Float[torch.Tensor, "n_points ..."]:
    r"""Compute Laplace-Beltrami at vertices using DEC cotangent formula.

    This is the **intrinsic** Laplacian - it automatically respects the
    manifold structure.

    .. math::

        \Delta f(v_0) = \frac{1}{|{\star}v_0|}
            \sum_{\text{edges from } v_0}
                \frac{|{\star}e|}{|e|} \, \bigl(f(v) - f(v_0)\bigr),

    where :math:`|{\star}v_0|` is the dual 0-cell volume (Voronoi cell around
    the vertex), :math:`|{\star}e|` is the dual 1-cell volume (dual to the
    edge), :math:`|e|` is the edge length, and the ratio
    :math:`|{\star}e| / |e|` is the cotangent weight.

    Parameters
    ----------
    mesh : Mesh
        Simplicial mesh.
    point_values : Float[torch.Tensor, "n_points ..."]
        Values at vertices.
    implementation : {"warp", "torch"} or None, optional
        Functional backend. Defaults to ``"torch"``.

    Returns
    -------
    Float[torch.Tensor, "n_points ..."]
        Laplacian at vertices, same shape as ``point_values``.
    """
    from physicsnemo.mesh.geometry.dual_meshes import (
        get_or_compute_cotan_weights_fem,
        get_or_compute_dual_volumes_0,
    )
    from physicsnemo.nn.functional.derivatives.mesh_cotan_laplacian import (
        mesh_cotan_laplacian,
    )

    ### Get cotangent weights and edges via FEM stiffness matrix (works for any dimension)
    cotan_weights, sorted_edges = get_or_compute_cotan_weights_fem(mesh)

    ### Normalize by Voronoi areas
    # Standard cotangent Laplacian: Δf_i = (1/A_voronoi_i) × accumulated_sum
    dual_volumes_0 = get_or_compute_dual_volumes_0(mesh)

    return mesh_cotan_laplacian(
        edges=sorted_edges,
        cotan_weights=cotan_weights,
        dual_volumes=dual_volumes_0,
        values=point_values,
        implementation=implementation,
    )


def compute_laplacian_points_lsq(
    mesh: "Mesh",
    point_values: Float[torch.Tensor, "n_points ..."],
    weight_power: float = 2.0,
    min_neighbors: int = 0,
    implementation: Literal["warp", "torch"] | None = "torch",
) -> Float[torch.Tensor, "n_points ..."]:
    r"""Compute an extrinsic double-LSQ Laplacian at vertices.

    First estimates the ambient gradient at every vertex by a weighted local
    least-squares fit, then applies the same LSQ differentiation to the
    gradient and takes its trace:

    .. math::

        \Delta f(x_i) \approx \operatorname{tr}\!\left[
        \operatorname{LSQGrad}\left(
        \operatorname{LSQGrad}(f)
        \right)_i\right].

    This is an **extrinsic** operator: derivatives are taken in the mesh's
    embedding coordinates.  On irregular or one-sided neighborhoods, applying
    LSQ twice can amplify first-derivative error; accuracy therefore depends on
    neighborhood quality and is generally lower near boundaries.  Use the DEC
    variant for an intrinsic Laplace--Beltrami operator on a simplicial surface.

    Parameters
    ----------
    mesh : Mesh
        Simplicial mesh.
    point_values : torch.Tensor
        Scalar or tensor values at vertices, shape ``(n_points, ...)``.
    weight_power : float, optional
        Exponent of inverse-distance neighbor weights. Defaults to ``2.0``.
    min_neighbors : int, optional
        Vertices with fewer neighbors receive zero. Defaults to ``0``.
    implementation : {"warp", "torch"} or None, optional
        Functional backend. Defaults to ``"torch"``.

    Returns
    -------
    torch.Tensor
        Estimated ambient Laplacian, with the same shape as ``point_values``.
    """
    from physicsnemo.nn.functional.derivatives.mesh_lsq_laplacian import (
        mesh_lsq_laplacian,
    )

    adjacency = mesh.get_point_to_points_adjacency()
    return mesh_lsq_laplacian(
        points=mesh.points,
        values=point_values,
        neighbor_offsets=adjacency.offsets,
        neighbor_indices=adjacency.indices,
        weight_power=weight_power,
        min_neighbors=min_neighbors,
        implementation=implementation,
    )


def compute_laplacian_cells_lsq(
    mesh: "Mesh",
    cell_values: Float[torch.Tensor, "n_cells ..."],
    weight_power: float = 2.0,
    min_neighbors: int = 0,
    implementation: Literal["warp", "torch"] | None = "torch",
) -> Float[torch.Tensor, "n_cells ..."]:
    r"""Compute an extrinsic double-LSQ Laplacian at cell centers.

    The operator estimates a weighted least-squares gradient over adjacent
    cell centroids, differentiates that gradient with a second LSQ fit, and
    takes the ambient trace:

    .. math::

        \Delta f(c_i) \approx \operatorname{tr}\!\left[
        \operatorname{LSQGrad}\left(
        \operatorname{LSQGrad}(f)
        \right)_i\right].

    This is an **extrinsic** approximation.  Its accuracy depends on the
    geometry and rank of the cell-neighbor stencil, and the second LSQ pass can
    amplify error on skewed meshes or near boundaries.  Cells with fewer than
    ``min_neighbors`` adjacent cells receive zero.

    Parameters
    ----------
    mesh : Mesh
        Simplicial mesh.
    cell_values : torch.Tensor
        Scalar or tensor values at cell centers, shape ``(n_cells, ...)``.
    weight_power : float, optional
        Exponent of inverse-distance neighbor weights. Defaults to ``2.0``.
    min_neighbors : int, optional
        Cells with fewer neighbors receive zero. Defaults to ``0``.
    implementation : {"warp", "torch"} or None, optional
        Functional backend. Defaults to ``"torch"``.

    Returns
    -------
    torch.Tensor
        Estimated ambient Laplacian, with the same shape as ``cell_values``.
    """
    from physicsnemo.nn.functional.derivatives.mesh_lsq_laplacian import (
        mesh_lsq_laplacian,
    )

    adjacency = mesh.get_cell_to_cells_adjacency(adjacency_codimension=1)
    return mesh_lsq_laplacian(
        points=mesh.cell_centroids,
        values=cell_values,
        neighbor_offsets=adjacency.offsets,
        neighbor_indices=adjacency.indices,
        weight_power=weight_power,
        min_neighbors=min_neighbors,
        implementation=implementation,
    )
