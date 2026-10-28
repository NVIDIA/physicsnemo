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

"""Weighted least-squares gradient reconstruction for unstructured meshes.

This implements the standard CFD approach for computing gradients on irregular
meshes using weighted least-squares fitting.

The method solves for the gradient that best fits the function differences
to neighboring points/cells, weighted by inverse distance.

Reference: Standard in CFD literature (Barth & Jespersen, AIAA 1989)
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import torch
from jaxtyping import Float

if TYPE_CHECKING:
    from physicsnemo.mesh.mesh import Mesh
    from physicsnemo.mesh.neighbors import Adjacency


def _lsq_gradients(
    points: Float[torch.Tensor, "n_entities n_spatial_dims"],
    values: Sequence[Float[torch.Tensor, "n_entities ..."]],
    adjacency: "Adjacency",
    weight_power: float,
    min_neighbors: int,
    implementation: Literal["torch", "warp"] | None,
    neighbors_span: bool,
) -> list[Float[torch.Tensor, "n_entities n_spatial_dims ..."]]:
    """LSQ gradients of several fields of one dtype, on the chosen backend.

    ``None`` picks Warp, which is faster, where it gives PyTorch's answer:
    ``float32`` data in one to three dimensions, where every neighborhood
    spans the space (``neighbors_span``). Where the neighbors do not determine
    every gradient component, Warp returns zero and PyTorch the minimum-norm
    solution. The ill-conditioned systems of curves and surfaces in a
    higher-dimensional space also lose accuracy in Warp's ``float32`` normal
    equations.
    """
    from physicsnemo.mesh.calculus._lsq_gradient import (
        MeshLSQGradient,
        mesh_lsq_gradient,
    )

    if implementation is None:
        use_warp = (
            neighbors_span
            and "warp" in MeshLSQGradient.available_implementations()
            and points.device.type in ("cpu", "cuda")
            and 0 < points.shape[0]
            and points.shape[1] <= 3
            and torch.float64 not in {points.dtype, *(v.dtype for v in values)}
        )
        implementation = "warp" if use_warp else "torch"
    elif implementation not in ("torch", "warp"):
        raise ValueError(
            f"Invalid {implementation=!r}. Must be 'torch', 'warp', or None."
        )

    if implementation == "torch":
        from physicsnemo.mesh.calculus._lsq_gradient._torch_impl import (
            _mesh_lsq_gradients_torch,
        )

        return _mesh_lsq_gradients_torch(
            points=points,
            values=values,
            neighbor_offsets=adjacency.offsets,
            neighbor_indices=adjacency.indices,
            weight_power=weight_power,
            min_neighbors=min_neighbors,
        )

    ### One Warp call for all fields: flatten, stack, and split the result.
    flat = [v.reshape(v.shape[0], -1) for v in values]
    gradients = mesh_lsq_gradient(
        points,
        torch.cat(flat, dim=1),
        adjacency.offsets,
        adjacency.indices,
        weight_power=weight_power,
        min_neighbors=min_neighbors,
        implementation="warp",
    )
    return [
        g.reshape(*g.shape[:2], *v.shape[1:])
        for g, v in zip(gradients.split([f.shape[1] for f in flat], dim=2), values)
    ]


def compute_point_gradient_lsq(
    mesh: "Mesh",
    point_values: Float[torch.Tensor, "n_points ..."],
    weight_power: float = 2.0,
    min_neighbors: int = 0,
) -> Float[torch.Tensor, "n_points n_spatial_dims ..."]:
    r"""Compute gradient at vertices using weighted least-squares reconstruction.

    For each vertex with center :math:`x_0` and value :math:`\varphi_0`, and
    a neighborhood :math:`N = \{x_1, \ldots, x_k\}` with values
    :math:`\varphi_1, \ldots, \varphi_k`, solves

    .. math::

        \min_{\nabla \varphi} \sum_{i = 1}^{k}
            w_i \, \bigl\| \nabla \varphi \cdot (x_i - x_0)
                          - (\varphi_i - \varphi_0) \bigr\|^2,

    with weights :math:`w_i = 1 / \|x_i - x_0\|^{\alpha}` (typically
    :math:`\alpha = 2`).

    Parameters
    ----------
    mesh : Mesh
        Simplicial mesh.
    point_values : Float[torch.Tensor, "n_points ..."]
        Values at vertices.
    weight_power : float
        Exponent :math:`\alpha` for inverse-distance weighting (default 2.0).
    min_neighbors : int
        Minimum neighbors required for gradient computation. Points with
        fewer neighbors get zero gradients. The default of 0 means all
        points are processed: ``lstsq`` naturally returns the minimum-norm
        solution for under-determined systems (fewer neighbors than spatial
        dims) and zero for isolated points with no neighbors.

    Returns
    -------
    Float[torch.Tensor, "n_points n_spatial_dims ..."]
        Gradients at vertices, shape ``(n_points, n_spatial_dims, ...)``.
        The derivative-coordinate axis is always axis 1. For a vector field,
        ``gradient[i, k, j]`` is :math:`\partial v_j / \partial x_k`.

    Notes
    -----
    Solves the weighted least-squares system
    :math:`(A^\top W A) \, \nabla \varphi = A^\top W b`, where, with
    :math:`k` neighbors and :math:`d` spatial dimensions,

    .. math::

        A &= [x_1 - x_0, x_2 - x_0, \ldots, x_k - x_0]^\top
            \quad (k \times d), \\
        b &= [\varphi_1 - \varphi_0, \varphi_2 - \varphi_0, \ldots,
              \varphi_k - \varphi_0]^\top
            \quad (k,), \\
        W &= \operatorname{diag}(w_1, w_2, \ldots, w_k)
            \quad (k \times k).

    Here ``k`` is the number of neighbors of the point and ``d`` equals
    ``n_spatial_dims``. Fully vectorized using batched operations. The
    systems of all points are assembled and solved at once where the normal
    equations solve them (float32 and 16-bit); otherwise points are grouped by
    neighbor count and processed in parallel to handle the ragged neighbor
    structure.
    """
    return _compute_point_gradients_lsq(
        mesh, [point_values], weight_power, min_neighbors
    )[0]


def _compute_point_gradients_lsq(
    mesh: "Mesh",
    point_values: Sequence[Float[torch.Tensor, "n_points ..."]],
    weight_power: float = 2.0,
    min_neighbors: int = 0,
    adjacency: "Adjacency | None" = None,
    implementation: Literal["torch", "warp"] | None = None,
) -> list[Float[torch.Tensor, "n_points n_spatial_dims ..."]]:
    """Gradients of several fields of one dtype at vertices, by
    :func:`compute_point_gradient_lsq`, sharing their LSQ systems."""
    # A vertex of a cell that fills the space has neighbors spanning it.
    neighbors_span = adjacency is None and mesh.n_manifold_dims == mesh.n_spatial_dims
    if adjacency is None:
        adjacency = mesh.get_point_to_points_adjacency()
    return _lsq_gradients(
        mesh.points,
        point_values,
        adjacency,
        weight_power,
        min_neighbors,
        implementation,
        neighbors_span,
    )


def compute_cell_gradient_lsq(
    mesh: "Mesh",
    cell_values: Float[torch.Tensor, "n_cells ..."],
    weight_power: float = 2.0,
) -> Float[torch.Tensor, "n_cells n_spatial_dims ..."]:
    r"""Compute gradient at cells using weighted least-squares reconstruction.

    Uses cell-to-cell adjacency to build a LSQ system around each cell centroid.

    Parameters
    ----------
    mesh : Mesh
        Simplicial mesh.
    cell_values : Float[torch.Tensor, "n_cells ..."]
        Values at cells.
    weight_power : float
        Exponent for inverse-distance weighting (default 2.0).

    Returns
    -------
    Float[torch.Tensor, "n_cells n_spatial_dims ..."]
        Gradients at cells, shape ``(n_cells, n_spatial_dims, ...)``.
        The derivative-coordinate axis is always axis 1. For a vector field,
        ``gradient[i, k, j]`` is :math:`\partial v_j / \partial x_k`.

    Notes
    -----
    Fully vectorized using batched operations. The systems of all cells are
    assembled and solved at once where the normal equations solve them
    (float32 and 16-bit); otherwise cells are grouped by neighbor count and
    processed in parallel.
    """
    return _compute_cell_gradients_lsq(mesh, [cell_values], weight_power)[0]


def _compute_cell_gradients_lsq(
    mesh: "Mesh",
    cell_values: Sequence[Float[torch.Tensor, "n_cells ..."]],
    weight_power: float = 2.0,
    implementation: Literal["torch", "warp"] | None = None,
) -> list[Float[torch.Tensor, "n_cells n_spatial_dims ..."]]:
    """Gradients of several fields of one dtype at cells, by
    :func:`compute_cell_gradient_lsq`, sharing their LSQ systems."""
    return _lsq_gradients(
        mesh.cell_centroids,
        cell_values,
        mesh.get_cell_to_cells_adjacency(adjacency_codimension=1),
        weight_power,
        0,  # Cells may have fewer neighbors than points.
        implementation,
        neighbors_span=False,  # A boundary cell may have just one neighbor.
    )
