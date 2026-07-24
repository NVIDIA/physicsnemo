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

"""General mesh topology utilities."""

from typing import TYPE_CHECKING

import torch
from jaxtyping import Int
from tensordict import TensorDict

from physicsnemo.mesh.boundaries._facet_extraction import extract_candidate_facets
from physicsnemo.utils._index_tuple_ops import unique_index_tuples

if TYPE_CHECKING:
    from physicsnemo.mesh.mesh import Mesh


def extract_unique_edges(
    mesh: "Mesh",
) -> tuple[Int[torch.Tensor, "n_edges 2"], Int[torch.Tensor, " n_candidates"]]:
    """Extract all unique edges from the mesh.

    For 1D meshes (cells are edges), the cells are deduplicated directly.
    For higher-dimensional meshes, edges are extracted via
    :func:`extract_candidate_facets` at the appropriate codimension.

    Parameters
    ----------
    mesh : Mesh
        Input mesh to extract edges from.

    Returns
    -------
    unique_edges : torch.Tensor
        Unique edge vertex indices, shape (n_edges, 2), canonically sorted
        so that ``unique_edges[:, 0] < unique_edges[:, 1]``.
    inverse_indices : torch.Tensor
        Mapping from candidate edges to unique edge indices.
        For 1D meshes, shape is (n_cells,).
        For n-manifolds with n > 1, shape is
        (n_cells * n_edges_per_cell,), which can be reshaped to
        (n_cells, n_edges_per_cell).

    Examples
    --------
    >>> from physicsnemo.mesh.primitives.basic import two_triangles_2d
    >>> triangle_mesh = two_triangles_2d.load()
    >>> edges, inverse = extract_unique_edges(triangle_mesh)
    >>> edges.shape[1]
    2
    """
    if mesh.n_manifold_dims == 0:
        # A 0-manifold has vertices but no one-dimensional simplices.  Keep the
        # edge shape stable so downstream DEC code can handle point clouds and
        # empty meshes without special-casing malformed ``(0, 0)`` tensors.
        return (
            torch.empty((0, 2), dtype=mesh.cells.dtype, device=mesh.cells.device),
            torch.empty((0,), dtype=torch.long, device=mesh.cells.device),
        )

    if mesh.n_manifold_dims == 1:
        ### 1D meshes: cells ARE edges - sort and deduplicate directly
        sorted_cells = torch.sort(mesh.cells, dim=1)[0]
        unique_edges, inverse_indices = unique_index_tuples(
            sorted_cells,
            index_bound=mesh.n_points,
            return_inverse=True,
        )
        return unique_edges, inverse_indices

    ### General case: extract edges as (n-1)-codimension facets of each cell
    candidate_edges, _parent_cell_indices = extract_candidate_facets(
        mesh.cells,
        manifold_codimension=mesh.n_manifold_dims - 1,
    )
    unique_edges, inverse_indices = unique_index_tuples(
        candidate_edges,
        index_bound=mesh.n_points,
        return_inverse=True,
    )
    return unique_edges, inverse_indices


def get_or_compute_unique_edges(
    mesh: "Mesh",
) -> tuple[Int[torch.Tensor, "n_edges 2"], Int[torch.Tensor, " n_candidates"]]:
    """Get cached unique edges and their candidate-to-edge mapping.

    Edge connectivity depends only on ``mesh.cells``, so both tensors live in
    the topology cache and remain valid across geometric transformations.  A
    legacy serialized mesh may not have a ``"topology"`` cache; it is created
    lazily in that case.

    Tensors created inside :func:`torch.inference_mode` cannot be saved by a
    later autograd operation.  When execution returns to grad-enabled mode,
    inference tensors are therefore recomputed and replaced by ordinary
    tensors before a DEC functional can save the edge indices for backward.

    Parameters
    ----------
    mesh : Mesh
        Input mesh.

    Returns
    -------
    unique_edges : torch.Tensor
        Canonically sorted unique edges, shape ``(n_edges, 2)``.
    inverse_indices : torch.Tensor
        Candidate-edge to unique-edge mapping.
    """
    topology_cache = mesh._cache.get("topology", None)
    if topology_cache is None:
        topology_cache = TensorDict({}, device=mesh.points.device)
        mesh._cache["topology"] = topology_cache

    unique_edges = topology_cache.get("unique_edges", None)
    inverse_indices = topology_cache.get("unique_edge_inverse_indices", None)
    needs_refresh = unique_edges is None or inverse_indices is None
    if torch.is_grad_enabled() and not needs_refresh:
        needs_refresh = unique_edges.is_inference() or inverse_indices.is_inference()

    if needs_refresh:
        unique_edges, inverse_indices = extract_unique_edges(mesh)
        topology_cache["unique_edges"] = unique_edges
        topology_cache["unique_edge_inverse_indices"] = inverse_indices

    return unique_edges, inverse_indices
