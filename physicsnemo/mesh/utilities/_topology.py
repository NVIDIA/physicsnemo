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

from physicsnemo.mesh.boundaries._facet_extraction import extract_candidate_facets

if TYPE_CHECKING:
    from physicsnemo.mesh.mesh import Mesh


def extract_unique_edges(mesh: "Mesh") -> tuple[torch.Tensor, torch.Tensor]:
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
    if mesh.n_manifold_dims == 1:
        ### 1D meshes: cells ARE edges - sort and deduplicate directly
        sorted_cells = torch.sort(mesh.cells, dim=1)[0]
        return torch.unique(sorted_cells, dim=0, return_inverse=True)

    ### General case: extract edges as (n-1)-codimension facets of each cell
    candidate_edges, _parent_cell_indices = extract_candidate_facets(
        mesh.cells,
        manifold_codimension=mesh.n_manifold_dims - 1,
    )
    return torch.unique(candidate_edges, dim=0, return_inverse=True)
