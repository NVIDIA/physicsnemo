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

"""Linear subdivision for simplicial meshes.

Linear subdivision is the simplest subdivision scheme: each edge is split at
its midpoint, and each n-simplex is divided into 2^n smaller simplices.
This is an interpolating scheme - original vertices remain unchanged.

Works for any manifold dimension and any spatial dimension (including higher
codimensions like curves in 3D or surfaces in 4D).
"""

from typing import TYPE_CHECKING

import torch

from physicsnemo.mesh.subdivision._data import (
    interpolate_point_data_to_edges,
    propagate_cell_data_to_children,
)
from physicsnemo.mesh.subdivision._topology import (
    extract_unique_edges,
    generate_child_cells,
    get_subdivision_pattern,
)

if TYPE_CHECKING:
    from physicsnemo.mesh.mesh import Mesh


def subdivide_linear(mesh: "Mesh") -> "Mesh":
    """Perform one level of linear subdivision on the mesh.

    Linear subdivision splits each n-simplex into 2^n child simplices by:
    1. Adding new vertices at edge midpoints
    2. Connecting vertices according to a subdivision pattern

    This is an interpolating scheme: original vertices keep their positions,
    and new vertices are placed exactly at edge midpoints.

    Properties:
    - Preserves manifold dimension and spatial dimension
    - Increases mesh resolution uniformly
    - Point data is interpolated to new vertices (averaged from endpoints)
    - Cell data is propagated to children (each child inherits parent's data)
    - Global data is preserved unchanged

    Parameters
    ----------
    mesh : Mesh
        Input mesh to subdivide (any manifold/spatial dimension)

    Returns
    -------
    Mesh
        Subdivided mesh with:
        - n_points = original_n_points + n_edges
        - n_cells = original_n_cells * 2^n_manifold_dims

    Examples
    --------
        >>> from physicsnemo.mesh.primitives.basic import two_triangles_2d
        >>> # Triangle mesh: 2 triangles -> 8 triangles
        >>> mesh = two_triangles_2d.load()
        >>> subdivided = subdivide_linear(mesh)
        >>> assert subdivided.n_cells == mesh.n_cells * 4  # 2^2 for 2D
    """
    from physicsnemo.mesh.mesh import Mesh

    ### Handle empty mesh
    if mesh.n_cells == 0:
        return mesh

    ### Extract unique edges from mesh
    unique_edges, edge_inverse = extract_unique_edges(mesh)
    n_original_points = mesh.n_points

    ### Compute edge midpoints
    # Shape: (n_edges, n_spatial_dims)
    edge_vertices = mesh.points[unique_edges]  # (n_edges, 2, n_spatial_dims)
    edge_midpoints = edge_vertices.mean(dim=1)  # Average the two endpoints

    ### Create new points array: original + midpoints
    # Shape: (n_original_points + n_edges, n_spatial_dims)
    new_points = torch.cat([mesh.points, edge_midpoints], dim=0)

    ### Interpolate point_data to edge midpoints
    new_point_data = interpolate_point_data_to_edges(
        point_data=mesh.point_data,
        edges=unique_edges,
        n_original_points=n_original_points,
    )

    ### Get subdivision pattern for this manifold dimension
    subdivision_pattern = get_subdivision_pattern(mesh.n_manifold_dims)
    subdivision_pattern = subdivision_pattern.to(mesh.cells.device)

    ### Generate child cells from parents
    child_cells, parent_indices = generate_child_cells(
        parent_cells=mesh.cells,
        edge_inverse=edge_inverse,
        n_original_points=n_original_points,
        subdivision_pattern=subdivision_pattern,
    )

    ### Propagate cell_data from parents to children
    new_cell_data = propagate_cell_data_to_children(
        cell_data=mesh.cell_data,
        parent_indices=parent_indices,
        n_total_children=len(child_cells),
    )

    ### Create and return subdivided mesh
    return Mesh(
        points=new_points,
        cells=child_cells,
        point_data=new_point_data,
        cell_data=new_cell_data,
        global_data=mesh.global_data,  # Preserved unchanged
    )
