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
    generate_child_cells,
    get_subdivision_pattern,
)
from physicsnemo.mesh.utilities._topology import extract_unique_edges

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

    ### Get subdivision pattern for this manifold dimension
    subdivision_pattern = get_subdivision_pattern(mesh.n_manifold_dims)
    # non_blocking: a small pageable host table, staged by CUDA before returning
    subdivision_pattern = subdivision_pattern.to(mesh.cells.device, non_blocking=True)
    n_children_per_parent = subdivision_pattern.shape[0]

    ### Generate child cells from parents
    # Done first so edge_inverse can be freed before the data is built.
    child_cells = generate_child_cells(
        parent_cells=mesh.cells,
        edge_inverse=edge_inverse,
        n_original_points=n_original_points,
        subdivision_pattern=subdivision_pattern,
    )
    del edge_inverse

    ### Create new points array: original + edge midpoints
    # Shape: (n_original_points + n_edges, n_spatial_dims). The (n_edges, 2,
    # n_spatial_dims) endpoint gather is a temporary of the mean.
    new_points = torch.cat([mesh.points, mesh.points[unique_edges].mean(dim=1)], dim=0)

    ### Interpolate point_data to edge midpoints
    new_point_data = interpolate_point_data_to_edges(
        point_data=mesh.point_data,
        edges=unique_edges,
        n_original_points=n_original_points,
    )
    del unique_edges

    ### Propagate cell_data from parents to children
    new_cell_data = propagate_cell_data_to_children(
        cell_data=mesh.cell_data,
        n_children_per_parent=n_children_per_parent,
        device=mesh.cells.device,
    )

    ### Create and return subdivided mesh
    result = Mesh(
        points=new_points,
        cells=child_cells,
        point_data=new_point_data,
        cell_data=new_cell_data,
        global_data=mesh.global_data.copy(),  # Shared tensors, own container
    )

    from physicsnemo.mesh.calculus.measure import (
        EFFECTIVE_MEASURE_KEY,
        _transfer_cell_measures,
    )

    if EFFECTIVE_MEASURE_KEY in mesh.cell_data:
        # Only explicit measures need each child's parent index
        parent_indices = torch.arange(
            mesh.n_cells, device=mesh.cells.device
        ).repeat_interleave(n_children_per_parent)
        _transfer_cell_measures(mesh, result, parent_indices)
    return result
