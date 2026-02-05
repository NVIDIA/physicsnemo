# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
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

"""Spatial dimension embedding and projection operations."""

import torch

from physicsnemo.mesh.mesh import Mesh
from physicsnemo.mesh.utilities._cache import CACHE_KEY


def embed_in_spatial_dims(
    mesh: Mesh,
    target_n_spatial_dims: int,
) -> Mesh:
    """Embed or project a mesh to a different number of spatial dimensions.

    This operation changes the spatial dimensionality of the mesh by adding or
    removing dimensions from the points array, while preserving the manifold
    structure and topology. New dimensions are appended to the end and initialized
    to zero. When projecting down, trailing dimensions are discarded.

    This is analogous to numpy.expand_dims() for the points array, but handles
    the full mesh structure including data fields and cached properties.

    Key behaviors:
        - Manifold dimension (n_manifold_dims) is preserved
        - Topology (cells connectivity) is preserved
        - Point/cell/global data are preserved as-is
        - Cached geometric properties are cleared (they depend on spatial embedding)

    Examples of use cases:
        - [2, 2] → [2, 3]: Embed 2D surface in 2D space into 3D space
        - [1, 3] → [1, 2]: Project 3D curve down to 2D plane
        - [2, 3] → [2, 4]: Embed 3D surface into 4D space

    Parameters
    ----------
    mesh : Mesh
        Input mesh to embed/project
    target_n_spatial_dims : int
        Target number of spatial dimensions. Must be >= 1.
        - If target > current: Points are padded with zeros in new dimensions
        - If target < current: Points are sliced to keep only first 'target' dims
        - If target == current: Returns mesh unchanged (no-op)

    Returns
    -------
    Mesh
        New mesh with modified spatial dimensions:
        - points shape: (n_points, target_n_spatial_dims)
        - n_manifold_dims: unchanged
        - cells: unchanged
        - point_data, cell_data: preserved (non-cached fields only)
        - Cached geometric properties: cleared (depend on spatial embedding)

    Raises
    ------
    ValueError
        If target_n_spatial_dims < 1
    ValueError
        If target_n_spatial_dims < n_manifold_dims (would create
        impossible configuration where manifold exceeds ambient space)

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.mesh import Mesh
    >>> # Embed 2D triangle mesh in 2D space into 3D space
    >>> points_2d = torch.tensor([[0., 0.], [1., 0.], [0., 1.]])
    >>> cells = torch.tensor([[0, 1, 2]])
    >>> mesh_2d = Mesh(points=points_2d, cells=cells)
    >>> assert mesh_2d.n_spatial_dims == 2
    >>>
    >>> # Embed in 3D (points become [x, y, 0])
    >>> mesh_3d = embed_in_spatial_dims(mesh_2d, target_n_spatial_dims=3)
    >>> assert mesh_3d.n_spatial_dims == 3
    >>> assert mesh_3d.points.shape == (3, 3)
    >>> assert torch.allclose(mesh_3d.points[0], torch.tensor([0., 0., 0.]))
    >>>
    >>> # Project back to 2D
    >>> mesh_2d_again = embed_in_spatial_dims(mesh_3d, target_n_spatial_dims=2)
    >>> assert torch.allclose(mesh_2d_again.points, points_2d)
    >>>
    >>> # Codimension changes affect normal computation
    >>> assert mesh_2d.codimension == 0  # no normals defined
    >>> assert mesh_3d.codimension == 1  # normals now defined!
    >>> assert mesh_3d.cell_normals.shape == (1, 3)

    Notes
    -----
    When spatial dimensions change, all cached geometric properties are cleared
    because they depend on the spatial embedding. This includes:
    - Cell/point normals (codimension changes)
    - Cell centroids (need padding/slicing)
    - Cell areas (intrinsically unchanged but cache is cleared for consistency)
    - Curvature values (depend on embedding)

    User data in point_data and cell_data is preserved as-is. If you have
    vector fields that should be padded/projected, you must handle this manually.
    """
    ### Validate inputs
    if target_n_spatial_dims < 1:
        raise ValueError(
            f"target_n_spatial_dims must be >= 1, got {target_n_spatial_dims=}"
        )

    if target_n_spatial_dims < mesh.n_manifold_dims:
        raise ValueError(
            f"Cannot embed {mesh.n_manifold_dims=}-dimensional manifold in "
            f"{target_n_spatial_dims=}-dimensional space.\n"
            f"Spatial dimensions must be >= manifold dimensions."
        )

    current_n_spatial_dims = mesh.n_spatial_dims

    ### Short-circuit if no change needed
    if target_n_spatial_dims == current_n_spatial_dims:
        return mesh

    ### Modify points array
    if target_n_spatial_dims > current_n_spatial_dims:
        # Pad with zeros in new dimensions (append to end)
        n_new_dims = target_n_spatial_dims - current_n_spatial_dims
        new_points = torch.nn.functional.pad(
            mesh.points,
            (0, n_new_dims),  # Pad last dimension
            mode="constant",
            value=0.0,
        )
    else:  # target_n_spatial_dims < current_n_spatial_dims
        # Slice to keep only first target_n_spatial_dims dimensions
        new_points = mesh.points[:, :target_n_spatial_dims]

    ### Preserve cells (topology unchanged)
    new_cells = mesh.cells

    ### Preserve user data, but clear cached properties
    # Cached properties depend on spatial embedding and must be recomputed
    new_point_data = mesh.point_data.exclude(CACHE_KEY)
    new_cell_data = mesh.cell_data.exclude(CACHE_KEY)
    new_global_data = mesh.global_data  # Global data is preserved as-is

    ### Create new mesh with modified spatial dimensions
    return Mesh(
        points=new_points,
        cells=new_cells,
        point_data=new_point_data,
        cell_data=new_cell_data,
        global_data=new_global_data,
    )
