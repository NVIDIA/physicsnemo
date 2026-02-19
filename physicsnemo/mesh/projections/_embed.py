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

"""Spatial dimension embedding operations."""

import torch

from physicsnemo.mesh.mesh import Mesh


def embed(
    mesh: Mesh,
    target_n_spatial_dims: int,
    *,
    insert_at: int | None = None,
) -> Mesh:
    """Embed a mesh into a higher-dimensional ambient space.

    Increases the spatial dimensionality of a mesh by inserting new zero-valued
    coordinate dimensions, while preserving the manifold structure and topology.
    This operation is non-destructive: the original coordinate values are retained
    in their respective dimensions.

    Key behaviors:
        - Manifold dimension (n_manifold_dims) is preserved
        - Topology (cell connectivity) is preserved
        - Point/cell/global data are preserved as-is
        - Cached geometric properties are cleared (they depend on spatial embedding)

    Use cases:
        - [2, 2] -> [2, 3]: Embed a flat 2D surface into 3D space
          (e.g., to enable surface normal computation via codimension-1)
        - [1, 2] -> [1, 3]: Embed a 2D curve into 3D space
        - [2, 3] -> [2, 4]: Embed a 3D surface into 4D space

    Parameters
    ----------
    mesh : Mesh
        Input mesh to embed.
    target_n_spatial_dims : int
        Target number of spatial dimensions. Must be >= current n_spatial_dims.
        If equal to current, returns mesh unchanged (no-op).
    insert_at : int | None, optional
        Index at which to insert the new zero-valued dimensions. The new
        dimensions form a contiguous block starting at this position, with
        semantics matching ``list.insert``. Valid range is
        ``0 <= insert_at <= n_current_spatial_dims``.

        - ``None`` (default): append new dimensions at the end.
          ``[x, y] -> [x, y, 0]``
        - ``0``: prepend new dimensions at the start.
          ``[x, y] -> [0, x, y]``
        - ``1``: insert new dimensions after the first coordinate.
          ``[x, y] -> [x, 0, y]``

    Returns
    -------
    Mesh
        New mesh with increased spatial dimensions:
        - points shape: ``(n_points, target_n_spatial_dims)``
        - n_manifold_dims: unchanged
        - cells: unchanged
        - point_data, cell_data: preserved (non-cached fields only)
        - Cached geometric properties: cleared

    Raises
    ------
    ValueError
        If ``target_n_spatial_dims < 1``.
    ValueError
        If ``target_n_spatial_dims < current n_spatial_dims`` (use
        :func:`project` to reduce dimensions).
    ValueError
        If ``insert_at`` is out of the valid range ``[0, n_spatial_dims]``.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.mesh import Mesh
    >>> from physicsnemo.mesh.projections import embed
    >>> points_2d = torch.tensor([[0., 0.], [1., 0.], [0., 1.]])
    >>> cells = torch.tensor([[0, 1, 2]])
    >>> mesh_2d = Mesh(points=points_2d, cells=cells)
    >>>
    >>> # Embed in 3D (default: append z=0)
    >>> mesh_3d = embed(mesh_2d, target_n_spatial_dims=3)
    >>> assert mesh_3d.points.shape == (3, 3)
    >>> assert torch.allclose(mesh_3d.points[:, 2], torch.zeros(3))
    >>>
    >>> # Embed with new dimension inserted at position 1: [x, y] -> [x, 0, y]
    >>> mesh_3d_mid = embed(mesh_2d, target_n_spatial_dims=3, insert_at=1)
    >>> assert torch.allclose(mesh_3d_mid.points[:, 0], points_2d[:, 0])
    >>> assert torch.allclose(mesh_3d_mid.points[:, 1], torch.zeros(3))
    >>> assert torch.allclose(mesh_3d_mid.points[:, 2], points_2d[:, 1])
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
    - Cell centroids (coordinate count changes)
    - Cell areas (intrinsically unchanged but cache is cleared for consistency)
    - Curvature values (depend on embedding)

    User data in ``point_data`` and ``cell_data`` is preserved as-is. If you have
    vector fields that should also be padded, you must handle this manually.

    See Also
    --------
    project : The inverse operation - reduce spatial dimensions.
    """
    ### Validate inputs
    if target_n_spatial_dims < 1:
        raise ValueError(
            f"target_n_spatial_dims must be >= 1, got {target_n_spatial_dims=}"
        )

    current_n_spatial_dims = mesh.n_spatial_dims

    if target_n_spatial_dims < current_n_spatial_dims:
        raise ValueError(
            f"Cannot embed: {target_n_spatial_dims=} < current "
            f"{current_n_spatial_dims=}. Use project() to reduce spatial dimensions."
        )

    ### Short-circuit if no change needed
    if target_n_spatial_dims == current_n_spatial_dims:
        return mesh

    n_new_dims = target_n_spatial_dims - current_n_spatial_dims

    ### Validate insert_at
    if insert_at is not None and not (0 <= insert_at <= current_n_spatial_dims):
        raise ValueError(
            f"insert_at must be in [0, {current_n_spatial_dims}], got {insert_at=}"
        )

    ### Construct new points array
    if insert_at is None or insert_at == current_n_spatial_dims:
        # Append zeros at end (fast path)
        new_points = torch.nn.functional.pad(
            mesh.points, (0, n_new_dims), mode="constant", value=0.0
        )
    elif insert_at == 0:
        # Prepend zeros at start (fast path)
        new_points = torch.nn.functional.pad(
            mesh.points, (n_new_dims, 0), mode="constant", value=0.0
        )
    else:
        # Insert zeros at arbitrary interior position
        prefix = mesh.points[:, :insert_at]
        zeros = torch.zeros(
            mesh.n_points,
            n_new_dims,
            dtype=mesh.points.dtype,
            device=mesh.points.device,
        )
        suffix = mesh.points[:, insert_at:]
        new_points = torch.cat([prefix, zeros, suffix], dim=1)

    ### Preserve cells (topology unchanged)
    new_cells = mesh.cells

    ### Preserve user data, but clear cached properties
    # Cached properties depend on spatial embedding and must be recomputed
    new_point_data = mesh.point_data
    new_cell_data = mesh.cell_data
    new_global_data = mesh.global_data

    ### Create new mesh with modified spatial dimensions
    return Mesh(
        points=new_points,
        cells=new_cells,
        point_data=new_point_data,
        cell_data=new_cell_data,
        global_data=new_global_data,
    )
