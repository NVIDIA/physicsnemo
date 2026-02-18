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

"""Spatial dimension projection operations."""

from collections.abc import Sequence

from physicsnemo.mesh.mesh import Mesh


def project(
    mesh: Mesh,
    target_n_spatial_dims: int | None = None,
    *,
    keep_dims: Sequence[int] | None = None,
) -> Mesh:
    """Project a mesh to a lower-dimensional ambient space.

    Reduces the spatial dimensionality of a mesh by discarding coordinate
    dimensions. The manifold structure and topology are preserved, but coordinate
    information in the removed dimensions is permanently lost.

    Two calling conventions are supported:
        - ``project(mesh, target_n_spatial_dims=N)``: keep the first N dimensions.
        - ``project(mesh, keep_dims=[i, j, ...])``: keep specific dimensions by
          index, in the order they should appear in the output.

    Exactly one of ``target_n_spatial_dims`` or ``keep_dims`` must be specified.

    Key behaviors:
        - Manifold dimension (n_manifold_dims) is preserved
        - Topology (cell connectivity) is preserved
        - Point/cell/global data are preserved as-is
        - Cached geometric properties are cleared
        - **Information in removed dimensions is discarded**

    Use cases:
        - [2, 3] -> [2, 2]: Project a 3D surface to the xy-plane
        - [1, 3] -> [1, 2]: Project a 3D curve down to 2D
        - [2, 3] -> [2, 2] via ``keep_dims=[0, 2]``: Project to the xz-plane

    Parameters
    ----------
    mesh : Mesh
        Input mesh to project.
    target_n_spatial_dims : int | None, optional
        Target number of spatial dimensions. Keeps the first N dimensions.
        Must be <= current n_spatial_dims and >= n_manifold_dims.
        If equal to current, returns mesh unchanged (no-op).
    keep_dims : Sequence[int] | None, optional
        Indices of spatial dimensions to retain, in the order they should
        appear in the output. For example, ``keep_dims=[0, 2]`` on a 3D mesh
        produces a 2D mesh with coordinates ``[x, z]``.

    Returns
    -------
    Mesh
        New mesh with reduced spatial dimensions:
        - points shape: ``(n_points, len(keep_dims))`` or
          ``(n_points, target_n_spatial_dims)``
        - n_manifold_dims: unchanged
        - cells: unchanged
        - point_data, cell_data: preserved (non-cached fields only)
        - Cached geometric properties: cleared

    Raises
    ------
    ValueError
        If both or neither of ``target_n_spatial_dims`` and ``keep_dims``
        are specified.
    ValueError
        If ``target_n_spatial_dims > current n_spatial_dims`` (use
        :func:`embed` to add dimensions).
    ValueError
        If any index in ``keep_dims`` is out of range.
    ValueError
        If the resulting number of spatial dimensions would be less than
        ``n_manifold_dims``.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.mesh import Mesh
    >>> from physicsnemo.mesh.projections import project
    >>> points = torch.tensor([[0., 1., 2.], [3., 4., 5.]])
    >>> cells = torch.tensor([[0, 1]])
    >>> mesh = Mesh(points=points, cells=cells)
    >>>
    >>> # Keep first 2 dimensions (drop z)
    >>> mesh_xy = project(mesh, target_n_spatial_dims=2)
    >>> assert torch.allclose(mesh_xy.points, torch.tensor([[0., 1.], [3., 4.]]))
    >>>
    >>> # Keep x and z (drop y) using keep_dims
    >>> mesh_xz = project(mesh, keep_dims=[0, 2])
    >>> assert torch.allclose(mesh_xz.points, torch.tensor([[0., 2.], [3., 5.]]))

    Notes
    -----
    This operation is lossy: coordinate values in removed dimensions cannot be
    recovered. If you need a reversible dimensionality change, use :func:`embed`
    followed by :func:`project`, ensuring the projected dimensions contain only
    values you added (e.g., zeros from embedding).

    When spatial dimensions change, all cached geometric properties are cleared
    because they depend on the spatial embedding (normals, centroids, areas,
    curvature). User data in ``point_data`` and ``cell_data`` is preserved as-is.

    See Also
    --------
    embed : The inverse operation - add spatial dimensions.
    """
    ### Validate mutually exclusive arguments
    if target_n_spatial_dims is not None and keep_dims is not None:
        raise ValueError(
            "Specify exactly one of target_n_spatial_dims or keep_dims, not both."
        )
    if target_n_spatial_dims is None and keep_dims is None:
        raise ValueError("Must specify either target_n_spatial_dims or keep_dims.")

    current_n_spatial_dims = mesh.n_spatial_dims

    ### Resolve keep_dims from the chosen calling convention
    if keep_dims is not None:
        keep_dims_list = list(keep_dims)
        result_n_dims = len(keep_dims_list)

        # Validate all indices are in range
        for idx in keep_dims_list:
            if not (0 <= idx < current_n_spatial_dims):
                raise ValueError(
                    f"keep_dims contains index {idx}, but mesh has only "
                    f"{current_n_spatial_dims} spatial dimensions "
                    f"(valid range: 0 to {current_n_spatial_dims - 1})."
                )
    else:
        if (
            target_n_spatial_dims is None
        ):  # pragma: no cover — unreachable after validation above
            raise ValueError(
                "target_n_spatial_dims must not be None when keep_dims is None."
            )

        if target_n_spatial_dims < 1:
            raise ValueError(
                f"target_n_spatial_dims must be >= 1, got {target_n_spatial_dims=}"
            )

        if target_n_spatial_dims > current_n_spatial_dims:
            raise ValueError(
                f"Cannot project: {target_n_spatial_dims=} > current "
                f"{current_n_spatial_dims=}. Use embed() to add spatial dimensions."
            )

        # Short-circuit: no-op if dimensions already match
        if target_n_spatial_dims == current_n_spatial_dims:
            return mesh

        keep_dims_list = list(range(target_n_spatial_dims))
        result_n_dims = target_n_spatial_dims

    ### Validate result has enough dims for the manifold
    if result_n_dims < mesh.n_manifold_dims:
        raise ValueError(
            f"Cannot project to {result_n_dims} spatial dimensions: mesh has "
            f"n_manifold_dims={mesh.n_manifold_dims}, and spatial dimensions "
            f"must be >= manifold dimensions."
        )

    ### Short-circuit: identity mapping (keep all dims in original order)
    if keep_dims_list == list(range(current_n_spatial_dims)):
        return mesh

    ### Construct new points by indexing selected dimensions
    new_points = mesh.points[:, keep_dims_list]

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
