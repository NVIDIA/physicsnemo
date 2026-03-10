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

"""Extrusion operations for generating higher-dimensional meshes."""

import torch
from tensordict import TensorDict

from physicsnemo.mesh.mesh import Mesh
from physicsnemo.mesh.projections._embed import embed


def extrude(
    mesh: Mesh,
    vector: torch.Tensor | list | tuple | None = None,
    capping: bool = False,
    allow_new_spatial_dims: bool = False,
) -> Mesh:
    """Extrude an N-dimensional mesh into an (N+1)-dimensional mesh.

    Extrusion sweeps a mesh along a direction to produce a mesh one manifold
    dimension higher - analogous to pushing a 2D cookie-cutter shape through
    space to create a solid 3D object. Each N-simplex in the input is swept into
    a prism and then tessellated into (N+1) child (N+1)-simplices.

    Common use cases:
        - **3D text and shapes**: extrude a 2D triangulated outline to create a
          solid tetrahedral volume (used by ``text_3d_3d`` in this library).
        - **Boundary layer meshing**: sweep a surface mesh along its normals to
          create a thin volumetric layer for near-wall CFD resolution.
        - **Space-time meshes**: extrude a spatial mesh along a time axis for
          space-time finite element discretizations.
        - **Structured mesh generation**: create 3D meshes from 2D cross-sections
          (e.g., a pipe from a circle, a channel from a rectangle).

    Dimensional behavior:
        - [N, M] -> [N+1, M]: Default case where M >= N+1
          (e.g., 2D surface in 3D -> 3D volume in 3D)
        - [N, M] -> [N+1, N+1]: When M < N+1 and ``allow_new_spatial_dims=True``,
          spatial dimensions are extended automatically.

    Parameters
    ----------
    mesh : Mesh
        Input mesh to extrude. Can be any manifold dimension N in any spatial
        dimension M >= N.
    vector : torch.Tensor | list | tuple | None, optional
        Extrusion direction and magnitude, shape ``(n_spatial_dims,)`` or
        broadcastable. If ``None``, defaults to a unit vector along the last
        spatial dimension: ``[0, 0, ..., 0, 1]``. For meshes where N+1 > M
        and ``allow_new_spatial_dims=True``, the default vector has shape
        ``(N+1,)`` with the last component set to 1.
    capping : bool, optional
        If True, cap the top and bottom of the extrusion to create a closed
        volume. Currently not implemented.
    allow_new_spatial_dims : bool, optional
        If True, allows extrusion to add new spatial dimensions when
        ``n_manifold_dims + 1 > n_spatial_dims``. The point coordinates are
        padded with zeros in the new dimensions (via :func:`embed`). If False
        (default), raises ``ValueError`` when spatial dimensions are insufficient.

    Returns
    -------
    Mesh
        Extruded mesh with:
        - ``n_manifold_dims = original_n_manifold_dims + 1``
        - ``n_spatial_dims = max(original_n_spatial_dims, n_manifold_dims)`` if
          ``allow_new_spatial_dims=True``, else ``original_n_spatial_dims``
        - ``n_points = 2 * original_n_points`` (original + extruded copies)
        - ``n_cells = (original_n_manifold_dims + 1) * original_n_cells``

    Raises
    ------
    ValueError
        If ``n_manifold_dims + 1 > n_spatial_dims`` and
        ``allow_new_spatial_dims=False``.
    NotImplementedError
        If ``capping=True`` (not yet implemented).

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.mesh import Mesh
    >>> # Extrude a triangle (2D) in 3D space to create a triangular prism
    >>> # tessellated into 3 tetrahedra
    >>> points = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    >>> cells = torch.tensor([[0, 1, 2]])
    >>> mesh = Mesh(points=points, cells=cells)
    >>> extruded = extrude(mesh, vector=[0., 0., 1.])
    >>> assert extruded.n_manifold_dims == 3  # tetrahedra
    >>> assert extruded.n_cells == 3  # one triangle -> three tetrahedra
    >>>
    >>> # Extrude an edge (1D) in 2D space to create triangles
    >>> points = torch.tensor([[0., 0.], [1., 0.]])
    >>> cells = torch.tensor([[0, 1]])
    >>> mesh = Mesh(points=points, cells=cells)
    >>> extruded = extrude(mesh, vector=[0., 1.])
    >>> assert extruded.n_manifold_dims == 2  # triangles
    >>> assert extruded.n_cells == 2  # one edge -> two triangles
    >>>
    >>> # Extrude a 2D surface in 2D space into 3D (auto-extending spatial dims)
    >>> points_2d = torch.tensor([[0., 0.], [1., 0.], [0., 1.]])
    >>> triangles = torch.tensor([[0, 1, 2]])
    >>> mesh_2d = Mesh(points_2d, triangles)  # [2, 2] mesh
    >>> extruded = extrude(mesh_2d, allow_new_spatial_dims=True)
    >>> assert extruded.n_spatial_dims == 3  # new dimension added

    Notes
    -----
    The tessellation pattern for an N-simplex with vertices [v0, v1, ..., vN]
    creates (N+1) child (N+1)-simplices:
        - Child i has vertices: [v0', v1', ..., vi', vi, vi+1, ..., vN]
    where primed vertices (v') are the extruded copies (offset by the extrusion
    vector).

    This tessellation preserves orientation and creates a valid simplicial complex.
    """
    ### Validate inputs
    if capping:
        raise NotImplementedError("Capping is not yet implemented. Use capping=False.")

    ### Determine target spatial dimensions and construct extrusion vector
    n_manifold_dims = mesh.n_manifold_dims
    n_spatial_dims = mesh.n_spatial_dims
    target_manifold_dims = n_manifold_dims + 1

    # Check if we have enough spatial dimensions for the extruded manifold
    if target_manifold_dims > n_spatial_dims:
        if not allow_new_spatial_dims:
            raise ValueError(
                f"Cannot extrude {n_manifold_dims=}-dimensional manifold in {n_spatial_dims=}-dimensional space "
                f"to {target_manifold_dims=}-dimensional manifold without increasing spatial dimensions.\n"
                f"Set allow_new_spatial_dims=True to add new spatial dimensions, or provide a custom vector."
            )
        # Extend spatial dimensions to accommodate extruded manifold
        target_spatial_dims = target_manifold_dims
    else:
        target_spatial_dims = n_spatial_dims

    # Construct or validate extrusion vector
    if vector is None:
        # Default: [0, 0, ..., 0, 1] in target spatial dimensions
        vector_tensor = torch.zeros(
            target_spatial_dims,
            dtype=mesh.points.dtype,
            device=mesh.points.device,
        )
        vector_tensor[-1] = 1.0
    else:
        # Convert to tensor if needed (torch.as_tensor avoids copies when possible)
        vector_tensor = torch.as_tensor(
            vector,
            dtype=mesh.points.dtype,
            device=mesh.points.device,
        )

        # Validate vector shape
        if vector_tensor.ndim != 1:
            raise ValueError(
                f"Extrusion vector must be 1D, got {vector_tensor.ndim=} with {vector_tensor.shape=}"
            )

    ### Pad mesh spatial dimensions if needed (delegate to embed)
    if target_spatial_dims > n_spatial_dims:
        mesh = embed(mesh, target_spatial_dims)

    original_points = mesh.points

    # Ensure vector has correct shape for broadcasting
    if vector_tensor.shape[0] != target_spatial_dims:
        if vector_tensor.shape[0] < target_spatial_dims:
            # Pad vector with zeros
            padding_width = target_spatial_dims - vector_tensor.shape[0]
            vector_tensor = torch.nn.functional.pad(
                vector_tensor,
                (0, padding_width),
                mode="constant",
                value=0.0,
            )
        else:
            raise ValueError(
                f"Extrusion vector has {vector_tensor.shape[0]} dimensions but "
                f"target spatial dimensions are {target_spatial_dims}"
            )

    ### Create extruded points
    extruded_points = original_points + vector_tensor.unsqueeze(0)

    # Concatenate: [original_points, extruded_points]
    all_points = torch.cat([original_points, extruded_points], dim=0)

    ### Tessellate cells
    # Each N-simplex becomes (N+1) child (N+1)-simplices
    # For parent cell with vertices [v0, v1, ..., vN]:
    #   Child i: [v0', v1', ..., vi', vi, vi+1, ..., vN]
    # where vi' = vi + n_original_points (extruded vertex index)

    n_original_points = mesh.n_points
    n_original_cells = mesh.n_cells
    n_vertices_per_parent = n_manifold_dims + 1  # N+1 vertices in N-simplex
    n_children_per_parent = n_manifold_dims + 1  # N+1 children from each parent
    n_vertices_per_child = target_manifold_dims + 1  # (N+1)+1 vertices in (N+1)-simplex

    if n_original_cells == 0:
        # Empty mesh: no cells to extrude
        extruded_cells = torch.empty(
            (0, n_vertices_per_child),
            dtype=mesh.cells.dtype,
            device=mesh.cells.device,
        )
    else:
        # Preallocate child cells array
        extruded_cells = torch.zeros(
            (n_original_cells * n_children_per_parent, n_vertices_per_child),
            dtype=mesh.cells.dtype,
            device=mesh.cells.device,
        )

        # Vectorized tessellation
        # For each parent cell, generate all children simultaneously
        parent_cells = mesh.cells  # Shape: (n_cells, n_vertices_per_parent)

        # NOTE: This loop iterates over child indices (bounded by simplex
        # dimension, not mesh size), so the iteration count is small (e.g. 3
        # for triangles -> tetrahedra).  For higher performance on very high-
        # dimensional simplices, this could be vectorized using a precomputed
        # tessellation pattern + torch.gather, similar to generate_child_cells
        # in subdivision/_topology.py.
        for child_idx in range(n_children_per_parent):
            # Child i has vertices: [v0', v1', ..., vi', vi, vi+1, ..., vN]
            # Extruded part: v0', v1', ..., vi' (child_idx + 1 vertices)
            # Original part: vi, vi+1, ..., vN (n_vertices_per_parent - child_idx vertices)

            child_vertices = []

            # Add extruded vertices [v0', v1', ..., vi']
            for j in range(child_idx + 1):
                extruded_vertex_indices = parent_cells[:, j] + n_original_points
                child_vertices.append(extruded_vertex_indices)

            # Add original vertices [vi, vi+1, ..., vN]
            for j in range(child_idx, n_vertices_per_parent):
                original_vertex_indices = parent_cells[:, j]
                child_vertices.append(original_vertex_indices)

            # Stack to form child cells: (n_cells, n_vertices_per_child)
            child_cells = torch.stack(child_vertices, dim=1)

            # Place in output array
            start_idx = child_idx * n_original_cells
            end_idx = (child_idx + 1) * n_original_cells
            extruded_cells[start_idx:end_idx] = child_cells

    ### Propagate data (excluding cached properties, which depend on geometry)
    filtered_point_data = mesh.point_data
    filtered_cell_data = mesh.cell_data
    # Point data: concatenate original and copy for extruded points
    if len(filtered_point_data.keys()) > 0:
        extruded_point_data = TensorDict.cat(
            [filtered_point_data, filtered_point_data.clone()],
            dim=0,
        )
    else:
        extruded_point_data = TensorDict(
            {},
            batch_size=torch.Size([all_points.shape[0]]),
            device=all_points.device,
        )

    # Cell data: replicate each parent cell's data (N+1) times.
    # Cells are ordered child-major: [c0_parent0, c0_parent1, ..., c1_parent0, ...],
    # so the parent index for cell i is i % n_original_cells.  Using .repeat()
    # tiles the entire parent block N+1 times, producing the correct ordering.
    if len(filtered_cell_data.keys()) > 0:
        extruded_cell_data = filtered_cell_data.apply(
            lambda t: t.repeat(n_children_per_parent, *([1] * (t.ndim - 1))),
            batch_size=torch.Size([extruded_cells.shape[0]]),
        )
    else:
        extruded_cell_data = TensorDict(
            {},
            batch_size=torch.Size([extruded_cells.shape[0]]),
            device=extruded_cells.device,
        )

    # Global data: preserve unchanged
    extruded_global_data = mesh.global_data

    ### Create and return extruded mesh
    return Mesh(
        points=all_points,
        cells=extruded_cells,
        point_data=extruded_point_data,
        cell_data=extruded_cell_data,
        global_data=extruded_global_data,
    )
