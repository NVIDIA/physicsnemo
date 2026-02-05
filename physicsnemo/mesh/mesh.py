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

from typing import TYPE_CHECKING, Any, Literal, Self, Sequence

import torch
import torch.nn.functional as F
from tensordict import TensorDict, tensorclass

from physicsnemo.mesh.transformations.geometric import (
    rotate,
    scale,
    transform,
    translate,
)
from physicsnemo.mesh.utilities._cache import get_cached, set_cached
from physicsnemo.mesh.utilities._padding import _pad_by_tiling_last, _pad_with_value
from physicsnemo.mesh.utilities._scatter_ops import scatter_aggregate
from physicsnemo.mesh.utilities.mesh_repr import format_mesh_repr
from physicsnemo.mesh.visualization.draw_mesh import draw_mesh


@tensorclass(tensor_only=True)
class Mesh:
    r"""A PyTorch-based, dimensionally-generic Mesh data structure.

    A ``Mesh`` is a discrete representation of an n-dimensional manifold embedded
    in m-dimensional Euclidean space (where n ≤ m). Field data can be associated
    with each point, with each cell, or globally with the mesh itself. This field
    data can be arbitrarily-dimensional (scalar fields, vector fields, or
    arbitrary-rank tensor fields) and semantically-rich (supporting string keys
    and nested data structures).

    **Simplices**

    The building block of a ``Mesh`` is a **simplex** (plural: **simplices**): a
    generalization of the notion of a triangle or tetrahedron to arbitrary
    dimensions. Consider these familiar examples of an n-dimensional simplex
    (an **n-simplex**):

    =========  ====================  =========================================
               Common Name           Description
    =========  ====================  =========================================
    0-simplex  Point                 A single vertex
    1-simplex  Line Segment / Edge   Connects 2 points; boundary: 2 0-simplices
    2-simplex  Triangle              Connects 3 points; boundary: 3 1-simplices
    3-simplex  Tetrahedron           Connects 4 points; boundary: 4 2-simplices
    =========  ====================  =========================================

    **Manifold Dimension**

    A ``Mesh`` is a collection of simplices that share vertices. Every simplex
    in a ``Mesh`` must have the same dimension; this shared dimension is called
    the **manifold dimension** (``n_manifold_dims``), representing the intrinsic
    dimensionality of each cell. A triangle has manifold dimension 2 regardless
    of whether it lives in 2D or 3D space.

    **Spatial Dimension and Codimension**

    The **spatial dimension** (``n_spatial_dims``) is the dimension of the
    embedding space where point coordinates live. A triangle mesh representing
    a 3D surface has ``n_spatial_dims=3`` but ``n_manifold_dims=2``.

    The difference, **codimension** = ``n_spatial_dims - n_manifold_dims``,
    determines whether unique normal vectors exist:

    - Codimension 1 (triangles in 3D, edges in 2D): unique unit normal (up to sign)
    - Codimension 0 (triangles in 2D, tets in 3D): no normal direction exists
    - Codimension > 1 (edges in 3D): infinitely many normal directions

    **Core Data Structure**

    A mesh is defined by two tensors:

    - ``points``: Vertex coordinates with shape :math:`(N_p, D_s)` where
      :math:`N_p` is the number of points and :math:`D_s` is the spatial
      dimension. For 1000 vertices in 3D: shape ``(1000, 3)``.

    - ``cells``: Cell connectivity with shape :math:`(N_c, D_m + 1)` where
      :math:`N_c` is the number of cells and :math:`D_m` is the manifold
      dimension. Each row lists point indices defining one simplex. For 500
      triangles: shape ``(500, 3)`` since each triangle references 3 vertices.

    **Attaching Field Data**

    Tensor data of any shape can be attached at three levels:

    - ``point_data``: Per-vertex quantities (temperature, velocity, embeddings)
    - ``cell_data``: Per-cell quantities (pressure, stress, material ID)
    - ``global_data``: Mesh-level quantities (simulation time, Reynolds number)

    All data is stored in ``TensorDict`` containers that move together with the
    mesh geometry under ``.to(device)`` calls.

    Parameters
    ----------
    points : torch.Tensor
        Vertex coordinates with shape :math:`(N_p, D_s)`. Must be floating-point.
    cells : torch.Tensor
        Cell connectivity with shape :math:`(N_c, D_m + 1)`. Each row contains
        indices into ``points`` defining one simplex. Must be integer dtype.
    point_data : TensorDict or dict[str, torch.Tensor], optional
        Per-vertex data. Dicts are automatically converted to TensorDict.
    cell_data : TensorDict or dict[str, torch.Tensor], optional
        Per-cell data. Dicts are automatically converted to TensorDict.
    global_data : TensorDict or dict[str, torch.Tensor], optional
        Mesh-level data. Dicts are automatically converted to TensorDict.

    Raises
    ------
    ValueError
        If ``points`` is not 2D, ``cells`` is not 2D, or manifold dimension
        exceeds spatial dimension.
    TypeError
        If ``cells`` has a floating-point dtype (indices must be integers).

    Examples
    --------
    Create a 2D triangular mesh (two triangles forming a unit square):

    >>> import torch
    >>> from physicsnemo.mesh import Mesh
    >>> points = torch.tensor([
    ...     [0.0, 0.0],  # vertex 0: bottom-left
    ...     [1.0, 0.0],  # vertex 1: bottom-right
    ...     [1.0, 1.0],  # vertex 2: top-right
    ...     [0.0, 1.0],  # vertex 3: top-left
    ... ])
    >>> cells = torch.tensor([
    ...     [0, 1, 2],  # triangle 0: vertices 0-1-2
    ...     [0, 2, 3],  # triangle 1: vertices 0-2-3
    ... ])
    >>> mesh = Mesh(points=points, cells=cells)
    >>> mesh.n_points, mesh.n_cells, mesh.n_spatial_dims, mesh.n_manifold_dims
    (4, 2, 2, 2)

    Attach field data at vertices and cells:

    >>> mesh = Mesh(
    ...     points=points,
    ...     cells=cells,
    ...     point_data={"temperature": torch.tensor([300., 350., 340., 310.])},
    ...     cell_data={"pressure": torch.tensor([101.3, 99.8])},
    ... )

    Move mesh and all data to GPU:

    >>> mesh_gpu = mesh.to("cuda")  # doctest: +SKIP

    Create an undirected graph (1-simplices in 3D):

    >>> nodes = torch.randn(100, 3)  # 100 vertices in 3D
    >>> edges = torch.randint(0, 100, (200, 2))  # 200 edges
    >>> graph = Mesh(points=nodes, cells=edges)
    >>> graph.n_manifold_dims, graph.n_spatial_dims
    (1, 3)

    Notes
    -----
    **Mixed Manifold Dimensions**

    To represent structures with multiple manifold dimensions (e.g., a
    tetrahedral volume mesh together with its triangular boundary surface),
    use separate ``Mesh`` objects for each dimension.

    **Non-Simplicial Elements**

    This class only supports simplicial cells. Non-simplicial elements must be
    subdivided before use:

    - **Quads** → split into 2 triangles each
    - **Hexahedra** → split into 5 or 6 tetrahedra each
    - **Polygons/polyhedra** → triangulate/tetrahedralize

    **Caching**

    Expensive geometric computations (centroids, areas, normals, etc.) are
    cached automatically under keys prefixed with ``_cache`` in the data
    dictionaries. The cache persists across repeated property accesses but is
    invalidated when creating new ``Mesh`` instances.
    """

    points: torch.Tensor  # shape: (n_points, n_spatial_dimensions)
    cells: torch.Tensor  # shape: (n_cells, n_manifold_dimensions + 1)
    point_data: TensorDict
    cell_data: TensorDict
    global_data: TensorDict

    def __init__(
        self,
        points: torch.Tensor,
        cells: torch.Tensor,
        point_data: TensorDict | dict[str, torch.Tensor] | None = None,
        cell_data: TensorDict | dict[str, torch.Tensor] | None = None,
        global_data: TensorDict | dict[str, torch.Tensor] | None = None,
    ) -> None:
        ### Assign tensorclass fields
        self.points = points
        self.cells = cells

        # For data fields, convert inputs to TensorDicts if needed
        if isinstance(point_data, TensorDict):
            point_data.batch_size = torch.Size(
                [self.n_points]
            )  # Ensure shape-compatible
        else:
            point_data = TensorDict(
                {} if point_data is None else dict(point_data),
                batch_size=torch.Size([self.n_points]),
                device=self.points.device,
            )
        self.point_data = point_data

        if isinstance(cell_data, TensorDict):
            cell_data.batch_size = torch.Size([self.n_cells])  # Ensure shape-compatible
        else:
            cell_data = TensorDict(
                {} if cell_data is None else dict(cell_data),
                batch_size=torch.Size([self.n_cells]),
                device=self.cells.device,
            )
        self.cell_data = cell_data

        if isinstance(global_data, TensorDict):
            global_data.batch_size = torch.Size([])  # Ensure shape-compatible
        else:
            global_data = TensorDict(
                {} if global_data is None else dict(global_data),
                batch_size=torch.Size([]),
                device=self.points.device,
            )
        self.global_data = global_data

        ### Validate shapes and dtypes
        if self.points.ndim != 2:
            raise ValueError(
                f"`points` must have shape (n_points, n_spatial_dimensions), but got {self.points.shape=}."
            )
        if self.cells.ndim != 2:
            raise ValueError(
                f"`cells` must have shape (n_cells, n_manifold_dimensions + 1), but got {self.cells.shape=}."
            )
        if self.n_manifold_dims > self.n_spatial_dims:
            raise ValueError(
                f"`n_manifold_dims` must be <= `n_spatial_dims`, but got {self.n_manifold_dims=} > {self.n_spatial_dims=}."
            )
        if torch.is_floating_point(self.cells):
            raise TypeError(
                f"`cells` must have an int-like dtype, but got {self.cells.dtype=}."
            )

    if TYPE_CHECKING:
        # Type stub for the `to` method dynamically added by @tensorclass.
        # This provides proper type hints without shadowing the runtime implementation.
        def to(self, *args: Any, **kwargs: Any) -> Self:
            """Move mesh and all attached data to specified device, dtype, or format.

            Maps this Mesh to another device and/or dtype. All tensors in ``points``,
            ``cells``, ``point_data``, ``cell_data``, and ``global_data`` are moved
            together.

            Parameters
            ----------
            *args : Any
                Positional arguments passed to the underlying tensorclass ``to`` method.
                Common usage: ``mesh.to("cuda")`` or ``mesh.to(torch.float32)``.
            **kwargs : Any
                Keyword arguments passed to the underlying tensorclass ``to`` method.

            Keyword Arguments
            -----------------
            device : torch.device, optional
                The desired device of the mesh.
            dtype : torch.dtype, optional
                The desired floating point or complex dtype of the mesh tensors.
            non_blocking : bool, optional
                Whether the operations should be non-blocking.
            memory_format : torch.memory_format, optional
                The desired memory format for 4D parameters and buffers.

            Returns
            -------
            Mesh
                A new Mesh instance on the target device/dtype, or the same mesh if
                no changes were required.

            Examples
            --------
            >>> mesh_gpu = mesh.to("cuda")
            >>> mesh_cpu = mesh.to(device="cpu")
            >>> mesh_fp16 = mesh.to(torch.float16)
            """
            ...

    @property
    def n_points(self) -> int:
        return self.points.shape[0]

    @property
    def n_spatial_dims(self) -> int:
        return self.points.shape[-1]

    @property
    def n_cells(self) -> int:
        return self.cells.shape[0]

    @property
    def n_manifold_dims(self) -> int:
        return self.cells.shape[-1] - 1

    @property
    def codimension(self) -> int:
        """Compute the codimension of the mesh.

        The codimension is the difference between the spatial dimension and the
        manifold dimension: codimension = n_spatial_dims - n_manifold_dims.

        Examples:
            - Edges (1-simplices) in 2D: codimension = 2 - 1 = 1 (codimension-1)
            - Triangles (2-simplices) in 3D: codimension = 3 - 2 = 1 (codimension-1)
            - Edges in 3D: codimension = 3 - 1 = 2 (codimension-2)
            - Points in 2D: codimension = 2 - 0 = 2 (codimension-2)

        Returns
        -------
        int
            The codimension of the mesh (always non-negative).
        """
        return self.n_spatial_dims - self.n_manifold_dims

    @property
    def cell_centroids(self) -> torch.Tensor:
        """Compute the centroids (geometric centers) of all cells.

        The centroid of a cell is computed as the arithmetic mean of its vertex positions.
        For an n-simplex with vertices (v0, v1, ..., vn), the centroid is:
            centroid = (v0 + v1 + ... + vn) / (n + 1)

        The result is cached in cell_data["_cache"]["centroids"] for efficiency.

        Returns
        -------
        torch.Tensor
            Tensor of shape (n_cells, n_spatial_dims) containing the centroid of each cell.
        """
        cached = get_cached(self.cell_data, "centroids")
        if cached is None:
            cached = self.points[self.cells].mean(dim=1)
            set_cached(self.cell_data, "centroids", cached)
        return cached

    @property
    def cell_areas(self) -> torch.Tensor:
        """Compute volumes (areas) of n-simplices using the Gram determinant method.

        This works for simplices of any manifold dimension embedded in any spatial dimension.
        For example: edges in 2D/3D, triangles in 2D/3D/4D, tetrahedra in 3D/4D, etc.

        The volume of an n-simplex with vertices (v0, v1, ..., vn) is:
            Volume = (1/n!) * sqrt(det(E^T @ E))
        where E is the matrix with columns (v1-v0, v2-v0, ..., vn-v0).

        Returns
        -------
        torch.Tensor
            Tensor of shape (n_cells,) containing the volume of each cell.
        """
        cached = get_cached(self.cell_data, "areas")
        if cached is None:
            ### Compute relative vectors from first vertex to all others
            # Shape: (n_cells, n_manifold_dims, n_spatial_dims)
            relative_vectors = (
                self.points[self.cells[:, 1:]] - self.points[self.cells[:, [0]]]
            )

            ### Compute Gram matrix: G = E^T @ E
            # E conceptually has shape (n_spatial_dims, n_manifold_dims) per cell
            # Gram matrix has shape (n_manifold_dims, n_manifold_dims) per cell
            # In batch form: (n_cells, n_manifold_dims, n_spatial_dims) @ (n_cells, n_spatial_dims, n_manifold_dims)
            gram_matrix = torch.matmul(
                relative_vectors,  # (n_cells, n_manifold_dims, n_spatial_dims)
                relative_vectors.transpose(
                    -2, -1
                ),  # (n_cells, n_spatial_dims, n_manifold_dims)
            )  # Result: (n_cells, n_manifold_dims, n_manifold_dims)

            ### Compute volume: sqrt(|det(G)|) / n!
            # Compute factorial using torch for small integers
            factorial = torch.arange(
                1, self.n_manifold_dims + 1, device=gram_matrix.device
            ).prod()

            cached = gram_matrix.det().abs().sqrt() / factorial
            set_cached(self.cell_data, "areas", cached)

        return cached

    @property
    def cell_normals(self) -> torch.Tensor:
        """Compute unit normal vectors for codimension-1 cells.

        Normal vectors are uniquely defined (up to orientation) only for codimension-1
        manifolds, where n_manifold_dims = n_spatial_dims - 1. This is because the
        perpendicular subspace to an (n-1)-dimensional manifold in n-dimensional space
        is 1-dimensional, yielding a unique normal direction.

        Examples of valid codimension-1 manifolds:
        - Edges (1-simplices) in 2D space: normal is a 2D vector
        - Triangles (2-simplices) in 3D space: normal is a 3D vector
        - Tetrahedron cells (3-simplices) in 4D space: normal is a 4D vector

        Examples of invalid higher-codimension cases:
        - Edges in 3D space: perpendicular space is 2D (no unique normal)
        - Points in 2D/3D space: perpendicular space is 2D/3D (no unique normal)

        The implementation uses the generalized cross product (Hodge star operator),
        computed via signed minor determinants. This generalizes:
        - 2D: 90° counterclockwise rotation of edge vector
        - 3D: Standard cross product of two edge vectors
        - nD: Determinant-based formula for (n-1) edge vectors in n-space

        Returns
        -------
        torch.Tensor
            Tensor of shape (n_cells, n_spatial_dims) containing unit normal vectors.

        Raises
        ------
        ValueError
            If the mesh is not codimension-1 (n_manifold_dims ≠ n_spatial_dims - 1).
        """
        cached = get_cached(self.cell_data, "normals")
        if cached is None:
            ### Validate codimension-1 requirement
            if self.codimension != 1:
                raise ValueError(
                    f"cell normals are only defined for codimension-1 manifolds.\n"
                    f"Got {self.n_manifold_dims=} and {self.n_spatial_dims=}.\n"
                    f"Required: n_manifold_dims = n_spatial_dims - 1 (codimension-1).\n"
                    f"Current codimension: {self.codimension}"
                )

            ### Compute relative vectors from first vertex to all others
            # Shape: (n_cells, n_manifold_dims, n_spatial_dims)
            # These form the rows of matrix E for each cell
            relative_vectors = (
                self.points[self.cells[:, 1:]] - self.points[self.cells[:, [0]]]
            )

            ### Compute normal using generalized cross product (Hodge star)
            # For (n-1) vectors in R^n represented as rows of matrix E,
            # the perpendicular vector has components:
            #   n_i = (-1)^(n-1+i) * det(E with column i removed)
            # This generalizes 2D rotation and 3D cross product.
            normal_components = []

            for i in range(self.n_spatial_dims):
                ### Select all columns except the i-th to form (n-1)×(n-1) submatrix
                cols_mask = torch.ones(
                    self.n_spatial_dims,
                    dtype=torch.bool,
                    device=relative_vectors.device,
                )
                cols_mask[i] = False
                submatrix = relative_vectors[
                    :, :, cols_mask
                ]  # (n_cells, n_manifold_dims, n_manifold_dims)

                ### Compute signed minor: (-1)^(n_manifold_dims + i) * det(submatrix)
                det = submatrix.det()  # (n_cells,)
                sign = (-1) ** (self.n_manifold_dims + i)
                normal_components.append(sign * det)

            ### Stack components and normalize to unit length
            normals = torch.stack(
                normal_components, dim=-1
            )  # (n_cells, n_spatial_dims)
            cached = F.normalize(normals, dim=-1)
            set_cached(self.cell_data, "normals", cached)

        return cached

    @property
    def point_normals(self) -> torch.Tensor:
        """Compute angle-area-weighted normal vectors at mesh vertices.

        This property returns the canonical/default point normals using combined
        angle and area weighting (Maya default). For other weighting schemes
        (unweighted, area, angle), use :meth:`compute_point_normals`.

        Angle-area weighting ensures that each face's contribution is weighted by
        both its area and the interior angle at the vertex, balancing both geometric
        factors for high-quality normals.

        The result is cached in point_data["_cache"]["normals"] for efficiency.

        Returns
        -------
        torch.Tensor
            Tensor of shape (n_points, n_spatial_dims) containing unit normal vectors
            at each vertex. For isolated points (with no adjacent cells), the normal
            is a zero vector.

        Raises
        ------
        ValueError
            If the mesh is not codimension-1 (n_manifold_dims ≠ n_spatial_dims - 1).

        See Also
        --------
        compute_point_normals : Compute point normals with explicit weighting choice.
        cell_normals : Compute cell (face) normals.

        Examples
        --------
            >>> # Triangle mesh in 3D
            >>> mesh = create_triangle_mesh_3d()  # doctest: +SKIP
            >>> normals = mesh.point_normals  # (n_points, 3), angle-area-weighted  # doctest: +SKIP
            >>> # Normals are unit vectors (or zero for isolated points)
            >>> assert torch.allclose(normals.norm(dim=-1), torch.ones(mesh.n_points), atol=1e-6)  # doctest: +SKIP
        """
        cached = get_cached(self.point_data, "normals")
        if cached is None:
            cached = self.compute_point_normals(weighting="angle_area")
            set_cached(self.point_data, "normals", cached)
        return cached

    def compute_point_normals(
        self,
        weighting: Literal["area", "unweighted", "angle", "angle_area"] = "angle_area",
    ) -> torch.Tensor:
        """Compute normal vectors at mesh vertices with specified weighting.

        For each point (vertex), computes a normal vector by averaging the normals
        of all adjacent cells. This provides a smooth approximation of the surface
        normal at each vertex.

        Four weighting schemes are available (following industry conventions from
        Autodesk Maya and 3ds Max):

        - **"area"** (default): Area-weighted averaging, where larger faces have more
          influence on the vertex normal. The normal at vertex v is computed as:
          ``point_normal_v = normalize(sum(cell_normal * cell_area))``.
          This reduces the influence of small sliver triangles.

        - **"unweighted"**: Simple averaging, where each adjacent face contributes
          equally regardless of size. The normal at vertex v is:
          ``point_normal_v = normalize(sum(cell_normal))``.
          This matches PyVista/VTK's ``compute_normals`` behavior.

        - **"angle"**: Angle-weighted averaging, where faces are weighted by the
          interior angle at the vertex. Faces with larger angles at the vertex
          have more influence. This often provides the most geometrically accurate
          normals for curved surfaces.

        - **"angle_area"**: Combined angle and area weighting, where each face's
          contribution is weighted by both its area and the angle at the vertex.
          This is the default in Maya and balances both geometric factors.

        Normal vectors are only well-defined for codimension-1 manifolds, where each
        cell has a unique normal direction. For higher codimensions, normals are
        ambiguous and this method will raise an error.

        Parameters
        ----------
        weighting : {"area", "unweighted", "angle", "angle_area"}
            Weighting scheme for averaging adjacent cell normals.
            - "area": Weight by cell area (larger faces have more influence).
            - "unweighted": Equal weight for all adjacent cells (matches PyVista/VTK).
            - "angle": Weight by interior angle at the vertex.
            - "angle_area": Weight by both angle and area (Maya default).

        Returns
        -------
        torch.Tensor
            Tensor of shape (n_points, n_spatial_dims) containing unit normal vectors
            at each vertex. For isolated points (with no adjacent cells), the normal
            is a zero vector.

        Raises
        ------
        ValueError
            If the mesh is not codimension-1 (n_manifold_dims ≠ n_spatial_dims - 1),
            if an invalid weighting scheme is specified, or if angle-based weighting
            is requested for 1-simplices (edges) which have no interior angle.

        See Also
        --------
        point_normals : Property returning angle-area-weighted normals (canonical default).
        cell_normals : Compute cell (face) normals.

        Examples
        --------
            >>> # Triangle mesh in 3D
            >>> mesh = create_triangle_mesh_3d()  # doctest: +SKIP
            >>> normals = mesh.compute_point_normals()  # area-weighted (default)  # doctest: +SKIP
            >>> normals_unweighted = mesh.compute_point_normals(weighting="unweighted")  # doctest: +SKIP
            >>> normals_angle = mesh.compute_point_normals(weighting="angle")  # doctest: +SKIP
            >>> # Normals are unit vectors (or zero for isolated points)
            >>> assert torch.allclose(normals.norm(dim=-1), torch.ones(mesh.n_points), atol=1e-6)  # doctest: +SKIP
        """
        valid_weightings = ("area", "unweighted", "angle", "angle_area")
        if weighting not in valid_weightings:
            raise ValueError(
                f"Invalid {weighting=}. Must be one of {valid_weightings}."
            )

        ### Validate codimension-1 requirement (same as cell_normals)
        if self.codimension != 1:
            raise ValueError(
                f"Point normals are only defined for codimension-1 manifolds.\n"
                f"Got {self.n_manifold_dims=} and {self.n_spatial_dims=}.\n"
                f"Required: n_manifold_dims = n_spatial_dims - 1 (codimension-1).\n"
                f"Current codimension: {self.codimension}"
            )

        ### Validate angle-based weighting requires 2+ manifold dims
        if weighting in ("angle", "angle_area") and self.n_manifold_dims < 2:
            raise ValueError(
                f"Angle-based weighting requires n_manifold_dims >= 2 "
                f"(cells must have interior angles).\n"
                f"Got {self.n_manifold_dims=}. Use 'area' or 'unweighted' instead."
            )

        ### Get cell normals (triggers computation if not cached)
        cell_normals = self.cell_normals  # (n_cells, n_spatial_dims)

        ### Initialize accumulated normals for each point
        accumulated_normals = torch.zeros(
            (self.n_points, self.n_spatial_dims),
            dtype=self.points.dtype,
            device=self.points.device,
        )

        n_vertices_per_cell = self.cells.shape[1]
        point_indices = self.cells.flatten()

        # Repeat cell normals for each vertex in the cell
        cell_normals_repeated = cell_normals.unsqueeze(1).expand(
            -1, n_vertices_per_cell, -1
        )
        cell_normals_flat = cell_normals_repeated.reshape(-1, self.n_spatial_dims)

        ### Compute weights based on scheme
        if weighting == "unweighted":
            weights = torch.ones(
                self.n_cells * n_vertices_per_cell,
                dtype=self.points.dtype,
                device=self.points.device,
            )

        elif weighting == "area":
            cell_areas = self.cell_areas
            weights = cell_areas.unsqueeze(1).expand(-1, n_vertices_per_cell).flatten()

        elif weighting in ("angle", "angle_area"):
            # Compute interior angles at each vertex of each cell
            # For a simplex, angle at vertex k is between edges to other vertices
            vertex_angles = (
                self._compute_vertex_angles()
            )  # (n_cells, n_vertices_per_cell)
            weights = vertex_angles.flatten()

            if weighting == "angle_area":
                # Multiply by cell area
                cell_areas = self.cell_areas
                area_weights = (
                    cell_areas.unsqueeze(1).expand(-1, n_vertices_per_cell).flatten()
                )
                weights = weights * area_weights

        ### Apply weights and accumulate
        normals_to_accumulate = cell_normals_flat * weights.unsqueeze(-1)

        point_indices_expanded = point_indices.unsqueeze(-1).expand(
            -1, self.n_spatial_dims
        )
        accumulated_normals.scatter_add_(
            dim=0,
            index=point_indices_expanded,
            src=normals_to_accumulate,
        )

        ### Normalize to get unit normals
        return F.normalize(accumulated_normals, dim=-1)

    def _compute_vertex_angles(self) -> torch.Tensor:
        """Compute generalized interior angles at each vertex of each cell.

        For an n-simplex, the "angle" at a vertex is computed using the unified
        formula that generalizes to arbitrary dimensions:

            Ω = 2 × arctan(√det(C) / (1 + Σᵢ<ⱼ Cᵢⱼ))

        where C is the correlation (normalized Gram) matrix of edge vectors:
            Cᵢⱼ = (eᵢ · eⱼ) / (|eᵢ| |eⱼ|)

        This formula reduces to:
        - For triangles (n=2): the planar interior angle θ
        - For tetrahedra (n=3): the solid angle Ω (steradians)
        - For higher n: the generalized solid angle

        Returns
        -------
        torch.Tensor
            Tensor of shape (n_cells, n_vertices_per_cell) containing the
            generalized angle at each vertex.

        Notes
        -----
        This formula is derived by recognizing that both the planar angle formula
        and the Van Oosterom & Strackee (1983) solid angle formula follow the
        same pattern when expressed in terms of the correlation matrix.

        The formula uses atan2 for numerical stability when the denominator
        approaches zero (nearly degenerate simplices).
        """
        n_edges = self.n_manifold_dims  # edges from each vertex

        # Get vertex positions: (n_cells, n_verts, n_spatial_dims)
        cell_vertices = self.points[self.cells]

        # For each vertex k, compute the edge vectors to the other n_edges vertices
        # Use roll to get shifted vertex positions:
        # rolled[:, :, i, :] = cell_vertices[:, (k + i + 1) % n_verts, :]

        # Build edge vectors for all vertices simultaneously
        # edges[k, i] = v_{(k+i+1) mod n_verts} - v_k
        # Shape: (n_cells, n_verts, n_edges, n_spatial_dims)
        edges = torch.stack(
            [
                torch.roll(cell_vertices, shifts=-(i + 1), dims=1) - cell_vertices
                for i in range(n_edges)
            ],
            dim=2,
        )

        # Compute edge lengths: (n_cells, n_verts, n_edges)
        edge_lengths = edges.norm(dim=-1)

        # Compute normalized edges: (n_cells, n_verts, n_edges, n_spatial_dims)
        edges_normalized = edges / edge_lengths.unsqueeze(-1).clamp(min=1e-10)

        # Compute correlation matrix C for each vertex of each cell
        # C[i,j] = normalized_edge_i · normalized_edge_j
        # Shape: (n_cells, n_verts, n_edges, n_edges)
        # Using einsum: C_ij = sum_d (edges_normalized[:,:,i,d] * edges_normalized[:,:,j,d])
        corr_matrix = torch.einsum(
            "cvid,cvjd->cvij", edges_normalized, edges_normalized
        )

        # Compute det(C) for each vertex: (n_cells, n_verts)
        det_C = torch.linalg.det(corr_matrix)

        # Compute sum of off-diagonal elements: Σᵢ<ⱼ Cᵢⱼ
        # For an n×n matrix, sum of upper triangle (excluding diagonal)
        # Create upper triangle mask
        triu_mask = torch.triu(
            torch.ones(n_edges, n_edges, device=self.points.device, dtype=torch.bool),
            diagonal=1,
        )
        # Sum off-diagonal: (n_cells, n_verts)
        sum_off_diag = corr_matrix[:, :, triu_mask].sum(dim=-1)

        # Denominator: 1 + Σᵢ<ⱼ Cᵢⱼ
        denominator = 1.0 + sum_off_diag

        # Numerator: √det(C) (use abs for numerical stability with near-degenerate cells)
        numerator = det_C.abs().sqrt()

        # Compute angle: Ω = 2 × arctan(numerator / denominator)
        # Use atan2 for numerical stability
        angles = 2.0 * torch.atan2(numerator, denominator)

        return angles

    @classmethod
    def merge(
        cls, meshes: Sequence["Mesh"], global_data_strategy: Literal["stack"] = "stack"
    ) -> "Mesh":
        """Merge multiple meshes into a single mesh.

        Parameters
        ----------
        meshes : Sequence[Mesh]
            List of Mesh objects to merge. All constituent tensors across all
            meshes must reside on the same device.
        global_data_strategy : {"stack"}
            Strategy for handling global_data. Currently only "stack" is supported,
            which stacks global_data fields along a new dimension.

        Returns
        -------
        Mesh
            A new Mesh object containing all the merged data.

        Raises
        ------
        ValueError
            If the meshes list is empty, or if meshes have inconsistent dimensions
            or cell_data keys.
        TypeError
            If any element in meshes is not a Mesh object.
        RuntimeError
            If tensors from different meshes reside on different devices.
        """
        ### Validate inputs
        if not torch.compiler.is_compiling():
            if len(meshes) == 0:
                raise ValueError("At least one Mesh must be provided to merge.")
            elif len(meshes) == 1:  # Short-circuit for speed in this case
                return meshes[0]
            if not all(isinstance(m, Mesh) for m in meshes):
                raise TypeError(
                    f"All objects must be Mesh types. Got:\n"
                    f"{[type(m) for m in meshes]=}"
                )
            # Check dimensional consistency across all meshes
            validations = {
                "spatial dimensions": [m.n_spatial_dims for m in meshes],
                "manifold dimensions": [m.n_manifold_dims for m in meshes],
            }
            for name, values in validations.items():
                if not all(v == values[0] for v in values):
                    raise ValueError(
                        f"All meshes must have the same {name}. Got:\n{values=}"
                    )
            # Check that all cell_data dicts have the same keys across all meshes
            if not all(
                m.cell_data.keys() == meshes[0].cell_data.keys() for m in meshes
            ):
                raise ValueError("All meshes must have the same cell_data keys.")

        ### Merge the meshes

        # Compute the number of points for each mesh, cumulatively, so that we can update
        # the point indices for the constituent cells arrays accordingly.
        n_points_for_meshes = torch.tensor(
            [m.n_points for m in meshes],
            device=meshes[0].points.device,
        )
        cumsum_n_points = torch.cumsum(n_points_for_meshes, dim=0)
        cell_index_offsets = cumsum_n_points.roll(1)
        cell_index_offsets[0] = 0

        if global_data_strategy == "stack":
            global_data = TensorDict.stack([m.global_data for m in meshes])
        else:
            raise ValueError(f"Invalid {global_data_strategy=}")

        return cls(
            points=torch.cat([m.points for m in meshes], dim=0),
            cells=torch.cat(
                [m.cells + offset for m, offset in zip(meshes, cell_index_offsets)],
                dim=0,
            ),
            point_data=TensorDict.cat([m.point_data for m in meshes], dim=0),
            cell_data=TensorDict.cat([m.cell_data for m in meshes], dim=0),
            global_data=global_data,
        )

    def slice_points(
        self,
        indices: int
        | slice
        | type(Ellipsis)  # ty: ignore[invalid-type-form]
        | None
        | torch.Tensor
        | Sequence[int | bool],
    ) -> "Mesh":
        """Returns a new Mesh with a subset of the points.

        This method filters points and automatically updates cells to maintain
        consistency. Cells that reference any removed points are also removed,
        and the remaining cells have their indices remapped to the new point
        numbering.

        Parameters
        ----------
        indices : int or slice or Ellipsis or None or torch.Tensor or Sequence
            Indices or mask to select points. Supports:
            - ``int``: Single point index
            - ``slice``: Python slice object
            - ``Ellipsis`` or ``None``: Keep all points (returns self)
            - ``torch.Tensor``: Integer indices or boolean mask
            - ``Sequence[int | bool]``: List/tuple of indices or boolean mask

        Returns
        -------
        Mesh
            New Mesh with subset of points. Cells that reference any removed
            points are also removed, and remaining cell indices are remapped.

        Examples
        --------
        >>> import torch
        >>> from physicsnemo.mesh import Mesh
        >>> # Create a mesh with 4 points and 2 triangular cells
        >>> points = torch.tensor([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
        >>> cells = torch.tensor([[0, 1, 2], [0, 2, 3]])
        >>> mesh = Mesh(points=points, cells=cells)
        >>> # Keep only points 0 and 2 - both cells are removed (they need points 1 or 3)
        >>> sliced = mesh.slice_points([0, 2])
        >>> sliced.n_points, sliced.n_cells
        (2, 0)
        >>> # Keep points 0, 1, 2 - first cell is preserved with remapped indices
        >>> sliced = mesh.slice_points([0, 1, 2])
        >>> sliced.n_points, sliced.n_cells
        (3, 1)
        >>> sliced.cells.tolist()
        [[0, 1, 2]]
        """
        ### Handle no-op cases: None or Ellipsis means keep all points
        if indices is None or indices is ...:
            return self

        ### Normalize indices to a 1D tensor of point indices to keep
        all_indices = torch.arange(self.n_points, device=self.points.device)
        if isinstance(indices, int):
            kept_indices = torch.tensor([indices], device=self.points.device)
        else:
            # Works for slice, Tensor (int or bool), and Sequence
            kept_indices = all_indices[indices]

        ### Build old-to-new point index mapping
        # old_to_new[old_idx] = new_idx if kept, else -1
        old_to_new = torch.full(
            (self.n_points,), -1, dtype=torch.long, device=self.points.device
        )
        old_to_new[kept_indices] = torch.arange(
            len(kept_indices), dtype=torch.long, device=self.points.device
        )

        ### Remap cells and filter out cells with any removed vertices
        remapped_cells = old_to_new[self.cells]  # (n_cells, n_verts_per_cell)
        valid_cells_mask = (remapped_cells >= 0).all(
            dim=-1
        )  # cells with all verts kept

        ### Extract valid cells with remapped indices
        new_cells = remapped_cells[valid_cells_mask]
        new_cell_data: TensorDict = self.cell_data[valid_cells_mask]  # type: ignore

        ### Slice points and point_data
        new_points = self.points[kept_indices]
        new_point_data: TensorDict = self.point_data[kept_indices]  # type: ignore

        return Mesh(
            points=new_points,
            cells=new_cells,
            point_data=new_point_data,
            cell_data=new_cell_data,
            global_data=self.global_data,
        )

    def slice_cells(
        self,
        indices: int
        | slice
        | type(Ellipsis)  # ty: ignore[invalid-type-form]
        | None
        | torch.Tensor
        | Sequence[int | bool | slice],
    ) -> "Mesh":
        """Returns a new Mesh with a subset of the cells.

        Parameters
        ----------
        indices : int or slice or torch.Tensor
            Indices or mask to select cells.

        Returns
        -------
        Mesh
            New Mesh with subset of cells.
        """
        if isinstance(indices, int):
            indices = torch.tensor([indices], device=self.cells.device)
        new_cell_data: TensorDict = self.cell_data[indices]  # type: ignore
        return Mesh(
            points=self.points,
            cells=self.cells[indices],
            point_data=self.point_data,
            cell_data=new_cell_data,
            global_data=self.global_data,
        )

    def cell_data_to_point_data(self, overwrite_keys: bool = False) -> "Mesh":
        """Convert cell data to point data by averaging.

        For each point, computes the average of the cell data values from all cells
        that contain that point. The resulting point data is added to the mesh's
        point_data dictionary. Original cell data is preserved.

        Parameters
        ----------
        overwrite_keys : bool
            If True, silently overwrite any existing point_data keys.
            If False, raise an error if a key already exists in point_data.

        Returns
        -------
        Mesh
            New Mesh with converted data added to point_data. Original cell_data is preserved.

        Raises
        ------
        ValueError
            If a cell_data key already exists in point_data and overwrite_keys=False.

        Examples
        --------
        >>> mesh = Mesh(points, cells, cell_data={"pressure": cell_pressures})  # doctest: +SKIP
        >>> mesh_with_point_data = mesh.cell_data_to_point_data()  # doctest: +SKIP
        >>> # Now mesh has both cell_data["pressure"] and point_data["pressure"]
        """
        ### Check for key conflicts
        if not overwrite_keys:
            for key in self.cell_data.exclude("_cache").keys():
                if key in self.point_data.keys():
                    raise ValueError(
                        f"Key {key!r} already exists in point_data. "
                        f"Set overwrite_keys=True to overwrite."
                    )

        ### Convert each cell data field to point data
        new_point_data = self.point_data.clone()

        # Get flat list of point indices and corresponding cell indices
        # self.cells shape: (n_cells, n_vertices_per_cell)
        n_vertices_per_cell = self.cells.shape[1]

        # Flatten: all point indices that appear in cells
        # Shape: (n_cells * n_vertices_per_cell,)
        point_indices = self.cells.flatten()

        # Corresponding cell index for each point
        # Shape: (n_cells * n_vertices_per_cell,)
        cell_indices = torch.arange(
            self.n_cells, device=self.points.device
        ).repeat_interleave(n_vertices_per_cell)

        for key, cell_values in self.cell_data.exclude("_cache").items():
            ### Use scatter aggregation utility to average cell values to points
            # Expand cell values to one entry per vertex
            src_data = cell_values[cell_indices]

            # Aggregate to points using mean
            point_values = scatter_aggregate(
                src_data=src_data,
                src_to_dst_mapping=point_indices,
                n_dst=self.n_points,
                weights=None,
                aggregation="mean",
            )

            new_point_data[key] = point_values

        ### Return new mesh with updated point data
        return Mesh(
            points=self.points,
            cells=self.cells,
            point_data=new_point_data,
            cell_data=self.cell_data,
            global_data=self.global_data,
        )

    def point_data_to_cell_data(self, overwrite_keys: bool = False) -> "Mesh":
        """Convert point data to cell data by averaging.

        For each cell, computes the average of the point data values from all points
        (vertices) that define that cell. The resulting cell data is added to the mesh's
        cell_data dictionary. Original point data is preserved.

        Parameters
        ----------
        overwrite_keys : bool
            If True, silently overwrite any existing cell_data keys.
            If False, raise an error if a key already exists in cell_data.

        Returns
        -------
        Mesh
            New Mesh with converted data added to cell_data. Original point_data is preserved.

        Raises
        ------
        ValueError
            If a point_data key already exists in cell_data and overwrite_keys=False.

        Examples
        --------
        >>> mesh = Mesh(points, cells, point_data={"temperature": point_temps})  # doctest: +SKIP
        >>> mesh_with_cell_data = mesh.point_data_to_cell_data()  # doctest: +SKIP
        >>> # Now mesh has both point_data["temperature"] and cell_data["temperature"]
        """
        ### Check for key conflicts
        if not overwrite_keys:
            for key in self.point_data.exclude("_cache").keys():
                if key in self.cell_data.keys():
                    raise ValueError(
                        f"Key {key!r} already exists in cell_data. "
                        f"Set overwrite_keys=True to overwrite."
                    )

        ### Convert each point data field to cell data
        new_cell_data = self.cell_data.clone()

        for key, point_values in self.point_data.exclude("_cache").items():
            # Get point values for each cell and average
            # cell_point_values shape: (n_cells, n_vertices_per_cell, ...)
            cell_point_values = point_values[self.cells]

            # Average over vertices dimension (dim=1)
            cell_values = cell_point_values.mean(dim=1)

            new_cell_data[key] = cell_values

        ### Return new mesh with updated cell data
        return Mesh(
            points=self.points,
            cells=self.cells,
            point_data=self.point_data,
            cell_data=new_cell_data,
            global_data=self.global_data,
        )

    def pad(
        self,
        target_n_points: int | None = None,
        target_n_cells: int | None = None,
        data_padding_value: float = torch.nan,
    ) -> "Mesh":
        """Pad points and cells arrays to specified sizes.

        This is the low-level padding method that performs the actual padding operation.
        Padding uses null/degenerate elements that don't affect computations:
        - Points: Additional points at the last existing point (preserves bounding box)
        - cells: Degenerate cells with all vertices at the last existing point (zero area)
        - cell data: NaN-valued padding for all cell data fields (default)

        Parameters
        ----------
        target_n_points : int or None, optional
            Target number of points. If None, no point padding is applied.
            Must be >= current n_points if specified. Also accepts SymInt for torch.compile.
        target_n_cells : int or None, optional
            Target number of cells. If None, no cell padding is applied.
            Must be >= current n_cells if specified. Also accepts SymInt for torch.compile.
        data_padding_value : float
            Value to use for padding data fields. Defaults to NaN.

        Returns
        -------
        Mesh
            A new Mesh with padded arrays. If both targets are None or equal to
            current sizes, returns self unchanged.

        Raises
        ------
        ValueError
            If target sizes are less than current sizes.

        Examples
        --------
        >>> mesh = Mesh(points, cells)  # 100 points, 200 cells  # doctest: +SKIP
        >>> padded = mesh.pad(target_n_points=128, target_n_cells=256)  # doctest: +SKIP
        >>> padded.n_points  # 128  # doctest: +SKIP
        >>> padded.n_cells   # 256  # doctest: +SKIP
        """
        # Validate inputs
        if not torch.compiler.is_compiling():
            if target_n_points is not None and target_n_points < self.n_points:
                raise ValueError(f"{target_n_points=} must be >= {self.n_points=}")
            if target_n_cells is not None and target_n_cells < self.n_cells:
                raise ValueError(f"{target_n_cells=} must be >= {self.n_cells=}")

        # Short-circuit if no padding needed
        if target_n_points is None and target_n_cells is None:
            return self

        # Determine actual target sizes
        if target_n_points is None:
            target_n_points = self.n_points
        if target_n_cells is None:
            target_n_cells = self.n_cells

        return self.__class__(
            points=_pad_by_tiling_last(self.points, target_n_points),
            cells=_pad_with_value(self.cells, target_n_cells, self.n_points - 1),
            point_data=self.point_data.apply(  # type: ignore
                lambda x: _pad_with_value(x, target_n_points, data_padding_value),
                batch_size=torch.Size([target_n_points]),
            ),
            cell_data=self.cell_data.apply(  # type: ignore
                lambda x: _pad_with_value(x, target_n_cells, data_padding_value),
                batch_size=torch.Size([target_n_cells]),
            ),
            global_data=self.global_data,
        )

    def pad_to_next_power(
        self, power: float = 1.5, data_padding_value: float = torch.nan
    ) -> "Mesh":
        """Pads points and cells arrays to their next power of `power` (integer-floored).

        This is useful for torch.compile with dynamic=False, where fixed tensor shapes
        are required. By padding to powers of a base (default 1.5), we can reuse compiled
        kernels across a reasonable range of mesh sizes while minimizing memory overhead.

        This method computes the target sizes as floor(power^n) for the smallest n such that
        the result is >= the current size, then calls .pad() to perform the actual padding.

        Parameters
        ----------
        power : float
            Base for computing the next power. Must be > 1.
            Provides a good balance between memory efficiency and compile cache hits.
        data_padding_value : float
            Value to use for padding data fields. Defaults to NaN.

        Returns
        -------
        Mesh
            A new Mesh with padded points and cells arrays. The padding uses
            null elements that don't affect geometric computations.

        Raises
        ------
        ValueError
            If power <= 1.

        Examples
        --------
        >>> mesh = Mesh(points, cells)  # 100 points, 200 cells  # doctest: +SKIP
        >>> padded = mesh.pad_to_next_power(power=1.5)  # doctest: +SKIP
        >>> # Points padded to floor(1.5^n) >= 100, cells to floor(1.5^m) >= 200
        >>> # For power=1.5: 100 points -> 129 points, 200 cells -> 216 cells
        >>> # Padding cells have zero area and don't affect computations
        """
        if not torch.compiler.is_compiling():
            if power <= 1:
                raise ValueError(f"power must be > 1, got {power=}")

        def next_power_size(current_size: int, base: float) -> int:
            """Calculate the next power of base (integer-floored) that is >= current_size."""
            # Clamp to at least 1 to avoid log(0) = -inf
            # Mathematically correct: for current_size <= 1, result is base^0 = 1
            # max() works with both int and SymInt during torch.compile
            safe_size = max(current_size, 1)

            # Solve for n: floor(base^n) >= current_size
            # n >= log(current_size) / log(base)
            n = (torch.tensor(safe_size).log() / torch.tensor(base).log()).ceil()
            return int(torch.tensor(base) ** n)

        target_n_points = next_power_size(self.n_points, power)
        target_n_cells = next_power_size(self.n_cells, power)

        return self.pad(
            target_n_points=target_n_points,
            target_n_cells=target_n_cells,
            data_padding_value=data_padding_value,
        )

    def draw(
        self,
        backend: Literal["matplotlib", "pyvista", "auto"] = "auto",
        show: bool = True,
        point_scalars: None | torch.Tensor | str | tuple[str, ...] = None,
        cell_scalars: None | torch.Tensor | str | tuple[str, ...] = None,
        cmap: str = "viridis",
        vmin: float | None = None,
        vmax: float | None = None,
        alpha_points: float = 1.0,
        alpha_cells: float = 1.0,
        alpha_edges: float = 1.0,
        show_edges: bool = True,
        ax=None,
        **kwargs,
    ):
        """Draw the mesh using matplotlib or PyVista backend.

        Provides interactive 3D or 2D visualization with support for scalar data
        coloring, transparency control, and automatic backend selection.

        Parameters
        ----------
        backend : {"auto", "matplotlib", "pyvista"}
            Visualization backend to use:
            - "auto": Automatically select based on n_spatial_dims
              (matplotlib for 0D/1D/2D, PyVista for 3D)
            - "matplotlib": Force matplotlib backend (supports 3D via mplot3d)
            - "pyvista": Force PyVista backend (requires n_spatial_dims <= 3)
        show : bool
            Whether to display the plot immediately (calls plt.show() or
            plotter.show()). If False, returns the plotter/axes for further
            customization before display.
        point_scalars : torch.Tensor or str or tuple[str, ...], optional
            Scalar data to color points. Mutually exclusive with cell_scalars. Can be:
            - None: Points use neutral color (black)
            - torch.Tensor: Direct scalar values, shape (n_points,) or
              (n_points, ...) where trailing dimensions are L2-normed
            - str or tuple[str, ...]: Key to lookup in mesh.point_data
        cell_scalars : torch.Tensor or str or tuple[str, ...], optional
            Scalar data to color cells. Mutually exclusive with point_scalars. Can be:
            - None: Cells use neutral color (lightblue if no scalars,
              lightgray if point_scalars active)
            - torch.Tensor: Direct scalar values, shape (n_cells,) or
              (n_cells, ...) where trailing dimensions are L2-normed
            - str or tuple[str, ...]: Key to lookup in mesh.cell_data
        cmap : str
            Colormap name for scalar visualization.
        vmin : float, optional
            Minimum value for colormap normalization. If None, uses data min.
        vmax : float, optional
            Maximum value for colormap normalization. If None, uses data max.
        alpha_points : float
            Opacity for points, range [0, 1].
        alpha_cells : float
            Opacity for cells/faces, range [0, 1].
        alpha_edges : float
            Opacity for cell edges, range [0, 1].
        show_edges : bool
            Whether to draw cell edges.
        ax : matplotlib.axes.Axes, optional
            (matplotlib only) Existing matplotlib axes to plot on. If None,
            creates new figure and axes.
        **kwargs : dict
            Additional backend-specific keyword arguments.

        Returns
        -------
        matplotlib.axes.Axes or pyvista.Plotter
            - matplotlib backend: matplotlib.axes.Axes object
            - PyVista backend: pyvista.Plotter object

        Raises
        ------
        ValueError
            If both point_scalars and cell_scalars are specified,
            or if n_spatial_dims is not supported by the chosen backend.
        ImportError
            If the chosen backend (matplotlib or pyvista) is not installed.

        Examples
        --------
        >>> # Draw mesh with automatic backend selection
        >>> mesh.draw()  # doctest: +SKIP
        >>>
        >>> # Color cells by pressure data
        >>> mesh.draw(cell_scalars="pressure", cmap="coolwarm")  # doctest: +SKIP
        >>>
        >>> # Color points by velocity magnitude (computing norm of vector field)
        >>> mesh.draw(point_scalars="velocity")  # velocity is (n_points, 3)  # doctest: +SKIP
        >>>
        >>> # Use nested TensorDict key
        >>> mesh.draw(cell_scalars=("flow", "temperature"))  # doctest: +SKIP
        >>>
        >>> # Customize and display later
        >>> ax = mesh.draw(show=False, backend="matplotlib")  # doctest: +SKIP
        >>> ax.set_title("My Mesh")  # doctest: +SKIP
        >>> import matplotlib.pyplot as plt  # doctest: +SKIP
        >>> plt.show()  # doctest: +SKIP
        """
        return draw_mesh(
            mesh=self,
            backend=backend,
            show=show,
            point_scalars=point_scalars,
            cell_scalars=cell_scalars,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            alpha_points=alpha_points,
            alpha_cells=alpha_cells,
            alpha_edges=alpha_edges,
            show_edges=show_edges,
            ax=ax,
            **kwargs,
        )

    def translate(
        self,
        offset: torch.Tensor | list | tuple,
    ) -> "Mesh":
        """Apply a translation to the mesh.

        Convenience wrapper for physicsnemo.mesh.transformations.translate().

        Parameters
        ----------
        offset : torch.Tensor or list or tuple
            Translation vector, shape (n_spatial_dims,).

        Returns
        -------
        Mesh
            New Mesh with translated geometry.
        """
        return translate(self, offset)

    def rotate(
        self,
        angle: float,
        axis: torch.Tensor | list | tuple | None = None,
        center: torch.Tensor | list | tuple | None = None,
        transform_point_data: bool = False,
        transform_cell_data: bool = False,
        transform_global_data: bool = False,
    ) -> "Mesh":
        """Rotate the mesh about an axis by a specified angle.

        Convenience wrapper for physicsnemo.mesh.transformations.rotate().

        Parameters
        ----------
        angle : float
            Rotation angle in radians.
        axis : torch.Tensor or list or tuple, optional
            Rotation axis vector. None for 2D, shape (3,) for 3D.
        center : torch.Tensor or list or tuple, optional
            Center point for rotation.
        transform_point_data : bool
            If True, rotate vector/tensor fields in point_data.
        transform_cell_data : bool
            If True, rotate vector/tensor fields in cell_data.
        transform_global_data : bool
            If True, rotate vector/tensor fields in global_data.

        Returns
        -------
        Mesh
            New Mesh with rotated geometry.
        """
        return rotate(
            self,
            angle,
            axis,
            center,
            transform_point_data,
            transform_cell_data,
            transform_global_data,
        )

    def scale(
        self,
        factor: float | torch.Tensor,
        center: torch.Tensor | None = None,
        transform_point_data: bool = False,
        transform_cell_data: bool = False,
        transform_global_data: bool = False,
        assume_invertible: bool | None = None,
    ) -> "Mesh":
        """Scale the mesh by specified factor(s).

        Convenience wrapper for physicsnemo.mesh.transformations.scale().

        Parameters
        ----------
        factor : float or torch.Tensor
            Scale factor (scalar) or factors (per-dimension).
        center : torch.Tensor, optional
            Center point for scaling.
        transform_point_data : bool
            If True, scale vector/tensor fields in point_data.
        transform_cell_data : bool
            If True, scale vector/tensor fields in cell_data.
        transform_global_data : bool
            If True, scale vector/tensor fields in global_data.
        assume_invertible : bool or None, optional
            Controls cache propagation:
            - True: Assume all factors are non-zero (compile-safe).
            - False: Skip cache propagation (compile-safe).
            - None: Check at runtime (may cause graph breaks).

        Returns
        -------
        Mesh
            New Mesh with scaled geometry.
        """
        return scale(
            self,
            factor,
            center,
            transform_point_data,
            transform_cell_data,
            transform_global_data,
            assume_invertible,
        )

    def transform(
        self,
        matrix: torch.Tensor,
        transform_point_data: bool = False,
        transform_cell_data: bool = False,
        transform_global_data: bool = False,
        assume_invertible: bool | None = None,
    ) -> "Mesh":
        """Apply a linear transformation to the mesh.

        Convenience wrapper for physicsnemo.mesh.transformations.transform().

        Parameters
        ----------
        matrix : torch.Tensor
            Transformation matrix, shape (new_n_spatial_dims, n_spatial_dims).
        transform_point_data : bool
            If True, transform vector/tensor fields in point_data.
        transform_cell_data : bool
            If True, transform vector/tensor fields in cell_data.
        transform_global_data : bool
            If True, transform vector/tensor fields in global_data.
        assume_invertible : bool or None, optional
            Controls cache propagation for square matrices:
            - True: Assume matrix is invertible (compile-safe).
            - False: Skip cache propagation (compile-safe).
            - None: Check at runtime (may cause graph breaks).

        Returns
        -------
        Mesh
            New Mesh with transformed geometry.
        """
        return transform(
            self,
            matrix,
            transform_point_data,
            transform_cell_data,
            transform_global_data,
            assume_invertible,
        )


### Override the tensorclass __repr__ with custom formatting
# Note: Must be done after class definition because @tensorclass overrides __repr__
# even when defined inside the class body
def _mesh_repr(self) -> str:
    return format_mesh_repr(self, exclude_cache=False)


Mesh.__repr__ = _mesh_repr  # type: ignore
