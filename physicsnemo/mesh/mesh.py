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

import math
import types
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Self, Sequence, cast

import torch
import torch.nn.functional as F
from tensordict import TensorDict, tensorclass

from physicsnemo.mesh.geometry._cell_areas import compute_cell_areas
from physicsnemo.mesh.geometry._cell_normals import compute_cell_normals
from physicsnemo.mesh.transformations.geometric import (
    rotate,
    scale,
    transform,
    translate,
)
from physicsnemo.mesh.utilities._padding import _pad_by_tiling_last, _pad_with_value
from physicsnemo.mesh.utilities._scatter_ops import scatter_aggregate
from physicsnemo.mesh.utilities.mesh_repr import format_mesh_repr
from physicsnemo.mesh.visualization.draw_mesh import draw_mesh


@tensorclass(tensor_only=True, shadow=True)
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

    **Dimension-Parametrized Types**

    ``Mesh`` supports subscript notation ``Mesh[manifold_dims, spatial_dims]``
    for type annotations and runtime ``isinstance`` checks::

        def compute_normals(mesh: Mesh[2, 3]) -> torch.Tensor:
            ...  # accepts only triangle meshes in 3D

        isinstance(mesh, Mesh[2, 3])   # True for triangles in 3D
        isinstance(mesh, Mesh[1, ...]) # True for any edge mesh

    Use ``...`` (Ellipsis) to leave a dimension unconstrained. The notation
    also supports ``.boundary`` to derive the boundary type::

        Mesh[2, 3].boundary  # -> Mesh[1, 3]

    See :meth:`__class_getitem__` for the full specification, including
    symbolic dimension expressions like ``Mesh["n-1", "n"]``.

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
    cells : torch.Tensor, optional
        Cell connectivity with shape :math:`(N_c, D_m + 1)`. Each row contains
        indices into ``points`` defining one simplex. Must be integer dtype.
        Defaults to an empty 0-simplex tensor for point-cloud meshes.
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

    Create a point cloud (no connectivity):

    >>> points = torch.randn(50, 3)
    >>> cloud = Mesh(points=points)
    >>> cloud.n_points, cloud.n_cells, cloud.n_manifold_dims
    (50, 0, 0)

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

    **Immutability**

    ``Mesh`` operations return new instances rather than modifying in place.
    For example, ``mesh.translate(offset)`` returns a new ``Mesh`` with
    translated points -- the original ``mesh`` is unchanged. This design
    enables safe caching of derived geometry (centroids, normals, curvature):
    cached values remain valid because the underlying ``points`` and ``cells``
    never change after construction.

    .. important::

       In-place modification of ``points`` or ``cells`` (e.g.,
       ``mesh.points[0] = ...``) is unsupported and will **silently
       invalidate** all cached properties. Always construct a new ``Mesh``
       instead.

    **Caching**

    Expensive geometric computations (centroids, areas, normals, curvature,
    adjacency) are cached in the ``_cache`` field -- a nested ``TensorDict``
    with ``"cell"``, ``"point"``, and ``"topology"`` sub-dicts. The cache is
    separate from ``point_data`` / ``cell_data``, so user data is never mixed
    with internal cached geometry.

    Caches are populated lazily on first access (e.g., the first call to
    ``mesh.cell_normals`` computes and caches the result; subsequent calls
    return the cached value). Because ``Mesh`` is effectively immutable (see
    above), cached values never go stale -- they remain valid for the
    lifetime of the ``Mesh`` instance.

    Geometric transforms (``translate``, ``rotate``, ``scale``, ``transform``)
    carry forward applicable cache entries to the new ``Mesh`` rather than
    discarding them. Topology caches are always preserved (transforms do not
    change connectivity). Geometric caches (areas, normals, centroids) are
    re-derived from the transform matrix where possible, avoiding
    recomputation from raw vertex data.

    Slicing operations (``slice_cells``, ``slice_points``) produce new
    ``Mesh`` instances with topology caches cleared (since connectivity has
    changed) but cell-level geometric caches sliced in lockstep.

    Access cached values directly via nested keys::

        mesh._cache["cell", "centroids"]   # shape (n_cells, n_dims)
        mesh._cache["point", "normals"]    # shape (n_points, n_dims)

    Prefer using properties (``mesh.cell_centroids``, ``mesh.point_normals``)
    over direct ``_cache`` access.
    """

    points: torch.Tensor  # shape: (n_points, n_spatial_dimensions)
    cells: torch.Tensor  # shape: (n_cells, n_manifold_dimensions + 1)
    point_data: TensorDict
    cell_data: TensorDict
    global_data: TensorDict
    _cache: TensorDict

    def __init__(
        self,
        points: torch.Tensor,
        cells: torch.Tensor | None = None,
        point_data: TensorDict | dict[str, torch.Tensor] | None = None,
        cell_data: TensorDict | dict[str, torch.Tensor] | None = None,
        global_data: TensorDict | dict[str, torch.Tensor] | None = None,
        *,
        _cache: TensorDict | None = None,
    ) -> None:
        self.points = points
        self.cells = cells  # type: ignore[assignment]  # normalized by __post_init__
        # The tensorclass setter silently drops entries from non-dict Mappings
        # (e.g. PyVista DataSetAttributes). Wrapping with dict() converts any
        # Mapping to a plain dict that the setter handles correctly.
        self.point_data = (  # type: ignore[assignment]  # normalized by __post_init__
            dict(point_data)
            if point_data is not None and not isinstance(point_data, TensorDict)
            else point_data
        )
        self.cell_data = (  # type: ignore[assignment]  # normalized by __post_init__ (coerced to TensorDict)
            dict(cell_data)
            if cell_data is not None and not isinstance(cell_data, TensorDict)
            else cell_data
        )
        self.global_data = (  # type: ignore[assignment]  # normalized by __post_init__
            dict(global_data)
            if global_data is not None and not isinstance(global_data, TensorDict)
            else global_data
        )
        self._cache = _cache  # type: ignore[assignment]  # normalized by __post_init__
        # tensorclass only auto-calls __post_init__ from the *generated* __init__
        # (same semantics as dataclasses). Since we define a custom __init__,
        # we must call it explicitly. During load(), tensorclass calls it
        # automatically, so __post_init__ is the single source of truth for
        # defaults, coercions, and validation.
        self.__post_init__()

    def __post_init__(self):
        """Normalize fields and validate invariants.

        Called automatically during ``load()`` by tensorclass, and explicitly
        from ``__init__`` during normal construction. This is the single source
        of truth for all default values, type coercions, and shape validation.
        """
        ### cells: default empty-cells sentinel for point clouds
        # The tensordict memmap format does not persist tensors with 0 elements,
        # so this also restores cells after deserialization.
        if self.cells is None:
            self.cells = torch.zeros(0, 1, dtype=torch.long, device=self.points.device)

        ### point_data: coerce dict -> TensorDict and enforce batch_size
        if isinstance(self.point_data, TensorDict):
            self.point_data.batch_size = torch.Size([self.n_points])
        else:
            self.point_data = TensorDict(
                {} if self.point_data is None else dict(self.point_data),
                batch_size=torch.Size([self.n_points]),
                device=self.points.device,
            )

        ### cell_data: coerce dict -> TensorDict and enforce batch_size
        if isinstance(self.cell_data, TensorDict):
            self.cell_data.batch_size = torch.Size([self.n_cells])
        else:
            self.cell_data = TensorDict(
                {} if self.cell_data is None else dict(self.cell_data),
                batch_size=torch.Size([self.n_cells]),
                device=self.cells.device,
            )

        ### global_data: coerce dict -> TensorDict and enforce batch_size
        if isinstance(self.global_data, TensorDict):
            self.global_data.batch_size = torch.Size([])
        else:
            self.global_data = TensorDict(
                {} if self.global_data is None else dict(self.global_data),
                device=self.points.device,
            )

        ### _cache: default empty cache structure
        if self._cache is None:
            self._cache = TensorDict(
                {
                    "cell": TensorDict({}, batch_size=[self.n_cells]),
                    "point": TensorDict({}, batch_size=[self.n_points]),
                    "topology": TensorDict({}),
                },
                device=self.points.device,
            )

        ### Validate shapes and dtypes
        if not torch.compiler.is_compiling():
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
            if self.points.device != self.cells.device:
                raise ValueError(
                    f"`points` and `cells` must be on the same device, "
                    f"but got {self.points.device=} and {self.cells.device=}."
                )

    @classmethod
    def __class_getitem__(cls, params: tuple) -> type:
        r"""Parametrize Mesh by manifold and spatial dimensions.

        Returns a synthetic type usable in type annotations and ``isinstance``
        checks. Always requires exactly two parameters; use ``...`` (Ellipsis)
        to leave a dimension unconstrained.

        Parameters
        ----------
        params : tuple
            A 2-tuple of ``(manifold_dims, spatial_dims)`` where each element
            is an ``int`` (concrete), ``str`` (symbolic, e.g. ``"n-1"``), or
            ``...`` (unconstrained).

        Returns
        -------
        type
            A parametrized Mesh type supporting ``isinstance`` checks.

        Raises
        ------
        TypeError
            If not exactly 2 parameters, or if parameter types are invalid.
        ValueError
            If concrete dimensions are negative or manifold exceeds spatial.

        Examples
        --------
        >>> Mesh[2, 3]
        Mesh[2, 3]
        >>> Mesh[1, ...]
        Mesh[1, ...]
        >>> Mesh[2, 3].boundary
        Mesh[1, 3]
        """
        from physicsnemo.mesh._mesh_spec import MeshDims, _get_mesh_spec

        if not isinstance(params, tuple):
            raise TypeError(
                f"Mesh[...] requires exactly 2 parameters (e.g. Mesh[2, 3] "
                f"or Mesh[2, ...]), got single parameter {params!r}"
            )
        if len(params) != 2:
            raise TypeError(
                f"Mesh[...] requires exactly 2 parameters, got {len(params)}"
            )

        n_manifold_dims = None if params[0] is ... else params[0]
        n_spatial_dims = None if params[1] is ... else params[1]

        return _get_mesh_spec(
            MeshDims(n_manifold_dims=n_manifold_dims, n_spatial_dims=n_spatial_dims)
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

        def clone(self) -> Self:
            """Return a deep clone of this Mesh.

            All tensors are copied (independent storage); the clone can
            be modified without affecting the original.
            """
            ...

        def save(
            self,
            prefix: str | Path | None = None,
            copy_existing: bool = False,
            *,
            num_threads: int = 0,
            return_early: bool = False,
            share_non_tensor: bool = False,
        ) -> Self:
            """Save the mesh to disk as memory-mapped tensors.

            Writes ``points``, ``cells``, ``point_data``, ``cell_data``,
            ``global_data``, and ``_cache`` to a directory tree of
            ``.memmap`` files.  Proxy for the tensorclass ``memmap()``
            method.

            This is the recommended serialization method. Compared to
            ``torch.save`` (pickle-based), memmap serialization is
            faster (parallel I/O across files), safer (no arbitrary code
            execution on load), and supports partial loading.

            Parameters
            ----------
            prefix : str, Path, or None
                Directory path where the memory-mapped files will be
                written.  If ``None``, a temporary directory is used.
            copy_existing : bool
                If ``True``, copy tensors that are already memory-mapped
                to the new location.
            num_threads : int
                Number of threads for parallel I/O (0 = sequential).
            return_early : bool
                If ``True``, return before all data is flushed to disk.
            share_non_tensor : bool
                If ``True``, share non-tensor data across processes.

            Returns
            -------
            Mesh
                A new Mesh backed by the on-disk memory-mapped storage.

            Examples
            --------
            >>> mesh.save("/path/to/mesh")  # doctest: +SKIP
            >>> reloaded = Mesh.load("/path/to/mesh")  # doctest: +SKIP
            """
            ...

        @classmethod
        def load(
            cls,
            prefix: str | Path,
            device: torch.device | None = None,
            non_blocking: bool = False,
        ) -> Self:
            """Load a previously saved mesh from disk.

            Reads a directory tree of memory-mapped tensors written by
            :meth:`save` and reconstructs the ``Mesh`` instance,
            including all attached ``point_data``, ``cell_data``, and
            ``global_data``.  Proxy for the tensorclass
            ``load_memmap()`` class method.

            Parameters
            ----------
            prefix : str or Path
                Path to the directory created by :meth:`save`.
            device : torch.device or None
                If provided, move all tensors to this device after
                loading.
            non_blocking : bool
                Whether device transfers should be non-blocking.

            Returns
            -------
            Mesh
                The reconstructed Mesh instance.

            Examples
            --------
            >>> mesh = Mesh.load("/path/to/mesh")  # doctest: +SKIP
            >>> mesh_gpu = Mesh.load("/path/to/mesh", device="cuda")  # doctest: +SKIP
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
        For an n-simplex with vertices (v0, v1, ..., vn), the centroid is
        ``centroid = (v0 + v1 + ... + vn) / (n + 1)``.

        The result is cached in ``_cache["cell", "centroids"]`` for efficiency.

        Returns
        -------
        torch.Tensor
            Tensor of shape (n_cells, n_spatial_dims) containing the centroid of each cell.
        """
        cached = self._cache.get(("cell", "centroids"), None)
        if cached is None:
            cached = self.points[self.cells].mean(dim=1)
            self._cache["cell", "centroids"] = cached
        return cached

    @property
    def cell_areas(self) -> torch.Tensor:
        """Compute volumes (areas) of n-simplices.

        This works for simplices of any manifold dimension embedded in any spatial dimension.
        For example: edges in 2D/3D, triangles in 2D/3D/4D, tetrahedra in 3D/4D, etc.

        Uses dimension-specific closed-form expressions for n <= 3 (Lagrange
        identity, scalar triple product, etc.) and falls back to the Gram
        determinant for higher dimensions.  See
        :func:`~physicsnemo.mesh.geometry._cell_areas.compute_cell_areas` for
        details.

        Returns
        -------
        torch.Tensor
            Tensor of shape (n_cells,) containing the volume of each cell.
        """
        cached = self._cache.get(("cell", "areas"), None)
        if cached is None:
            relative_vectors = (
                self.points[self.cells[:, 1:]] - self.points[self.cells[:, [0]]]
            )
            cached = compute_cell_areas(relative_vectors)
            self._cache["cell", "areas"] = cached

        return cached

    @property
    def cell_normals(self) -> torch.Tensor:
        """Compute unit normal vectors for codimension-1 cells.

        Normal vectors are uniquely defined (up to orientation) only for
        codimension-1 manifolds, where ``n_manifold_dims = n_spatial_dims - 1``.

        Uses dimension-specific closed-form expressions for d=2 (rotation)
        and d=3 (cross product), falling back to signed minor determinants
        for higher dimensions.  See
        :func:`~physicsnemo.mesh.geometry._cell_normals.compute_cell_normals`
        for details.

        Returns
        -------
        torch.Tensor
            Tensor of shape (n_cells, n_spatial_dims) containing unit normal vectors.

        Raises
        ------
        ValueError
            If the mesh is not codimension-1 (n_manifold_dims ≠ n_spatial_dims - 1).
        """
        cached = self._cache.get(("cell", "normals"), None)
        if cached is None:
            if self.codimension != 1:
                raise ValueError(
                    f"cell normals are only defined for codimension-1 manifolds.\n"
                    f"Got {self.n_manifold_dims=} and {self.n_spatial_dims=}.\n"
                    f"Required: n_manifold_dims = n_spatial_dims - 1 (codimension-1).\n"
                    f"Current codimension: {self.codimension}"
                )
            relative_vectors = (
                self.points[self.cells[:, 1:]] - self.points[self.cells[:, [0]]]
            )
            cached = compute_cell_normals(relative_vectors)
            self._cache["cell", "normals"] = cached

        return cached

    @property
    def point_normals(self) -> torch.Tensor:
        """Compute weighted normal vectors at mesh vertices.

        This property returns the canonical/default point normals. For 2D+
        manifolds (surfaces, volumes), angle-area weighting is used, which
        balances face area and vertex interior angle for high-quality normals.
        For 1D manifolds (curves), area weighting (i.e. segment-length
        weighting) is used, since interior angles are not defined for edges.

        For explicit weighting control, use :meth:`compute_point_normals`.

        The result is cached in ``_cache["point", "normals"]`` for efficiency.

        Returns
        -------
        torch.Tensor
            Tensor of shape (n_points, n_spatial_dims) containing unit normal vectors
            at each vertex. For isolated points (with no adjacent cells), the normal
            is a zero vector.

        Raises
        ------
        ValueError
            If the mesh is not codimension-1 (n_manifold_dims != n_spatial_dims - 1).

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
        cached = self._cache.get(("point", "normals"), None)
        if cached is None:
            weighting = "area" if self.n_manifold_dims < 2 else "angle_area"
            cached = self.compute_point_normals(weighting=weighting)
            self._cache["point", "normals"] = cached
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

        - **"area"**: Area-weighted averaging, where larger faces have more
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

        - **"angle_area"** (default): Combined angle and area weighting, where each face's
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
            from physicsnemo.mesh.geometry._angles import compute_vertex_angles

            vertex_angles = compute_vertex_angles(
                self
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

    @property
    def gaussian_curvature_vertices(self) -> torch.Tensor:
        """Compute intrinsic Gaussian curvature at mesh vertices.

        Uses the angle defect method from discrete differential geometry:
            K = (full_angle - Σ angles) / voronoi_area

        This is an intrinsic measure of curvature (Theorema Egregium) that works
        for any codimension, as it depends only on distances within the manifold.

        Signed curvature:
        - Positive: Elliptic/convex (sphere-like)
        - Zero: Flat/parabolic (plane-like)
        - Negative: Hyperbolic/saddle (saddle-like)

        The result is cached in ``_cache["point", "gaussian_curvature"]`` for efficiency.

        Returns
        -------
        torch.Tensor
            Tensor of shape (n_points,) containing signed Gaussian curvature.
            Isolated vertices have NaN curvature.

        Notes
        -----
        Satisfies discrete Gauss-Bonnet theorem:
            Σ_vertices (K_i * A_i) = 2π * χ(M)

        Examples
        --------
        >>> from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral
        >>> # Sphere of radius r has K = 1/r²
        >>> sphere = sphere_icosahedral.load(radius=2.0, subdivisions=3)
        >>> K = sphere.gaussian_curvature_vertices
        >>> # K.mean() ≈ 0.25 (= 1/(2.0)²)
        """
        cached = self._cache.get(("point", "gaussian_curvature"), None)
        if cached is None:
            from physicsnemo.mesh.curvature import gaussian_curvature_vertices

            cached = gaussian_curvature_vertices(self)
            self._cache["point", "gaussian_curvature"] = cached

        return cached

    @property
    def gaussian_curvature_cells(self) -> torch.Tensor:
        """Compute Gaussian curvature at cell centers using dual mesh concept.

        Treats cell centroids as vertices of a dual mesh and computes curvature
        based on angles between connections to adjacent cell centroids.

        The result is cached in ``_cache["cell", "gaussian_curvature"]`` for efficiency.

        Returns
        -------
        torch.Tensor
            Tensor of shape (n_cells,) containing Gaussian curvature at cells.

        Examples
        --------
        >>> from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral
        >>> mesh = sphere_icosahedral.load(subdivisions=2)
        >>> K_cells = mesh.gaussian_curvature_cells
        """
        cached = self._cache.get(("cell", "gaussian_curvature"), None)
        if cached is None:
            from physicsnemo.mesh.curvature import gaussian_curvature_cells

            cached = gaussian_curvature_cells(self)
            self._cache["cell", "gaussian_curvature"] = cached

        return cached

    @property
    def mean_curvature_vertices(self) -> torch.Tensor:
        """Compute extrinsic mean curvature at mesh vertices.

        Uses the cotangent Laplace-Beltrami operator:
            H = (1/2) * ||L @ points|| / voronoi_area

        Mean curvature is an extrinsic measure (depends on embedding) and is
        only defined for codimension-1 manifolds where normal vectors exist.

        For 2D surfaces: H = (k1 + k2) / 2 where k1, k2 are principal curvatures

        Signed curvature:
        - Positive: Convex (sphere exterior with outward normals)
        - Negative: Concave (sphere interior with outward normals)
        - Zero: Minimal surface (soap film)

        The result is cached in ``_cache["point", "mean_curvature"]`` for efficiency.

        Returns
        -------
        torch.Tensor
            Tensor of shape (n_points,) containing signed mean curvature.
            Isolated vertices have NaN curvature.

        Raises
        ------
        ValueError
            If mesh is not codimension-1.

        Examples
        --------
        >>> from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral
        >>> # Sphere of radius r has H = 1/r
        >>> sphere = sphere_icosahedral.load(radius=2.0, subdivisions=3)
        >>> H = sphere.mean_curvature_vertices
        >>> # H.mean() ≈ 0.5 (= 1/2.0)
        """
        cached = self._cache.get(("point", "mean_curvature"), None)
        if cached is None:
            from physicsnemo.mesh.curvature import mean_curvature_vertices

            cached = mean_curvature_vertices(self)
            self._cache["point", "mean_curvature"] = cached

        return cached

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
            elif len(meshes) == 1:  # Return a shallow copy to avoid aliasing
                return meshes[0].clone()
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
            ref_keys = set(
                meshes[0].cell_data.keys(include_nested=True, leaves_only=True)
            )
            if not all(
                set(m.cell_data.keys(include_nested=True, leaves_only=True)) == ref_keys
                for m in meshes
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
        | types.EllipsisType
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
        # cast: TensorDict[bool_mask] returns TensorCollection | Tensor statically;
        # the runtime is always TensorDict because cell_data is itself a TensorDict.
        new_cell_data = cast(TensorDict, self.cell_data[valid_cells_mask])

        ### Slice points and point_data
        new_points = self.points[kept_indices]
        new_point_data = cast(TensorDict, self.point_data[kept_indices])

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
        | types.EllipsisType
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
        new_cell_data = cast(TensorDict, self.cell_data[indices])
        new_cache = TensorDict(
            {
                "cell": self._cache["cell"][indices],
                "point": self._cache["point"],
                "topology": TensorDict({}),
            },
            device=self.points.device,
        )
        return Mesh(
            points=self.points,
            cells=self.cells[indices],
            point_data=self.point_data,
            cell_data=new_cell_data,
            global_data=self.global_data,
            _cache=new_cache,
        )

    def sample_random_points_on_cells(
        self,
        cell_indices: Sequence[int] | torch.Tensor | None = None,
        alpha: float = 1.0,
    ) -> torch.Tensor:
        """Sample random points on specified cells of the mesh.

        Uses a Dirichlet distribution to generate barycentric coordinates, which are
        then used to compute random points as weighted combinations of cell vertices.
        The concentration parameter alpha controls the distribution of samples within
        each cell (simplex).

        This is a convenience method that delegates to physicsnemo.mesh.sampling.sample_random_points_on_cells.

        Parameters
        ----------
        cell_indices : Sequence[int] or torch.Tensor or None, optional
            Indices of cells to sample from. Can be a Sequence or tensor.
            Allows repeated indices to sample multiple points from the same cell.
            If None, samples one point from each cell (equivalent to arange(n_cells)).
            Shape: (n_samples,) where n_samples is the number of points to sample.
        alpha : float, optional
            Concentration parameter for the Dirichlet distribution. Controls how
            samples are distributed within each cell:
            - alpha = 1.0: Uniform distribution over the simplex (default)
            - alpha > 1.0: Concentrates samples toward the center of each cell
            - alpha < 1.0: Concentrates samples toward vertices and edges

        Returns
        -------
        torch.Tensor
            Random points on cells, shape (n_samples, n_spatial_dims). Each point lies
            within its corresponding cell. If cell_indices is None, n_samples = n_cells.

        Raises
        ------
        NotImplementedError
            If alpha != 1.0 and torch.compile is being used.
            This is due to a PyTorch limitation with Gamma distributions under torch.compile.
        IndexError
            If any cell_indices are out of bounds.

        Examples
        --------
        >>> import torch
        >>> from physicsnemo.mesh.primitives.basic import two_triangles_2d
        >>> mesh = two_triangles_2d.load()
        >>> # Sample one point from each cell uniformly
        >>> points = mesh.sample_random_points_on_cells()
        >>> assert points.shape == (mesh.n_cells, mesh.n_spatial_dims)
        """
        from physicsnemo.mesh.sampling import sample_random_points_on_cells

        return sample_random_points_on_cells(
            mesh=self,
            cell_indices=cell_indices,
            alpha=alpha,
        )

    def sample_data_at_points(
        self,
        query_points: torch.Tensor,
        data_source: Literal["cells", "points"] = "cells",
        multiple_cells_strategy: Literal["mean", "nan"] = "mean",
        project_onto_nearest_cell: bool = False,
        tolerance: float = 1e-6,
        bvh: Any = None,
    ) -> "TensorDict":
        """Extract or interpolate mesh data at specified query points.

        This method retrieves mesh data at arbitrary spatial locations. Note that
        "sample" here means "extract/query at specific points" - NOT random sampling.
        For random point sampling, see :meth:`sample_random_points_on_cells`.

        Containment queries are BVH-accelerated (O(n_queries * log(n_cells))).

        Parameters
        ----------
        query_points : torch.Tensor
            Query point locations, shape (n_queries, n_spatial_dims).
        data_source : {"cells", "points"}, optional
            How to retrieve data:
            - "cells": Use cell data directly (no interpolation)
            - "points": Interpolate point data using barycentric coordinates
        multiple_cells_strategy : {"mean", "nan"}, optional
            How to handle query points in multiple cells:
            - "mean": Return arithmetic mean of values from all containing cells
            - "nan": Return NaN for ambiguous points
        project_onto_nearest_cell : bool, optional
            If True, snaps each query point to the centroid of the nearest cell
            before containment testing. Useful for codimension != 0 manifolds.
        tolerance : float, optional
            Tolerance for considering a point inside a cell.
        bvh : BVH or None, optional
            Pre-built Bounding Volume Hierarchy. If ``None`` (default), one is
            built automatically. For repeated queries, pre-build with
            ``BVH.from_mesh(mesh)`` and pass it here to avoid redundant work.

        Returns
        -------
        TensorDict
            Data for each query point. Values are NaN for query points outside
            the mesh.

        Examples
        --------
        >>> import torch
        >>> from physicsnemo.mesh.primitives.basic import two_triangles_2d
        >>> mesh = two_triangles_2d.load()
        >>> mesh.cell_data["pressure"] = torch.tensor([1.0, 2.0])
        >>> query_pts = torch.tensor([[0.3, 0.3], [0.8, 0.5]])
        >>> data = mesh.sample_data_at_points(query_pts, data_source="cells")
        """
        from physicsnemo.mesh.sampling import sample_data_at_points

        return sample_data_at_points(
            mesh=self,
            query_points=query_points,
            data_source=data_source,
            multiple_cells_strategy=multiple_cells_strategy,
            project_onto_nearest_cell=project_onto_nearest_cell,
            tolerance=tolerance,
            bvh=bvh,
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
            src_keys = set(self.cell_data.keys(include_nested=True, leaves_only=True))
            dst_keys = set(self.point_data.keys(include_nested=True, leaves_only=True))
            conflicts = src_keys & dst_keys
            if conflicts:
                raise ValueError(
                    f"Keys {conflicts} already exist in point_data. "
                    f"Set overwrite_keys=True to overwrite."
                )

        ### Convert each cell data field to point data via scatter aggregation
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

        converted = self.cell_data.apply(
            lambda cell_values: scatter_aggregate(
                src_data=cell_values[cell_indices],
                src_to_dst_mapping=point_indices,
                n_dst=self.n_points,
                weights=None,
                aggregation="mean",
            ),
            batch_size=torch.Size([self.n_points]),
        )
        new_point_data.update(converted)

        ### Return new mesh with updated point data
        return Mesh(
            points=self.points,
            cells=self.cells,
            point_data=new_point_data,
            cell_data=self.cell_data,
            global_data=self.global_data,
            _cache=self._cache,
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
            src_keys = set(self.point_data.keys(include_nested=True, leaves_only=True))
            dst_keys = set(self.cell_data.keys(include_nested=True, leaves_only=True))
            conflicts = src_keys & dst_keys
            if conflicts:
                raise ValueError(
                    f"Keys {conflicts} already exist in cell_data. "
                    f"Set overwrite_keys=True to overwrite."
                )

        ### Convert each point data field to cell data by averaging over cell vertices
        new_cell_data = self.cell_data.clone()

        converted = self.point_data.apply(
            lambda point_values: point_values[self.cells].mean(dim=1),
            batch_size=torch.Size([self.n_cells]),
        )
        new_cell_data.update(converted)

        ### Return new mesh with updated cell data
        return Mesh(
            points=self.points,
            cells=self.cells,
            point_data=self.point_data,
            cell_data=new_cell_data,
            global_data=self.global_data,
            _cache=self._cache,
        )

    def get_facet_mesh(
        self,
        manifold_codimension: int = 1,
        data_source: Literal["points", "cells"] = "cells",
        data_aggregation: Literal["mean", "area_weighted", "inverse_distance"] = "mean",
        target_counts: "list[int] | Literal['boundary', 'shared', 'interior', 'all']" = "all",
    ) -> "Mesh":
        """Extract k-codimension facet mesh from this n-dimensional mesh.

        Extracts all (n-k)-simplices from the current n-simplicial mesh. For example:
        - Triangle mesh (2-simplices) → edge mesh (1-simplices) [codimension=1, default]
        - Triangle mesh (2-simplices) → vertex mesh (0-simplices) [codimension=2]
        - Tetrahedral mesh (3-simplices) → triangular facet mesh (2-simplices) [codimension=1, default]
        - Tetrahedral mesh (3-simplices) → edge mesh (1-simplices) [codimension=2]

        The resulting mesh shares the same vertex positions but has connectivity
        representing the lower-dimensional simplices. Data can be inherited from
        either the parent cells or the boundary points.

        Parameters
        ----------
        manifold_codimension : int, optional
            Codimension of extracted mesh relative to parent.

            - 1: Extract (n-1)-facets (default, immediate boundaries of all cells)
            - 2: Extract (n-2)-facets (e.g., edges from tets, vertices from triangles)
            - k: Extract (n-k)-facets
        data_source : {"points", "cells"}, optional
            Source of data inheritance:

            - "cells": Facets inherit from parent cells they bound. When multiple
              cells share a facet, data is aggregated according to data_aggregation.
            - "points": Facets inherit from their boundary vertices. Data from
              multiple boundary points is averaged.
        data_aggregation : {"mean", "area_weighted", "inverse_distance"}, optional
            Strategy for aggregating data from multiple sources
            (only applies when data_source="cells"):

            - "mean": Simple arithmetic mean
            - "area_weighted": Weighted by parent cell areas
            - "inverse_distance": Weighted by inverse distance from facet centroid
              to parent cell centroids
        target_counts : list[int] | {"boundary", "shared", "interior", "all"}, optional
            Which facets to keep based on how many parent cells share them:

            - "all": Keep all unique facets (default)
            - "boundary": Keep only boundary facets (appearing in exactly 1 cell)
            - "shared": Keep only shared facets (appearing in 2+ cells)
            - "interior": Keep only interior facets (appearing in exactly 2 cells)
            - list[int]: Keep facets with counts matching any value in the list

        Returns
        -------
        Mesh
            New Mesh with n_manifold_dims = self.n_manifold_dims - manifold_codimension,
            embedded in the same spatial dimension. The mesh shares the same points array
            but has new cells connectivity and aggregated cell_data.

        Raises
        ------
        ValueError
            If manifold_codimension is too large for this mesh
            (would result in negative manifold dimension).

        Examples
        --------
        >>> from physicsnemo.mesh.primitives.basic import two_triangles_2d
        >>> # Extract edges from a triangle mesh (codimension 1)
        >>> triangle_mesh = two_triangles_2d.load()
        >>> edge_mesh = triangle_mesh.get_facet_mesh(manifold_codimension=1)
        >>> assert edge_mesh.n_manifold_dims == 1  # edges
        >>>
        >>> # Extract vertices from a triangle mesh (codimension 2)
        >>> vertex_mesh = triangle_mesh.get_facet_mesh(manifold_codimension=2)
        >>> assert vertex_mesh.n_manifold_dims == 0  # vertices
        >>> facet_mesh = triangle_mesh.get_facet_mesh(
        ...     data_source="cells",
        ...     data_aggregation="area_weighted"
        ... )
        """
        ### Validate that extraction is possible
        new_manifold_dims = self.n_manifold_dims - manifold_codimension
        if new_manifold_dims < 0:
            raise ValueError(
                f"Cannot extract facet mesh with {manifold_codimension=} from mesh with {self.n_manifold_dims=}.\n"
                f"Would result in negative manifold dimension ({new_manifold_dims=}).\n"
                f"Maximum allowed codimension is {self.n_manifold_dims}."
            )

        ### Call kernel to extract facet mesh data
        from physicsnemo.mesh.boundaries import extract_facet_mesh_data

        facet_cells, facet_cell_data = extract_facet_mesh_data(
            parent_mesh=self,
            manifold_codimension=manifold_codimension,
            data_source=data_source,
            data_aggregation=data_aggregation,
            target_counts=target_counts,
        )

        ### Create and return new Mesh
        return Mesh(
            points=self.points,  # Share the same points
            cells=facet_cells,  # New connectivity for sub-simplices
            point_data=self.point_data.clone(),
            cell_data=facet_cell_data,  # Aggregated cell data
            global_data=self.global_data,  # Share global data
        )

    def get_boundary_mesh(
        self,
        data_source: Literal["points", "cells"] = "cells",
        data_aggregation: Literal["mean", "area_weighted", "inverse_distance"] = "mean",
    ) -> "Mesh":
        """Extract the boundary surface of this mesh.

        Convenience wrapper around :meth:`get_facet_mesh` that extracts only
        boundary facets (those appearing in exactly one parent cell).

        See :meth:`get_facet_mesh` for full parameter documentation.

        Parameters
        ----------
        data_source : {"points", "cells"}, optional
            Source of data inheritance. Default: "cells".
        data_aggregation : {"mean", "area_weighted", "inverse_distance"}, optional
            Strategy for aggregating data. Default: "mean".

        Returns
        -------
        Mesh
            Boundary mesh containing only boundary facets.

        Notes
        -----
        For meshes with internal cavities (like volume meshes with voids or
        drivaerML-style automotive meshes), this returns BOTH the exterior
        surface and any interior cavity surfaces. All facets that appear in
        exactly one parent cell are included, regardless of whether they face
        "outward" or "inward".

        Examples
        --------
        >>> from physicsnemo.mesh.primitives.procedural import lumpy_ball
        >>> from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral
        >>> # Extract triangular surface of a volume mesh
        >>> vol_mesh = lumpy_ball.load(n_shells=2, subdivisions=1)
        >>> surface_mesh = vol_mesh.get_boundary_mesh()
        >>> assert surface_mesh.n_manifold_dims == 2  # triangles
        >>>
        >>> # For a closed watertight sphere
        >>> sphere = sphere_icosahedral.load(subdivisions=3)
        >>> boundary = sphere.get_boundary_mesh()
        >>> assert boundary.n_cells == 0  # no boundary
        """
        return self.get_facet_mesh(
            manifold_codimension=1,
            data_source=data_source,
            data_aggregation=data_aggregation,
            target_counts="boundary",
        )

    def to_edge_graph(self) -> "Mesh[1, ...]":
        r"""Return a 1D Mesh whose cells are the unique edges of this mesh.

        Each edge (pair of vertices connected in a cell) appears exactly once.
        The resulting Mesh has the same ``points`` array, with ``cells`` of
        shape :math:`(E, 2)` where *E* is the number of unique edges.

        Cell data from the parent mesh is aggregated onto edges via the
        facet extraction pipeline (mean aggregation by default).

        Returns
        -------
        Mesh[1, ...]
            A 1D mesh (``n_manifold_dims == 1``) with edge cells.

        Examples
        --------
        >>> import torch
        >>> from physicsnemo.mesh import Mesh
        >>> points = torch.tensor([[0., 0.], [1., 0.], [0.5, 1.]])
        >>> cells = torch.tensor([[0, 1, 2]])
        >>> mesh = Mesh(points=points, cells=cells)
        >>> edge_graph = mesh.to_edge_graph()
        >>> assert isinstance(edge_graph, Mesh[1, ...])
        >>> assert edge_graph.n_cells == 3  # triangle has 3 edges
        """
        codim = self.n_manifold_dims - 1
        return self.get_facet_mesh(manifold_codimension=codim, target_counts="all")

    def to_dual_graph(self) -> "Mesh[1, ...]":
        r"""Return a 1D Mesh representing the cell-adjacency (dual) graph.

        Points are the cell centroids of this mesh.  Cells are
        :math:`(E, 2)` line segments connecting pairs of cells that share a
        codimension-1 facet (e.g., cells sharing an edge in 2D or a face in
        3D).  The parent mesh's ``cell_data`` becomes the ``point_data`` of the
        returned Mesh, since each dual-graph node corresponds to a parent cell.

        Returns
        -------
        Mesh[1, ...]
            A 1D mesh (``n_manifold_dims == 1``) whose points are cell
            centroids and whose cells encode the cell-neighbor adjacency.

        Examples
        --------
        >>> import torch
        >>> from physicsnemo.mesh import Mesh
        >>> # Two triangles sharing an edge
        >>> points = torch.tensor([[0., 0.], [1., 0.], [0.5, 1.], [1.5, 1.]])
        >>> cells = torch.tensor([[0, 1, 2], [1, 3, 2]])
        >>> mesh = Mesh(points=points, cells=cells)
        >>> dual = mesh.to_dual_graph()
        >>> assert isinstance(dual, Mesh[1, ...])
        >>> assert dual.n_cells == 1  # 1 shared edge -> 1 dual edge
        """
        adj = self.get_cell_to_cells_adjacency(adjacency_codimension=1)
        sources, targets = adj.expand_to_pairs()

        # Keep only upper-triangular pairs (source < target) to avoid
        # counting each neighbor relationship twice.
        mask = sources < targets
        edges = torch.stack([sources[mask], targets[mask]], dim=1)

        return Mesh(
            points=self.cell_centroids,
            cells=edges,
            point_data=self.cell_data,
            global_data=self.global_data,
        )

    def to_point_cloud(
        self, point_source: "Literal['vertices', 'cell_centroids']" = "vertices"
    ) -> "Mesh[0, ...]":
        r"""Return a 0D Mesh (point cloud) with no cell connectivity.

        Parameters
        ----------
        point_source : {"vertices", "cell_centroids"}
            What becomes the points of the returned Mesh:

            - ``"vertices"`` (default): Uses mesh vertices as points,
              preserving ``point_data``.
            - ``"cell_centroids"``: Uses cell centroids as points,
              mapping ``cell_data`` to ``point_data``.

        Returns
        -------
        Mesh[0, ...]
            A 0D mesh (``n_manifold_dims == 0``) with no cells.

        Examples
        --------
        >>> import torch
        >>> from physicsnemo.mesh import Mesh
        >>> points = torch.tensor([[0., 0.], [1., 0.], [0.5, 1.]])
        >>> cells = torch.tensor([[0, 1, 2]])
        >>> mesh = Mesh(points=points, cells=cells)
        >>> pc = mesh.to_point_cloud()
        >>> assert isinstance(pc, Mesh[0, ...])
        >>> assert pc.n_points == 3
        """
        if point_source == "vertices":
            return Mesh(
                points=self.points,
                point_data=self.point_data,
                global_data=self.global_data,
            )
        elif point_source == "cell_centroids":
            return Mesh(
                points=self.cell_centroids,
                point_data=self.cell_data,
                global_data=self.global_data,
            )
        else:
            raise ValueError(
                f"Invalid {point_source=!r}. Must be 'vertices' or 'cell_centroids'."
            )

    def is_watertight(self) -> bool:
        """Check if mesh is watertight (has no boundary).

        A mesh is watertight if every codimension-1 facet is shared by exactly 2 cells.
        This means the mesh forms a closed surface/volume with no holes or gaps.

        Returns
        -------
        bool
            True if mesh is watertight (no boundary facets), False otherwise.

        Examples
        --------
        >>> from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral, cylinder_open
        >>> # Closed sphere is watertight
        >>> sphere = sphere_icosahedral.load(subdivisions=3)
        >>> assert sphere.is_watertight() == True
        >>>
        >>> # Open cylinder with holes at ends
        >>> cylinder = cylinder_open.load()
        >>> assert cylinder.is_watertight() == False
        """
        from physicsnemo.mesh.boundaries import is_watertight

        return is_watertight(self)

    def is_manifold(
        self,
        check_level: Literal["facets", "edges", "full"] = "full",
    ) -> bool:
        """Check if mesh is a valid topological manifold.

        A mesh is a manifold if it locally looks like Euclidean space at every point.
        This function checks various topological constraints depending on the check level.

        Parameters
        ----------
        check_level : {"facets", "edges", "full"}, optional
            Level of checking to perform:
            - "facets": Only check codimension-1 facets (each appears 1-2 times)
            - "edges": Check facets + edge neighborhoods (for 2D/3D meshes)
            - "full": Complete manifold validation (default)

        Returns
        -------
        bool
            True if mesh passes the specified manifold checks, False otherwise.

        Notes
        -----
        This function checks topological constraints but does not check for
        geometric self-intersections (which would require expensive spatial queries).

        Examples
        --------
        >>> from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral, cylinder_open
        >>> # Valid manifold (sphere)
        >>> sphere = sphere_icosahedral.load(subdivisions=3)
        >>> assert sphere.is_manifold() == True
        >>>
        >>> # Manifold with boundary (open cylinder)
        >>> cylinder = cylinder_open.load()
        >>> assert cylinder.is_manifold() == True  # manifold with boundary is OK
        """
        from physicsnemo.mesh.boundaries import is_manifold

        return is_manifold(self, check_level=check_level)

    def _cached_adjacency(self, cache_key: str, compute_fn, **kwargs):
        r"""Look up or compute-and-cache a topological adjacency.

        All four ``get_*_adjacency`` methods delegate here. The
        ``offsets`` and ``indices`` tensors are stored as a sub-TensorDict
        under ``_cache["topology", "{cache_key}"]``.

        Parameters
        ----------
        cache_key : str
            Key under ``"topology"``, e.g. ``"point_to_points"`` or
            ``"cell_to_cells_codim_1"``.
        compute_fn : callable
            ``(mesh, **kwargs) -> Adjacency`` invoked on cache miss.
        **kwargs
            Forwarded to *compute_fn*.

        Returns
        -------
        Adjacency
            Cached or freshly computed adjacency.
        """
        from physicsnemo.mesh.neighbors import Adjacency

        cached = self._cache.get(("topology", cache_key), None)
        if cached is not None:
            return Adjacency(
                offsets=cached["offsets"],
                indices=cached["indices"],
            )
        result = compute_fn(self, **kwargs)
        self._cache["topology", cache_key] = TensorDict(
            {"offsets": result.offsets, "indices": result.indices},
        )
        return result

    def get_point_to_cells_adjacency(self):
        """Compute the star of each vertex (all cells containing each point).

        For each point in the mesh, finds all cells that contain that point. This
        is the graph-theoretic "star" operation on vertices.

        The result is cached in ``_cache["topology", ...]`` for efficiency.
        Adjacency depends only on topology (cells), not geometry (points), so
        the cache is preserved through geometric transforms.

        Returns
        -------
        Adjacency
            Adjacency where adjacency.to_list()[i] contains all cell indices that
            contain point i. Isolated points (not in any cells) have empty lists.

        Examples
        --------
        >>> from physicsnemo.mesh.primitives.basic import two_triangles_2d
        >>> mesh = two_triangles_2d.load()
        >>> adj = mesh.get_point_to_cells_adjacency()
        >>> # Get cells containing point 0
        >>> cells_of_point_0 = adj.to_list()[0]
        """
        from physicsnemo.mesh.neighbors import get_point_to_cells_adjacency

        return self._cached_adjacency("point_to_cells", get_point_to_cells_adjacency)

    def get_point_to_points_adjacency(self):
        """Compute point-to-point adjacency (graph edges of the mesh).

        For each point, finds all other points that share a cell with it. In simplicial
        meshes, this is equivalent to finding all points connected by an edge.

        The result is cached in ``_cache["topology", ...]`` for efficiency.
        Adjacency depends only on topology (cells), not geometry (points), so
        the cache is preserved through geometric transforms.

        Returns
        -------
        Adjacency
            Adjacency where adjacency.to_list()[i] contains all point indices that
            share a cell (edge) with point i. Isolated points have empty lists.

        Examples
        --------
        >>> from physicsnemo.mesh.primitives.basic import two_triangles_2d
        >>> mesh = two_triangles_2d.load()
        >>> adj = mesh.get_point_to_points_adjacency()
        >>> # Get neighbors of point 0
        >>> neighbors_of_point_0 = adj.to_list()[0]
        """
        from physicsnemo.mesh.neighbors import get_point_to_points_adjacency

        return self._cached_adjacency("point_to_points", get_point_to_points_adjacency)

    def get_cell_to_cells_adjacency(self, adjacency_codimension: int = 1):
        """Compute cell-to-cells adjacency based on shared facets.

        Two cells are considered adjacent if they share a k-codimension facet.

        The result is cached in ``_cache["topology", ...]`` for efficiency,
        keyed by ``adjacency_codimension``. Adjacency depends only on topology
        (cells), not geometry (points), so the cache is preserved through
        geometric transforms.

        Parameters
        ----------
        adjacency_codimension : int, optional
            Codimension of shared facets defining adjacency.

            - 1 (default): Cells must share a codimension-1 facet (e.g., triangles
              sharing an edge, tetrahedra sharing a triangular face)
            - 2: Cells must share a codimension-2 facet (e.g., tetrahedra sharing
              an edge)
            - k: Cells must share a codimension-k facet

        Returns
        -------
        Adjacency
            Adjacency where adjacency.to_list()[i] contains all cell indices that
            share a k-codimension facet with cell i.

        Examples
        --------
        >>> from physicsnemo.mesh.primitives.basic import two_triangles_2d
        >>> mesh = two_triangles_2d.load()
        >>> adj = mesh.get_cell_to_cells_adjacency(adjacency_codimension=1)
        >>> # Get cells sharing an edge with cell 0
        >>> neighbors_of_cell_0 = adj.to_list()[0]
        """
        from physicsnemo.mesh.neighbors import get_cell_to_cells_adjacency

        return self._cached_adjacency(
            f"cell_to_cells_codim_{adjacency_codimension}",
            get_cell_to_cells_adjacency,
            adjacency_codimension=adjacency_codimension,
        )

    def get_cell_to_points_adjacency(self):
        """Get the vertices (points) that comprise each cell.

        This is a simple wrapper around the cells array that returns it in the
        standard Adjacency format for consistency with other neighbor queries.

        The result is cached in ``_cache["topology", ...]`` for efficiency.

        Returns
        -------
        Adjacency
            Adjacency where adjacency.to_list()[i] contains all point indices that
            are vertices of cell i. For simplicial meshes, all cells have the same
            number of vertices (n_manifold_dims + 1).

        Examples
        --------
        >>> from physicsnemo.mesh.primitives.basic import two_triangles_2d
        >>> mesh = two_triangles_2d.load()
        >>> adj = mesh.get_cell_to_points_adjacency()
        >>> # Get vertices of cell 0
        >>> vertices_of_cell_0 = adj.to_list()[0]
        """
        from physicsnemo.mesh.neighbors import get_cell_to_points_adjacency

        return self._cached_adjacency("cell_to_points", get_cell_to_points_adjacency)

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
            point_data=self.point_data.apply(
                lambda x: _pad_with_value(x, target_n_points, data_padding_value),
                batch_size=torch.Size([target_n_points]),
            ),
            cell_data=self.cell_data.apply(
                lambda x: _pad_with_value(x, target_n_cells, data_padding_value),
                batch_size=torch.Size([target_n_cells]),
            ),
            global_data=self.global_data,
            _cache=TensorDict(
                {
                    "cell": self._cache["cell"].apply(
                        lambda x: _pad_with_value(x, target_n_cells, 0.0),
                        batch_size=torch.Size([target_n_cells]),
                    ),
                    "point": self._cache["point"].apply(
                        lambda x: _pad_with_value(x, target_n_points, 0.0),
                        batch_size=torch.Size([target_n_points]),
                    ),
                    "topology": TensorDict({}),
                },
                device=self.points.device,
            ),
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
            n = math.ceil(math.log(safe_size) / math.log(base))
            return int(base**n)

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
        backend_options: dict[str, Any] | None = None,
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
        ax : matplotlib.axes.Axes or pyvista.Plotter, optional
            Existing canvas to draw on. For matplotlib, a matplotlib Axes;
            for PyVista, a pyvista Plotter. If ``None``, a new figure/plotter
            is created. Use this to overlay multiple meshes on the same scene.
        backend_options : dict[str, Any], optional
            Additional keyword arguments forwarded to the underlying
            visualization backend (e.g. PyVista's ``plotter.add_mesh()``).

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
            backend_options=backend_options,
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
        axis: torch.Tensor | list | tuple | Literal["x", "y", "z"] | None = None,
        center: torch.Tensor | list | tuple | None = None,
        transform_point_data: bool | TensorDict = False,
        transform_cell_data: bool | TensorDict = False,
        transform_global_data: bool | TensorDict = False,
    ) -> "Mesh":
        """Rotate the mesh about an axis by a specified angle.

        Convenience wrapper for physicsnemo.mesh.transformations.rotate().

        Parameters
        ----------
        angle : float
            Rotation angle in radians.
        axis : torch.Tensor or list or tuple or {"x", "y", "z"}, optional
            Rotation axis vector. None for 2D, shape (3,) for 3D.
            String literals "x", "y", "z" are converted to unit vectors
            (1,0,0), (0,1,0), (0,0,1) respectively.
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
        transform_point_data: bool | TensorDict = False,
        transform_cell_data: bool | TensorDict = False,
        transform_global_data: bool | TensorDict = False,
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
        transform_point_data: bool | TensorDict = False,
        transform_cell_data: bool | TensorDict = False,
        transform_global_data: bool | TensorDict = False,
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

    def compute_point_derivatives(
        self,
        keys: str | tuple[str, ...] | list[str | tuple[str, ...]] | None = None,
        method: Literal["lsq", "dec"] = "lsq",
        gradient_type: Literal["intrinsic", "extrinsic", "both"] = "intrinsic",
    ) -> "Mesh":
        """Compute gradients of point_data fields.

        This is a convenience method that delegates to physicsnemo.mesh.calculus.compute_point_derivatives.

        Parameters
        ----------
        keys : str or tuple[str, ...] or list[str | tuple[str, ...]] or None, optional
            Fields to compute gradients of. Options:
            - None: All non-cached fields (excludes "_cache" subdictionary)
            - str: Single field name (e.g., "pressure")
            - tuple: Nested path (e.g., ("flow", "temperature"))
            - list: Multiple fields (e.g., ["pressure", "velocity"])
        method : {"lsq", "dec"}, optional
            Discretization method:
            - "lsq": Weighted least-squares reconstruction (default, CFD standard)
            - "dec": Discrete Exterior Calculus (differential geometry)
        gradient_type : {"intrinsic", "extrinsic", "both"}, optional
            Type of gradient:
            - "intrinsic": Project onto manifold tangent space (default)
            - "extrinsic": Full ambient space gradient
            - "both": Compute and store both

        Returns
        -------
        Mesh
            Self (mesh) with gradient fields added to point_data (modified in place).
            Field naming: "{field}_gradient" or "{field}_gradient_intrinsic/extrinsic"

        Examples
        --------
        >>> import torch
        >>> from physicsnemo.mesh.primitives.basic import two_triangles_2d
        >>> mesh = two_triangles_2d.load()
        >>> mesh.point_data["pressure"] = torch.randn(mesh.n_points)
        >>> # Compute gradient of pressure
        >>> mesh_grad = mesh.compute_point_derivatives(keys="pressure")
        >>> grad_p = mesh_grad.point_data["pressure_gradient"]
        """
        from physicsnemo.mesh.calculus import compute_point_derivatives

        return compute_point_derivatives(
            mesh=self,
            keys=keys,
            method=method,
            gradient_type=gradient_type,
        )

    def compute_cell_derivatives(
        self,
        keys: str | tuple[str, ...] | list[str | tuple[str, ...]] | None = None,
        method: Literal["lsq", "dec"] = "lsq",
        gradient_type: Literal["intrinsic", "extrinsic", "both"] = "intrinsic",
    ) -> "Mesh":
        """Compute gradients of cell_data fields.

        This is a convenience method that delegates to
        :func:`physicsnemo.mesh.calculus.compute_cell_derivatives`.

        Parameters
        ----------
        keys : str or tuple[str, ...] or list[str | tuple[str, ...]] or None, optional
            Fields to compute gradients of (same format as compute_point_derivatives).
        method : {"lsq"}, optional
            Discretization method for cell-centered data. Currently only
            ``"lsq"`` (weighted least-squares) is implemented. DEC
            gradients for cell-centered data are not available because the
            standard DEC exterior derivative maps vertex 0-forms to edge
            1-forms; there is no analogous cell-to-cell operator in the
            primal DEC complex.
        gradient_type : {"intrinsic", "extrinsic", "both"}, optional
            Type of gradient to compute.

        Returns
        -------
        Mesh
            A new Mesh with gradient fields added to ``cell_data``.

        Raises
        ------
        NotImplementedError
            If ``method="dec"`` is requested.

        Examples
        --------
        >>> import torch
        >>> from physicsnemo.mesh.primitives.basic import two_triangles_2d
        >>> mesh = two_triangles_2d.load()
        >>> mesh.cell_data["pressure"] = torch.randn(mesh.n_cells)
        >>> # Compute gradient of cell-centered pressure
        >>> mesh_grad = mesh.compute_cell_derivatives(keys="pressure")
        """
        from physicsnemo.mesh.calculus import compute_cell_derivatives

        return compute_cell_derivatives(
            mesh=self,
            keys=keys,
            method=method,
            gradient_type=gradient_type,
        )

    def integrate(
        self,
        field: str | tuple[str, ...] | torch.Tensor,
        data_source: Literal["cells", "points"] = "cells",
    ) -> torch.Tensor:
        r"""Integrate a field over the mesh domain.

        Computes :math:`\int_\Omega f\,d\Omega` using the appropriate
        quadrature rule for the field's discretization.  Cell data is
        treated as piecewise-constant (P0); point data is treated as
        piecewise-linear (P1) via the vertex-averaging rule (exact for
        linear fields, second-order accurate for smooth fields).

        The manifold dimension determines the measure automatically:
        arc length for ``Mesh[1, ...]``, surface area for ``Mesh[2, ...]``,
        volume for ``Mesh[3, ...]``, etc.

        Parameters
        ----------
        field : str, tuple[str, ...], or torch.Tensor
            Field to integrate:

            - ``str`` or ``tuple``: looked up in ``cell_data`` or
              ``point_data`` according to *data_source*.
            - ``torch.Tensor``: used directly.
        data_source : {"cells", "points"}
            Whether *field* is cell-centered (P0) or vertex-centered (P1).

        Returns
        -------
        torch.Tensor
            Integral value.  Shape matches ``field.shape[1:]`` (trailing
            dimensions are preserved: scalar -> 0-d, vector -> 1-d, etc.).

        Raises
        ------
        KeyError
            If *field* is a string key not present in the specified
            data source.
        ValueError
            If the mesh has no cells, or if a raw tensor has the wrong
            leading dimension for the specified *data_source*.

        Examples
        --------
        >>> import torch
        >>> from physicsnemo.mesh import Mesh
        >>> pts = torch.tensor([[0., 0.], [1., 0.], [0.5, 1.]])
        >>> cells = torch.tensor([[0, 1, 2]])
        >>> mesh = Mesh(points=pts, cells=cells)
        >>> mesh.cell_data["p"] = torch.tensor([3.0])
        >>> mesh.integrate("p")
        tensor(1.5000)
        """
        from physicsnemo.mesh.calculus.integration import integrate

        return integrate(
            mesh=self,
            field=field,
            data_source=data_source,
        )

    def integrate_flux(
        self,
        field: str | tuple[str, ...] | torch.Tensor,
        data_source: Literal["cells", "points"] = "cells",
    ) -> torch.Tensor:
        r"""Compute the surface flux integral for codimension-1 meshes.

        Computes :math:`\int_\Gamma \mathbf{F} \cdot \mathbf{n}\,d\Gamma`,
        the oriented flux of a vector field through the mesh surface.  Only
        defined for codimension-1 meshes where unique cell normals exist.

        Parameters
        ----------
        field : str, tuple[str, ...], or torch.Tensor
            Vector field with last dimension equal to ``n_spatial_dims``.
        data_source : {"cells", "points"}
            Whether *field* is cell-centered or vertex-centered.

        Returns
        -------
        torch.Tensor
            Scalar flux value (0-d tensor).

        Raises
        ------
        KeyError
            If *field* is a string key not present in the specified
            data source.
        ValueError
            If the mesh is not codimension-1, or if the field's last
            dimension does not match ``n_spatial_dims``.

        Examples
        --------
        >>> import torch
        >>> from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral
        >>> sphere = sphere_icosahedral.load(subdivisions=2)
        >>> # Constant field through a closed surface -> zero flux
        >>> v = torch.ones(sphere.n_cells, 3)
        >>> sphere.integrate_flux(v).abs() < 1e-5
        tensor(True)
        """
        from physicsnemo.mesh.calculus.integration import integrate_flux

        return integrate_flux(
            mesh=self,
            field=field,
            data_source=data_source,
        )

    def validate(
        self,
        check_degenerate_cells: bool = True,
        check_duplicate_vertices: bool = True,
        check_inverted_cells: bool = False,
        check_out_of_bounds: bool = True,
        check_manifoldness: bool = False,
        tolerance: float = 1e-10,
        raise_on_error: bool = False,
    ):
        """Validate mesh integrity and detect common errors.

        Convenience method that delegates to physicsnemo.mesh.validation.validate_mesh.

        Parameters
        ----------
        check_degenerate_cells : bool, optional
            Check for zero/negative area cells.
        check_duplicate_vertices : bool, optional
            Check for coincident vertices.
        check_inverted_cells : bool, optional
            Check for negative orientation.
        check_out_of_bounds : bool, optional
            Check cell indices are valid.
        check_manifoldness : bool, optional
            Check manifold topology (2D only).
        tolerance : float, optional
            Tolerance for geometric checks.
        raise_on_error : bool, optional
            Raise ValueError on first error vs return report.

        Returns
        -------
        dict
            Dictionary with validation results.

        Examples
        --------
        >>> from physicsnemo.mesh.primitives.basic import two_triangles_2d
        >>> mesh = two_triangles_2d.load()
        >>> report = mesh.validate()
        >>> assert report["valid"] == True
        """
        from physicsnemo.mesh.validation import validate_mesh

        return validate_mesh(
            mesh=self,
            check_degenerate_cells=check_degenerate_cells,
            check_duplicate_vertices=check_duplicate_vertices,
            check_inverted_cells=check_inverted_cells,
            check_out_of_bounds=check_out_of_bounds,
            check_manifoldness=check_manifoldness,
            tolerance=tolerance,
            raise_on_error=raise_on_error,
        )

    @property
    def quality_metrics(self):
        """Compute geometric quality metrics for all cells.

        Returns
        -------
        TensorDict
            Per-cell quality metrics:
            - aspect_ratio: max_edge / characteristic_length
            - edge_length_ratio: max_edge / min_edge
            - min_angle, max_angle: Interior angles (triangles only)
            - quality_score: Combined metric in [0,1] (1.0 is perfect)

        Examples
        --------
        >>> from physicsnemo.mesh.primitives.basic import two_triangles_2d
        >>> mesh = two_triangles_2d.load()
        >>> metrics = mesh.quality_metrics
        >>> assert "quality_score" in metrics.keys()
        """
        from physicsnemo.mesh.validation import compute_quality_metrics

        return compute_quality_metrics(self)

    @property
    def statistics(self):
        """Compute summary statistics for mesh.

        Returns
        -------
        dict
            Mesh statistics including counts, edge length distributions,
            area distributions, and quality metrics.

        Examples
        --------
        >>> from physicsnemo.mesh.primitives.basic import two_triangles_2d
        >>> mesh = two_triangles_2d.load()
        >>> stats = mesh.statistics
        >>> assert "n_points" in stats and "n_cells" in stats
        """
        from physicsnemo.mesh.validation import compute_mesh_statistics

        return compute_mesh_statistics(self)

    def subdivide(
        self,
        levels: int = 1,
        filter: Literal["linear", "butterfly", "loop"] = "linear",
    ) -> "Mesh":
        """Subdivide the mesh using iterative application of subdivision schemes.

        Subdivision refines the mesh by splitting each n-simplex into 2^n child
        simplices. Multiple subdivision schemes are supported, each with different
        geometric and smoothness properties.

        This method applies the chosen subdivision scheme iteratively for the
        specified number of levels. Each level independently subdivides the
        current mesh.

        Parameters
        ----------
        levels : int, optional
            Number of subdivision iterations to perform. Each level
            increases mesh resolution exponentially:

            - 0: No subdivision (returns original mesh)
            - 1: Each cell splits into 2^n children
            - 2: Each cell splits into 4^n children
            - k: Each cell splits into (2^k)^n children
        filter : {"linear", "butterfly", "loop"}, optional
            Subdivision scheme to use:

            - "linear": Simple midpoint subdivision (interpolating).
              New vertices at exact edge midpoints. Works for any dimension.
              Preserves original vertices.
            - "butterfly": Weighted stencil subdivision (interpolating).
              New vertices use weighted neighbor stencils for smoother results.
              Currently only supports 2D manifolds (triangular meshes).
              Preserves original vertices.
            - "loop": Valence-based subdivision (approximating).
              Both old and new vertices are repositioned for C² smoothness.
              Currently only supports 2D manifolds (triangular meshes).
              Original vertices move to new positions.

        Returns
        -------
        Mesh
            Subdivided mesh with refined geometry and connectivity.
            - Manifold and spatial dimensions are preserved
            - Point data is interpolated to new vertices
            - Cell data is propagated from parents to children
            - Global data is preserved unchanged

        Raises
        ------
        ValueError
            If levels < 0 or if filter is not one of the supported schemes.
        NotImplementedError
            If butterfly/loop filter used with non-2D manifold.

        Notes
        -----
        Multi-level subdivision is achieved by iterative application.
        For levels=3, this is equivalent to calling subdivide(levels=1)
        three times in sequence. This is the standard approach for all
        subdivision schemes.

        Examples
        --------
        >>> from physicsnemo.mesh.primitives.basic import two_triangles_2d
        >>> # Linear subdivision of triangular mesh
        >>> mesh = two_triangles_2d.load()
        >>> refined = mesh.subdivide(levels=2, filter="linear")
        >>> # Each triangle splits into 4, twice: 2 -> 8 -> 32 triangles
        >>> assert refined.n_cells == mesh.n_cells * 16
        """
        from physicsnemo.mesh.subdivision import (
            subdivide_butterfly,
            subdivide_linear,
            subdivide_loop,
        )

        ### Validate inputs
        if levels < 0:
            raise ValueError(f"levels must be >= 0, got {levels=}")

        ### Apply subdivision iteratively
        mesh = self
        for _ in range(levels):
            if filter == "linear":
                mesh = subdivide_linear(mesh)
            elif filter == "butterfly":
                mesh = subdivide_butterfly(mesh)
            elif filter == "loop":
                mesh = subdivide_loop(mesh)
            else:
                raise ValueError(
                    f"Invalid {filter=}. Must be one of: 'linear', 'butterfly', 'loop'"
                )

        return mesh

    def clean(
        self,
        tolerance: float = 1e-12,
        merge_points: bool = True,
        remove_duplicate_cells: bool = True,
        remove_unused_points: bool = True,
    ) -> "Mesh":
        r"""Clean and repair this mesh.

        Performs up to three cleaning operations in sequence:

        1. **Merge duplicate points** (``merge_points``): Finds points
           within ``tolerance`` L2 distance using BVH spatial queries and
           merges them into a single representative.  Point data values
           are averaged across merged groups.  Cost: :math:`O(N \log N)`
           where *N* is the number of points.  This is the most expensive
           step - on meshes with millions of points it can take tens of
           seconds.
        2. **Remove duplicate cells** (``remove_duplicate_cells``): Sorts
           vertex indices within each cell and removes cells that share
           the same vertex set.  Cost: :math:`O(C \log C)` where *C* is
           the number of cells.  Typically fast.
        3. **Remove unused points** (``remove_unused_points``): Drops
           points not referenced by any cell and compacts the point
           array.  Cost: :math:`O(N + C \cdot V)` where *V* is vertices
           per cell.  Very fast (linear scatter + mask).

        This is useful after importing meshes from external sources (VTK,
        STL, CAD) that may have redundant geometry.  For programmatic mesh
        operations like ``slice_cells`` that don't create duplicates, you
        can disable the expensive steps and only keep
        ``remove_unused_points=True`` for a large speedup.

        Parameters
        ----------
        tolerance : float, optional
            Absolute L2 distance threshold for merging duplicate points.
        merge_points : bool, optional
            Whether to merge spatially-duplicate points (default True).
        remove_duplicate_cells : bool, optional
            Whether to remove cells with identical vertex sets (default True).
        remove_unused_points : bool, optional
            Whether to drop points not referenced by any cell (default True).

        Returns
        -------
        Mesh
            Cleaned mesh with same structure but repaired topology.

        Examples
        --------
        >>> import torch
        >>> from physicsnemo.mesh import Mesh
        >>> # Mesh with duplicate points
        >>> points = torch.tensor([[0., 0.], [1., 0.], [0., 0.], [1., 1.]])
        >>> cells = torch.tensor([[0, 1, 3], [2, 1, 3]])
        >>> mesh = Mesh(points=points, cells=cells)
        >>> cleaned = mesh.clean()
        >>> assert cleaned.n_points == 3  # points 0 and 2 merged
        >>>
        >>> # Fast path: only remove unreferenced points (after slice_cells, etc.)
        >>> subset = mesh.slice_cells(torch.tensor([0]))
        >>> compacted = subset.clean(
        ...     merge_points=False,
        ...     remove_duplicate_cells=False,
        ...     remove_unused_points=True,
        ... )
        """
        from physicsnemo.mesh.repair import clean_mesh

        cleaned, _stats = clean_mesh(
            mesh=self,
            tolerance=tolerance,
            merge_points=merge_points,
            deduplicate_cells=remove_duplicate_cells,
            drop_unused_points=remove_unused_points,
        )
        return cleaned

    def strip_caches(self) -> "Mesh":
        r"""Return a new mesh with all cached values removed.

        Cached values (stored under the ``_cache`` key in data TensorDicts) are
        computed lazily for expensive operations like normals, areas, and curvature.
        This method creates a new mesh without these cached values, which is useful
        for:

        - Accurate benchmarking (prevents false performance benefits from caching)
        - Reducing memory usage
        - Forcing recomputation of cached values

        Returns
        -------
        Mesh
            A new mesh with the same geometry and data, but without cached values.

        Examples
        --------
        >>> from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral
        >>> mesh = sphere_icosahedral.load(subdivisions=2)
        >>> _ = mesh.cell_normals  # Triggers caching
        >>> mesh_clean = mesh.strip_caches()  # Remove cached normals
        """
        return Mesh(
            points=self.points,
            cells=self.cells,
            point_data=self.point_data,
            cell_data=self.cell_data,
            global_data=self.global_data,
        )


### Override the tensorclass __repr__ with custom formatting
# Note: Must be done after class definition because @tensorclass overrides __repr__
# even when defined inside the class body
def _mesh_repr(self) -> str:
    return format_mesh_repr(self)


Mesh.__repr__ = _mesh_repr  # type: ignore[method-assign]  # ty: ignore[invalid-assignment]
