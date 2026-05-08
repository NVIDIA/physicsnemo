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

from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Self

import torch
from jaxtyping import Float
from tensordict import TensorDict, tensorclass

from physicsnemo.mesh.mesh import Mesh
from physicsnemo.mesh.utilities.mesh_repr import format_mesh_repr

if TYPE_CHECKING:
    import matplotlib.axes
    import pyvista


@tensorclass
class DomainMesh:
    r"""A simulation domain represented as an interior mesh with named boundary meshes.

    A ``DomainMesh`` groups an interior :class:`Mesh` (either a volumetric mesh
    with full connectivity or a point cloud) together with zero or more boundary
    :class:`Mesh` objects keyed by boundary condition type (e.g. ``"no_slip"``,
    ``"inlet"``, ``"farfield"``), plus optional domain-level metadata in
    ``global_data``.

    The semantic contract is that the boundary meshes, if merged, form a
    watertight enclosure around the interior mesh. This is documented but not
    enforced at construction time; call :meth:`is_boundary_watertight` to
    verify explicitly.

    Because ``DomainMesh`` is a tensorclass, standard TensorDict operations
    like :meth:`to`, :meth:`clone`, and :meth:`pin_memory` propagate to
    ``interior``, all ``boundaries``, and ``global_data`` automatically.

    Parameters
    ----------
    interior : Mesh
        The interior region mesh. Can be a volumetric mesh with full simplicial
        connectivity (triangles, tetrahedra) or a bare point cloud.
    boundaries : dict[str, Mesh] or TensorDict[str, Mesh], optional
        Boundary condition meshes keyed by BC type name. If a ``dict`` is
        provided, it is automatically converted to a :class:`TensorDict`.
        Defaults to an empty collection.
    global_data : dict[str, torch.Tensor] or TensorDict, optional
        Domain-level quantities that apply to the entire simulation (e.g.
        Reynolds number, angle of attack, Mach number). If a ``dict`` is
        provided, it is automatically converted to a :class:`TensorDict`.
        Defaults to an empty collection.

    Raises
    ------
    TypeError
        If ``interior`` is not a :class:`Mesh`, or if any value in
        ``boundaries`` is not a :class:`Mesh`.
    ValueError
        If any boundary mesh has a different ``n_spatial_dims`` than
        ``interior``.

    Examples
    --------
    Create a domain with a volumetric interior and two boundary patches:

    >>> import torch
    >>> from physicsnemo.mesh import Mesh, DomainMesh
    >>> interior = Mesh(points=torch.randn(100, 3))
    >>> wall = Mesh(
    ...     points=torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]),
    ...     cells=torch.tensor([[0, 1, 2]]),
    ... )
    >>> inlet = Mesh(
    ...     points=torch.tensor([[2., 0., 0.], [3., 0., 0.], [2., 1., 0.]]),
    ...     cells=torch.tensor([[0, 1, 2]]),
    ... )
    >>> dm = DomainMesh(
    ...     interior=interior,
    ...     boundaries={"no_slip": wall, "inlet": inlet},
    ...     global_data={"Re": torch.tensor(1e6), "AoA": torch.tensor(5.0)},
    ... )
    >>> dm.n_boundaries
    2
    >>> dm.boundary_names
    ['inlet', 'no_slip']

    Create a domain with no boundaries (e.g. a standalone point cloud):

    >>> dm = DomainMesh(interior=Mesh(points=torch.randn(50, 3)))
    >>> dm.n_boundaries
    0

    Move everything to GPU:

    >>> dm_gpu = dm.to("cuda")  # doctest: +SKIP
    """

    interior: Mesh
    boundaries: TensorDict[str, Mesh]
    global_data: TensorDict

    def __init__(
        self,
        interior: Mesh,
        boundaries: dict[str, Mesh] | TensorDict | None = None,
        global_data: dict[str, torch.Tensor] | TensorDict | None = None,
    ) -> None:
        self.interior = interior
        self.boundaries = boundaries  # normalized by __post_init__
        self.global_data = global_data  # normalized by __post_init__
        # tensorclass only auto-calls __post_init__ from the *generated* __init__
        # (same semantics as dataclasses). Since we define a custom __init__,
        # we must call it explicitly. During load(), tensorclass calls it
        # automatically, so __post_init__ is the single source of truth for
        # defaults, coercions, and validation.
        self.__post_init__()

    def __post_init__(self) -> None:
        """Normalize fields and validate invariants.

        Called automatically during ``load()`` by tensorclass, and explicitly
        from ``__init__`` during normal construction. This is the single source
        of truth for all default values, type coercions, and shape validation.
        """
        ### boundaries: coerce dict -> TensorDict, None -> empty TensorDict
        if isinstance(self.boundaries, dict):
            self.boundaries = TensorDict(self.boundaries, batch_size=[])
        elif self.boundaries is None:
            self.boundaries = TensorDict({}, batch_size=[])
        else:
            self.boundaries.batch_size = torch.Size([])

        ### global_data: coerce dict -> TensorDict, None -> empty TensorDict
        if isinstance(self.global_data, TensorDict):
            self.global_data.batch_size = torch.Size([])
        else:
            self.global_data = TensorDict(
                {} if self.global_data is None else dict(self.global_data),
                batch_size=torch.Size([]),
            )

        ### Validate types and dimensional consistency
        if not torch.compiler.is_compiling():
            if not isinstance(self.interior, Mesh):
                raise TypeError(
                    f"`interior` must be a Mesh, got {type(self.interior).__name__}."
                )
            expected_spatial_dims = self.interior.n_spatial_dims
            for name in self.boundaries.keys():
                bc_mesh = self.boundaries[name]
                if not isinstance(bc_mesh, Mesh):
                    raise TypeError(
                        f"All boundary values must be Mesh instances, but "
                        f"boundaries[{name!r}] is {type(bc_mesh).__name__}."
                    )
                if bc_mesh.n_spatial_dims != expected_spatial_dims:
                    raise ValueError(
                        f"All meshes must share the same spatial dimension "
                        f"({expected_spatial_dims}), but boundaries[{name!r}] "
                        f"has n_spatial_dims={bc_mesh.n_spatial_dims}."
                    )

    def apply_to_meshes(
        self,
        fn: Callable[[Mesh], Mesh],
        *,
        interior: bool = True,
        boundaries: bool = True,
    ) -> "DomainMesh":
        r"""Apply a Mesh-to-Mesh function to meshes in the domain.

        By default, ``fn`` is called on the ``interior`` and on each boundary
        mesh. Use the keyword flags to apply selectively. Components that are
        skipped are cloned unchanged. Domain-level ``global_data`` is always
        cloned unchanged.

        All built-in operations (``translate``, ``rotate``, ``subdivide``,
        ``clean``, etc.) delegate here.

        This is distinct from the inherited tensorclass :meth:`apply`, which
        recursively maps a ``Tensor -> Tensor`` callable across every leaf
        tensor. Use :meth:`apply` for tensor-level transforms (e.g. dtype
        casting) and :meth:`apply_to_meshes` for mesh-level transforms.

        Parameters
        ----------
        fn : Callable[[Mesh], Mesh]
            A function that takes a :class:`Mesh` and returns a :class:`Mesh`.
        interior : bool
            If ``True`` (default), apply ``fn`` to the interior mesh.
        boundaries : bool
            If ``True`` (default), apply ``fn`` to every boundary mesh.

        Returns
        -------
        DomainMesh
            New domain with the transformed meshes.

        Examples
        --------
        Convert every mesh to a point cloud (drop connectivity):

        >>> dm_cloud = dm.apply_to_meshes(lambda m: Mesh(points=m.points))  # doctest: +SKIP

        Subdivide only the boundaries (e.g. to match a finer interior):

        >>> dm2 = dm.apply_to_meshes(  # doctest: +SKIP
        ...     lambda m: m.subdivide(levels=1), boundaries=True, interior=False
        ... )
        """
        return DomainMesh(
            interior=fn(self.interior) if interior else self.interior.clone(),
            boundaries=(
                self.boundaries.apply(fn, call_on_nested=True)
                if boundaries
                else self.boundaries.clone()
            ),
            global_data=self.global_data.clone(),
        )

    if TYPE_CHECKING:

        def to(self, *args: Any, **kwargs: Any) -> Self:
            """Move domain and all attached data to specified device/dtype.

            All tensors in ``interior``, every mesh in ``boundaries``, and
            ``global_data`` are moved together.

            Parameters
            ----------
            *args : Any
                Positional arguments passed to the underlying tensorclass
                ``to`` method.  Common usage: ``dm.to("cuda")`` or
                ``dm.to(torch.float32)``.
            **kwargs : Any
                Keyword arguments passed to the underlying tensorclass
                ``to`` method.

            Keyword Arguments
            -----------------
            device : torch.device, optional
                The desired device.
            dtype : torch.dtype, optional
                The desired floating-point or complex dtype.
            non_blocking : bool, optional
                Whether the transfer should be non-blocking.

            Returns
            -------
            DomainMesh
                A new DomainMesh on the target device/dtype, or the same
                instance if no changes were required.

            Examples
            --------
            >>> dm_gpu = dm.to("cuda")  # doctest: +SKIP
            >>> dm_cpu = dm.to(device="cpu")  # doctest: +SKIP
            """
            ...

        def clone(self) -> Self:
            """Return a deep clone of this DomainMesh.

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
            """Save the domain mesh to disk as memory-mapped tensors.

            Writes ``interior``, all ``boundaries``, and ``global_data``
            to a directory tree of ``.memmap`` files.  Proxy for the
            tensorclass ``memmap()`` method.

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
            DomainMesh
                A new DomainMesh backed by the on-disk memory-mapped
                storage.

            Examples
            --------
            >>> dm.save("/path/to/domain_mesh")  # doctest: +SKIP
            >>> reloaded = DomainMesh.load("/path/to/domain_mesh")  # doctest: +SKIP
            """
            ...

        @classmethod
        def load(
            cls,
            prefix: str | Path,
            device: torch.device | None = None,
            non_blocking: bool = False,
        ) -> Self:
            """Load a previously saved domain mesh from disk.

            Reads a directory tree of memory-mapped tensors written by
            :meth:`save` and reconstructs the ``DomainMesh`` instance,
            including the ``interior`` mesh, all ``boundaries``, and
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
            DomainMesh
                The reconstructed DomainMesh instance.

            Examples
            --------
            >>> dm = DomainMesh.load("/path/to/domain_mesh")  # doctest: +SKIP
            """
            ...

    ### Geometric Transforms

    def translate(
        self,
        offset: Float[torch.Tensor, " n_spatial_dims"] | Sequence[float],
    ) -> "DomainMesh":
        r"""Translate all meshes in the domain by a constant offset.

        Delegates to :meth:`Mesh.translate` for each mesh.

        Parameters
        ----------
        offset : torch.Tensor or Sequence[float]
            Translation vector, shape :math:`(S,)` where :math:`S` is
            ``n_spatial_dims``.

        Returns
        -------
        DomainMesh
            New domain with translated geometry.
        """
        return self.apply_to_meshes(lambda m: m.translate(offset=offset))

    def rotate(
        self,
        angle: float,
        axis: Float[torch.Tensor, " n_spatial_dims"]
        | Sequence[float]
        | Literal["x", "y", "z"]
        | None = None,
        center: Float[torch.Tensor, " n_spatial_dims"] | Sequence[float] | None = None,
        transform_point_data: bool | TensorDict = False,
        transform_cell_data: bool | TensorDict = False,
        transform_global_data: bool | TensorDict = False,
    ) -> "DomainMesh":
        r"""Rotate all meshes in the domain about an axis.

        Builds a rotation matrix and delegates to :meth:`transform`.
        Center handling uses translate-rotate-translate at the domain
        level, so domain-level :attr:`global_data` vectors are correctly
        rotated but not translated (vectors are translation-invariant).

        Parameters
        ----------
        angle : float
            Rotation angle in radians.
        axis : torch.Tensor or Sequence[float] or {"x", "y", "z"}, optional
            Rotation axis vector, shape :math:`(D_s,)`. Use ``None`` for 2D.
        center : torch.Tensor or Sequence[float], optional
            Center point for rotation, shape :math:`(D_s,)`.
        transform_point_data : bool or TensorDict
            Controls transformation of ``point_data`` fields. ``True``
            transforms all compatible fields; a ``TensorDict`` (or
            ``dict``) with scalar bool leaves selects specific fields.
        transform_cell_data : bool or TensorDict
            Same semantics, for ``cell_data``.
        transform_global_data : bool or TensorDict
            Same semantics, for each mesh's ``global_data`` and the
            domain-level :attr:`global_data`.

        Returns
        -------
        DomainMesh
            New domain with rotated geometry.
        """
        if center is not None:
            c = torch.as_tensor(
                center,
                device=self.interior.points.device,
                dtype=self.interior.points.dtype,
            )
            return (
                self.translate(-c)
                .rotate(
                    angle=angle,
                    axis=axis,
                    center=None,
                    transform_point_data=transform_point_data,
                    transform_cell_data=transform_cell_data,
                    transform_global_data=transform_global_data,
                )
                .translate(c)
            )

        from physicsnemo.mesh.transformations.geometric import rotation_matrix

        R = rotation_matrix(
            angle=angle,
            axis=axis,
            n_spatial_dims=self.interior.n_spatial_dims,
            device=self.interior.points.device,
            dtype=self.interior.points.dtype,
        )
        return self.transform(
            matrix=R,
            transform_point_data=transform_point_data,
            transform_cell_data=transform_cell_data,
            transform_global_data=transform_global_data,
            assume_invertible=True,
        )

    def scale(
        self,
        factor: float | Float[torch.Tensor, " n_spatial_dims"],
        center: Float[torch.Tensor, " n_spatial_dims"] | Sequence[float] | None = None,
        transform_point_data: bool | TensorDict = False,
        transform_cell_data: bool | TensorDict = False,
        transform_global_data: bool | TensorDict = False,
        assume_invertible: bool | None = None,
    ) -> "DomainMesh":
        r"""Scale all meshes in the domain by specified factor(s).

        Builds a scale matrix and delegates to :meth:`transform`.
        Center handling uses translate-scale-translate at the domain
        level.

        Parameters
        ----------
        factor : float or torch.Tensor
            Scale factor (scalar) or per-dimension factors, shape :math:`(D_s,)`.
        center : torch.Tensor or Sequence[float], optional
            Center point for scaling, shape :math:`(D_s,)`.
        transform_point_data : bool or TensorDict
            Controls transformation of ``point_data`` fields. ``True``
            transforms all compatible fields; a ``TensorDict`` (or
            ``dict``) with scalar bool leaves selects specific fields.
        transform_cell_data : bool or TensorDict
            Same semantics, for ``cell_data``.
        transform_global_data : bool or TensorDict
            Same semantics, for each mesh's ``global_data`` and the
            domain-level :attr:`global_data`.
        assume_invertible : bool or None, optional
            Controls cache propagation.  See :meth:`Mesh.scale`.

        Returns
        -------
        DomainMesh
            New domain with scaled geometry.
        """
        if center is not None:
            c = torch.as_tensor(
                center,
                device=self.interior.points.device,
                dtype=self.interior.points.dtype,
            )
            return (
                self.translate(-c)
                .scale(
                    factor=factor,
                    center=None,
                    transform_point_data=transform_point_data,
                    transform_cell_data=transform_cell_data,
                    transform_global_data=transform_global_data,
                    assume_invertible=assume_invertible,
                )
                .translate(c)
            )

        from physicsnemo.mesh.transformations.geometric import scale_matrix

        M = scale_matrix(
            factor=factor,
            n_spatial_dims=self.interior.n_spatial_dims,
            device=self.interior.points.device,
            dtype=self.interior.points.dtype,
        )
        return self.transform(
            matrix=M,
            transform_point_data=transform_point_data,
            transform_cell_data=transform_cell_data,
            transform_global_data=transform_global_data,
            assume_invertible=assume_invertible,
        )

    def transform(
        self,
        matrix: Float[torch.Tensor, "new_n_spatial_dims n_spatial_dims"],
        transform_point_data: bool | TensorDict = False,
        transform_cell_data: bool | TensorDict = False,
        transform_global_data: bool | TensorDict = False,
        assume_invertible: bool | None = None,
    ) -> "DomainMesh":
        r"""Apply a linear transformation to all meshes in the domain.

        This is the single point of contact for domain-level
        :attr:`global_data` transformation. Both :meth:`rotate` and
        :meth:`scale` delegate here after building their matrix.

        Parameters
        ----------
        matrix : torch.Tensor
            Transformation matrix, shape :math:`(S', S)`.
        transform_point_data : bool or TensorDict
            Controls transformation of ``point_data`` fields. ``True``
            transforms all compatible fields; a ``TensorDict`` (or
            ``dict``) with scalar bool leaves selects specific fields.
        transform_cell_data : bool or TensorDict
            Same semantics, for ``cell_data``.
        transform_global_data : bool or TensorDict
            Same semantics, for each mesh's ``global_data`` and the
            domain-level :attr:`global_data`.
        assume_invertible : bool or None, optional
            Controls cache propagation.  See :meth:`Mesh.transform`.

        Returns
        -------
        DomainMesh
            New domain with transformed geometry.
        """
        result = self.apply_to_meshes(
            lambda m: m.transform(
                matrix=matrix,
                transform_point_data=transform_point_data,
                transform_cell_data=transform_cell_data,
                transform_global_data=transform_global_data,
                assume_invertible=assume_invertible,
            )
        )
        if transform_global_data is not False:
            from physicsnemo.mesh.transformations.geometric import (
                _normalize_transform_mask,
                _transform_tensordict,
            )

            _transform_tensordict(
                result.global_data,
                matrix,
                self.interior.n_spatial_dims,
                "global_data",
                mask=_normalize_transform_mask(transform_global_data),
            )
        return result

    ### Cleanup / Refinement

    def clean(
        self,
        tolerance: float = 1e-12,
        merge_points: bool = True,
        remove_duplicate_cells: bool = True,
        remove_unused_points: bool = True,
    ) -> "DomainMesh":
        r"""Clean and repair all meshes in the domain.

        Delegates to :meth:`Mesh.clean` for each mesh independently.

        Parameters
        ----------
        tolerance : float, optional
            L2 distance threshold for merging duplicate points.
        merge_points : bool, optional
            Whether to merge spatially-duplicate points.
        remove_duplicate_cells : bool, optional
            Whether to remove cells with identical vertex sets.
        remove_unused_points : bool, optional
            Whether to drop points not referenced by any cell.

        Returns
        -------
        DomainMesh
            New domain with cleaned meshes.
        """
        return self.apply_to_meshes(
            lambda m: m.clean(
                tolerance=tolerance,
                merge_points=merge_points,
                remove_duplicate_cells=remove_duplicate_cells,
                remove_unused_points=remove_unused_points,
            )
        )

    def strip_caches(self) -> "DomainMesh":
        r"""Remove cached geometry from all meshes in the domain.

        Delegates to :meth:`Mesh.strip_caches` for each mesh.

        Returns
        -------
        DomainMesh
            New domain with all cached values cleared.
        """
        return self.apply_to_meshes(lambda m: m.strip_caches())

    def subdivide(
        self,
        levels: int = 1,
        filter: Literal["linear", "butterfly", "loop"] = "linear",
    ) -> "DomainMesh":
        r"""Subdivide all meshes in the domain.

        Delegates to :meth:`Mesh.subdivide` for each mesh.

        Parameters
        ----------
        levels : int, optional
            Number of subdivision iterations.
        filter : {"linear", "butterfly", "loop"}, optional
            Subdivision scheme.  See :meth:`Mesh.subdivide`.

        Returns
        -------
        DomainMesh
            New domain with subdivided meshes.
        """
        return self.apply_to_meshes(lambda m: m.subdivide(levels=levels, filter=filter))

    ### Data Operations

    def cell_data_to_point_data(self, overwrite_keys: bool = False) -> "DomainMesh":
        r"""Convert cell data to point data on all meshes in the domain.

        Delegates to :meth:`Mesh.cell_data_to_point_data` for each mesh.

        Parameters
        ----------
        overwrite_keys : bool
            If ``True``, silently overwrite existing ``point_data`` keys.

        Returns
        -------
        DomainMesh
            New domain with converted data on all meshes.
        """
        return self.apply_to_meshes(
            lambda m: m.cell_data_to_point_data(overwrite_keys=overwrite_keys)
        )

    def point_data_to_cell_data(self, overwrite_keys: bool = False) -> "DomainMesh":
        r"""Convert point data to cell data on all meshes in the domain.

        Delegates to :meth:`Mesh.point_data_to_cell_data` for each mesh.

        Parameters
        ----------
        overwrite_keys : bool
            If ``True``, silently overwrite existing ``cell_data`` keys.

        Returns
        -------
        DomainMesh
            New domain with converted data on all meshes.
        """
        return self.apply_to_meshes(
            lambda m: m.point_data_to_cell_data(overwrite_keys=overwrite_keys)
        )

    def compute_point_derivatives(
        self,
        keys: str | tuple[str, ...] | list[str | tuple[str, ...]] | None = None,
        method: Literal["lsq", "dec"] = "lsq",
        gradient_type: Literal["intrinsic", "extrinsic", "both"] = "intrinsic",
    ) -> "DomainMesh":
        r"""Compute gradients of point_data fields on all meshes.

        Delegates to :meth:`Mesh.compute_point_derivatives` for each mesh.

        Parameters
        ----------
        keys : str or tuple or list or None, optional
            Fields to differentiate.  ``None`` for all non-cached fields.
        method : {"lsq", "dec"}, optional
            Discretization method.
        gradient_type : {"intrinsic", "extrinsic", "both"}, optional
            Type of gradient to compute.

        Returns
        -------
        DomainMesh
            Domain with gradient fields added to each mesh's ``point_data``.
        """
        return self.apply_to_meshes(
            lambda m: m.compute_point_derivatives(
                keys=keys, method=method, gradient_type=gradient_type
            )
        )

    def compute_cell_derivatives(
        self,
        keys: str | tuple[str, ...] | list[str | tuple[str, ...]] | None = None,
        method: Literal["lsq", "dec"] = "lsq",
        gradient_type: Literal["intrinsic", "extrinsic", "both"] = "intrinsic",
    ) -> "DomainMesh":
        r"""Compute gradients of cell_data fields on all meshes.

        Delegates to :meth:`Mesh.compute_cell_derivatives` for each mesh.

        Parameters
        ----------
        keys : str or tuple or list or None, optional
            Fields to differentiate.  ``None`` for all non-cached fields.
        method : {"lsq", "dec"}, optional
            Discretization method.
        gradient_type : {"intrinsic", "extrinsic", "both"}, optional
            Type of gradient to compute.

        Returns
        -------
        DomainMesh
            Domain with gradient fields added to each mesh's ``cell_data``.
        """
        return self.apply_to_meshes(
            lambda m: m.compute_cell_derivatives(
                keys=keys, method=method, gradient_type=gradient_type
            )
        )

    ### Validation

    def validate(
        self,
        check_degenerate_cells: bool = True,
        check_duplicate_vertices: bool = True,
        check_inverted_cells: bool = False,
        check_out_of_bounds: bool = True,
        check_manifoldness: bool = False,
        tolerance: float = 1e-10,
        raise_on_error: bool = False,
    ) -> dict[str, Any]:
        r"""Validate all meshes in the domain and aggregate results.

        Delegates to :meth:`Mesh.validate` for the interior and each boundary
        mesh, then aggregates the results into a domain-level report.

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
            Check manifold topology.
        tolerance : float, optional
            Tolerance for geometric checks.
        raise_on_error : bool, optional
            Raise ``ValueError`` on first error vs return report.

        Returns
        -------
        dict[str, Any]
            Aggregated validation report with keys:

            - ``"interior"``: validation report for the interior mesh
              (``Mapping[str, bool | int | torch.Tensor]``, see
              :meth:`Mesh.validate`).
            - ``"boundaries"``: ``dict[str, Mapping[str, ...]]`` of per-boundary
              reports.
            - ``"valid"``: ``bool``, ``True`` only if all meshes pass validation.
        """
        kwargs: dict[str, Any] = dict(
            check_degenerate_cells=check_degenerate_cells,
            check_duplicate_vertices=check_duplicate_vertices,
            check_inverted_cells=check_inverted_cells,
            check_out_of_bounds=check_out_of_bounds,
            check_manifoldness=check_manifoldness,
            tolerance=tolerance,
            raise_on_error=raise_on_error,
        )
        interior_report = self.interior.validate(**kwargs)
        boundary_reports = {
            name: self.boundaries[name].validate(**kwargs)
            for name in self.boundary_names
        }
        return {
            "interior": interior_report,
            "boundaries": boundary_reports,
            "valid": interior_report["valid"]
            and all(r["valid"] for r in boundary_reports.values()),
        }

    ### Properties

    @property
    def boundary_names(self) -> list[str]:
        """Sorted list of boundary condition names.

        Returns
        -------
        list[str]
            The keys of ``boundaries``, sorted alphabetically.
        """
        return sorted(self.boundaries.keys())

    @property
    def n_boundaries(self) -> int:
        """Number of boundary meshes.

        Returns
        -------
        int
            The number of entries in ``boundaries``.
        """
        return len(self.boundary_names)

    ### Methods

    def all_meshes(self) -> Iterator[tuple[str, Mesh]]:
        """Iterate over all meshes in the domain.

        Yields the interior mesh first (keyed ``"interior"``), then each
        boundary mesh in sorted key order.

        Yields
        ------
        tuple[str, Mesh]
            ``(name, mesh)`` pairs. The first pair is always
            ``("interior", self.interior)``.

        Examples
        --------
        >>> for name, mesh in dm.all_meshes():
        ...     print(f"{name}: {mesh.n_points} points")  # doctest: +SKIP
        interior: 100 points
        inlet: 3 points
        no_slip: 3 points
        """
        yield "interior", self.interior
        for name in self.boundary_names:
            yield name, self.boundaries[name]

    def __iter__(self) -> Iterator[tuple[str, Mesh]]:
        r"""Iterate over all meshes in the domain.

        Equivalent to :meth:`all_meshes`; yields the interior mesh first
        (keyed ``"interior"``), then each boundary mesh in sorted key order.

        Yields
        ------
        tuple[str, Mesh]
            ``(name, mesh)`` pairs.

        Examples
        --------
        >>> for name, mesh in dm:
        ...     print(f"{name}: {mesh.n_points} points")  # doctest: +SKIP
        """
        yield from self.all_meshes()

    def merge_boundaries(self, preserve_data: bool = False) -> Mesh:
        """Merge all boundary meshes into a single :class:`Mesh`.

        Produces a mesh containing the concatenated points and cells from
        every boundary. By default, ``point_data`` and ``cell_data`` are
        stripped before merging because boundaries typically carry
        heterogeneous fields (different keys per boundary), which
        :meth:`Mesh.merge` cannot concatenate.

        Parameters
        ----------
        preserve_data : bool
            If ``False`` (default), strip ``point_data`` and ``cell_data``
            from each boundary before merging - the safe choice for the
            typical CFD case where each boundary carries its own field set.
            If ``True``, delegate directly to :meth:`Mesh.merge`, which
            preserves data but requires that all boundaries share the same
            ``cell_data`` keys and have ``point_data`` that can be
            concatenated. Use this when every boundary has a consistent
            set of fields.

        Returns
        -------
        Mesh
            A single mesh containing the concatenated points and cells from
            every boundary. Data fields are included only if
            ``preserve_data`` is ``True``.

        Raises
        ------
        ValueError
            If there are no boundary meshes to merge, if boundary meshes
            have incompatible manifold dimensions, or (when
            ``preserve_data=True``) if their data keys are inconsistent.
        """
        if self.n_boundaries == 0:
            raise ValueError("No boundary meshes to merge.")
        boundaries = [self.boundaries[name] for name in self.boundary_names]
        if preserve_data:
            return Mesh.merge(boundaries)
        geometry_only = [Mesh(points=b.points, cells=b.cells) for b in boundaries]
        return Mesh.merge(geometry_only)

    def is_boundary_watertight(self, tolerance: float = 1e-6) -> bool:
        r"""Check whether the merged boundary meshes form a watertight surface.

        Merges all boundary meshes via :meth:`merge_boundaries`, deduplicates
        coincident vertices with :meth:`Mesh.clean`, and calls
        :meth:`Mesh.is_watertight` on the result. The clean step is necessary
        because independently-meshed boundary patches share physical vertices
        that become duplicated during merge - and float32 round-off from any
        prior transform may prevent an exact-match merge.

        Parameters
        ----------
        tolerance : float, optional
            L2 distance threshold for merging coincident boundary vertices
            before the topology check. The default ``1e-6`` is deliberately
            looser than :meth:`Mesh.clean`'s ``1e-12`` so it absorbs float32
            round-off (~1e-7 relative) on the duplicated vertices that
            ``merge_boundaries`` produces from independently-meshed patches.
            For coordinates that span much smaller or much larger than ~1,
            pass an explicit value (e.g. ``1e-6 * max_extent`` of the bbox).

        Returns
        -------
        bool
            ``True`` if the merged boundary surface is watertight (every
            codimension-1 facet is shared by exactly 2 cells), ``False``
            otherwise. Returns ``False`` if there are no boundary meshes.

        Notes
        -----
        This is not free to compute: the :meth:`Mesh.clean` step performs a
        BVH-based duplicate-point merge that scales as :math:`O(N \log N)`
        in the total boundary vertex count :math:`N`, and dominates the
        runtime. Callers that need the result repeatedly should cache it.
        """
        if self.n_boundaries == 0:
            return False
        return self.merge_boundaries().clean(tolerance=tolerance).is_watertight()

    def draw(
        self,
        *,
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
        show_edges: bool = False,
        boundary_kwargs: dict[str, Any] | None = None,
        ax: "matplotlib.axes.Axes | pyvista.Plotter | None" = None,
        backend_options: dict[str, Any] | None = None,
    ) -> "matplotlib.axes.Axes | pyvista.Plotter":
        r"""Draw the domain: interior with optional scalar coloring, boundaries overlaid.

        Renders the interior as the primary visual layer, then overlays every
        boundary on the same canvas. The interior parameter set mirrors
        :meth:`Mesh.draw` exactly, with two intentional changes: the call is
        keyword-only, and ``show_edges`` defaults to ``False`` (rather than
        ``True``) because dense interior meshes are typically more readable
        without edges. Both matplotlib and PyVista backends are supported.

        Parameters
        ----------
        backend, show, point_scalars, cell_scalars, cmap, vmin, vmax, alpha_points, alpha_cells, alpha_edges, show_edges, ax, backend_options
            Forwarded to :meth:`Mesh.draw` for the **interior** mesh. See
            :meth:`Mesh.draw` for full descriptions.
        boundary_kwargs : dict, optional
            Keyword arguments forwarded to :meth:`Mesh.draw` for **every**
            boundary mesh. Defaults are tuned for unobtrusive overlay:

            - ``alpha_points = 0`` (boundary vertices are not scattered).
            - ``alpha_cells = 0.3`` when boundaries are 2-D surfaces,
              ``1.0`` when they are 1-D curves. Auto-detected from the
              first boundary's :attr:`Mesh.n_manifold_dims`.
            - ``show_edges = False``.

            User-supplied keys override these defaults. To color individual
            boundaries by their own scalar fields, compose :meth:`Mesh.draw`
            calls directly (see Examples).

        Returns
        -------
        matplotlib.axes.Axes or pyvista.Plotter
            The canvas, for further customization when ``show=False``.

        Examples
        --------
        Default visualization with pressure coloring on the interior:

        >>> dm.draw(point_scalars="p", cmap="RdBu_r", vmin=-200, vmax=200)  # doctest: +SKIP

        Translucent boundaries with edges visible:

        >>> dm.draw(  # doctest: +SKIP
        ...     point_scalars="p",
        ...     boundary_kwargs={"alpha_cells": 0.5, "show_edges": True},
        ... )

        Customize and display later by setting axis limits on the returned canvas:

        >>> ax = dm.draw(point_scalars="p", show=False)  # doctest: +SKIP
        >>> ax.set_xlim(-2, 4); ax.set_ylim(-3, 3)       # doctest: +SKIP

        Per-boundary scalar coloring (manual composition - color the no-slip
        wall by its own ``shear`` field while the interior shows pressure):

        >>> ax = dm.interior.draw(point_scalars="p", show=False)  # doctest: +SKIP
        >>> dm.boundaries["wall"].draw(                           # doctest: +SKIP
        ...     ax=ax, cell_scalars="shear", cmap="hot", show=False,
        ... )
        >>> for name in dm.boundary_names:                        # doctest: +SKIP
        ...     if name == "wall":
        ...         continue
        ...     dm.boundaries[name].draw(
        ...         ax=ax, alpha_cells=0.3, alpha_points=0,
        ...         show_edges=False, show=False,
        ...     )
        >>> import matplotlib.pyplot as plt; plt.show()           # doctest: +SKIP
        """
        ### Auto-pick boundary opacity from the boundary's manifold dim:
        ### 2-D surfaces would otherwise occlude the interior; 1-D curves
        ### are thin lines and stay legible at full opacity.
        if self.n_boundaries > 0:
            first_bdy = self.boundaries[self.boundary_names[0]]
            auto_alpha_cells = 0.3 if first_bdy.n_manifold_dims >= 2 else 1.0
        else:
            auto_alpha_cells = 1.0  # unused

        boundary_defaults: dict[str, Any] = {
            "alpha_points": 0,
            "alpha_cells": auto_alpha_cells,
            "show_edges": False,
        }
        boundary_defaults.update(boundary_kwargs or {})

        ### Draw interior; if no boundaries follow, this is the layer that
        ### triggers the eventual ``.show()``.
        has_boundaries = self.n_boundaries > 0
        canvas = self.interior.draw(
            backend=backend,
            show=show and not has_boundaries,
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

        ### Overlay boundaries; the last one triggers ``.show()`` if requested.
        names = self.boundary_names
        last = names[-1] if names else None
        for name in names:
            self.boundaries[name].draw(
                ax=canvas,
                backend=backend,
                show=show and (name is last),
                **boundary_defaults,
            )
        return canvas

    ### Repr is defined after the class body (see below) because
    ### @tensorclass overwrites __repr__ even when defined inline.


### Override the tensorclass __repr__ with custom formatting.
# Must be done after class definition because @tensorclass overrides __repr__
# even when defined inside the class body (same pattern as Mesh).
def _domain_mesh_repr(self: DomainMesh) -> str:
    """Format a readable summary of the domain mesh."""
    lines = ["DomainMesh("]

    ### Interior - indent data fields one level under "interior:"
    interior_repr = format_mesh_repr(self.interior)
    first, *rest = interior_repr.split("\n")
    lines.append(f"    interior: {first}")
    lines.extend(f"    {line}" for line in rest)

    ### Boundaries - indent data fields one level under each boundary key
    bc_names = self.boundary_names
    if not bc_names:
        lines.append("    boundaries: {}")
    else:
        lines.append("    boundaries:")
        max_bc_len = max(len(n) for n in bc_names)
        for name in bc_names:
            bc_mesh = self.boundaries[name]
            bc_repr = format_mesh_repr(bc_mesh)
            first, *rest = bc_repr.split("\n")
            lines.append(f"        {name.ljust(max_bc_len)}: {first}")
            lines.extend(f"        {line}" for line in rest)

    ### Global data (only if non-empty)
    gd_keys = sorted(self.global_data.keys())
    if gd_keys:
        items = ", ".join(f"{k}: {tuple(self.global_data[k].shape)}" for k in gd_keys)
        lines.append(f"    global_data: {{{items}}}")

    lines.append(")")
    return "\n".join(lines)


DomainMesh.__repr__ = _domain_mesh_repr  # type: ignore[method-assign]  # ty: ignore[invalid-assignment]
