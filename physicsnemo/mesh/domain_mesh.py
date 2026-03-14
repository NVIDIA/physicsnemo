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

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, Self

import torch
from tensordict import TensorDict, tensorclass

from physicsnemo.mesh._mesh_spec import MeshDims, _get_mesh_spec
from physicsnemo.mesh.mesh import Mesh
from physicsnemo.mesh.utilities.mesh_repr import format_mesh_repr


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
    enforced at construction time; call :meth:`check_boundary_watertight` to
    verify explicitly.

    Because ``DomainMesh`` is a tensorclass, standard TensorDict operations
    like :meth:`to`, :meth:`clone`, and :meth:`pin_memory` propagate to
    ``interior``, all ``boundaries``, and ``global_data`` automatically.

    Supports parametric dimension syntax via ``DomainMesh[m, s]``, which
    constrains the interior to ``Mesh[m, s]`` and boundaries to
    ``Mesh[m-1, s]``. This enables dimension-aware type annotations and
    runtime ``isinstance`` checks (see :meth:`__class_getitem__`).

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
        ``interior``, or (when the interior has cells) a different
        ``n_manifold_dims`` than ``interior.n_manifold_dims - 1``.

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

    interior: Mesh[" m", " s"]
    boundaries: TensorDict[str, Mesh[" m-1", " s"]]
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

    def __post_init__(self):
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
            interior_manifold_dims = self.interior.n_manifold_dims
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
                if (
                    interior_manifold_dims > 0
                    and bc_mesh.n_manifold_dims != interior_manifold_dims - 1
                ):
                    raise ValueError(
                        f"Boundary meshes must have n_manifold_dims="
                        f"{interior_manifold_dims - 1} "
                        f"(interior.n_manifold_dims - 1), but "
                        f"boundaries[{name!r}] has "
                        f"n_manifold_dims={bc_mesh.n_manifold_dims}."
                    )

    @classmethod
    def __class_getitem__(cls, params: tuple) -> type:
        r"""Parametrize DomainMesh by interior manifold and spatial dimensions.

        Returns a synthetic type usable in type annotations and ``isinstance``
        checks. The spec ``DomainMesh[m, s]`` constrains the interior to
        ``Mesh[m, s]`` and all boundary meshes to ``Mesh[m-1, s]``.
        Always requires exactly two parameters; use ``...`` (Ellipsis) to
        leave a dimension unconstrained.

        Parameters
        ----------
        params : tuple
            A 2-tuple of ``(manifold_dims, spatial_dims)`` where each element
            is an ``int`` (concrete), ``str`` (symbolic, e.g. ``"n-1"``), or
            ``...`` (unconstrained).

        Returns
        -------
        type
            A parametrized DomainMesh type supporting ``isinstance`` checks,
            with ``.interior_type`` and ``.boundary_type`` navigation
            properties.

        Raises
        ------
        TypeError
            If not exactly 2 parameters, or if parameter types are invalid.
        ValueError
            If concrete dimensions are negative or manifold exceeds spatial.

        Examples
        --------
        >>> DomainMesh[3, 3]
        DomainMesh[3, 3]
        >>> DomainMesh[2, ...]
        DomainMesh[2, ...]
        >>> DomainMesh[3, 3].interior_type
        Mesh[3, 3]
        >>> DomainMesh[3, 3].boundary_type
        Mesh[2, 3]
        """
        if not isinstance(params, tuple):
            raise TypeError(
                f"DomainMesh[...] requires exactly 2 parameters "
                f"(e.g. DomainMesh[3, 3] or DomainMesh[2, ...]), "
                f"got single parameter {params!r}"
            )
        if len(params) != 2:
            raise TypeError(
                f"DomainMesh[...] requires exactly 2 parameters, "
                f"got {len(params)}"
            )

        n_manifold_dims = None if params[0] is ... else params[0]
        n_spatial_dims = None if params[1] is ... else params[1]

        return _get_domain_mesh_spec(
            MeshDims(n_manifold_dims=n_manifold_dims, n_spatial_dims=n_spatial_dims)
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
            """Return a shallow clone of this DomainMesh.

            All tensor storage is shared with the original; metadata and
            TensorDict structure are independent copies.
            """
            ...

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
        return len(list(self.boundaries.keys()))

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

    def merge_boundaries(self) -> Mesh:
        """Merge all boundary meshes into a single :class:`Mesh`.

        Delegates to :meth:`Mesh.merge`. All boundary meshes must have the
        same manifold dimension and compatible ``cell_data`` keys.

        Returns
        -------
        Mesh
            A single mesh containing the concatenated points, cells, and data
            from every boundary mesh.

        Raises
        ------
        ValueError
            If there are no boundary meshes to merge, or if boundary meshes
            have incompatible dimensions or ``cell_data`` keys.
        """
        boundary_meshes = [self.boundaries[name] for name in self.boundary_names]
        if not boundary_meshes:
            raise ValueError("No boundary meshes to merge.")
        return Mesh.merge(boundary_meshes)

    def check_boundary_watertight(self) -> bool:
        """Check whether the merged boundary meshes form a watertight surface.

        Merges all boundary meshes via :meth:`merge_boundaries` and calls
        :meth:`Mesh.is_watertight` on the result.

        Returns
        -------
        bool
            ``True`` if the merged boundary surface is watertight (every
            codimension-1 facet is shared by exactly 2 cells), ``False``
            otherwise. Returns ``False`` if there are no boundary meshes.
        """
        if self.n_boundaries == 0:
            return False
        return self.merge_boundaries().is_watertight()

    ### Repr

    def __repr__(self) -> str:
        """Format a readable summary of the domain mesh."""
        lines = ["DomainMesh("]

        ### Interior
        lines.append(f"    interior: {format_mesh_repr(self.interior)}")

        ### Boundaries
        bc_names = self.boundary_names
        if not bc_names:
            lines.append("    boundaries: {}")
        else:
            lines.append("    boundaries:")
            max_bc_len = max(len(n) for n in bc_names)
            for name in bc_names:
                bc_mesh = self.boundaries[name]
                bc_repr = format_mesh_repr(bc_mesh)
                # First line gets the key prefix; continuation lines are indented
                first, *rest = bc_repr.split("\n")
                key_prefix = f"        {name.ljust(max_bc_len)}: "
                lines.append(f"{key_prefix}{first}")
                cont_indent = " " * len(key_prefix)
                lines.extend(f"{cont_indent}{line}" for line in rest)

        ### Global data (only if non-empty)
        gd_keys = sorted(self.global_data.keys())
        if gd_keys:
            items = ", ".join(
                f"{k}: {tuple(self.global_data[k].shape)}" for k in gd_keys
            )
            lines.append(f"    global_data: {{{items}}}")

        lines.append(")")
        return "\n".join(lines)


### Metaclass for parametrized DomainMesh types


class _DomainMeshSpecMeta(type):
    r"""Metaclass enabling ``isinstance(dm, DomainMesh[3, 3])`` checks.

    Each instance of this metaclass is a synthetic type representing a
    dimension-constrained DomainMesh. The constraint applies to the interior
    mesh (must satisfy ``Mesh[m, s]``) and all boundary meshes (must satisfy
    ``Mesh[m-1, s]``). It is not a subclass of DomainMesh and cannot be
    instantiated - it exists purely for ``isinstance`` checks, ``repr``,
    and derived-type navigation.
    """

    _mesh_dims: MeshDims

    def __instancecheck__(cls, instance: object) -> bool:
        if not type.__instancecheck__(DomainMesh, instance):
            return False
        dm: DomainMesh = instance  # type: ignore[assignment]
        interior_spec = _get_mesh_spec(cls._mesh_dims)
        if not isinstance(dm.interior, interior_spec):
            return False
        try:
            boundary_spec = _get_mesh_spec(cls._mesh_dims.boundary)
        except (ValueError, TypeError):
            # m=0 or m=None: boundary dims can't be derived, skip check
            return True
        for name in dm.boundary_names:
            if not isinstance(dm.boundaries[name], boundary_spec):
                return False
        return True

    def __repr__(cls) -> str:
        return f"DomainMesh[{cls._mesh_dims}]"

    @property
    def interior_type(cls) -> type:
        """``DomainMesh[m, s].interior_type`` gives ``Mesh[m, s]``."""
        return _get_mesh_spec(cls._mesh_dims)

    @property
    def boundary_type(cls) -> type:
        """``DomainMesh[m, s].boundary_type`` gives ``Mesh[m-1, s]``."""
        return _get_mesh_spec(cls._mesh_dims.boundary)


### Cached factory

_domain_mesh_spec_cache: dict[MeshDims, type] = {}


def _get_domain_mesh_spec(dims: MeshDims) -> type:
    r"""Get or create a parametrized DomainMesh type for the given dimension spec.

    Results are cached so that ``DomainMesh[3, 3] is DomainMesh[3, 3]`` holds.

    Parameters
    ----------
    dims : MeshDims
        The dimension specification for the interior mesh.

    Returns
    -------
    type
        A ``_DomainMeshSpecMeta`` instance usable with ``isinstance`` and as a
        type annotation.
    """
    if dims not in _domain_mesh_spec_cache:
        _domain_mesh_spec_cache[dims] = _DomainMeshSpecMeta(
            f"DomainMesh[{dims}]", (), {"_mesh_dims": dims}
        )
    return _domain_mesh_spec_cache[dims]
