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

r"""Parametric mesh type specifications for dimension-aware type hints.

Provides :class:`MeshDims` and the machinery behind ``Mesh[m, s]`` subscript
syntax, enabling dimension-aware type annotations and runtime ``isinstance``
checks:

.. code-block:: python

    def compute_normals(mesh: Mesh[2, 3]) -> torch.Tensor:
        ...

    assert isinstance(mesh, Mesh[2, 3])  # True for a triangle mesh in 3D

"""

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from physicsnemo.mesh.mesh import Mesh


### Symbolic dimension expression parsing

_DIM_EXPR_PATTERN = re.compile(r"^([a-zA-Z_]\w*)(?:([+-])(\d+))?$")


def _parse_dim_expr(expr: str) -> tuple[str, int]:
    r"""Parse a symbolic dimension expression into ``(variable, offset)``.

    Accepts patterns like ``"n"``, ``"n-1"``, ``"dim+2"``.

    Parameters
    ----------
    expr : str
        A symbolic dimension string.

    Returns
    -------
    tuple[str, int]
        The variable name and integer offset.

    Raises
    ------
    ValueError
        If ``expr`` doesn't match the ``var`` or ``var±int`` pattern.
    """
    match = _DIM_EXPR_PATTERN.match("".join(expr.split()))
    if match is None:
        raise ValueError(
            f"Invalid symbolic dimension expression {expr!r}. "
            f"Expected 'var' or 'var±int' (e.g. 'n', 'n-1', 'dim+2')."
        )
    variable = match.group(1)
    if match.group(2) is None:
        return variable, 0
    sign = 1 if match.group(2) == "+" else -1
    return variable, sign * int(match.group(3))


### MeshDims specification


@dataclass(frozen=True)
class MeshDims:
    r"""Dimension specification for a parametric Mesh type.

    Stores manifold and spatial dimension constraints that can be concrete
    (integers), symbolic (strings like ``"n-1"``), or unconstrained (``None``).
    Frozen and hashable so it can serve as a cache key for parametrized Mesh
    types returned by ``Mesh[m, s]``.

    Parameters
    ----------
    n_manifold_dims : int | str | None
        Manifold (topological) dimension constraint. An ``int`` for a concrete
        value, a ``str`` for a symbolic expression, or ``None`` for unconstrained.
    n_spatial_dims : int | str | None
        Spatial (embedding) dimension constraint, same semantics.

    Raises
    ------
    ValueError
        If concrete dimensions are negative, or manifold exceeds spatial.
    TypeError
        If dimensions are not ``int``, ``str``, or ``None``.
    """

    n_manifold_dims: int | str | None = None
    n_spatial_dims: int | str | None = None

    # Caches for parsed symbolic expressions, populated in __post_init__ when
    # the corresponding dimension is a string. Excluded from init / equality /
    # hash / repr so they don't leak into the public dataclass surface.
    _m_parsed: tuple[str, int] | None = field(
        default=None, init=False, repr=False, compare=False, hash=False
    )
    _s_parsed: tuple[str, int] | None = field(
        default=None, init=False, repr=False, compare=False, hash=False
    )

    def __post_init__(self) -> None:
        for name, val in [
            ("n_manifold_dims", self.n_manifold_dims),
            ("n_spatial_dims", self.n_spatial_dims),
        ]:
            if val is not None and not isinstance(val, (int, str)):
                raise TypeError(
                    f"{name} must be int, str, or None, got {type(val).__name__}"
                )

        if isinstance(self.n_manifold_dims, int) and self.n_manifold_dims < 0:
            raise ValueError(
                f"n_manifold_dims must be non-negative, got {self.n_manifold_dims}"
            )
        if isinstance(self.n_spatial_dims, int) and self.n_spatial_dims < 0:
            raise ValueError(
                f"n_spatial_dims must be non-negative, got {self.n_spatial_dims}"
            )
        if (
            isinstance(self.n_manifold_dims, int)
            and isinstance(self.n_spatial_dims, int)
            and self.n_manifold_dims > self.n_spatial_dims
        ):
            raise ValueError(
                f"n_manifold_dims ({self.n_manifold_dims}) cannot exceed "
                f"n_spatial_dims ({self.n_spatial_dims})"
            )

        # Validate symbolic expressions eagerly and cache parsed results.
        # A symbolic dimension paired with None is meaningless (no codimension
        # constraint can be derived), so reject it early. Frozen dataclass
        # requires object.__setattr__ to populate the post-init cache fields.
        # The isinstance() checks below also narrow the str type for
        # _parse_dim_expr's signature.
        if isinstance(self.n_manifold_dims, str):
            if self.n_spatial_dims is None:
                raise TypeError(
                    f"Symbolic n_manifold_dims={self.n_manifold_dims!r} requires a "
                    f"paired n_spatial_dims (got None). Use both symbolic dims, "
                    f"e.g. Mesh['{self.n_manifold_dims}', 'n']."
                )
            object.__setattr__(self, "_m_parsed", _parse_dim_expr(self.n_manifold_dims))
        if isinstance(self.n_spatial_dims, str):
            if self.n_manifold_dims is None:
                raise TypeError(
                    f"Symbolic n_spatial_dims={self.n_spatial_dims!r} requires a "
                    f"paired n_manifold_dims (got None). Use both symbolic dims, "
                    f"e.g. Mesh['n', '{self.n_spatial_dims}']."
                )
            object.__setattr__(self, "_s_parsed", _parse_dim_expr(self.n_spatial_dims))

    def matches(self, mesh: "Mesh") -> bool:
        r"""Check whether a mesh instance satisfies this dimension spec.

        Concrete (``int``) constraints require exact equality. Symbolic (``str``)
        constraints with a shared variable validate the implied codimension.
        Unconstrained (``None``) dimensions match anything.

        Parameters
        ----------
        mesh : Mesh
            The mesh instance to check.

        Returns
        -------
        bool
            ``True`` if the mesh's dimensions satisfy all constraints.
        """
        if (
            isinstance(self.n_manifold_dims, int)
            and mesh.n_manifold_dims != self.n_manifold_dims
        ):
            return False
        if (
            isinstance(self.n_spatial_dims, int)
            and mesh.n_spatial_dims != self.n_spatial_dims
        ):
            return False
        if isinstance(self.n_manifold_dims, str) and isinstance(
            self.n_spatial_dims, str
        ):
            return self._check_symbolic_constraint(mesh)
        return True

    def _check_symbolic_constraint(self, mesh: "Mesh") -> bool:
        r"""Validate symbolic codimension constraints against a mesh.

        When both dimensions use the same variable (e.g. ``"n-1"`` and ``"n"``),
        the expected codimension is ``spatial_offset - manifold_offset``.
        Different variables impose no constraint.
        """
        # By construction (matches() only delegates here when both dims are
        # str), __post_init__ has populated both parsed caches. The if-guard
        # is defensive and acts as a type narrower for the parsed tuples.
        if self._m_parsed is None or self._s_parsed is None:
            return True
        m_var, m_off = self._m_parsed
        s_var, s_off = self._s_parsed
        if m_var != s_var:
            return True
        expected_codim = s_off - m_off
        return mesh.codimension == expected_codim

    @property
    def is_concrete(self) -> bool:
        """Whether both dimensions are concrete integers."""
        return isinstance(self.n_manifold_dims, int) and isinstance(
            self.n_spatial_dims, int
        )

    @property
    def boundary(self) -> "MeshDims":
        r"""Derive the boundary dimension spec (manifold dim decremented by 1).

        For concrete dimensions, ``MeshDims(2, 3).boundary`` gives
        ``MeshDims(1, 3)``. For symbolic dimensions, ``MeshDims("n", "n+1").boundary``
        gives ``MeshDims("n-1", "n+1")``.

        Returns
        -------
        MeshDims
            A new spec with ``n_manifold_dims`` reduced by 1.

        Raises
        ------
        ValueError
            If ``n_manifold_dims`` is 0.
        TypeError
            If ``n_manifold_dims`` is unconstrained (``None``).
        """
        if isinstance(self.n_manifold_dims, int):
            if self.n_manifold_dims == 0:
                raise ValueError("0-dimensional manifold has no boundary")
            return MeshDims(self.n_manifold_dims - 1, self.n_spatial_dims)
        if isinstance(self.n_manifold_dims, str):
            m_var, m_off = _parse_dim_expr(self.n_manifold_dims)
            new_offset = m_off - 1
            if new_offset == 0:
                new_expr = m_var
            elif new_offset > 0:
                new_expr = f"{m_var}+{new_offset}"
            else:
                new_expr = f"{m_var}{new_offset}"
            return MeshDims(new_expr, self.n_spatial_dims)
        raise TypeError("Cannot derive boundary of unconstrained manifold dimension")

    def _format_dim(self, value: int | str | None) -> str:
        """Format a single dimension value for display."""
        if value is None:
            return "..."
        if isinstance(value, str):
            return repr(value)
        return str(value)

    def __str__(self) -> str:
        return (
            f"{self._format_dim(self.n_manifold_dims)}, "
            f"{self._format_dim(self.n_spatial_dims)}"
        )


### Metaclass for parametrized Mesh types


class _MeshSpecMeta(type):
    r"""Metaclass enabling ``isinstance(mesh, Mesh[2, 3])`` checks.

    Each instance of this metaclass is a synthetic type representing a
    dimension-constrained Mesh. It is not a subclass of Mesh and cannot be
    instantiated - it exists purely for ``isinstance`` checks, ``repr``,
    and derived-type navigation (e.g. ``.boundary``).
    """

    _mesh_dims: MeshDims

    def __instancecheck__(cls, instance: object) -> bool:
        from physicsnemo.mesh.mesh import Mesh

        # Mesh's metaclass is plain ``type`` (not _MeshSpecMeta), so this
        # ``isinstance`` call dispatches to the default implementation - no
        # recursion risk - and narrows ``instance`` to ``Mesh`` for the
        # subsequent matches() call.
        if not isinstance(instance, Mesh):
            return False
        return cls._mesh_dims.matches(instance)

    def __repr__(cls) -> str:
        return f"Mesh[{cls._mesh_dims}]"

    @property
    def boundary(cls) -> type:
        """The boundary type: ``Mesh[m, s].boundary`` gives ``Mesh[m-1, s]``."""
        return _get_mesh_spec(cls._mesh_dims.boundary)


### Cached factory

_mesh_spec_cache: dict[MeshDims, type] = {}


def _get_mesh_spec(dims: MeshDims) -> type:
    r"""Get or create a parametrized Mesh type for the given dimension spec.

    Results are cached so that ``Mesh[2, 3] is Mesh[2, 3]`` holds.

    Parameters
    ----------
    dims : MeshDims
        The dimension specification.

    Returns
    -------
    type
        A ``_MeshSpecMeta`` instance usable with ``isinstance`` and as a
        type annotation.
    """
    if dims not in _mesh_spec_cache:
        _mesh_spec_cache[dims] = _MeshSpecMeta(
            f"Mesh[{dims}]", (), {"_mesh_dims": dims}
        )
    return _mesh_spec_cache[dims]
