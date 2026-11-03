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

r"""Named physical fields and their transformation laws.

A mesh model consumes and produces *named* fields (``"pressure"``,
``"velocity"``, ``"stress"``).  A plain concatenation of their channels is
not enough: a scalar, a Cartesian vector and a rank-2 tensor transform
differently under a change of frame.  This module lets a model declare, per
field, how it transforms, and validate data against that declaration.

Two types, both read-only ``dict`` subclasses, so a declaration serializes to
JSON exactly as it is written (models record their constructor arguments as
JSON in checkpoints):

* :class:`RankSpec` is one field's law: its tensor ``rank`` (0 scalar,
  1 vector, 2 rank-2 tensor, ...), whether a rank >= 2 tensor is
  ``symmetric`` under index permutation, and its ``parity`` (``"even"`` for a
  true tensor, ``"odd"`` for a pseudotensor that flips sign under an improper
  rotation).  It holds the attributes that differ from their defaults:
  ``RankSpec(2, symmetric=True)`` is ``{"rank": 2, "symmetric": True}``.
  Multiple channels are multiple named fields, never an extra unnamed
  feature axis.
* :class:`FieldSchema` maps dotted field names to :class:`RankSpec`.
  :meth:`FieldSchema.parse` is the single entry point for the declaration
  grammar that constructors and configuration files use:

  .. code-block:: python

      FieldSchema.parse({
          "pressure": {"rank": 0},
          "velocity": RankSpec(rank=1),
          "surface": {"shear": {"rank": 1}},        # a nested group ...
          "surface.heat_flux": {"rank": 0},          # ... or the same nesting, dotted
          "stress": {"rank": 2, "symmetric": True},
      })

  A field is a :class:`RankSpec` or a mapping of values (``{"rank": 2,
  "symmetric": True}``); a group is a mapping of fields or groups, and ``{}``
  is an empty group.  A mapping that mixes values with nested mappings is
  refused.  Because the two are told apart by their values rather than by
  key names, a nested field may itself be called ``"rank"`` or ``"parity"``.
  Nested and dotted forms flatten to the same dotted name, and a name cannot
  be both a field and a group of fields.  Validation is construction: an
  invalid schema cannot exist.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal, NoReturn, TypeAlias, Union

from tensordict import TensorDict

_SEP = "."
_ATTRIBUTES = frozenset({"rank", "symmetric", "parity"})


class _ReadOnlyDict(dict):
    """A ``dict`` whose contents are fixed at construction.

    Subclassing ``dict`` keeps instances JSON-serializable as they stand, the
    same design as DoMINO's ``Config``.  Pickling and copying go through each
    subclass's ``__reduce__``, which calls the validating constructor.
    """

    __slots__ = ()

    def _read_only(self, *args: Any, **kwargs: Any) -> NoReturn:
        raise TypeError(f"{type(self).__name__} is immutable")

    __setitem__ = __delitem__ = __ior__ = _read_only
    clear = pop = popitem = setdefault = update = _read_only


class RankSpec(_ReadOnlyDict):
    r"""Transformation law of one named tensor field.

    A read-only ``dict`` of the attributes that differ from their defaults,
    so it serializes to JSON as the declaration a user writes:
    ``RankSpec(2, symmetric=True)`` is ``{"rank": 2, "symmetric": True}``.
    Equal laws are equal specs, however they were spelled.

    Parameters
    ----------
    rank : int
        Number of spatial indices: 0 for a scalar, 1 for a vector, 2 for a
        rank-2 tensor, and so on.
    symmetric : bool, default=False
        Whether the field is fully symmetric under any permutation of its
        indices.  Only meaningful for ``rank >= 2``; rejected below that.
    parity : {"even", "odd"}, default="even"
        ``"even"`` for a true tensor and ``"odd"`` for a pseudotensor, which
        additionally flips sign under an improper rotation.

    Examples
    --------
    >>> spec = RankSpec.parse({"rank": 2, "symmetric": True})
    >>> spec
    RankSpec(rank=2, symmetric=True)
    >>> spec.shape(3), spec.numel(3)
    ((3, 3), 9)
    >>> spec == RankSpec(2, symmetric=True, parity="even")
    True
    """

    __slots__ = ()

    def __init__(
        self,
        rank: int,
        symmetric: bool = False,
        parity: Literal["even", "odd"] = "even",
    ) -> None:
        if isinstance(rank, bool) or not isinstance(rank, int):
            raise TypeError(f"rank must be an integer, got {rank!r}")
        if rank < 0:
            raise ValueError(f"rank must be non-negative, got {rank}")
        if not isinstance(symmetric, bool):
            raise TypeError(f"symmetric must be a bool, got {symmetric!r}")
        if symmetric and rank < 2:
            raise ValueError(
                f"symmetric=True is only meaningful for rank >= 2, got rank {rank}"
            )
        if parity not in ("even", "odd"):
            raise ValueError(f"parity must be 'even' or 'odd', got {parity!r}")
        attributes: dict[str, Any] = {"rank": rank}
        if symmetric:
            attributes["symmetric"] = True
        if parity != "even":
            attributes["parity"] = parity
        dict.__init__(self, attributes)

    @property
    def rank(self) -> int:
        r"""Number of spatial indices."""
        return self["rank"]

    @property
    def symmetric(self) -> bool:
        r"""Whether the field is fully symmetric under index permutation."""
        return self.get("symmetric", False)

    @property
    def parity(self) -> Literal["even", "odd"]:
        r"""``"even"`` for a true tensor, ``"odd"`` for a pseudotensor."""
        return self.get("parity", "even")

    def __repr__(self) -> str:
        return f"RankSpec({', '.join(f'{k}={v!r}' for k, v in self.items())})"

    def __hash__(self) -> int:
        return hash((self.rank, self.symmetric, self.parity))

    def __reduce__(self) -> tuple[type[RankSpec], tuple[int, bool, str]]:
        return (type(self), (self.rank, self.symmetric, self.parity))

    def numel(self, n_spatial_dims: int) -> int:
        r"""Number of components per point, ``n_spatial_dims ** rank``: the
        element count of :meth:`shape`."""
        return n_spatial_dims**self.rank

    def shape(self, n_spatial_dims: int) -> tuple[int, ...]:
        r"""Trailing component shape: ``()`` at rank 0, ``(n_spatial_dims,) * rank``
        above."""
        return () if self.rank == 0 else (n_spatial_dims,) * self.rank

    @classmethod
    def parse(cls, spec: RankSpecLike, *, label: str = "field") -> RankSpec:
        r"""Normalize a field declaration to a :class:`RankSpec`.

        Parameters
        ----------
        spec : RankSpec or Mapping
            An existing :class:`RankSpec`, or a mapping with the required key
            ``"rank"`` and the optional keys ``"symmetric"`` and ``"parity"``.
        label : str, default="field"
            Name of the declaration in error messages.

        Raises
        ------
        TypeError
            If ``spec`` is neither form (a bare integer is refused: write
            ``{"rank": 0}``), or a key has the wrong type.
        ValueError
            If the mapping has unknown keys or lacks ``"rank"``, or a value is
            invalid (negative rank, bad parity, symmetric below rank 2).
        """
        if isinstance(spec, cls):
            return spec
        if not isinstance(spec, Mapping):
            hint = (
                " (write {'rank': %d})" % spec
                if isinstance(spec, int) and not isinstance(spec, bool)
                else ""
            )
            raise TypeError(
                f"{label} must be a RankSpec or a mapping with a 'rank' key; "
                f"got {type(spec).__name__}{hint}"
            )
        unknown = set(spec) - _ATTRIBUTES
        if unknown:
            ### No known key at all: most likely a group whose fields were
            ### written as bare values (``{"surface": {"pressure": 0}}``).
            hint = (
                ""
                if _ATTRIBUTES & set(spec)
                else "; a group of fields maps each name to a declaration such "
                "as {'rank': 0}"
            )
            raise ValueError(
                f"{label} has unknown keys {sorted(map(str, unknown))!r}; "
                f"allowed keys are {sorted(_ATTRIBUTES)!r}{hint}"
            )
        if "rank" not in spec:
            raise ValueError(f"{label} mapping must contain a 'rank' key")
        try:
            return cls(**spec)
        except (TypeError, ValueError) as error:
            raise type(error)(f"{label}: {error}") from None


RankSpecLike: TypeAlias = Union[RankSpec, Mapping[str, Any]]
"""One field's declaration as :meth:`RankSpec.parse` accepts it."""

# TODO: replace with ``type FieldSchemaLike = ...`` after Python 3.11 support is
# dropped (PEP 695).
FieldSchemaLike: TypeAlias = Union[
    "FieldSchema", Mapping[str, Union[RankSpecLike, Mapping[str, Any]]]
]
"""A schema declaration as :meth:`FieldSchema.parse` accepts it: a
:class:`FieldSchema`, or a mapping from names to field declarations and
nested groups."""


def _is_group(value: Mapping[Any, Any], label: str) -> bool:
    """Whether a mapping in a declaration is a group of fields.

    A mapping of mappings (or ``{}``) is a group; a mapping of plain values is
    a field declaration, for :meth:`RankSpec.parse` to check.  A mapping that
    mixes the two is refused.
    """
    nested = sorted(str(k) for k, v in value.items() if isinstance(v, Mapping))
    if nested and len(nested) < len(value):
        plain = sorted(str(k) for k, v in value.items() if not isinstance(v, Mapping))
        raise ValueError(
            f"{label} mixes field attributes {plain!r} with nested fields "
            f"{nested!r}; a field maps 'rank' (and optionally 'symmetric' and "
            f"'parity') to values, and a group maps names to fields"
        )
    return bool(nested) or not value


def _split(name: object, label: str) -> tuple[str, ...]:
    """A dotted field name as its path components, refusing non-strings and
    empty components."""
    if not isinstance(name, str):
        raise TypeError(f"Field names in {label} must be strings; got {name!r}")
    path = tuple(name.split(_SEP))
    if not all(path):
        raise ValueError(f"Field name {name!r} in {label} has an empty path component")
    return path


class FieldSchema(_ReadOnlyDict):
    r"""Read-only ``dict`` from dotted field names to :class:`RankSpec`.

    Build one with :meth:`parse` from the declaration grammar (nested groups,
    dotted names, ``{"rank": ...}`` fields), or directly from a flat mapping
    of :class:`RankSpec` values.  Either way the schema is validated on
    construction: names are non-empty dotted strings, no name is declared
    twice, and no name is both a field and a group of fields.  Insertion order
    is kept.  A schema serializes to JSON as ``{dotted name: declaration}``,
    which :meth:`parse` reads back to an equal schema.

    Parameters
    ----------
    fields : Mapping[str, RankSpec]
        Flat mapping from dotted field name to its transformation law.

    Examples
    --------
    >>> schema = FieldSchema.parse({"pressure": {"rank": 0}, "fluid": {"velocity": {"rank": 1}}})
    >>> list(schema)
    ['pressure', 'fluid.velocity']
    >>> schema.ranks
    {'pressure': 0, 'fluid.velocity': 1}
    >>> schema.key("fluid.velocity")
    ('fluid', 'velocity')
    >>> import json
    >>> json.dumps(schema)
    '{"pressure": {"rank": 0}, "fluid.velocity": {"rank": 1}}'
    """

    __slots__ = ()

    def __init__(self, fields: Mapping[str, RankSpec]) -> None:
        if not isinstance(fields, Mapping):
            raise TypeError(f"fields must be a mapping, got {type(fields).__name__}")
        paths: list[tuple[str, ...]] = []
        for name, spec in fields.items():
            paths.append(_split(name, "FieldSchema"))
            if not isinstance(spec, RankSpec):
                raise TypeError(
                    f"FieldSchema[{name!r}] must be a RankSpec, got "
                    f"{type(spec).__name__}; use FieldSchema.parse for the "
                    f"declaration grammar"
                )
        _check_paths(paths, "FieldSchema")
        dict.__init__(self, fields)

    def __reduce__(self) -> tuple[type[FieldSchema], tuple[dict[str, RankSpec]]]:
        return (type(self), (dict(self),))

    def __repr__(self) -> str:
        return f"FieldSchema({dict(self)!r})"

    # -- construction --------------------------------------------------------

    @classmethod
    def parse(cls, spec: FieldSchemaLike, *, label: str = "fields") -> FieldSchema:
        r"""Parse the declaration grammar into a schema.

        Parameters
        ----------
        spec : FieldSchemaLike
            A :class:`FieldSchema` (returned unchanged), or a mapping whose
            values are fields (a :class:`RankSpec`, or a mapping of values
            such as ``{"rank": 0}``) or nested groups (a mapping of fields or
            groups).  A name may contain ``"."`` to denote the same nesting.
        label : str, default="fields"
            Name of the declaration in error messages, e.g. ``"outputs"``.

        Raises
        ------
        TypeError
            If ``spec`` is not a mapping, a name is not a string, or a field
            declaration is not an accepted form.
        ValueError
            If a field declaration is invalid, a mapping mixes field
            attributes with nested fields, a name is declared twice, a name
            has an empty path component, or a name is both a field and a group
            of fields.
        """
        if isinstance(spec, cls):
            return spec
        if not isinstance(spec, Mapping):
            raise TypeError(f"{label} must be a mapping, got {type(spec).__name__}")

        fields: dict[str, RankSpec] = {}
        paths: list[tuple[str, ...]] = []

        def _walk(group: Mapping[str, Any], prefix: tuple[str, ...]) -> None:
            for name, value in group.items():
                path = (*prefix, *_split(name, label))
                dotted = _SEP.join(path)
                where = f"{label}[{dotted!r}]"
                if (
                    isinstance(value, Mapping)
                    and not isinstance(value, RankSpec)
                    and _is_group(value, where)
                ):
                    _walk(value, path)
                    continue
                fields[dotted] = RankSpec.parse(value, label=where)
                paths.append(path)

        _walk(spec, ())
        _check_paths(paths, label)
        return cls(fields)

    @classmethod
    def from_tensordict(cls, data: TensorDict) -> FieldSchema:
        r"""The schema implied by a TensorDict's leaf shapes.

        A leaf's rank is its number of non-batch dimensions: for point data
        with batch size ``(N,)``, ``(N,)`` is rank 0 and ``(N, D)`` is rank 1.
        Symmetry and parity cannot be read from shapes and take their
        defaults.
        """
        return cls(
            {name: RankSpec(rank=rank) for name, rank in _leaf_ranks(data).items()}
        )

    # -- queries ---------------------------------------------------------------

    @property
    def ranks(self) -> dict[str, int]:
        r"""Dotted field name to integer rank, in schema order."""
        return {name: spec.rank for name, spec in self.items()}

    def count(self, rank: int) -> int:
        r"""Number of fields of the given rank."""
        return sum(1 for spec in self.values() if spec.rank == rank)

    @staticmethod
    def key(name: str) -> str | tuple[str, ...]:
        r"""A dotted field name as a TensorDict key: ``"a"`` -> ``"a"``,
        ``"a.b"`` -> ``("a", "b")``."""
        path = tuple(name.split(_SEP))
        return path[0] if len(path) == 1 else path

    def check(self, data: TensorDict, *, label: str) -> None:
        r"""Raise unless ``data`` holds every field of this schema at its rank.

        Additional leaves in ``data`` are allowed, whatever their names.
        Missing fields and rank mismatches are reported together.

        Parameters
        ----------
        data : TensorDict
            The data to check.
        label : str
            Name of the data in the error message, e.g. ``"boundary data"``.

        Raises
        ------
        ValueError
            If a field is missing or has a different rank.
        """
        actual = _leaf_ranks(data)
        declared = self.ranks
        lines = [
            f"  - missing field {name!r} (declared rank {declared[name]})"
            for name in declared
            if name not in actual
        ]
        lines.extend(
            f"  - rank mismatch for {name!r}: declared {declared[name]}, "
            f"got {actual[name]}"
            for name in declared
            if name in actual and declared[name] != actual[name]
        )
        if lines:
            raise ValueError(
                f"{label} does not contain its declared fields:\n" + "\n".join(lines)
            )


def _leaf_ranks(data: TensorDict) -> dict[str, int]:
    """Dotted leaf name to number of non-batch dimensions, for every leaf."""
    return {
        _SEP.join(key) if isinstance(key, tuple) else key: value.ndim - data.batch_dims
        for key, value in data.items(include_nested=True, leaves_only=True)
    }


def _check_paths(paths: list[tuple[str, ...]], label: str) -> None:
    """Refuse duplicate names and names that are both a field and a group,
    reporting the first offender in declaration order."""
    seen: set[tuple[str, ...]] = set()
    for path in paths:
        if path in seen:
            raise ValueError(
                f"{label} declares the field {_SEP.join(path)!r} more than once"
            )
        seen.add(path)
    for path in paths:
        for depth in range(1, len(path)):
            if path[:depth] in seen:
                raise ValueError(
                    f"{label}: {_SEP.join(path[:depth])!r} is a group of "
                    f"{_SEP.join(path)!r}; a name cannot be both a field and a "
                    f"group of fields"
                )


### Names removed in 2.3, each with what replaces it. Importing one raises an
### ImportError that names the replacement instead of Python's bare "cannot
### import name"; the old functions are not kept, since they read the old
### integer-leaf grammar.
_REMOVED_IN_2_3 = {
    "RankSpecDict": "FieldSchemaLike (a declaration) or FieldSchema (a parsed schema)",
    "flatten_rank_spec": "FieldSchema.parse(spec).ranks",
    "rank_counts": "FieldSchema.parse(spec).count(rank)",
    "ranks_from_tensordict": "FieldSchema.from_tensordict(data).ranks (flat dotted names)",
    "validate_data_contains_ranks": (
        "FieldSchema.parse(declared_ranks).check(data, label=source_label)"
    ),
}


def _missing_attribute(module: str, name: str) -> NoReturn:
    """Raise for an attribute ``module`` lacks: an ImportError naming the
    replacement of a name removed in 2.3, an AttributeError otherwise."""
    if name in _REMOVED_IN_2_3:
        raise ImportError(
            f"{module}.{name} was removed in PhysicsNeMo 2.3, when "
            f"physicsnemo.mesh.fields moved to FieldSchema and RankSpec; use "
            f"{_REMOVED_IN_2_3[name]} instead. Fields are now declared as "
            f"{{'rank': n}} rather than as bare integers.",
            name=module,
        )
    raise AttributeError(f"module {module!r} has no attribute {name!r}")


### Hidden from type checkers, so that they keep reporting unknown names (these
### included) as missing instead of accepting any attribute of this module.
if not TYPE_CHECKING:

    def __getattr__(name: str) -> NoReturn:
        """Point imports of removed names at their replacements (PEP 562)."""
        _missing_attribute(__name__, name)


__all__ = [
    "FieldSchema",
    "FieldSchemaLike",
    "RankSpec",
    "RankSpecLike",
]
