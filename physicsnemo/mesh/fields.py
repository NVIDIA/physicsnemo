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

Two types:

* :class:`RankSpec` is one field's law: its tensor ``rank`` (0 scalar,
  1 vector, 2 rank-2 tensor, ...), whether a rank >= 2 tensor is
  ``symmetric`` under index permutation, and its ``parity`` (``"even"`` for a
  true tensor, ``"odd"`` for a pseudotensor that flips sign under an improper
  rotation).  Multiple channels are multiple named fields, never an extra
  unnamed feature axis.
* :class:`FieldSchema` is an immutable mapping from field names to
  :class:`RankSpec`.  :meth:`FieldSchema.parse` is the single entry point for
  the declaration grammar that constructors and configuration files use:

  .. code-block:: python

      FieldSchema.parse({
          "pressure": {"rank": 0},
          "velocity": RankSpec(rank=1),
          "surface": {"shear": {"rank": 1}},        # a nested group ...
          "surface.heat_flux": {"rank": 0},          # ... or the same nesting, dotted
          "stress": {"rank": 2, "symmetric": True},
      })

  A mapping with a ``"rank"`` key is a field; any other mapping is a group of
  fields.  Nested and dotted forms flatten to the same dotted name, and a
  name cannot be both a field and a group of fields.  Validation is
  construction: an invalid schema cannot exist.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias, Union

from tensordict import TensorDict

_SEP = "."


@dataclass(frozen=True)
class RankSpec:
    r"""Transformation law of one named tensor field.

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
    >>> RankSpec.parse({"rank": 2, "symmetric": True}).shape(3)
    (3, 3)
    """

    rank: int
    symmetric: bool = False
    parity: Literal["even", "odd"] = "even"

    def __post_init__(self) -> None:
        if isinstance(self.rank, bool) or not isinstance(self.rank, int):
            raise TypeError(f"rank must be an integer, got {self.rank!r}")
        if self.rank < 0:
            raise ValueError(f"rank must be non-negative, got {self.rank}")
        if not isinstance(self.symmetric, bool):
            raise TypeError(f"symmetric must be a bool, got {self.symmetric!r}")
        if self.symmetric and self.rank < 2:
            raise ValueError(
                f"symmetric=True is only meaningful for rank >= 2, got rank {self.rank}"
            )
        if self.parity not in ("even", "odd"):
            raise ValueError(f"parity must be 'even' or 'odd', got {self.parity!r}")

    def size(self, dim: int) -> int:
        r"""Number of components per point in ``dim`` spatial dimensions."""
        return dim**self.rank

    def shape(self, dim: int) -> tuple[int, ...]:
        r"""Trailing component shape: ``()`` at rank 0, ``(dim,) * rank`` above."""
        return () if self.rank == 0 else (dim,) * self.rank

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
            hint = " (write {'rank': %d})" % spec if isinstance(spec, int) else ""
            raise TypeError(
                f"{label} must be a RankSpec or a mapping with a 'rank' key; "
                f"got {type(spec).__name__}{hint}"
            )
        unknown = set(spec) - {"rank", "symmetric", "parity"}
        if unknown:
            raise ValueError(
                f"{label} has unknown keys {sorted(map(str, unknown))!r}; "
                f"allowed keys are ['parity', 'rank', 'symmetric']"
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


def _is_field(value: object) -> bool:
    """A mapping with a ``"rank"`` key or a RankSpec is a field; any other
    mapping is a group.  Anything else is a (bad) field, so that
    :meth:`RankSpec.parse` reports it."""
    return not isinstance(value, Mapping) or "rank" in value


def _split(name: object, label: str) -> tuple[str, ...]:
    if not isinstance(name, str):
        raise TypeError(f"Field names in {label} must be strings; got {name!r}")
    path = tuple(name.split(_SEP))
    if not all(path):
        raise ValueError(f"Field name {name!r} in {label} has an empty path component")
    return path


@dataclass(frozen=True, eq=False, repr=False)
class FieldSchema(Mapping[str, RankSpec]):
    r"""Immutable mapping from dotted field names to :class:`RankSpec`.

    Build one with :meth:`parse` from the declaration grammar (nested groups,
    dotted names, ``{"rank": ...}`` leaves), or directly from a flat mapping
    of :class:`RankSpec` values.  Either way the schema is validated on
    construction: names are non-empty dotted strings, no name is declared
    twice, and no name is both a field and a group of fields.  Insertion order
    is kept.

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
    """

    fields: Mapping[str, RankSpec]

    def __post_init__(self) -> None:
        if not isinstance(self.fields, Mapping):
            raise TypeError(
                f"fields must be a mapping, got {type(self.fields).__name__}"
            )
        paths: list[tuple[str, ...]] = []
        for name, spec in self.fields.items():
            paths.append(_split(name, "FieldSchema"))
            if not isinstance(spec, RankSpec):
                raise TypeError(
                    f"FieldSchema[{name!r}] must be a RankSpec, got "
                    f"{type(spec).__name__}; use FieldSchema.parse for the "
                    f"declaration grammar"
                )
        _check_paths(paths, "FieldSchema")
        object.__setattr__(self, "fields", dict(self.fields))

    # -- construction --------------------------------------------------------

    @classmethod
    def parse(cls, spec: FieldSchemaLike, *, label: str = "fields") -> FieldSchema:
        r"""Parse the declaration grammar into a schema.

        Parameters
        ----------
        spec : FieldSchemaLike
            A :class:`FieldSchema` (returned unchanged), or a mapping whose
            values are field declarations (:class:`RankSpec` or a mapping with
            a ``"rank"`` key) or nested groups (any other mapping).  A name may
            contain ``"."`` to denote the same nesting.
        label : str, default="fields"
            Name of the declaration in error messages, e.g. ``"outputs"``.

        Raises
        ------
        TypeError
            If ``spec`` or a group is not a mapping, a name is not a string,
            or a field declaration is not an accepted form.
        ValueError
            If a field declaration is invalid, a name is declared twice, a
            name has an empty path component, or a name is both a field and a
            group of fields.
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
                if not _is_field(value):
                    _walk(value, path)
                    continue
                dotted = _SEP.join(path)
                fields[dotted] = RankSpec.parse(value, label=f"{label}[{dotted!r}]")
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
        fields: dict[str, RankSpec] = {}
        for key, value in data.items(include_nested=True, leaves_only=True):
            name = key if isinstance(key, str) else _SEP.join(key)
            fields[name] = RankSpec(rank=value.ndim - data.batch_dims)
        return cls(fields)

    # -- Mapping protocol ------------------------------------------------------

    def __getitem__(self, name: str) -> RankSpec:
        return self.fields[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self.fields)

    def __len__(self) -> int:
        return len(self.fields)

    def __repr__(self) -> str:
        return f"FieldSchema({self.fields!r})"

    # -- queries ---------------------------------------------------------------

    @property
    def ranks(self) -> dict[str, int]:
        r"""Dotted field name to integer rank, in schema order."""
        return {name: spec.rank for name, spec in self.fields.items()}

    def count(self, rank: int) -> int:
        r"""Number of fields of the given rank."""
        return sum(1 for spec in self.fields.values() if spec.rank == rank)

    @staticmethod
    def key(name: str) -> str | tuple[str, ...]:
        r"""A dotted field name as a TensorDict key: ``"a"`` -> ``"a"``,
        ``"a.b"`` -> ``("a", "b")``."""
        path = tuple(name.split(_SEP))
        return path[0] if len(path) == 1 else path

    def check(self, data: TensorDict, *, label: str) -> None:
        r"""Raise unless ``data`` holds every field of this schema at its rank.

        Additional leaves in ``data`` are allowed.  Missing fields and rank
        mismatches are reported together.

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
        actual = FieldSchema.from_tensordict(data).ranks
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


def _check_paths(paths: list[tuple[str, ...]], label: str) -> None:
    """Refuse duplicate names and names that are both a field and a group."""
    seen: set[tuple[str, ...]] = set()
    for path in paths:
        if path in seen:
            raise ValueError(
                f"{label} declares the field {_SEP.join(path)!r} more than once"
            )
        seen.add(path)
    for path in seen:
        for depth in range(1, len(path)):
            if path[:depth] in seen:
                raise ValueError(
                    f"{label}: {_SEP.join(path[:depth])!r} is a group of "
                    f"{_SEP.join(path)!r}; a name cannot be both a field and a "
                    f"group of fields"
                )


__all__ = [
    "FieldSchema",
    "FieldSchemaLike",
    "RankSpec",
    "RankSpecLike",
]
