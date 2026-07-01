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

r"""Semantic scalar and vector field layout utilities.

Mesh models commonly need to turn named physical fields into dense channel
tensors.  A plain concatenation is not sufficient: a scalar field and a
Cartesian vector field transform differently under a change of frame.  This
module keeps those two representations separate while giving them a stable,
name-based channel order.

``RankSpecDict`` describes the semantic tensor rank of every named leaf.  The
currently supported :class:`FieldLayout` leaves are

* rank 0: one scalar per point, with shape ``(N,)``;
* rank 1: one polar vector per point, with shape ``(N, D)``.

Consequently, a rank-1 leaf is not an arbitrary feature axis of length ``D``:
it is explicitly promised to transform as a polar vector.  Axial vectors and
higher-order tensors require additional representation metadata and are not
silently treated as rank-1 fields here.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Collection
from copy import deepcopy
from typing import NamedTuple, TypeAlias, Union

import torch
from tensordict import TensorDict

# TODO: replace with ``type RankSpecDict = ...`` after Python 3.11 support is
# dropped (PEP 695).
RankSpecDict: TypeAlias = dict[str, Union[int, "RankSpecDict"]]


def flatten_rank_spec(rank_spec: RankSpecDict, sep: str = ".") -> dict[str, int]:
    r"""Flatten a nested rank specification to separator-joined names.

    The insertion order of ``rank_spec`` is retained for compatibility.  Code
    that assigns positional channels should sort the returned names, as
    :class:`FieldLayout` does.

    Parameters
    ----------
    rank_spec : RankSpecDict
        Mapping from field names to integer ranks or nested mappings.
    sep : str, default="."
        Separator used to join nested path components.

    Returns
    -------
    dict[str, int]
        Flat field-name to semantic-rank mapping.

    Examples
    --------
    >>> flatten_rank_spec({"pressure": 0, "velocity": 1})
    {'pressure': 0, 'velocity': 1}
    >>> flatten_rank_spec({"fluid": {"pressure": 0, "velocity": 1}})
    {'fluid.pressure': 0, 'fluid.velocity': 1}
    """
    result: dict[str, int] = {}
    for key, value in rank_spec.items():
        if isinstance(value, dict):
            for sub_key, rank in flatten_rank_spec(value, sep=sep).items():
                result[f"{key}{sep}{sub_key}"] = rank
        else:
            result[key] = value
    return result


def validate_rank_spec(
    rank_spec: RankSpecDict,
    *,
    allowed_ranks: Collection[int] | None = None,
    source_label: str = "rank_spec",
) -> None:
    r"""Validate the structure and integer leaves of a rank specification.

    Parameters
    ----------
    rank_spec : RankSpecDict
        Rank specification to validate.
    allowed_ranks : Collection[int] or None, optional
        If supplied, every leaf must belong to this collection.  Otherwise all
        non-negative integer ranks are accepted.
    source_label : str, default="rank_spec"
        Human-readable name used in error messages.

    Raises
    ------
    TypeError
        If a key is not a string or a leaf is not an integer.
    ValueError
        If a rank is negative or is not in ``allowed_ranks``.
    """
    allowed = None if allowed_ranks is None else frozenset(allowed_ranks)

    def _validate(spec: RankSpecDict, path: tuple[str, ...]) -> None:
        for key, value in spec.items():
            if not isinstance(key, str):
                raise TypeError(
                    f"All keys in {source_label} must be strings; got "
                    f"{key!r} at {'.'.join(path) or '<root>'}"
                )
            leaf_path = (*path, key)
            if isinstance(value, dict):
                _validate(value, leaf_path)
                continue
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    f"Rank for {'.'.join(leaf_path)!r} in {source_label} must "
                    f"be an integer, got {value!r}"
                )
            if value < 0:
                raise ValueError(
                    f"Rank for {'.'.join(leaf_path)!r} in {source_label} must "
                    f"be non-negative, got {value}"
                )
            if allowed is not None and value not in allowed:
                raise ValueError(
                    f"Rank for {'.'.join(leaf_path)!r} in {source_label} must "
                    f"be one of {sorted(allowed)}, got {value}"
                )

    if not isinstance(rank_spec, dict):
        raise TypeError(
            f"{source_label} must be a dict, got {type(rank_spec).__name__}"
        )
    _validate(rank_spec, ())


def rank_counts(rank_spec: RankSpecDict) -> Counter[int]:
    r"""Count leaves of each semantic rank in ``rank_spec``."""
    return Counter(flatten_rank_spec(rank_spec).values())


def ranks_from_tensordict(td: TensorDict) -> RankSpecDict:
    r"""Derive semantic-rank-shaped metadata from TensorDict leaf shapes.

    A leaf rank is its number of non-batch dimensions.  For a point-field
    TensorDict with batch size ``(N,)``, ``(N,)`` is therefore rank 0 and
    ``(N, D)`` is rank 1.
    """
    result: RankSpecDict = {}
    for key in td.keys():
        value = td[key]
        if isinstance(value, TensorDict):
            result[key] = ranks_from_tensordict(value)  # ty: ignore[invalid-assignment]
        else:
            result[key] = value.ndim - td.batch_dims  # ty: ignore[invalid-assignment]
    return result


def validate_data_contains_ranks(
    *,
    data: TensorDict,
    declared_ranks: RankSpecDict,
    source_label: str,
) -> None:
    r"""Validate that ``data`` contains every declared leaf at its stated rank.

    Additional leaves in ``data`` are allowed.  Missing leaves and rank
    mismatches are reported together to make schema errors easier to diagnose.
    """
    validate_rank_spec(declared_ranks, source_label="declared_ranks")
    declared = flatten_rank_spec(declared_ranks)
    actual = flatten_rank_spec(ranks_from_tensordict(data))

    lines = [
        f"  - missing leaf {key!r} (declared rank {declared[key]})"
        for key in sorted(declared.keys() - actual.keys())
    ]
    lines.extend(
        [
            f"  - rank mismatch for {key!r}: declared {declared[key]}, "
            f"got {actual[key]}"
            for key in sorted(declared.keys() & actual.keys())
            if declared[key] != actual[key]
        ]
    )
    if lines:
        raise ValueError(
            f"{source_label} does not contain its declared rank spec:\n"
            + "\n".join(lines)
        )


class ScalarVectorFields(NamedTuple):
    r"""Dense scalar and polar-vector channels for a collection of points.

    ``scalars`` has shape ``(N, C_s)`` and ``vectors`` has shape
    ``(N, C_v, D)``.  Shape validation belongs to the :class:`FieldLayout`
    that defines ``C_s``, ``C_v``, and ``D``.
    """

    scalars: torch.Tensor
    vectors: torch.Tensor


def _rank_entries(
    rank_spec: RankSpecDict,
    path: tuple[str, ...] = (),
) -> list[tuple[tuple[str, ...], int]]:
    """Return rank leaves as unambiguous key paths."""
    entries: list[tuple[tuple[str, ...], int]] = []
    for key, value in rank_spec.items():
        leaf_path = (*path, key)
        if isinstance(value, dict):
            entries.extend(_rank_entries(value, leaf_path))
        else:
            entries.append((leaf_path, value))
    return entries


def _td_key(path: tuple[str, ...]) -> str | tuple[str, ...]:
    """Use a string for top-level TensorDict keys and tuples for nested keys."""
    return path[0] if len(path) == 1 else path


class FieldLayout:
    r"""Deterministic packing layout for named scalar and polar-vector fields.

    Channel order is lexicographic by the flattened field name and therefore
    does not depend on dictionary or TensorDict construction order.  Scalars
    and vectors are packed separately so that equivariant code cannot
    accidentally mix their transformation laws.

    Parameters
    ----------
    rank_spec : RankSpecDict
        Nested or flat field schema.  Only rank-0 and rank-1 leaves are
        supported.
    spatial_dim : int
        Cartesian vector dimension ``D``.
    sep : str, default="."
        Separator used only for public flattened names and sorting.

    Notes
    -----
    Each rank-0 leaf must have shape ``(N,)`` and each rank-1 leaf must have
    shape ``(N, D)``.  Multiple channels must be represented by multiple named
    leaves rather than an extra unnamed feature dimension.
    """

    def __init__(
        self,
        rank_spec: RankSpecDict,
        spatial_dim: int,
        *,
        sep: str = ".",
    ) -> None:
        validate_rank_spec(
            rank_spec,
            allowed_ranks=(0, 1),
            source_label="rank_spec",
        )
        if isinstance(spatial_dim, bool) or not isinstance(spatial_dim, int):
            raise TypeError(f"spatial_dim must be an integer, got {spatial_dim!r}")
        if spatial_dim < 1:
            raise ValueError(f"spatial_dim must be positive, got {spatial_dim}")
        if not isinstance(sep, str):
            raise TypeError(f"sep must be a string, got {type(sep).__name__}")
        if not sep:
            raise ValueError("sep must not be empty")

        entries = [
            (sep.join(path), path, rank) for path, rank in _rank_entries(rank_spec)
        ]
        if not entries:
            raise ValueError("rank_spec must contain at least one field leaf")

        # A literal key containing ``sep`` can otherwise collide with a nested
        # path in public names and make deterministic ordering ambiguous.
        names = [name for name, _, _ in entries]
        if len(names) != len(set(names)):
            raise ValueError(
                f"rank_spec contains field paths that collide when flattened "
                f"with separator {sep!r}"
            )

        entries.sort(key=lambda entry: entry[0])
        self._rank_spec = deepcopy(rank_spec)
        self.spatial_dim = spatial_dim
        self.sep = sep
        self._entries = tuple(entries)
        self._scalar_entries = tuple(entry for entry in entries if entry[2] == 0)
        self._vector_entries = tuple(entry for entry in entries if entry[2] == 1)

    @property
    def rank_spec(self) -> RankSpecDict:
        """A defensive copy of the named-field schema."""
        return deepcopy(self._rank_spec)

    @property
    def flat_rank_spec(self) -> dict[str, int]:
        """Field ranks in deterministic packed-name order."""
        return {name: rank for name, _, rank in self._entries}

    @property
    def scalar_names(self) -> tuple[str, ...]:
        """Names of scalar channels in packed order."""
        return tuple(name for name, _, _ in self._scalar_entries)

    @property
    def vector_names(self) -> tuple[str, ...]:
        """Names of polar-vector channels in packed order."""
        return tuple(name for name, _, _ in self._vector_entries)

    @property
    def n_scalars(self) -> int:
        """Number of packed scalar channels."""
        return len(self._scalar_entries)

    @property
    def n_vectors(self) -> int:
        """Number of packed polar-vector channels."""
        return len(self._vector_entries)

    def pack(self, data: TensorDict) -> ScalarVectorFields:
        r"""Pack named point fields into deterministic scalar/vector channels.

        Extra leaves in ``data`` are ignored.  All selected leaves must share
        dtype and device, in addition to satisfying the semantic shapes in this
        layout.
        """
        if data.batch_dims != 1:
            raise ValueError(
                "FieldLayout expects a point TensorDict with one batch dimension "
                f"(N,), got batch_size={tuple(data.batch_size)}"
            )
        validate_data_contains_ranks(
            data=data,
            declared_ranks=self._rank_spec,
            source_label="data",
        )

        n_points = data.batch_size[0]
        selected: list[torch.Tensor] = []
        scalar_tensors: list[torch.Tensor] = []
        vector_tensors: list[torch.Tensor] = []

        for name, path, rank in self._entries:
            tensor = data[_td_key(path)]
            expected = (n_points,) if rank == 0 else (n_points, self.spatial_dim)
            if tuple(tensor.shape) != expected:
                kind = "scalar" if rank == 0 else "polar vector"
                raise ValueError(
                    f"Field {name!r} is declared as a {kind} and must have "
                    f"shape {expected}, got {tuple(tensor.shape)}"
                )
            selected.append(tensor)
            if rank == 0:
                scalar_tensors.append(tensor)
            else:
                vector_tensors.append(tensor)

        reference = selected[0]
        for name, tensor in zip(
            (name for name, _, _ in self._entries), selected, strict=True
        ):
            if tensor.device != reference.device:
                raise ValueError(
                    f"All packed fields must share a device; field {name!r} is "
                    f"on {tensor.device}, expected {reference.device}"
                )
            if tensor.dtype != reference.dtype:
                raise ValueError(
                    f"All packed fields must share a dtype; field {name!r} has "
                    f"{tensor.dtype}, expected {reference.dtype}"
                )

        scalars = (
            torch.stack(scalar_tensors, dim=-1)
            if scalar_tensors
            else reference.new_empty((n_points, 0))
        )
        vectors = (
            torch.stack(vector_tensors, dim=-2)
            if vector_tensors
            else reference.new_empty((n_points, 0, self.spatial_dim))
        )
        return ScalarVectorFields(scalars=scalars, vectors=vectors)

    def unpack(self, fields: ScalarVectorFields) -> TensorDict:
        r"""Unpack scalar/vector channels to a TensorDict matching ``rank_spec``."""
        scalars, vectors = fields
        if scalars.ndim != 2:
            raise ValueError(
                f"scalars must have shape (N, {self.n_scalars}), got "
                f"{tuple(scalars.shape)}"
            )
        if vectors.ndim != 3:
            raise ValueError(
                f"vectors must have shape (N, {self.n_vectors}, "
                f"{self.spatial_dim}), got {tuple(vectors.shape)}"
            )

        n_points = scalars.shape[0]
        expected_scalars = (n_points, self.n_scalars)
        expected_vectors = (n_points, self.n_vectors, self.spatial_dim)
        if tuple(scalars.shape) != expected_scalars:
            raise ValueError(
                f"scalars must have shape {expected_scalars}, got "
                f"{tuple(scalars.shape)}"
            )
        if tuple(vectors.shape) != expected_vectors:
            raise ValueError(
                f"vectors must have shape {expected_vectors}, got "
                f"{tuple(vectors.shape)}"
            )
        if vectors.shape[0] != n_points:
            raise ValueError(
                "scalars and vectors must contain the same number of points"
            )
        if vectors.device != scalars.device:
            raise ValueError("scalars and vectors must be on the same device")
        if vectors.dtype != scalars.dtype:
            raise ValueError("scalars and vectors must have the same dtype")

        nested: dict[str, object] = {}

        def _assign(path: tuple[str, ...], tensor: torch.Tensor) -> None:
            target = nested
            for key in path[:-1]:
                child = target.setdefault(key, {})
                # The schema validation guarantees that a path component is
                # either always a group or always a leaf.
                target = child  # type: ignore[assignment]
            target[path[-1]] = tensor

        for index, (_, path, _) in enumerate(self._scalar_entries):
            _assign(path, scalars[:, index])
        for index, (_, path, _) in enumerate(self._vector_entries):
            _assign(path, vectors[:, index, :])

        return TensorDict(
            nested,
            batch_size=torch.Size([n_points]),
            device=scalars.device,
        )


__all__ = [
    "FieldLayout",
    "RankSpecDict",
    "ScalarVectorFields",
    "flatten_rank_spec",
    "rank_counts",
    "ranks_from_tensordict",
    "validate_data_contains_ranks",
    "validate_rank_spec",
]
