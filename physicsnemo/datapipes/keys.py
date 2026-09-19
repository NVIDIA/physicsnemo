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

"""
Nested-key helpers for addressing fields inside (possibly nested) TensorDicts.

A :class:`~tensordict.TensorDict` is a tree: a key is either a plain string
(top-level leaf or sub-TensorDict) or a tuple of strings addressing a leaf
inside nested sub-TensorDicts, e.g. ``("solution", "gauge_pressure")``.
TensorDict never parses separators inside a string, so ``td["a.b"]`` looks up
a literal top-level key named ``"a.b"``.

YAML / Hydra configs can only express strings (and lists of strings), so
every datapipe component that takes a field name from a config routes it
through :func:`as_nested_key`. The convention is:

- A string is split on ``"."``: ``"solution.gauge_pressure"`` addresses the
  leaf ``("solution", "gauge_pressure")``. This matches the separator used by
  ``TensorDict.flatten_keys`` and by the existing ``Rename`` / ``Purge`` /
  ``RestructureTensorDict`` transforms.
- A list / tuple of strings is taken verbatim as the key path (the form to use
  from Python when the key is already a tuple).

``"."`` and ``"/"`` are therefore reserved in field names: ``"."`` is the
config-string nesting separator and ``"/"`` is the on-disk one (VTK, HDF5,
zarr). A leaf whose own name contains either character is not supported;
name it something else.

Everything else in this module exists so that downstream code never has to
iterate ``td.keys()`` without ``include_nested=True, leaves_only=True``, and
never has to flatten / unflatten a TensorDict just to reach a nested field.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import TypeAlias

import torch
from tensordict import TensorDict

### Public alias for a TensorDict key: a top-level string or a nested path.
NestedKey: TypeAlias = str | tuple[str, ...]

### Separator used when a nested key is written as a single config string.
### This is the default ``separator`` of ``TensorDict.flatten_keys`` /
### ``unflatten_keys``; tensordict does not export it as a constant, so it is
### spelled out here and pinned by a test.
KEY_SEPARATOR = "."


def as_nested_key(key: str | Sequence[str]) -> NestedKey:
    r"""Normalize a config-supplied field name into a TensorDict key.

    Parameters
    ----------
    key : str or sequence of str
        Either a ``"."``-separated path string (``"solution.pressure"``) or an
        explicit sequence of path components (``["solution", "pressure"]``).
        A single-component path collapses to a plain string, matching how
        TensorDict itself normalizes 1-tuples.

    Returns
    -------
    str or tuple[str, ...]
        ``"pressure"`` for a top-level field, ``("solution", "pressure")``
        for a nested one.

    Raises
    ------
    TypeError
        If ``key`` is neither a string nor a sequence of strings.
    ValueError
        If any path component is empty (e.g. ``"a..b"`` or ``""``).

    Examples
    --------
    >>> as_nested_key("pressure")
    'pressure'
    >>> as_nested_key("solution.pressure")
    ('solution', 'pressure')
    >>> as_nested_key(["solution", "pressure"])
    ('solution', 'pressure')
    """
    if isinstance(key, str):
        parts: tuple[str, ...] = tuple(key.split(KEY_SEPARATOR))
    elif isinstance(key, Sequence):
        parts = tuple(key)
        if not all(isinstance(p, str) for p in parts):
            raise TypeError(f"Nested key components must all be strings, got {key!r}.")
    else:
        raise TypeError(
            f"Field name must be a string or a sequence of strings, "
            f"got {type(key).__name__}: {key!r}."
        )
    if not parts or any(p == "" for p in parts):
        raise ValueError(
            f"Invalid field name {key!r}: every '{KEY_SEPARATOR}'-separated "
            f"component must be non-empty."
        )
    return parts[0] if len(parts) == 1 else parts


def key_to_str(key: NestedKey) -> str:
    r"""Render a TensorDict key as a single ``"."``-joined string.

    Inverse of :func:`as_nested_key` for display, error messages, and any
    flat namespace (logging tags, file names) that needs a string.
    """
    if isinstance(key, str):
        return key
    return KEY_SEPARATOR.join(key)


def leaf_keys(td: TensorDict) -> list[NestedKey]:
    r"""All leaf (tensor) keys of ``td``, descending into nested sub-TensorDicts."""
    return list(td.keys(include_nested=True, leaves_only=True))


def format_leaf_keys(td: TensorDict) -> list[str]:
    r"""Sorted ``"."``-joined leaf key names, for error messages."""
    return sorted(key_to_str(k) for k in leaf_keys(td))


def with_leaf_name(key: NestedKey, fn: Callable[[str], str]) -> NestedKey:
    r"""Apply ``fn`` to the last path component of ``key``, keeping its parents.

    Used to derive output names from input names without collapsing the
    nesting, e.g. ``with_leaf_name(("solution", "v"), lambda n: f"knn_{n}")``
    gives ``("solution", "knn_v")``.
    """
    if isinstance(key, str):
        return fn(key)
    return (*key[:-1], fn(key[-1]))


def get_leaf(td: TensorDict, key: NestedKey, *, what: str = "Field") -> torch.Tensor:
    r"""Look up a leaf tensor, with actionable errors for the two common mistakes.

    Parameters
    ----------
    td : TensorDict
        Container to read from.
    key : str or tuple[str, ...]
        Key as returned by :func:`as_nested_key`.
    what : str
        Noun used in error messages (``"Field"``, ``"Reference key"``, ...).

    Raises
    ------
    KeyError
        If ``key`` is absent. The message lists the available leaf keys with
        their nesting spelled out, rather than only the top-level names.
    TypeError
        If ``key`` names a sub-TensorDict (a group of fields) rather than a
        single tensor field.
    """
    ### ``TensorDict.get`` returns ``None`` for a missing key (including when
    ### a path component is missing); it raises ``ValueError`` when a path
    ### tries to descend *through* a leaf tensor. Both are "not found" here.
    try:
        value = td.get(key, None)
    except ValueError:
        value = None
    if value is None:
        raise KeyError(
            f"{what} {key_to_str(key)!r} not found in data. "
            f"Available fields: {format_leaf_keys(td)}"
        )
    if isinstance(value, TensorDict):
        raise TypeError(
            f"{what} {key_to_str(key)!r} names a group of fields "
            f"{format_leaf_keys(value)}, not a single field. Address one of "
            f"its leaves, e.g. {key_to_str(key)!r} + '.' + <leaf name>."
        )
    return value


def require_keys(
    td: TensorDict, keys: Iterable[NestedKey], *, what: str = "Field"
) -> None:
    r"""Raise ``KeyError`` listing the available leaves if any key is absent from ``td``.

    Use before ``td.select(*keys)``: TensorDict's own error for a path that
    runs through a leaf tensor is an ``AttributeError``, and its message for
    a missing key lists only the top level.
    """
    missing = [key_to_str(k) for k in keys if k not in td]
    if missing:
        raise KeyError(
            f"{what}{'s' if len(missing) > 1 else ''} {missing} not found in data. "
            f"Available fields: {format_leaf_keys(td)}"
        )


def present_keys(td: TensorDict, keys: Iterable[NestedKey]) -> list[NestedKey]:
    r"""The subset of ``keys`` present in ``td``, for null-safe ``exclude`` / ``select``."""
    return [k for k in keys if k in td]


def _prune_empty_ancestors(td: TensorDict, key: NestedKey) -> None:
    """Delete the groups above ``key`` that are now empty, walking upward."""
    parent = key[:-1] if isinstance(key, tuple) else ()
    while parent:
        group = td.get(parent, None)
        if isinstance(group, TensorDict) and group.is_empty():
            td.del_(parent)
            parent = parent[:-1]
        else:
            break


def exclude_keys(td: TensorDict, keys: Iterable[NestedKey]) -> TensorDict:
    r"""``td.exclude(*keys)`` that tolerates missing keys and prunes emptied groups.

    Removing the last leaf of a group with plain ``exclude`` leaves an empty
    sub-TensorDict behind; this drops such groups so the result has no empty
    containers the caller did not ask for. Leaf tensors are shared with ``td``.
    """
    keys = present_keys(td, keys)
    out = td.exclude(*keys)
    for key in keys:
        _prune_empty_ancestors(out, key)
    return out


def _nested_pairs(keys: Sequence[NestedKey]) -> tuple[NestedKey, NestedKey] | None:
    """Return ``(outer, inner)`` if one key path is a strict prefix of another."""
    paths = [((k,) if isinstance(k, str) else k, k) for k in keys]
    for outer_path, outer in paths:
        for inner_path, inner in paths:
            if (
                len(inner_path) > len(outer_path)
                and inner_path[: len(outer_path)] == outer_path
            ):
                return outer, inner
    return None


def rename_keys(
    td: TensorDict,
    mapping: Mapping[NestedKey, NestedKey],
    *,
    strict: bool,
) -> TensorDict:
    r"""Return a copy of ``td`` with keys renamed, nested keys included.

    Both sides of ``mapping`` may be top-level or nested keys; a leaf can be
    moved between groups (``("raw", "p") -> "pressure"``) and a whole group
    can be renamed (``"raw" -> "solution"``). Every source is read from the
    original tree before any destination is written, so chains
    (``a -> b, b -> c``) and swaps (``a -> b, b -> a``) are safe. A mapping
    that names both a group and a key inside it (``"raw"`` and
    ``("raw", "p")``) is rejected: the group move carries its children with
    it, so the result would depend on which entry ran first. Rename the group
    and then its leaves in two steps instead. Leaf tensors are shared with
    ``td``; only the container tree is copied.

    Parameters
    ----------
    td : TensorDict
        Source container. Not modified.
    mapping : Mapping
        ``{old_key: new_key}`` with keys already normalized by
        :func:`as_nested_key`.
    strict : bool
        If True, raise ``KeyError`` when an ``old_key`` is missing. If False,
        missing sources are skipped.

    Raises
    ------
    KeyError
        In strict mode, when a source key is absent.
    ValueError
        When a destination key already exists and is not itself being
        renamed away, when two sources map to the same destination, or when
        one source or destination path lies inside another.
    """
    present = [old for old in mapping if old in td]
    if strict:
        missing = [key_to_str(old) for old in mapping if old not in td]
        if missing:
            raise KeyError(
                f"Keys not found in data: {missing}. "
                f"Available fields: {format_leaf_keys(td)}"
            )
    conflicts = [
        key_to_str(mapping[old])
        for old in present
        if mapping[old] in td and mapping[old] not in present
    ]
    if conflicts:
        raise ValueError(f"New key names conflict with existing keys: {conflicts}")
    destinations = [mapping[old] for old in present]
    duplicates = sorted(
        {key_to_str(d) for d in destinations if destinations.count(d) > 1}
    )
    if duplicates:
        raise ValueError(f"Several keys are renamed to the same name: {duplicates}")
    for label, keys in (("source", present), ("destination", destinations)):
        overlap = _nested_pairs(keys)
        if overlap:
            outer, inner = overlap
            raise ValueError(
                f"Rename {label} {key_to_str(inner)!r} lies inside {label} "
                f"{key_to_str(outer)!r}; a group and a key inside it cannot be "
                f"renamed in one mapping. Rename the group first, then its keys."
            )

    ### ``clone(recurse=False)`` copies the container tree (including nested
    ### sub-TensorDicts) but not the tensors, so renaming inside a sub-TD
    ### cannot leak back into ``td``. Detach every source before writing any
    ### destination so a chain ``a -> b, b -> c`` (or a swap) cannot
    ### overwrite a value that is still waiting to be moved.
    out = td.clone(recurse=False)
    moved = [(mapping[old], out.get(old)) for old in present]
    for old in present:
        out.del_(old)
    for new, value in moved:
        out.set(new, value)
    ### Hoisting the last leaf out of a group leaves an empty sub-TensorDict
    ### behind; drop such groups so the result has no empty containers the
    ### caller did not ask for.
    for old in present:
        _prune_empty_ancestors(out, old)
    return out


def as_nested_keys(keys: Iterable[str | Sequence[str]] | None) -> list[NestedKey]:
    r"""Vectorized :func:`as_nested_key`; ``None`` maps to an empty list."""
    return [as_nested_key(k) for k in keys] if keys is not None else []
