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
Forward-kwargs spec resolver.

Each model YAML supplies a `forward_kwargs:` block that maps the model's
`forward()` keyword arguments to declarative paths into a `DomainMesh` (or
`Mesh`). At collate time, this module turns that spec into the actual
`dict[str, Any]` ready to splat into `model.forward(**kwargs)`.

Spec value semantics:

- Strings are dotted paths into the source, walked via getattr-then-getitem
  (e.g., ``"interior.points"``, ``"boundaries.vehicle.cell_data.normals"``,
  ``"global_data.U_inf"``). The empty string ``""`` and the literal ``"."``
  resolve to the source itself.
- Numbers (int / float / bool) become 0-d float32 tensors.
- Lists trigger ``torch.cat`` along the last dim of the resolved entries
  (used for tensor-input models that want concatenated feature vectors).
- Dicts recurse and produce dict-valued kwargs (used for mesh-input models
  with dict-valued args like GLOBE's ``boundary_meshes`` and
  ``reference_lengths``).
- "Modifier" dicts -- dicts containing a ``"source"`` key plus one of the
  recognized modifier keys (currently only ``"expand_like"``) -- are
  deferred to a second resolution pass so they may reference other already-
  resolved kwargs. ``{source: <path>, expand_like: <other_kwarg>}``
  extracts ``source`` and expands the result along axis ``-2`` to match
  the resolved value of ``<other_kwarg>``; this is how a per-sample tensor
  (e.g. a freestream vector) gets broadcast across the per-cell axis of a
  per-element kwarg.

Targets are extracted separately by :func:`extract_targets` from the
DomainMesh's ``interior.point_data``, by convention.
"""

from __future__ import annotations

from typing import Any

import torch
from tensordict import TensorDict

from physicsnemo.mesh import DomainMesh, Mesh

from utils import FieldType


### ---------------------------------------------------------------------------
### Path walking
### ---------------------------------------------------------------------------


def walk_path(source: Any, path: str) -> Any:
    """Resolve a dotted path against `source` via getattr-then-getitem.

    Args:
        source: The object to walk into (typically a `DomainMesh`, `Mesh`,
            or `TensorDict`).
        path: Dotted path string. Empty string or ``"."`` returns `source`
            itself.

    Returns:
        The value at the end of the path.

    Raises:
        KeyError: If a path segment cannot be resolved either as an attribute
            or an item.
    """
    if path in ("", "."):
        return source

    obj = source
    for part in path.split("."):
        try:
            obj = getattr(obj, part)
        except AttributeError:
            ### Fall back to indexing. Different container types raise
            ### different errors for missing keys / unsupported indexing
            ### (KeyError on dict / TensorDict, TypeError on plain
            ### tensors, ValueError on tensorclass for non-indexable
            ### strings, IndexError on out-of-range numeric indices).
            ### Normalize all of them into a single KeyError so callers
            ### only have to handle one type.
            try:
                obj = obj[part]
            ### LookupError covers KeyError (dict / TensorDict) and
            ### IndexError (out-of-range numeric); TypeError covers plain
            ### tensors not supporting string indexing; ValueError covers
            ### tensorclass rejecting non-indexable strings.
            except (TypeError, ValueError, LookupError) as e:
                raise KeyError(
                    f"Cannot resolve segment {part!r} of {path!r} on "
                    f"{type(obj).__name__}: {e}"
                ) from e
    return obj


### ---------------------------------------------------------------------------
### Spec resolution
### ---------------------------------------------------------------------------


_MODIFIER_KEYS = frozenset({"expand_like"})


def _is_modifier_spec(spec: Any) -> bool:
    """Whether `spec` is a modifier dict (has 'source' + a recognized modifier)."""
    return (
        isinstance(spec, dict)
        and "source" in spec
        and any(k in _MODIFIER_KEYS for k in spec)
    )


def resolve_spec(spec: Any, source: Any) -> Any:
    """Resolve a single spec value against `source`.

    See module docstring for spec semantics. This function does NOT handle
    modifier specs (`{source: ..., expand_like: ...}`); modifiers are deferred
    by :func:`resolve_forward_kwargs` to a second pass that has access to
    other already-resolved kwargs.

    Args:
        spec: The YAML-decoded spec value (string, number, list, or dict).
        source: The DomainMesh / Mesh / TensorDict to resolve paths against.

    Returns:
        The resolved value -- a tensor, a Mesh, a TensorDict, a dict of
        resolved values, etc.

    Raises:
        ValueError: If `spec` is a modifier dict (which must be handled by
            :func:`resolve_forward_kwargs`).
    """
    if isinstance(spec, str):
        return walk_path(source, spec)

    ### `bool` is a subclass of `int`, so without this guard `True` /
    ### `False` would silently coerce to `tensor(1.0)` / `tensor(0.0)` --
    ### almost always a config bug (e.g. someone meant a numeric flag).
    ### Reject explicitly so the YAML author sees an actionable error
    ### rather than a downstream shape / dtype mismatch.
    if isinstance(spec, bool):
        raise TypeError(
            f"Boolean spec values are not supported in forward_kwargs "
            f"(got {spec!r}). Use 0 / 1 explicitly if you really meant a "
            f"numeric flag, or wire the bool through a different mechanism."
        )

    if isinstance(spec, (int, float)):
        return torch.tensor(float(spec), dtype=torch.float32)

    if isinstance(spec, list):
        ### List = concatenate resolved tensors along the last dim. Pad ndim
        ### up to the max so e.g. (N,) and (N, 3) become (N, 1) + (N, 3).
        resolved = [resolve_spec(item, source) for item in spec]
        if not all(isinstance(t, torch.Tensor) for t in resolved):
            raise TypeError(
                f"List specs must resolve to tensors, got "
                f"{[type(t).__name__ for t in resolved]!r}"
            )
        max_ndim = max(t.ndim for t in resolved)
        resolved = [t.unsqueeze(-1) if t.ndim < max_ndim else t for t in resolved]
        return torch.cat(resolved, dim=-1)

    if isinstance(spec, dict):
        if _is_modifier_spec(spec):
            raise ValueError(
                f"Modifier specs ({spec!r}) must be resolved via "
                f"resolve_forward_kwargs, not resolve_spec directly."
            )
        return {k: resolve_spec(v, source) for k, v in spec.items()}

    if spec is None:
        return None

    ### Pass through anything else unchanged (already-resolved tensors etc.).
    return spec


def _apply_modifier(
    mod_spec: dict[str, Any],
    source: DomainMesh | Mesh,
    resolved_kwargs: dict[str, Any],
) -> Any:
    """Apply a modifier spec (e.g., ``{source: ..., expand_like: ...}``).

    Resolves ``mod_spec["source"]`` and then applies any recognized modifier
    in turn. ``expand_like`` expands the resolved value's axis ``-2`` to
    match the resolved value of the referenced kwarg.
    """
    src_value = resolve_spec(mod_spec["source"], source)

    if "expand_like" in mod_spec:
        ref_key = mod_spec["expand_like"]
        if ref_key not in resolved_kwargs:
            raise KeyError(
                f"expand_like references {ref_key!r}, but no such kwarg "
                f"was resolved (available: {sorted(resolved_kwargs)!r})."
            )
        ref = resolved_kwargs[ref_key]
        if not isinstance(ref, torch.Tensor):
            raise TypeError(
                f"expand_like reference {ref_key!r} must resolve to a tensor, "
                f"got {type(ref).__name__}."
            )
        if not isinstance(src_value, torch.Tensor):
            raise TypeError(
                f"expand_like source must resolve to a tensor, got "
                f"{type(src_value).__name__}."
            )
        ### `expand_like` broadcasts source along the reference's per-element
        ### (axis -2) dimension, so the reference must have one. A 0-D or 1-D
        ### reference (e.g. a per-element scalar field at axis -1) is almost
        ### always a config bug; fail fast with a useful message instead of
        ### the bare IndexError that ref.shape[-2] would produce.
        if ref.ndim < 2:
            raise ValueError(
                f"expand_like reference {ref_key!r} must be at least 2-D so "
                f"axis -2 exists; got shape {tuple(ref.shape)} (ndim="
                f"{ref.ndim}). For a 1-D per-element reference, drop the "
                f"expand_like modifier and use a plain path spec instead."
            )
        ### Pad source to ref.ndim (which is >= 2) so axis -2 exists.
        while src_value.ndim < ref.ndim:
            src_value = src_value.unsqueeze(0)
        target_shape = list(src_value.shape)
        target_shape[-2] = ref.shape[-2]
        src_value = src_value.expand(*target_shape)

    return src_value


def resolve_forward_kwargs(
    spec_dict: dict[str, Any],
    source: DomainMesh | Mesh,
) -> dict[str, Any]:
    """Resolve a top-level forward_kwargs spec into actual model.forward kwargs.

    Two-pass strategy:

    1. Resolve every non-modifier spec first.
    2. Resolve modifier specs (e.g. ``expand_like``) using the first-pass
       results so users can reference any other kwarg without worrying
       about declaration order.

    Args:
        spec_dict: Top-level spec dict from the model YAML.
        source: The DomainMesh / Mesh to resolve paths against.

    Returns:
        Dict ready for ``model(**kwargs)``.
    """
    resolved: dict[str, Any] = {}

    ### Pass 1 -- everything that doesn't depend on other kwargs.
    deferred: dict[str, dict[str, Any]] = {}
    for key, value in spec_dict.items():
        if _is_modifier_spec(value):
            deferred[key] = value
        else:
            resolved[key] = resolve_spec(value, source)

    ### Pass 2 -- modifier specs that may reference resolved kwargs.
    for key, mod_spec in deferred.items():
        resolved[key] = _apply_modifier(mod_spec, source, resolved)

    return resolved


### ---------------------------------------------------------------------------
### Target extraction
### ---------------------------------------------------------------------------


def extract_targets(
    domain: DomainMesh | Mesh,
    target_config: dict[str, FieldType],
) -> TensorDict:
    """Pull target tensors from a DomainMesh's ``interior.point_data`` by name.

    Targets always live at ``interior.point_data.<name>`` by the recipe's
    DomainMesh contract -- this convention is what lets the dataset YAMLs
    stay model-agnostic.

    Args:
        domain: The dataset's output. Usually a `DomainMesh`; for backward
            compatibility this also accepts a bare `Mesh` (uses its
            ``point_data`` directly).
        target_config: ``{name: type}`` mapping. Only the keys are used;
            types are validated upstream by `LossCalculator` /
            `MetricCalculator`.

    Returns:
        TensorDict with exactly the keys in ``target_config`` and the same
        ``batch_size`` / device as the source ``point_data``. Iteration order
        matches the order of ``target_config``.

    Raises:
        KeyError: If a name in ``target_config`` is not present in
            ``interior.point_data``.
    """
    if isinstance(domain, DomainMesh):
        source_td: TensorDict = domain.interior.point_data
        location = "interior.point_data"
    elif isinstance(domain, Mesh):
        source_td = domain.point_data
        location = "point_data"
    else:
        raise TypeError(f"Expected DomainMesh or Mesh, got {type(domain).__name__}.")

    available = set(source_td.keys())
    missing = [name for name in target_config if name not in available]
    if missing:
        raise KeyError(
            f"Target fields {missing!r} not found in {location} "
            f"(available: {sorted(available)!r})."
        )
    return source_td.select(*target_config)
