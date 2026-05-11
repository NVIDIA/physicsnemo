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

"""Collate function for the DomainMesh-native unified external aero recipe.

The collate turns a single dataset sample (a `DomainMesh`) into a dict with
exactly two keys:

- ``"forward_kwargs"`` : ready to splat into ``model.forward(**...)``.
- ``"targets"``        : `TensorDict` of prediction targets extracted from
                          ``interior.point_data``. ``batch_size`` is
                          ``[N]`` in mesh-input mode and ``[1, N]`` in
                          tensor-input mode (the leading 1 comes from a
                          ``targets.unsqueeze(0)`` performed here).

Two `input_type`s are supported:

- ``"mesh"``    -- mesh-native model (e.g. GLOBE). Forward kwargs are
                   passed through as resolved by `forward_kwargs.py`:
                   tensors stay shape ``(N, ...)``, Mesh / DomainMesh
                   objects pass through, scalar literals stay 0-d.
- ``"tensors"`` -- transformer / point-cloud model (GeoTransolver,
                   Transolver, FLARE, DoMINO, ...). Every tensor in
                   ``forward_kwargs`` is padded to ``ndim >= 2`` and then
                   prepended with a batch dim of 1; targets get a single
                   ``unsqueeze(0)`` (TensorDict auto-grows every leaf).
                   Token-style features ``(D,)`` become ``(1, 1, D)``;
                   per-element features ``(N, C)`` become ``(1, N, C)``.

`batch_size > 1` is not implemented anywhere in the recipe (it raises a
`NotImplementedError` upstream in `train.py`); this collate enforces
``len(samples) == 1`` defensively.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from jaxtyping import Float

from forward_kwargs import extract_targets, resolve_forward_kwargs
from output_normalize import IOType
from utils import FieldType


### ---------------------------------------------------------------------------
### Batch-dim helpers
### ---------------------------------------------------------------------------


def _add_batch_dim_token(t: torch.Tensor) -> Float[torch.Tensor, "1 n c"]:
    """Pad a tensor up to ``ndim >= 2`` and prepend a batch dim of 1.

    Used for forward_kwargs values so token-style features (`(D,)` global
    feature vectors, `()` scalar literals) become 3-D ``(1, 1, D)`` shapes
    that line up with per-element features ``(1, N, C)`` after batch wrap.

    - 0-d ``()``     -> ``(1, 1)``  -> ``(1, 1, 1)`` (scalar literal)
    - 1-d ``(D,)``   -> ``(1, D)``  -> ``(1, 1, D)`` (token feature)
    - 2-d ``(N, C)`` -> ``(N, C)``  -> ``(1, N, C)`` (per-element feature)
    """
    while t.ndim < 2:
        t = t.unsqueeze(0)
    return t.unsqueeze(0)


def _add_batch_dim_recursive(value: Any, *, leaf_fn) -> Any:
    """Apply *leaf_fn* to every tensor in a (possibly nested) value.

    Non-tensor values are passed through unchanged. Dicts and lists / tuples
    are walked recursively.
    """
    if isinstance(value, torch.Tensor):
        return leaf_fn(value)
    if isinstance(value, dict):
        return {
            k: _add_batch_dim_recursive(v, leaf_fn=leaf_fn) for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return type(value)(_add_batch_dim_recursive(v, leaf_fn=leaf_fn) for v in value)
    return value


### ---------------------------------------------------------------------------
### Factory
### ---------------------------------------------------------------------------


def build_collate_fn(
    input_type: IOType,
    forward_kwargs_spec: dict[str, Any],
    target_config: dict[str, FieldType],
) -> Callable[[list[tuple[Any, Any]]], dict[str, Any]]:
    """Build a collate function for a given model contract.

    Args:
        input_type: One of ``"mesh"`` or ``"tensors"``. Controls whether
            tensor values are batch-wrapped on the way out.
        forward_kwargs_spec: The model YAML's ``forward_kwargs:`` block,
            mapping model.forward kwarg names to declarative spec values
            (paths, lists, nested dicts, modifiers; see
            :mod:`forward_kwargs`).
        target_config: ``{name: scalar|vector}`` mapping. Only the keys
            are used here -- types are validated downstream by the loss
            and metric calculators.

    Returns:
        A collate function suitable for ``DataLoader(collate_fn=...)``.
        It returns a dict with keys ``"forward_kwargs"`` and ``"targets"``.

    Raises:
        ValueError: If ``input_type`` is not ``"mesh"`` or ``"tensors"``.
    """
    if input_type not in ("mesh", "tensors"):
        raise ValueError(f"input_type must be 'mesh' or 'tensors', got {input_type!r}")

    add_batch_dim = input_type == "tensors"

    def collate_fn(samples: list[tuple[Any, Any]]) -> dict[str, Any]:
        ### Single-sample contract; train.py raises a clearer error upstream.
        if len(samples) != 1:
            raise NotImplementedError(
                f"This recipe requires exactly 1 sample per batch, got "
                f"len(samples)={len(samples)}. Every model in the recipe "
                f"assumes B=1; the YAML batch_size field is reserved for "
                f"future use."
            )
        domain, _metadata = samples[0]

        forward_kwargs = resolve_forward_kwargs(forward_kwargs_spec, domain)
        targets = extract_targets(domain, target_config)

        if add_batch_dim:
            ### forward_kwargs values get padded up to ndim>=2 first (so 1-D
            ### token features like `U_inf (3,)` become `(1, 1, 3)` tokens
            ### compatible with `(1, N, C)` per-element features).
            forward_kwargs = _add_batch_dim_recursive(
                forward_kwargs, leaf_fn=_add_batch_dim_token
            )
            ### TensorDict.unsqueeze grows the batch_size and every leaf in
            ### lock-step: batch_size [N] -> [1, N]; pressure (N,) -> (1, N);
            ### wss (N, 3) -> (1, N, 3). Per-element scalars stay (1, N) so
            ### they line up with the model's (1, N) scalar output.
            targets = targets.unsqueeze(0)

        return {"forward_kwargs": forward_kwargs, "targets": targets}

    return collate_fn
