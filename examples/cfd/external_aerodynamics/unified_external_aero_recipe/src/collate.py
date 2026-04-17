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

"""Data-to-model mapping for converting datapipe output to model batch format.

The datapipe produces ``(TensorDict, metadata_dict)`` tuples.  A *mapping
specification* defines how TensorDict fields are extracted, optionally
concatenated, and assembled into the ``dict[str, Tensor]`` batch expected
by ``model.forward()``.

Mapping specs are plain dictionaries registered in :data:`MODEL_MAPPINGS`::

    MODEL_MAPPINGS = {
        "geotransolver_automotive_surface": {
            "geometry":         "input/points",
            "local_embedding":  ["input/points", "input/normals"],
            "local_positions":  "input/points",
            "global_embedding": "input/U_inf",
            "fields":           ["output/pressure", "output/wss"],
        },
    }

Each value is either:

* A **string** path (``"group/key"``) — extract that tensor directly.
* A **list** of paths — extract each tensor, then concatenate along the
  last dimension.

The ``"fields"`` key is treated as the prediction target by the training
loop (popped from the batch before ``model(**batch)``).
"""

from __future__ import annotations

from typing import Callable

import torch
from tensordict import TensorDict

MappingSpec = dict[str, str | list[str]]


# ---------------------------------------------------------------------------
# Mapping registry — add new model mappings here
# The idea here is to build a dictionary to map datapipe outputs
# to model inputs.  We can make it relatively targeted between
# model and application, and you can extend it to new models / domains.
# ---------------------------------------------------------------------------

MODEL_MAPPINGS: dict[str, MappingSpec] = {
    # Automotive surface: concatenates points+normals into local_embedding
    # (breaks equivariance by design — GeoTransolver learns to disentangle).
    "geotransolver_automotive_surface": {
        "geometry": "input/points",
        "local_embedding": ["input/points", "input/normals"],
        "local_positions": "input/points",
        "global_embedding": "input/U_inf",
        "fields": ["output/pressure", "output/wss"],
    },
    # High-lift airplane surface: compressible fields (P, T, rho, U, tau_wall).
    "geotransolver_highlift_surface": {
        "geometry": "input/points",
        "local_embedding": ["input/points", "input/normals"],
        "local_positions": "input/points",
        "global_embedding": "input/U_inf",
        "fields": [
            "output/pressure",
            "output/temperature",
            "output/density",
            "output/velocity",
            "output/tau_wall",
        ],
    },
    # High-lift airplane volume: point cloud without normals.
    "geotransolver_highlift_volume": {
        "geometry": "input/points",
        "local_embedding": "input/points",
        "local_positions": "input/points",
        "global_embedding": "input/U_inf",
        "fields": [
            "output/pressure",
            "output/temperature",
            "output/density",
            "output/velocity",
        ],
    },
    # Automotive volume: point cloud without normals, incompressible fields.
    "geotransolver_automotive_volume": {
        "geometry": "input/points",
        "local_embedding": "input/points",
        "local_positions": "input/points",
        "global_embedding": "input/U_inf",
        "fields": ["output/velocity", "output/pressure", "output/nut"],
    },
    # Automotive surface (Transolver): embedding = points+normals, fx = freestream velocity.
    # fx is broadcast from (B,1,3) to (B,N,3) via broadcast_global in train.py.
    "transolver_automotive_surface": {
        "embedding": ["input/points", "input/normals"],
        "fx": "input/U_inf",
        "fields": ["output/pressure", "output/wss"],
    },
    # Automotive volume (Transolver): embedding = points only, fx = freestream velocity.
    "transolver_automotive_volume": {
        "embedding": "input/points",
        "fx": "input/U_inf",
        "fields": ["output/velocity", "output/pressure", "output/nut"],
    },
}


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


def _extract(td: TensorDict, path: str) -> torch.Tensor:
    """Extract a tensor from a TensorDict using a ``/``-separated path."""
    keys = path.split("/")
    result = td
    for key in keys:
        result = result[key]
    return result


def _resolve_spec(td: TensorDict, spec: str | list[str]) -> torch.Tensor:
    """Resolve one mapping spec to a single tensor.

    - String spec: extract and ensure at least 2-D (adds a leading token dim).
    - List spec: extract each path, align ndim, and concatenate along last dim.
    """
    if isinstance(spec, str):
        tensor = _extract(td, spec)
        # Scalars / 1-D vectors (e.g. U_inf as (3,)) need a leading
        # token dimension so they stack to (B, 1, D).
        while tensor.ndim < 2:
            tensor = tensor.unsqueeze(0)
        return tensor

    tensors = [_extract(td, s) for s in spec]
    # Align ndim before concatenation (e.g. pressure (N,) with
    # wss (N, 3) — unsqueeze pressure to (N, 1)).
    max_ndim = max(t.ndim for t in tensors)
    tensors = [t.unsqueeze(-1) if t.ndim < max_ndim else t for t in tensors]
    return torch.cat(tensors, dim=-1)


def map_data_to_model(
    samples: list[tuple[TensorDict, dict]],
    mapping: MappingSpec,
) -> dict[str, torch.Tensor]:
    """Stack datapipe samples into a model-ready batch.

    Each sample is a ``(data, metadata)`` tuple where ``data`` is a TensorDict
    with groups produced by
    :class:`~physicsnemo.datapipes.transforms.mesh.RestructureTensorDict`.

    Parameters
    ----------
    samples : list[tuple[TensorDict, dict]]
        List of ``(data, metadata)`` pairs from the datapipe.
    mapping : dict[str, str | list[str]]
        Mapping from model batch keys to datapipe TensorDict paths.
        A string value extracts that field directly; a list of strings
        extracts each field and concatenates them along the last dimension.

    Returns
    -------
    dict[str, torch.Tensor]
        Batch dictionary ready for model consumption.
    """
    accumulators: dict[str, list[torch.Tensor]] = {key: [] for key in mapping}

    for data, _meta in samples:
        for model_key, spec in mapping.items():
            accumulators[model_key].append(_resolve_spec(data, spec))

    return {key: torch.stack(vals) for key, vals in accumulators.items()}


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_collate_fn(
    mapping: str | MappingSpec,
) -> Callable[[list[tuple[TensorDict, dict]]], dict[str, torch.Tensor]]:
    """Return a collate function that applies a data-to-model mapping.

    Parameters
    ----------
    mapping : str or dict
        Either a key in :data:`MODEL_MAPPINGS` or an explicit mapping dict.

    Returns
    -------
    Callable
        A function suitable for ``DataLoader(collate_fn=...)``.

    Raises
    ------
    ValueError
        If *mapping* is a string not found in :data:`MODEL_MAPPINGS`.
    """
    if isinstance(mapping, str):
        if mapping not in MODEL_MAPPINGS:
            raise ValueError(
                f"Unknown mapping {mapping!r}. Available: {list(MODEL_MAPPINGS.keys())}"
            )
        resolved = MODEL_MAPPINGS[mapping]
    else:
        resolved = mapping

    def collate_fn(
        samples: list[tuple[TensorDict, dict]],
    ) -> dict[str, torch.Tensor]:
        return map_data_to_model(samples, resolved)

    return collate_fn
