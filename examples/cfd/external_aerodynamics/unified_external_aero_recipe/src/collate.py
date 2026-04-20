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
    # High-lift airplane volume: SDF + normals from STL surface.
    "geotransolver_highlift_volume": {
        "geometry": "input/points",
        "local_embedding": ["input/points", "input/sdf", "input/sdf_normals"],
        "local_positions": "input/points",
        "global_embedding": "input/U_inf",
        "fields": [
            "output/pressure",
            "output/temperature",
            "output/density",
            "output/velocity",
        ],
    },
    # Automotive volume: SDF + normals from STL surface, incompressible fields.
    "geotransolver_automotive_volume": {
        "geometry": "input/points",
        "local_embedding": ["input/points", "input/sdf", "input/sdf_normals"],
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
    # Automotive volume (Transolver): SDF + normals from STL surface, fx = freestream velocity.
    "transolver_automotive_volume": {
        "embedding": ["input/points", "input/sdf", "input/sdf_normals"],
        "fx": "input/U_inf",
        "fields": ["output/velocity", "output/pressure", "output/nut"],
    },
    # Automotive surface (FLARE): same interface as Transolver (fx + embedding).
    "flare_automotive_surface": {
        "embedding": ["input/points", "input/normals"],
        "fx": "input/U_inf",
        "fields": ["output/pressure", "output/wss"],
    },
    # Automotive volume (FLARE): SDF + normals from STL surface.
    "flare_automotive_volume": {
        "embedding": ["input/points", "input/sdf", "input/sdf_normals"],
        "fx": "input/U_inf",
        "fields": ["output/velocity", "output/pressure", "output/nut"],
    },
    # -----------------------------------------------------------------------
    # DoMINO mappings
    # -----------------------------------------------------------------------
    # DoMINO.forward() takes a single ``data_dict`` argument instead of
    # keyword-per-tensor.  The ``_wrap_as`` key tells the collate layer to
    # pack every non-``fields`` tensor into a nested dict with that name so
    # that ``model(**batch)`` expands to ``model(data_dict={...})``.
    #
    # The datapipe keys below assume the dataset YAML's
    # RestructureTensorDict produces DoMINO-compatible names under input/.
    # -----------------------------------------------------------------------
    "domino_automotive_surface": {
        "_wrap_as": "data_dict",
        "geometry_coordinates": "input/geometry_coordinates",
        "surf_grid": "input/surf_grid",
        "sdf_surf_grid": "input/sdf_surf_grid",
        "global_params_values": "input/global_params_values",
        "global_params_reference": "input/global_params_reference",
        "pos_surface_center_of_mass": "input/pos_surface_center_of_mass",
        "surface_mesh_centers": "input/surface_mesh_centers",
        "surface_mesh_neighbors": "input/surface_mesh_neighbors",
        "surface_normals": "input/surface_normals",
        "surface_neighbors_normals": "input/surface_neighbors_normals",
        "surface_areas": "input/surface_areas",
        "surface_neighbors_areas": "input/surface_neighbors_areas",
        "surface_min_max": "input/surface_min_max",
        "fields": ["output/pressure", "output/wss"],
    },
    "domino_automotive_volume": {
        "_wrap_as": "data_dict",
        "geometry_coordinates": "input/geometry_coordinates",
        "grid": "input/grid",
        "surf_grid": "input/surf_grid",
        "sdf_grid": "input/sdf_grid",
        "sdf_surf_grid": "input/sdf_surf_grid",
        "sdf_nodes": "input/sdf_nodes",
        "global_params_values": "input/global_params_values",
        "global_params_reference": "input/global_params_reference",
        "pos_volume_closest": "input/pos_volume_closest",
        "pos_volume_center_of_mass": "input/pos_volume_center_of_mass",
        "volume_mesh_centers": "input/volume_mesh_centers",
        "volume_min_max": "input/volume_min_max",
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
    *,
    wrap_as: str | None = None,
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
        Keys starting with ``_`` are metadata and are skipped.
    wrap_as : str or None, optional
        When set, all non-``fields`` tensors are packed into a nested dict
        under this key.  Used for models whose ``forward()`` accepts a
        single dict argument (e.g. DoMINO's ``data_dict``).

    Returns
    -------
    dict[str, torch.Tensor | dict[str, torch.Tensor]]
        Batch dictionary ready for model consumption.
    """
    real_mapping = {k: v for k, v in mapping.items() if not k.startswith("_")}
    accumulators: dict[str, list[torch.Tensor]] = {key: [] for key in real_mapping}

    for data, _meta in samples:
        for model_key, spec in real_mapping.items():
            accumulators[model_key].append(_resolve_spec(data, spec))

    batch = {key: torch.stack(vals) for key, vals in accumulators.items()}

    if wrap_as is not None:
        fields = batch.pop("fields")
        return {wrap_as: batch, "fields": fields}

    return batch


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

    wrap_as = resolved.get("_wrap_as")

    def collate_fn(
        samples: list[tuple[TensorDict, dict]],
    ) -> dict[str, torch.Tensor]:
        return map_data_to_model(samples, resolved, wrap_as=wrap_as)

    return collate_fn
