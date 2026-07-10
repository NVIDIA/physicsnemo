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

"""Shared utilities for mesh deformation operations."""

from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict

if TYPE_CHECKING:
    from physicsnemo.mesh.mesh import Mesh


def _resolve_point_field(
    mesh: "Mesh",
    value: str | tuple[str, ...] | torch.Tensor,
    *,
    argument_name: str,
    owner_label: str | None = None,
) -> torch.Tensor:
    """Resolve a raw tensor or nested ``point_data`` key."""
    if isinstance(value, torch.Tensor):
        return value
    if not isinstance(value, (str, tuple)):
        raise TypeError(
            f"{argument_name} must be a tensor or point_data key/path, got "
            f"{type(value).__name__}"
        )
    point_data_label = (
        "point_data" if owner_label is None else f"{owner_label}.point_data"
    )
    try:
        resolved = mesh.point_data[value]
    except (AttributeError, KeyError, ValueError):
        available = list(mesh.point_data.keys(include_nested=True, leaves_only=True))
        raise KeyError(
            f"{argument_name} field {value!r} not found in {point_data_label}. "
            f"Available keys: {available}"
        ) from None
    if not isinstance(resolved, torch.Tensor):
        raise TypeError(
            f"{argument_name} field {value!r} in {point_data_label} must "
            "resolve to a torch.Tensor"
        )
    return resolved


def _mesh_with_deformed_points(mesh: "Mesh", points: torch.Tensor) -> "Mesh":
    """Construct a geometry-invalidated mesh while retaining topology caches."""
    from physicsnemo.mesh.mesh import Mesh

    device = points.device
    cache = TensorDict(
        {
            "cell": TensorDict({}, batch_size=[mesh.n_cells], device=device),
            "point": TensorDict({}, batch_size=[mesh.n_points], device=device),
            "topology": mesh._cache.get("topology", TensorDict({}, device=device)),
        },
        device=device,
    )
    return Mesh(
        points=points,
        cells=mesh.cells,
        point_data=mesh.point_data,
        cell_data=mesh.cell_data,
        global_data=mesh.global_data,
        _cache=cache,
    )
