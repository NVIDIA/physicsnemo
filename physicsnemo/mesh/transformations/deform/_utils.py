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

from dataclasses import replace
from typing import TYPE_CHECKING

import torch
from jaxtyping import Bool, Float
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


def _resolve_domain_point_weights(
    components: list[tuple[str, "Mesh"]],
    point_weights: str | tuple[str, ...] | None,
    reference: Float[torch.Tensor, "..."],
    reference_name: str,
) -> list[
    Bool[torch.Tensor, " n_component_points"]
    | Float[torch.Tensor, " n_component_points"]
]:
    """Validate components against a reference tensor and resolve point weights.

    Every component's points must share the reference dtype and device. When a
    ``point_weights`` key is supplied, it is resolved independently on each
    component and validated to one common bool or floating dtype.
    """
    resolved_point_weights: list[torch.Tensor] = []
    for label, component in components:
        if component.points.device != reference.device:
            raise ValueError(
                f"{label} and {reference_name} must be on the same device, got "
                f"{component.points.device} and {reference.device}"
            )
        if component.points.dtype != reference.dtype:
            raise TypeError(
                f"{label} and {reference_name} must have the same dtype, got "
                f"{component.points.dtype} and {reference.dtype}"
            )
        if point_weights is not None:
            component_point_weights = _resolve_point_field(
                component,
                point_weights,
                argument_name="point_weights",
                owner_label=label,
            )
            if tuple(component_point_weights.shape) != (component.n_points,):
                raise ValueError(
                    f"point_weights field {point_weights!r} in "
                    f"{label}.point_data must have "
                    f"shape ({component.n_points},), got "
                    f"{tuple(component_point_weights.shape)}"
                )
            if component_point_weights.device != component.points.device:
                raise ValueError(
                    f"point_weights field {point_weights!r} in "
                    f"{label}.point_data and points must be on the same "
                    f"device, got {component_point_weights.device} and "
                    f"{component.points.device}"
                )
            if (
                component_point_weights.dtype != torch.bool
                and not torch.is_floating_point(component_point_weights)
            ):
                raise TypeError(
                    f"point_weights field {point_weights!r} in "
                    f"{label}.point_data must have bool or floating-point "
                    f"dtype, got {component_point_weights.dtype}"
                )
            if (
                component_point_weights.dtype != torch.bool
                and component_point_weights.dtype != component.points.dtype
            ):
                raise TypeError(
                    f"point_weights field {point_weights!r} in "
                    f"{label}.point_data and points must have the same dtype "
                    "for floating weights, got "
                    f"{component_point_weights.dtype} and {component.points.dtype}"
                )
            if (
                resolved_point_weights
                and component_point_weights.dtype != resolved_point_weights[0].dtype
            ):
                raise TypeError(
                    f"point_weights field {point_weights!r} must have one "
                    f"common dtype across all components. {label}.point_data "
                    f"has {component_point_weights.dtype}, expected "
                    f"{resolved_point_weights[0].dtype}"
                )
            resolved_point_weights.append(component_point_weights)
    return resolved_point_weights


def _mesh_with_deformed_points(
    mesh: "Mesh",
    points: Float[torch.Tensor, "n_points n_spatial_dims"],
) -> "Mesh":
    """Construct a geometry-invalidated mesh while retaining topology caches."""
    device = points.device
    cache = TensorDict(
        {
            "cell": TensorDict({}, batch_size=[mesh.n_cells], device=device),
            "point": TensorDict({}, batch_size=[mesh.n_points], device=device),
            "topology": mesh._cache.get("topology", TensorDict({}, device=device)),
        },
        device=device,
    )
    return replace(mesh, points=points, _cache=cache)
