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

from __future__ import annotations

import torch

from physicsnemo.nn.functional.derivatives.mesh_lsq_gradient._warp_impl import (
    mesh_lsq_gradient_warp,
)


def _validate_vector_field(points: torch.Tensor, vector_field: torch.Tensor) -> None:
    """Validate divergence-specific vector-field shape constraints."""
    if vector_field.ndim != 2:
        raise ValueError(
            "mesh_lsq_divergence: vector_field must have shape "
            f"(n_entities, dims), got {vector_field.shape=}"
        )
    if vector_field.shape != points.shape:
        raise ValueError(
            "mesh_lsq_divergence: vector_field shape must match points shape, "
            f"got {vector_field.shape} and {points.shape}"
        )


def mesh_lsq_divergence_warp(
    points: torch.Tensor,
    vector_field: torch.Tensor,
    neighbor_offsets: torch.Tensor,
    neighbor_indices: torch.Tensor,
    weight_power: float = 2.0,
    min_neighbors: int = 0,
    safe_epsilon: float | None = None,
) -> torch.Tensor:
    """Compute LSQ mesh divergence using the Warp LSQ-gradient backend."""
    _validate_vector_field(points, vector_field)
    jacobian = mesh_lsq_gradient_warp(
        points=points,
        values=vector_field,
        neighbor_offsets=neighbor_offsets,
        neighbor_indices=neighbor_indices,
        weight_power=weight_power,
        min_neighbors=min_neighbors,
        safe_epsilon=safe_epsilon,
    )
    return torch.diagonal(jacobian, dim1=1, dim2=2).sum(dim=-1)
