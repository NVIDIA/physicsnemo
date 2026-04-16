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

from .utils import resolve_safe_epsilon, validate_inputs


def mesh_lsq_gradient_torch(
    points: torch.Tensor,
    values: torch.Tensor,
    neighbor_offsets: torch.Tensor,
    neighbor_indices: torch.Tensor,
    weight_power: float = 2.0,
    min_neighbors: int = 0,
    safe_epsilon: float | None = None,
) -> torch.Tensor:
    """Compute weighted LSQ mesh gradients with PyTorch tensor ops."""
    ### Validate inputs before building LSQ systems.
    validate_inputs(
        points=points,
        values=values,
        neighbor_offsets=neighbor_offsets,
        neighbor_indices=neighbor_indices,
        min_neighbors=min_neighbors,
    )

    ### Normalize dtypes/layout for stable downstream linear algebra.
    points = points.contiguous()
    values = values.contiguous()
    neighbor_offsets = neighbor_offsets.to(
        dtype=torch.int64, device=points.device
    ).contiguous()
    neighbor_indices = neighbor_indices.to(
        dtype=torch.int64, device=points.device
    ).contiguous()

    n_entities = points.shape[0]
    n_dims = points.shape[1]
    value_shape = values.shape[1:]
    counts = neighbor_offsets[1:] - neighbor_offsets[:-1]

    ### Flatten component dimensions so scalar and tensor fields share one solve path.
    values_flat = values.reshape(n_entities, -1)
    n_components = values_flat.shape[1]
    gradients_flat = torch.zeros(
        (n_entities, n_dims, n_components),
        dtype=values.dtype,
        device=values.device,
    )

    points_cast = points.to(dtype=values.dtype)
    dist_eps = resolve_safe_epsilon(safe_epsilon=safe_epsilon, dtype=points_cast.dtype)

    ### Process one dense batch per neighbor-count group (mesh-module strategy).
    unique_counts = torch.unique(counts)
    for count_tensor in unique_counts:
        n_neighbors = int(count_tensor.item())
        if n_neighbors < min_neighbors or n_neighbors == 0:
            continue

        entity_indices = torch.where(counts == count_tensor)[0]
        if entity_indices.numel() == 0:
            continue

        offsets_group = neighbor_offsets[entity_indices]
        col_range = torch.arange(n_neighbors, device=points.device, dtype=torch.int64)
        flat_indices = offsets_group.unsqueeze(1) + col_range.unsqueeze(0)
        neighbors = neighbor_indices[flat_indices].to(torch.long)

        center_points = points_cast[entity_indices]
        relative = points_cast[neighbors] - center_points.unsqueeze(1)

        values_center = values_flat[entity_indices]
        delta_values = values_flat[neighbors] - values_center.unsqueeze(1)

        dist2 = (relative * relative).sum(dim=-1).clamp_min(dist_eps)
        sqrt_w = dist2.pow(-0.25 * weight_power).unsqueeze(-1)

        A_weighted = sqrt_w * relative
        b_weighted = sqrt_w * delta_values

        solution = torch.linalg.lstsq(
            A_weighted,
            b_weighted,
            rcond=None,
        ).solution
        gradients_flat[entity_indices] = solution

    ### Restore gradient output shape.
    gradients = gradients_flat.reshape(n_entities, n_dims, *value_shape)

    return gradients
