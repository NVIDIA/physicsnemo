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

import functools
import itertools
from collections.abc import Sequence

import torch

from physicsnemo.utils._small_linalg import (
    _lstsq_solver,
    _segment_lstsq_solver,
    _uses_normal_equations,
    batched_lstsq,
)

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
    return _mesh_lsq_gradients_torch(
        points=points,
        values=[values],
        neighbor_offsets=neighbor_offsets,
        neighbor_indices=neighbor_indices,
        weight_power=weight_power,
        min_neighbors=min_neighbors,
        safe_epsilon=safe_epsilon,
    )[0]


def _mesh_lsq_gradients_torch(
    points: torch.Tensor,
    values: Sequence[torch.Tensor],
    neighbor_offsets: torch.Tensor,
    neighbor_indices: torch.Tensor,
    weight_power: float = 2.0,
    min_neighbors: int = 0,
    safe_epsilon: float | None = None,
    *,
    grouped: bool = False,
) -> list[torch.Tensor]:
    """Weighted LSQ mesh gradients of several fields of one dtype.

    Each gradient is that of :func:`mesh_lsq_gradient_torch` for its field, bit
    for bit. The fields share the work that does not depend on them: the
    weighted systems and their factorization. Each field is then solved on its
    own, so the peak memory is that of one field.

    Where the normal equations solve the systems (float32 and 16-bit, see
    :func:`~physicsnemo.utils._small_linalg.batched_lstsq`), every entity's
    system is ``n_dims x n_dims``: they are assembled and solved at once, by
    :func:`~physicsnemo.utils._small_linalg._segment_lstsq_solver`. Otherwise,
    or with ``grouped``, the entities are solved in groups of one neighbor
    count each.
    """
    ### Validate inputs before building LSQ systems.
    for field in values:
        validate_inputs(
            points=points,
            values=field,
            neighbor_offsets=neighbor_offsets,
            neighbor_indices=neighbor_indices,
            min_neighbors=min_neighbors,
        )
    dtypes = {field.dtype for field in values}
    if len(dtypes) > 1:
        raise ValueError(f"values must share one dtype, got {dtypes=}")
    if not values:
        return []
    (dtype,) = dtypes

    ### Normalize dtypes/layout for stable downstream linear algebra.
    points = points.contiguous()
    neighbor_offsets = neighbor_offsets.to(
        dtype=torch.int64, device=points.device
    ).contiguous()
    neighbor_indices = neighbor_indices.to(
        dtype=torch.int64, device=points.device
    ).contiguous()

    n_entities = points.shape[0]
    n_dims = points.shape[1]
    counts = neighbor_offsets[1:] - neighbor_offsets[:-1]

    ### Flatten component dimensions so scalar and tensor fields share one solve path.
    # Several fields: each group gathers its rows from the field as it is and
    # flattens those, so no field is copied whole (they would all stay alive).
    n_components = [field.shape[1:].numel() for field in values]
    several = len(values) > 1
    values_flat = (
        list(values)
        if several
        else [values[0].contiguous().reshape(n_entities, n_components[0])]
    )

    def zero_gradients(i: int) -> torch.Tensor:
        """Zero gradients of field ``i``, flattened."""
        shape = (n_entities, n_dims, n_components[i])
        return torch.zeros(shape, dtype=dtype, device=values[i].device)

    # Several fields: each field's gradients are created before its first
    # solve, as one field at a time would
    gradients_flat = [None if several else zero_gradients(0) for _ in values_flat]

    points_cast = points.to(dtype=dtype)
    dist_eps = resolve_safe_epsilon(safe_epsilon=safe_epsilon, dtype=points_cast.dtype)

    if not grouped and _uses_normal_equations(dtype, n_dims):
        ### Assemble every entity's system at once, from one weighted row per edge
        # The rows are those of the groups below, by the same operations.
        n_edges = neighbor_indices.shape[0]
        sources = torch.repeat_interleave(counts, output_size=n_edges)
        relative = points_cast[neighbor_indices]
        relative -= points_cast[sources]
        dist2 = (relative * relative).sum(dim=-1).clamp_min(dist_eps)
        sqrt_w = dist2.pow(-0.25 * weight_power).unsqueeze(-1)
        del dist2
        solve = _segment_lstsq_solver(sqrt_w * relative, neighbor_offsets)
        del relative
        # Entities with too few neighbors are not solved: zero gradients
        unsolved = counts < min_neighbors if min_neighbors > 1 else None

        gradients = []
        for field, n_field_components in zip(values, n_components):
            delta_values = field[neighbor_indices].reshape(n_edges, n_field_components)
            delta_values -= field[sources].reshape(n_edges, n_field_components)
            b_weighted = sqrt_w * delta_values
            del delta_values
            field_gradients = solve(b_weighted)
            del b_weighted
            if unsolved is not None:
                field_gradients = field_gradients.masked_fill(
                    unsolved[:, None, None], 0.0
                )
            gradients.append(
                field_gradients.reshape(n_entities, n_dims, *field.shape[1:])
            )
            del field_gradients
        return gradients

    ### Process one dense batch per neighbor-count group (mesh-module strategy).
    # A stable sort lists each group's entities contiguously, in ascending index
    # order; one transfer brings every group's count and size to the host.
    order = torch.argsort(counts, stable=True)
    unique_counts, group_sizes = torch.unique_consecutive(
        counts[order], return_counts=True
    )
    group_counts, group_sizes = torch.stack((unique_counts, group_sizes)).tolist()
    group_starts = itertools.accumulate(group_sizes, initial=0)
    groups = list(zip(group_counts, group_starts, group_sizes))
    if several:
        # The group of the largest systems first: the peak memory then falls
        # where only the gradients of the fields solved so far exist
        groups.sort(key=lambda group: group[0] * group[2], reverse=True)
    for n_neighbors, group_start, group_size in groups:
        group_end = group_start + group_size
        if n_neighbors < min_neighbors or n_neighbors == 0:
            continue

        entity_indices = order[group_start:group_end]

        offsets_group = neighbor_offsets[entity_indices]
        col_range = torch.arange(n_neighbors, device=points.device, dtype=torch.int64)
        flat_indices = offsets_group.unsqueeze(1) + col_range.unsqueeze(0)
        neighbors = neighbor_indices[flat_indices].to(torch.long)

        center_points = points_cast[entity_indices]
        relative = points_cast[neighbors] - center_points.unsqueeze(1)

        dist2 = (relative * relative).sum(dim=-1).clamp_min(dist_eps)
        sqrt_w = dist2.pow(-0.25 * weight_power).unsqueeze(-1)

        A_weighted = sqrt_w * relative
        del flat_indices, center_points, relative, dist2

        ### Factorize the weighted system once, for all fields
        if not several:
            solve = functools.partial(batched_lstsq, A_weighted)
        else:
            # The neighbors are gathered again for each field: kept alive, they
            # would raise the peak memory above that of one field.
            del neighbors
            solve = _lstsq_solver(A_weighted)
        del A_weighted  # kept by the solver if it needs it

        for i, field_flat in enumerate(values_flat):
            if several:
                neighbors = neighbor_indices[
                    offsets_group.unsqueeze(1) + col_range.unsqueeze(0)
                ]
            values_center = field_flat[entity_indices]
            neighbor_values = field_flat[neighbors]
            if several:  # gathered from the field as it is: flatten them
                values_center = values_center.reshape(group_size, -1)
                neighbor_values = neighbor_values.reshape(group_size, n_neighbors, -1)
            delta_values = neighbor_values - values_center.unsqueeze(1)
            del neighbor_values
            b_weighted = sqrt_w * delta_values
            # Only the weighted system is needed from here on
            del neighbors, values_center, delta_values
            if i == len(values_flat) - 1:
                del sqrt_w
            if gradients_flat[i] is None:
                gradients_flat[i] = zero_gradients(i)
            gradients_flat[i][entity_indices] = solve(b_weighted)
            del b_weighted
        del solve  # before the next group's

    ### Restore gradient output shape (zero for fields never solved).
    return [
        (zero_gradients(i) if gradients is None else gradients).reshape(
            n_entities, n_dims, *field.shape[1:]
        )
        for i, (gradients, field) in enumerate(zip(gradients_flat, values))
    ]
