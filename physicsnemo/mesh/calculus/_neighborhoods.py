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

"""Batched neighborhood iteration for ragged adjacency structures.

Provides a generator that groups mesh entities by neighbor count and yields
dense batches of neighborhood data, enabling efficient vectorized processing
without Python-level loops over individual entities.

Used by the LSQ gradient solvers (standard and intrinsic) and PCA tangent
space estimation to avoid duplicating the grouping + extraction boilerplate.
"""

from collections.abc import Iterator
from typing import TYPE_CHECKING, NamedTuple

import torch

if TYPE_CHECKING:
    from physicsnemo.mesh.neighbors._adjacency import Adjacency


class NeighborhoodBatch(NamedTuple):
    """A batch of entities that share the same neighbor count.

    All tensors in a batch have a leading dimension of ``n_group`` (the number
    of entities in this batch) and a second dimension of ``n_neighbors``.

    Parameters
    ----------
    entity_indices : torch.Tensor
        Global indices of the entities in this batch, shape ``(n_group,)``.
    neighbor_indices : torch.Tensor
        Global indices of each entity's neighbors, shape ``(n_group, n_neighbors)``.
    relative_positions : torch.Tensor
        Vectors from each entity to its neighbors,
        shape ``(n_group, n_neighbors, n_spatial_dims)``.
    n_neighbors : int
        Number of neighbors per entity in this batch (same for all entities).
    """

    entity_indices: torch.Tensor
    neighbor_indices: torch.Tensor
    relative_positions: torch.Tensor
    n_neighbors: int


def iter_neighborhood_batches(
    positions: torch.Tensor,
    adjacency: "Adjacency",
    *,
    min_neighbors: int = 0,
    max_neighbors: int | None = None,
) -> Iterator[NeighborhoodBatch]:
    """Iterate over neighborhoods grouped by neighbor count.

    Groups entities by their (possibly clamped) neighbor count and yields
    dense batches containing entity indices, neighbor indices, and relative
    position vectors. This converts a ragged adjacency structure into a
    sequence of regular batches suitable for ``torch.linalg`` operations.

    Parameters
    ----------
    positions : torch.Tensor
        Entity positions, shape ``(n_entities, n_spatial_dims)``.
    adjacency : Adjacency
        Adjacency structure mapping entities to their neighbors. Must have
        ``offsets`` and ``indices`` attributes (CSR format).
    min_neighbors : int
        Skip entities with fewer than this many (effective) neighbors.
    max_neighbors : int | None
        Clamp each entity's neighbor count to at most this value.
        Useful for PCA where only the k nearest neighbors are needed.
        If ``None``, use all neighbors.

    Yields
    ------
    NeighborhoodBatch
        One batch per unique (effective) neighbor count that meets the
        ``min_neighbors`` threshold. Batches are yielded in order of
        ascending neighbor count.
    """
    device = positions.device

    ### Compute per-entity neighbor counts from CSR offsets
    neighbor_counts = adjacency.offsets[1:] - adjacency.offsets[:-1]

    ### Optionally clamp to max_neighbors
    if max_neighbors is not None:
        effective_counts = torch.minimum(
            neighbor_counts,
            torch.tensor(max_neighbors, dtype=neighbor_counts.dtype, device=device),
        )
    else:
        effective_counts = neighbor_counts

    ### Group by effective neighbor count
    unique_counts, inverse_indices = torch.unique(effective_counts, return_inverse=True)

    ### Yield one batch per unique count
    for count_idx, count_tensor in enumerate(unique_counts):
        n_neighbors = int(count_tensor)

        # Skip groups below the minimum threshold
        if n_neighbors < min_neighbors:
            continue

        # Find all entities in this group
        entity_indices = torch.where(inverse_indices == count_idx)[0]
        if len(entity_indices) == 0:
            continue

        ### Extract neighbor indices for the entire group at once
        # Build a (n_group, n_neighbors) index matrix into adjacency.indices
        offsets_group = adjacency.offsets[entity_indices]  # (n_group,)
        col_range = torch.arange(n_neighbors, device=device)  # (n_neighbors,)
        flat_indices = offsets_group.unsqueeze(1) + col_range.unsqueeze(0)
        neighbor_indices = adjacency.indices[flat_indices]  # (n_group, n_neighbors)

        ### Gather positions and compute relative vectors
        center = positions[entity_indices]  # (n_group, n_spatial_dims)
        neighbor_pos = positions[
            neighbor_indices
        ]  # (n_group, n_neighbors, n_spatial_dims)
        relative = neighbor_pos - center.unsqueeze(1)

        yield NeighborhoodBatch(
            entity_indices=entity_indices,
            neighbor_indices=neighbor_indices,
            relative_positions=relative,
            n_neighbors=n_neighbors,
        )
