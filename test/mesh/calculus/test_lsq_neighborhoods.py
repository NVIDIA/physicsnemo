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

"""Neighborhood batching for LSQ gradients: grouping, exactness, and syncs."""

import pytest
import torch

from physicsnemo.mesh import Mesh
from physicsnemo.mesh.calculus._lsq_intrinsic import (
    compute_point_gradient_lsq_intrinsic,
)
from physicsnemo.mesh.calculus._neighborhoods import iter_neighborhood_batches
from physicsnemo.mesh.neighbors._adjacency import Adjacency
from physicsnemo.mesh.primitives.planar import structured_grid
from test.mesh.mesh.test_slicing_sync import _cuda_sync_budget


def _ragged_adjacency(device) -> tuple[torch.Tensor, Adjacency, list[list[int]]]:
    """Random positions and neighbor lists of 0 to 6 entries each."""
    generator = torch.Generator().manual_seed(0)
    n_entities = 40
    counts = torch.randint(0, 7, (n_entities,), generator=generator)
    neighbors = [
        torch.randint(0, n_entities, (int(c),), generator=generator).tolist()
        for c in counts
    ]
    offsets = torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)])
    indices = torch.tensor(sum(neighbors, []), dtype=torch.long)
    adjacency = Adjacency(offsets=offsets.to(device), indices=indices.to(device))
    positions = torch.randn(n_entities, 3, generator=generator).to(device)
    return positions, adjacency, neighbors


@pytest.mark.parametrize("min_neighbors, max_neighbors", [(0, None), (2, None), (1, 3)])
def test_batches_match_reference_grouping(min_neighbors, max_neighbors, device):
    positions, adjacency, neighbors = _ragged_adjacency(device)
    clamp = (
        (lambda n: n) if max_neighbors is None else (lambda n: min(n, max_neighbors))
    )
    expected = {}
    for entity, entity_neighbors in enumerate(neighbors):
        k = clamp(len(entity_neighbors))
        if k >= min_neighbors:
            expected.setdefault(k, []).append((entity, entity_neighbors[:k]))

    batches = list(
        iter_neighborhood_batches(
            positions,
            adjacency,
            min_neighbors=min_neighbors,
            max_neighbors=max_neighbors,
        )
    )

    # One batch per count, in ascending count order, entities in ascending order
    assert [b.n_neighbors for b in batches] == sorted(expected)
    for batch in batches:
        entities = [e for e, _ in expected[batch.n_neighbors]]
        assert batch.entity_indices.tolist() == entities
        assert batch.neighbor_indices.tolist() == [
            n for _, n in expected[batch.n_neighbors]
        ]
        torch.testing.assert_close(
            batch.relative_positions,
            positions[batch.neighbor_indices] - positions[batch.entity_indices, None],
        )


def test_no_entities_yield_no_batches(device):
    adjacency = Adjacency(
        offsets=torch.zeros(1, dtype=torch.long, device=device),
        indices=torch.zeros(0, dtype=torch.long, device=device),
    )
    positions = torch.zeros(0, 3, device=device)
    assert list(iter_neighborhood_batches(positions, adjacency)) == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("max_neighbors", [None, 3])
def test_batching_synchronizes_twice_for_any_number_of_groups(max_neighbors):
    """Finding the groups and reading their sizes: two syncs, not two per group."""
    positions, adjacency, _ = _ragged_adjacency("cuda")
    list(iter_neighborhood_batches(positions, adjacency, max_neighbors=max_neighbors))
    torch.cuda.synchronize()

    with _cuda_sync_budget(2):
        list(
            iter_neighborhood_batches(positions, adjacency, max_neighbors=max_neighbors)
        )
    torch.cuda.synchronize()


@pytest.mark.parametrize("value_shape", [(), (2,), (2, 3)])
def test_intrinsic_gradient_is_exact_for_linear_fields_on_a_plane(value_shape, device):
    """LSQ reproduces linear data exactly, so only the tangent projection remains."""
    grid = structured_grid.load(n_x=7, n_y=6, device=device)
    # Embed the planar grid in a tilted plane of 3D space
    basis = torch.tensor(
        [[0.6, 0.0, 0.8], [0.0, 1.0, 0.0]], dtype=torch.float64, device=device
    )
    mesh = Mesh(points=grid.points.double() @ basis, cells=grid.cells)
    generator = torch.Generator().manual_seed(1)
    ambient_gradient = torch.randn(3, *value_shape, generator=generator).double()
    ambient_gradient = ambient_gradient.to(device)
    values = torch.einsum("ps,s...->p...", mesh.points, ambient_gradient)

    gradient = compute_point_gradient_lsq_intrinsic(mesh, values)

    projector = basis.T @ basis  # orthogonal projection onto the plane
    expected = torch.einsum("st,t...->s...", projector, ambient_gradient)
    torch.testing.assert_close(
        gradient, expected.expand(mesh.n_points, *expected.shape)
    )
