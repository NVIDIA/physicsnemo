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

import pytest
import torch

from physicsnemo.experimental.models.meshtransolver import (
    ContactGraph,
    FunctionalContactGraphBuilder,
    SparseContactBlock,
    merge_contact_graphs,
    smooth_contact_cutoff,
)


@pytest.mark.parametrize("implementation", ["torch", "warp"])
def test_nearest_after_exclusions_and_coincident_points(implementation):
    positions = torch.zeros(7, 3)
    positions[:, 0] = torch.tensor([0, 0, 0.1, 0.2, 0.3, 0.4, 5])
    # The nearest structural neighbors must not consume the top-k slots.
    structural = torch.tensor([[0, 0, 0], [2, 3, 4]])
    builder = FunctionalContactGraphBuilder(0.5, 2, implementation)
    graph = builder(positions, structural)
    selected = graph.edge_index[0, graph.edge_index[1] == 0]
    assert selected.tolist() == [1, 5]
    assert not (graph.edge_index[0] == graph.edge_index[1]).any()
    for a, b in structural.t():
        assert not ((graph.edge_index[0] == a) & (graph.edge_index[1] == b)).any()
        assert not ((graph.edge_index[0] == b) & (graph.edge_index[1] == a)).any()
    assert torch.bincount(graph.edge_index[1]).max() <= 2
    assert torch.isfinite(graph.edge_features).all()


@pytest.mark.parametrize("implementation", ["torch", "warp"])
def test_ragged_interleaved_batches_extra_exclusions_and_reuse(implementation):
    positions = torch.tensor(
        [[0.0, 0, 0], [0.1, 0, 0], [0.2, 0, 0], [0.3, 0, 0], [0.4, 0, 0]]
    )
    batch = torch.tensor([8, 3, 8, 3, 8])
    builder = FunctionalContactGraphBuilder(1, 3, implementation)
    topology = builder.prepare_topology(
        5,
        batch=batch,
        device=positions.device,
        extra_exclusion_edges=torch.tensor([[0], [2]]),
    )
    graph = builder(positions, topology=topology)
    source, destination = graph.edge_index
    assert torch.equal(batch[source], batch[destination])
    assert set(zip(source.tolist(), destination.tolist())) == {
        (0, 4),
        (4, 0),
        (2, 4),
        (4, 2),
        (1, 3),
        (3, 1),
    }
    # Reuse only static topology; live positions must drive each new discovery.
    moved = positions.clone()
    moved[4, 0] = 10
    graph = builder(moved, topology=topology)
    assert not (graph.edge_index == 4).any()


def test_live_geometry_velocity_weights_gradcheck():
    positions = torch.tensor(
        [[0.0, 0, 0], [0.4, 0.2, 0.1], [0.8, -0.1, 0.1]],
        dtype=torch.double,
        requires_grad=True,
    )
    velocities = torch.randn_like(positions, requires_grad=True)
    builder = FunctionalContactGraphBuilder(
        2, 2, "torch", include_velocity=True, velocity_scale=3, smooth_cutoff=True
    )

    def features(p, v):
        graph = builder(p, velocities=v)
        return torch.cat((graph.edge_features.flatten(), graph.edge_weights))

    assert torch.autograd.gradcheck(features, (positions, velocities))
    graph = builder(positions, velocities=velocities)
    source, destination = graph.edge_index
    torch.testing.assert_close(
        graph.edge_features[:, 8:11], (velocities[source] - velocities[destination]) / 3
    )


def test_smooth_cutoff_and_biased_node_correction_vanish():
    distance = torch.tensor([0.0, 1.0 - 1e-6, 1.0, 1.1], requires_grad=True)
    weights = smooth_contact_cutoff(distance, 1)
    (derivative,) = torch.autograd.grad(weights.sum(), distance)
    assert weights[0] == 1 and torch.equal(weights[2:], torch.zeros(2))
    assert derivative[2] == 0 and abs(derivative[1]) < 1e-5
    nodes = torch.randn(3, 8, requires_grad=True)
    block = SparseContactBlock(8, gate_init=1)
    graph = ContactGraph(
        torch.tensor([[0], [1]]),
        torch.ones(1, 8),
        torch.zeros(1, dtype=torch.bool),
        torch.zeros(1),
    )
    torch.testing.assert_close(block(nodes, graph), nodes, atol=0, rtol=0)


@pytest.mark.parametrize("kind", ["empty", "nodes", "obstacle"])
def test_all_parameters_have_gradients_for_distributed_reduction(kind):
    block = SparseContactBlock(8, gate_init=0.1)
    nodes = torch.randn(3, 8, requires_grad=True)
    graph = ContactGraph.empty(nodes.device, nodes.dtype)
    if kind != "empty":
        graph = ContactGraph(
            torch.tensor([[0], [1]]),
            torch.randn(1, 8),
            torch.tensor([kind == "obstacle"]),
        )
    result = block(nodes, graph)
    result.square().sum().backward()
    for name, parameter in block.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        if kind == "empty":
            assert torch.count_nonzero(parameter.grad) == 0, name
    if kind == "empty":
        torch.testing.assert_close(result, nodes, atol=0, rtol=0)


def test_merge_weights_preserves_gradients_and_unweighted_edges():
    weights = torch.tensor([0.4], requires_grad=True)
    graph = ContactGraph(
        torch.tensor([[0], [1]]), torch.zeros(1, 8), torch.tensor([False]), weights
    )
    unweighted = ContactGraph(
        graph.edge_index, graph.edge_features, graph.obstacle_mask
    )
    merged = merge_contact_graphs(graph, unweighted).to(torch.device("cpu"))
    torch.testing.assert_close(merged.edge_weights, torch.tensor([0.4, 1.0]))
    merged.edge_weights.sum().backward()
    assert weights.grad == 1


@pytest.mark.parametrize("implementation", ["torch", "warp"])
def test_empty_and_single_node(implementation):
    builder = FunctionalContactGraphBuilder(
        1, 16, implementation, include_velocity=True
    )
    for n in (0, 1):
        graph = builder(torch.zeros(n, 3), velocities=torch.zeros(n, 3))
        assert graph.edge_index.shape == (2, 0)
        assert graph.edge_features.shape == (0, 12)


def test_invalid_topology_and_thickness():
    builder = FunctionalContactGraphBuilder(1)
    positions = torch.zeros(2, 3)
    with pytest.raises(ValueError, match="different batch"):
        builder(positions, torch.tensor([[0], [1]]), batch=torch.arange(2))
    with pytest.raises(ValueError, match="bounds"):
        builder(positions, torch.tensor([[0], [2]]))
    with pytest.raises(ValueError, match="finite nonnegative"):
        builder(positions, shell_thickness=torch.tensor([1.0, float("nan")]))
