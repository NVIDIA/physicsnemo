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

"""Adversarial regressions for continuous, bounded contact-v2 responses."""

import pytest
import torch

from physicsnemo.experimental.models.meshtransolver import (
    FunctionalContactGraphBuilder,
    SparseContactBlock,
)
from physicsnemo.experimental.models.meshtransolver.contact import contact_edge_features


def builder(k=3, implementation="torch", **kwargs):
    return FunctionalContactGraphBuilder(
        3.0,
        k,
        implementation,
        smooth_cutoff=True,
        selection_taper=True,
        normal_epsilon=0.1,
        **kwargs,
    )


@pytest.mark.parametrize("implementation", ["torch", "warp"])
@pytest.mark.parametrize("k", [1, 3, 16])
def test_neighbor_exchange_has_no_finite_jump(implementation, k):
    torch.manual_seed(10)
    graph_builder = builder(k, implementation)
    dtype = torch.float32 if implementation == "warp" else torch.double
    block = SparseContactBlock(8, gate_init=1).to(dtype)
    latents = torch.randn(k + 2, 8, dtype=dtype)
    points = torch.zeros(k + 2, 3, dtype=dtype)
    points[1:k, 0] = torch.linspace(0.2, 0.7, k - 1)
    points[k, 0], points[k + 1, 0] = 1, -1

    def response(epsilon):
        displaced = points.clone()
        displaced[k, 0] += epsilon
        graph = graph_builder(displaced)
        return block(latents, graph, return_correction=True)[0]

    jumps = [
        (response(eps) - response(-eps)).norm().item() for eps in (1e-2, 1e-4, 1e-6)
    ]
    assert jumps[-1] < 2e-6, jumps
    assert jumps[-1] < jumps[0] * 0.005, jumps


@pytest.mark.parametrize("implementation", ["torch", "warp"])
@pytest.mark.parametrize("n,k", [(0, 3), (1, 3), (2, 3), (8, 1), (8, 3)])
def test_coincident_clusters_padding_and_empty(implementation, n, k):
    positions = torch.zeros(n, 3, requires_grad=True)
    graph = builder(k, implementation)(positions)
    assert torch.isfinite(graph.edge_features).all()
    assert torch.isfinite(graph.edge_weights).all()
    if n > k + 1:
        assert torch.equal(graph.edge_weights, torch.zeros_like(graph.edge_weights))
    if graph.edge_features.numel():
        (graph.edge_features.sum() + graph.edge_weights.sum()).backward()
        assert torch.isfinite(positions.grad).all()


@pytest.mark.parametrize("separation", [0, 1e-9, 1e-6, 1e-3, 0.1, 1.0])
def test_regularized_direction_gradient_is_physically_bounded(separation):
    relative = torch.tensor(
        [[separation, 0.0, 0.0]], dtype=torch.double, requires_grad=True
    )

    def direction(r):
        return contact_edge_features(r, r.new_zeros(1), 3, normal_epsilon=0.1)[:, 5:8]

    jacobian = torch.autograd.functional.jacobian(direction, relative)
    assert torch.isfinite(jacobian).all()
    assert jacobian.abs().max() <= 10 + 1e-10
    assert torch.autograd.gradcheck(direction, (relative,), eps=1e-7)


def test_live_buffer_distance_velocity_and_weights_gradcheck():
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.4, 0.1, 0.0],
            [0.8, -0.2, 0.1],
            [1.3, 0.1, 0.2],
            [2.0, 0.4, 0.1],
        ],
        dtype=torch.double,
        requires_grad=True,
    )
    velocities = torch.randn_like(points, requires_grad=True)
    graph_builder = builder(2, include_velocity=True, activation_distance=1.5)

    def output(p, v):
        graph = graph_builder(p, velocities=v)
        return torch.cat((graph.edge_features.ravel(), graph.edge_weights))

    assert torch.autograd.gradcheck(output, (points, velocities))
    graph = graph_builder(points, velocities=velocities)
    # Query zero keeps nodes 1 and 2; buffer node 3 must still receive gradient.
    loss = graph.edge_weights[graph.edge_index[1] == 0].sum()
    gradient = torch.autograd.grad(loss, points)[0]
    assert gradient[3].norm() > 0


def test_gap_band_has_zero_response_and_slope_at_boundary():
    graph_builder = builder(3, activation_distance=0.5)
    for gap in (0.5, 0.50001, 1.0):
        points = torch.tensor(
            [[0.0, 0.0, 0.0], [gap + 0.2, 0.0, 0.0]], requires_grad=True
        )
        graph = graph_builder(points, shell_thickness=torch.full((2,), 0.2))
        assert graph.edge_weights.max() < 1e-12
        gradient = torch.autograd.grad(graph.edge_weights.sum(), points)[0]
        assert gradient.abs().max() < 1e-5


def test_weights_rigid_motion_and_galilean_invariance():
    torch.manual_seed(5)
    points = torch.randn(10, 3, dtype=torch.double)
    velocity = torch.randn_like(points)
    rotation, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.double))
    graph_builder = builder(include_velocity=True)
    a = graph_builder(points, velocities=velocity)
    b = graph_builder(points @ rotation + 7, velocities=velocity @ rotation + 13)
    assert torch.equal(a.edge_index, b.edge_index)
    torch.testing.assert_close(a.edge_weights, b.edge_weights)
    torch.testing.assert_close(a.edge_features[:, 3:5], b.edge_features[:, 3:5])
    torch.testing.assert_close(a.edge_features[:, 11], b.edge_features[:, 11])
    torch.testing.assert_close(
        a.edge_features[:, 5:8] @ rotation, b.edge_features[:, 5:8]
    )


def test_node_permutation_equivariance_even_for_boundary_ties():
    torch.manual_seed(91)
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
        ]
    )
    permutation = torch.tensor([2, 4, 0, 1, 3])
    latent = torch.randn(5, 8)
    block = SparseContactBlock(8, gate_init=1)
    graph_builder = builder(2)
    a = block(latent, graph_builder(points))
    b = block(latent[permutation], graph_builder(points[permutation]))
    torch.testing.assert_close(a[permutation], b)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -1, 0])
def test_invalid_regularization_rejected(invalid):
    with pytest.raises(ValueError):
        FunctionalContactGraphBuilder(3, normal_epsilon=invalid)
    with pytest.raises(ValueError):
        FunctionalContactGraphBuilder(3, activation_distance=invalid)


def test_nonfinite_state_fails_before_search():
    with pytest.raises(ValueError, match="finite"):
        builder()(torch.full((2, 3), float("nan")))


def test_pruning_exact_zero_messages_preserves_output_and_gradients():
    import copy

    torch.manual_seed(24)
    points = torch.tensor(
        [[0.0, 0, 0], [0.3, 0.1, 0], [1.5, 0, 0], [2.0, 0, 0]], dtype=torch.double
    )
    latents = torch.randn(4, 8, dtype=torch.double)
    block = SparseContactBlock(8, gate_init=1).double()
    reference = None
    counts = []
    for prune in (False, True):
        current = copy.deepcopy(block)
        p = points.clone().requires_grad_()
        h = latents.clone().requires_grad_()
        graph = builder(3, activation_distance=0.6, prune_zero_weight=prune)(p)
        counts.append(graph.edge_index.shape[1])
        output = current(h, graph)
        output.square().sum().backward()
        tensors = [output, p.grad, h.grad] + [
            param.grad for param in current.parameters()
        ]
        if reference is None:
            reference = tensors
        else:
            for a, b in zip(tensors, reference):
                torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-10)
    assert counts[1] < counts[0]
