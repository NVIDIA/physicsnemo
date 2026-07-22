# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the tensor-only as-rigid-as-possible energy."""

import inspect
from typing import Literal, get_type_hints

import pytest
import torch

import physicsnemo.nn.functional as functional
from benchmarks.physicsnemo.nn.functional.registry import FUNCTIONAL_SPECS
from physicsnemo.core.function_spec import FunctionSpec
from physicsnemo.nn.functional import arap_energy
from physicsnemo.nn.functional.geometry import ARAPEnergy


def _complete_edges(num_points: int, device: torch.device | str = "cpu"):
    """Return every undirected edge of a complete graph."""

    return torch.combinations(
        torch.arange(num_points, device=device, dtype=torch.int64), r=2
    )


def _proper_orthogonal_matrix(
    num_dims: int,
    dtype: torch.dtype,
    device: torch.device | str,
) -> torch.Tensor:
    """Construct a deterministic proper orthogonal matrix."""

    generator = torch.Generator(device=device).manual_seed(4101 + num_dims)
    matrix = torch.randn(
        (num_dims, num_dims), generator=generator, dtype=dtype, device=device
    )
    orthogonal, _ = torch.linalg.qr(matrix)
    final_sign = torch.where(
        torch.linalg.det(orthogonal) < 0,
        -torch.ones((), dtype=dtype, device=device),
        torch.ones((), dtype=dtype, device=device),
    )
    correction = torch.cat(
        (
            torch.ones(num_dims - 1, dtype=dtype, device=device),
            final_sign.unsqueeze(0),
        )
    )
    return orthogonal * correction.unsqueeze(0)


def _reduced_benchmark_case(args, kwargs, num_points=32):
    """Reduce a generated workload while retaining its batch and weight modes."""

    reference, deformed, _ = args
    reference = reference[..., :num_points, :].detach().clone()
    deformed = deformed[..., :num_points, :].detach().clone()
    edge_start = torch.arange(num_points - 1, device=reference.device)
    edges = torch.stack((edge_start, edge_start + 1), dim=-1)
    weights = kwargs["edge_weights"]
    if weights is not None:
        weights = weights[: num_points - 1].detach().clone()
    return (reference, deformed, edges), {"edge_weights": weights}


def test_public_exports_function_spec_and_benchmark_registry():
    assert arap_energy.__name__ == "arap_energy"
    assert arap_energy.__module__ == ("physicsnemo.nn.functional.geometry.deform.arap")
    assert issubclass(ARAPEnergy, FunctionSpec)
    assert ARAPEnergy in FUNCTIONAL_SPECS
    assert not hasattr(functional, "ARAPEnergy")
    assert list(inspect.signature(arap_energy).parameters) == [
        "reference_points",
        "deformed_points",
        "edges",
        "edge_weights",
        "implementation",
    ]
    assert ARAPEnergy.implementations() == ("torch",)
    assert get_type_hints(arap_energy)["implementation"] == Literal["torch"] | None


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("num_dims", [2, 3, 4])
def test_energy_is_zero_for_proper_rigid_motion(device, dtype, num_dims):
    device = torch.device(device)
    generator = torch.Generator(device=device).manual_seed(4201 + num_dims)
    reference = torch.randn(
        (num_dims + 2, num_dims), generator=generator, dtype=dtype, device=device
    )
    rotation = _proper_orthogonal_matrix(num_dims, dtype, device)
    translation = torch.linspace(-0.7, 0.4, num_dims, dtype=dtype, device=device)
    deformed = reference @ rotation.mT + translation
    edges = _complete_edges(reference.shape[0], device)

    energy = arap_energy(reference, deformed, edges)

    assert energy.shape == ()
    tolerance = 2.0e-4 if dtype == torch.float32 else 1.0e-11
    torch.testing.assert_close(energy, torch.zeros_like(energy), atol=tolerance, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_uniform_scale_has_known_energy(dtype):
    reference = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=dtype
    )
    edges = _complete_edges(4)
    weights = torch.tensor([0.5, 1.0, 1.5, 0.75, 1.25, 2.0], dtype=dtype)
    scale = 1.4
    reference_edges = reference[edges[:, 0]] - reference[edges[:, 1]]
    expected = (scale - 1) ** 2 * (weights * reference_edges.square().sum(dim=-1)).sum()

    actual = arap_energy(reference, scale * reference, edges, weights)

    torch.testing.assert_close(actual, expected)


def test_improper_reflection_is_not_a_rigid_motion():
    reference = torch.tensor(
        [[0.0, 0.0], [2.0, 0.0], [0.2, 1.0], [1.4, 1.7]],
        dtype=torch.float64,
    )
    reflection = torch.tensor([[-1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    edges = _complete_edges(4)

    energy = arap_energy(reference, reference @ reflection.mT, edges)

    assert energy > 0.1


def test_batched_inputs_return_one_energy_per_batch():
    reference = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        dtype=torch.float64,
    )
    edges = _complete_edges(4)
    rotation = torch.tensor([[0.0, -1.0], [1.0, 0.0]], dtype=torch.float64)
    references = torch.stack((reference, reference))
    deformed = torch.stack((reference @ rotation.mT + 2.0, 1.25 * reference))

    actual = arap_energy(references, deformed, edges)
    expected_scaled = arap_energy(reference, 1.25 * reference, edges)

    assert actual.shape == (2,)
    torch.testing.assert_close(
        actual[0], torch.zeros_like(actual[0]), atol=1e-12, rtol=0
    )
    torch.testing.assert_close(actual[1], expected_scaled)


def test_envelope_theorem_gradients_match_finite_differences():
    reference = torch.tensor(
        [[0.0, 0.0], [1.1, 0.1], [-0.2, 1.0], [1.0, 1.3]],
        dtype=torch.float64,
        requires_grad=True,
    )
    deformed = torch.tensor(
        [[0.1, -0.05], [1.0, 0.2], [-0.1, 0.9], [1.2, 1.1]],
        dtype=torch.float64,
        requires_grad=True,
    )
    weights = torch.tensor(
        [0.7, 1.1, 0.8, 1.3, 0.9, 1.2],
        dtype=torch.float64,
        requires_grad=True,
    )
    edges = _complete_edges(4)

    assert torch.autograd.gradcheck(
        lambda ref, deform, weight: arap_energy(ref, deform, edges, weight),
        (reference, deformed, weights),
        eps=1.0e-6,
        atol=2.0e-5,
        rtol=2.0e-4,
    )


def test_repeated_singular_values_have_finite_first_derivatives():
    reference = torch.tensor(
        [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    deformed = reference.detach().clone().requires_grad_()
    edges = _complete_edges(4)

    gradients = torch.autograd.grad(
        arap_energy(reference, deformed, edges), (reference, deformed)
    )

    for gradient in gradients:
        assert torch.isfinite(gradient).all()
        torch.testing.assert_close(
            gradient, torch.zeros_like(gradient), atol=1e-12, rtol=0
        )


def test_empty_edge_graph_returns_differentiable_zero():
    reference = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
    deformed = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
    edges = torch.empty((0, 2), dtype=torch.int64)
    weights = torch.empty(0, dtype=torch.float64, requires_grad=True)

    energy = arap_energy(reference, deformed, edges, weights)
    gradients = torch.autograd.grad(energy, (reference, deformed, weights))

    torch.testing.assert_close(energy, torch.zeros_like(energy))
    for gradient in gradients:
        torch.testing.assert_close(gradient, torch.zeros_like(gradient))


@pytest.mark.parametrize(
    ("mutate", "error", "match"),
    [
        (
            lambda ref, deform, edges, weights: (
                ref.unsqueeze(0).unsqueeze(0),
                deform,
                edges,
                weights,
            ),
            ValueError,
            "reference_points must have shape",
        ),
        (
            lambda ref, deform, edges, weights: (
                ref[:, :1],
                deform[:, :1],
                edges,
                weights,
            ),
            ValueError,
            "coordinate dimension must be at least 2",
        ),
        (
            lambda ref, deform, edges, weights: (
                ref.to(torch.float16),
                deform.to(torch.float16),
                edges,
                weights.to(torch.float16),
            ),
            TypeError,
            "reference_points must have dtype",
        ),
        (
            lambda ref, deform, edges, weights: (ref, deform[:-1], edges, weights),
            ValueError,
            "must have identical shapes",
        ),
        (
            lambda ref, deform, edges, weights: (
                ref,
                deform.to(torch.float32),
                edges,
                weights,
            ),
            TypeError,
            "must have the same dtype",
        ),
        (
            lambda ref, deform, edges, weights: (ref, deform, edges[:, 0], weights),
            ValueError,
            "edges must have shape",
        ),
        (
            lambda ref, deform, edges, weights: (
                ref,
                deform,
                edges.to(torch.int32),
                weights,
            ),
            TypeError,
            "edges must have dtype torch.int64",
        ),
        (
            lambda ref, deform, edges, weights: (ref, deform, edges, weights[:-1]),
            ValueError,
            "edge_weights must have shape",
        ),
        (
            lambda ref, deform, edges, weights: (
                ref,
                deform,
                edges,
                weights.to(torch.float32),
            ),
            TypeError,
            "edge_weights and point tensors must have the same dtype",
        ),
    ],
)
def test_structural_validation(mutate, error, match):
    reference = torch.randn(4, 2, dtype=torch.float64)
    deformed = torch.randn(4, 2, dtype=torch.float64)
    edges = _complete_edges(4)
    weights = torch.ones(edges.shape[0], dtype=torch.float64)
    args = mutate(reference, deformed, edges, weights)

    with pytest.raises(error, match=match):
        arap_energy(*args)


def test_out_of_range_edges_are_rejected_by_torch_indexing():
    points = torch.randn(4, 2)
    edges = torch.tensor([[0, 4]], dtype=torch.int64)

    with pytest.raises((IndexError, RuntimeError)):
        arap_energy(points, points, edges)


def test_unknown_backend_is_rejected():
    points = torch.randn(4, 2)
    edges = _complete_edges(4)

    with pytest.raises(KeyError, match="No implementation named 'warp'"):
        arap_energy(points, points, edges, implementation="warp")


def test_torch_compile_fullgraph():
    reference = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    deformed = reference + torch.tensor(
        [[0.0, 0.0], [0.1, 0.0], [0.0, -0.05], [0.08, 0.03]]
    )
    edges = _complete_edges(4)
    weights = torch.linspace(0.5, 1.5, edges.shape[0])

    def operation(ref, deform, topology, edge_weights):
        return arap_energy(
            ref,
            deform,
            topology,
            edge_weights,
            implementation="torch",
        )

    expected = operation(reference, deformed, edges, weights)
    actual = torch.compile(operation, fullgraph=True, backend="eager")(
        reference, deformed, edges, weights
    )

    torch.testing.assert_close(actual, expected)


def test_benchmark_generators_and_hooks(device):
    device = torch.device(device)
    forward_labels = []
    for label, args, kwargs in ARAPEnergy.make_inputs_forward(device=device):
        forward_labels.append(label)
        args, kwargs = _reduced_benchmark_case(args, kwargs)
        output = ARAPEnergy.dispatch(*args, implementation="torch", **kwargs)
        ARAPEnergy.compare_forward(output, output.detach().clone())
    assert forward_labels == [case[0] for case in ARAPEnergy._FORWARD_BENCHMARK_CASES]

    backward_labels = []
    for label, args, kwargs in ARAPEnergy.make_inputs_backward(device=device):
        backward_labels.append(label)
        args, kwargs = _reduced_benchmark_case(args, kwargs)
        reference, deformed, edges = args
        weights = kwargs["edge_weights"]
        reference.requires_grad_(label.endswith("-all"))
        deformed.requires_grad_(True)
        weights.requires_grad_(label.endswith("-all"))
        output = ARAPEnergy.dispatch(
            reference,
            deformed,
            edges,
            edge_weights=weights,
            implementation="torch",
        )
        output.sum().backward()
        assert deformed.grad is not None
        assert torch.isfinite(deformed.grad).all()
        ARAPEnergy.compare_backward(deformed.grad, deformed.grad.detach().clone())
    assert backward_labels == [case[0] for case in ARAPEnergy._BACKWARD_BENCHMARK_CASES]
