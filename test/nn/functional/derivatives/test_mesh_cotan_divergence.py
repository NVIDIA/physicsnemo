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

from physicsnemo.mesh import Mesh
from physicsnemo.nn.functional import mesh_cotan_divergence
from physicsnemo.nn.functional.derivatives import MeshCotanDivergence
from test.conftest import requires_module
from test.nn.functional.derivatives._mesh_vector_calculus_test_utils import (
    check_backend_backward_parity,
    check_backend_forward_parity,
    make_simple_cotan_case,
)


def test_mesh_cotan_divergence_simple_chain(device: str):
    points, edges, weights, volumes = make_simple_cotan_case(device)
    vector_field = points.clone()

    output = mesh_cotan_divergence(
        points,
        edges,
        weights,
        volumes,
        vector_field,
        implementation="torch",
    )

    expected = torch.tensor([0.5, 1.0, -1.5], device=points.device)
    torch.testing.assert_close(output, expected)


@requires_module("warp")
def test_mesh_cotan_divergence_clamped_volume_gradient(device: str):
    points, edges, weights, volumes = make_simple_cotan_case(device)
    volumes[0] = 0.0
    vector_field = points.clone()

    torch_volumes = volumes.clone().requires_grad_(True)
    torch_output = mesh_cotan_divergence(
        points,
        edges,
        weights,
        torch_volumes,
        vector_field,
        implementation="torch",
    )
    torch_output.sum().backward()

    warp_volumes = volumes.clone().requires_grad_(True)
    warp_output = mesh_cotan_divergence(
        points,
        edges,
        weights,
        warp_volumes,
        vector_field,
        implementation="warp",
    )
    warp_output.sum().backward()

    assert torch_volumes.grad is not None
    assert warp_volumes.grad is not None
    assert torch_volumes.grad[0] == 0
    torch.testing.assert_close(warp_volumes.grad, torch_volumes.grad)


@requires_module("warp")
def test_mesh_cotan_divergence_mixed_volume_dtype(device: str):
    points, edges, weights, volumes = make_simple_cotan_case(device)
    vector_field = points.to(torch.float32)
    points = points.to(torch.float64)
    weights = weights.to(torch.float64)
    volumes = volumes.to(torch.float64)

    torch_output = mesh_cotan_divergence(
        points,
        edges,
        weights,
        volumes,
        vector_field,
        implementation="torch",
    )
    warp_output = mesh_cotan_divergence(
        points,
        edges,
        weights,
        volumes,
        vector_field,
        implementation="warp",
    )
    assert torch_output.dtype == vector_field.dtype
    assert warp_output.dtype == vector_field.dtype
    torch.testing.assert_close(warp_output, torch_output)


@requires_module("warp")
def test_mesh_cotan_divergence_backend_forward_parity(device: str):
    check_backend_forward_parity(device, MeshCotanDivergence)


@requires_module("warp")
def test_mesh_cotan_divergence_backend_backward_parity(device: str):
    check_backend_backward_parity(device, MeshCotanDivergence, (0, 2, 3, 4))


def _make_irregular_case(device: str, edge_dtype: torch.dtype):
    """Build a case with more edges than points to exercise fake shapes."""
    points = torch.tensor(
        [[0.0, 0.0], [1.0, 0.2], [2.0, -0.1], [3.0, 0.4]],
        dtype=torch.float32,
        device=device,
    )
    edges = torch.tensor(
        [[0, 1], [1, 2], [2, 3], [0, 2], [1, 3]],
        dtype=edge_dtype,
        device=device,
    )
    weights = torch.tensor(
        [1.0, 0.7, 0.8, 0.3, 0.2], dtype=torch.float32, device=device
    )
    volumes = torch.tensor([1.0, 1.2, 0.9, 1.1], dtype=torch.float32, device=device)
    vector_field = torch.tensor(
        [[0.2, 0.4], [0.7, -0.1], [0.3, 0.9], [-0.2, 0.8]],
        dtype=torch.float32,
        device=device,
    )
    return points, edges, weights, volumes, vector_field


@requires_module("warp")
def test_mesh_cotan_divergence_inference_geometry_backward(device: str):
    with torch.inference_mode():
        points, edges, weights, volumes, vector_field = _make_irregular_case(
            device, torch.int64
        )

    vector_field = vector_field.clone().requires_grad_(True)
    output = mesh_cotan_divergence(
        points,
        edges,
        weights,
        volumes,
        vector_field,
        implementation="warp",
    )
    output.square().sum().backward()

    assert vector_field.grad is not None
    assert torch.isfinite(vector_field.grad).all()


@requires_module("warp")
def test_mesh_constructed_in_inference_mode_supports_training_backward(device: str):
    with torch.inference_mode():
        mesh = Mesh(
            points=torch.tensor(
                [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
                device=device,
            ),
            cells=torch.tensor([[0, 1, 2], [1, 3, 2]], device=device),
        )
        mesh.divergence(
            torch.randn((mesh.n_points, 2), device=device),
            method="dec",
            implementation="warp",
        )

    vector_field = torch.randn((mesh.n_points, 2), device=device, requires_grad=True)
    output = mesh.divergence(
        vector_field,
        method="dec",
        implementation="warp",
    )
    output.square().sum().backward()

    assert vector_field.grad is not None
    assert torch.isfinite(vector_field.grad).all()


def _run_with_grad_selection(
    device: str,
    edge_dtype: torch.dtype,
    implementation: str,
    grad_indices: tuple[int, ...],
):
    args = list(_make_irregular_case(device, edge_dtype))
    for index in grad_indices:
        args[index] = args[index].detach().clone().requires_grad_(True)
    output = mesh_cotan_divergence(*args, implementation=implementation)
    output.square().sum().backward()
    return output.detach(), tuple(args[index].grad for index in (0, 2, 3, 4))


@requires_module("warp")
@pytest.mark.parametrize("edge_dtype", [torch.int32, torch.int64])
def test_mesh_cotan_divergence_edge_dtype_forward_backward(
    device: str, edge_dtype: torch.dtype
):
    torch_output, torch_grads = _run_with_grad_selection(
        device, edge_dtype, "torch", (0, 2, 3, 4)
    )
    warp_output, warp_grads = _run_with_grad_selection(
        device, edge_dtype, "warp", (0, 2, 3, 4)
    )

    MeshCotanDivergence.compare_forward(warp_output, torch_output)
    for warp_grad, torch_grad in zip(warp_grads, torch_grads):
        assert warp_grad is not None and torch_grad is not None
        MeshCotanDivergence.compare_backward(warp_grad, torch_grad)


@requires_module("warp")
@pytest.mark.parametrize("grad_index", [0, 2, 3, 4])
def test_mesh_cotan_divergence_selective_backward(device: str, grad_index: int):
    torch_output, torch_grads = _run_with_grad_selection(
        device, torch.int64, "torch", (grad_index,)
    )
    warp_output, warp_grads = _run_with_grad_selection(
        device, torch.int64, "warp", (grad_index,)
    )

    MeshCotanDivergence.compare_forward(warp_output, torch_output)
    selected_position = (0, 2, 3, 4).index(grad_index)
    for position, (warp_grad, torch_grad) in enumerate(zip(warp_grads, torch_grads)):
        if position == selected_position:
            assert warp_grad is not None and torch_grad is not None
            MeshCotanDivergence.compare_backward(warp_grad, torch_grad)
        else:
            assert warp_grad is None and torch_grad is None


@requires_module("warp")
def test_mesh_cotan_divergence_noncontiguous_inputs(device: str):
    points, edges, weights, volumes, vector_field = _make_irregular_case(
        device, torch.int64
    )
    points = points.T.contiguous().T
    edges = edges.T.contiguous().T
    weights = torch.stack((weights, weights), dim=1)[:, 0]
    volumes = torch.stack((volumes, volumes), dim=1)[:, 0]
    vector_field = vector_field.T.contiguous().T
    assert not points.is_contiguous()
    assert not edges.is_contiguous()
    assert not weights.is_contiguous()
    assert not volumes.is_contiguous()
    assert not vector_field.is_contiguous()

    torch_output = mesh_cotan_divergence(
        points, edges, weights, volumes, vector_field, implementation="torch"
    )
    warp_output = mesh_cotan_divergence(
        points, edges, weights, volumes, vector_field, implementation="warp"
    )
    MeshCotanDivergence.compare_forward(warp_output, torch_output)


@requires_module("warp")
def test_mesh_cotan_divergence_edgeless_backward(device: str):
    points = torch.randn((4, 2), dtype=torch.float32, device=device).requires_grad_(
        True
    )
    edges = torch.empty((0, 2), dtype=torch.int64, device=device)
    weights = torch.empty((0,), dtype=torch.float32, device=device).requires_grad_(True)
    volumes = torch.ones((4,), dtype=torch.float32, device=device).requires_grad_(True)
    vector_field = torch.randn(
        (4, 2), dtype=torch.float32, device=device, requires_grad=True
    )

    output = mesh_cotan_divergence(
        points, edges, weights, volumes, vector_field, implementation="warp"
    )
    torch.testing.assert_close(output, torch.zeros_like(output))
    output.sum().backward()
    for tensor in (points, weights, volumes, vector_field):
        assert tensor.grad is not None
        torch.testing.assert_close(tensor.grad, torch.zeros_like(tensor))


@requires_module("warp")
def test_mesh_cotan_divergence_empty_mesh_backward(device: str):
    points = torch.empty((0, 2), dtype=torch.float32, device=device, requires_grad=True)
    edges = torch.empty((0, 2), dtype=torch.int32, device=device)
    weights = torch.empty((0,), dtype=torch.float32, device=device, requires_grad=True)
    volumes = torch.empty((0,), dtype=torch.float32, device=device, requires_grad=True)
    vector_field = torch.empty(
        (0, 2), dtype=torch.float32, device=device, requires_grad=True
    )

    output = mesh_cotan_divergence(
        points, edges, weights, volumes, vector_field, implementation="warp"
    )
    assert output.shape == (0,)
    output.sum().backward()
    for tensor in (points, weights, volumes, vector_field):
        assert tensor.grad is not None
        assert tensor.grad.shape == tensor.shape


def _first_and_second_grads(device: str, implementation: str):
    args = list(_make_irregular_case(device, torch.int64))
    differentiable = []
    for index in (0, 2, 3, 4):
        args[index] = args[index].detach().clone().requires_grad_(True)
        differentiable.append(args[index])
    output = mesh_cotan_divergence(*args, implementation=implementation)
    first = torch.autograd.grad(
        output.square().sum(), differentiable, create_graph=True
    )
    second_loss = sum(gradient.square().sum() for gradient in first)
    second = torch.autograd.grad(second_loss, differentiable)
    return first, second


@requires_module("warp")
def test_mesh_cotan_divergence_higher_order_backward(device: str):
    torch_first, torch_second = _first_and_second_grads(device, "torch")
    warp_first, warp_second = _first_and_second_grads(device, "warp")
    for warp_grad, torch_grad in zip(warp_first, torch_first):
        MeshCotanDivergence.compare_backward(warp_grad, torch_grad)
    for warp_grad, torch_grad in zip(warp_second, torch_second):
        torch.testing.assert_close(warp_grad, torch_grad, atol=2e-5, rtol=2e-5)


@requires_module("warp")
def test_mesh_cotan_divergence_compiled_weight_backward(device: str):
    points, edges, weights, volumes, vector_field = _make_irregular_case(
        device, torch.int64
    )
    assert edges.shape[0] != points.shape[0]

    reference_weights = weights.detach().clone().requires_grad_(True)
    reference_loss = (
        mesh_cotan_divergence(
            points,
            edges,
            reference_weights,
            volumes,
            vector_field,
            implementation="torch",
        )
        .square()
        .sum()
    )
    reference_loss.backward()

    def loss_fn(input_weights: torch.Tensor) -> torch.Tensor:
        return (
            mesh_cotan_divergence(
                points,
                edges,
                input_weights,
                volumes,
                vector_field,
                implementation="warp",
            )
            .square()
            .sum()
        )

    compiled_loss_fn = torch.compile(loss_fn, backend="aot_eager", fullgraph=True)
    compiled_weights = weights.detach().clone().requires_grad_(True)
    compiled_loss = compiled_loss_fn(compiled_weights)
    compiled_loss.backward()

    assert compiled_weights.grad is not None
    assert compiled_weights.grad.shape == (edges.shape[0],)
    MeshCotanDivergence.compare_backward(compiled_weights.grad, reference_weights.grad)
