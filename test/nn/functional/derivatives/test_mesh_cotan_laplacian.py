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

from physicsnemo.nn.functional import mesh_cotan_laplacian
from physicsnemo.nn.functional.derivatives import MeshCotanLaplacian
from test.conftest import requires_module
from test.nn.functional.derivatives._mesh_vector_calculus_test_utils import (
    check_backend_backward_parity,
    check_backend_forward_parity,
    make_simple_cotan_case,
)


def test_mesh_cotan_laplacian_simple_chain(device: str):
    _points, edges, weights, volumes = make_simple_cotan_case(device)
    values = torch.tensor([0.0, 1.0, 3.0], device=torch.device(device))

    output = mesh_cotan_laplacian(
        edges,
        weights,
        volumes,
        values,
        implementation="torch",
    )

    expected = torch.tensor([1.0, 1.0, -2.0], device=values.device)
    torch.testing.assert_close(output, expected)


def test_mesh_cotan_laplacian_empty(device: str):
    torch_device = torch.device(device)
    edges = torch.empty((0, 2), dtype=torch.int64, device=torch_device)
    weights = torch.empty((0,), dtype=torch.float32, device=torch_device)
    volumes = torch.empty((0,), dtype=torch.float32, device=torch_device)
    values = torch.empty((0,), dtype=torch.float32, device=torch_device)

    output = mesh_cotan_laplacian(
        edges, weights, volumes, values, implementation="torch"
    )
    assert output.shape == values.shape
    assert output.dtype == values.dtype


@requires_module("warp")
def test_mesh_cotan_laplacian_empty_backend_parity(device: str):
    torch_device = torch.device(device)
    edges = torch.empty((0, 2), dtype=torch.int64, device=torch_device)
    weights = torch.empty((0,), dtype=torch.float32, device=torch_device)
    volumes = torch.empty((0,), dtype=torch.float32, device=torch_device)
    values = torch.empty((0, 2), dtype=torch.float32, device=torch_device)

    torch_output = mesh_cotan_laplacian(
        edges, weights, volumes, values, implementation="torch"
    )
    warp_output = mesh_cotan_laplacian(
        edges, weights, volumes, values, implementation="warp"
    )
    torch.testing.assert_close(warp_output, torch_output)


@requires_module("warp")
def test_mesh_cotan_laplacian_clamped_volume_gradient(device: str):
    _points, edges, weights, volumes = make_simple_cotan_case(device)
    volumes[0] = 0.0
    values = torch.tensor([0.0, 1.0, 3.0], device=volumes.device)

    torch_volumes = volumes.clone().requires_grad_(True)
    torch_output = mesh_cotan_laplacian(
        edges, weights, torch_volumes, values, implementation="torch"
    )
    torch_output.sum().backward()

    warp_volumes = volumes.clone().requires_grad_(True)
    warp_output = mesh_cotan_laplacian(
        edges, weights, warp_volumes, values, implementation="warp"
    )
    warp_output.sum().backward()

    assert torch_volumes.grad is not None
    assert warp_volumes.grad is not None
    assert torch_volumes.grad[0] == 0
    torch.testing.assert_close(warp_volumes.grad, torch_volumes.grad)


@requires_module("warp")
def test_mesh_cotan_laplacian_mixed_volume_dtype(device: str):
    _points, edges, weights, volumes = make_simple_cotan_case(device)
    values = torch.tensor([0.0, 1.0, 3.0], device=volumes.device)
    volumes = volumes.to(torch.float64)

    torch_output = mesh_cotan_laplacian(
        edges, weights, volumes, values, implementation="torch"
    )
    warp_output = mesh_cotan_laplacian(
        edges, weights, volumes, values, implementation="warp"
    )
    assert torch_output.dtype == values.dtype
    assert warp_output.dtype == values.dtype
    torch.testing.assert_close(warp_output, torch_output)


@requires_module("warp")
def test_mesh_cotan_laplacian_backend_forward_parity(device: str):
    check_backend_forward_parity(device, MeshCotanLaplacian)


@requires_module("warp")
def test_mesh_cotan_laplacian_backend_backward_parity(device: str):
    check_backend_backward_parity(device, MeshCotanLaplacian, (1, 2, 3))


def _make_irregular_case(device: str, edge_dtype: torch.dtype):
    """Build a multicomponent case with a nontrivial edge/point ratio."""
    edges = torch.tensor(
        [[0, 1], [1, 2], [2, 3], [0, 2], [1, 3]],
        dtype=edge_dtype,
        device=device,
    )
    weights = torch.tensor(
        [1.0, 0.7, 0.8, 0.3, 0.2], dtype=torch.float32, device=device
    )
    volumes = torch.tensor([1.0, 1.2, 0.9, 1.1], dtype=torch.float32, device=device)
    values = torch.tensor(
        [
            [[0.2, 0.4], [0.7, -0.1]],
            [[0.3, 0.9], [-0.2, 0.8]],
            [[1.1, -0.3], [0.5, 0.6]],
            [[-0.4, 0.2], [0.9, -0.7]],
        ],
        dtype=torch.float32,
        device=device,
    )
    return edges, weights, volumes, values


def _run_with_grad_selection(
    device: str,
    edge_dtype: torch.dtype,
    implementation: str,
    grad_indices: tuple[int, ...],
):
    args = list(_make_irregular_case(device, edge_dtype))
    for index in grad_indices:
        args[index] = args[index].detach().clone().requires_grad_(True)
    output = mesh_cotan_laplacian(*args, implementation=implementation)
    output.square().sum().backward()
    return output.detach(), tuple(args[index].grad for index in (1, 2, 3))


@requires_module("warp")
@pytest.mark.parametrize("edge_dtype", [torch.int32, torch.int64])
def test_mesh_cotan_laplacian_edge_dtype_forward_backward(
    device: str, edge_dtype: torch.dtype
):
    torch_output, torch_grads = _run_with_grad_selection(
        device, edge_dtype, "torch", (1, 2, 3)
    )
    warp_output, warp_grads = _run_with_grad_selection(
        device, edge_dtype, "warp", (1, 2, 3)
    )

    MeshCotanLaplacian.compare_forward(warp_output, torch_output)
    for warp_grad, torch_grad in zip(warp_grads, torch_grads):
        assert warp_grad is not None and torch_grad is not None
        MeshCotanLaplacian.compare_backward(warp_grad, torch_grad)


@requires_module("warp")
@pytest.mark.parametrize("grad_index", [1, 2, 3])
def test_mesh_cotan_laplacian_selective_backward(device: str, grad_index: int):
    torch_output, torch_grads = _run_with_grad_selection(
        device, torch.int64, "torch", (grad_index,)
    )
    warp_output, warp_grads = _run_with_grad_selection(
        device, torch.int64, "warp", (grad_index,)
    )

    MeshCotanLaplacian.compare_forward(warp_output, torch_output)
    selected_position = (1, 2, 3).index(grad_index)
    for position, (warp_grad, torch_grad) in enumerate(zip(warp_grads, torch_grads)):
        if position == selected_position:
            assert warp_grad is not None and torch_grad is not None
            MeshCotanLaplacian.compare_backward(warp_grad, torch_grad)
        else:
            assert warp_grad is None and torch_grad is None


@requires_module("warp")
def test_mesh_cotan_laplacian_noncontiguous_inputs(device: str):
    edges, weights, volumes, values = _make_irregular_case(device, torch.int64)
    edges = edges.T.contiguous().T
    weights = torch.stack((weights, weights), dim=1)[:, 0]
    volumes = torch.stack((volumes, volumes), dim=1)[:, 0]
    values = values.movedim(0, -1).contiguous().movedim(-1, 0)
    assert not edges.is_contiguous()
    assert not weights.is_contiguous()
    assert not volumes.is_contiguous()
    assert not values.is_contiguous()

    torch_output = mesh_cotan_laplacian(
        edges, weights, volumes, values, implementation="torch"
    )
    warp_output = mesh_cotan_laplacian(
        edges, weights, volumes, values, implementation="warp"
    )
    MeshCotanLaplacian.compare_forward(warp_output, torch_output)


@requires_module("warp")
def test_mesh_cotan_laplacian_edgeless_backward(device: str):
    edges = torch.empty((0, 2), dtype=torch.int64, device=device)
    weights = torch.empty((0,), dtype=torch.float32, device=device, requires_grad=True)
    volumes = torch.ones((4,), dtype=torch.float32, device=device, requires_grad=True)
    values = torch.randn((4, 2), dtype=torch.float32, device=device, requires_grad=True)

    output = mesh_cotan_laplacian(
        edges, weights, volumes, values, implementation="warp"
    )
    torch.testing.assert_close(output, torch.zeros_like(output))
    output.sum().backward()
    for tensor in (weights, volumes, values):
        assert tensor.grad is not None
        torch.testing.assert_close(tensor.grad, torch.zeros_like(tensor))


@requires_module("warp")
def test_mesh_cotan_laplacian_empty_mesh_backward(device: str):
    edges = torch.empty((0, 2), dtype=torch.int32, device=device)
    weights = torch.empty((0,), dtype=torch.float32, device=device, requires_grad=True)
    volumes = torch.empty((0,), dtype=torch.float32, device=device, requires_grad=True)
    values = torch.empty((0, 2), dtype=torch.float32, device=device, requires_grad=True)

    output = mesh_cotan_laplacian(
        edges, weights, volumes, values, implementation="warp"
    )
    assert output.shape == values.shape
    output.sum().backward()
    for tensor in (weights, volumes, values):
        assert tensor.grad is not None
        assert tensor.grad.shape == tensor.shape


@requires_module("warp")
def test_mesh_cotan_laplacian_inference_geometry_then_backward(device: str):
    with torch.inference_mode():
        edges, weights, volumes, _ = _make_irregular_case(device, torch.int64)

    values = torch.randn(
        (volumes.shape[0], 2),
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    output = mesh_cotan_laplacian(
        edges, weights, volumes, values, implementation="warp"
    )
    output.square().sum().backward()

    assert values.grad is not None
    assert torch.isfinite(values.grad).all()


@requires_module("warp")
def test_mesh_cotan_laplacian_zero_components_backward(device: str):
    edges, weights, volumes, _values = _make_irregular_case(device, torch.int64)
    weights.requires_grad_(True)
    volumes.requires_grad_(True)
    values = torch.empty(
        (volumes.shape[0], 0),
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )

    output = mesh_cotan_laplacian(
        edges, weights, volumes, values, implementation="warp"
    )
    assert output.shape == values.shape
    output.sum().backward()
    for tensor in (weights, volumes, values):
        assert tensor.grad is not None
        torch.testing.assert_close(tensor.grad, torch.zeros_like(tensor))


@requires_module("warp")
@pytest.mark.parametrize("bad_index", [-1, 4])
def test_mesh_cotan_laplacian_rejects_invalid_edges(device: str, bad_index: int):
    edges, weights, volumes, values = _make_irregular_case(device, torch.int64)
    edges = edges.clone()
    edges[0, 0] = bad_index
    with pytest.raises(ValueError, match="0 <= index < n_points"):
        mesh_cotan_laplacian(edges, weights, volumes, values, implementation="warp")


def _first_and_second_grads(device: str, implementation: str):
    args = list(_make_irregular_case(device, torch.int64))
    differentiable = []
    for index in (1, 2, 3):
        args[index] = args[index].detach().clone().requires_grad_(True)
        differentiable.append(args[index])
    output = mesh_cotan_laplacian(*args, implementation=implementation)
    first = torch.autograd.grad(
        output.square().sum(), differentiable, create_graph=True
    )
    second_loss = sum(gradient.square().sum() for gradient in first)
    second = torch.autograd.grad(second_loss, differentiable)
    return first, second


@requires_module("warp")
def test_mesh_cotan_laplacian_higher_order_backward(device: str):
    torch_first, torch_second = _first_and_second_grads(device, "torch")
    warp_first, warp_second = _first_and_second_grads(device, "warp")
    for warp_grad, torch_grad in zip(warp_first, torch_first):
        MeshCotanLaplacian.compare_backward(warp_grad, torch_grad)
    for warp_grad, torch_grad in zip(warp_second, torch_second):
        torch.testing.assert_close(warp_grad, torch_grad, atol=2e-5, rtol=2e-5)


@requires_module("warp")
def test_mesh_cotan_laplacian_compiled_forward(device: str):
    edges, weights, volumes, values = _make_irregular_case(device, torch.int64)

    def forward_fn(input_values: torch.Tensor) -> torch.Tensor:
        return mesh_cotan_laplacian(
            edges, weights, volumes, input_values, implementation="warp"
        )

    compiled_forward = torch.compile(forward_fn, backend="aot_eager", fullgraph=True)
    MeshCotanLaplacian.compare_forward(compiled_forward(values), forward_fn(values))


@requires_module("warp")
def test_mesh_cotan_laplacian_compiled_backward(device: str):
    edges, weights, volumes, values = _make_irregular_case(device, torch.int64)
    assert edges.shape[0] != values.shape[0]

    reference_weights = weights.detach().clone().requires_grad_(True)
    reference_volumes = volumes.detach().clone().requires_grad_(True)
    reference_values = values.detach().clone().requires_grad_(True)
    reference_loss = (
        mesh_cotan_laplacian(
            edges,
            reference_weights,
            reference_volumes,
            reference_values,
            implementation="torch",
        )
        .square()
        .sum()
    )
    reference_loss.backward()

    def loss_fn(
        input_weights: torch.Tensor,
        input_volumes: torch.Tensor,
        input_values: torch.Tensor,
    ) -> torch.Tensor:
        return (
            mesh_cotan_laplacian(
                edges,
                input_weights,
                input_volumes,
                input_values,
                implementation="warp",
            )
            .square()
            .sum()
        )

    compiled_loss_fn = torch.compile(loss_fn, backend="aot_eager", fullgraph=True)
    compiled_weights = weights.detach().clone().requires_grad_(True)
    compiled_volumes = volumes.detach().clone().requires_grad_(True)
    compiled_values = values.detach().clone().requires_grad_(True)
    compiled_loss = compiled_loss_fn(
        compiled_weights, compiled_volumes, compiled_values
    )
    compiled_loss.backward()

    assert compiled_weights.grad is not None
    assert compiled_weights.grad.shape == (edges.shape[0],)
    for compiled_arg, reference_arg in (
        (compiled_weights, reference_weights),
        (compiled_volumes, reference_volumes),
        (compiled_values, reference_values),
    ):
        assert compiled_arg.grad is not None and reference_arg.grad is not None
        MeshCotanLaplacian.compare_backward(compiled_arg.grad, reference_arg.grad)
