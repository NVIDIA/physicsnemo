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

from physicsnemo.nn.functional import uniform_grid_gradient
from physicsnemo.nn.functional.derivatives import UniformGridGradient
from test.conftest import requires_module
from test.nn.functional._parity_utils import clone_case


# Build periodic analytic fields for derivative correctness checks.
def _make_periodic_field(device: str, dims: int, derivative_order: int):
    torch_device = torch.device(device)

    if dims == 1:
        n0 = 512
        x0 = torch.arange(n0, device=torch_device, dtype=torch.float32) / float(n0)
        field = torch.sin(2.0 * torch.pi * x0)
        spacing = 1.0 / float(n0)
        if derivative_order == 1:
            expected = (2.0 * torch.pi) * torch.cos(2.0 * torch.pi * x0).unsqueeze(0)
        else:
            expected = (
                -((2.0 * torch.pi) ** 2) * torch.sin(2.0 * torch.pi * x0)
            ).unsqueeze(0)
        return field, spacing, expected

    if dims == 2:
        n0, n1 = 192, 160
        x0 = torch.arange(n0, device=torch_device, dtype=torch.float32) / float(n0)
        x1 = torch.arange(n1, device=torch_device, dtype=torch.float32) / float(n1)
        xx, yy = torch.meshgrid(x0, x1, indexing="ij")
        field = torch.sin(2.0 * torch.pi * xx) + 0.5 * torch.cos(4.0 * torch.pi * yy)
        spacing = (1.0 / float(n0), 1.0 / float(n1))
        if derivative_order == 1:
            deriv_x = (2.0 * torch.pi) * torch.cos(2.0 * torch.pi * xx)
            deriv_y = -2.0 * torch.pi * torch.sin(4.0 * torch.pi * yy)
        else:
            deriv_x = -((2.0 * torch.pi) ** 2) * torch.sin(2.0 * torch.pi * xx)
            deriv_y = -8.0 * (torch.pi**2) * torch.cos(4.0 * torch.pi * yy)
        expected = torch.stack((deriv_x, deriv_y), dim=0)
        return field, spacing, expected

    n0, n1, n2 = 80, 72, 64
    x0 = torch.arange(n0, device=torch_device, dtype=torch.float32) / float(n0)
    x1 = torch.arange(n1, device=torch_device, dtype=torch.float32) / float(n1)
    x2 = torch.arange(n2, device=torch_device, dtype=torch.float32) / float(n2)
    xx, yy, zz = torch.meshgrid(x0, x1, x2, indexing="ij")
    field = (
        torch.sin(2.0 * torch.pi * xx)
        + 0.5 * torch.cos(2.0 * torch.pi * yy)
        + 0.25 * torch.sin(4.0 * torch.pi * zz)
    )
    spacing = (1.0 / float(n0), 1.0 / float(n1), 1.0 / float(n2))
    if derivative_order == 1:
        deriv_x = (2.0 * torch.pi) * torch.cos(2.0 * torch.pi * xx)
        deriv_y = -1.0 * torch.pi * torch.sin(2.0 * torch.pi * yy)
        deriv_z = 1.0 * torch.pi * torch.cos(4.0 * torch.pi * zz)
    else:
        deriv_x = -((2.0 * torch.pi) ** 2) * torch.sin(2.0 * torch.pi * xx)
        deriv_y = -2.0 * (torch.pi**2) * torch.cos(2.0 * torch.pi * yy)
        deriv_z = -4.0 * (torch.pi**2) * torch.sin(4.0 * torch.pi * zz)
    expected = torch.stack((deriv_x, deriv_y, deriv_z), dim=0)
    return field, spacing, expected


# Validate torch backend against analytic periodic derivatives.
@pytest.mark.parametrize("dims", [1, 2, 3])
@pytest.mark.parametrize("derivative_order", [1, 2])
@pytest.mark.parametrize("order", [2, 4])
def test_uniform_grid_gradient_torch(
    device: str, dims: int, derivative_order: int, order: int
):
    field, spacing, expected = _make_periodic_field(device, dims, derivative_order)
    output = UniformGridGradient.dispatch(
        field,
        spacing=spacing,
        order=order,
        derivative_orders=derivative_order,
        implementation="torch",
    )
    torch.testing.assert_close(output, expected, atol=5e-2, rtol=5e-2)


# Validate unified derivative-order requests concatenate outputs deterministically.
@pytest.mark.parametrize("dims", [1, 2, 3])
def test_uniform_grid_gradient_torch_combined_orders(device: str, dims: int):
    field, spacing, expected_first = _make_periodic_field(
        device, dims, derivative_order=1
    )
    _, _, expected_second = _make_periodic_field(device, dims, derivative_order=2)

    output = UniformGridGradient.dispatch(
        field,
        spacing=spacing,
        order=2,
        derivative_orders=(1, 2),
        include_mixed=False,
        implementation="torch",
    )
    expected = torch.cat((expected_first, expected_second), dim=0)
    torch.testing.assert_close(output, expected, atol=5e-2, rtol=5e-2)


# Validate mixed second derivatives are available through unified API.
@pytest.mark.parametrize("dims", [2, 3])
def test_uniform_grid_gradient_torch_second_order_mixed(device: str, dims: int):
    field, spacing, _expected = _make_periodic_field(device, dims, derivative_order=2)
    output = UniformGridGradient.dispatch(
        field,
        spacing=spacing,
        order=2,
        derivative_orders=2,
        include_mixed=True,
        implementation="torch",
    )
    expected_count = dims + (dims * (dims - 1)) // 2
    assert output.shape[0] == expected_count


# Validate higher-order stencil improves analytic error for smooth fields.
@pytest.mark.parametrize("dims", [1, 2, 3])
def test_uniform_grid_gradient_torch_order4_more_accurate(device: str, dims: int):
    field, spacing, expected = _make_periodic_field(device, dims, derivative_order=1)
    out_o2 = UniformGridGradient.dispatch(
        field,
        spacing=spacing,
        order=2,
        derivative_orders=1,
        implementation="torch",
    )
    out_o4 = UniformGridGradient.dispatch(
        field,
        spacing=spacing,
        order=4,
        derivative_orders=1,
        implementation="torch",
    )

    err_o2 = torch.linalg.vector_norm((out_o2 - expected).reshape(-1)).item()
    err_o4 = torch.linalg.vector_norm((out_o4 - expected).reshape(-1)).item()
    assert err_o4 < err_o2


# Validate warp backend forward parity against torch for representative cases.
@requires_module("warp")
def test_uniform_grid_gradient_backend_forward_parity(device: str):
    for _label, args, kwargs in UniformGridGradient.make_inputs_forward(device=device):
        args_torch, kwargs_torch = clone_case(args, kwargs)
        args_warp, kwargs_warp = clone_case(args, kwargs)

        out_torch = UniformGridGradient.dispatch(
            *args_torch,
            implementation="torch",
            **kwargs_torch,
        )
        out_warp = UniformGridGradient.dispatch(
            *args_warp,
            implementation="warp",
            **kwargs_warp,
        )
        UniformGridGradient.compare_forward(out_warp, out_torch)


# Validate warp backend against analytic periodic derivatives.
@requires_module("warp")
@pytest.mark.parametrize("dims", [1, 2, 3])
@pytest.mark.parametrize("derivative_order", [1, 2])
def test_uniform_grid_gradient_warp(device: str, dims: int, derivative_order: int):
    field, spacing, expected = _make_periodic_field(device, dims, derivative_order)
    output = UniformGridGradient.dispatch(
        field,
        spacing=spacing,
        order=2,
        derivative_orders=derivative_order,
        implementation="warp",
    )
    torch.testing.assert_close(output, expected, atol=7e-2, rtol=7e-2)


# Validate warp backend backward parity against torch for representative workloads.
@requires_module("warp")
def test_uniform_grid_gradient_backend_backward_parity(device: str):
    for _label, args, kwargs in UniformGridGradient.make_inputs_backward(device=device):
        args_torch, kwargs_torch = clone_case(args, kwargs)
        args_warp, kwargs_warp = clone_case(args, kwargs)

        out_torch = UniformGridGradient.dispatch(
            *args_torch,
            implementation="torch",
            **kwargs_torch,
        )
        out_torch.square().mean().backward()
        grad_torch = args_torch[0].grad
        assert grad_torch is not None

        out_warp = UniformGridGradient.dispatch(
            *args_warp,
            implementation="warp",
            **kwargs_warp,
        )
        out_warp.square().mean().backward()
        grad_warp = args_warp[0].grad
        assert grad_warp is not None

        UniformGridGradient.compare_backward(grad_warp, grad_torch)


# Validate warp backend backward parity for fused combined-order requests.
@requires_module("warp")
@pytest.mark.parametrize("dims", [1, 2, 3])
def test_uniform_grid_gradient_backend_backward_combined_orders(device: str, dims: int):
    field, spacing, _ = _make_periodic_field(device, dims, derivative_order=1)

    field_torch = field.to(torch.float32).detach().clone().requires_grad_(True)
    field_warp = field.to(torch.float32).detach().clone().requires_grad_(True)

    out_torch = UniformGridGradient.dispatch(
        field_torch,
        spacing=spacing,
        order=2,
        derivative_orders=(1, 2),
        include_mixed=False,
        implementation="torch",
    )
    out_warp = UniformGridGradient.dispatch(
        field_warp,
        spacing=spacing,
        order=2,
        derivative_orders=(1, 2),
        include_mixed=False,
        implementation="warp",
    )

    grad_seed = torch.randn_like(out_torch)
    grad_torch = torch.autograd.grad(
        outputs=out_torch,
        inputs=field_torch,
        grad_outputs=grad_seed,
        create_graph=False,
        retain_graph=False,
        allow_unused=False,
    )[0]
    grad_warp = torch.autograd.grad(
        outputs=out_warp,
        inputs=field_warp,
        grad_outputs=grad_seed,
        create_graph=False,
        retain_graph=False,
        allow_unused=False,
    )[0]

    assert grad_torch is not None
    assert grad_warp is not None
    torch.testing.assert_close(grad_warp, grad_torch, atol=7e-2, rtol=7e-2)


# Validate warp backend mixed second-derivative backward parity against torch.
@requires_module("warp")
@pytest.mark.parametrize("dims", [2, 3])
def test_uniform_grid_gradient_backend_backward_mixed_orders(device: str, dims: int):
    field, spacing, _ = _make_periodic_field(device, dims, derivative_order=1)

    field_torch = field.to(torch.float32).detach().clone().requires_grad_(True)
    field_warp = field.to(torch.float32).detach().clone().requires_grad_(True)

    out_torch = UniformGridGradient.dispatch(
        field_torch,
        spacing=spacing,
        order=2,
        derivative_orders=2,
        include_mixed=True,
        implementation="torch",
    )
    out_warp = UniformGridGradient.dispatch(
        field_warp,
        spacing=spacing,
        order=2,
        derivative_orders=2,
        include_mixed=True,
        implementation="warp",
    )

    grad_seed = torch.randn_like(out_torch)
    grad_torch = torch.autograd.grad(
        outputs=out_torch,
        inputs=field_torch,
        grad_outputs=grad_seed,
        create_graph=False,
        retain_graph=False,
        allow_unused=False,
    )[0]
    grad_warp = torch.autograd.grad(
        outputs=out_warp,
        inputs=field_warp,
        grad_outputs=grad_seed,
        create_graph=False,
        retain_graph=False,
        allow_unused=False,
    )[0]

    assert grad_torch is not None
    assert grad_warp is not None
    torch.testing.assert_close(grad_warp, grad_torch, atol=7e-2, rtol=7e-2)


# Validate benchmark input generation contract for forward inputs.
def test_uniform_grid_gradient_make_inputs_forward(device: str):
    label, args, kwargs = next(
        iter(UniformGridGradient.make_inputs_forward(device=device))
    )
    assert isinstance(label, str)
    assert isinstance(args, tuple)
    assert isinstance(kwargs, dict)

    field = args[0]
    assert field.ndim in (1, 2, 3)
    assert torch.is_floating_point(field)

    output = UniformGridGradient.dispatch(
        *args,
        implementation="torch",
        **kwargs,
    )
    assert output.shape[0] == field.ndim


# Validate benchmark input generation contract for backward inputs.
def test_uniform_grid_gradient_make_inputs_backward(device: str):
    label, args, kwargs = next(
        iter(UniformGridGradient.make_inputs_backward(device=device))
    )
    assert isinstance(label, str)
    assert isinstance(args, tuple)
    assert isinstance(kwargs, dict)

    field = args[0]
    assert field.requires_grad

    output = UniformGridGradient.dispatch(
        *args,
        implementation="torch",
        **kwargs,
    )
    output.square().mean().backward()
    assert field.grad is not None


# Validate compare-forward hook contract.
def test_uniform_grid_gradient_compare_forward_contract(device: str):
    _label, args, kwargs = next(
        iter(UniformGridGradient.make_inputs_forward(device=device))
    )
    output = UniformGridGradient.dispatch(*args, implementation="torch", **kwargs)
    reference = output.detach().clone()
    UniformGridGradient.compare_forward(output, reference)


# Validate compare-backward hook contract.
def test_uniform_grid_gradient_compare_backward_contract(device: str):
    _label, args, kwargs = next(
        iter(UniformGridGradient.make_inputs_backward(device=device))
    )
    field = args[0]

    output = UniformGridGradient.dispatch(*args, implementation="torch", **kwargs)
    output.square().mean().backward()

    assert field.grad is not None
    UniformGridGradient.compare_backward(field.grad, field.grad.detach().clone())


# Validate exported functional API and error handling paths.
def test_uniform_grid_gradient_error_handling(device: str):
    field = torch.randn(16, device=device, dtype=torch.float32)

    output = uniform_grid_gradient(field, spacing=1.0)
    assert output.shape == (1, 16)
    assert output.dtype == torch.float32

    with pytest.raises(ValueError, match="supports 1D-3D fields"):
        UniformGridGradient.dispatch(
            torch.randn(4, 4, 4, 4, device=device, dtype=torch.float32),
            implementation="torch",
        )

    with pytest.raises(TypeError, match="floating-point"):
        UniformGridGradient.dispatch(
            torch.ones(8, device=device, dtype=torch.int32),
            implementation="torch",
        )

    with pytest.raises(ValueError, match="spacing must have"):
        UniformGridGradient.dispatch(
            torch.randn(8, 8, device=device, dtype=torch.float32),
            spacing=(1.0,),
            implementation="torch",
        )

    with pytest.raises(ValueError, match="strictly positive"):
        UniformGridGradient.dispatch(
            torch.randn(8, 8, device=device, dtype=torch.float32),
            spacing=(1.0, 0.0),
            implementation="torch",
        )

    with pytest.raises(ValueError, match="supports"):
        UniformGridGradient.dispatch(
            torch.randn(8, 8, device=device, dtype=torch.float32),
            order=6,
            implementation="torch",
        )

    with pytest.raises(TypeError, match="integer"):
        UniformGridGradient.dispatch(
            torch.randn(8, 8, device=device, dtype=torch.float32),
            order=2.0,  # type: ignore[arg-type]
            implementation="torch",
        )

    with pytest.raises(ValueError, match="supports derivative orders"):
        UniformGridGradient.dispatch(
            torch.randn(8, 8, device=device, dtype=torch.float32),
            derivative_orders=3,
            implementation="torch",
        )

    with pytest.raises(TypeError, match="include_mixed must be a bool"):
        UniformGridGradient.dispatch(
            torch.randn(8, 8, device=device, dtype=torch.float32),
            derivative_orders=2,
            include_mixed=1,  # type: ignore[arg-type]
            implementation="torch",
        )

    with pytest.raises(ValueError, match="only valid when requesting 2nd derivatives"):
        UniformGridGradient.dispatch(
            torch.randn(8, 8, device=device, dtype=torch.float32),
            derivative_orders=1,
            include_mixed=True,
            implementation="torch",
        )

    out = UniformGridGradient.dispatch(
        torch.randn(8, 8, device=device, dtype=torch.float32),
        derivative_orders=2,
        include_mixed=True,
        implementation="torch",
    )
    assert out.shape[0] == 3
