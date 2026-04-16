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

from physicsnemo.nn.functional import rectilinear_grid_gradient
from physicsnemo.nn.functional.derivatives import RectilinearGridGradient
from test.conftest import requires_module
from test.nn.functional._parity_utils import clone_case


# Build analytic periodic fields on nonuniform rectilinear coordinates.
def _make_periodic_case(device: str, dims: int, derivative_order: int):
    torch_device = torch.device(device)
    amp0 = 0.04 if derivative_order == 1 else 0.0
    amp1 = 0.03 if derivative_order == 1 else 0.0
    amp2 = 0.02 if derivative_order == 1 else 0.0

    if dims == 1:
        n0 = 1024
        s0 = torch.linspace(0.0, 1.0, n0 + 1, device=torch_device)[:-1]
        x0 = s0 + amp0 * torch.sin(2.0 * torch.pi * s0)
        field = torch.sin(2.0 * torch.pi * x0)
        if derivative_order == 1:
            expected = (2.0 * torch.pi) * torch.cos(2.0 * torch.pi * x0).unsqueeze(0)
        else:
            expected = (
                -((2.0 * torch.pi) ** 2) * torch.sin(2.0 * torch.pi * x0)
            ).unsqueeze(0)
        return field, (x0.to(torch.float32),), 1.0, expected

    if dims == 2:
        n0, n1 = 320, 256
        s0 = torch.linspace(0.0, 1.0, n0 + 1, device=torch_device)[:-1]
        s1 = torch.linspace(0.0, 1.0, n1 + 1, device=torch_device)[:-1]
        x0 = s0 + amp0 * torch.sin(2.0 * torch.pi * s0)
        x1 = s1 + amp1 * torch.sin(2.0 * torch.pi * s1)
        xx, yy = torch.meshgrid(x0, x1, indexing="ij")
        field = torch.sin(2.0 * torch.pi * xx) + 0.5 * torch.cos(2.0 * torch.pi * yy)
        if derivative_order == 1:
            deriv_x = (2.0 * torch.pi) * torch.cos(2.0 * torch.pi * xx)
            deriv_y = -1.0 * torch.pi * torch.sin(2.0 * torch.pi * yy)
        else:
            deriv_x = -((2.0 * torch.pi) ** 2) * torch.sin(2.0 * torch.pi * xx)
            deriv_y = -2.0 * (torch.pi**2) * torch.cos(2.0 * torch.pi * yy)
        expected = torch.stack((deriv_x, deriv_y), dim=0)
        return field, (x0.to(torch.float32), x1.to(torch.float32)), (1.0, 1.0), expected

    n0, n1, n2 = 120, 96, 80
    s0 = torch.linspace(0.0, 1.0, n0 + 1, device=torch_device)[:-1]
    s1 = torch.linspace(0.0, 1.0, n1 + 1, device=torch_device)[:-1]
    s2 = torch.linspace(0.0, 1.0, n2 + 1, device=torch_device)[:-1]
    x0 = s0 + amp0 * torch.sin(2.0 * torch.pi * s0)
    x1 = s1 + amp1 * torch.sin(2.0 * torch.pi * s1)
    x2 = s2 + amp2 * torch.sin(2.0 * torch.pi * s2)
    xx, yy, zz = torch.meshgrid(x0, x1, x2, indexing="ij")
    field = (
        torch.sin(2.0 * torch.pi * xx)
        + 0.5 * torch.cos(2.0 * torch.pi * yy)
        + 0.25 * torch.sin(2.0 * torch.pi * zz)
    )
    if derivative_order == 1:
        deriv_x = (2.0 * torch.pi) * torch.cos(2.0 * torch.pi * xx)
        deriv_y = -1.0 * torch.pi * torch.sin(2.0 * torch.pi * yy)
        deriv_z = 0.5 * torch.pi * torch.cos(2.0 * torch.pi * zz)
    else:
        deriv_x = -((2.0 * torch.pi) ** 2) * torch.sin(2.0 * torch.pi * xx)
        deriv_y = -2.0 * (torch.pi**2) * torch.cos(2.0 * torch.pi * yy)
        deriv_z = -(torch.pi**2) * torch.sin(2.0 * torch.pi * zz)
    expected = torch.stack((deriv_x, deriv_y, deriv_z), dim=0)
    return (
        field,
        (x0.to(torch.float32), x1.to(torch.float32), x2.to(torch.float32)),
        (1.0, 1.0, 1.0),
        expected,
    )


# Validate torch backend against analytic periodic derivatives.
@pytest.mark.parametrize("dims", [1, 2, 3])
@pytest.mark.parametrize("derivative_order", [1, 2])
def test_rectilinear_grid_gradient_torch(device: str, dims: int, derivative_order: int):
    field, coordinates, periods, expected = _make_periodic_case(
        device, dims, derivative_order
    )
    output = RectilinearGridGradient.dispatch(
        field.to(torch.float32),
        coordinates,
        periods=periods,
        derivative_orders=derivative_order,
        implementation="torch",
    )
    atol, rtol = (6e-1, 1e-1) if derivative_order == 2 and dims == 1 else (3e-2, 3e-2)
    torch.testing.assert_close(output, expected, atol=atol, rtol=rtol)


# Validate unified derivative-order requests concatenate outputs deterministically.
@pytest.mark.parametrize("dims", [1, 2, 3])
def test_rectilinear_grid_gradient_torch_combined_orders(device: str, dims: int):
    field, coordinates, periods, _expected_first = _make_periodic_case(
        device, dims, derivative_order=1
    )
    first_only = RectilinearGridGradient.dispatch(
        field.to(torch.float32),
        coordinates,
        periods=periods,
        derivative_orders=1,
        include_mixed=False,
        implementation="torch",
    )
    second_only = RectilinearGridGradient.dispatch(
        field.to(torch.float32),
        coordinates,
        periods=periods,
        derivative_orders=2,
        include_mixed=False,
        implementation="torch",
    )
    output = RectilinearGridGradient.dispatch(
        field.to(torch.float32),
        coordinates,
        periods=periods,
        derivative_orders=(1, 2),
        include_mixed=False,
        implementation="torch",
    )
    expected = torch.cat((first_only, second_only), dim=0)
    torch.testing.assert_close(output, expected, atol=3e-2, rtol=3e-2)


# Validate mixed second derivatives are available through unified API.
@pytest.mark.parametrize("dims", [2, 3])
def test_rectilinear_grid_gradient_torch_second_order_mixed(device: str, dims: int):
    field, coordinates, periods, _ = _make_periodic_case(
        device, dims, derivative_order=2
    )
    output = RectilinearGridGradient.dispatch(
        field.to(torch.float32),
        coordinates,
        periods=periods,
        derivative_orders=2,
        include_mixed=True,
        implementation="torch",
    )
    expected_count = dims + (dims * (dims - 1)) // 2
    assert output.shape[0] == expected_count


# Validate warp backend against analytic periodic derivatives.
@requires_module("warp")
@pytest.mark.parametrize("dims", [1, 2, 3])
@pytest.mark.parametrize("derivative_order", [1, 2])
def test_rectilinear_grid_gradient_warp(device: str, dims: int, derivative_order: int):
    field, coordinates, periods, expected = _make_periodic_case(
        device, dims, derivative_order
    )
    output = RectilinearGridGradient.dispatch(
        field.to(torch.float32),
        coordinates,
        periods=periods,
        derivative_orders=derivative_order,
        implementation="warp",
    )
    atol, rtol = (6e-1, 1e-1) if derivative_order == 2 and dims == 1 else (4e-2, 4e-2)
    torch.testing.assert_close(output, expected, atol=atol, rtol=rtol)


# Validate warp backend forward parity against torch across benchmark cases.
@requires_module("warp")
def test_rectilinear_grid_gradient_backend_forward_parity(device: str):
    for _label, args, kwargs in RectilinearGridGradient.make_inputs_forward(
        device=device
    ):
        args_torch, kwargs_torch = clone_case(args, kwargs)
        args_warp, kwargs_warp = clone_case(args, kwargs)

        out_torch = RectilinearGridGradient.dispatch(
            *args_torch,
            implementation="torch",
            **kwargs_torch,
        )
        out_warp = RectilinearGridGradient.dispatch(
            *args_warp,
            implementation="warp",
            **kwargs_warp,
        )
        RectilinearGridGradient.compare_forward(out_warp, out_torch)


# Validate warp fused first+second path parity against torch.
@requires_module("warp")
@pytest.mark.parametrize("dims", [1, 2, 3])
def test_rectilinear_grid_gradient_backend_forward_combined_orders(
    device: str, dims: int
):
    field, coordinates, periods, _ = _make_periodic_case(
        device, dims, derivative_order=1
    )
    out_torch = RectilinearGridGradient.dispatch(
        field.to(torch.float32),
        coordinates,
        periods=periods,
        derivative_orders=(1, 2),
        include_mixed=False,
        implementation="torch",
    )
    out_warp = RectilinearGridGradient.dispatch(
        field.to(torch.float32),
        coordinates,
        periods=periods,
        derivative_orders=(1, 2),
        include_mixed=False,
        implementation="warp",
    )
    RectilinearGridGradient.compare_forward(out_warp, out_torch)


# Validate warp backend backward parity against torch.
@requires_module("warp")
def test_rectilinear_grid_gradient_backend_backward_parity(device: str):
    for _label, args, kwargs in RectilinearGridGradient.make_inputs_backward(
        device=device
    ):
        args_torch, kwargs_torch = clone_case(args, kwargs)
        args_warp, kwargs_warp = clone_case(args, kwargs)

        out_torch = RectilinearGridGradient.dispatch(
            *args_torch,
            implementation="torch",
            **kwargs_torch,
        )
        grad_seed = torch.randn_like(out_torch)
        grad_torch = torch.autograd.grad(
            outputs=out_torch,
            inputs=args_torch[0],
            grad_outputs=grad_seed,
            create_graph=False,
            retain_graph=False,
            allow_unused=False,
        )[0]
        assert grad_torch is not None

        out_warp = RectilinearGridGradient.dispatch(
            *args_warp,
            implementation="warp",
            **kwargs_warp,
        )
        grad_warp = torch.autograd.grad(
            outputs=out_warp,
            inputs=args_warp[0],
            grad_outputs=grad_seed,
            create_graph=False,
            retain_graph=False,
            allow_unused=False,
        )[0]
        assert grad_warp is not None

        RectilinearGridGradient.compare_backward(grad_warp, grad_torch)


# Validate warp backend backward parity for fused combined-order requests.
@requires_module("warp")
@pytest.mark.parametrize("dims", [1, 2, 3])
def test_rectilinear_grid_gradient_backend_backward_combined_orders(
    device: str, dims: int
):
    field, coordinates, periods, _ = _make_periodic_case(
        device, dims, derivative_order=1
    )

    field_torch = field.to(torch.float32).detach().clone().requires_grad_(True)
    field_warp = field.to(torch.float32).detach().clone().requires_grad_(True)

    out_torch = RectilinearGridGradient.dispatch(
        field_torch,
        coordinates,
        periods=periods,
        derivative_orders=(1, 2),
        include_mixed=False,
        implementation="torch",
    )
    out_warp = RectilinearGridGradient.dispatch(
        field_warp,
        coordinates,
        periods=periods,
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


# Validate benchmark input generation contract for forward inputs.
def test_rectilinear_grid_gradient_make_inputs_forward(device: str):
    label, args, kwargs = next(
        iter(RectilinearGridGradient.make_inputs_forward(device=device))
    )
    assert isinstance(label, str)
    assert isinstance(args, tuple)
    assert isinstance(kwargs, dict)

    field, coordinates = args
    assert field.ndim in (1, 2, 3)
    assert len(coordinates) == field.ndim

    output = RectilinearGridGradient.dispatch(
        *args,
        implementation="torch",
        **kwargs,
    )
    assert output.shape[0] == field.ndim


# Validate benchmark input generation contract for backward inputs.
def test_rectilinear_grid_gradient_make_inputs_backward(device: str):
    label, args, kwargs = next(
        iter(RectilinearGridGradient.make_inputs_backward(device=device))
    )
    assert isinstance(label, str)
    assert isinstance(args, tuple)
    assert isinstance(kwargs, dict)

    field = args[0]
    assert field.requires_grad

    output = RectilinearGridGradient.dispatch(
        *args,
        implementation="torch",
        **kwargs,
    )
    output.square().mean().backward()
    assert field.grad is not None


# Validate compare-forward hook contract.
def test_rectilinear_grid_gradient_compare_forward_contract(device: str):
    _label, args, kwargs = next(
        iter(RectilinearGridGradient.make_inputs_forward(device=device))
    )
    output = RectilinearGridGradient.dispatch(*args, implementation="torch", **kwargs)
    reference = output.detach().clone()
    RectilinearGridGradient.compare_forward(output, reference)


# Validate compare-backward hook contract.
def test_rectilinear_grid_gradient_compare_backward_contract(device: str):
    _label, args, kwargs = next(
        iter(RectilinearGridGradient.make_inputs_backward(device=device))
    )
    field = args[0]

    output = RectilinearGridGradient.dispatch(*args, implementation="torch", **kwargs)
    output.square().mean().backward()

    assert field.grad is not None
    RectilinearGridGradient.compare_backward(field.grad, field.grad.detach().clone())


# Validate exported API and input validation paths.
def test_rectilinear_grid_gradient_error_handling(device: str):
    x = torch.linspace(0.0, 1.0, 17, device=device)[:-1]
    field = torch.sin(2.0 * torch.pi * x).to(torch.float32)

    output = rectilinear_grid_gradient(field, (x.to(torch.float32),), periods=1.0)
    assert output.shape == (1, 16)

    with pytest.raises(ValueError, match="supports 1D-3D fields"):
        RectilinearGridGradient.dispatch(
            torch.randn(2, 2, 2, 2, device=device, dtype=torch.float32),
            (x, x, x, x),
            periods=1.0,
            implementation="torch",
        )

    with pytest.raises(ValueError, match="must contain one axis tensor"):
        RectilinearGridGradient.dispatch(
            torch.randn(32, 32, device=device, dtype=torch.float32),
            (torch.linspace(0.0, 1.0, 32, device=device),),
            periods=1.0,
            implementation="torch",
        )

    with pytest.raises(ValueError, match="strictly increasing"):
        bad_x = torch.tensor([0.0, 0.3, 0.2, 0.8], device=device, dtype=torch.float32)
        bad_f = torch.randn(4, device=device, dtype=torch.float32)
        RectilinearGridGradient.dispatch(
            bad_f,
            (bad_x,),
            periods=1.0,
            implementation="torch",
        )

    with pytest.raises(ValueError, match="must be larger than coordinate span"):
        RectilinearGridGradient.dispatch(
            torch.randn(16, device=device, dtype=torch.float32),
            (torch.linspace(0.0, 1.0, 16, device=device, dtype=torch.float32),),
            periods=0.8,
            implementation="torch",
        )

    with pytest.raises(ValueError, match="supports derivative orders"):
        RectilinearGridGradient.dispatch(
            torch.randn(16, device=device, dtype=torch.float32),
            (torch.linspace(0.0, 1.0, 16, device=device, dtype=torch.float32),),
            periods=1.0,
            derivative_orders=3,
            implementation="torch",
        )

    with pytest.raises(TypeError, match="include_mixed must be a bool"):
        RectilinearGridGradient.dispatch(
            torch.randn(16, device=device, dtype=torch.float32),
            (torch.linspace(0.0, 1.0, 16, device=device, dtype=torch.float32),),
            periods=1.0,
            derivative_orders=2,
            include_mixed=1,  # type: ignore[arg-type]
            implementation="torch",
        )

    with pytest.raises(ValueError, match="only valid when requesting 2nd derivatives"):
        RectilinearGridGradient.dispatch(
            torch.randn(16, device=device, dtype=torch.float32),
            (torch.linspace(0.0, 1.0, 16, device=device, dtype=torch.float32),),
            periods=1.0,
            derivative_orders=1,
            include_mixed=True,
            implementation="torch",
        )

    with pytest.raises(
        ValueError, match="mixed derivatives require at least 2D inputs"
    ):
        RectilinearGridGradient.dispatch(
            torch.randn(16, device=device, dtype=torch.float32),
            (torch.linspace(0.0, 1.0, 16, device=device, dtype=torch.float32),),
            periods=1.0,
            derivative_orders=2,
            include_mixed=True,
            implementation="torch",
        )
