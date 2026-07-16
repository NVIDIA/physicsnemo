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

from physicsnemo.nn.functional import uniform_grid_laplacian
from physicsnemo.nn.functional.derivatives import UniformGridLaplacian
from test.conftest import requires_module
from test.nn.functional._parity_utils import clone_case


def _make_periodic_scalar_field(device: str, dims: int):
    torch_device = torch.device(device)
    wave_number = 2.0 * torch.pi

    if dims == 1:
        n0 = 384
        x0 = torch.arange(n0, device=torch_device, dtype=torch.float32) / float(n0)
        field = torch.sin(wave_number * x0)
        spacing = (1.0 / float(n0),)
        expected = -(wave_number**2) * torch.sin(wave_number * x0)
        return field, spacing, expected

    if dims == 2:
        n0, n1 = 128, 112
        x0 = torch.arange(n0, device=torch_device, dtype=torch.float32) / float(n0)
        x1 = torch.arange(n1, device=torch_device, dtype=torch.float32) / float(n1)
        xx, yy = torch.meshgrid(x0, x1, indexing="ij")
        field = torch.sin(wave_number * xx) + 0.5 * torch.cos(wave_number * yy)
        spacing = (1.0 / float(n0), 1.0 / float(n1))
        expected = -(wave_number**2) * (
            torch.sin(wave_number * xx) + 0.5 * torch.cos(wave_number * yy)
        )
        return field, spacing, expected

    n0, n1, n2 = 56, 48, 40
    x0 = torch.arange(n0, device=torch_device, dtype=torch.float32) / float(n0)
    x1 = torch.arange(n1, device=torch_device, dtype=torch.float32) / float(n1)
    x2 = torch.arange(n2, device=torch_device, dtype=torch.float32) / float(n2)
    xx, yy, zz = torch.meshgrid(x0, x1, x2, indexing="ij")
    field = (
        torch.sin(wave_number * xx)
        + 0.5 * torch.cos(wave_number * yy)
        + 0.25 * torch.sin(wave_number * zz)
    )
    spacing = (1.0 / float(n0), 1.0 / float(n1), 1.0 / float(n2))
    expected = -(wave_number**2) * field
    return field, spacing, expected


@pytest.mark.parametrize("dims", [1, 2, 3])
@pytest.mark.parametrize("order", [2, 4])
def test_uniform_grid_laplacian_torch(device: str, dims: int, order: int):
    field, spacing, expected = _make_periodic_scalar_field(device, dims)
    output = UniformGridLaplacian.dispatch(
        field,
        spacing=spacing,
        order=order,
        implementation="torch",
    )
    torch.testing.assert_close(output, expected, atol=2e-1, rtol=8e-2)


def test_uniform_grid_laplacian_public_function(device: str):
    field, spacing, expected = _make_periodic_scalar_field(device, dims=2)
    output = uniform_grid_laplacian(
        field,
        spacing=spacing,
        order=2,
        implementation="torch",
    )
    torch.testing.assert_close(output, expected, atol=2e-1, rtol=8e-2)


@requires_module("warp")
def test_uniform_grid_laplacian_backend_forward_parity(device: str):
    for _label, args, kwargs in UniformGridLaplacian.make_inputs_forward(device=device):
        args_torch, kwargs_torch = clone_case(args, kwargs)
        args_warp, kwargs_warp = clone_case(args, kwargs)

        out_torch = UniformGridLaplacian.dispatch(
            *args_torch,
            implementation="torch",
            **kwargs_torch,
        )
        out_warp = UniformGridLaplacian.dispatch(
            *args_warp,
            implementation="warp",
            **kwargs_warp,
        )
        UniformGridLaplacian.compare_forward(out_warp, out_torch)


def test_uniform_grid_laplacian_compare_forward_contract(device: str):
    field, spacing, _expected = _make_periodic_scalar_field(device, dims=2)
    output = UniformGridLaplacian.dispatch(
        field,
        spacing=spacing,
        order=2,
        implementation="torch",
    )
    UniformGridLaplacian.compare_forward(output, output.detach().clone())


@requires_module("warp")
def test_uniform_grid_laplacian_backend_backward_parity(device: str):
    for _label, args, kwargs in UniformGridLaplacian.make_inputs_backward(
        device=device
    ):
        args_torch, kwargs_torch = clone_case(args, kwargs)
        args_warp, kwargs_warp = clone_case(args, kwargs)

        out_torch = UniformGridLaplacian.dispatch(
            *args_torch,
            implementation="torch",
            **kwargs_torch,
        )
        out_torch.square().mean().backward()
        grad_torch = args_torch[0].grad
        assert grad_torch is not None

        out_warp = UniformGridLaplacian.dispatch(
            *args_warp,
            implementation="warp",
            **kwargs_warp,
        )
        out_warp.square().mean().backward()
        grad_warp = args_warp[0].grad
        assert grad_warp is not None

        UniformGridLaplacian.compare_backward(grad_warp, grad_torch)


def test_uniform_grid_laplacian_compare_backward_contract(device: str):
    field, spacing, _expected = _make_periodic_scalar_field(device, dims=2)
    field = field.detach().clone().requires_grad_(True)
    output = UniformGridLaplacian.dispatch(
        field,
        spacing=spacing,
        order=2,
        implementation="torch",
    )
    output.square().mean().backward()
    assert field.grad is not None
    UniformGridLaplacian.compare_backward(
        field.grad,
        field.grad.detach().clone(),
    )


def test_uniform_grid_laplacian_error_handling(device: str):
    with pytest.raises(TypeError, match="floating-point"):
        UniformGridLaplacian.dispatch(
            torch.ones((8, 8), device=device, dtype=torch.int64),
            implementation="torch",
        )

    with pytest.raises(ValueError, match="1D-3D"):
        UniformGridLaplacian.dispatch(
            torch.ones((2, 2, 2, 2), device=device),
            implementation="torch",
        )


@requires_module("warp")
def test_uniform_grid_laplacian_warp_rejects_non_integer_order(device: str):
    with pytest.raises(TypeError, match="order must be an integer"):
        UniformGridLaplacian.dispatch(
            torch.ones((8, 8), device=device),
            order=2.5,
            implementation="warp",
        )


def test_uniform_grid_laplacian_make_inputs_forward(device: str):
    label, args, kwargs = next(
        iter(UniformGridLaplacian.make_inputs_forward(device=device))
    )
    assert isinstance(label, str)
    assert isinstance(args, tuple)
    assert isinstance(kwargs, dict)

    field = args[0]
    assert field.ndim in (1, 2, 3)

    output = UniformGridLaplacian.dispatch(
        *args,
        implementation="torch",
        **kwargs,
    )
    assert output.shape == field.shape


def test_uniform_grid_laplacian_make_inputs_backward(device: str):
    _label, args, kwargs = next(
        iter(UniformGridLaplacian.make_inputs_backward(device=device))
    )
    field = args[0]
    assert field.requires_grad

    output = UniformGridLaplacian.dispatch(
        *args,
        implementation="torch",
        **kwargs,
    )
    output.square().mean().backward()
    assert field.grad is not None
