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

from physicsnemo.nn.functional import uniform_grid_curl
from physicsnemo.nn.functional.derivatives import UniformGridCurl
from test.conftest import requires_module
from test.nn.functional._parity_utils import clone_case


def _make_periodic_vector_field(device: str, dims: int):
    torch_device = torch.device(device)
    wave_number = 2.0 * torch.pi

    if dims == 2:
        n0, n1 = 160, 144
        x0 = torch.arange(n0, device=torch_device, dtype=torch.float32) / float(n0)
        x1 = torch.arange(n1, device=torch_device, dtype=torch.float32) / float(n1)
        xx, yy = torch.meshgrid(x0, x1, indexing="ij")
        vector_field = torch.stack(
            (
                torch.sin(wave_number * yy),
                torch.cos(wave_number * xx),
            ),
            dim=0,
        )
        spacing = (1.0 / float(n0), 1.0 / float(n1))
        expected = -wave_number * torch.sin(wave_number * xx) - wave_number * torch.cos(
            wave_number * yy
        )
        return vector_field, spacing, expected

    n0, n1, n2 = 64, 56, 48
    x0 = torch.arange(n0, device=torch_device, dtype=torch.float32) / float(n0)
    x1 = torch.arange(n1, device=torch_device, dtype=torch.float32) / float(n1)
    x2 = torch.arange(n2, device=torch_device, dtype=torch.float32) / float(n2)
    xx, yy, zz = torch.meshgrid(x0, x1, x2, indexing="ij")
    vector_field = torch.stack(
        (
            torch.sin(wave_number * yy),
            torch.cos(wave_number * zz),
            torch.sin(wave_number * xx),
        ),
        dim=0,
    )
    spacing = (1.0 / float(n0), 1.0 / float(n1), 1.0 / float(n2))
    expected = torch.stack(
        (
            wave_number * torch.sin(wave_number * zz),
            -wave_number * torch.cos(wave_number * xx),
            -wave_number * torch.cos(wave_number * yy),
        ),
        dim=0,
    )
    return vector_field, spacing, expected


@pytest.mark.parametrize("dims", [2, 3])
@pytest.mark.parametrize("order", [2, 4])
def test_uniform_grid_curl_torch(device: str, dims: int, order: int):
    vector_field, spacing, expected = _make_periodic_vector_field(device, dims)
    output = UniformGridCurl.dispatch(
        vector_field,
        spacing=spacing,
        order=order,
        implementation="torch",
    )
    torch.testing.assert_close(output, expected, atol=8e-2, rtol=8e-2)


def test_uniform_grid_curl_public_function(device: str):
    vector_field, spacing, expected = _make_periodic_vector_field(device, dims=2)
    output = uniform_grid_curl(
        vector_field,
        spacing=spacing,
        order=2,
        implementation="torch",
    )
    torch.testing.assert_close(output, expected, atol=8e-2, rtol=8e-2)


@requires_module("warp")
def test_uniform_grid_curl_backend_forward_parity(device: str):
    for _label, args, kwargs in UniformGridCurl.make_inputs_forward(device=device):
        args_torch, kwargs_torch = clone_case(args, kwargs)
        args_warp, kwargs_warp = clone_case(args, kwargs)

        out_torch = UniformGridCurl.dispatch(
            *args_torch,
            implementation="torch",
            **kwargs_torch,
        )
        out_warp = UniformGridCurl.dispatch(
            *args_warp,
            implementation="warp",
            **kwargs_warp,
        )
        UniformGridCurl.compare_forward(out_warp, out_torch)


def test_uniform_grid_curl_compare_forward_contract(device: str):
    vector_field, spacing, _expected = _make_periodic_vector_field(device, dims=2)
    output = UniformGridCurl.dispatch(
        vector_field,
        spacing=spacing,
        order=2,
        implementation="torch",
    )
    UniformGridCurl.compare_forward(output, output.detach().clone())


@requires_module("warp")
def test_uniform_grid_curl_backend_backward_parity(device: str):
    for _label, args, kwargs in UniformGridCurl.make_inputs_backward(device=device):
        args_torch, kwargs_torch = clone_case(args, kwargs)
        args_warp, kwargs_warp = clone_case(args, kwargs)

        out_torch = UniformGridCurl.dispatch(
            *args_torch,
            implementation="torch",
            **kwargs_torch,
        )
        out_torch.square().mean().backward()
        grad_torch = args_torch[0].grad
        assert grad_torch is not None

        out_warp = UniformGridCurl.dispatch(
            *args_warp,
            implementation="warp",
            **kwargs_warp,
        )
        out_warp.square().mean().backward()
        grad_warp = args_warp[0].grad
        assert grad_warp is not None

        UniformGridCurl.compare_backward(grad_warp, grad_torch)


def test_uniform_grid_curl_compare_backward_contract(device: str):
    vector_field, spacing, _expected = _make_periodic_vector_field(device, dims=2)
    vector_field = vector_field.detach().clone().requires_grad_(True)
    output = UniformGridCurl.dispatch(
        vector_field,
        spacing=spacing,
        order=2,
        implementation="torch",
    )
    output.square().mean().backward()
    assert vector_field.grad is not None
    UniformGridCurl.compare_backward(
        vector_field.grad,
        vector_field.grad.detach().clone(),
    )


def test_uniform_grid_curl_error_handling(device: str):
    with pytest.raises(TypeError, match="floating-point"):
        UniformGridCurl.dispatch(
            torch.ones((2, 8, 8), device=device, dtype=torch.int64),
            implementation="torch",
        )

    with pytest.raises(ValueError, match="2D or 3D"):
        UniformGridCurl.dispatch(
            torch.ones((1, 8), device=device),
            implementation="torch",
        )

    with pytest.raises(ValueError, match="shape\\[0\\]"):
        UniformGridCurl.dispatch(
            torch.ones((3, 8, 8), device=device),
            implementation="torch",
        )


@requires_module("warp")
def test_uniform_grid_curl_warp_rejects_non_integer_order(device: str):
    with pytest.raises(TypeError, match="order must be an integer"):
        UniformGridCurl.dispatch(
            torch.ones((2, 8, 8), device=device),
            order=2.5,
            implementation="warp",
        )


def test_uniform_grid_curl_make_inputs_forward(device: str):
    label, args, kwargs = next(iter(UniformGridCurl.make_inputs_forward(device=device)))
    assert isinstance(label, str)
    assert isinstance(args, tuple)
    assert isinstance(kwargs, dict)

    vector_field = args[0]
    assert vector_field.ndim in (3, 4)
    assert vector_field.shape[0] == vector_field.ndim - 1

    output = UniformGridCurl.dispatch(
        *args,
        implementation="torch",
        **kwargs,
    )
    if vector_field.ndim == 3:
        assert output.shape == vector_field.shape[1:]
    else:
        assert output.shape == vector_field.shape


def test_uniform_grid_curl_make_inputs_backward(device: str):
    _label, args, kwargs = next(
        iter(UniformGridCurl.make_inputs_backward(device=device))
    )
    vector_field = args[0]
    assert vector_field.requires_grad

    output = UniformGridCurl.dispatch(
        *args,
        implementation="torch",
        **kwargs,
    )
    output.square().mean().backward()
    assert vector_field.grad is not None
