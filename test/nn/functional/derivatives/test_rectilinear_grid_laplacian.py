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

from physicsnemo.nn.functional import rectilinear_grid_laplacian
from physicsnemo.nn.functional.derivatives import RectilinearGridLaplacian
from physicsnemo.nn.functional.derivatives.rectilinear_grid_laplacian._torch_impl import (
    rectilinear_grid_laplacian_torch,
)
from test.conftest import requires_module
from test.nn.functional._parity_utils import clone_case


def _make_periodic_case(device: str, dims: int):
    torch_device = torch.device(device)
    if dims == 1:
        n0 = 1024
        x0 = torch.linspace(0.0, 1.0, n0 + 1, device=torch_device)[:-1]
        field = torch.sin(2.0 * torch.pi * x0)
        expected = -((2.0 * torch.pi) ** 2) * torch.sin(2.0 * torch.pi * x0)
        return field, (x0.to(torch.float32),), 1.0, expected

    if dims == 2:
        n0, n1 = 320, 256
        x0 = torch.linspace(0.0, 1.0, n0 + 1, device=torch_device)[:-1]
        x1 = torch.linspace(0.0, 1.0, n1 + 1, device=torch_device)[:-1]
        xx, yy = torch.meshgrid(x0, x1, indexing="ij")
        field = torch.sin(2.0 * torch.pi * xx) + 0.5 * torch.cos(2.0 * torch.pi * yy)
        expected = -((2.0 * torch.pi) ** 2) * torch.sin(2.0 * torch.pi * xx) - 2.0 * (
            torch.pi**2
        ) * torch.cos(2.0 * torch.pi * yy)
        return field, (x0.to(torch.float32), x1.to(torch.float32)), (1.0, 1.0), expected

    n0, n1, n2 = 120, 96, 80
    x0 = torch.linspace(0.0, 1.0, n0 + 1, device=torch_device)[:-1]
    x1 = torch.linspace(0.0, 1.0, n1 + 1, device=torch_device)[:-1]
    x2 = torch.linspace(0.0, 1.0, n2 + 1, device=torch_device)[:-1]
    xx, yy, zz = torch.meshgrid(x0, x1, x2, indexing="ij")
    field = (
        torch.sin(2.0 * torch.pi * xx)
        + 0.5 * torch.cos(2.0 * torch.pi * yy)
        + 0.25 * torch.sin(2.0 * torch.pi * zz)
    )
    expected = (
        -((2.0 * torch.pi) ** 2) * torch.sin(2.0 * torch.pi * xx)
        - 2.0 * (torch.pi**2) * torch.cos(2.0 * torch.pi * yy)
        - (torch.pi**2) * torch.sin(2.0 * torch.pi * zz)
    )
    return (
        field,
        (x0.to(torch.float32), x1.to(torch.float32), x2.to(torch.float32)),
        (1.0, 1.0, 1.0),
        expected,
    )


@pytest.mark.parametrize("dims", [1, 2, 3])
def test_rectilinear_grid_laplacian_torch(device: str, dims: int):
    field, coordinates, periods, expected = _make_periodic_case(device, dims)
    output = rectilinear_grid_laplacian_torch(
        field.to(torch.float32),
        coordinates,
        periods=periods,
    )
    torch.testing.assert_close(output, expected, atol=6e-1, rtol=1e-1)


@requires_module("warp")
@pytest.mark.parametrize("dims", [1, 2, 3])
def test_rectilinear_grid_laplacian_warp(device: str, dims: int):
    field, coordinates, periods, expected = _make_periodic_case(device, dims)
    output = RectilinearGridLaplacian.dispatch(
        field.to(torch.float32),
        coordinates,
        periods=periods,
        implementation="warp",
    )
    torch.testing.assert_close(output, expected, atol=6e-1, rtol=1e-1)


@requires_module("warp")
def test_rectilinear_grid_laplacian_backend_forward_parity(device: str):
    for _label, args, kwargs in RectilinearGridLaplacian.make_inputs_forward(
        device=device
    ):
        args_torch, kwargs_torch = clone_case(args, kwargs)
        args_warp, kwargs_warp = clone_case(args, kwargs)

        out_torch = RectilinearGridLaplacian.dispatch(
            *args_torch,
            implementation="torch",
            **kwargs_torch,
        )
        out_warp = RectilinearGridLaplacian.dispatch(
            *args_warp,
            implementation="warp",
            **kwargs_warp,
        )
        RectilinearGridLaplacian.compare_forward(out_warp, out_torch)


@requires_module("warp")
def test_rectilinear_grid_laplacian_backend_backward_parity(device: str):
    for _label, args, kwargs in RectilinearGridLaplacian.make_inputs_backward(
        device=device
    ):
        args_torch, kwargs_torch = clone_case(args, kwargs)
        args_warp, kwargs_warp = clone_case(args, kwargs)

        out_torch = RectilinearGridLaplacian.dispatch(
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

        out_warp = RectilinearGridLaplacian.dispatch(
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

        assert grad_torch is not None
        assert grad_warp is not None
        RectilinearGridLaplacian.compare_backward(grad_warp, grad_torch)


def test_rectilinear_grid_laplacian_make_inputs_forward(device: str):
    label, args, kwargs = next(
        iter(RectilinearGridLaplacian.make_inputs_forward(device=device))
    )
    assert isinstance(label, str)
    field, coordinates = args
    assert len(coordinates) == field.ndim

    output = RectilinearGridLaplacian.dispatch(
        *args,
        implementation="torch",
        **kwargs,
    )
    assert output.shape == field.shape


def test_rectilinear_grid_laplacian_make_inputs_backward(device: str):
    _label, args, kwargs = next(
        iter(RectilinearGridLaplacian.make_inputs_backward(device=device))
    )
    field = args[0]
    assert field.requires_grad

    output = RectilinearGridLaplacian.dispatch(
        *args,
        implementation="torch",
        **kwargs,
    )
    output.square().mean().backward()
    assert field.grad is not None


def test_rectilinear_grid_laplacian_error_handling(device: str):
    x = torch.linspace(0.0, 1.0, 17, device=device)[:-1].to(torch.float32)
    field = torch.sin(2.0 * torch.pi * x)

    output = rectilinear_grid_laplacian(field, (x,), periods=1.0)
    assert output.shape == (16,)

    with pytest.raises(ValueError, match="supports 1D-3D fields"):
        rectilinear_grid_laplacian_torch(
            torch.randn(2, 2, 2, 2, device=device, dtype=torch.float32),
            (x, x, x, x),
            periods=1.0,
        )

    with pytest.raises(ValueError, match="must contain one axis tensor"):
        rectilinear_grid_laplacian_torch(
            torch.randn(16, 16, device=device, dtype=torch.float32),
            (x,),
            periods=1.0,
        )
