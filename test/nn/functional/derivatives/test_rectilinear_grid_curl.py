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

from physicsnemo.nn.functional import rectilinear_grid_curl
from physicsnemo.nn.functional.derivatives import RectilinearGridCurl
from physicsnemo.nn.functional.derivatives.rectilinear_grid_curl._torch_impl import (
    rectilinear_grid_curl_torch,
)
from test.conftest import requires_module
from test.nn.functional._parity_utils import clone_case


def _make_periodic_case(device: str, dims: int):
    torch_device = torch.device(device)
    if dims == 2:
        n0, n1 = 320, 256
        s0 = torch.linspace(0.0, 1.0, n0 + 1, device=torch_device)[:-1]
        s1 = torch.linspace(0.0, 1.0, n1 + 1, device=torch_device)[:-1]
        x0 = s0 + 0.04 * torch.sin(2.0 * torch.pi * s0)
        x1 = s1 + 0.03 * torch.sin(2.0 * torch.pi * s1)
        xx, yy = torch.meshgrid(x0, x1, indexing="ij")
        vector_field = torch.stack(
            (
                0.5 * torch.cos(2.0 * torch.pi * yy),
                torch.sin(2.0 * torch.pi * xx),
            ),
            dim=0,
        )
        expected = (2.0 * torch.pi) * torch.cos(
            2.0 * torch.pi * xx
        ) + torch.pi * torch.sin(2.0 * torch.pi * yy)
        return (
            vector_field,
            (x0.to(torch.float32), x1.to(torch.float32)),
            (1.0, 1.0),
            expected,
        )

    n0, n1, n2 = 120, 96, 80
    s0 = torch.linspace(0.0, 1.0, n0 + 1, device=torch_device)[:-1]
    s1 = torch.linspace(0.0, 1.0, n1 + 1, device=torch_device)[:-1]
    s2 = torch.linspace(0.0, 1.0, n2 + 1, device=torch_device)[:-1]
    x0 = s0 + 0.04 * torch.sin(2.0 * torch.pi * s0)
    x1 = s1 + 0.03 * torch.sin(2.0 * torch.pi * s1)
    x2 = s2 + 0.02 * torch.sin(2.0 * torch.pi * s2)
    xx, yy, zz = torch.meshgrid(x0, x1, x2, indexing="ij")
    vector_field = torch.stack(
        (
            torch.sin(2.0 * torch.pi * yy),
            torch.sin(2.0 * torch.pi * zz),
            torch.sin(2.0 * torch.pi * xx),
        ),
        dim=0,
    )
    expected = torch.stack(
        (
            -(2.0 * torch.pi) * torch.cos(2.0 * torch.pi * zz),
            -(2.0 * torch.pi) * torch.cos(2.0 * torch.pi * xx),
            -(2.0 * torch.pi) * torch.cos(2.0 * torch.pi * yy),
        ),
        dim=0,
    )
    return (
        vector_field,
        (x0.to(torch.float32), x1.to(torch.float32), x2.to(torch.float32)),
        (1.0, 1.0, 1.0),
        expected,
    )


@pytest.mark.parametrize("dims", [2, 3])
def test_rectilinear_grid_curl_torch(device: str, dims: int):
    vector_field, coordinates, periods, expected = _make_periodic_case(device, dims)
    output = rectilinear_grid_curl_torch(
        vector_field.to(torch.float32),
        coordinates,
        periods=periods,
    )
    torch.testing.assert_close(output, expected, atol=5e-2, rtol=5e-2)


@requires_module("warp")
@pytest.mark.parametrize("dims", [2, 3])
def test_rectilinear_grid_curl_warp(device: str, dims: int):
    vector_field, coordinates, periods, expected = _make_periodic_case(device, dims)
    output = RectilinearGridCurl.dispatch(
        vector_field.to(torch.float32),
        coordinates,
        periods=periods,
        implementation="warp",
    )
    torch.testing.assert_close(output, expected, atol=6e-2, rtol=6e-2)


@requires_module("warp")
def test_rectilinear_grid_curl_backend_forward_parity(device: str):
    for _label, args, kwargs in RectilinearGridCurl.make_inputs_forward(device=device):
        args_torch, kwargs_torch = clone_case(args, kwargs)
        args_warp, kwargs_warp = clone_case(args, kwargs)

        out_torch = RectilinearGridCurl.dispatch(
            *args_torch,
            implementation="torch",
            **kwargs_torch,
        )
        out_warp = RectilinearGridCurl.dispatch(
            *args_warp,
            implementation="warp",
            **kwargs_warp,
        )
        RectilinearGridCurl.compare_forward(out_warp, out_torch)


@requires_module("warp")
def test_rectilinear_grid_curl_backend_backward_parity(device: str):
    for _label, args, kwargs in RectilinearGridCurl.make_inputs_backward(device=device):
        args_torch, kwargs_torch = clone_case(args, kwargs)
        args_warp, kwargs_warp = clone_case(args, kwargs)

        out_torch = RectilinearGridCurl.dispatch(
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

        out_warp = RectilinearGridCurl.dispatch(
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
        RectilinearGridCurl.compare_backward(grad_warp, grad_torch)


def test_rectilinear_grid_curl_make_inputs_forward(device: str):
    for label, args, kwargs in RectilinearGridCurl.make_inputs_forward(device=device):
        assert isinstance(label, str)
        vector_field, coordinates = args
        grid_ndim = vector_field.ndim - 1
        assert vector_field.shape[0] == grid_ndim
        assert len(coordinates) == grid_ndim

        output = RectilinearGridCurl.dispatch(
            *args,
            implementation="torch",
            **kwargs,
        )
        expected_shape = (
            vector_field.shape[1:] if grid_ndim == 2 else vector_field.shape
        )
        assert output.shape == expected_shape


def test_rectilinear_grid_curl_make_inputs_backward(device: str):
    _label, args, kwargs = next(
        iter(RectilinearGridCurl.make_inputs_backward(device=device))
    )
    vector_field = args[0]
    assert vector_field.requires_grad

    output = RectilinearGridCurl.dispatch(
        *args,
        implementation="torch",
        **kwargs,
    )
    output.square().mean().backward()
    assert vector_field.grad is not None


def test_rectilinear_grid_curl_error_handling(device: str):
    x0 = torch.linspace(0.0, 1.0, 17, device=device)[:-1].to(torch.float32)
    x1 = torch.linspace(0.0, 1.0, 19, device=device)[:-1].to(torch.float32)
    xx, yy = torch.meshgrid(x0, x1, indexing="ij")
    vector_field = torch.stack(
        (
            0.5 * torch.cos(2.0 * torch.pi * yy),
            torch.sin(2.0 * torch.pi * xx),
        ),
        dim=0,
    )

    output = rectilinear_grid_curl(vector_field, (x0, x1), periods=(1.0, 1.0))
    assert output.shape == (16, 18)

    with pytest.raises(ValueError, match="expects a 2D or 3D"):
        rectilinear_grid_curl_torch(
            torch.randn(1, 16, device=device, dtype=torch.float32),
            (x0,),
            periods=1.0,
        )

    with pytest.raises(ValueError, match="vector_field.shape\\[0\\] must match"):
        rectilinear_grid_curl_torch(
            torch.randn(3, 16, 18, device=device, dtype=torch.float32),
            (x0, x1),
            periods=(1.0, 1.0),
        )
