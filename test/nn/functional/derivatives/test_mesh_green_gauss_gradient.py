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

from physicsnemo.nn.functional import mesh_green_gauss_gradient
from physicsnemo.nn.functional.derivatives import MeshGreenGaussGradient
from physicsnemo.nn.functional.derivatives.mesh_green_gauss_gradient.utils import (
    build_neighbors,
)
from test.conftest import requires_module
from test.nn.functional._parity_utils import clone_case


# Build a deterministic structured triangular mesh.
def _build_case(device: str, nx: int = 36, ny: int = 32):
    torch_device = torch.device(device)
    x = torch.linspace(0.0, 1.0, nx, device=torch_device, dtype=torch.float32)
    y = torch.linspace(0.0, 1.0, ny, device=torch_device, dtype=torch.float32)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    points = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=-1)

    cells = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            p00 = i * ny + j
            p10 = (i + 1) * ny + j
            p01 = i * ny + (j + 1)
            p11 = (i + 1) * ny + (j + 1)
            cells.append((p00, p10, p11))
            cells.append((p00, p11, p01))
    cells = torch.tensor(cells, device=torch_device, dtype=torch.int64)
    return points.contiguous(), cells.contiguous()


def _build_case_3d(device: str, nx: int = 12, ny: int = 10, nz: int = 8):
    torch_device = torch.device(device)
    x = torch.linspace(0.0, 1.0, nx, device=torch_device, dtype=torch.float32)
    y = torch.linspace(0.0, 1.0, ny, device=torch_device, dtype=torch.float32)
    z = torch.linspace(0.0, 1.0, nz, device=torch_device, dtype=torch.float32)
    xx, yy, zz = torch.meshgrid(x, y, z, indexing="ij")
    points = torch.stack((xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)), dim=-1)

    def _idx(i: int, j: int, k: int) -> int:
        return (i * ny + j) * nz + k

    cells = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            for k in range(nz - 1):
                p000 = _idx(i, j, k)
                p100 = _idx(i + 1, j, k)
                p010 = _idx(i, j + 1, k)
                p110 = _idx(i + 1, j + 1, k)
                p001 = _idx(i, j, k + 1)
                p101 = _idx(i + 1, j, k + 1)
                p011 = _idx(i, j + 1, k + 1)
                p111 = _idx(i + 1, j + 1, k + 1)

                cells.append((p000, p100, p110, p111))
                cells.append((p000, p110, p010, p111))
                cells.append((p000, p010, p011, p111))
                cells.append((p000, p011, p001, p111))
                cells.append((p000, p001, p101, p111))
                cells.append((p000, p101, p100, p111))

    cells = torch.tensor(cells, device=torch_device, dtype=torch.int64)
    return points.contiguous(), cells.contiguous()


# Validate torch Green-Gauss reconstruction on a linear field.
def test_mesh_green_gauss_gradient_torch(device: str):
    points, cells = _build_case(device=device, nx=40, ny=34)
    neighbors = build_neighbors(cells)
    centroids = points[cells].mean(dim=1)
    coeff = torch.tensor([2.0, -3.0], device=points.device, dtype=torch.float32)
    values = (centroids * coeff).sum(dim=-1)

    output = MeshGreenGaussGradient.dispatch(
        points,
        cells,
        neighbors,
        values,
        implementation="torch",
    )
    interior = (neighbors >= 0).all(dim=1)
    expected = coeff.view(1, -1).expand(interior.sum(), -1)
    torch.testing.assert_close(output[interior], expected, atol=5e-2, rtol=5e-2)


def test_mesh_green_gauss_gradient_torch_3d(device: str):
    points, cells = _build_case_3d(device=device, nx=11, ny=9, nz=7)
    neighbors = build_neighbors(cells)
    values = torch.sin(points[cells].mean(dim=1).sum(dim=-1))

    output = MeshGreenGaussGradient.dispatch(
        points,
        cells,
        neighbors,
        values,
        implementation="torch",
    )
    assert output.shape == (cells.shape[0], points.shape[1])
    assert torch.isfinite(output).all()


# Validate warp Green-Gauss reconstruction on a linear field.
@requires_module("warp")
def test_mesh_green_gauss_gradient_warp(device: str):
    points, cells = _build_case(device=device, nx=40, ny=34)
    neighbors = build_neighbors(cells)
    centroids = points[cells].mean(dim=1)
    coeff = torch.tensor([2.0, -3.0], device=points.device, dtype=torch.float32)
    values = (centroids * coeff).sum(dim=-1)

    output = MeshGreenGaussGradient.dispatch(
        points,
        cells,
        neighbors,
        values,
        implementation="warp",
    )
    interior = (neighbors >= 0).all(dim=1)
    expected = coeff.view(1, -1).expand(interior.sum(), -1)
    torch.testing.assert_close(output[interior], expected, atol=5e-2, rtol=5e-2)


@requires_module("warp")
def test_mesh_green_gauss_gradient_warp_3d(device: str):
    points, cells = _build_case_3d(device=device, nx=11, ny=9, nz=7)
    neighbors = build_neighbors(cells)
    values = torch.sin(points[cells].mean(dim=1).sum(dim=-1))

    output_torch = MeshGreenGaussGradient.dispatch(
        points,
        cells,
        neighbors,
        values,
        implementation="torch",
    )

    output_warp = MeshGreenGaussGradient.dispatch(
        points,
        cells,
        neighbors,
        values,
        implementation="warp",
    )
    MeshGreenGaussGradient.compare_forward(output_warp, output_torch)


# Validate warp backend forward parity against torch across benchmark cases.
@requires_module("warp")
def test_mesh_green_gauss_gradient_backend_forward_parity(device: str):
    for _label, args, kwargs in MeshGreenGaussGradient.make_inputs_forward(
        device=device
    ):
        args_torch, kwargs_torch = clone_case(args, kwargs)
        args_warp, kwargs_warp = clone_case(args, kwargs)

        out_torch = MeshGreenGaussGradient.dispatch(
            *args_torch,
            implementation="torch",
            **kwargs_torch,
        )
        out_warp = MeshGreenGaussGradient.dispatch(
            *args_warp,
            implementation="warp",
            **kwargs_warp,
        )
        MeshGreenGaussGradient.compare_forward(out_warp, out_torch)


# Validate warp backend backward parity against torch on value gradients.
@requires_module("warp")
def test_mesh_green_gauss_gradient_backend_backward_parity(device: str):
    for _label, args, kwargs in MeshGreenGaussGradient.make_inputs_backward(
        device=device
    ):
        args_torch, kwargs_torch = clone_case(args, kwargs)
        args_warp, kwargs_warp = clone_case(args, kwargs)

        out_torch = MeshGreenGaussGradient.dispatch(
            *args_torch,
            implementation="torch",
            **kwargs_torch,
        )
        out_torch.square().mean().backward()
        grad_torch = args_torch[3].grad
        assert grad_torch is not None

        out_warp = MeshGreenGaussGradient.dispatch(
            *args_warp,
            implementation="warp",
            **kwargs_warp,
        )
        out_warp.square().mean().backward()
        grad_warp = args_warp[3].grad
        assert grad_warp is not None

        MeshGreenGaussGradient.compare_backward(grad_warp, grad_torch)


@requires_module("warp")
def test_mesh_green_gauss_gradient_warp_supports_point_gradients(device: str):
    points, cells = _build_case(device=device, nx=26, ny=22)
    neighbors = build_neighbors(cells)
    centroids = points[cells].mean(dim=1)
    base_values = (
        torch.sin(2.0 * torch.pi * centroids[:, 0])
        + 0.25 * torch.cos(2.0 * torch.pi * centroids[:, 1])
    ).to(torch.float32)

    points_warp = points.detach().clone().requires_grad_(True)
    values_warp = base_values.detach().clone().requires_grad_(True)
    out_warp = MeshGreenGaussGradient.dispatch(
        points_warp,
        cells,
        neighbors,
        values_warp,
        implementation="warp",
    )
    out_warp.square().mean().backward()
    grad_points_warp = points_warp.grad
    grad_values_warp = values_warp.grad
    assert grad_points_warp is not None
    assert grad_values_warp is not None
    assert torch.isfinite(grad_points_warp).all()
    assert torch.isfinite(grad_values_warp).all()
    assert torch.any(grad_points_warp != 0.0)


# Validate benchmark input generation contract for forward inputs.
def test_mesh_green_gauss_gradient_make_inputs_forward(device: str):
    label, args, kwargs = next(
        iter(MeshGreenGaussGradient.make_inputs_forward(device=device))
    )
    assert isinstance(label, str)
    assert isinstance(args, tuple)
    assert isinstance(kwargs, dict)

    points, cells, neighbors, values = args
    assert points.ndim == 2
    assert cells.ndim == 2
    assert neighbors.shape == (cells.shape[0], cells.shape[1])
    assert values.shape[0] == cells.shape[0]

    output = MeshGreenGaussGradient.dispatch(
        *args,
        implementation="torch",
        **kwargs,
    )
    assert output.shape[0] == cells.shape[0]
    assert output.shape[1] == points.shape[1]


# Validate benchmark input generation contract for backward inputs.
def test_mesh_green_gauss_gradient_make_inputs_backward(device: str):
    label, args, kwargs = next(
        iter(MeshGreenGaussGradient.make_inputs_backward(device=device))
    )
    assert isinstance(label, str)
    assert isinstance(args, tuple)
    assert isinstance(kwargs, dict)

    values = args[3]
    assert values.requires_grad

    output = MeshGreenGaussGradient.dispatch(
        *args,
        implementation="torch",
        **kwargs,
    )
    output.square().mean().backward()
    assert values.grad is not None


# Validate compare-forward hook contract.
def test_mesh_green_gauss_gradient_compare_forward_contract(device: str):
    _label, args, kwargs = next(
        iter(MeshGreenGaussGradient.make_inputs_forward(device=device))
    )
    output = MeshGreenGaussGradient.dispatch(*args, implementation="torch", **kwargs)
    reference = output.detach().clone()
    MeshGreenGaussGradient.compare_forward(output, reference)


# Validate compare-backward hook contract.
def test_mesh_green_gauss_gradient_compare_backward_contract(device: str):
    _label, args, kwargs = next(
        iter(MeshGreenGaussGradient.make_inputs_backward(device=device))
    )
    values = args[3]

    output = MeshGreenGaussGradient.dispatch(*args, implementation="torch", **kwargs)
    output.square().mean().backward()

    assert values.grad is not None
    MeshGreenGaussGradient.compare_backward(values.grad, values.grad.detach().clone())


# Validate exported API and input validation paths.
def test_mesh_green_gauss_gradient_error_handling(device: str):
    points, cells = _build_case(device=device, nx=16, ny=14)
    values = torch.randn(cells.shape[0], device=points.device, dtype=torch.float32)

    neighbors = build_neighbors(cells)
    output = mesh_green_gauss_gradient(points, cells, neighbors, values)
    assert output.shape[0] == cells.shape[0]
    assert output.shape[1] == points.shape[1]

    with pytest.raises(ValueError, match="supports dims in"):
        bad_points = torch.randn(
            points.shape[0], 4, device=points.device, dtype=torch.float32
        )
        MeshGreenGaussGradient.dispatch(
            bad_points,
            cells,
            neighbors,
            values,
            implementation="torch",
        )

    with pytest.raises(ValueError, match="must contain 3 vertices"):
        bad_cells = torch.randint(
            0,
            points.shape[0],
            (cells.shape[0], 4),
            device=points.device,
            dtype=torch.int64,
        )
        MeshGreenGaussGradient.dispatch(
            points,
            bad_cells,
            neighbors,
            values,
            implementation="torch",
        )

    with pytest.raises(ValueError, match="leading dimension must match n_cells"):
        MeshGreenGaussGradient.dispatch(
            points,
            cells,
            neighbors,
            values[:-1],
            implementation="torch",
        )

    with pytest.raises(ValueError, match="neighbors shape must match"):
        MeshGreenGaussGradient.dispatch(
            points,
            cells,
            neighbors[:, :-1],
            values,
            implementation="torch",
        )
