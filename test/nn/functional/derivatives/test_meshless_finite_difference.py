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

from __future__ import annotations

import pytest
import torch

from physicsnemo.nn.functional import meshless_fd_derivatives
from physicsnemo.nn.functional.derivatives import MeshlessFDDerivatives
from physicsnemo.nn.functional.derivatives.meshless_finite_difference._torch_impl import (
    meshless_fd_stencil_points_torch,
)


def _spacing_for_dim(dim: int) -> float | tuple[float, ...]:
    """Return representative spacing values by dimensionality."""
    if dim == 1:
        return 0.01
    if dim == 2:
        return (0.01, 0.015)
    return (0.01, 0.015, 0.02)


def _analytic_values(points: torch.Tensor) -> torch.Tensor:
    """Evaluate a smooth two-channel scalar field at arbitrary points."""
    x = points[..., 0]
    if points.shape[-1] == 1:
        ch0 = torch.sin(2.0 * x) + 0.3 * x.square()
        ch1 = torch.cos(1.5 * x) - 0.1 * x
        return torch.stack((ch0, ch1), dim=-1)

    y = points[..., 1]
    if points.shape[-1] == 2:
        ch0 = torch.sin(1.4 * x) * torch.cos(0.7 * y) + 0.2 * x * y
        ch1 = x.square() + y.pow(3)
        return torch.stack((ch0, ch1), dim=-1)

    z = points[..., 2]
    ch0 = torch.sin(1.2 * x) * torch.cos(0.8 * y) * torch.sin(0.6 * z) + 0.1 * x * y * z
    ch1 = x.square() + 0.5 * y.square() - z
    return torch.stack((ch0, ch1), dim=-1)


def _analytic_first_derivatives(points: torch.Tensor) -> torch.Tensor:
    """Evaluate analytic first derivatives for the two-channel test field."""
    x = points[:, 0]
    if points.shape[1] == 1:
        dx = torch.stack(
            (2.0 * torch.cos(2.0 * x) + 0.6 * x, -1.5 * torch.sin(1.5 * x) - 0.1),
            dim=-1,
        )
        return dx.unsqueeze(0)

    y = points[:, 1]
    if points.shape[1] == 2:
        dfdx = torch.stack(
            (
                1.4 * torch.cos(1.4 * x) * torch.cos(0.7 * y) + 0.2 * y,
                2.0 * x,
            ),
            dim=-1,
        )
        dfdy = torch.stack(
            (
                -0.7 * torch.sin(1.4 * x) * torch.sin(0.7 * y) + 0.2 * x,
                3.0 * y.square(),
            ),
            dim=-1,
        )
        return torch.stack((dfdx, dfdy), dim=0)

    z = points[:, 2]
    dfdx = torch.stack(
        (
            1.2 * torch.cos(1.2 * x) * torch.cos(0.8 * y) * torch.sin(0.6 * z)
            + 0.1 * y * z,
            2.0 * x,
        ),
        dim=-1,
    )
    dfdy = torch.stack(
        (
            -0.8 * torch.sin(1.2 * x) * torch.sin(0.8 * y) * torch.sin(0.6 * z)
            + 0.1 * x * z,
            y,
        ),
        dim=-1,
    )
    dfdz = torch.stack(
        (
            0.6 * torch.sin(1.2 * x) * torch.cos(0.8 * y) * torch.cos(0.6 * z)
            + 0.1 * x * y,
            -torch.ones_like(x),
        ),
        dim=-1,
    )
    return torch.stack((dfdx, dfdy, dfdz), dim=0)


def _analytic_second_derivatives(points: torch.Tensor) -> torch.Tensor:
    """Evaluate analytic second derivatives (pure then mixed axis pairs)."""
    x = points[:, 0]
    if points.shape[1] == 1:
        d2xx = torch.stack(
            (
                -4.0 * torch.sin(2.0 * x) + 0.6,
                -2.25 * torch.cos(1.5 * x),
            ),
            dim=-1,
        )
        return d2xx.unsqueeze(0)

    y = points[:, 1]
    if points.shape[1] == 2:
        d2xx = torch.stack(
            (
                -1.96 * torch.sin(1.4 * x) * torch.cos(0.7 * y),
                2.0 * torch.ones_like(x),
            ),
            dim=-1,
        )
        d2yy = torch.stack(
            (
                -0.49 * torch.sin(1.4 * x) * torch.cos(0.7 * y),
                6.0 * y,
            ),
            dim=-1,
        )
        d2xy = torch.stack(
            (
                -0.98 * torch.cos(1.4 * x) * torch.sin(0.7 * y) + 0.2,
                torch.zeros_like(x),
            ),
            dim=-1,
        )
        return torch.stack((d2xx, d2yy, d2xy), dim=0)

    z = points[:, 2]
    d2xx = torch.stack(
        (
            -1.44 * torch.sin(1.2 * x) * torch.cos(0.8 * y) * torch.sin(0.6 * z),
            2.0 * torch.ones_like(x),
        ),
        dim=-1,
    )
    d2yy = torch.stack(
        (
            -0.64 * torch.sin(1.2 * x) * torch.cos(0.8 * y) * torch.sin(0.6 * z),
            torch.ones_like(x),
        ),
        dim=-1,
    )
    d2zz = torch.stack(
        (
            -0.36 * torch.sin(1.2 * x) * torch.cos(0.8 * y) * torch.sin(0.6 * z),
            torch.zeros_like(x),
        ),
        dim=-1,
    )
    d2xy = torch.stack(
        (
            -0.96 * torch.cos(1.2 * x) * torch.sin(0.8 * y) * torch.sin(0.6 * z)
            + 0.1 * z,
            torch.zeros_like(x),
        ),
        dim=-1,
    )
    d2xz = torch.stack(
        (
            0.72 * torch.cos(1.2 * x) * torch.cos(0.8 * y) * torch.cos(0.6 * z)
            + 0.1 * y,
            torch.zeros_like(x),
        ),
        dim=-1,
    )
    d2yz = torch.stack(
        (
            -0.48 * torch.sin(1.2 * x) * torch.sin(0.8 * y) * torch.cos(0.6 * z)
            + 0.1 * x,
            torch.zeros_like(x),
        ),
        dim=-1,
    )
    return torch.stack((d2xx, d2yy, d2zz, d2xy, d2xz, d2yz), dim=0)


# Validate stencil-point generation for representative dimensions.
@pytest.mark.parametrize("dim", [1, 2, 3])
def test_meshless_fd_derivatives_stencil_points_torch(device: str, dim: int):
    points = torch.rand(32, dim, device=device, dtype=torch.float32)
    spacing = _spacing_for_dim(dim)

    stencil_points = meshless_fd_stencil_points_torch(
        points,
        spacing=spacing,
        include_center=True,
    )

    assert stencil_points.shape == (32, 3**dim, dim)
    center_index = (3**dim) // 2
    torch.testing.assert_close(stencil_points[:, center_index], points)


# Validate meshless first derivatives against analytic derivatives.
@pytest.mark.parametrize("dim", [1, 2, 3])
def test_meshless_fd_derivatives_torch_first_order(device: str, dim: int):
    points = torch.rand(128, dim, device=device, dtype=torch.float32)
    spacing = _spacing_for_dim(dim)
    stencil_points = meshless_fd_stencil_points_torch(points, spacing=spacing)
    stencil_values = _analytic_values(stencil_points)
    expected = _analytic_first_derivatives(points)

    output = MeshlessFDDerivatives.dispatch(
        stencil_values,
        spacing=spacing,
        derivative_orders=1,
        include_mixed=False,
        implementation="torch",
    )
    torch.testing.assert_close(output, expected, atol=5e-3, rtol=5e-3)


# Validate meshless second derivatives, including mixed terms, against analytics.
@pytest.mark.parametrize("dim", [1, 2, 3])
def test_meshless_fd_derivatives_torch_second_order(device: str, dim: int):
    points = torch.rand(128, dim, device=device, dtype=torch.float32)
    spacing = _spacing_for_dim(dim)
    stencil_points = meshless_fd_stencil_points_torch(points, spacing=spacing)
    stencil_values = _analytic_values(stencil_points)
    expected = _analytic_second_derivatives(points)

    output = MeshlessFDDerivatives.dispatch(
        stencil_values,
        spacing=spacing,
        derivative_orders=2,
        include_mixed=(dim > 1),
        implementation="torch",
    )
    torch.testing.assert_close(output, expected, atol=7e-3, rtol=7e-3)


# Validate unified derivative-order requests concatenate outputs deterministically.
@pytest.mark.parametrize("dim", [1, 2, 3])
def test_meshless_fd_derivatives_torch_combined_orders(device: str, dim: int):
    points = torch.rand(128, dim, device=device, dtype=torch.float32)
    spacing = _spacing_for_dim(dim)
    stencil_points = meshless_fd_stencil_points_torch(points, spacing=spacing)
    stencil_values = _analytic_values(stencil_points)
    expected_first = _analytic_first_derivatives(points)
    expected_second = _analytic_second_derivatives(points)

    output = MeshlessFDDerivatives.dispatch(
        stencil_values,
        spacing=spacing,
        derivative_orders=(1, 2),
        include_mixed=(dim > 1),
        implementation="torch",
    )
    expected = torch.cat((expected_first, expected_second), dim=0)
    torch.testing.assert_close(output, expected, atol=7e-3, rtol=7e-3)


# Validate benchmark input generation contract for derivative forward inputs.
def test_meshless_fd_derivatives_make_inputs_forward(device: str):
    label, args, kwargs = next(
        iter(MeshlessFDDerivatives.make_inputs_forward(device=device))
    )
    assert isinstance(label, str)
    assert isinstance(args, tuple)
    assert isinstance(kwargs, dict)

    output = MeshlessFDDerivatives.dispatch(*args, implementation="torch", **kwargs)
    assert output.ndim in (2, 3)


# Validate benchmark input generation contract for derivative backward inputs.
def test_meshless_fd_derivatives_make_inputs_backward(device: str):
    label, args, kwargs = next(
        iter(MeshlessFDDerivatives.make_inputs_backward(device=device))
    )
    assert isinstance(label, str)
    assert isinstance(args, tuple)
    assert isinstance(kwargs, dict)

    stencil_values = args[0]
    assert stencil_values.requires_grad

    output = MeshlessFDDerivatives.dispatch(*args, implementation="torch", **kwargs)
    output.square().mean().backward()
    assert stencil_values.grad is not None


# Validate exported API and error handling branches for meshless FD functionals.
def test_meshless_fd_derivatives_error_handling(device: str):
    points = torch.rand(16, 2, device=device, dtype=torch.float32)
    stencil_points = meshless_fd_stencil_points_torch(points, spacing=(0.01, 0.02))
    assert stencil_points.shape == (16, 9, 2)

    values = _analytic_values(stencil_points)
    derivs = meshless_fd_derivatives(values, spacing=(0.01, 0.02), derivative_orders=1)
    assert derivs.shape == (2, 16, 2)

    with pytest.raises(ValueError, match="shape"):
        meshless_fd_stencil_points_torch(
            torch.rand(16, device=device, dtype=torch.float32)
        )

    with pytest.raises(TypeError, match="floating-point"):
        meshless_fd_stencil_points_torch(
            torch.ones(16, 2, device=device, dtype=torch.int32)
        )

    with pytest.raises(ValueError, match="must have 2 entries"):
        meshless_fd_stencil_points_torch(
            torch.rand(16, 2, device=device, dtype=torch.float32),
            spacing=(0.1,),
        )

    with pytest.raises(ValueError, match="strictly positive"):
        meshless_fd_stencil_points_torch(
            torch.rand(16, 2, device=device, dtype=torch.float32),
            spacing=(0.1, 0.0),
        )

    with pytest.raises(ValueError, match="must have shape"):
        MeshlessFDDerivatives.dispatch(
            torch.rand(16, device=device, dtype=torch.float32),
            implementation="torch",
        )

    with pytest.raises(TypeError, match="floating-point"):
        MeshlessFDDerivatives.dispatch(
            torch.ones(16, 9, device=device, dtype=torch.int32),
            implementation="torch",
        )

    with pytest.raises(ValueError, match="must be 3, 9, or 27"):
        MeshlessFDDerivatives.dispatch(
            torch.rand(16, 5, device=device, dtype=torch.float32),
            implementation="torch",
        )

    with pytest.raises(ValueError, match="supports derivative orders"):
        MeshlessFDDerivatives.dispatch(
            torch.rand(16, 9, device=device, dtype=torch.float32),
            derivative_orders=3,
            implementation="torch",
        )

    with pytest.raises(ValueError, match="only valid when requesting 2nd derivatives"):
        MeshlessFDDerivatives.dispatch(
            torch.rand(16, 9, device=device, dtype=torch.float32),
            derivative_orders=1,
            include_mixed=True,
            implementation="torch",
        )

    with pytest.raises(ValueError, match="must have 2 entries"):
        MeshlessFDDerivatives.dispatch(
            torch.rand(16, 9, device=device, dtype=torch.float32),
            spacing=(0.1,),
            implementation="torch",
        )

    with pytest.raises(ValueError, match="mixed derivatives require at least 2D"):
        MeshlessFDDerivatives.dispatch(
            torch.rand(16, 3, device=device, dtype=torch.float32),
            derivative_orders=2,
            include_mixed=True,
            implementation="torch",
        )
