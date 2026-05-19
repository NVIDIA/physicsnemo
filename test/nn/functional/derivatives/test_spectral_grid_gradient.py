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

from physicsnemo.nn.functional import spectral_grid_gradient
from physicsnemo.nn.functional.derivatives import SpectralGridGradient


def _make_periodic_test_case(
    device: str,
    dim: int,
) -> tuple[torch.Tensor, tuple[float, ...], torch.Tensor, torch.Tensor]:
    """Build periodic fields with analytic first and second derivatives."""
    torch_device = torch.device(device)

    if dim == 1:
        n0 = 512
        l0 = 2.0
        x0 = torch.arange(n0, device=torch_device, dtype=torch.float64) * (l0 / n0)
        k0 = 2.0 * torch.pi / l0

        field = torch.sin(k0 * x0) + 0.25 * torch.cos(2.0 * k0 * x0)
        first = torch.stack(
            [
                k0 * torch.cos(k0 * x0) - 0.5 * k0 * torch.sin(2.0 * k0 * x0),
            ],
            dim=0,
        )
        second = torch.stack(
            [
                -(k0 * k0) * torch.sin(k0 * x0) - (k0 * k0) * torch.cos(2.0 * k0 * x0),
            ],
            dim=0,
        )
        return field, (l0,), first, second

    if dim == 2:
        n0, n1 = 256, 224
        l0, l1 = 2.0, 1.5
        x0 = torch.arange(n0, device=torch_device, dtype=torch.float64) * (l0 / n0)
        x1 = torch.arange(n1, device=torch_device, dtype=torch.float64) * (l1 / n1)
        xx, yy = torch.meshgrid(x0, x1, indexing="ij")
        k0 = 2.0 * torch.pi / l0
        k1 = 2.0 * torch.pi / l1

        ax = k0 * xx + 0.3
        by = k1 * yy - 0.2
        field = torch.sin(ax) * torch.cos(by)

        dfdx = k0 * torch.cos(ax) * torch.cos(by)
        dfdy = -k1 * torch.sin(ax) * torch.sin(by)
        d2fdx2 = -(k0 * k0) * torch.sin(ax) * torch.cos(by)
        d2fdy2 = -(k1 * k1) * torch.sin(ax) * torch.cos(by)
        d2fdxdy = -k0 * k1 * torch.cos(ax) * torch.sin(by)

        first = torch.stack((dfdx, dfdy), dim=0)
        second = torch.stack((d2fdx2, d2fdy2, d2fdxdy), dim=0)
        return field, (l0, l1), first, second

    n0, n1, n2 = 128, 112, 96
    l0, l1, l2 = 2.0, 1.5, 1.25
    x0 = torch.arange(n0, device=torch_device, dtype=torch.float64) * (l0 / n0)
    x1 = torch.arange(n1, device=torch_device, dtype=torch.float64) * (l1 / n1)
    x2 = torch.arange(n2, device=torch_device, dtype=torch.float64) * (l2 / n2)
    xx, yy, zz = torch.meshgrid(x0, x1, x2, indexing="ij")
    k0 = 2.0 * torch.pi / l0
    k1 = 2.0 * torch.pi / l1
    k2 = 2.0 * torch.pi / l2

    ax = k0 * xx + 0.2
    by = k1 * yy - 0.4
    cz = k2 * zz + 0.1
    field = torch.sin(ax) * torch.cos(by) * torch.sin(cz)

    dfdx = k0 * torch.cos(ax) * torch.cos(by) * torch.sin(cz)
    dfdy = -k1 * torch.sin(ax) * torch.sin(by) * torch.sin(cz)
    dfdz = k2 * torch.sin(ax) * torch.cos(by) * torch.cos(cz)

    d2fdx2 = -(k0 * k0) * torch.sin(ax) * torch.cos(by) * torch.sin(cz)
    d2fdy2 = -(k1 * k1) * torch.sin(ax) * torch.cos(by) * torch.sin(cz)
    d2fdz2 = -(k2 * k2) * torch.sin(ax) * torch.cos(by) * torch.sin(cz)
    d2fdxdy = -k0 * k1 * torch.cos(ax) * torch.sin(by) * torch.sin(cz)
    d2fdxdz = k0 * k2 * torch.cos(ax) * torch.cos(by) * torch.cos(cz)
    d2fdydz = -k1 * k2 * torch.sin(ax) * torch.sin(by) * torch.cos(cz)

    first = torch.stack((dfdx, dfdy, dfdz), dim=0)
    second = torch.stack((d2fdx2, d2fdy2, d2fdz2, d2fdxdy, d2fdxdz, d2fdydz), dim=0)
    return field, (l0, l1, l2), first, second


# Validate torch backend first-order derivatives against analytic periodic fields.
@pytest.mark.parametrize("dim", [1, 2, 3])
def test_spectral_grid_gradient_torch_first_order(device: str, dim: int):
    field, lengths, first_expected, _ = _make_periodic_test_case(device=device, dim=dim)
    output = SpectralGridGradient.dispatch(
        field,
        lengths=lengths,
        derivative_orders=1,
        include_mixed=False,
        implementation="torch",
    )
    torch.testing.assert_close(output, first_expected, atol=1e-4, rtol=1e-4)


# Validate torch backend second-order derivatives against analytic periodic fields.
@pytest.mark.parametrize("dim", [1, 2, 3])
def test_spectral_grid_gradient_torch_second_order(device: str, dim: int):
    field, lengths, _, second_expected = _make_periodic_test_case(
        device=device, dim=dim
    )
    output = SpectralGridGradient.dispatch(
        field,
        lengths=lengths,
        derivative_orders=2,
        include_mixed=(dim > 1),
        implementation="torch",
    )
    torch.testing.assert_close(output, second_expected, atol=1e-4, rtol=1e-4)


# Validate unified derivative-order requests concatenate outputs deterministically.
@pytest.mark.parametrize("dim", [1, 2, 3])
def test_spectral_grid_gradient_torch_combined_orders(device: str, dim: int):
    field, lengths, first_expected, second_expected = _make_periodic_test_case(
        device=device, dim=dim
    )
    output = SpectralGridGradient.dispatch(
        field,
        lengths=lengths,
        derivative_orders=(1, 2),
        include_mixed=(dim > 1),
        implementation="torch",
    )
    expected = torch.cat((first_expected, second_expected), dim=0)
    torch.testing.assert_close(output, expected, atol=1e-4, rtol=1e-4)


# Validate benchmark input generation contract for forward inputs.
def test_spectral_grid_gradient_make_inputs_forward(device: str):
    label, args, kwargs = next(
        iter(SpectralGridGradient.make_inputs_forward(device=device))
    )
    assert isinstance(label, str)
    assert isinstance(args, tuple)
    assert isinstance(kwargs, dict)

    field = args[0]
    output = SpectralGridGradient.dispatch(*args, implementation="torch", **kwargs)
    expected_count = field.ndim
    if kwargs["derivative_orders"] == 2 and kwargs["include_mixed"]:
        expected_count += (field.ndim * (field.ndim - 1)) // 2
    assert output.shape[0] == expected_count


# Validate benchmark input generation contract for backward inputs.
def test_spectral_grid_gradient_make_inputs_backward(device: str):
    label, args, kwargs = next(
        iter(SpectralGridGradient.make_inputs_backward(device=device))
    )
    assert isinstance(label, str)
    assert isinstance(args, tuple)
    assert isinstance(kwargs, dict)

    field = args[0]
    assert field.requires_grad

    output = SpectralGridGradient.dispatch(*args, implementation="torch", **kwargs)
    output.square().mean().backward()
    assert field.grad is not None


# Validate exported API and validation error paths.
def test_spectral_grid_gradient_error_handling(device: str):
    field = torch.randn(64, device=device, dtype=torch.float32)
    output = spectral_grid_gradient(field, lengths=2.0, derivative_orders=1)
    assert output.shape == (1, 64)

    with pytest.raises(ValueError, match="supports 1D-3D fields"):
        SpectralGridGradient.dispatch(
            torch.randn(4, 4, 4, 4, device=device, dtype=torch.float32),
            implementation="torch",
        )

    with pytest.raises(TypeError, match="floating-point"):
        SpectralGridGradient.dispatch(
            torch.ones(64, device=device, dtype=torch.int64),
            implementation="torch",
        )

    with pytest.raises(ValueError, match="supports derivative orders"):
        SpectralGridGradient.dispatch(
            field,
            derivative_orders=3,
            implementation="torch",
        )

    with pytest.raises(ValueError, match="only valid when requesting 2nd derivatives"):
        SpectralGridGradient.dispatch(
            field,
            derivative_orders=1,
            include_mixed=True,
            implementation="torch",
        )

    with pytest.raises(
        ValueError, match="mixed derivatives require at least 2D inputs"
    ):
        SpectralGridGradient.dispatch(
            field,
            derivative_orders=2,
            include_mixed=True,
            implementation="torch",
        )

    with pytest.raises(ValueError, match="must have 2 entries"):
        SpectralGridGradient.dispatch(
            torch.randn(32, 32, device=device, dtype=torch.float32),
            lengths=(1.0,),
            implementation="torch",
        )

    with pytest.raises(ValueError, match="strictly positive"):
        SpectralGridGradient.dispatch(
            torch.randn(32, 32, device=device, dtype=torch.float32),
            lengths=(1.0, 0.0),
            implementation="torch",
        )
