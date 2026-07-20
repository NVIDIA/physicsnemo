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

from physicsnemo.nn.functional import (
    spectral_grid_divergence,
    spectral_grid_gradient,
    spectral_grid_laplacian,
)
from physicsnemo.nn.functional.derivatives import SpectralGridLaplacian


def _make_periodic_scalar_field(
    device: str,
    dim: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, tuple[float, ...], torch.Tensor]:
    """Build a periodic scalar field and its exact Laplacian."""
    shapes = ((31,), (23, 19), (17, 15, 13))
    all_lengths = (2.0, 1.5, 1.25)
    shape = shapes[dim - 1]
    lengths = all_lengths[:dim]
    axes = tuple(
        torch.arange(size, device=device, dtype=dtype) * (length / size)
        for size, length in zip(shape, lengths)
    )
    mesh = torch.meshgrid(*axes, indexing="ij")
    k = tuple(2.0 * torch.pi / length for length in lengths)
    amplitudes = (1.0, -0.5, 0.25)
    modes = (1, 2, 3)
    phases = (0.1, -0.2, 0.3)

    field = torch.full_like(mesh[0], 2.5)
    expected = torch.zeros_like(mesh[0])
    for axis in range(dim):
        term = amplitudes[axis] * torch.sin(
            modes[axis] * k[axis] * mesh[axis] + phases[axis]
        )
        field = field + term
        expected = expected - (modes[axis] * k[axis]) ** 2 * term

    coupled = torch.full_like(mesh[0], 0.2)
    for axis in range(dim):
        coupled = coupled * torch.cos(k[axis] * mesh[axis] + 0.15 * (axis + 1))
    field = field + coupled
    expected = expected - sum(value**2 for value in k) * coupled
    return field, lengths, expected


@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_spectral_grid_laplacian_analytic(
    device: str,
    dim: int,
    dtype: torch.dtype,
):
    field, lengths, expected = _make_periodic_scalar_field(device, dim, dtype)
    output = SpectralGridLaplacian.dispatch(
        field,
        lengths=lengths,
        implementation="torch",
    )
    atol = 2e-3 if dtype == torch.float32 else 1e-10
    rtol = 2e-4 if dtype == torch.float32 else 1e-10
    torch.testing.assert_close(output, expected, atol=atol, rtol=rtol)


def test_spectral_grid_laplacian_public_function(device: str):
    field, lengths, expected = _make_periodic_scalar_field(
        device,
        dim=2,
        dtype=torch.float64,
    )
    output = spectral_grid_laplacian(field, lengths=lengths)
    torch.testing.assert_close(output, expected, atol=1e-10, rtol=1e-10)


def test_divergence_of_gradient_matches_laplacian(device: str):
    field, lengths, _expected = _make_periodic_scalar_field(
        device,
        dim=3,
        dtype=torch.float64,
    )
    gradient = spectral_grid_gradient(field, lengths=lengths, derivative_orders=1)
    divergence = spectral_grid_divergence(gradient, lengths=lengths)
    laplacian = spectral_grid_laplacian(field, lengths=lengths)
    torch.testing.assert_close(divergence, laplacian, atol=1e-10, rtol=1e-10)


def test_even_grid_nyquist_mode_contract(device: str):
    size = 8
    length = 2.0
    field = torch.where(
        torch.arange(size, device=device) % 2 == 0,
        1.0,
        -1.0,
    ).to(torch.float64)

    gradient = spectral_grid_gradient(field, lengths=length, derivative_orders=1)
    divergence = spectral_grid_divergence(field.unsqueeze(0), lengths=length)
    divergence_of_gradient = spectral_grid_divergence(gradient, lengths=length)
    laplacian = spectral_grid_laplacian(field, lengths=length)
    nyquist_wavenumber = torch.pi * size / length

    torch.testing.assert_close(
        gradient, torch.zeros_like(gradient), atol=1e-12, rtol=0.0
    )
    torch.testing.assert_close(
        divergence,
        torch.zeros_like(divergence),
        atol=1e-12,
        rtol=0.0,
    )
    torch.testing.assert_close(
        divergence_of_gradient,
        torch.zeros_like(divergence_of_gradient),
        atol=1e-12,
        rtol=0.0,
    )
    torch.testing.assert_close(
        laplacian,
        -(nyquist_wavenumber**2) * field,
        atol=1e-10,
        rtol=1e-10,
    )


def test_spectral_grid_laplacian_gradcheck(device: str):
    field = torch.randn(
        (5, 7),
        device=device,
        dtype=torch.float64,
        requires_grad=True,
    )
    assert torch.autograd.gradcheck(
        lambda value: spectral_grid_laplacian(
            value,
            lengths=(2.0, 1.5),
            implementation="torch",
        ),
        (field,),
        fast_mode=True,
        eps=1e-6,
        atol=1e-5,
        rtol=1e-3,
    )


def test_spectral_grid_laplacian_benchmark_inputs(device: str):
    label, args, kwargs = next(
        iter(SpectralGridLaplacian.make_inputs_forward(device=device))
    )
    assert isinstance(label, str)
    field = args[0]
    output = SpectralGridLaplacian.dispatch(*args, implementation="torch", **kwargs)
    assert output.shape == field.shape

    _label, backward_args, backward_kwargs = next(
        iter(SpectralGridLaplacian.make_inputs_backward(device=device))
    )
    backward_field = backward_args[0]
    output = SpectralGridLaplacian.dispatch(
        *backward_args,
        implementation="torch",
        **backward_kwargs,
    )
    output.square().mean().backward()
    assert backward_field.grad is not None
    assert torch.isfinite(backward_field.grad).all()


def test_spectral_grid_laplacian_preserves_low_precision_dtype(device: str):
    for dtype in (torch.float16, torch.bfloat16):
        field = torch.randn((9, 7), device=device, dtype=dtype)
        output = spectral_grid_laplacian(field, lengths=(2.0, 1.5))
        assert output.dtype == dtype
        assert torch.isfinite(output).all()


def test_spectral_grid_laplacian_error_handling(device: str):
    with pytest.raises(TypeError, match="floating-point"):
        spectral_grid_laplacian(torch.ones((9, 7), device=device, dtype=torch.int64))

    with pytest.raises(ValueError, match="supports 1D-3D fields"):
        spectral_grid_laplacian(torch.ones((3, 3, 3, 3), device=device))

    with pytest.raises(ValueError, match="must have 2 entries"):
        spectral_grid_laplacian(
            torch.ones((9, 7), device=device),
            lengths=(1.0,),
        )

    with pytest.raises(ValueError, match="strictly positive"):
        spectral_grid_laplacian(
            torch.ones((9, 7), device=device),
            lengths=(1.0, -1.0),
        )

    with pytest.raises(ValueError, match="finite and strictly positive"):
        spectral_grid_laplacian(
            torch.ones((9, 7), device=device),
            lengths=(1.0, float("inf")),
        )
