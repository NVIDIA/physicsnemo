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

from physicsnemo.nn.functional import spectral_grid_curl, spectral_grid_gradient
from physicsnemo.nn.functional.derivatives import SpectralGridCurl


def _make_periodic_vector_field(
    device: str,
    dim: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, tuple[float, ...], torch.Tensor]:
    """Build a periodic vector field and its exact right-handed curl."""
    shape = (23, 19) if dim == 2 else (17, 15, 13)
    lengths = (2.0, 1.5) if dim == 2 else (2.0, 1.5, 1.25)
    axes = tuple(
        torch.arange(size, device=device, dtype=dtype) * (length / size)
        for size, length in zip(shape, lengths)
    )
    mesh = torch.meshgrid(*axes, indexing="ij")
    k = tuple(2.0 * torch.pi / length for length in lengths)

    if dim == 2:
        x, y = mesh
        u = 0.5 * torch.cos(k[0] * x - 0.3) * torch.sin(3.0 * k[1] * y + 0.4)
        v = torch.sin(2.0 * k[0] * x + 0.1) * torch.cos(k[1] * y - 0.2)
        expected = 2.0 * k[0] * torch.cos(2.0 * k[0] * x + 0.1) * torch.cos(
            k[1] * y - 0.2
        ) - 1.5 * k[1] * torch.cos(k[0] * x - 0.3) * torch.cos(3.0 * k[1] * y + 0.4)
        return torch.stack((u, v)), lengths, expected

    x, y, z = mesh
    field = torch.stack(
        (
            torch.sin(k[1] * y + 0.1) * torch.cos(2.0 * k[2] * z - 0.2),
            torch.sin(k[2] * z + 0.2) * torch.cos(2.0 * k[0] * x - 0.3),
            torch.sin(k[0] * x + 0.3) * torch.cos(2.0 * k[1] * y - 0.4),
        )
    )
    expected = torch.stack(
        (
            -2.0 * k[1] * torch.sin(k[0] * x + 0.3) * torch.sin(2.0 * k[1] * y - 0.4)
            - k[2] * torch.cos(k[2] * z + 0.2) * torch.cos(2.0 * k[0] * x - 0.3),
            -2.0 * k[2] * torch.sin(k[1] * y + 0.1) * torch.sin(2.0 * k[2] * z - 0.2)
            - k[0] * torch.cos(k[0] * x + 0.3) * torch.cos(2.0 * k[1] * y - 0.4),
            -2.0 * k[0] * torch.sin(k[2] * z + 0.2) * torch.sin(2.0 * k[0] * x - 0.3)
            - k[1] * torch.cos(k[1] * y + 0.1) * torch.cos(2.0 * k[2] * z - 0.2),
        )
    )
    return field, lengths, expected


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_spectral_grid_curl_analytic(
    device: str,
    dim: int,
    dtype: torch.dtype,
):
    vector_field, lengths, expected = _make_periodic_vector_field(device, dim, dtype)
    output = SpectralGridCurl.dispatch(
        vector_field,
        lengths=lengths,
        implementation="torch",
    )
    tolerance = 2e-4 if dtype == torch.float32 else 1e-10
    torch.testing.assert_close(output, expected, atol=tolerance, rtol=tolerance)


def test_spectral_grid_curl_public_function(device: str):
    vector_field, lengths, expected = _make_periodic_vector_field(
        device,
        dim=2,
        dtype=torch.float64,
    )
    output = spectral_grid_curl(vector_field, lengths=lengths)
    torch.testing.assert_close(output, expected, atol=1e-10, rtol=1e-10)


def test_spectral_grid_curl_of_gradient_is_zero(device: str):
    shape = (11, 9, 7)
    lengths = (2.0, 1.5, 1.25)
    axes = tuple(
        torch.arange(size, device=device, dtype=torch.float64) * (length / size)
        for size, length in zip(shape, lengths)
    )
    x, y, z = torch.meshgrid(*axes, indexing="ij")
    field = (
        torch.sin(2.0 * torch.pi * x / lengths[0] + 0.1)
        * torch.cos(2.0 * torch.pi * y / lengths[1] - 0.2)
        * torch.sin(2.0 * torch.pi * z / lengths[2] + 0.3)
    )
    gradient = spectral_grid_gradient(field, lengths=lengths, derivative_orders=1)
    output = spectral_grid_curl(gradient, lengths=lengths)
    torch.testing.assert_close(output, torch.zeros_like(output), atol=1e-10, rtol=0.0)


@pytest.mark.parametrize(
    ("shape", "lengths"),
    [
        ((2, 5, 7), (2.0, 1.5)),
        ((3, 3, 5, 7), (2.0, 1.5, 1.25)),
    ],
)
def test_spectral_grid_curl_gradcheck(
    device: str,
    shape: tuple[int, ...],
    lengths: tuple[float, ...],
):
    vector_field = torch.randn(
        shape,
        device=device,
        dtype=torch.float64,
        requires_grad=True,
    )
    assert torch.autograd.gradcheck(
        lambda value: spectral_grid_curl(
            value,
            lengths=lengths,
            implementation="torch",
        ),
        (vector_field,),
        fast_mode=True,
        eps=1e-6,
        atol=1e-5,
        rtol=1e-3,
    )


def test_spectral_grid_curl_benchmark_inputs(device: str):
    cases = list(SpectralGridCurl.make_inputs_forward(device=device))
    assert len(cases) == 2
    for label, args, kwargs in cases:
        assert isinstance(label, str)
        vector_field = args[0]
        output = SpectralGridCurl.dispatch(*args, implementation="torch", **kwargs)
        expected_shape = (
            vector_field.shape[1:] if vector_field.ndim == 3 else vector_field.shape
        )
        assert output.shape == expected_shape

    _label, backward_args, backward_kwargs = next(
        iter(SpectralGridCurl.make_inputs_backward(device=device))
    )
    backward_field = backward_args[0]
    output = SpectralGridCurl.dispatch(
        *backward_args,
        implementation="torch",
        **backward_kwargs,
    )
    output.square().mean().backward()
    assert backward_field.grad is not None
    assert torch.isfinite(backward_field.grad).all()


def test_spectral_grid_curl_preserves_low_precision_dtype(device: str):
    for dtype in (torch.float16, torch.bfloat16):
        vector_field = torch.randn((2, 9, 7), device=device, dtype=dtype)
        output = spectral_grid_curl(vector_field, lengths=(2.0, 1.5))
        assert output.dtype == dtype
        assert torch.isfinite(output).all()


def test_spectral_grid_curl_error_handling(device: str):
    with pytest.raises(TypeError, match="floating-point"):
        spectral_grid_curl(torch.ones((2, 9, 7), device=device, dtype=torch.int64))

    with pytest.raises(ValueError, match="2D or 3D"):
        spectral_grid_curl(torch.ones((1, 9), device=device))

    with pytest.raises(ValueError, match=r"shape\[0\]"):
        spectral_grid_curl(torch.ones((3, 9, 7), device=device))

    with pytest.raises(ValueError, match="must have 2 entries"):
        spectral_grid_curl(
            torch.ones((2, 9, 7), device=device),
            lengths=(1.0,),
        )

    with pytest.raises(ValueError, match="finite and strictly positive"):
        spectral_grid_curl(
            torch.ones((2, 9, 7), device=device),
            lengths=(1.0, float("nan")),
        )
