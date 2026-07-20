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

from physicsnemo.nn.functional import spectral_grid_divergence
from physicsnemo.nn.functional.derivatives import SpectralGridDivergence


def _make_periodic_vector_field(
    device: str,
    dim: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, tuple[float, ...], torch.Tensor]:
    """Build an anisotropic periodic vector field and its exact divergence."""
    shapes = ((31,), (23, 19), (17, 15, 13))
    all_lengths = (2.0, 1.5, 1.25)
    shape = shapes[dim - 1]
    lengths = all_lengths[:dim]
    axes = tuple(
        torch.arange(size, device=device, dtype=dtype) * (length / size)
        for size, length in zip(shape, lengths)
    )
    mesh = torch.meshgrid(*axes, indexing="ij")
    wavenumbers = tuple(2.0 * torch.pi / length for length in lengths)
    amplitudes = (1.0, -0.5, 0.25)
    modes = (1, 2, 3)
    phases = (0.1, -0.2, 0.3)

    components = []
    derivatives = []
    for component in range(dim):
        transverse = torch.ones_like(mesh[component])
        for axis in range(dim):
            if axis != component:
                transverse = transverse * torch.cos(
                    wavenumbers[axis] * mesh[axis] + 0.2 * (axis + component + 1)
                )
        phase = modes[component] * wavenumbers[component] * mesh[component]
        phase = phase + phases[component]
        components.append(amplitudes[component] * torch.sin(phase) * transverse)
        derivatives.append(
            amplitudes[component]
            * modes[component]
            * wavenumbers[component]
            * torch.cos(phase)
            * transverse
        )

    return torch.stack(components), lengths, torch.stack(derivatives).sum(dim=0)


@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_spectral_grid_divergence_analytic(
    device: str,
    dim: int,
    dtype: torch.dtype,
):
    vector_field, lengths, expected = _make_periodic_vector_field(device, dim, dtype)
    output = SpectralGridDivergence.dispatch(
        vector_field,
        lengths=lengths,
        implementation="torch",
    )
    tolerance = 2e-4 if dtype == torch.float32 else 1e-10
    torch.testing.assert_close(output, expected, atol=tolerance, rtol=tolerance)


def test_spectral_grid_divergence_public_function(device: str):
    vector_field, lengths, expected = _make_periodic_vector_field(
        device,
        dim=2,
        dtype=torch.float64,
    )
    output = spectral_grid_divergence(vector_field, lengths=lengths)
    torch.testing.assert_close(output, expected, atol=1e-10, rtol=1e-10)


def test_spectral_grid_divergence_gradcheck(device: str):
    vector_field = torch.randn(
        (2, 5, 7),
        device=device,
        dtype=torch.float64,
        requires_grad=True,
    )
    assert torch.autograd.gradcheck(
        lambda value: spectral_grid_divergence(
            value,
            lengths=(2.0, 1.5),
            implementation="torch",
        ),
        (vector_field,),
        fast_mode=True,
        eps=1e-6,
        atol=1e-5,
        rtol=1e-3,
    )


def test_spectral_grid_divergence_benchmark_inputs(device: str):
    label, args, kwargs = next(
        iter(SpectralGridDivergence.make_inputs_forward(device=device))
    )
    assert isinstance(label, str)
    vector_field = args[0]
    output = SpectralGridDivergence.dispatch(
        *args,
        implementation="torch",
        **kwargs,
    )
    assert output.shape == vector_field.shape[1:]

    _label, backward_args, backward_kwargs = next(
        iter(SpectralGridDivergence.make_inputs_backward(device=device))
    )
    backward_field = backward_args[0]
    output = SpectralGridDivergence.dispatch(
        *backward_args,
        implementation="torch",
        **backward_kwargs,
    )
    output.square().mean().backward()
    assert backward_field.grad is not None
    assert torch.isfinite(backward_field.grad).all()


def test_spectral_grid_divergence_preserves_low_precision_dtype(device: str):
    for dtype in (torch.float16, torch.bfloat16):
        vector_field = torch.randn((2, 9, 7), device=device, dtype=dtype)
        output = spectral_grid_divergence(vector_field, lengths=(2.0, 1.5))
        assert output.dtype == dtype
        assert torch.isfinite(output).all()


def test_spectral_grid_divergence_error_handling(device: str):
    with pytest.raises(TypeError, match="floating-point"):
        spectral_grid_divergence(
            torch.ones((2, 9, 7), device=device, dtype=torch.int64)
        )

    with pytest.raises(ValueError, match=r"shape\[0\]"):
        spectral_grid_divergence(torch.ones((3, 9, 7), device=device))

    with pytest.raises(ValueError, match="must have 2 entries"):
        spectral_grid_divergence(
            torch.ones((2, 9, 7), device=device),
            lengths=(1.0,),
        )

    with pytest.raises(ValueError, match="strictly positive"):
        spectral_grid_divergence(
            torch.ones((2, 9, 7), device=device),
            lengths=(1.0, 0.0),
        )

    with pytest.raises(ValueError, match="finite and strictly positive"):
        spectral_grid_divergence(
            torch.ones((2, 9, 7), device=device),
            lengths=(1.0, float("nan")),
        )
