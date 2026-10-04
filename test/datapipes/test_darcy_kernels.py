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

import numpy as np
import pytest
import warp as wp

from physicsnemo.datapipes.benchmarks.kernels.finite_difference import (
    darcy_mgrid_jacobi_iterative_batched_2d,
)


@pytest.mark.parametrize("factor", [1, 2, 4])
@pytest.mark.parametrize("variable_permeability", [False, True])
def test_darcy_jacobi_central_differences(device, factor, variable_permeability):
    """Compare every updated grid point with independent central differences."""
    length = 9
    dx = 1.0 / length
    rng = np.random.default_rng(17)
    pressure = rng.normal(size=(2, length, length)).astype(np.float32)
    permeability = np.full_like(pressure, 2.0)
    if variable_permeability:
        permeability += rng.uniform(0.0, 1.0, pressure.shape).astype(np.float32)
    source = 0.75
    expected = np.full_like(pressure, -99.0)
    extent = length if factor == 1 else (length - 1) // factor
    h = dx * factor

    def value(field, b, i, j, clamp=False):
        if clamp:
            return float(field[b, np.clip(i, 0, length - 1), np.clip(j, 0, length - 1)])
        if i < 0 or i >= length or j < 0 or j >= length:
            return 0.0
        return float(field[b, i, j])

    for b in range(2):
        for i in range(extent):
            for j in range(extent):
                x, y = factor * i + factor - 1, factor * j + factor - 1
                k = float(permeability[b, x, y])
                px0, px1 = (
                    value(pressure, b, x + offset, y) for offset in (-factor, factor)
                )
                py0, py1 = (
                    value(pressure, b, x, y + offset) for offset in (-factor, factor)
                )
                kx0, kx1 = (
                    value(permeability, b, x + offset, y, True)
                    for offset in (-factor, factor)
                )
                ky0, ky1 = (
                    value(permeability, b, x, y + offset, True)
                    for offset in (-factor, factor)
                )
                grad_k = np.array([kx1 - kx0, ky1 - ky0]) / (2 * h)
                grad_p = np.array([px1 - px0, py1 - py0]) / (2 * h)
                expected[b, x, y] = (
                    k * (px0 + px1 + py0 + py1) / h**2 + np.dot(grad_k, grad_p) + source
                ) / (4 * k / h**2)

    actual = wp.full(pressure.shape, -99.0, dtype=float, device=device)
    wp.launch(
        darcy_mgrid_jacobi_iterative_batched_2d,
        dim=(2, extent, extent),
        inputs=[
            wp.array(pressure, dtype=float, device=device),
            actual,
            wp.array(permeability, dtype=float, device=device),
            source,
            length,
            length,
            dx,
            factor,
        ],
        device=device,
    )
    np.testing.assert_allclose(actual.numpy(), expected, rtol=2e-6, atol=2e-7)


@pytest.mark.parametrize("factor", [1, 2, 4])
def test_darcy_jacobi_preserves_linear_manufactured_solution(device, factor):
    """An affine solution with affine permeability has a constant source."""
    length = 17
    dx = 1.0 / length
    x, y = np.meshgrid(np.arange(length) * dx, np.arange(length) * dx, indexing="ij")
    pressure = (x + 2 * y)[None].astype(np.float32)
    permeability = (2 + 0.3 * x + 0.2 * y)[None].astype(np.float32)
    # -div(k grad(p)) = -(0.3 * 1 + 0.2 * 2).
    source = -0.7
    extent = length if factor == 1 else (length - 1) // factor
    actual = wp.zeros(pressure.shape, dtype=float, device=device)
    wp.launch(
        darcy_mgrid_jacobi_iterative_batched_2d,
        dim=(1, extent, extent),
        inputs=[
            wp.array(pressure, dtype=float, device=device),
            actual,
            wp.array(permeability, dtype=float, device=device),
            source,
            length,
            length,
            dx,
            factor,
        ],
        device=device,
    )
    indices = factor * np.arange(extent) + factor - 1
    interior = indices[(indices >= factor) & (indices + factor < length)]
    np.testing.assert_allclose(
        actual.numpy()[0][np.ix_(interior, interior)],
        pressure[0][np.ix_(interior, interior)],
        rtol=2e-6,
        atol=2e-7,
    )
