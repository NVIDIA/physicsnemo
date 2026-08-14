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

import sys
from pathlib import Path

import torch

EXAMPLE_ROOT = Path(__file__).resolve().parents[2] / "examples" / "tcad" / "fp_ddm"
sys.path.insert(0, str(EXAMPLE_ROOT))

from fpddm.elasticity import Elasticity2DSolver  # noqa: E402
from fpddm.elasticity_example import run_elasticity  # noqa: E402


def _checkerboard_problem(size: int):
    spacing = 1.0 / (size - 1)
    solver = Elasticity2DSolver(
        spacing=(spacing, spacing),
        max_iter=4000,
        tolerance=1.0e-11,
        device="cpu",
        dtype=torch.float64,
    )
    row = torch.arange(size)[:, None]
    column = torch.arange(size)[None, :]
    checkerboard = (1.0 - 2.0 * ((row + column) % 2)).to(torch.float64)
    mode = torch.zeros((1, 2, size, size), dtype=torch.float64)
    mode[:, 0] = checkerboard
    mask = torch.zeros_like(mode, dtype=torch.bool)
    mask[..., (0, -1), :] = True
    mask[..., :, (0, -1)] = True
    mode.masked_fill_(mask, 0.0)
    young_modulus = torch.ones((1, size, size), dtype=torch.float64)
    poisson_ratio = torch.full_like(young_modulus, 0.25)
    return solver, mode, mask, young_modulus, poisson_ratio


def test_plane_stress_operator_matches_affine_solution():
    solver = Elasticity2DSolver(spacing=(0.3, 0.2), device="cpu", dtype=torch.float64)
    y = 0.3 * torch.arange(7, dtype=torch.float64)
    x = 0.2 * torch.arange(8, dtype=torch.float64)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    displacement = torch.stack((2.0 * yy + 3.0 * xx, -yy + 4.0 * xx))[None]
    young_modulus = torch.full((1, 7, 8), 200.0, dtype=torch.float64)
    poisson_ratio = torch.full_like(young_modulus, 0.25)

    strain = solver.strain(displacement)
    expected_strain = torch.tensor([[2.0, 1.0], [1.0, 4.0]], dtype=torch.float64).view(
        1, 2, 2, 1, 1
    )
    torch.testing.assert_close(
        strain, expected_strain.expand_as(strain), atol=1.0e-12, rtol=0.0
    )

    shear_modulus = 200.0 / (2.0 * 1.25)
    lame_lambda = 200.0 * 0.25 / (1.0 - 0.25**2)
    expected_stress = (
        2.0 * shear_modulus * expected_strain
        + lame_lambda
        * expected_strain.diagonal(dim1=1, dim2=2).sum(-1).view(1, 1, 1, 1, 1)
        * torch.eye(2, dtype=torch.float64).view(1, 2, 2, 1, 1)
    )
    stress = solver.stress(displacement, young_modulus, poisson_ratio)
    torch.testing.assert_close(
        stress, expected_stress.expand_as(stress), atol=1.0e-10, rtol=0.0
    )
    expected_von_mises = torch.sqrt(
        expected_stress[:, 0, 0].square()
        - expected_stress[:, 0, 0] * expected_stress[:, 1, 1]
        + expected_stress[:, 1, 1].square()
        + 3.0 * expected_stress[:, 0, 1].square()
    )
    torch.testing.assert_close(
        solver.von_mises(displacement, young_modulus, poisson_ratio),
        expected_von_mises.expand(1, 7, 8),
        atol=1.0e-10,
        rtol=0.0,
    )
    residual = solver.residual(
        displacement,
        young_modulus,
        poisson_ratio,
        torch.zeros_like(displacement),
    )
    torch.testing.assert_close(
        residual[..., 2:-2, 2:-2],
        torch.zeros_like(residual[..., 2:-2, 2:-2]),
        atol=1.0e-9,
        rtol=0.0,
    )


def test_elasticity_operator_penalizes_checkerboard_mode():
    solver, mode, _, young_modulus, poisson_ratio = _checkerboard_problem(17)
    residual = solver.residual(
        mode, young_modulus, poisson_ratio, torch.zeros_like(mode)
    )
    interior_rms = residual[..., 2:-2, 2:-2].square().mean().sqrt()

    assert float(interior_rms) * solver.spacing[0] ** 2 > 1.0


def test_checkerboard_force_response_decays_under_refinement():
    amplitudes = []
    for size in (17, 33):
        solver, force, mask, young_modulus, poisson_ratio = _checkerboard_problem(size)
        displacement = solver.solve(
            young_modulus,
            poisson_ratio,
            force,
            torch.zeros_like(force),
            mask,
        )
        amplitudes.append(float(displacement.abs().max()))

    refinement_ratio = amplitudes[1] / amplitudes[0]
    assert 0.15 < refinement_ratio < 0.35


def test_elasticity_ddm_matches_monolithic_reference(tmp_path):
    result = run_elasticity(
        tmp_path,
        size=16,
        max_iterations=100,
        tolerance=1.0e-7,
        device="cpu",
        visualize=False,
    )

    assert result.converged
    assert len(result.metrics) < 100
    assert result.metrics[-1]["loss_interface"] < result.metrics[0]["loss_interface"]
    reference_error = torch.linalg.vector_norm(
        result.reference_displacement - result.exact_displacement
    ) / torch.linalg.vector_norm(result.exact_displacement)
    assert reference_error < 2.0e-3
    assert result.displacement_relative_error < 5.0e-4
    assert result.stress_relative_error < 5.0e-3
    assert float(result.reference_stress.square().mean().sqrt()) > 1.0e-4
