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

"""Tests for the steel bridge CAE reference model and the PINN formulation."""

import numpy as np
import pytest
import torch
from steel_bridge_cae import (
    QuadMesh,
    SteelBridge,
    _element_matrices,
    _gauss_points,
    solve_plane_stress,
    von_mises,
)
from steel_bridge_pinn import (
    BridgeEnergyLoss,
    BridgePINN,
    BridgeSampler,
    analytic_scales,
    stress_tensor,
)

TOL = dict(rtol=1e-6, atol=1e-3)


@pytest.fixture(scope="module")
def bridge() -> SteelBridge:
    return SteelBridge()


@pytest.fixture(scope="module")
def coarse_solution(bridge: SteelBridge):
    return solve_plane_stress(
        bridge, QuadMesh(nx=60, ny=12, span=bridge.span, depth=bridge.depth)
    )


# --------------------------------------------------------------------------- FEA
def test_element_matches_uniform_strain(bridge: SteelBridge):
    """A bilinear quad must reproduce a constant strain state exactly (patch test)."""
    e_xx, e_yy, g_xy = 1.0e-3, -2.0e-4, 5.0e-4
    strain = np.array([e_xx, e_yy, g_xy])
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    displacement = np.stack(
        [
            e_xx * nodes[:, 0] + 0.5 * g_xy * nodes[:, 1],
            e_yy * nodes[:, 1] + 0.5 * g_xy * nodes[:, 0],
        ],
        axis=-1,
    )
    xy = nodes[None, ...]
    determinant, operators = _element_matrices(xy, bridge.thickness)
    _, weights = _gauss_points()
    for k, b in enumerate(operators):
        # Shape functions are isoparametric, so the strain is constant per element.
        recovered = b[0] @ displacement.ravel()
        assert np.allclose(recovered, strain, atol=1e-12)
        assert determinant[0, k] == pytest.approx(0.25, rel=1e-12)
        assert weights[k] == pytest.approx(1.0)


def test_element_is_free_of_rigid_body_forces(bridge: SteelBridge):
    """Translations and small rotations must not produce internal forces."""
    xy = np.array([[0.0, 0.0], [0.4, 0.0], [0.4, 0.2], [0.0, 0.2]])[None, ...]
    determinant, operators = _element_matrices(xy, bridge.thickness)
    stiffness = bridge.constitutive_matrix
    _, weights = _gauss_points()
    ke = np.zeros((1, 8, 8))
    for k, b in enumerate(operators):
        ke += (
            np.einsum("eki,kl,elj->eij", b, stiffness, b)
            * (determinant[:, k] * weights[k] * bridge.thickness)[:, None, None]
        )

    translation = np.tile([0.3, -0.7], 4)
    rotation = np.concatenate([[-1.0e-6 * y, 1.0e-6 * x] for x, y in xy[0]])
    for mode in (translation, rotation):
        assert np.allclose(ke[0] @ mode, 0.0, atol=1e-6)


def test_reactions_balance_the_applied_load(bridge: SteelBridge, coarse_solution):
    """Vertical reactions must equal deck loads plus self weight."""
    applied = bridge.total_load + (
        bridge.density * bridge.gravity * bridge.span * bridge.depth * bridge.thickness
    )
    assert coarse_solution.reactions[:, 1].sum() == pytest.approx(applied, rel=1e-8)


def test_deflection_matches_beam_theory(bridge: SteelBridge):
    """The finite element model must agree with Euler-Bernoulli theory."""
    solution = solve_plane_stress(
        bridge, QuadMesh(nx=240, ny=48, span=bridge.span, depth=bridge.depth)
    )
    theory = bridge.beam_theory()
    assert solution.midspan_deflection() == pytest.approx(
        theory.midspan_deflection, rel=0.02
    )
    # Bending stress at the bottom flange midspan.
    gauge = solution.stress_at(np.array([[0.5 * bridge.span, 0.0]]))[0, 0]
    assert gauge == pytest.approx(theory.max_bending_stress, rel=0.05)


def test_mesh_refinement_improves_deflection(bridge: SteelBridge):
    errors = []
    for nx, ny in ((60, 12), (120, 24)):
        solution = solve_plane_stress(
            bridge, QuadMesh(nx=nx, ny=ny, span=bridge.span, depth=bridge.depth)
        )
        errors.append(
            abs(
                solution.midspan_deflection() / bridge.beam_theory().midspan_deflection
                - 1.0
            )
        )
    assert errors[1] < errors[0]


def test_sensor_measurements_match_the_field(bridge: SteelBridge, coarse_solution):
    points, displacement = coarse_solution.sensor_measurements(16, seed=5)
    assert points.shape == (16, 2)
    assert np.allclose(
        displacement, coarse_solution.displacement_at(points), atol=1e-12
    )
    assert np.all(np.abs(displacement) < 10 * bridge.beam_theory().midspan_deflection)


def test_von_mises_limits():
    uniaxial = np.array([[2.0, 0.0, 0.0]])
    assert von_mises(uniaxial)[0] == pytest.approx(2.0)
    shear = np.array([[0.0, 0.0, 3.0]])
    assert von_mises(shear)[0] == pytest.approx(3.0 * np.sqrt(3.0))
    biaxial = np.array([[1.0, 1.0, 0.0]])
    assert von_mises(biaxial)[0] == pytest.approx(1.0)


# --------------------------------------------------------------------------- PINN
def _uniform_stress_model(bridge: SteelBridge, strain: tuple[float, float, float]):
    """Analytic displacement field with a constant strain state."""

    class Model(torch.nn.Module):
        def forward(self, xy: torch.Tensor) -> torch.Tensor:
            u_x = strain[0] * xy[:, 0] + 0.5 * strain[2] * xy[:, 1]
            u_y = strain[1] * xy[:, 1] + 0.5 * strain[2] * xy[:, 0]
            return torch.stack([u_x, u_y], dim=-1)

    return Model()


def test_stress_tensor_reproduces_plane_stress_law(bridge: SteelBridge):
    """A constant-strain field must give the plane-stress constitutive response."""
    strain = (1.0e-3, -2.0e-4, 4.0e-4)
    model = _uniform_stress_model(bridge, strain)
    points = torch.rand(32, 2) * torch.tensor([bridge.span, bridge.depth])
    sigma = stress_tensor(model, points, bridge)

    e, nu = bridge.youngs_modulus, bridge.poisson_ratio
    factor = e / (1.0 - nu**2)
    expected_xx = factor * (strain[0] + nu * strain[1])
    expected_yy = factor * (strain[1] + nu * strain[0])
    expected_xy = bridge.shear_modulus * strain[2]
    assert torch.allclose(
        sigma[:, 0], torch.full_like(sigma[:, 0], expected_xx), rtol=1e-6
    )
    assert torch.allclose(
        sigma[:, 1], torch.full_like(sigma[:, 1], expected_yy), rtol=1e-6
    )
    assert torch.allclose(
        sigma[:, 2], torch.full_like(sigma[:, 2], expected_xy), rtol=1e-6
    )


def test_bearing_masks_enforce_zero_displacement(bridge: SteelBridge):
    scales = analytic_scales(bridge)
    model = BridgePINN(bridge, scales.displacement)
    # Trainable parameters are zero-initialised, so perturb them to test the mask.
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(0.1)

    left_pad = torch.tensor([[0.0, 0.0], [0.0, 0.5 * bridge.bearing_length]])
    right_pad = torch.tensor(
        [[bridge.span, 0.0], [bridge.span, 0.5 * bridge.bearing_length]]
    )
    assert torch.allclose(model(left_pad), torch.zeros(2, 2), atol=1e-12)
    assert torch.allclose(model(right_pad)[:, 1], torch.zeros(2), atol=1e-12)

    free = torch.tensor([[0.5 * bridge.span, bridge.depth]])
    left, right = model.bearing_masks(free)
    assert float(left) < 1e-6
    assert float(right) < 1e-6


def test_zero_initialised_model_has_zero_energy(bridge: SteelBridge):
    scales = analytic_scales(bridge)
    model = BridgePINN(bridge, scales.displacement)
    energy_loss = BridgeEnergyLoss(bridge=bridge, scales=scales)
    sampler = BridgeSampler(bridge, seed=0)
    value, contributions = energy_loss.energy(model, sampler, 256, 16)
    assert abs(value) < 1e-12
    assert abs(contributions["strain_energy"]) < 1e-12


def test_energy_decreases_with_deflection(bridge: SteelBridge):
    """The load must do positive work, so deflecting downwards lowers the energy."""
    scales = analytic_scales(bridge)
    energy_loss = BridgeEnergyLoss(bridge=bridge, scales=scales)
    sampler = BridgeSampler(bridge, seed=0)

    def beam_like(amplitude: float) -> torch.nn.Module:
        class Model(torch.nn.Module):
            def forward(self, xy: torch.Tensor) -> torch.Tensor:
                shape = torch.sin(np.pi * xy[:, 0] / bridge.span)
                return torch.stack(
                    [torch.zeros_like(shape), -amplitude * shape], dim=-1
                )

        return Model()

    low, _ = energy_loss.energy(beam_like(1.0e-4), sampler, 4096, 64)
    high, _ = energy_loss.energy(beam_like(2.0e-4), sampler, 4096, 64)
    assert high.detach().item() < low.detach().item()


def test_data_loss_is_zero_for_perfect_fit(bridge: SteelBridge):
    scales = analytic_scales(bridge)
    energy_loss = BridgeEnergyLoss(bridge=bridge, scales=scales)
    points = torch.tensor([[1.0, 0.5], [8.0, 0.9]])
    target = torch.tensor([[-1.0e-3, -2.0e-3], [-1.5e-3, -3.0e-3]])

    class Model(torch.nn.Module):
        def forward(self, xy: torch.Tensor) -> torch.Tensor:
            return target

    residual = energy_loss.data(Model(), points, target)
    assert torch.allclose(residual, torch.zeros_like(residual))
