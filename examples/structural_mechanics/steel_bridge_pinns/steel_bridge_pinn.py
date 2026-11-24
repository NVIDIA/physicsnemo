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

"""Physics-informed neural network for the steel bridge plane-stress problem.

The network represents the displacement field ``(u_x, u_y)`` of the bridge web as a
function of the physical coordinates.  Physics enters through the **total potential
energy** of linear elasticity,

.. math::
    \\Pi[u] = \\tfrac{1}{2}\\int_\\Omega \\varepsilon : \\mathbb{C} : \\varepsilon \\, t
        \\, dA + \\int_\\Omega \\rho g \\, u_y \\, t \\, dA
        + \\int_{\\Gamma_t} \\boldsymbol{t} \\cdot \\boldsymbol{u} \\, t \\, ds ,

which is minimised with a Monte-Carlo estimate of the integrals.  This variational
(energy) form only needs first derivatives of the network.  The strong form of the
equilibrium equations was also tried - its residuals on a slender domain
(:math:`L/H = 10` here) are dominated by transverse derivatives and the optimisation
collapses to the trivial zero field, see the notes in ``README.md``.

The bearing pads are imposed *exactly* through smooth multiplicative masks on the
network output.  A handful of CAE measurements can be blended into the same objective,
either as a data-consistency term or as a supervised warm start of the network.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
from steel_bridge_cae import SteelBridge

from physicsnemo.models.mlp.fully_connected import FullyConnected

__all__ = [
    "LossScales",
    "analytic_scales",
    "BridgePINN",
    "BridgeSampler",
    "BridgeEnergyLoss",
    "stress_tensor",
    "von_mises_stress",
]


def von_mises_stress(stress: torch.Tensor) -> torch.Tensor:
    """Plane-stress von Mises equivalent stress for a ``(N, 3)`` tensor."""
    s_xx, s_yy, s_xy = stress[..., 0], stress[..., 1], stress[..., 2]
    return torch.sqrt(s_xx**2 - s_xx * s_yy + s_yy**2 + 3.0 * s_xy**2)


def _displacement_and_gradients(
    model: nn.Module, xy: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Displacement and its two gradient components at ``xy``."""
    xy = xy.detach().clone().requires_grad_(True)
    displacement = model(xy)
    grad_u = torch.autograd.grad(
        displacement[:, 0:1],
        xy,
        grad_outputs=torch.ones_like(displacement[:, 0:1]),
        create_graph=True,
    )[0]
    grad_v = torch.autograd.grad(
        displacement[:, 1:2],
        xy,
        grad_outputs=torch.ones_like(displacement[:, 1:2]),
        create_graph=True,
    )[0]
    return displacement, grad_u, grad_v


def stress_tensor(
    model: nn.Module, xy: torch.Tensor, bridge: SteelBridge
) -> torch.Tensor:
    """Plane-stress tensor ``(sigma_xx, sigma_yy, sigma_xy)`` at ``xy``.

    Derivatives are taken by automatic differentiation of the network, so stresses are
    available anywhere in the domain without a mesh.
    """
    _, grad_u, grad_v = _displacement_and_gradients(model, xy)
    e_xx = grad_u[:, 0:1]
    e_yy = grad_v[:, 1:2]
    g_xy = grad_u[:, 1:2] + grad_v[:, 0:1]

    e, nu = bridge.youngs_modulus, bridge.poisson_ratio
    factor = e / (1.0 - nu**2)
    s_xx = factor * (e_xx + nu * e_yy)
    s_yy = factor * (e_yy + nu * e_xx)
    s_xy = bridge.shear_modulus * g_xy
    return torch.cat([s_xx, s_yy, s_xy], dim=-1)


def strain_energy_density(
    model: nn.Module, xy: torch.Tensor, bridge: SteelBridge
) -> tuple[torch.Tensor, torch.Tensor]:
    """Strain energy density ``1/2 sigma : epsilon`` and displacement at ``xy``."""
    displacement, grad_u, grad_v = _displacement_and_gradients(model, xy)
    e_xx = grad_u[:, 0:1]
    e_yy = grad_v[:, 1:2]
    g_xy = grad_u[:, 1:2] + grad_v[:, 0:1]

    e, nu = bridge.youngs_modulus, bridge.poisson_ratio
    factor = e / (1.0 - nu**2)
    s_xx = factor * (e_xx + nu * e_yy)
    s_yy = factor * (e_yy + nu * e_xx)
    s_xy = bridge.shear_modulus * g_xy
    density = 0.5 * (s_xx * e_xx + s_yy * e_yy + s_xy * g_xy)
    return density, displacement


@dataclass(frozen=True)
class LossScales:
    """Reference scales used to non-dimensionalise the objective terms."""

    displacement: float
    stress: float
    traction: float
    energy: float


def analytic_scales(bridge: SteelBridge) -> LossScales:
    """Scales from Euler-Bernoulli theory, i.e. computed without any CAE data."""
    theory = bridge.beam_theory()
    displacement = max(theory.midspan_deflection, 1e-12)
    energy = sum(
        bridge.patch_traction * bridge.thickness * (stop - start) * displacement
        for start, stop in bridge.load_patch_bounds
    )
    return LossScales(
        displacement=displacement,
        stress=max(theory.max_bending_stress, 1e-12),
        traction=max(bridge.patch_traction, 1e-12),
        energy=max(energy, 1e-12),
    )


class BridgePINN(nn.Module):
    """Displacement-field network for the bridge web.

    Coordinates are normalised to ``[-1, 1]`` before entering the MLP and the output is
    scaled to metres.  The bearing pads are enforced *exactly* by multiplying the
    network output with smooth masks that vanish on the pads, so no Dirichlet penalty
    term (and no associated weight trade-off) is required.
    """

    def __init__(
        self,
        bridge: SteelBridge,
        displacement_scale: float,
        num_layers: int = 5,
        layer_size: int = 48,
        activation: str = "tanh",
        skip_connections: bool = False,
        bearing_halo: float = 0.1,
        zero_output_init: bool = True,
    ):
        super().__init__()
        self.span = bridge.span
        self.depth = bridge.depth
        self.bearing_length = bridge.bearing_length
        self.bearing_halo = bearing_halo
        self.displacement_scale = displacement_scale
        self.net = FullyConnected(
            in_features=2,
            out_features=2,
            num_layers=num_layers,
            layer_size=layer_size,
            activation_fn=activation,
            skip_connections=skip_connections,
        )
        if zero_output_init:
            # A random initial field has residuals that are orders of magnitude larger
            # than the load terms; starting from the stress-free state keeps the first
            # optimisation steps well conditioned.
            nn.init.zeros_(self.net.final_layer.linear.weight)
            nn.init.zeros_(self.net.final_layer.linear.bias)

    def bearing_masks(self, xy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Smooth masks that equal one on the left and right bearing pads."""
        above = torch.clamp(xy[:, 1] - self.bearing_length, min=0.0) ** 2
        halo = self.bearing_halo**2
        left = torch.exp(-(xy[:, 0] ** 2 + above) / halo)[:, None]
        right = torch.exp(-((self.span - xy[:, 0]) ** 2 + above) / halo)[:, None]
        return left, right

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        """Map physical coordinates ``(N, 2)`` to displacement ``(N, 2)`` in metres."""
        normalised = torch.stack(
            [2.0 * xy[:, 0] / self.span - 1.0, 2.0 * xy[:, 1] / self.depth - 1.0],
            dim=-1,
        )
        raw = self.net(normalised) * self.displacement_scale
        left, right = self.bearing_masks(xy)
        return torch.cat(
            [(1.0 - left) * raw[:, 0:1], (1.0 - left) * (1.0 - right) * raw[:, 1:2]],
            dim=-1,
        )


class BridgeSampler:
    """Monte-Carlo quadrature points for the energy functional."""

    def __init__(self, bridge: SteelBridge, seed: int = 0):
        self.bridge = bridge
        self.generator = torch.Generator().manual_seed(seed)

    def interior(self, n_points: int) -> torch.Tensor:
        """Uniform points inside the web."""
        uniform = torch.rand(n_points, 2, generator=self.generator)
        return torch.stack(
            [uniform[:, 0] * self.bridge.span, uniform[:, 1] * self.bridge.depth], -1
        )

    def deck_patches(
        self, n_per_patch: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Points on the loaded deck patches with their surface weights and traction.

        Returns
        -------
        points, weights, traction
            ``points`` lie on ``y = H``, ``weights`` are the edge lengths each point
            represents (``ds``) and ``traction`` is the imposed pressure in Pa.
        """
        points, weights, traction = [], [], []
        for start, stop in self.bridge.load_patch_bounds:
            uniform = torch.rand(n_per_patch, 2, generator=self.generator)
            points.append(
                torch.stack(
                    [
                        start + (stop - start) * uniform[:, 0],
                        torch.full((n_per_patch,), self.bridge.depth),
                    ],
                    -1,
                )
            )
            weights.append(torch.full((n_per_patch,), (stop - start) / n_per_patch))
            traction.append(
                torch.tensor([0.0, -self.bridge.patch_traction]).repeat(n_per_patch, 1)
            )
        return torch.cat(points), torch.cat(weights), torch.cat(traction)

    def bearings(self, n_points: int) -> torch.Tensor:
        """Points inside the two bearing pads (used to verify the hard constraints)."""
        half = n_points // 2
        left = torch.stack(
            [
                torch.zeros(half),
                self.bridge.bearing_length * torch.rand(half, generator=self.generator),
            ],
            -1,
        )
        right = torch.stack(
            [
                torch.full((n_points - half,), self.bridge.span),
                self.bridge.bearing_length
                * torch.rand(n_points - half, generator=self.generator),
            ],
            -1,
        )
        return torch.cat([left, right])

    @staticmethod
    def data_points(points, displacement) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert NumPy sensor readings to tensors."""
        return (
            torch.as_tensor(points, dtype=torch.float32),
            torch.as_tensor(displacement, dtype=torch.float32),
        )


@dataclass
class BridgeEnergyLoss:
    """Total potential energy, data term and diagnostics of the bridge PINN."""

    bridge: SteelBridge
    scales: LossScales
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))

    @property
    def volume(self) -> float:
        return self.bridge.span * self.bridge.depth * self.bridge.thickness

    def energy(
        self, model: nn.Module, sampler: BridgeSampler, n_interior: int, n_patch: int
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Non-dimensional total potential energy and its contributions.

        Note the sign convention: the strain energy is positive, whereas the self
        weight and the deck load contribute negative terms, so a lower (more negative)
        total energy means a more compliant, load-carrying field.
        """
        interior = sampler.interior(n_interior)
        density, displacement = strain_energy_density(model, interior, self.bridge)
        strain_energy = density.mean() * self.volume

        self_weight = (
            self.bridge.density * self.bridge.gravity * displacement[:, 1:2]
        ).mean() * self.volume

        points, weights, traction = sampler.deck_patches(n_patch)
        deck = model(points)
        external = ((traction * deck).sum(-1) * weights * self.bridge.thickness).sum()

        # Total potential energy: strain energy + gravitational potential - work done
        # by the applied deck load.
        total = (strain_energy + self_weight - external) / self.scales.energy
        contributions = {
            "strain_energy": float((strain_energy / self.scales.energy).detach()),
            "self_weight": float((self_weight / self.scales.energy).detach()),
            "external_work": float((external / self.scales.energy).detach()),
        }
        return total, contributions

    def data(
        self, model: nn.Module, xy: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Non-dimensional mismatch with the sparse CAE measurements."""
        return (model(xy) - target) / self.scales.displacement

    # ------------------------------------------------------------- diagnostics
    def stress(self, model: nn.Module, xy: torch.Tensor) -> torch.Tensor:
        """Plane-stress tensor in Pa at ``xy``."""
        return stress_tensor(model, xy, self.bridge)

    def von_mises(self, model: nn.Module, xy: torch.Tensor) -> torch.Tensor:
        """Von Mises stress in Pa at ``xy``."""
        return von_mises_stress(stress_tensor(model, xy, self.bridge))

    def energy_balance(
        self, model: nn.Module, sampler: BridgeSampler, n_interior: int, n_patch: int
    ) -> dict[str, float]:
        """Ratio ``2 U / |W_ext|``, equal to one for a field in equilibrium."""
        _, contributions = self.energy(model, sampler, n_interior, n_patch)
        external = contributions["external_work"]
        if external == 0.0:
            return {"energy_balance": float("nan")}
        return {"energy_balance": 2.0 * contributions["strain_energy"] / abs(external)}
