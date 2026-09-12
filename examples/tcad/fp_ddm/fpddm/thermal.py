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

"""Matrix-free finite-volume reference solver for the thermal equation."""

from __future__ import annotations

import torch


def _harmonic_mean(
    first: torch.Tensor, second: torch.Tensor, eps: float = 1.0e-12
) -> torch.Tensor:
    return 2.0 * first * second / (first + second + eps)


class Heat2DSolver:
    """Solve ``-div(k grad(T)) = q`` with preconditioned conjugate gradients."""

    def __init__(
        self,
        dx: float = 1.0,
        dy: float = 1.0,
        max_iter: int = 1000,
        tolerance: float = 1.0e-12,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Configure grid spacing, convergence, device, and arithmetic type."""

        self.dx = dx
        self.dy = dy
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.device = torch.device(
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.dtype = dtype

    def solve(
        self,
        conductivity: torch.Tensor,
        heat_source: torch.Tensor,
        temperature_bc: torch.Tensor,
        dirichlet_mask: torch.Tensor,
        neumann_mask: torch.Tensor,
        neumann_value: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Solve a batch of rectangular thermal systems and return temperature."""

        conductivity = conductivity.to(self.device, self.dtype)
        heat_source = heat_source.to(self.device, self.dtype)
        temperature_bc = temperature_bc.to(self.device, self.dtype)
        neumann_value = (
            neumann_value
            if neumann_value is not None
            else torch.zeros_like(heat_source)
        ).to(self.device, self.dtype)
        dirichlet = dirichlet_mask.to(self.device).bool()
        neumann = neumann_mask.to(self.device).bool() & ~dirichlet

        batch, height, width = conductivity.shape
        dx2 = self.dx * self.dx
        dy2 = self.dy * self.dy

        # Harmonic face values preserve normal heat flux across material jumps.
        east = torch.zeros_like(conductivity)
        west = torch.zeros_like(conductivity)
        north = torch.zeros_like(conductivity)
        south = torch.zeros_like(conductivity)
        east[:, :, :-1] = _harmonic_mean(
            conductivity[:, :, :-1], conductivity[:, :, 1:]
        )
        west[:, :, 1:] = east[:, :, :-1]
        north[:, 1:, :] = _harmonic_mean(
            conductivity[:, 1:, :], conductivity[:, :-1, :]
        )
        south[:, :-1, :] = north[:, 1:, :]

        rows = torch.arange(height, device=self.device).view(1, height, 1)
        columns = torch.arange(width, device=self.device).view(1, 1, width)
        left = neumann & (columns == 0)
        right = neumann & (columns == width - 1)
        top = neumann & (rows == 0)
        bottom = neumann & (rows == height - 1)
        has_left = bool(left.any())
        has_right = bool(right.any())
        has_top = bool(top.any())
        has_bottom = bool(bottom.any())

        # Boundary rows replace the thermal equation with the prescribed data.
        rhs = heat_source.clone()
        rhs[dirichlet] = temperature_bc[dirichlet]
        for mask, active in (
            (left, has_left),
            (right, has_right),
            (top, has_top),
            (bottom, has_bottom),
        ):
            if active:
                rhs[mask] = neumann_value[mask]

        diagonal = (east + west) / dx2 + (north + south) / dy2
        diagonal = diagonal.clone()
        diagonal[dirichlet] = 1.0
        if has_left:
            diagonal[left] = east[left] / self.dx
        if has_right:
            diagonal[right] = west[right] / self.dx
        if has_top:
            diagonal[top] = south[top] / self.dy
        if has_bottom:
            diagonal[bottom] = north[bottom] / self.dy
        inverse_diagonal = 1.0 / diagonal.clamp_min(1.0e-12)

        # Applying the stencil directly avoids assembling one sparse matrix per
        # generated material field.
        def apply_operator(values: torch.Tensor) -> torch.Tensor:
            shifted_right = torch.roll(values, -1, 2)
            shifted_left = torch.roll(values, 1, 2)
            shifted_down = torch.roll(values, -1, 1)
            shifted_up = torch.roll(values, 1, 1)
            result = -(
                east * (shifted_right - values) / dx2
                + west * (shifted_left - values) / dx2
                + south * (shifted_down - values) / dy2
                + north * (shifted_up - values) / dy2
            )
            result[dirichlet] = values[dirichlet]
            if has_left:
                coefficient = east[left] / self.dx
                result[left] = coefficient * (values[left] - shifted_right[left])
            if has_right:
                coefficient = west[right] / self.dx
                result[right] = coefficient * (values[right] - shifted_left[right])
            if has_top:
                coefficient = south[top] / self.dy
                result[top] = coefficient * (values[top] - shifted_down[top])
            if has_bottom:
                coefficient = north[bottom] / self.dy
                result[bottom] = coefficient * (values[bottom] - shifted_up[bottom])
            return result

        # All systems advance together, while the active mask freezes batches
        # that converge before their neighbors.
        temperature = torch.zeros_like(temperature_bc)
        temperature[dirichlet] = temperature_bc[dirichlet]
        residual = rhs - apply_operator(temperature)
        preconditioned = inverse_diagonal * residual
        direction = preconditioned.clone()
        residual_dot = (residual * preconditioned).flatten(1).sum(1)
        rhs_norm = torch.linalg.vector_norm(rhs.flatten(1), dim=1).clamp_min(1.0e-12)
        active = torch.ones(batch, dtype=torch.bool, device=self.device)

        for _ in range(self.max_iter):
            operator_direction = apply_operator(direction)
            denominator = (direction * operator_direction).flatten(1).sum(1)
            invalid = active & ((denominator <= 1.0e-20) | ~torch.isfinite(denominator))
            if invalid.any():
                direction[invalid] = residual[invalid]
                operator_direction = apply_operator(direction)
                denominator = (direction * operator_direction).flatten(1).sum(1)

            safe_denominator = denominator.clamp_min(1.0e-12)
            alpha = torch.where(active, residual_dot / safe_denominator, 0.0)
            temperature += alpha.view(batch, 1, 1) * direction
            residual -= alpha.view(batch, 1, 1) * operator_direction
            preconditioned = inverse_diagonal * residual
            new_residual_dot = (residual * preconditioned).flatten(1).sum(1)
            relative_residual = new_residual_dot.clamp_min(0).sqrt() / rhs_norm
            active &= relative_residual >= self.tolerance
            active &= torch.isfinite(relative_residual)
            if not active.any():
                break
            beta = torch.where(
                active,
                new_residual_dot / residual_dot.clamp_min(1.0e-20),
                0.0,
            ).view(batch, 1, 1)
            direction = preconditioned + beta * direction
            direction[~active] = 0.0
            residual_dot = new_residual_dot
        return temperature
