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

"""Matrix-free plane-stress elasticity solver for FP-DDM patches."""

from __future__ import annotations

from collections.abc import Sequence

import torch


def _inner(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Return one Euclidean inner product per batch item."""

    return (first * second).flatten(1).sum(1)


def _batch_view(values: torch.Tensor, ndim: int) -> torch.Tensor:
    return values.reshape(values.shape[0], *((1,) * (ndim - 1)))


class Elasticity2DSolver:
    r"""Solve ``-div(sigma) = f`` for two-dimensional displacement.

    Displacement uses ``[B, 2, H, W]`` with components ``(u_y, u_x)`` and
    spacing is the explicit pair ``(dy, dx)``. Strain and stress use
    ``[B, 2, 2, H, W]`` with both tensor axes ordered ``(y, x)``.

    Equilibrium uses fully integrated bilinear quadrilateral elements, which
    give a coercive regular-grid operator without checkerboard modes. Material
    values are interpolated from the grid points at each quadrature point.
    BiCGSTAB solves the resulting matrix-free system with component-wise
    Dirichlet constraints.
    """

    stencil_radius = 1

    def __init__(
        self,
        spacing: Sequence[float] = (1.0, 1.0),
        max_iter: int = 1000,
        tolerance: float = 1.0e-10,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Configure spacing, convergence, device, and arithmetic type."""

        if len(spacing) != 2 or any(value <= 0.0 for value in spacing):
            raise ValueError("spacing must be the two positive values (dy, dx)")
        if max_iter <= 0 or tolerance < 0.0:
            raise ValueError("max_iter must be positive and tolerance non-negative")
        self.spacing = tuple(float(value) for value in spacing)
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.device = torch.device(
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.dtype = dtype

    def _displacement(self, values: torch.Tensor, name: str) -> torch.Tensor:
        values = values.to(self.device, self.dtype)
        if values.ndim != 4 or values.shape[1] != 2:
            raise ValueError(f"{name} must have shape [batch, 2, height, width]")
        if min(values.shape[-2:]) < 2:
            raise ValueError("each spatial axis needs at least two grid points")
        return values

    @staticmethod
    def _scalar_field(
        values: torch.Tensor | float,
        displacement: torch.Tensor,
        name: str,
    ) -> torch.Tensor:
        values = torch.as_tensor(
            values, dtype=displacement.dtype, device=displacement.device
        )
        if values.ndim == 3 and values.shape[0] == displacement.shape[0]:
            values = values.unsqueeze(1)
        target = (displacement.shape[0], 1, *displacement.shape[-2:])
        try:
            return torch.broadcast_to(values, target)
        except RuntimeError as error:
            raise ValueError(
                f"{name} must broadcast to [batch, 1, height, width]"
            ) from error

    @staticmethod
    def _derivative(values: torch.Tensor, axis: int, step: float) -> torch.Tensor:
        tensor_axis = axis + 2
        first = (
            values.select(tensor_axis, 1) - values.select(tensor_axis, 0)
        ).unsqueeze(tensor_axis) / step
        last = (
            values.select(tensor_axis, -1) - values.select(tensor_axis, -2)
        ).unsqueeze(tensor_axis) / step
        if values.shape[tensor_axis] == 2:
            return torch.cat((first, last), dim=tensor_axis)
        lower = [slice(None)] * values.ndim
        upper = [slice(None)] * values.ndim
        lower[tensor_axis] = slice(None, -2)
        upper[tensor_axis] = slice(2, None)
        centered = (values[tuple(upper)] - values[tuple(lower)]) / (2.0 * step)
        return torch.cat((first, centered, last), dim=tensor_axis)

    def strain(self, displacement: torch.Tensor) -> torch.Tensor:
        r"""Return the small strain ``(grad(u) + grad(u)^T) / 2``."""

        displacement = self._displacement(displacement, "displacement")
        gradient = torch.stack(
            [
                self._derivative(displacement, axis, step)
                for axis, step in enumerate(self.spacing)
            ],
            dim=2,
        )
        return 0.5 * (gradient + gradient.transpose(1, 2))

    def _material(
        self,
        displacement: torch.Tensor,
        young_modulus: torch.Tensor | float,
        poisson_ratio: torch.Tensor | float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        young_modulus = self._scalar_field(young_modulus, displacement, "young_modulus")
        poisson_ratio = self._scalar_field(poisson_ratio, displacement, "poisson_ratio")
        shear_modulus = young_modulus / (2.0 * (1.0 + poisson_ratio))
        lame_lambda = young_modulus * poisson_ratio / (1.0 - poisson_ratio.square())
        return lame_lambda, shear_modulus

    def stress(
        self,
        displacement: torch.Tensor,
        young_modulus: torch.Tensor | float,
        poisson_ratio: torch.Tensor | float,
    ) -> torch.Tensor:
        """Return the plane-stress Cauchy tensor."""

        displacement = self._displacement(displacement, "displacement")
        strain = self.strain(displacement)
        identity = torch.eye(
            2, dtype=displacement.dtype, device=displacement.device
        ).view(1, 2, 2, 1, 1)
        trace = strain.diagonal(dim1=1, dim2=2).sum(-1)
        lame_lambda, shear_modulus = self._material(
            displacement, young_modulus, poisson_ratio
        )
        return (
            2.0 * shear_modulus.unsqueeze(2) * strain
            + lame_lambda.unsqueeze(2) * trace.unsqueeze(1).unsqueeze(1) * identity
        )

    def von_mises(
        self,
        displacement: torch.Tensor,
        young_modulus: torch.Tensor | float,
        poisson_ratio: torch.Tensor | float,
    ) -> torch.Tensor:
        """Return plane-stress von Mises stress with shape ``[B, H, W]``."""

        stress = self.stress(displacement, young_modulus, poisson_ratio)
        sigma_yy = stress[:, 0, 0]
        sigma_xx = stress[:, 1, 1]
        sigma_yx = stress[:, 0, 1]
        return (
            (
                sigma_yy.square()
                - sigma_yy * sigma_xx
                + sigma_xx.square()
                + 3.0 * sigma_yx.square()
            )
            .clamp_min(0.0)
            .sqrt()
        )

    @staticmethod
    def _element_values(values: torch.Tensor) -> torch.Tensor:
        """Gather the four corner values of every grid cell."""

        return torch.stack(
            (
                values[..., :-1, :-1],
                values[..., :-1, 1:],
                values[..., 1:, :-1],
                values[..., 1:, 1:],
            ),
            dim=-1,
        )

    def _equilibrium(
        self,
        displacement: torch.Tensor,
        young_modulus: torch.Tensor,
        poisson_ratio: torch.Tensor,
    ) -> torch.Tensor:
        """Return the volume-normalized Q1 finite-element internal force."""

        dy, dx = self.spacing
        corners = self._element_values(displacement)
        young_corners = self._element_values(young_modulus)
        poisson_corners = self._element_values(poisson_ratio)
        element_force = torch.zeros_like(corners)
        quadrature_points = (-(3.0**-0.5), 3.0**-0.5)
        area_weight = 0.25 * dy * dx

        for eta in quadrature_points:
            for xi in quadrature_points:
                shape = (
                    displacement.new_tensor(
                        (
                            (1.0 - eta) * (1.0 - xi),
                            (1.0 - eta) * (1.0 + xi),
                            (1.0 + eta) * (1.0 - xi),
                            (1.0 + eta) * (1.0 + xi),
                        )
                    )
                    * 0.25
                )
                derivative_y = displacement.new_tensor(
                    (-(1.0 - xi), -(1.0 + xi), 1.0 - xi, 1.0 + xi)
                ) / (2.0 * dy)
                derivative_x = displacement.new_tensor(
                    (-(1.0 - eta), 1.0 - eta, -(1.0 + eta), 1.0 + eta)
                ) / (2.0 * dx)
                gradient_y = (corners * derivative_y).sum(-1)
                gradient_x = (corners * derivative_x).sum(-1)
                young = (young_corners * shape).sum(-1)
                poisson = (poisson_corners * shape).sum(-1)
                shear = young / (2.0 * (1.0 + poisson))
                lame = young * poisson / (1.0 - poisson.square())
                sigma_yy = (lame + 2.0 * shear) * gradient_y[:, 0:1]
                sigma_yy += lame * gradient_x[:, 1:2]
                sigma_xx = lame * gradient_y[:, 0:1]
                sigma_xx += (lame + 2.0 * shear) * gradient_x[:, 1:2]
                sigma_yx = shear * (gradient_x[:, 0:1] + gradient_y[:, 1:2])
                element_force[:, 0:1] += area_weight * (
                    sigma_yy.unsqueeze(-1) * derivative_y
                    + sigma_yx.unsqueeze(-1) * derivative_x
                )
                element_force[:, 1:2] += area_weight * (
                    sigma_yx.unsqueeze(-1) * derivative_y
                    + sigma_xx.unsqueeze(-1) * derivative_x
                )

        result = torch.zeros_like(displacement)
        result[..., :-1, :-1] += element_force[..., 0]
        result[..., :-1, 1:] += element_force[..., 1]
        result[..., 1:, :-1] += element_force[..., 2]
        result[..., 1:, 1:] += element_force[..., 3]
        volume = torch.full_like(displacement[:, :1], dy * dx)
        volume[..., (0, -1), :] *= 0.5
        volume[..., :, (0, -1)] *= 0.5
        return result / volume

    def residual(
        self,
        displacement: torch.Tensor,
        young_modulus: torch.Tensor | float,
        poisson_ratio: torch.Tensor | float,
        body_force: torch.Tensor,
    ) -> torch.Tensor:
        """Return ``-div(sigma) - body_force`` on the full grid."""

        displacement = self._displacement(displacement, "displacement")
        body_force = self._displacement(body_force, "body_force")
        young_modulus = self._scalar_field(young_modulus, displacement, "young_modulus")
        poisson_ratio = self._scalar_field(poisson_ratio, displacement, "poisson_ratio")
        return (
            self._equilibrium(displacement, young_modulus, poisson_ratio) - body_force
        )

    @torch.no_grad()
    def solve(
        self,
        young_modulus: torch.Tensor,
        poisson_ratio: torch.Tensor,
        body_force: torch.Tensor,
        displacement_bc: torch.Tensor,
        dirichlet_mask: torch.Tensor,
        initial: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Solve a batch with component-wise displacement constraints."""

        boundary = self._displacement(displacement_bc, "displacement_bc")
        body_force = self._displacement(body_force, "body_force")
        mask = dirichlet_mask.to(self.device).bool()
        if mask.shape != boundary.shape:
            raise ValueError("dirichlet_mask must match displacement_bc")
        initial = (
            torch.zeros_like(boundary)
            if initial is None
            else self._displacement(initial, "initial")
        )
        young_modulus = self._scalar_field(young_modulus, boundary, "young_modulus")
        poisson_ratio = self._scalar_field(poisson_ratio, boundary, "poisson_ratio")
        free = ~mask
        fixed = torch.where(mask, boundary, torch.zeros_like(boundary))
        lame_lambda, shear_modulus = self._material(
            boundary, young_modulus, poisson_ratio
        )
        dy, dx = self.spacing
        normal_modulus = lame_lambda + 2.0 * shear_modulus
        inverse_scale = torch.cat(
            (
                normal_modulus / dy**2 + shear_modulus / dx**2,
                shear_modulus / dy**2 + normal_modulus / dx**2,
            ),
            dim=1,
        ).reciprocal()

        def stiffness(values: torch.Tensor) -> torch.Tensor:
            return self._equilibrium(values, young_modulus, poisson_ratio)

        def apply(values: torch.Tensor) -> torch.Tensor:
            free_values = torch.where(free, values, torch.zeros_like(values))
            return torch.where(mask, values, inverse_scale * stiffness(free_values))

        fixed_residual = self.residual(
            fixed,
            young_modulus,
            poisson_ratio,
            body_force,
        )
        right_hand_side = torch.where(
            free, -inverse_scale * fixed_residual, torch.zeros_like(fixed_residual)
        )
        solution = torch.where(free, initial, torch.zeros_like(initial))
        residual = right_hand_side - apply(solution)
        shadow = residual.clone()
        direction = torch.zeros_like(residual)
        operator_direction = torch.zeros_like(residual)
        rhs_norm = _inner(right_hand_side, right_hand_side).sqrt()
        threshold = self.tolerance * rhs_norm.clamp_min(1.0)
        active = _inner(residual, residual).sqrt() > threshold
        rho_previous = torch.ones_like(rhs_norm)
        alpha = torch.ones_like(rhs_norm)
        omega = torch.ones_like(rhs_norm)
        tiny = torch.finfo(boundary.dtype).tiny

        for iteration in range(self.max_iter):
            if not bool(active.any()):
                break
            rho = _inner(shadow, residual)
            valid = rho.abs() > tiny
            active &= valid & torch.isfinite(rho)
            if iteration == 0:
                candidate = residual
            else:
                safe_rho_previous = torch.where(
                    rho_previous.abs() > tiny, rho_previous, torch.ones_like(rho)
                )
                safe_omega = torch.where(
                    omega.abs() > tiny, omega, torch.ones_like(omega)
                )
                beta = (rho / safe_rho_previous) * (alpha / safe_omega)
                candidate = residual + _batch_view(beta, residual.ndim) * (
                    direction - _batch_view(omega, residual.ndim) * operator_direction
                )
            direction = torch.where(
                _batch_view(active, residual.ndim), candidate, direction
            )
            operator_direction = apply(direction)
            denominator = _inner(shadow, operator_direction)
            valid = denominator.abs() > tiny
            active &= valid & torch.isfinite(denominator)
            safe_denominator = torch.where(
                valid, denominator, torch.ones_like(denominator)
            )
            alpha = torch.where(active, rho / safe_denominator, alpha)
            intermediate = residual - _batch_view(alpha, residual.ndim) * (
                operator_direction
            )
            solution += (
                _batch_view(active, residual.ndim)
                * _batch_view(alpha, residual.ndim)
                * direction
            )
            converged = _inner(intermediate, intermediate).sqrt() <= threshold
            residual = torch.where(
                _batch_view(active & converged, residual.ndim),
                intermediate,
                residual,
            )
            active &= ~converged
            if not bool(active.any()):
                break
            operator_intermediate = apply(intermediate)
            denominator = _inner(operator_intermediate, operator_intermediate)
            valid = denominator > tiny
            active &= valid & torch.isfinite(denominator)
            omega_candidate = _inner(operator_intermediate, intermediate) / (
                denominator.clamp_min(tiny)
            )
            active &= torch.isfinite(omega_candidate) & (omega_candidate.abs() > tiny)
            omega = torch.where(active, omega_candidate, omega)
            solution += (
                _batch_view(active, residual.ndim)
                * _batch_view(omega, residual.ndim)
                * intermediate
            )
            candidate = intermediate - _batch_view(omega, residual.ndim) * (
                operator_intermediate
            )
            residual = torch.where(
                _batch_view(active, residual.ndim), candidate, residual
            )
            active &= _inner(residual, residual).sqrt() > threshold
            rho_previous = rho

        final_residual = right_hand_side - apply(solution)
        residual_norm = _inner(final_residual, final_residual).sqrt()
        if not bool(
            torch.all(torch.isfinite(residual_norm) & (residual_norm <= threshold))
        ):
            raise RuntimeError("elasticity solve did not converge")
        return fixed + torch.where(free, solution, torch.zeros_like(solution))


__all__ = ["Elasticity2DSolver"]
