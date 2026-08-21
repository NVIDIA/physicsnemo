# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pure-Torch Sobolev deformation with a P1 stiffness operator."""

from __future__ import annotations

import math

import torch
from torch.autograd.function import once_differentiable


def _scatter_cell_vertices(
    values: torch.Tensor,
    cells: torch.Tensor,
    num_points: int,
) -> torch.Tensor:
    """Accumulate cell-local vertex values into global point values."""

    batch_size, num_cells, num_cell_points = values.shape[:3]
    trailing_shape = values.shape[3:]
    flattened_width = math.prod(trailing_shape) if trailing_shape else 1

    values_flat = values.reshape(
        batch_size,
        num_cells * num_cell_points,
        flattened_width,
    )
    indices = (
        cells.reshape(1, -1, 1)
        .expand(batch_size, -1, flattened_width)
        .to(dtype=torch.long)
    )
    output = values.new_zeros((batch_size, num_points, flattened_width))
    output.scatter_add_(1, indices, values_flat)
    return output.reshape(batch_size, num_points, *trailing_shape)


def _assemble_p1_operators(
    points: torch.Tensor,
    cells: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Assemble a uniform vertex mass and cell-local P1 stiffness matrices."""

    batch_size, num_points, num_spatial_dims = points.shape
    num_cells, num_cell_points = cells.shape
    num_manifold_dims = num_cell_points - 1

    cell_points = points[:, cells.to(dtype=torch.long)]
    edge_matrix = cell_points[:, :, 1:, :] - cell_points[:, :, :1, :]
    gram = edge_matrix @ edge_matrix.transpose(-1, -2)
    gram_determinant = torch.linalg.det(gram)

    finite_positive = torch.isfinite(gram_determinant) & (gram_determinant > 0)
    valid_geometry = finite_positive.all()
    if torch.compiler.is_compiling():
        torch._assert_async(
            valid_geometry,
            "cells must be finite, nondegenerate simplices for Sobolev deformation",
        )
    else:
        from torch._subclasses.fake_tensor import is_fake

        if is_fake(points):
            torch._assert_async(
                valid_geometry,
                "cells must be finite, nondegenerate simplices for Sobolev deformation",
            )
        elif not bool(valid_geometry):
            raise ValueError(
                "cells must be finite, nondegenerate simplices for Sobolev deformation"
            )

    cell_measures = torch.sqrt(gram_determinant.clamp_min(0))
    factorial = torch.arange(
        1,
        num_manifold_dims + 1,
        dtype=points.dtype,
        device=points.device,
    ).prod()
    cell_measures = cell_measures / factorial

    basis_gradient_map = torch.cat(
        (
            -points.new_ones((1, num_manifold_dims)),
            torch.eye(
                num_manifold_dims,
                dtype=points.dtype,
                device=points.device,
            ),
        ),
        dim=0,
    )
    gram_inverse = torch.linalg.inv(gram)
    local_stiffness = cell_measures[..., None, None] * (
        basis_gradient_map[None, None] @ gram_inverse @ basis_gradient_map.T[None, None]
    )

    local_mass = (
        cell_measures[..., None].expand(-1, -1, num_cell_points).div(num_cell_points)
    )
    lumped_mass = _scatter_cell_vertices(local_mass, cells, num_points)
    stiffness_diagonal = _scatter_cell_vertices(
        local_stiffness.diagonal(dim1=-2, dim2=-1),
        cells,
        num_points,
    )

    connected = lumped_mass > 0
    connected_count = connected.sum(dim=1, keepdim=True).clamp_min(1)
    mean_mass = lumped_mass.sum(dim=1, keepdim=True) / connected_count
    mean_mass = torch.where(
        connected.any(dim=1, keepdim=True),
        mean_mass,
        torch.ones_like(mean_mass),
    )
    mass = mean_mass.expand_as(lumped_mass)
    return mass, local_stiffness, stiffness_diagonal


def _apply_stiffness(
    values: torch.Tensor,
    cells: torch.Tensor,
    local_stiffness: torch.Tensor,
) -> torch.Tensor:
    """Apply the assembled P1 stiffness matrix to a vertex field."""

    cell_values = values[:, cells.to(dtype=torch.long)]
    local_values = local_stiffness @ cell_values
    return _scatter_cell_vertices(local_values, cells, values.shape[1])


def _solve_helmholtz_pcg(
    right_hand_side: torch.Tensor,
    initial: torch.Tensor,
    mass: torch.Tensor,
    cells: torch.Tensor,
    local_stiffness: torch.Tensor,
    stiffness_diagonal: torch.Tensor,
    free_points: torch.Tensor,
    length_scale: float,
    max_iterations: int,
    tolerance: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve the uniform-mass Helmholtz system with Jacobi PCG."""

    length_scale_squared = length_scale * length_scale
    mass_vector = mass.unsqueeze(-1)
    free_vector = free_points.unsqueeze(-1)

    def apply_operator(values: torch.Tensor) -> torch.Tensor:
        free_values = torch.where(free_vector, values, torch.zeros_like(values))
        applied = mass_vector * free_values
        applied = applied + length_scale_squared * _apply_stiffness(
            free_values,
            cells,
            local_stiffness,
        )
        return torch.where(free_vector, applied, values)

    diagonal = mass + length_scale_squared * stiffness_diagonal
    diagonal = torch.where(free_points, diagonal, torch.ones_like(diagonal))
    diagonal = diagonal.clamp_min(torch.finfo(diagonal.dtype).tiny).unsqueeze(-1)

    solution = torch.where(free_vector, initial, torch.zeros_like(initial))
    residual = right_hand_side - apply_operator(solution)
    preconditioned = residual / diagonal
    direction = preconditioned

    reduce_dims = (-2, -1)
    residual_inner = (residual * preconditioned).sum(dim=reduce_dims)
    right_norm_squared = right_hand_side.square().sum(dim=reduce_dims)
    tolerance_squared = tolerance * tolerance
    tiny = torch.finfo(right_hand_side.dtype).tiny
    threshold = tolerance_squared * right_norm_squared.clamp_min(tiny)
    residual_norm_squared = residual.square().sum(dim=reduce_dims)
    active = residual_norm_squared > threshold
    breakdown = torch.zeros_like(active)

    for _ in range(max_iterations):
        active_field = active[:, None, None]
        direction = torch.where(active_field, direction, torch.zeros_like(direction))
        operator_direction = apply_operator(direction)
        denominator = (direction * operator_direction).sum(dim=reduce_dims)
        valid = (
            active
            & torch.isfinite(residual_inner)
            & torch.isfinite(denominator)
            & (denominator > tiny)
        )
        breakdown = breakdown | (active & ~valid)
        step = torch.where(
            valid,
            residual_inner / denominator.clamp_min(tiny),
            torch.zeros_like(denominator),
        )

        solution = solution + step[:, None, None] * direction
        residual = residual - step[:, None, None] * operator_direction
        preconditioned = residual / diagonal
        next_inner = (residual * preconditioned).sum(dim=reduce_dims)
        next_norm_squared = residual.square().sum(dim=reduce_dims)
        finite_update = torch.isfinite(next_inner) & torch.isfinite(next_norm_squared)
        breakdown = breakdown | (valid & ~finite_update)
        next_active = finite_update & (next_norm_squared > threshold)
        ratio = torch.where(
            valid & next_active,
            next_inner / residual_inner.clamp_min(tiny),
            torch.zeros_like(next_inner),
        )
        direction = preconditioned + ratio[:, None, None] * direction
        residual_inner = next_inner
        active = valid & next_active

    solution = torch.where(free_vector, solution, torch.zeros_like(solution))
    converged = ~(active | breakdown)
    return solution, converged


class _SobolevDisplacement(torch.autograd.Function):
    """Solve the Helmholtz system with an implicit first-order backward."""

    @staticmethod
    def forward(
        ctx,
        points: torch.Tensor,
        cells: torch.Tensor,
        displacement: torch.Tensor,
        free_points: torch.Tensor,
        length_scale: float,
        max_iterations: int,
        tolerance: float,
    ) -> torch.Tensor:
        mass, local_stiffness, stiffness_diagonal = _assemble_p1_operators(
            points,
            cells,
        )
        free_vector = free_points.unsqueeze(-1)
        weighted_displacement = torch.where(
            free_vector,
            displacement,
            torch.zeros_like(displacement),
        )
        right_hand_side = mass.unsqueeze(-1) * weighted_displacement
        solution, converged = _solve_helmholtz_pcg(
            right_hand_side=right_hand_side,
            initial=weighted_displacement,
            mass=mass,
            cells=cells,
            local_stiffness=local_stiffness,
            stiffness_diagonal=stiffness_diagonal,
            free_points=free_points,
            length_scale=length_scale,
            max_iterations=max_iterations,
            tolerance=tolerance,
        )
        convergence_message = (
            "Sobolev deformation PCG did not converge. Increase max_iterations "
            "or relax tolerance"
        )
        if torch.compiler.is_compiling():
            torch._assert_async(converged.all(), convergence_message)
        else:
            from torch._subclasses.fake_tensor import is_fake

            if is_fake(converged):
                torch._assert_async(converged.all(), convergence_message)
            elif not bool(converged.all()):
                raise RuntimeError(convergence_message)
        ctx.save_for_backward(
            points,
            cells,
            displacement,
            free_points,
            solution,
            mass,
            local_stiffness,
            stiffness_diagonal,
        )
        ctx.length_scale = length_scale
        ctx.max_iterations = max_iterations
        ctx.tolerance = tolerance
        return solution

    @staticmethod
    @once_differentiable
    def backward(
        ctx,
        output_adjoint: torch.Tensor,
    ) -> tuple[
        torch.Tensor | None,
        None,
        torch.Tensor | None,
        None,
        None,
        None,
        None,
    ]:
        (
            points,
            cells,
            displacement,
            free_points,
            solution,
            mass,
            local_stiffness,
            stiffness_diagonal,
        ) = ctx.saved_tensors
        free_vector = free_points.unsqueeze(-1)
        adjoint_right_hand_side = torch.where(
            free_vector,
            output_adjoint,
            torch.zeros_like(output_adjoint),
        )
        system_adjoint, converged = _solve_helmholtz_pcg(
            right_hand_side=adjoint_right_hand_side,
            initial=torch.zeros_like(output_adjoint),
            mass=mass,
            cells=cells,
            local_stiffness=local_stiffness,
            stiffness_diagonal=stiffness_diagonal,
            free_points=free_points,
            length_scale=ctx.length_scale,
            max_iterations=ctx.max_iterations,
            tolerance=ctx.tolerance,
        )
        convergence_message = (
            "Sobolev adjoint PCG did not converge. Increase max_iterations or "
            "relax tolerance"
        )
        if torch.compiler.is_compiling():
            torch._assert_async(converged.all(), convergence_message)
        else:
            from torch._subclasses.fake_tensor import is_fake

            if is_fake(converged):
                torch._assert_async(converged.all(), convergence_message)
            elif not bool(converged.all()):
                raise RuntimeError(convergence_message)

        points_adjoint = None
        if ctx.needs_input_grad[0]:

            def implicit_geometry_objective(
                differentiable_points: torch.Tensor,
            ) -> torch.Tensor:
                differentiable_mass, differentiable_stiffness, _ = (
                    _assemble_p1_operators(differentiable_points, cells)
                )
                weighted_displacement = torch.where(
                    free_vector,
                    displacement.detach(),
                    torch.zeros_like(displacement),
                )
                applied_solution = (
                    differentiable_mass.unsqueeze(-1) * solution.detach()
                    + ctx.length_scale
                    * ctx.length_scale
                    * _apply_stiffness(
                        solution.detach(),
                        cells,
                        differentiable_stiffness,
                    )
                )
                residual = (
                    differentiable_mass.unsqueeze(-1) * weighted_displacement
                    - applied_solution
                )
                residual = torch.where(
                    free_vector,
                    residual,
                    torch.zeros_like(residual),
                )
                return (system_adjoint.detach() * residual).sum()

            points_adjoint = torch.func.grad(implicit_geometry_objective)(
                points.detach()
            )

        displacement_adjoint = None
        if ctx.needs_input_grad[2]:
            displacement_adjoint = mass.unsqueeze(-1) * system_adjoint
            displacement_adjoint = torch.where(
                free_vector,
                displacement_adjoint,
                torch.zeros_like(displacement_adjoint),
            )

        return (
            points_adjoint,
            None,
            displacement_adjoint,
            None,
            None,
            None,
            None,
        )


def sobolev_deform_points_torch(
    points: torch.Tensor,
    cells: torch.Tensor,
    displacement: torch.Tensor,
    fixed_points: torch.Tensor | None,
    *,
    length_scale: float,
    max_iterations: int,
    tolerance: float,
) -> torch.Tensor:
    r"""Apply a differentiable uniform-mass P1 Sobolev deformation."""

    if fixed_points is None:
        free_points = torch.ones(
            points.shape[:2],
            dtype=torch.bool,
            device=points.device,
        )
    else:
        free_points = ~fixed_points

    free_vector = free_points.unsqueeze(-1)
    weighted_displacement = torch.where(
        free_vector,
        displacement,
        torch.zeros_like(displacement),
    )
    if length_scale == 0:
        return points + weighted_displacement

    filtered_displacement = _SobolevDisplacement.apply(
        points,
        cells,
        displacement,
        free_points,
        length_scale,
        max_iterations,
        tolerance,
    )
    return points + filtered_displacement


__all__ = ["sobolev_deform_points_torch"]
