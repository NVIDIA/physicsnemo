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

"""Pure-Torch local/global template-fitting implementation."""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch.autograd.function import once_differentiable

from ..point_to_mesh_distance import point_to_mesh_distance
from ._polar import proper_rotation


def _unique_edges(triangles: torch.Tensor) -> torch.Tensor:
    """Return canonical unique undirected edges of a triangle mesh."""

    edges = torch.cat(
        (
            triangles[:, (0, 1)],
            triangles[:, (1, 2)],
            triangles[:, (2, 0)],
        ),
        dim=0,
    )
    return torch.sort(edges, dim=-1).values.unique(dim=0)


def _laplacian_apply(values: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    """Apply the uniform graph Laplacian without materializing a sparse matrix."""

    edge_start, edge_end = edges.unbind(dim=-1)
    differences = values.index_select(0, edge_start) - values.index_select(0, edge_end)
    output = torch.zeros_like(values)
    output = output.index_add(0, edge_start, differences)
    return output.index_add(0, edge_end, -differences)


def _system_apply(
    values: torch.Tensor,
    edges: torch.Tensor,
    fit_weight: float,
    arap_weight: float,
) -> torch.Tensor:
    """Apply ``fit_weight * I + 2 * arap_weight * L``."""

    return fit_weight * values + 2.0 * arap_weight * _laplacian_apply(values, edges)


def _system_diagonal(
    num_points: int,
    edges: torch.Tensor,
    dtype: torch.dtype,
    fit_weight: float,
    arap_weight: float,
) -> torch.Tensor:
    """Return the Jacobi diagonal of the matrix-free fitting system."""

    degree = torch.zeros(num_points, dtype=dtype, device=edges.device)
    ones = torch.ones(edges.shape[0], dtype=dtype, device=edges.device)
    degree = degree.index_add(0, edges[:, 0], ones)
    degree = degree.index_add(0, edges[:, 1], ones)
    return fit_weight + 2.0 * arap_weight * degree


@torch.no_grad()
def _cg_solve_eager(
    right_hand_side: torch.Tensor,
    edges: torch.Tensor,
    fit_weight: float,
    arap_weight: float,
    tolerance: float,
    max_iterations: int,
) -> torch.Tensor:
    """Solve the shared-coordinate SPD system with Jacobi-preconditioned CG."""

    solution = torch.zeros_like(right_hand_side)
    residual = right_hand_side.clone()
    right_hand_side_norm = torch.linalg.vector_norm(right_hand_side)
    right_hand_side_norm_value = float(right_hand_side_norm)
    if right_hand_side_norm_value == 0.0:
        return solution
    convergence_threshold = tolerance * right_hand_side_norm_value
    if float(torch.linalg.vector_norm(residual)) <= convergence_threshold:
        return solution

    diagonal = _system_diagonal(
        right_hand_side.shape[0],
        edges,
        right_hand_side.dtype,
        fit_weight,
        arap_weight,
    )
    preconditioned = residual / diagonal.unsqueeze(-1)
    direction = preconditioned.clone()
    residual_dot = torch.sum(residual * preconditioned)

    for _ in range(max_iterations):
        matrix_direction = _system_apply(direction, edges, fit_weight, arap_weight)
        denominator = torch.sum(direction * matrix_direction)
        denominator_value = float(denominator)
        if not math.isfinite(denominator_value) or denominator_value <= 0.0:
            raise RuntimeError(
                "conjugate-gradient fitting solve encountered a non-SPD operator"
            )

        step_size = residual_dot / denominator
        solution = solution + step_size * direction
        residual = residual - step_size * matrix_direction
        if float(torch.linalg.vector_norm(residual)) <= convergence_threshold:
            return solution

        preconditioned = residual / diagonal.unsqueeze(-1)
        next_residual_dot = torch.sum(residual * preconditioned)
        direction = preconditioned + (next_residual_dot / residual_dot) * direction
        residual_dot = next_residual_dot

    relative_residual = (
        float(torch.linalg.vector_norm(residual)) / right_hand_side_norm_value
    )
    raise RuntimeError(
        "conjugate-gradient fitting solve did not converge within "
        f"{max_iterations} iterations (scaled residual {relative_residual:.3e})"
    )


@torch.no_grad()
def _cg_solve_device(
    right_hand_side: torch.Tensor,
    edges: torch.Tensor,
    fit_weight: float,
    arap_weight: float,
    tolerance: float,
    max_iterations: int,
) -> torch.Tensor:
    """Solve CG with tensor-controlled iteration and no host scalar readback."""

    diagonal = _system_diagonal(
        right_hand_side.shape[0],
        edges,
        right_hand_side.dtype,
        fit_weight,
        arap_weight,
    )
    solution = torch.zeros_like(right_hand_side)
    residual = right_hand_side.clone()
    preconditioned = residual / diagonal.unsqueeze(-1)
    direction = preconditioned.clone()
    residual_dot = torch.sum(residual * preconditioned)
    right_hand_side_norm = torch.linalg.vector_norm(right_hand_side)
    convergence_threshold = tolerance * right_hand_side_norm
    converged = right_hand_side_norm <= convergence_threshold
    valid = (
        torch.isfinite(right_hand_side).all()
        & torch.isfinite(diagonal).all()
        & torch.all(diagonal > 0.0)
        & (converged | (torch.isfinite(residual_dot) & (residual_dot > 0.0)))
    )
    iteration = torch.zeros((), dtype=torch.int64, device=right_hand_side.device)

    def condition(
        iteration,
        solution,
        residual,
        direction,
        residual_dot,
        converged,
        valid,
    ):
        del solution, residual, direction, residual_dot
        return (iteration < max_iterations) & (~converged) & valid

    def body(
        iteration,
        solution,
        residual,
        direction,
        residual_dot,
        converged,
        valid,
    ):
        del converged
        matrix_direction = _system_apply(
            direction,
            edges,
            fit_weight,
            arap_weight,
        )
        denominator = torch.sum(direction * matrix_direction)
        denominator_valid = (
            torch.isfinite(denominator)
            & (denominator > 0.0)
            & torch.isfinite(residual_dot)
            & (residual_dot > 0.0)
        )
        safe_denominator = torch.where(
            denominator_valid,
            denominator,
            torch.ones_like(denominator),
        )
        step_size = residual_dot / safe_denominator
        next_solution = solution + step_size * direction
        next_residual = residual - step_size * matrix_direction
        next_residual_norm = torch.linalg.vector_norm(next_residual)
        next_converged = next_residual_norm <= convergence_threshold

        next_preconditioned = next_residual / diagonal.unsqueeze(-1)
        next_residual_dot = torch.sum(next_residual * next_preconditioned)
        safe_residual_dot = torch.where(
            residual_dot > 0.0,
            residual_dot,
            torch.ones_like(residual_dot),
        )
        beta = torch.where(
            next_converged,
            torch.zeros_like(next_residual_dot),
            next_residual_dot / safe_residual_dot,
        )
        next_direction = torch.where(
            next_converged,
            torch.zeros_like(direction),
            next_preconditioned + beta * direction,
        )
        next_valid = (
            valid
            & denominator_valid
            & torch.isfinite(step_size)
            & torch.isfinite(next_residual_norm)
            & torch.isfinite(next_residual_dot)
            & torch.isfinite(beta)
            & (next_converged | (next_residual_dot > 0.0))
        )
        return (
            iteration + 1,
            next_solution,
            next_residual,
            next_direction,
            next_residual_dot,
            next_converged,
            next_valid,
        )

    (
        _,
        solution,
        _,
        _,
        _,
        _,
        valid,
    ) = torch.while_loop(
        condition,
        body,
        (
            iteration,
            solution,
            residual,
            direction,
            residual_dot,
            converged,
            valid,
        ),
    )
    # Recompute the residual from the returned solution instead of trusting the
    # loop-carried convergence flag. AOTAutograd/Inductor can otherwise retain
    # a stale flag when this solve is inlined into a larger compiled backward,
    # even though the returned solution satisfies the requested tolerance.
    final_residual = right_hand_side - _system_apply(
        solution,
        edges,
        fit_weight,
        arap_weight,
    )
    converged = torch.linalg.vector_norm(final_residual) <= convergence_threshold
    torch._assert_async(
        valid,
        "conjugate-gradient fitting solve encountered non-finite values or a "
        "non-SPD operator",
    )
    torch._assert_async(
        converged,
        "conjugate-gradient fitting solve did not converge within "
        f"{max_iterations} iterations",
    )
    return solution


def _cg_solve(
    right_hand_side: torch.Tensor,
    edges: torch.Tensor,
    fit_weight: float,
    arap_weight: float,
    tolerance: float,
    max_iterations: int,
) -> torch.Tensor:
    """Use eager diagnostics on CPU and tensor control on accelerator devices."""

    if right_hand_side.device.type == "cpu" and not torch.compiler.is_compiling():
        return _cg_solve_eager(
            right_hand_side,
            edges,
            fit_weight,
            arap_weight,
            tolerance,
            max_iterations,
        )
    return _cg_solve_device(
        right_hand_side,
        edges,
        fit_weight,
        arap_weight,
        tolerance,
        max_iterations,
    )


class _ImplicitSPDLinearSolve(torch.autograd.Function):
    """SPD solve whose first backward is a second matrix-free adjoint solve."""

    @staticmethod
    def forward(
        ctx,
        right_hand_side: torch.Tensor,
        edges: torch.Tensor,
        fit_weight: float,
        arap_weight: float,
        tolerance: float,
        max_iterations: int,
    ) -> torch.Tensor:
        """Run graph-free CG and save the constant operator description."""

        ctx.save_for_backward(edges)
        ctx.fit_weight = fit_weight
        ctx.arap_weight = arap_weight
        ctx.tolerance = tolerance
        ctx.max_iterations = max_iterations
        return _cg_solve(
            right_hand_side,
            edges,
            fit_weight,
            arap_weight,
            tolerance,
            max_iterations,
        )

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: torch.Tensor):
        """Apply ``A^{-T}`` to the output cotangent."""

        (edges,) = ctx.saved_tensors
        grad_right_hand_side = _cg_solve(
            grad_output,
            edges,
            ctx.fit_weight,
            ctx.arap_weight,
            ctx.tolerance,
            ctx.max_iterations,
        )
        return grad_right_hand_side, None, None, None, None, None


def _implicit_spd_solve(
    right_hand_side: torch.Tensor,
    edges: torch.Tensor,
    fit_weight: float,
    arap_weight: float,
    tolerance: float,
    max_iterations: int,
) -> torch.Tensor:
    """Call the custom first-order differentiable global solve."""

    return _ImplicitSPDLinearSolve.apply(
        right_hand_side,
        edges,
        fit_weight,
        arap_weight,
        tolerance,
        max_iterations,
    )


def _local_rotations(
    deformed_points: torch.Tensor,
    reference_points: torch.Tensor,
    edges: torch.Tensor,
) -> torch.Tensor:
    """Fit one proper rotation to every uniform-weight point one-ring."""

    edge_start, edge_end = edges.unbind(dim=-1)
    deformed_edges = deformed_points.index_select(
        0, edge_start
    ) - deformed_points.index_select(0, edge_end)
    reference_edges = reference_points.index_select(
        0, edge_start
    ) - reference_points.index_select(0, edge_end)
    outer_products = deformed_edges.unsqueeze(-1) * reference_edges.unsqueeze(-2)
    covariance = torch.zeros(
        (reference_points.shape[0], 3, 3),
        dtype=reference_points.dtype,
        device=reference_points.device,
    )
    covariance = covariance.index_add(0, edge_start, outer_products)
    covariance = covariance.index_add(0, edge_end, outer_products)
    return proper_rotation(covariance)


def _arap_right_hand_side(
    reference_points: torch.Tensor,
    rotations: torch.Tensor,
    edges: torch.Tensor,
) -> torch.Tensor:
    """Assemble ``B(R)`` for the uniform-weight ARAP global step."""

    edge_start, edge_end = edges.unbind(dim=-1)
    reference_edges = reference_points.index_select(
        0, edge_start
    ) - reference_points.index_select(0, edge_end)
    average_rotation = 0.5 * (
        rotations.index_select(0, edge_start) + rotations.index_select(0, edge_end)
    )
    edge_values = torch.matmul(average_rotation, reference_edges.unsqueeze(-1)).squeeze(
        -1
    )
    output = torch.zeros_like(reference_points)
    output = output.index_add(0, edge_start, edge_values)
    return output.index_add(0, edge_end, -edge_values)


def fit_template_points_torch(
    template_points: torch.Tensor,
    template_triangles: torch.Tensor,
    target_points: torch.Tensor,
    target_triangles: torch.Tensor,
    fit_weight: float,
    arap_weight: float,
    steps: int,
    cg_tolerance: float,
    cg_max_iterations: int,
    point_implementation: Literal["torch", "warp"],
) -> torch.Tensor:
    """Run Torch local/global steps with the selected correspondence backend."""

    if steps == 0:
        return template_points.clone()

    edges = _unique_edges(template_triangles)
    fitted_points = template_points
    for _ in range(steps):
        _, closest_points = point_to_mesh_distance(
            target_points,
            target_triangles,
            fitted_points,
            squared=True,
            implementation=point_implementation,
        )
        if arap_weight == 0.0:
            right_hand_side = fit_weight * closest_points
        else:
            rotations = _local_rotations(fitted_points, template_points, edges)
            arap_rhs = _arap_right_hand_side(template_points, rotations, edges)
            right_hand_side = fit_weight * closest_points + 2.0 * arap_weight * arap_rhs
        fitted_points = _implicit_spd_solve(
            right_hand_side,
            edges,
            fit_weight,
            arap_weight,
            cg_tolerance,
            cg_max_iterations,
        )
    return fitted_points


__all__ = ["fit_template_points_torch"]
