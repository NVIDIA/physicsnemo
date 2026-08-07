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

"""Torch custom-op integration for Warp-backed Sobolev deformation."""

from __future__ import annotations

import torch
import warp as wp
from warp.optim.linear import LinearOperator, cg

from physicsnemo.core.function_spec import FunctionSpec

from .sobolev_kernels import (
    ASSEMBLY_KERNELS,
    GEOMETRY_PULLBACK_KERNELS,
    accumulate_mass_geometry_coefficient,
    displacement_pullback,
    finalize_system,
    helmholtz_base_matvec,
    helmholtz_stiffness_matvec,
    initialize_adjoint,
    jacobi_matvec,
)

wp.init()
wp.config.log_level = wp.LOG_WARNING


_WARP_DTYPES = {
    torch.float32: wp.float32,
    torch.float64: wp.float64,
}
_TRUE_RESIDUAL_RESTART_INTERVAL = 32


def _warp_dtype(dtype: torch.dtype):
    try:
        return _WARP_DTYPES[dtype]
    except KeyError:
        raise TypeError(
            f"Warp Sobolev deformation supports float32 and float64, got {dtype}"
        ) from None


def _wp_array(tensor: torch.Tensor, dtype):
    """Create a full zero-copy Warp array wrapper around a Torch tensor."""

    return wp.from_torch(
        tensor.detach(),
        dtype=dtype,
        return_ctype=False,
        requires_grad=False,
    )


def _validate_normalized_inputs(
    points: torch.Tensor,
    cells: torch.Tensor,
    displacement: torch.Tensor,
    free_points: torch.Tensor,
) -> None:
    """Validate the normalized tensors accepted by the private custom op."""

    if points.ndim != 3 or displacement.shape != points.shape:
        raise ValueError("points and displacement must be aligned rank-3 tensors")
    if points.device.type != "cuda":
        raise ValueError("the Warp Sobolev backend requires CUDA tensors")
    if cells.ndim != 2 or cells.shape[1] not in ASSEMBLY_KERNELS:
        raise ValueError(
            "Warp Sobolev deformation supports segment, triangle, and tetrahedron cells"
        )
    if free_points.shape != points.shape[:2] or free_points.dtype != torch.bool:
        raise ValueError("free_points must be bool with shape (batch, num_points)")
    if cells.dtype != torch.int64:
        raise TypeError("normalized Warp Sobolev cells must have dtype torch.int64")
    _warp_dtype(points.dtype)
    if displacement.dtype != points.dtype:
        raise TypeError("points and displacement must have the same dtype")
    if not (points.device == cells.device == displacement.device == free_points.device):
        raise ValueError("all Warp Sobolev tensors must be on the same device")
    if cells.numel() and not bool(((cells >= 0) & (cells < points.shape[1])).all()):
        raise ValueError("cells contain indices outside the point range")


def _connected_point_count(cells: torch.Tensor, num_points: int) -> int:
    """Return the topology-only count of vertices referenced by a cell."""

    if cells.numel() == 0:
        return 0
    connected = torch.zeros(num_points, dtype=torch.bool, device=cells.device)
    connected[cells.reshape(-1)] = True
    return int(connected.sum().item())


def _point_incidence(
    cells: torch.Tensor,
    num_points: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build deterministic point-to-cell-local incidence for matvec gathers."""

    point_indices = cells.reshape(-1)
    incidence = torch.argsort(point_indices, stable=True)
    counts = torch.bincount(point_indices, minlength=num_points)
    offsets = torch.empty(
        num_points + 1,
        dtype=torch.int64,
        device=cells.device,
    )
    offsets[0] = 0
    torch.cumsum(counts, dim=0, out=offsets[1:])
    return offsets, incidence


def _launch_assembly(
    points: torch.Tensor,
    cells: torch.Tensor,
    displacement: torch.Tensor,
    free_points: torch.Tensor,
    length_scale: float,
    *,
    use_displacement_initial: bool,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int,
]:
    """Assemble cell-local P1 data and initialize one Helmholtz solve."""

    batch, num_points, num_dims = points.shape
    num_cells, num_cell_points = cells.shape
    dtype = points.dtype
    device = points.device
    local_stiffness = torch.zeros(
        (batch, num_cells, num_cell_points, num_cell_points),
        dtype=dtype,
        device=device,
    )
    stiffness_diagonal = torch.zeros((batch, num_points), dtype=dtype, device=device)
    mass_sum = torch.zeros((batch,), dtype=dtype, device=device)
    invalid_geometry = torch.zeros((batch,), dtype=torch.int32, device=device)
    mass = torch.empty((batch, num_points), dtype=dtype, device=device)
    right_hand_side = torch.empty_like(points)
    initial = torch.empty_like(points)
    connected_count = _connected_point_count(cells, num_points)
    warp_dtype = _warp_dtype(dtype)
    wp_device, wp_stream = FunctionSpec.warp_launch_context(points)

    with FunctionSpec.warp_stream_scope(wp_stream, sync_enter=False):
        if batch * num_cells:
            wp.launch(
                ASSEMBLY_KERNELS[num_cell_points],
                dim=(batch, num_cells),
                inputs=[
                    _wp_array(points, warp_dtype),
                    _wp_array(cells, wp.int64),
                    _wp_array(local_stiffness, warp_dtype),
                    _wp_array(stiffness_diagonal, warp_dtype),
                    _wp_array(mass_sum, warp_dtype),
                    _wp_array(invalid_geometry, wp.int32),
                    int(num_dims),
                ],
                device=wp_device,
                stream=wp_stream,
            )

        # Geometry diagnostics intentionally synchronize.  The public operation
        # already documents CUDA Graph capture as unsupported.
        if bool(invalid_geometry.any()):
            raise ValueError(
                "cells must be finite, nondegenerate simplices for Sobolev deformation"
            )

        if batch * num_points * num_dims:
            wp.launch(
                finalize_system,
                dim=(batch, num_points, num_dims),
                inputs=[
                    _wp_array(displacement, warp_dtype),
                    _wp_array(free_points, wp.bool),
                    _wp_array(mass_sum, warp_dtype),
                    _wp_array(stiffness_diagonal, warp_dtype),
                    int(connected_count),
                    warp_dtype(length_scale * length_scale),
                    int(use_displacement_initial),
                    _wp_array(mass, warp_dtype),
                    _wp_array(right_hand_side, warp_dtype),
                    _wp_array(initial, warp_dtype),
                ],
                device=wp_device,
                stream=wp_stream,
            )

    return (
        right_hand_side,
        initial,
        mass,
        local_stiffness,
        stiffness_diagonal,
        connected_count,
    )


def _make_linear_operators(
    right_hand_side: torch.Tensor,
    cells: torch.Tensor,
    free_points: torch.Tensor,
    mass: torch.Tensor,
    local_stiffness: torch.Tensor,
    stiffness_diagonal: torch.Tensor,
    length_scale: float,
) -> tuple[LinearOperator, LinearOperator, object, object]:
    """Construct matrix-free Helmholtz and Jacobi Warp operators."""

    batch, num_points, num_dims = right_hand_side.shape
    num_cells, num_cell_points = cells.shape
    total_dofs = batch * num_points * num_dims
    warp_dtype = _warp_dtype(right_hand_side.dtype)
    rhs_flat = right_hand_side.reshape(-1)
    rhs_wp = _wp_array(rhs_flat, warp_dtype)
    cells_wp = _wp_array(cells, wp.int64)
    free_wp = _wp_array(free_points, wp.bool)
    mass_wp = _wp_array(mass, warp_dtype)
    stiffness_wp = _wp_array(local_stiffness, warp_dtype)
    diagonal_wp = _wp_array(stiffness_diagonal, warp_dtype)
    batch_offsets_t = torch.arange(
        batch + 1, dtype=torch.int32, device=right_hand_side.device
    ) * (num_points * num_dims)
    batch_offsets_wp = _wp_array(batch_offsets_t, wp.int32)
    point_offsets_t, point_incidence_t = _point_incidence(cells, num_points)
    point_offsets_wp = _wp_array(point_offsets_t, wp.int64)
    point_incidence_wp = _wp_array(point_incidence_t, wp.int64)
    length_scale_squared = length_scale * length_scale

    def helmholtz_matvec(x, y, z, alpha, beta):
        alpha_t = warp_dtype(alpha)
        beta_t = warp_dtype(beta)
        wp.launch(
            helmholtz_base_matvec,
            dim=total_dofs,
            inputs=[
                x,
                y,
                mass_wp,
                free_wp,
                int(num_points),
                int(num_dims),
                alpha_t,
                beta_t,
                z,
            ],
            device=rhs_wp.device,
        )
        if num_cells:
            wp.launch(
                helmholtz_stiffness_matvec,
                dim=(batch, num_points, num_dims),
                inputs=[
                    x,
                    cells_wp,
                    stiffness_wp,
                    free_wp,
                    point_offsets_wp,
                    point_incidence_wp,
                    int(num_points),
                    int(num_dims),
                    int(num_cell_points),
                    warp_dtype(float(alpha) * length_scale_squared),
                    z,
                ],
                device=rhs_wp.device,
            )

    def jacobi(x, y, z, alpha, beta):
        alpha_t = warp_dtype(alpha)
        beta_t = warp_dtype(beta)
        wp.launch(
            jacobi_matvec,
            dim=total_dofs,
            inputs=[
                x,
                y,
                diagonal_wp,
                int(num_points),
                int(num_dims),
                alpha_t,
                beta_t,
                z,
            ],
            device=rhs_wp.device,
        )

    shape = (total_dofs, total_dofs)
    operator = LinearOperator(
        shape,
        warp_dtype,
        rhs_wp.device,
        matvec=helmholtz_matvec,
        batch_offsets=batch_offsets_wp,
    )
    preconditioner = LinearOperator(
        shape,
        warp_dtype,
        rhs_wp.device,
        matvec=jacobi,
        batch_offsets=batch_offsets_wp,
    )
    # Keep the Torch incidence tensors and Warp RHS wrapper alive with the
    # operators.
    owners = (batch_offsets_t, point_offsets_t, point_incidence_t)
    return operator, preconditioner, rhs_wp, owners


def _solve_preassembled(
    right_hand_side: torch.Tensor,
    initial: torch.Tensor,
    cells: torch.Tensor,
    free_points: torch.Tensor,
    mass: torch.Tensor,
    local_stiffness: torch.Tensor,
    stiffness_diagonal: torch.Tensor,
    *,
    length_scale: float,
    max_iterations: int,
    tolerance: float,
    convergence_message: str,
) -> torch.Tensor:
    """Run restarted Warp CG and verify every batch with the true residual."""

    batch, num_points, num_dims = right_hand_side.shape
    if batch * num_points * num_dims == 0:
        return initial

    right_scale = right_hand_side.abs().amax(dim=(1, 2), keepdim=True)
    right_scale = torch.where(
        torch.isfinite(right_scale) & (right_scale > 0),
        right_scale,
        torch.ones_like(right_scale),
    )
    solve_right_hand_side = right_hand_side / right_scale
    solve_initial = initial / right_scale
    operator, preconditioner, rhs_wp, offsets_owner = _make_linear_operators(
        solve_right_hand_side,
        cells,
        free_points,
        mass,
        local_stiffness,
        stiffness_diagonal,
        length_scale,
    )
    warp_dtype = _warp_dtype(solve_right_hand_side.dtype)
    solution_wp = _wp_array(solve_initial.reshape(-1), warp_dtype)
    applied = torch.empty_like(solve_right_hand_side)
    applied_wp = _wp_array(applied.reshape(-1), warp_dtype)
    wp_device, wp_stream = FunctionSpec.warp_launch_context(solve_right_hand_side)
    reduce_shape = (batch, num_points * num_dims)
    right_norm_squared = solve_right_hand_side.reshape(reduce_shape).square().sum(dim=1)
    tiny = torch.finfo(solve_right_hand_side.dtype).tiny
    threshold = tolerance * tolerance * right_norm_squared.clamp_min(tiny)

    for iteration_start in range(
        0,
        max_iterations,
        _TRUE_RESIDUAL_RESTART_INTERVAL,
    ):
        chunk_iterations = min(
            _TRUE_RESIDUAL_RESTART_INTERVAL,
            max_iterations - iteration_start,
        )
        with FunctionSpec.warp_stream_scope(wp_stream, sync_enter=False):
            cg(
                operator,
                rhs_wp,
                solution_wp,
                tol=float(tolerance),
                atol=0.0,
                maxiter=int(chunk_iterations),
                M=preconditioner,
                check_every=int(chunk_iterations),
                use_cuda_graph=False,
            )
            operator.matvec(
                solution_wp,
                applied_wp,
                applied_wp,
                warp_dtype(1.0),
                warp_dtype(0.0),
            )

        # Recursive CG residuals can drift from b - A x in finite precision.
        # Recompute the true residual and restart within the same iteration
        # budget when a batch has not reached the public tolerance.
        residual = solve_right_hand_side - applied
        residual_norm_squared = residual.reshape(reduce_shape).square().sum(dim=1)
        converged = (
            torch.isfinite(solve_initial.reshape(reduce_shape)).all(dim=1)
            & torch.isfinite(residual_norm_squared)
            & (residual_norm_squared <= threshold)
        )
        # Preserve owners through the synchronization in bool().
        _ = offsets_owner
        if bool(converged.all()):
            return solve_initial * right_scale

    raise RuntimeError(convergence_message)


@torch.library.custom_op(
    "physicsnemo::sobolev_displacement_warp_impl",
    mutates_args=(),
    tags=(torch.Tag.nondeterministic_bitwise, torch.Tag.cudagraph_unsafe),
)
def sobolev_displacement_warp_impl(
    points: torch.Tensor,
    cells: torch.Tensor,
    displacement: torch.Tensor,
    free_points: torch.Tensor,
    length_scale: float,
    max_iterations: int,
    tolerance: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Solve the normalized Sobolev displacement system with Warp."""

    _validate_normalized_inputs(points, cells, displacement, free_points)
    points_c = points.contiguous()
    cells_c = cells.contiguous()
    displacement_c = displacement.contiguous()
    free_c = free_points.contiguous()
    rhs, initial, mass, local_stiffness, diagonal, _ = _launch_assembly(
        points_c,
        cells_c,
        displacement_c,
        free_c,
        float(length_scale),
        use_displacement_initial=True,
    )
    solution = _solve_preassembled(
        rhs,
        initial,
        cells_c,
        free_c,
        mass,
        local_stiffness,
        diagonal,
        length_scale=float(length_scale),
        max_iterations=int(max_iterations),
        tolerance=float(tolerance),
        convergence_message=(
            "Sobolev deformation PCG did not converge. Increase max_iterations "
            "or relax tolerance"
        ),
    )
    return solution, mass, local_stiffness, diagonal


@sobolev_displacement_warp_impl.register_fake
def _sobolev_displacement_warp_fake(
    points: torch.Tensor,
    cells: torch.Tensor,
    displacement: torch.Tensor,
    free_points: torch.Tensor,
    length_scale: float,
    max_iterations: int,
    tolerance: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    _ = displacement, free_points, length_scale, max_iterations, tolerance
    batch, num_points, _ = points.shape
    num_cells, num_cell_points = cells.shape
    return (
        torch.empty_like(points, memory_format=torch.contiguous_format),
        torch.empty((batch, num_points), dtype=points.dtype, device=points.device),
        torch.empty(
            (batch, num_cells, num_cell_points, num_cell_points),
            dtype=points.dtype,
            device=points.device,
        ),
        torch.empty((batch, num_points), dtype=points.dtype, device=points.device),
    )


@torch.library.custom_op(
    "physicsnemo::sobolev_displacement_warp_backward_impl",
    mutates_args=(),
    schema=(
        "(Tensor output_adjoint, Tensor points, Tensor cells, Tensor displacement, "
        "Tensor free_points, Tensor solution, Tensor mass, Tensor local_stiffness, "
        "Tensor stiffness_diagonal, float length_scale, int max_iterations, "
        "float tolerance, bool need_points=True, bool need_displacement=True) -> "
        "(Tensor?, Tensor?)"
    ),
    tags=(torch.Tag.nondeterministic_bitwise, torch.Tag.cudagraph_unsafe),
)
def sobolev_displacement_warp_backward_impl(
    output_adjoint: torch.Tensor,
    points: torch.Tensor,
    cells: torch.Tensor,
    displacement: torch.Tensor,
    free_points: torch.Tensor,
    solution: torch.Tensor,
    mass: torch.Tensor,
    local_stiffness: torch.Tensor,
    stiffness_diagonal: torch.Tensor,
    length_scale: float,
    max_iterations: int,
    tolerance: float,
    need_points: bool = True,
    need_displacement: bool = True,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Evaluate the explicit first-order implicit Sobolev pullback."""

    output_adjoint_c = output_adjoint.contiguous()
    points_c = points.contiguous()
    cells_c = cells.contiguous()
    displacement_c = displacement.contiguous()
    free_c = free_points.contiguous()
    solution_c = solution.contiguous()
    mass_c = mass.contiguous()
    stiffness_c = local_stiffness.contiguous()
    diagonal_c = stiffness_diagonal.contiguous()
    rhs = torch.empty_like(output_adjoint_c)
    system_adjoint = torch.empty_like(output_adjoint_c)
    batch, num_points, num_dims = points_c.shape
    num_cells, num_cell_points = cells_c.shape
    warp_dtype = _warp_dtype(points.dtype)
    wp_device, wp_stream = FunctionSpec.warp_launch_context(points_c)

    with FunctionSpec.warp_stream_scope(wp_stream, sync_enter=False):
        if batch * num_points * num_dims:
            wp.launch(
                initialize_adjoint,
                dim=(batch, num_points, num_dims),
                inputs=[
                    _wp_array(output_adjoint_c, warp_dtype),
                    _wp_array(free_c, wp.bool),
                    _wp_array(rhs, warp_dtype),
                    _wp_array(system_adjoint, warp_dtype),
                ],
                device=wp_device,
                stream=wp_stream,
            )

    system_adjoint = _solve_preassembled(
        rhs,
        system_adjoint,
        cells_c,
        free_c,
        mass_c,
        stiffness_c,
        diagonal_c,
        length_scale=float(length_scale),
        max_iterations=int(max_iterations),
        tolerance=float(tolerance),
        convergence_message=(
            "Sobolev adjoint PCG did not converge. Increase max_iterations or "
            "relax tolerance"
        ),
    )

    points_adjoint = torch.zeros_like(points_c) if need_points else None
    displacement_adjoint = (
        torch.empty_like(displacement_c) if need_displacement else None
    )
    connected_count = _connected_point_count(cells_c, num_points) if need_points else 0
    mass_coefficient = (
        torch.zeros((batch,), dtype=points.dtype, device=points.device)
        if need_points and connected_count > 0
        else None
    )

    with FunctionSpec.warp_stream_scope(wp_stream, sync_enter=False):
        if need_displacement and batch * num_points * num_dims:
            wp.launch(
                displacement_pullback,
                dim=(batch, num_points, num_dims),
                inputs=[
                    _wp_array(system_adjoint, warp_dtype),
                    _wp_array(mass_c, warp_dtype),
                    _wp_array(free_c, wp.bool),
                    _wp_array(displacement_adjoint, warp_dtype),
                ],
                device=wp_device,
                stream=wp_stream,
            )

        if need_points and connected_count > 0 and batch * num_cells * num_dims:
            wp.launch(
                accumulate_mass_geometry_coefficient,
                dim=(batch, num_points, num_dims),
                inputs=[
                    _wp_array(displacement_c, warp_dtype),
                    _wp_array(solution_c, warp_dtype),
                    _wp_array(system_adjoint, warp_dtype),
                    _wp_array(free_c, wp.bool),
                    _wp_array(mass_coefficient, warp_dtype),
                ],
                device=wp_device,
                stream=wp_stream,
            )
            wp.launch(
                GEOMETRY_PULLBACK_KERNELS[num_cell_points],
                dim=(batch, num_cells),
                inputs=[
                    _wp_array(points_c, warp_dtype),
                    _wp_array(cells_c, wp.int64),
                    _wp_array(solution_c, warp_dtype),
                    _wp_array(system_adjoint, warp_dtype),
                    _wp_array(mass_coefficient, warp_dtype),
                    int(connected_count),
                    warp_dtype(length_scale * length_scale),
                    int(num_dims),
                    _wp_array(points_adjoint, warp_dtype),
                ],
                device=wp_device,
                stream=wp_stream,
            )

    return points_adjoint, displacement_adjoint


@sobolev_displacement_warp_backward_impl.register_fake
def _sobolev_displacement_warp_backward_fake(
    output_adjoint: torch.Tensor,
    points: torch.Tensor,
    cells: torch.Tensor,
    displacement: torch.Tensor,
    free_points: torch.Tensor,
    solution: torch.Tensor,
    mass: torch.Tensor,
    local_stiffness: torch.Tensor,
    stiffness_diagonal: torch.Tensor,
    length_scale: float,
    max_iterations: int,
    tolerance: float,
    need_points: bool = True,
    need_displacement: bool = True,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    _ = (
        output_adjoint,
        cells,
        free_points,
        solution,
        mass,
        local_stiffness,
        stiffness_diagonal,
        length_scale,
        max_iterations,
        tolerance,
    )
    return (
        torch.empty_like(points, memory_format=torch.contiguous_format)
        if need_points
        else None,
        torch.empty_like(displacement, memory_format=torch.contiguous_format)
        if need_displacement
        else None,
    )


def _setup_sobolev_context(
    ctx: torch.autograd.function.FunctionCtx,
    inputs: tuple,
    output: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    (
        points,
        cells,
        displacement,
        free_points,
        length_scale,
        max_iterations,
        tolerance,
    ) = inputs
    solution, mass, local_stiffness, stiffness_diagonal = output
    ctx.save_for_backward(
        points.contiguous(),
        cells.contiguous(),
        displacement.contiguous(),
        free_points.contiguous(),
        solution,
        mass,
        local_stiffness,
        stiffness_diagonal,
    )
    ctx.length_scale = float(length_scale)
    ctx.max_iterations = int(max_iterations)
    ctx.tolerance = float(tolerance)
    ctx.mark_non_differentiable(mass, local_stiffness, stiffness_diagonal)
    ctx.set_materialize_grads(False)


def _backward_sobolev(
    ctx: torch.autograd.function.FunctionCtx,
    output_adjoint: torch.Tensor | None,
    _mass_adjoint: torch.Tensor | None,
    _stiffness_adjoint: torch.Tensor | None,
    _diagonal_adjoint: torch.Tensor | None,
) -> tuple[torch.Tensor | None, ...]:
    needs = ctx.needs_input_grad
    if output_adjoint is None or not (needs[0] or needs[2]):
        return None, None, None, None, None, None, None
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
    points_adjoint, displacement_adjoint = sobolev_displacement_warp_backward_impl(
        output_adjoint,
        points,
        cells,
        displacement,
        free_points,
        solution,
        mass,
        local_stiffness,
        stiffness_diagonal,
        ctx.length_scale,
        ctx.max_iterations,
        ctx.tolerance,
        bool(needs[0]),
        bool(needs[2]),
    )
    return (
        points_adjoint if needs[0] else None,
        None,
        displacement_adjoint if needs[2] else None,
        None,
        None,
        None,
        None,
    )


sobolev_displacement_warp_impl.register_autograd(
    _backward_sobolev,
    setup_context=_setup_sobolev_context,
)


def sobolev_deform_points_warp(
    points: torch.Tensor,
    cells: torch.Tensor,
    displacement: torch.Tensor,
    fixed_points: torch.Tensor | None,
    *,
    length_scale: float,
    max_iterations: int,
    tolerance: float,
) -> torch.Tensor:
    """Apply normalized differentiable Sobolev deformation with Warp."""

    free_points = (
        torch.ones(points.shape[:2], dtype=torch.bool, device=points.device)
        if fixed_points is None
        else ~fixed_points
    )
    free_vector = free_points.unsqueeze(-1)
    weighted_displacement = torch.where(
        free_vector, displacement, torch.zeros_like(displacement)
    )
    if length_scale == 0 or points.numel() == 0 or cells.numel() == 0:
        return points + weighted_displacement
    solution, _, _, _ = sobolev_displacement_warp_impl(
        points,
        cells,
        displacement,
        free_points,
        float(length_scale),
        int(max_iterations),
        float(tolerance),
    )
    return points + solution


__all__ = [
    "sobolev_deform_points_warp",
    "sobolev_displacement_warp_backward_impl",
    "sobolev_displacement_warp_impl",
]
