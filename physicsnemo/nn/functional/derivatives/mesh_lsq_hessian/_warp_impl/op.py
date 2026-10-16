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

"""Torch custom-op integration for Warp mesh LSQ Hessians."""

from __future__ import annotations

import math

import torch
import warp as wp

from physicsnemo.core.function_spec import FunctionSpec

from ..utils import (
    resolve_min_neighbors,
    resolve_safe_epsilon,
    validate_inputs,
    validate_rcond,
    validate_weight_power,
)
from .launch_backward import launch_backward_points, launch_backward_values
from .launch_forward import launch_factorization, launch_forward

wp.init()
wp.config.log_level = wp.LOG_WARNING


@torch.library.custom_op("physicsnemo::mesh_lsq_hessian_warp_impl", mutates_args=())
def mesh_lsq_hessian_impl(
    points: torch.Tensor,
    values: torch.Tensor,
    neighbor_offsets: torch.Tensor,
    neighbor_indices: torch.Tensor,
    weight_power: float,
    min_neighbors: int,
    distance_epsilon: float,
    requested_rcond: float,
) -> torch.Tensor:
    """Compute direct quadratic-LSQ Hessians with Warp kernels."""
    validate_inputs(
        points=points,
        values=values,
        neighbor_offsets=neighbor_offsets,
        neighbor_indices=neighbor_indices,
        min_neighbors=int(min_neighbors),
    )
    points_fp32 = points.to(dtype=torch.float32).contiguous()
    values_fp32 = values.to(dtype=torch.float32).contiguous()
    offsets_i32 = neighbor_offsets.to(
        dtype=torch.int32,
        device=points.device,
    ).contiguous()
    indices_i32 = neighbor_indices.to(
        dtype=torch.int32,
        device=points.device,
    ).contiguous()

    n_entities, n_dims = points_fp32.shape
    value_shape = values.shape[1:]
    n_components = math.prod(value_shape) if value_shape else 1
    values_flat_fp32 = values_fp32.reshape(n_entities, n_components)
    hessians_flat = torch.zeros(
        (n_entities, n_dims, n_dims, n_components),
        dtype=torch.float32,
        device=values.device,
    )
    if n_entities > 0 and n_components > 0:
        wp_device, wp_stream = FunctionSpec.warp_launch_context(points_fp32)
        q_coefficients, r_factor, permutation, fit_info, full_rank = (
            launch_factorization(
                points_fp32=points_fp32,
                offsets_i32=offsets_i32,
                indices_i32=indices_i32,
                weight_power=float(weight_power),
                min_neighbors=int(min_neighbors),
                distance_epsilon=float(distance_epsilon),
                requested_rcond=float(requested_rcond),
                wp_device=wp_device,
                wp_stream=wp_stream,
            )
        )
        launch_forward(
            points_fp32=points_fp32,
            values_flat_fp32=values_flat_fp32,
            offsets_i32=offsets_i32,
            indices_i32=indices_i32,
            weight_power=float(weight_power),
            distance_epsilon=float(distance_epsilon),
            q_coefficients=q_coefficients,
            r_factor=r_factor,
            permutation=permutation,
            fit_info=fit_info,
            full_rank=full_rank,
            hessians_flat=hessians_flat,
            wp_device=wp_device,
            wp_stream=wp_stream,
        )

    output = hessians_flat.reshape(n_entities, n_dims, n_dims, *value_shape)
    if output.dtype != values.dtype:
        output = output.to(dtype=values.dtype)
    return output


@mesh_lsq_hessian_impl.register_fake
def _mesh_lsq_hessian_impl_fake(
    points: torch.Tensor,
    values: torch.Tensor,
    neighbor_offsets: torch.Tensor,
    neighbor_indices: torch.Tensor,
    weight_power: float,
    min_neighbors: int,
    distance_epsilon: float,
    requested_rcond: float,
) -> torch.Tensor:
    """Propagate shape and dtype through fake-tensor tracing."""
    _ = (
        neighbor_offsets,
        neighbor_indices,
        weight_power,
        min_neighbors,
        distance_epsilon,
        requested_rcond,
    )
    return torch.empty(
        (values.shape[0], points.shape[1], points.shape[1], *values.shape[1:]),
        dtype=values.dtype,
        device=values.device,
    )


@torch.library.custom_op(
    "physicsnemo::mesh_lsq_hessian_warp_backward_impl",
    mutates_args=(),
)
def mesh_lsq_hessian_backward_impl(
    points: torch.Tensor,
    values: torch.Tensor,
    neighbor_offsets: torch.Tensor,
    neighbor_indices: torch.Tensor,
    grad_output: torch.Tensor,
    weight_power: float,
    min_neighbors: int,
    distance_epsilon: float,
    requested_rcond: float,
    needs_points: bool,
    needs_values: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the explicit Warp adjoints behind an opaque dispatcher boundary."""
    points_fp32 = points.to(dtype=torch.float32).contiguous()
    values_fp32 = values.to(dtype=torch.float32).contiguous()
    offsets_i32 = neighbor_offsets.to(
        dtype=torch.int32,
        device=points.device,
    ).contiguous()
    indices_i32 = neighbor_indices.to(
        dtype=torch.int32,
        device=points.device,
    ).contiguous()

    n_entities, n_dims = points_fp32.shape
    value_shape = values.shape[1:]
    n_components = math.prod(value_shape) if value_shape else 1
    values_flat_fp32 = values_fp32.reshape(n_entities, n_components)
    grad_output_flat = grad_output.to(dtype=torch.float32).reshape(
        n_entities,
        n_dims,
        n_dims,
        n_components,
    )
    grad_output_flat = grad_output_flat.contiguous()
    grad_values_flat = torch.zeros(
        (n_entities, n_components),
        dtype=torch.float32,
        device=values.device,
    )
    grad_points_fp32 = torch.zeros_like(points_fp32, dtype=torch.float32)
    if n_entities > 0 and n_components > 0 and (needs_points or needs_values):
        wp_device, wp_stream = FunctionSpec.warp_launch_context(points_fp32)
        # Recompute geometry factors inside the opaque backward op instead of
        # retaining five O(n_entities) workspaces in the autograd context. This
        # keeps the saved state small and the backward compatible with AOTAutograd.
        q_coefficients, r_factor, permutation, fit_info, full_rank = (
            launch_factorization(
                points_fp32=points_fp32,
                offsets_i32=offsets_i32,
                indices_i32=indices_i32,
                weight_power=float(weight_power),
                min_neighbors=int(min_neighbors),
                distance_epsilon=float(distance_epsilon),
                requested_rcond=float(requested_rcond),
                wp_device=wp_device,
                wp_stream=wp_stream,
            )
        )
        if needs_values:
            launch_backward_values(
                points_fp32=points_fp32,
                offsets_i32=offsets_i32,
                indices_i32=indices_i32,
                weight_power=float(weight_power),
                distance_epsilon=float(distance_epsilon),
                q_coefficients=q_coefficients,
                r_factor=r_factor,
                permutation=permutation,
                fit_info=fit_info,
                full_rank=full_rank,
                grad_output_flat=grad_output_flat,
                grad_values_flat=grad_values_flat,
                wp_device=wp_device,
                wp_stream=wp_stream,
            )
        if needs_points:
            launch_backward_points(
                points_fp32=points_fp32,
                values_flat_fp32=values_flat_fp32,
                offsets_i32=offsets_i32,
                indices_i32=indices_i32,
                weight_power=float(weight_power),
                distance_epsilon=float(distance_epsilon),
                q_coefficients=q_coefficients,
                r_factor=r_factor,
                permutation=permutation,
                fit_info=fit_info,
                full_rank=full_rank,
                grad_output_flat=grad_output_flat,
                grad_points_fp32=grad_points_fp32,
                wp_device=wp_device,
                wp_stream=wp_stream,
            )

    return grad_points_fp32, grad_values_flat.reshape(values.shape)


@mesh_lsq_hessian_backward_impl.register_fake
def _mesh_lsq_hessian_backward_impl_fake(
    points: torch.Tensor,
    values: torch.Tensor,
    neighbor_offsets: torch.Tensor,
    neighbor_indices: torch.Tensor,
    grad_output: torch.Tensor,
    weight_power: float,
    min_neighbors: int,
    distance_epsilon: float,
    requested_rcond: float,
    needs_points: bool,
    needs_values: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Propagate explicit-adjoint metadata through fake-tensor tracing."""
    _ = (
        neighbor_offsets,
        neighbor_indices,
        grad_output,
        weight_power,
        min_neighbors,
        distance_epsilon,
        requested_rcond,
        needs_points,
        needs_values,
    )
    return (
        torch.empty_like(points, dtype=torch.float32),
        torch.empty_like(values, dtype=torch.float32),
    )


def _setup_mesh_lsq_hessian_context(
    ctx: torch.autograd.function.FunctionCtx,
    inputs: tuple,
    output: torch.Tensor,
) -> None:
    """Save original inputs required by the explicit Warp backward."""
    (
        points,
        values,
        neighbor_offsets,
        neighbor_indices,
        weight_power,
        min_neighbors,
        distance_epsilon,
        requested_rcond,
    ) = inputs
    _ = output
    ctx.save_for_backward(
        points,
        values,
        neighbor_offsets,
        neighbor_indices,
    )
    ctx.points_dtype = points.dtype
    ctx.values_dtype = values.dtype
    ctx.weight_power = float(weight_power)
    ctx.min_neighbors = int(min_neighbors)
    ctx.distance_epsilon = float(distance_epsilon)
    ctx.requested_rcond = float(requested_rcond)


def _backward_mesh_lsq_hessian(
    ctx: torch.autograd.function.FunctionCtx,
    grad_output: torch.Tensor,
) -> tuple[
    torch.Tensor | None,
    torch.Tensor | None,
    None,
    None,
    None,
    None,
    None,
    None,
]:
    """Differentiate the Warp reconstruction with respect to points and values."""
    needs_points = ctx.needs_input_grad[0]
    needs_values = ctx.needs_input_grad[1]
    if grad_output is None or (not needs_points and not needs_values):
        return None, None, None, None, None, None, None, None

    points, values, neighbor_offsets, neighbor_indices = ctx.saved_tensors
    grad_points_fp32, grad_values_fp32 = mesh_lsq_hessian_backward_impl(
        points,
        values,
        neighbor_offsets,
        neighbor_indices,
        grad_output,
        ctx.weight_power,
        ctx.min_neighbors,
        ctx.distance_epsilon,
        ctx.requested_rcond,
        bool(needs_points),
        bool(needs_values),
    )

    grad_values = None
    if needs_values:
        grad_values = grad_values_fp32
        if grad_values.dtype != ctx.values_dtype:
            grad_values = grad_values.to(dtype=ctx.values_dtype)
    grad_points = None
    if needs_points:
        grad_points = grad_points_fp32
        if grad_points.dtype != ctx.points_dtype:
            grad_points = grad_points.to(dtype=ctx.points_dtype)
    return grad_points, grad_values, None, None, None, None, None, None


mesh_lsq_hessian_impl.register_autograd(
    _backward_mesh_lsq_hessian,
    setup_context=_setup_mesh_lsq_hessian_context,
)


def mesh_lsq_hessian_warp(
    points: torch.Tensor,
    values: torch.Tensor,
    neighbor_offsets: torch.Tensor,
    neighbor_indices: torch.Tensor,
    weight_power: float = 2.0,
    min_neighbors: int | None = None,
    safe_epsilon: float | None = None,
    rcond: float | None = None,
) -> torch.Tensor:
    """Compute quadratic-LSQ mesh Hessians with Warp.

    Warp computes in ``float32`` and casts the result back to ``values.dtype``.
    The explicit adjoint differentiates both values and point coordinates for
    accepted full-rank fits; the discrete rank decision itself is non-smooth.
    """
    if points.ndim != 2:
        raise ValueError(
            f"points must have shape (n_entities, dims), got {points.shape=}"
        )
    resolved_min_neighbors = resolve_min_neighbors(
        min_neighbors,
        n_dims=points.shape[1],
    )
    resolved_weight_power = validate_weight_power(weight_power)
    resolved_rcond = validate_rcond(rcond)
    distance_epsilon = resolve_safe_epsilon(
        safe_epsilon=safe_epsilon,
        dtype=torch.float32,
    )
    requested_rcond = -1.0 if resolved_rcond is None else resolved_rcond
    return mesh_lsq_hessian_impl(
        points,
        values,
        neighbor_offsets,
        neighbor_indices,
        float(resolved_weight_power),
        int(resolved_min_neighbors),
        float(distance_epsilon),
        float(requested_rcond),
    )


__all__ = ["mesh_lsq_hessian_warp"]
