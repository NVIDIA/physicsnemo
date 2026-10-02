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

"""Coordinate-adjoint kernel for mesh LSQ Hessians."""

from __future__ import annotations

import warp as wp

from ..utils import (
    Mat99f,
    Vec9f,
    back_substitute,
    coefficient_count,
    design_gradient,
    design_value,
    forward_substitute_transpose,
    output_gradient_to_coefficients,
    row_sqrt_weight,
)


@wp.func
def _row_coordinate_cotangent(
    n_dims: int,
    x: float,
    y: float,
    z: float,
    normalized_distance_squared: float,
    weight_power: float,
    distance_epsilon: float,
    max_log_sqrt_weight: float,
    delta_value: float,
    solution: Vec9f,
    coefficient_adjoint: Vec9f,
) -> wp.vec3f:  # pragma: no cover
    """Differentiate one weighted Taylor row with respect to normalized r."""
    n_coefficients = coefficient_count(n_dims)
    sqrt_weight = row_sqrt_weight(
        normalized_distance_squared,
        weight_power,
        distance_epsilon,
        max_log_sqrt_weight,
    )
    weight = sqrt_weight * sqrt_weight
    predicted = float(0.0)
    design_adjoint_dot = float(0.0)
    for coefficient in range(9):
        if coefficient < n_coefficients:
            design = design_value(n_dims, coefficient, x, y, z)
            predicted += design * solution[coefficient]
            design_adjoint_dot += design * coefficient_adjoint[coefficient]
    residual = delta_value - predicted
    design_cotangent = Vec9f()
    for coefficient in range(9):
        if coefficient < n_coefficients:
            design_cotangent[coefficient] = weight * (
                residual * coefficient_adjoint[coefficient]
                - design_adjoint_dot * solution[coefficient]
            )
    coordinate_cotangent = design_gradient(
        n_dims,
        x,
        y,
        z,
        design_cotangent,
    )
    if normalized_distance_squared > distance_epsilon:
        weight_factor = (
            -weight_power
            * weight
            * design_adjoint_dot
            * residual
            / normalized_distance_squared
        )
        coordinate_cotangent += weight_factor * wp.vec3f(x, y, z)
    return coordinate_cotangent


@wp.kernel(enable_backward=False)
def backward_points_kernel(
    points: wp.array2d(dtype=wp.float32),
    values: wp.array2d(dtype=wp.float32),
    offsets: wp.array(dtype=wp.int32),
    indices: wp.array(dtype=wp.int32),
    n_dims: int,
    weight_power: float,
    distance_epsilon: float,
    q_coefficients: wp.array3d(dtype=wp.float32),
    r_factor_global: wp.array3d(dtype=wp.float32),
    permutation: wp.array2d(dtype=wp.int32),
    fit_info: wp.array2d(dtype=wp.float32),
    full_rank: wp.array(dtype=wp.int32),
    grad_output: wp.array4d(dtype=wp.float32),
    grad_points: wp.array2d(dtype=wp.float32),
):  # pragma: no cover
    """Differentiate accepted least-squares fits with respect to coordinates."""
    entity, component = wp.tid()
    if full_rank[entity] == 0:
        return

    n_coefficients = coefficient_count(n_dims)
    scale_squared = fit_info[entity, 0]
    max_log_sqrt_weight = fit_info[entity, 1]
    effective_rows = fit_info[entity, 2]
    inverse_scale = 1.0 / wp.sqrt(scale_squared)
    center_x = points[entity, 0]
    center_y = float(0.0)
    center_z = float(0.0)
    if n_dims > 1:
        center_y = points[entity, 1]
    if n_dims > 2:
        center_z = points[entity, 2]
    center_value = values[entity, component]
    start = offsets[entity]
    end = offsets[entity + 1]

    qtb = Vec9f()
    for edge in range(start, end):
        neighbor = indices[edge]
        dx = points[neighbor, 0] - center_x
        dy = float(0.0)
        dz = float(0.0)
        if n_dims > 1:
            dy = points[neighbor, 1] - center_y
        if n_dims > 2:
            dz = points[neighbor, 2] - center_z
        distance_squared = dx * dx + dy * dy + dz * dz
        if distance_squared > 0.0:
            x = dx * inverse_scale
            y = dy * inverse_scale
            z = dz * inverse_scale
            sqrt_weight = row_sqrt_weight(
                distance_squared / scale_squared,
                weight_power,
                distance_epsilon,
                max_log_sqrt_weight,
            )
            delta_value = values[neighbor, component] - center_value
            for row in range(9):
                if row < n_coefficients:
                    q_value = float(0.0)
                    for coefficient in range(9):
                        if coefficient < n_coefficients:
                            q_value += q_coefficients[
                                entity, row, coefficient
                            ] * design_value(
                                n_dims,
                                coefficient,
                                x,
                                y,
                                z,
                            )
                    qtb[row] += sqrt_weight * sqrt_weight * q_value * delta_value

    r_local = Mat99f()
    for row in range(9):
        if row < n_coefficients:
            for column in range(9):
                if column < n_coefficients:
                    r_local[row, column] = r_factor_global[entity, row, column]
    solution_permuted = back_substitute(n_coefficients, r_local, qtb)
    solution = Vec9f()
    for row in range(9):
        if row < n_coefficients:
            solution[permutation[entity, row]] = solution_permuted[row]

    # For M=A^TWA and g=dL/dc, solve M p=g using R^T u=P^Tg and Rp=u.
    coefficient_cotangent = output_gradient_to_coefficients(
        n_dims,
        entity,
        component,
        1.0 / scale_squared,
        grad_output,
    )
    cotangent_permuted = Vec9f()
    for row in range(9):
        if row < n_coefficients:
            cotangent_permuted[row] = coefficient_cotangent[permutation[entity, row]]
    intermediate = forward_substitute_transpose(
        n_coefficients,
        r_local,
        cotangent_permuted,
    )
    adjoint_permuted = back_substitute(
        n_coefficients,
        r_local,
        intermediate,
    )
    coefficient_adjoint = Vec9f()
    for row in range(9):
        if row < n_coefficients:
            coefficient_adjoint[permutation[entity, row]] = adjoint_permuted[row]

    # Physical Hessians are c_quadratic / scale_squared.
    scale_cotangent = float(0.0)
    for coefficient in range(9):
        if coefficient < n_coefficients:
            scale_cotangent -= (
                coefficient_cotangent[coefficient]
                * solution[coefficient]
                / scale_squared
            )

    # First pass: z=r/sqrt(scale_squared) couples every row through the RMS
    # scale, so its shared cotangent must be complete before any edge gradient
    # can be finalized. Recomputing row cotangents avoids dynamic local storage.
    for edge in range(start, end):
        neighbor = indices[edge]
        dx = points[neighbor, 0] - center_x
        dy = float(0.0)
        dz = float(0.0)
        if n_dims > 1:
            dy = points[neighbor, 1] - center_y
        if n_dims > 2:
            dz = points[neighbor, 2] - center_z
        distance_squared = dx * dx + dy * dy + dz * dz
        if distance_squared > 0.0:
            x = dx * inverse_scale
            y = dy * inverse_scale
            z = dz * inverse_scale
            coordinate_cotangent = _row_coordinate_cotangent(
                n_dims,
                x,
                y,
                z,
                distance_squared / scale_squared,
                weight_power,
                distance_epsilon,
                max_log_sqrt_weight,
                values[neighbor, component] - center_value,
                solution,
                coefficient_adjoint,
            )
            scale_cotangent -= (
                0.5 * wp.dot(coordinate_cotangent, wp.vec3f(x, y, z)) / scale_squared
            )

    # Second pass: apply the now-complete scale cotangent to each edge.
    center_gradient = wp.vec3f(0.0, 0.0, 0.0)
    for edge in range(start, end):
        neighbor = indices[edge]
        dx = points[neighbor, 0] - center_x
        dy = float(0.0)
        dz = float(0.0)
        if n_dims > 1:
            dy = points[neighbor, 1] - center_y
        if n_dims > 2:
            dz = points[neighbor, 2] - center_z
        distance_squared = dx * dx + dy * dy + dz * dz
        if distance_squared > 0.0:
            x = dx * inverse_scale
            y = dy * inverse_scale
            z = dz * inverse_scale
            coordinate_cotangent = _row_coordinate_cotangent(
                n_dims,
                x,
                y,
                z,
                distance_squared / scale_squared,
                weight_power,
                distance_epsilon,
                max_log_sqrt_weight,
                values[neighbor, component] - center_value,
                solution,
                coefficient_adjoint,
            )
            edge_vector = wp.vec3f(dx, dy, dz)
            edge_gradient = (
                inverse_scale * coordinate_cotangent
                + (2.0 * scale_cotangent / effective_rows) * edge_vector
            )
            wp.atomic_add(grad_points, neighbor, 0, edge_gradient[0])
            center_gradient -= edge_gradient
            if n_dims > 1:
                wp.atomic_add(grad_points, neighbor, 1, edge_gradient[1])
            if n_dims > 2:
                wp.atomic_add(grad_points, neighbor, 2, edge_gradient[2])

    wp.atomic_add(grad_points, entity, 0, center_gradient[0])
    if n_dims > 1:
        wp.atomic_add(grad_points, entity, 1, center_gradient[1])
    if n_dims > 2:
        wp.atomic_add(grad_points, entity, 2, center_gradient[2])


__all__ = ["backward_points_kernel"]
