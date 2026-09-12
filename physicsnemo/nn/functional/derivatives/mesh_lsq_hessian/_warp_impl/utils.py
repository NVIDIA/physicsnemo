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

"""Shared fixed-width algebra for mesh LSQ Hessian Warp kernels."""

from __future__ import annotations

import warp as wp

Mat99f = wp.types.matrix(shape=(9, 9), dtype=wp.float32)
Vec9f = wp.types.vector(length=9, dtype=wp.float32)
Vec9i = wp.types.vector(length=9, dtype=wp.int32)


@wp.func
def coefficient_count(n_dims: int) -> int:  # pragma: no cover
    """Return the active Taylor coefficient count inside Warp kernels."""
    result = int(2)
    if n_dims == 2:
        result = 5
    elif n_dims == 3:
        result = 9
    return result


@wp.func
def design_value(
    n_dims: int,
    column: int,
    x: float,
    y: float,
    z: float,
) -> float:  # pragma: no cover
    """Evaluate one normalized quadratic Taylor design column."""
    value = float(0.0)
    if column == 0:
        value = x
    elif column == 1:
        if n_dims == 1:
            value = 0.5 * x * x
        else:
            value = y
    elif column == 2:
        if n_dims == 2:
            value = 0.5 * x * x
        else:
            value = z
    elif column == 3:
        if n_dims == 2:
            value = x * y
        else:
            value = 0.5 * x * x
    elif column == 4:
        if n_dims == 2:
            value = 0.5 * y * y
        else:
            value = x * y
    elif column == 5:
        value = x * z
    elif column == 6:
        value = 0.5 * y * y
    elif column == 7:
        value = y * z
    elif column == 8:
        value = 0.5 * z * z
    return value


@wp.func
def row_sqrt_weight(
    normalized_distance_squared: float,
    weight_power: float,
    distance_epsilon: float,
    max_log_sqrt_weight: float,
) -> float:  # pragma: no cover
    """Evaluate the recentered square-root inverse-distance weight."""
    weight_distance_squared = wp.max(
        normalized_distance_squared,
        distance_epsilon,
    )
    log_sqrt_weight = -0.25 * weight_power * wp.log(weight_distance_squared)
    return wp.exp(log_sqrt_weight - max_log_sqrt_weight)


@wp.func
def back_substitute(
    n_coefficients: int,
    r_factor: Mat99f,
    right_hand_side: Vec9f,
) -> Vec9f:  # pragma: no cover
    """Solve an upper-triangular system stored in a fixed 9x9 matrix."""
    solution = Vec9f()
    for reverse_index in range(9):
        step = n_coefficients - 1 - reverse_index
        if step >= 0:
            value = right_hand_side[step]
            for column in range(9):
                if column > step and column < n_coefficients:
                    value -= r_factor[step, column] * solution[column]
            solution[step] = value / r_factor[step, step]
    return solution


@wp.func
def forward_substitute_transpose(
    n_coefficients: int,
    r_factor: Mat99f,
    right_hand_side: Vec9f,
) -> Vec9f:  # pragma: no cover
    """Solve a lower-triangular system defined by the transpose of R."""
    solution = Vec9f()
    for step in range(9):
        if step < n_coefficients:
            value = right_hand_side[step]
            for row in range(9):
                if row < step:
                    value -= r_factor[row, step] * solution[row]
            solution[step] = value / r_factor[step, step]
    return solution


@wp.func
def output_gradient_to_coefficients(
    n_dims: int,
    entity: int,
    component: int,
    inverse_scale_squared: float,
    grad_output: wp.array4d(dtype=wp.float32),
) -> Vec9f:  # pragma: no cover
    """Map a full symmetric-Hessian cotangent to packed fit coefficients."""
    gradient = Vec9f()
    if n_dims == 1:
        gradient[1] = grad_output[entity, 0, 0, component] * inverse_scale_squared
    elif n_dims == 2:
        gradient[2] = grad_output[entity, 0, 0, component] * inverse_scale_squared
        gradient[3] = (
            grad_output[entity, 0, 1, component] + grad_output[entity, 1, 0, component]
        ) * inverse_scale_squared
        gradient[4] = grad_output[entity, 1, 1, component] * inverse_scale_squared
    else:
        gradient[3] = grad_output[entity, 0, 0, component] * inverse_scale_squared
        gradient[4] = (
            grad_output[entity, 0, 1, component] + grad_output[entity, 1, 0, component]
        ) * inverse_scale_squared
        gradient[5] = (
            grad_output[entity, 0, 2, component] + grad_output[entity, 2, 0, component]
        ) * inverse_scale_squared
        gradient[6] = grad_output[entity, 1, 1, component] * inverse_scale_squared
        gradient[7] = (
            grad_output[entity, 1, 2, component] + grad_output[entity, 2, 1, component]
        ) * inverse_scale_squared
        gradient[8] = grad_output[entity, 2, 2, component] * inverse_scale_squared
    return gradient


@wp.func
def design_gradient(
    n_dims: int,
    x: float,
    y: float,
    z: float,
    coefficient_gradient: Vec9f,
) -> wp.vec3f:  # pragma: no cover
    """Contract packed Taylor-column derivatives with their cotangents."""
    gradient_x = coefficient_gradient[0]
    gradient_y = float(0.0)
    gradient_z = float(0.0)
    if n_dims == 1:
        gradient_x += coefficient_gradient[1] * x
    elif n_dims == 2:
        gradient_y = coefficient_gradient[1]
        gradient_x += coefficient_gradient[2] * x + coefficient_gradient[3] * y
        gradient_y += coefficient_gradient[3] * x + coefficient_gradient[4] * y
    else:
        gradient_y = coefficient_gradient[1]
        gradient_z = coefficient_gradient[2]
        gradient_x += (
            coefficient_gradient[3] * x
            + coefficient_gradient[4] * y
            + coefficient_gradient[5] * z
        )
        gradient_y += (
            coefficient_gradient[4] * x
            + coefficient_gradient[6] * y
            + coefficient_gradient[7] * z
        )
        gradient_z += (
            coefficient_gradient[5] * x
            + coefficient_gradient[7] * y
            + coefficient_gradient[8] * z
        )
    return wp.vec3f(gradient_x, gradient_y, gradient_z)


__all__ = [
    "Mat99f",
    "Vec9f",
    "Vec9i",
    "back_substitute",
    "coefficient_count",
    "design_gradient",
    "design_value",
    "forward_substitute_transpose",
    "output_gradient_to_coefficients",
    "row_sqrt_weight",
]
