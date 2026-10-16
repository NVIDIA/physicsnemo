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

"""Forward solve kernel for mesh LSQ Hessians."""

from __future__ import annotations

import warp as wp

from ..utils import (
    Mat99f,
    Vec9f,
    back_substitute,
    coefficient_count,
    design_value,
    row_sqrt_weight,
)


@wp.kernel(enable_backward=False)
def forward_kernel(
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
    hessians: wp.array4d(dtype=wp.float32),
):  # pragma: no cover
    """Apply stored QR factors to one entity and value component."""
    entity, component = wp.tid()
    if full_rank[entity] == 0:
        return

    n_coefficients = coefficient_count(n_dims)
    scale_squared = fit_info[entity, 0]
    max_log_sqrt_weight = fit_info[entity, 1]
    inverse_scale = 1.0 / wp.sqrt(scale_squared)
    center_x = points[entity, 0]
    center_y = float(0.0)
    center_z = float(0.0)
    if n_dims > 1:
        center_y = points[entity, 1]
    if n_dims > 2:
        center_z = points[entity, 2]
    center_value = values[entity, component]

    qtb = Vec9f()
    start = offsets[entity]
    end = offsets[entity + 1]
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
    permuted_solution = back_substitute(n_coefficients, r_local, qtb)
    solution = Vec9f()
    for row in range(9):
        if row < n_coefficients:
            solution[permutation[entity, row]] = permuted_solution[row]

    inverse_scale_squared = 1.0 / scale_squared
    if n_dims == 1:
        hessians[entity, 0, 0, component] = solution[1] * inverse_scale_squared
    elif n_dims == 2:
        hessians[entity, 0, 0, component] = solution[2] * inverse_scale_squared
        hessians[entity, 0, 1, component] = solution[3] * inverse_scale_squared
        hessians[entity, 1, 0, component] = solution[3] * inverse_scale_squared
        hessians[entity, 1, 1, component] = solution[4] * inverse_scale_squared
    else:
        hessians[entity, 0, 0, component] = solution[3] * inverse_scale_squared
        hessians[entity, 0, 1, component] = solution[4] * inverse_scale_squared
        hessians[entity, 1, 0, component] = solution[4] * inverse_scale_squared
        hessians[entity, 0, 2, component] = solution[5] * inverse_scale_squared
        hessians[entity, 2, 0, component] = solution[5] * inverse_scale_squared
        hessians[entity, 1, 1, component] = solution[6] * inverse_scale_squared
        hessians[entity, 1, 2, component] = solution[7] * inverse_scale_squared
        hessians[entity, 2, 1, component] = solution[7] * inverse_scale_squared
        hessians[entity, 2, 2, component] = solution[8] * inverse_scale_squared


__all__ = ["forward_kernel"]
