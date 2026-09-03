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

"""Rank-revealing factorization kernel for mesh LSQ Hessians."""

from __future__ import annotations

import warp as wp

from ..utils import (
    Mat99f as _Mat99f,
)
from ..utils import (
    Vec9i as _Vec9i,
)
from ..utils import (
    coefficient_count as _coefficient_count,
)
from ..utils import (
    design_value as _design_value,
)
from ..utils import (
    row_sqrt_weight as _row_sqrt_weight,
)


@wp.kernel(enable_backward=False)
def factorize_kernel(
    points: wp.array2d(dtype=wp.float32),
    offsets: wp.array(dtype=wp.int32),
    indices: wp.array(dtype=wp.int32),
    n_dims: int,
    weight_power: float,
    min_neighbors: int,
    distance_epsilon: float,
    requested_rcond: float,
    q_coefficients: wp.array3d(dtype=wp.float32),
    r_factor: wp.array3d(dtype=wp.float32),
    permutation: wp.array2d(dtype=wp.int32),
    fit_info: wp.array2d(dtype=wp.float32),
    full_rank: wp.array(dtype=wp.int32),
):  # pragma: no cover
    """Build a rank-revealing column-pivoted QR factor per entity."""
    entity = wp.tid()
    start = offsets[entity]
    end = offsets[entity + 1]
    count = end - start
    n_coefficients = _coefficient_count(n_dims)
    if count < min_neighbors or count == 0:
        return

    center_x = points[entity, 0]
    center_y = float(0.0)
    center_z = float(0.0)
    if n_dims > 1:
        center_y = points[entity, 1]
    if n_dims > 2:
        center_z = points[entity, 2]

    effective_rows = int(0)
    distance_sum = float(0.0)
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
            effective_rows += 1
            distance_sum += distance_squared

    if effective_rows < n_coefficients:
        return

    scale_squared = distance_sum / float(effective_rows)
    if not wp.isfinite(scale_squared) or scale_squared <= 0.0:
        return
    inverse_scale = 1.0 / wp.sqrt(scale_squared)

    max_log_sqrt_weight = float(-3.402823466e38)
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
            normalized_distance_squared = distance_squared / scale_squared
            weight_distance_squared = wp.max(
                normalized_distance_squared,
                distance_epsilon,
            )
            log_sqrt_weight = -0.25 * weight_power * wp.log(weight_distance_squared)
            max_log_sqrt_weight = wp.max(
                max_log_sqrt_weight,
                log_sqrt_weight,
            )

    if not wp.isfinite(max_log_sqrt_weight):
        return

    # q_local stores each orthonormal weighted column as coefficients in the
    # original design basis. projections[k, j] is q_k^T B_j.
    q_local = _Mat99f()
    projections = _Mat99f()
    r_local = _Mat99f()
    permutation_local = _Vec9i()
    first_pivot = float(0.0)
    factorization_valid = int(1)

    for step in range(9):
        if step < n_coefficients and factorization_valid == 1:
            best_column = int(-1)
            best_norm_squared = float(-1.0)

            # Explicit residual norms avoid the loss of rank information that
            # normal equations would cause in float32.
            for column in range(9):
                if column < n_coefficients:
                    selected = int(0)
                    for prior in range(9):
                        if prior < step and permutation_local[prior] == column:
                            selected = 1

                    if selected == 0:
                        norm_squared = float(0.0)
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
                                sqrt_weight = _row_sqrt_weight(
                                    distance_squared / scale_squared,
                                    weight_power,
                                    distance_epsilon,
                                    max_log_sqrt_weight,
                                )
                                residual = _design_value(
                                    n_dims,
                                    column,
                                    x,
                                    y,
                                    z,
                                )
                                for prior in range(9):
                                    if prior < step:
                                        q_value = float(0.0)
                                        for coefficient in range(9):
                                            if coefficient < n_coefficients:
                                                q_value += q_local[
                                                    prior, coefficient
                                                ] * _design_value(
                                                    n_dims,
                                                    coefficient,
                                                    x,
                                                    y,
                                                    z,
                                                )
                                        residual -= projections[prior, column] * q_value
                                weighted_residual = sqrt_weight * residual
                                norm_squared += weighted_residual * weighted_residual

                        if norm_squared > best_norm_squared:
                            best_norm_squared = norm_squared
                            best_column = column

            pivot = wp.sqrt(wp.max(best_norm_squared, 0.0))
            if step == 0:
                first_pivot = pivot

            relative_rcond = requested_rcond
            if requested_rcond < 0.0:
                relative_rcond = (
                    float(wp.max(effective_rows, n_coefficients))
                    * 1.1920928955078125e-7
                )
            cutoff = relative_rcond * first_pivot
            if (
                best_column < 0
                or not wp.isfinite(pivot)
                or pivot <= cutoff
                or pivot <= 0.0
            ):
                factorization_valid = 0

            if factorization_valid == 1:
                permutation_local[step] = best_column
                r_local[step, step] = pivot
                for prior in range(9):
                    if prior < step:
                        r_local[prior, step] = projections[prior, best_column]

                for coefficient in range(9):
                    if coefficient < n_coefficients:
                        residual_coefficient = float(0.0)
                        if coefficient == best_column:
                            residual_coefficient = 1.0
                        for prior in range(9):
                            if prior < step:
                                residual_coefficient -= (
                                    projections[prior, best_column]
                                    * q_local[prior, coefficient]
                                )
                        q_local[step, coefficient] = residual_coefficient / pivot

                for column in range(9):
                    if column < n_coefficients:
                        projection = float(0.0)
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
                                sqrt_weight = _row_sqrt_weight(
                                    distance_squared / scale_squared,
                                    weight_power,
                                    distance_epsilon,
                                    max_log_sqrt_weight,
                                )
                                q_value = float(0.0)
                                for coefficient in range(9):
                                    if coefficient < n_coefficients:
                                        q_value += q_local[
                                            step, coefficient
                                        ] * _design_value(
                                            n_dims,
                                            coefficient,
                                            x,
                                            y,
                                            z,
                                        )
                                column_value = _design_value(
                                    n_dims,
                                    column,
                                    x,
                                    y,
                                    z,
                                )
                                projection += (
                                    sqrt_weight * sqrt_weight * q_value * column_value
                                )
                        projections[step, column] = projection

    if factorization_valid == 0:
        return

    for row in range(9):
        if row < n_coefficients:
            permutation[entity, row] = permutation_local[row]
            for column in range(9):
                if column < n_coefficients:
                    q_coefficients[entity, row, column] = q_local[row, column]
                    r_factor[entity, row, column] = r_local[row, column]
    fit_info[entity, 0] = scale_squared
    fit_info[entity, 1] = max_log_sqrt_weight
    fit_info[entity, 2] = float(effective_rows)
    full_rank[entity] = 1


__all__ = ["factorize_kernel"]
