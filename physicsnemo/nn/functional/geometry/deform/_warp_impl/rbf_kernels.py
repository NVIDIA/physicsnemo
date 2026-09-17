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

"""Warp kernels for thin-plate-spline radial-basis fields.

The forward and pullback kernels stream over controls or query points instead
of materializing the dense pairwise kernel matrix. Generic kernels are
instantiated for both supported floating-point precisions below.
"""

from typing import Any

import warp as wp


@wp.func
def _thin_plate_spline_from_squared_distance(squared_distance: Any):
    """Evaluate ``r**2 * log(r)`` from ``r**2``, with ``phi(0) = 0``."""

    zero = type(squared_distance)(0.0)
    if squared_distance == zero:
        return zero
    return type(squared_distance)(0.5) * squared_distance * wp.log(squared_distance)


@wp.func
def _thin_plate_spline_coordinate_factor(squared_distance: Any):
    """Return the factor multiplying ``x - c`` in the TPS gradient."""

    zero = type(squared_distance)(0.0)
    if squared_distance == zero:
        return zero
    return wp.log(squared_distance) + type(squared_distance)(1.0)


@wp.kernel
def _rbf_forward(
    points: wp.array3d(dtype=Any),
    controls: wp.array3d(dtype=Any),
    radial_coefficients: wp.array3d(dtype=Any),
    polynomial_coefficients: wp.array3d(dtype=Any),
    n_controls: int,
    n_dims: int,
    n_polynomial_terms: int,
    field: wp.array3d(dtype=Any),
):
    """Evaluate a TPS field and its optional affine polynomial tail."""

    b, i = wp.tid()
    zero = type(points[b, i, 0])(0.0)

    for k in range(n_dims):
        value = zero
        if n_polynomial_terms != 0:
            value = polynomial_coefficients[b, 0, k]
            for d in range(n_dims):
                value = value + points[b, i, d] * polynomial_coefficients[b, d + 1, k]
        field[b, i, k] = value

    for j in range(n_controls):
        squared_distance = zero
        for d in range(n_dims):
            delta = points[b, i, d] - controls[b, j, d]
            squared_distance = squared_distance + delta * delta
        phi = _thin_plate_spline_from_squared_distance(squared_distance)
        for k in range(n_dims):
            field[b, i, k] = field[b, i, k] + phi * radial_coefficients[b, j, k]


@wp.kernel
def _rbf_point_backward(
    grad_field: wp.array3d(dtype=Any),
    points: wp.array3d(dtype=Any),
    controls: wp.array3d(dtype=Any),
    radial_coefficients: wp.array3d(dtype=Any),
    polynomial_coefficients: wp.array3d(dtype=Any),
    n_controls: int,
    n_dims: int,
    n_polynomial_terms: int,
    grad_points: wp.array3d(dtype=Any),
):
    """Evaluate the query-coordinate pullback without atomics."""

    b, i = wp.tid()
    zero = type(points[b, i, 0])(0.0)
    for d in range(n_dims):
        value = zero
        if n_polynomial_terms != 0:
            for k in range(n_dims):
                value = value + (
                    grad_field[b, i, k] * polynomial_coefficients[b, d + 1, k]
                )
        grad_points[b, i, d] = value

    for j in range(n_controls):
        squared_distance = zero
        for d in range(n_dims):
            delta = points[b, i, d] - controls[b, j, d]
            squared_distance = squared_distance + delta * delta
        coordinate_factor = _thin_plate_spline_coordinate_factor(squared_distance)
        coefficient_dot = zero
        for k in range(n_dims):
            coefficient_dot = coefficient_dot + (
                grad_field[b, i, k] * radial_coefficients[b, j, k]
            )
        for d in range(n_dims):
            delta = points[b, i, d] - controls[b, j, d]
            grad_points[b, i, d] = (
                grad_points[b, i, d] + coordinate_factor * delta * coefficient_dot
            )


@wp.kernel
def _rbf_control_and_radial_backward(
    grad_field: wp.array3d(dtype=Any),
    points: wp.array3d(dtype=Any),
    controls: wp.array3d(dtype=Any),
    radial_coefficients: wp.array3d(dtype=Any),
    n_points: int,
    n_dims: int,
    query_block_size: int,
    need_controls: int,
    need_radial_coefficients: int,
    grad_controls: wp.array3d(dtype=Any),
    grad_radial_coefficients: wp.array3d(dtype=Any),
):
    """Accumulate chunked control and radial-coefficient pullbacks."""

    b, j, d, query_block = wp.tid()
    zero = type(controls[b, j, 0])(0.0)
    control_value = zero
    radial_value = zero
    query_start = query_block * query_block_size
    query_stop = wp.min(query_start + query_block_size, n_points)

    for i in range(query_start, query_stop):
        squared_distance = zero
        for coordinate in range(n_dims):
            delta = points[b, i, coordinate] - controls[b, j, coordinate]
            squared_distance = squared_distance + delta * delta

        if need_radial_coefficients != 0:
            phi = _thin_plate_spline_from_squared_distance(squared_distance)
            radial_value = radial_value + phi * grad_field[b, i, d]

        if need_controls != 0:
            coordinate_factor = _thin_plate_spline_coordinate_factor(squared_distance)
            coefficient_dot = zero
            for k in range(n_dims):
                coefficient_dot = coefficient_dot + (
                    grad_field[b, i, k] * radial_coefficients[b, j, k]
                )
            delta = points[b, i, d] - controls[b, j, d]
            control_value = control_value - coordinate_factor * delta * coefficient_dot

    # One atomic per output and query block bounds contention while exposing
    # enough independent work for large meshes.
    if need_controls != 0:
        wp.atomic_add(grad_controls, b, j, d, control_value)
    if need_radial_coefficients != 0:
        wp.atomic_add(grad_radial_coefficients, b, j, d, radial_value)


@wp.kernel
def _rbf_polynomial_backward(
    grad_field: wp.array3d(dtype=Any),
    points: wp.array3d(dtype=Any),
    n_points: int,
    query_block_size: int,
    grad_polynomial_coefficients: wp.array3d(dtype=Any),
):
    """Accumulate a chunked affine-polynomial coefficient pullback."""

    b, p, k, query_block = wp.tid()
    zero = type(points[b, 0, 0])(0.0)
    value = zero
    query_start = query_block * query_block_size
    query_stop = wp.min(query_start + query_block_size, n_points)
    for i in range(query_start, query_stop):
        feature = type(zero)(1.0)
        if p > 0:
            feature = points[b, i, p - 1]
        value = value + feature * grad_field[b, i, k]
    wp.atomic_add(grad_polynomial_coefficients, b, p, k, value)


def _precision_overload(kernel, dtype, array3d_args):
    """Instantiate one concrete-precision overload of a generic kernel."""

    return wp.overload(
        kernel,
        {name: wp.array3d(dtype=dtype) for name in array3d_args},
    )


_FORWARD_ARRAYS = (
    "points",
    "controls",
    "radial_coefficients",
    "polynomial_coefficients",
    "field",
)
_POINT_BACKWARD_ARRAYS = (
    "grad_field",
    "points",
    "controls",
    "radial_coefficients",
    "polynomial_coefficients",
    "grad_points",
)
_CONTROL_BACKWARD_ARRAYS = (
    "grad_field",
    "points",
    "controls",
    "radial_coefficients",
    "grad_controls",
    "grad_radial_coefficients",
)
_POLYNOMIAL_BACKWARD_ARRAYS = (
    "grad_field",
    "points",
    "grad_polynomial_coefficients",
)

rbf_forward_f32 = _precision_overload(_rbf_forward, wp.float32, _FORWARD_ARRAYS)
rbf_forward_f64 = _precision_overload(_rbf_forward, wp.float64, _FORWARD_ARRAYS)
rbf_point_backward_f32 = _precision_overload(
    _rbf_point_backward, wp.float32, _POINT_BACKWARD_ARRAYS
)
rbf_point_backward_f64 = _precision_overload(
    _rbf_point_backward, wp.float64, _POINT_BACKWARD_ARRAYS
)
rbf_control_and_radial_backward_f32 = _precision_overload(
    _rbf_control_and_radial_backward, wp.float32, _CONTROL_BACKWARD_ARRAYS
)
rbf_control_and_radial_backward_f64 = _precision_overload(
    _rbf_control_and_radial_backward, wp.float64, _CONTROL_BACKWARD_ARRAYS
)
rbf_polynomial_backward_f32 = _precision_overload(
    _rbf_polynomial_backward, wp.float32, _POLYNOMIAL_BACKWARD_ARRAYS
)
rbf_polynomial_backward_f64 = _precision_overload(
    _rbf_polynomial_backward, wp.float64, _POLYNOMIAL_BACKWARD_ARRAYS
)


__all__ = [
    "rbf_control_and_radial_backward_f32",
    "rbf_control_and_radial_backward_f64",
    "rbf_forward_f32",
    "rbf_forward_f64",
    "rbf_point_backward_f32",
    "rbf_point_backward_f64",
    "rbf_polynomial_backward_f32",
    "rbf_polynomial_backward_f64",
]
