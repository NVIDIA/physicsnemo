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

"""Warp kernels for the proper-rotation Procrustes projection."""

from typing import Any

import warp as wp


@wp.func
def _jacobi_rotate_4(
    matrix: wp.mat44d,
    eigenvectors: wp.mat44d,
    first: int,
    second: int,
):
    """Apply one symmetric Jacobi rotation to a 4x4 eigensystem."""

    off_diagonal = matrix[first, second]
    first_diagonal = matrix[first, first]
    second_diagonal = matrix[second, second]
    zero = wp.float64(0.0)
    one = wp.float64(1.0)
    two = wp.float64(2.0)
    threshold = wp.float64(1.0e-15) * (
        wp.abs(first_diagonal) + wp.abs(second_diagonal) + wp.float64(1.0e-300)
    )
    if wp.abs(off_diagonal) > threshold:
        ratio = (second_diagonal - first_diagonal) / (two * off_diagonal)
        tangent = zero
        if ratio >= zero:
            tangent = one / (ratio + wp.sqrt(one + ratio * ratio))
        else:
            tangent = -one / (-ratio + wp.sqrt(one + ratio * ratio))
        cosine = one / wp.sqrt(one + tangent * tangent)
        sine = tangent * cosine

        for index in range(4):
            if index != first and index != second:
                value_first = matrix[index, first]
                value_second = matrix[index, second]
                rotated_first = cosine * value_first - sine * value_second
                rotated_second = sine * value_first + cosine * value_second
                matrix[index, first] = rotated_first
                matrix[first, index] = rotated_first
                matrix[index, second] = rotated_second
                matrix[second, index] = rotated_second

        cosine_squared = cosine * cosine
        sine_squared = sine * sine
        twice_sine_cosine = two * sine * cosine
        matrix[first, first] = (
            cosine_squared * first_diagonal
            - twice_sine_cosine * off_diagonal
            + sine_squared * second_diagonal
        )
        matrix[second, second] = (
            sine_squared * first_diagonal
            + twice_sine_cosine * off_diagonal
            + cosine_squared * second_diagonal
        )
        matrix[first, second] = zero
        matrix[second, first] = zero

        for index in range(4):
            vector_first = eigenvectors[index, first]
            vector_second = eigenvectors[index, second]
            eigenvectors[index, first] = cosine * vector_first - sine * vector_second
            eigenvectors[index, second] = sine * vector_first + cosine * vector_second

    return matrix, eigenvectors


@wp.func
def _proper_rotation_quaternion_3d(covariance: wp.mat33d):
    """Return the determinant-one maximizer of ``trace(R * covariance)``."""

    zero = wp.float64(0.0)
    one = wp.float64(1.0)
    two = wp.float64(2.0)
    trace = covariance[0, 0] + covariance[1, 1] + covariance[2, 2]
    davenport = wp.mat44d(
        trace,
        covariance[1, 2] - covariance[2, 1],
        covariance[2, 0] - covariance[0, 2],
        covariance[0, 1] - covariance[1, 0],
        covariance[1, 2] - covariance[2, 1],
        covariance[0, 0] - covariance[1, 1] - covariance[2, 2],
        covariance[0, 1] + covariance[1, 0],
        covariance[0, 2] + covariance[2, 0],
        covariance[2, 0] - covariance[0, 2],
        covariance[0, 1] + covariance[1, 0],
        -covariance[0, 0] + covariance[1, 1] - covariance[2, 2],
        covariance[1, 2] + covariance[2, 1],
        covariance[0, 1] - covariance[1, 0],
        covariance[0, 2] + covariance[2, 0],
        covariance[1, 2] + covariance[2, 1],
        -covariance[0, 0] - covariance[1, 1] + covariance[2, 2],
    )
    eigenvectors = wp.mat44d(
        one,
        zero,
        zero,
        zero,
        zero,
        one,
        zero,
        zero,
        zero,
        zero,
        one,
        zero,
        zero,
        zero,
        zero,
        one,
    )

    # Cyclic Jacobi sweeps converge the four-dimensional symmetric problem to
    # double precision without relying on Warp's approximate 3x3 SVD.
    for _ in range(12):
        davenport, eigenvectors = _jacobi_rotate_4(davenport, eigenvectors, 0, 1)
        davenport, eigenvectors = _jacobi_rotate_4(davenport, eigenvectors, 0, 2)
        davenport, eigenvectors = _jacobi_rotate_4(davenport, eigenvectors, 0, 3)
        davenport, eigenvectors = _jacobi_rotate_4(davenport, eigenvectors, 1, 2)
        davenport, eigenvectors = _jacobi_rotate_4(davenport, eigenvectors, 1, 3)
        davenport, eigenvectors = _jacobi_rotate_4(davenport, eigenvectors, 2, 3)

    largest_index = 0
    largest_value = davenport[0, 0]
    for index in range(1, 4):
        if davenport[index, index] > largest_value:
            largest_index = index
            largest_value = davenport[index, index]

    scalar = eigenvectors[0, largest_index]
    x_value = eigenvectors[1, largest_index]
    y_value = eigenvectors[2, largest_index]
    z_value = eigenvectors[3, largest_index]
    inverse_norm = one / wp.sqrt(
        scalar * scalar + x_value * x_value + y_value * y_value + z_value * z_value
    )
    scalar *= inverse_norm
    x_value *= inverse_norm
    y_value *= inverse_norm
    z_value *= inverse_norm

    return wp.mat33d(
        one - two * (y_value * y_value + z_value * z_value),
        two * (x_value * y_value - z_value * scalar),
        two * (x_value * z_value + y_value * scalar),
        two * (x_value * y_value + z_value * scalar),
        one - two * (x_value * x_value + z_value * z_value),
        two * (y_value * z_value - x_value * scalar),
        two * (x_value * z_value - y_value * scalar),
        two * (y_value * z_value + x_value * scalar),
        one - two * (x_value * x_value + y_value * y_value),
    )


@wp.kernel
def _proper_rotation_forward_1d(
    covariance: wp.array(dtype=Any),
    rotation: wp.array(dtype=Any),
    symmetric_factor: wp.array(dtype=Any),
):
    """Project a one-dimensional covariance onto ``SO(1)``."""

    batch_index = wp.tid()
    rotation[batch_index] = type(covariance[batch_index])(1.0)
    symmetric_factor[batch_index] = covariance[batch_index]


@wp.kernel
def _proper_rotation_backward_1d(
    grad_rotation: wp.array(dtype=Any),
    rotation: wp.array(dtype=Any),
    symmetric_factor: wp.array(dtype=Any),
    grad_covariance: wp.array(dtype=Any),
):
    """Return the zero derivative of the sole proper 1D rotation."""

    batch_index = wp.tid()
    _ = rotation[batch_index]
    _ = symmetric_factor[batch_index]
    grad_covariance[batch_index] = type(grad_rotation[batch_index])(0.0)


@wp.kernel
def proper_rotation_forward_2d_f64(
    covariance: wp.array(dtype=wp.mat22d),
    rotation: wp.array(dtype=wp.mat22d),
    symmetric_factor: wp.array(dtype=wp.mat22d),
):
    """Project two-dimensional covariance matrices onto ``SO(2)``."""

    batch_index = wp.tid()
    covariance_value = covariance[batch_index]
    cosine_numerator = covariance_value[0, 0] + covariance_value[1, 1]
    sine_numerator = covariance_value[0, 1] - covariance_value[1, 0]
    norm = wp.sqrt(
        cosine_numerator * cosine_numerator + sine_numerator * sine_numerator
    )
    zero = wp.float64(0.0)
    one = wp.float64(1.0)
    half = wp.float64(0.5)
    cosine = one
    sine = zero
    if norm > zero:
        cosine = cosine_numerator / norm
        sine = sine_numerator / norm
    rotation_value = wp.mat22d(cosine, -sine, sine, cosine)
    factor = rotation_value * covariance_value
    rotation[batch_index] = rotation_value
    symmetric_factor[batch_index] = half * (factor + wp.transpose(factor))


@wp.kernel
def proper_rotation_forward_3d_f64(
    covariance: wp.array(dtype=wp.mat33d),
    rotation: wp.array(dtype=wp.mat33d),
    symmetric_factor: wp.array(dtype=wp.mat33d),
):
    """Project three-dimensional covariance matrices onto ``SO(3)``."""

    batch_index = wp.tid()
    covariance_value = covariance[batch_index]
    rotation_value = _proper_rotation_quaternion_3d(covariance_value)
    factor = rotation_value * covariance_value
    rotation[batch_index] = rotation_value
    symmetric_factor[batch_index] = wp.float64(0.5) * (factor + wp.transpose(factor))


@wp.kernel
def proper_rotation_backward_2d_f64(
    grad_rotation: wp.array(dtype=wp.mat22d),
    rotation: wp.array(dtype=wp.mat22d),
    symmetric_factor: wp.array(dtype=wp.mat22d),
    grad_covariance: wp.array(dtype=wp.mat22d),
):
    """Apply the analytic proper-rotation VJP in two dimensions."""

    batch_index = wp.tid()
    gradient = grad_rotation[batch_index]
    rotation_value = rotation[batch_index]
    factor = symmetric_factor[batch_index]
    # At the optimum H = R A is symmetric. The polar-factor pullback solves
    # H Z + Z H = K for skew Z; in 2D its sole denominator is trace(H).
    generator = wp.float64(0.5) * (
        gradient * wp.transpose(rotation_value)
        - rotation_value * wp.transpose(gradient)
    )
    value = generator[0, 1] / (factor[0, 0] + factor[1, 1])
    zero = wp.float64(0.0)
    solution = wp.mat22d(zero, value, -value, zero)
    grad_covariance[batch_index] = (
        wp.float64(-2.0) * wp.transpose(rotation_value) * solution
    )


@wp.kernel
def proper_rotation_backward_3d_f64(
    grad_rotation: wp.array(dtype=wp.mat33d),
    rotation: wp.array(dtype=wp.mat33d),
    symmetric_factor: wp.array(dtype=wp.mat33d),
    grad_covariance: wp.array(dtype=wp.mat33d),
):
    """Apply the analytic proper-rotation VJP in three dimensions."""

    batch_index = wp.tid()
    gradient = grad_rotation[batch_index]
    rotation_value = rotation[batch_index]
    factor = symmetric_factor[batch_index]
    # Writing skew matrices as cross-product vectors turns the Sylvester
    # equation H Z + Z H = K into (trace(H) I - H) z = k.
    generator = wp.float64(0.5) * (
        gradient * wp.transpose(rotation_value)
        - rotation_value * wp.transpose(gradient)
    )
    trace = factor[0, 0] + factor[1, 1] + factor[2, 2]
    system = wp.mat33d(
        trace - factor[0, 0],
        -factor[0, 1],
        -factor[0, 2],
        -factor[1, 0],
        trace - factor[1, 1],
        -factor[1, 2],
        -factor[2, 0],
        -factor[2, 1],
        trace - factor[2, 2],
    )
    generator_vector = wp.vec3d(generator[2, 1], generator[0, 2], generator[1, 0])
    solution_vector = wp.inverse(system) * generator_vector
    solution = wp.mat33d(
        wp.float64(0.0),
        -solution_vector[2],
        solution_vector[1],
        solution_vector[2],
        wp.float64(0.0),
        -solution_vector[0],
        -solution_vector[1],
        solution_vector[0],
        wp.float64(0.0),
    )
    grad_covariance[batch_index] = (
        wp.float64(-2.0) * wp.transpose(rotation_value) * solution
    )


def _overload_1d(kernel, dtype):
    """Instantiate a scalar-array Procrustes kernel overload."""

    if kernel is _proper_rotation_backward_1d:
        return wp.overload(
            kernel,
            {
                "grad_rotation": wp.array(dtype=dtype),
                "rotation": wp.array(dtype=dtype),
                "symmetric_factor": wp.array(dtype=dtype),
                "grad_covariance": wp.array(dtype=dtype),
            },
        )
    return wp.overload(
        kernel,
        {
            "covariance": wp.array(dtype=dtype),
            "rotation": wp.array(dtype=dtype),
            "symmetric_factor": wp.array(dtype=dtype),
        },
    )


proper_rotation_forward_1d_f32 = _overload_1d(_proper_rotation_forward_1d, wp.float32)
proper_rotation_forward_1d_f64 = _overload_1d(_proper_rotation_forward_1d, wp.float64)
proper_rotation_backward_1d_f32 = _overload_1d(_proper_rotation_backward_1d, wp.float32)
proper_rotation_backward_1d_f64 = _overload_1d(_proper_rotation_backward_1d, wp.float64)


__all__ = [
    "proper_rotation_backward_1d_f32",
    "proper_rotation_backward_1d_f64",
    "proper_rotation_backward_2d_f64",
    "proper_rotation_backward_3d_f64",
    "proper_rotation_forward_1d_f32",
    "proper_rotation_forward_1d_f64",
    "proper_rotation_forward_2d_f64",
    "proper_rotation_forward_3d_f64",
]
