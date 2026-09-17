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

"""Warp kernels for compact Shepard morphing.

Every kernel is written once as a generic Warp kernel (``typing.Any``
annotations, ``type(...)`` scalar constructors) and instantiated for float32
and float64 through :func:`warp.overload`, so the two precisions share one
numerical definition by construction.
"""

from typing import Any

import warp as wp


@wp.func
def _normalized_component(point: Any, control: Any, radius: Any):
    one = type(point)(1.0)
    delta = point - control
    value = delta / radius
    if not wp.isfinite(delta):
        value = point / radius - control / radius
    if wp.isnan(value):
        value = one
    return wp.clamp(value, -one, one)


@wp.func
def _normalized_distance(
    points: wp.array3d(dtype=Any),
    controls: wp.array3d(dtype=Any),
    radii: wp.array2d(dtype=Any),
    b: int,
    i: int,
    j: int,
    n_dims: int,
):
    radius = radii[b, j]
    zero = type(radius)(0.0)
    maximum = zero
    for d in range(n_dims):
        value = _normalized_component(points[b, i, d], controls[b, j, d], radius)
        maximum = wp.max(maximum, wp.abs(value))
    if maximum == zero:
        return zero
    norm_squared = zero
    for d in range(n_dims):
        value = _normalized_component(points[b, i, d], controls[b, j, d], radius)
        scaled = value / maximum
        norm_squared = norm_squared + scaled * scaled
    return maximum * wp.sqrt(norm_squared)


@wp.func
def _wendland_phi(q: Any):
    """Evaluate the compact Wendland weight."""

    one = type(q)(1.0)
    t = one - q
    return t * t * t * t * (type(q)(4.0) * q + one)


@wp.func
def _wendland_phi_prime(q: Any):
    """Evaluate the compact Wendland weight derivative."""

    t = type(q)(1.0) - q
    return -type(q)(20.0) * q * t * t * t


@wp.kernel
def _shepard_forward(
    points: wp.array3d(dtype=Any),
    controls: wp.array3d(dtype=Any),
    control_displacements: wp.array3d(dtype=Any),
    radii: wp.array2d(dtype=Any),
    n_controls: int,
    n_dims: int,
    save_auxiliaries: int,
    save_correction: int,
    field: wp.array3d(dtype=Any),
    min_q: wp.array2d(dtype=Any),
    denominator: wp.array2d(dtype=Any),
    exact_count_out: wp.array2d(dtype=wp.int32),
    reference_index_out: wp.array2d(dtype=wp.int32),
    correction: wp.array3d(dtype=Any),
):
    """Interpolate a compact Shepard displacement field."""

    b, i = wp.tid()
    zero = type(points[b, i, 0])(0.0)
    one = type(zero)(1.0)
    exact_count = int(0)
    # Active normalized distances satisfy q < 1, so any value above one is an
    # unambiguous "no active control seen" placeholder; reference_index == -1
    # is the authoritative no-active flag.
    minimum = type(zero)(2.0)
    reference_index = int(-1)

    for j in range(n_controls):
        q = _normalized_distance(points, controls, radii, b, i, j, n_dims)
        if q == zero:
            exact_count = exact_count + 1
        elif q < one and q < minimum:
            minimum = q
            reference_index = j

    if save_auxiliaries != 0:
        exact_count_out[b, i] = exact_count
        reference_index_out[b, i] = reference_index
    if exact_count > 0:
        inv_count = one / type(zero)(exact_count)
        for d in range(n_dims):
            field[b, i, d] = zero
            if save_correction != 0:
                correction[b, i, d] = zero
        for j in range(n_controls):
            q = _normalized_distance(points, controls, radii, b, i, j, n_dims)
            if q == zero:
                for d in range(n_dims):
                    field[b, i, d] = field[b, i, d] + control_displacements[b, j, d]
        for d in range(n_dims):
            field[b, i, d] = field[b, i, d] * inv_count
        if save_auxiliaries != 0:
            min_q[b, i] = one
            denominator[b, i] = type(zero)(exact_count)
            reference_index_out[b, i] = int(-1)
        return

    if reference_index == int(-1):
        if save_auxiliaries != 0:
            min_q[b, i] = one
            denominator[b, i] = one
        for d in range(n_dims):
            field[b, i, d] = zero
            if save_correction != 0:
                correction[b, i, d] = zero
        return

    # Multiplying every handle weight and the stationary background by the
    # same minimum q^2 keeps the quotient unchanged. Evaluating the handle
    # ratio as (minimum_q / q)^2 avoids overflow and q^2 underflow.
    if save_auxiliaries != 0:
        min_q[b, i] = minimum
    reference_phi = _wendland_phi(minimum)
    background = minimum * minimum / reference_phi
    denom = background
    for d in range(n_dims):
        value = -background * control_displacements[b, reference_index, d]
        if save_correction != 0:
            correction[b, i, d] = value
        else:
            # When geometry gradients are unnecessary, field doubles as
            # per-query scratch so no correction tensor needs to be allocated.
            field[b, i, d] = value

    for j in range(n_controls):
        q = _normalized_distance(points, controls, radii, b, i, j, n_dims)
        if q > zero and q < one:
            phi = _wendland_phi(q)
            ratio = minimum / q
            a = ratio * ratio * phi / reference_phi
            denom = denom + a
            if j != reference_index:
                for d in range(n_dims):
                    value = a * (
                        control_displacements[b, j, d]
                        - control_displacements[b, reference_index, d]
                    )
                    if save_correction != 0:
                        correction[b, i, d] = correction[b, i, d] + value
                    else:
                        field[b, i, d] = field[b, i, d] + value

    if save_auxiliaries != 0:
        denominator[b, i] = denom
    for d in range(n_dims):
        if save_correction != 0:
            correction[b, i, d] = correction[b, i, d] / denom
            field[b, i, d] = (
                control_displacements[b, reference_index, d] + correction[b, i, d]
            )
        else:
            field[b, i, d] = (
                control_displacements[b, reference_index, d] + field[b, i, d] / denom
            )


@wp.kernel
def _shepard_backward(
    points: wp.array3d(dtype=Any),
    controls: wp.array3d(dtype=Any),
    control_displacements: wp.array3d(dtype=Any),
    radii: wp.array2d(dtype=Any),
    min_q: wp.array2d(dtype=Any),
    denominator: wp.array2d(dtype=Any),
    exact_count: wp.array2d(dtype=wp.int32),
    reference_index: wp.array2d(dtype=wp.int32),
    correction: wp.array3d(dtype=Any),
    grad_field: wp.array3d(dtype=Any),
    n_dims: int,
    need_controls: int,
    need_control_displacements: int,
    need_radii: int,
    grad_controls: wp.array3d(dtype=Any),
    grad_control_displacements: wp.array3d(dtype=Any),
    grad_radii: wp.array2d(dtype=Any),
):
    """Accumulate the control-centric Shepard pullback."""

    b, i, j = wp.tid()
    zero = type(points[b, i, 0])(0.0)
    one = type(zero)(1.0)
    two = type(zero)(2.0)
    q = _normalized_distance(points, controls, radii, b, i, j, n_dims)
    coincident = q == zero

    count = exact_count[b, i]
    if count > 0:
        if need_control_displacements != 0 and coincident:
            inv_count = one / type(zero)(count)
            for d in range(n_dims):
                wp.atomic_add(
                    grad_control_displacements,
                    b,
                    j,
                    d,
                    grad_field[b, i, d] * inv_count,
                )
        return

    if coincident or q >= one:
        return
    radius = radii[b, j]
    phi = _wendland_phi(q)
    minimum = min_q[b, i]
    denom = denominator[b, i]
    reference_phi = _wendland_phi(minimum)
    scaled_denom = reference_phi * denom
    ratio = minimum / q
    ratio_squared = ratio * ratio
    a = ratio_squared * phi

    ref = reference_index[b, i]
    if need_control_displacements != 0:
        for d in range(n_dims):
            wp.atomic_add(
                grad_control_displacements,
                b,
                j,
                d,
                (a / scaled_denom) * grad_field[b, i, d],
            )

    if need_controls == 0 and need_radii == 0:
        return

    phi_prime = _wendland_phi_prime(q)
    base_dot = zero
    correction_dot = zero
    reference_dot = zero
    for d in range(n_dims):
        g = grad_field[b, i, d]
        base_dot = base_dot + g * (
            control_displacements[b, j, d] - control_displacements[b, ref, d]
        )
        correction_dot = correction_dot + g * correction[b, i, d]
        reference_dot = reference_dot + g * control_displacements[b, ref, d]

    dot = base_dot - correction_dot
    d_a_d_q = ratio_squared * (phi_prime - two * phi / q)
    q_d_a_d_q = ratio_squared * (q * phi_prime - two * phi)
    minimum_d_a_d_q = ratio_squared * (minimum * phi_prime - two * phi * ratio)
    gamma = zero
    q_gamma = zero
    if dot != zero:
        gamma = (dot / scaled_denom) * d_a_d_q
        q_gamma = (dot / scaled_denom) * q_d_a_d_q
    elif j == ref and minimum * minimum == zero:
        gamma = (
            reference_dot * minimum * minimum_d_a_d_q / (scaled_denom * scaled_denom)
        )
        q_gamma = (
            reference_dot
            * minimum
            * (q * minimum_d_a_d_q)
            / (scaled_denom * scaled_denom)
        )

    if need_controls != 0:
        for d in range(n_dims):
            normalized_delta = _normalized_component(
                points[b, i, d], controls[b, j, d], radius
            )
            value = zero
            if normalized_delta != zero:
                value = (gamma / radius) * (normalized_delta / q)
            wp.atomic_sub(grad_controls, b, j, d, value)
    if need_radii != 0:
        wp.atomic_add(grad_radii, b, j, -q_gamma / radius)


@wp.kernel
def _shepard_point_backward(
    points: wp.array3d(dtype=Any),
    controls: wp.array3d(dtype=Any),
    control_displacements: wp.array3d(dtype=Any),
    radii: wp.array2d(dtype=Any),
    min_q: wp.array2d(dtype=Any),
    denominator: wp.array2d(dtype=Any),
    exact_count: wp.array2d(dtype=wp.int32),
    reference_index: wp.array2d(dtype=wp.int32),
    correction: wp.array3d(dtype=Any),
    grad_field: wp.array3d(dtype=Any),
    n_controls: int,
    n_dims: int,
    grad_points: wp.array3d(dtype=Any),
):
    """Query-centric point pullback with no inter-control atomics."""

    b, i = wp.tid()
    zero = type(points[b, i, 0])(0.0)
    one = type(zero)(1.0)
    two = type(zero)(2.0)
    for d in range(n_dims):
        grad_points[b, i, d] = zero
    if exact_count[b, i] > 0:
        return

    minimum = min_q[b, i]
    denom = denominator[b, i]
    reference_phi = _wendland_phi(minimum)
    scaled_denom = reference_phi * denom
    ref = reference_index[b, i]

    for j in range(n_controls):
        q = _normalized_distance(points, controls, radii, b, i, j, n_dims)
        coincident = q == zero
        if not coincident and q < one:
            radius = radii[b, j]
            phi = _wendland_phi(q)
            phi_prime = _wendland_phi_prime(q)
            ratio = minimum / q
            ratio_squared = ratio * ratio
            base_dot = zero
            correction_dot = zero
            reference_dot = zero
            for d in range(n_dims):
                g = grad_field[b, i, d]
                base_dot = base_dot + g * (
                    control_displacements[b, j, d] - control_displacements[b, ref, d]
                )
                correction_dot = correction_dot + g * correction[b, i, d]
                reference_dot = reference_dot + g * control_displacements[b, ref, d]
            dot = base_dot - correction_dot
            d_a_d_q = ratio_squared * (phi_prime - two * phi / q)
            minimum_d_a_d_q = ratio_squared * (minimum * phi_prime - two * phi * ratio)
            gamma = zero
            if dot != zero:
                gamma = (dot / scaled_denom) * d_a_d_q
            elif j == ref and minimum * minimum == zero:
                gamma = (
                    reference_dot
                    * minimum
                    * minimum_d_a_d_q
                    / (scaled_denom * scaled_denom)
                )
            for d in range(n_dims):
                normalized_delta = _normalized_component(
                    points[b, i, d], controls[b, j, d], radius
                )
                if normalized_delta != zero:
                    grad_points[b, i, d] = grad_points[b, i, d] + (
                        (gamma / radius) * (normalized_delta / q)
                    )


def _precision_overload(kernel, dtype, array3d_args, array2d_args):
    """Instantiate one concrete-precision overload of a generic kernel."""

    arg_types = {name: wp.array3d(dtype=dtype) for name in array3d_args}
    arg_types.update({name: wp.array2d(dtype=dtype) for name in array2d_args})
    return wp.overload(kernel, arg_types)


_FORWARD_3D = ("points", "controls", "control_displacements", "field", "correction")
_FORWARD_2D = ("radii", "min_q", "denominator")
_BACKWARD_3D = (
    "points",
    "controls",
    "control_displacements",
    "correction",
    "grad_field",
    "grad_controls",
    "grad_control_displacements",
)
_BACKWARD_2D = ("radii", "min_q", "denominator", "grad_radii")
_POINT_BACKWARD_3D = (
    "points",
    "controls",
    "control_displacements",
    "correction",
    "grad_field",
    "grad_points",
)
_POINT_BACKWARD_2D = ("radii", "min_q", "denominator")

shepard_forward_f32 = _precision_overload(
    _shepard_forward, wp.float32, _FORWARD_3D, _FORWARD_2D
)
shepard_forward_f64 = _precision_overload(
    _shepard_forward, wp.float64, _FORWARD_3D, _FORWARD_2D
)
shepard_backward_f32 = _precision_overload(
    _shepard_backward, wp.float32, _BACKWARD_3D, _BACKWARD_2D
)
shepard_backward_f64 = _precision_overload(
    _shepard_backward, wp.float64, _BACKWARD_3D, _BACKWARD_2D
)
shepard_point_backward_f32 = _precision_overload(
    _shepard_point_backward, wp.float32, _POINT_BACKWARD_3D, _POINT_BACKWARD_2D
)
shepard_point_backward_f64 = _precision_overload(
    _shepard_point_backward, wp.float64, _POINT_BACKWARD_3D, _POINT_BACKWARD_2D
)


__all__ = [
    "shepard_backward_f32",
    "shepard_backward_f64",
    "shepard_forward_f32",
    "shepard_forward_f64",
    "shepard_point_backward_f32",
    "shepard_point_backward_f64",
]
