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

"""Warp kernels for reference-relative deformation energies.

The simplex kernels are specialized for each supported coordinate/simplex
dimension. This keeps the generated Warp adjoints free of runtime loop-carried
state while retaining one numerical definition for float32 and float64.
"""

from __future__ import annotations

from typing import Any, Callable

import warp as wp


@wp.func
def _sort_simplex_indices(
    vertex_0: wp.int64,
    vertex_1: wp.int64,
    vertex_2: wp.int64,
    vertex_3: wp.int64,
):
    """Canonicalize cell order without a separate device sort or allocation."""

    index_0 = vertex_0
    index_1 = vertex_1
    index_2 = vertex_2
    index_3 = vertex_3
    if index_0 > index_1:
        temporary = index_0
        index_0 = index_1
        index_1 = temporary
    if index_2 > index_3:
        temporary = index_2
        index_2 = index_3
        index_3 = temporary
    if index_0 > index_2:
        temporary = index_0
        index_0 = index_2
        index_2 = temporary
    if index_1 > index_3:
        temporary = index_1
        index_1 = index_3
        index_3 = temporary
    if index_1 > index_2:
        temporary = index_1
        index_1 = index_2
        index_2 = temporary
    return wp.vec4l(index_0, index_1, index_2, index_3)


@wp.func
def _edge_determinant_1(
    values: wp.array3d(dtype=Any),
    batch: int,
    origin: wp.int64,
    vertex_1: wp.int64,
    vertex_2: wp.int64,
    vertex_3: wp.int64,
):
    return values[batch, vertex_1, 0] - values[batch, origin, 0]


@wp.func
def _edge_determinant_2(
    values: wp.array3d(dtype=Any),
    batch: int,
    origin: wp.int64,
    vertex_1: wp.int64,
    vertex_2: wp.int64,
    vertex_3: wp.int64,
):
    x10 = values[batch, vertex_1, 0] - values[batch, origin, 0]
    x11 = values[batch, vertex_1, 1] - values[batch, origin, 1]
    x20 = values[batch, vertex_2, 0] - values[batch, vertex_1, 0]
    x21 = values[batch, vertex_2, 1] - values[batch, vertex_1, 1]
    return x10 * x21 - x11 * x20


@wp.func
def _edge_determinant_3(
    values: wp.array3d(dtype=Any),
    batch: int,
    origin: wp.int64,
    vertex_1: wp.int64,
    vertex_2: wp.int64,
    vertex_3: wp.int64,
):
    x10 = values[batch, vertex_1, 0] - values[batch, origin, 0]
    x11 = values[batch, vertex_1, 1] - values[batch, origin, 1]
    x12 = values[batch, vertex_1, 2] - values[batch, origin, 2]
    x20 = values[batch, vertex_2, 0] - values[batch, vertex_1, 0]
    x21 = values[batch, vertex_2, 1] - values[batch, vertex_1, 1]
    x22 = values[batch, vertex_2, 2] - values[batch, vertex_1, 2]
    x30 = values[batch, vertex_3, 0] - values[batch, vertex_2, 0]
    x31 = values[batch, vertex_3, 1] - values[batch, vertex_2, 1]
    x32 = values[batch, vertex_3, 2] - values[batch, vertex_2, 2]
    return (
        x10 * (x21 * x32 - x22 * x31)
        - x20 * (x11 * x32 - x12 * x31)
        + x30 * (x11 * x22 - x12 * x21)
    )


_EDGE_DETERMINANT: dict[int, Callable] = {
    1: _edge_determinant_1,
    2: _edge_determinant_2,
    3: _edge_determinant_3,
}


def _name_kernel(function: Callable, name: str):
    """Give a closure-specialized kernel a stable, unique Warp key."""

    function.__name__ = name
    function.__qualname__ = name
    return wp.kernel(function)


def _make_simplex_stvk_kernel(coordinate_dimension: int, simplex_dimension: int):
    factorial = (1, 1, 2, 6)[simplex_dimension]
    inverse_dimension = (0.0, 1.0, 0.5, 1.0 / 3.0)[simplex_dimension]

    def kernel(
        points: wp.array3d(dtype=Any),
        reference_points: wp.array3d(dtype=Any),
        simplices: wp.array2d(dtype=wp.int64),
        num_points: int,
        lame_lambda: Any,
        shear_modulus: Any,
        terms: wp.array2d(dtype=Any),
    ):
        batch, primitive = wp.tid()
        vertex_0 = simplices[primitive, 0]
        vertex_1 = simplices[primitive, 1]
        vertex_2 = wp.int64(num_points)
        vertex_3 = wp.int64(num_points)
        valid = vertex_0 >= 0 and vertex_0 < num_points
        valid = valid and vertex_1 >= 0 and vertex_1 < num_points
        if wp.static(simplex_dimension >= 2):
            vertex_2 = simplices[primitive, 2]
            valid = valid and vertex_2 >= 0 and vertex_2 < num_points
        if wp.static(simplex_dimension == 3):
            vertex_3 = simplices[primitive, 3]
            valid = valid and vertex_3 >= 0 and vertex_3 < num_points
        if not valid:
            terms[batch, primitive] = type(terms[batch, primitive])(wp.nan)
            return
        ordered_vertices = _sort_simplex_indices(vertex_0, vertex_1, vertex_2, vertex_3)
        vertex_0 = ordered_vertices[0]
        vertex_1 = ordered_vertices[1]
        vertex_2 = ordered_vertices[2]
        vertex_3 = ordered_vertices[3]

        zero = type(points[batch, vertex_0, 0])(0.0)
        one = type(zero)(1.0)

        # Current and reference edge rows. Unused coordinates and edges stay
        # zero so the same scalar algebra specializes cleanly for all m <= D.
        c0x = points[batch, vertex_1, 0] - points[batch, vertex_0, 0]
        c0y = zero
        c0z = zero
        r0x = (
            reference_points[batch, vertex_1, 0] - reference_points[batch, vertex_0, 0]
        )
        r0y = zero
        r0z = zero
        if wp.static(coordinate_dimension >= 2):
            c0y = points[batch, vertex_1, 1] - points[batch, vertex_0, 1]
            r0y = (
                reference_points[batch, vertex_1, 1]
                - reference_points[batch, vertex_0, 1]
            )
        if wp.static(coordinate_dimension == 3):
            c0z = points[batch, vertex_1, 2] - points[batch, vertex_0, 2]
            r0z = (
                reference_points[batch, vertex_1, 2]
                - reference_points[batch, vertex_0, 2]
            )

        c1x = zero
        c1y = zero
        c1z = zero
        r1x = zero
        r1y = zero
        r1z = zero
        if wp.static(simplex_dimension >= 2):
            c1x = points[batch, vertex_2, 0] - points[batch, vertex_1, 0]
            r1x = (
                reference_points[batch, vertex_2, 0]
                - reference_points[batch, vertex_1, 0]
            )
            if wp.static(coordinate_dimension >= 2):
                c1y = points[batch, vertex_2, 1] - points[batch, vertex_1, 1]
                r1y = (
                    reference_points[batch, vertex_2, 1]
                    - reference_points[batch, vertex_1, 1]
                )
            if wp.static(coordinate_dimension == 3):
                c1z = points[batch, vertex_2, 2] - points[batch, vertex_1, 2]
                r1z = (
                    reference_points[batch, vertex_2, 2]
                    - reference_points[batch, vertex_1, 2]
                )

        c2x = zero
        c2y = zero
        c2z = zero
        r2x = zero
        r2y = zero
        r2z = zero
        if wp.static(simplex_dimension == 3):
            c2x = points[batch, vertex_3, 0] - points[batch, vertex_2, 0]
            c2y = points[batch, vertex_3, 1] - points[batch, vertex_2, 1]
            c2z = points[batch, vertex_3, 2] - points[batch, vertex_2, 2]
            r2x = (
                reference_points[batch, vertex_3, 0]
                - reference_points[batch, vertex_2, 0]
            )
            r2y = (
                reference_points[batch, vertex_3, 1]
                - reference_points[batch, vertex_2, 1]
            )
            r2z = (
                reference_points[batch, vertex_3, 2]
                - reference_points[batch, vertex_2, 2]
            )

        finite = wp.isfinite(c0x) and wp.isfinite(r0x)
        if wp.static(coordinate_dimension >= 2):
            finite = finite and wp.isfinite(c0y) and wp.isfinite(r0y)
        if wp.static(coordinate_dimension == 3):
            finite = finite and wp.isfinite(c0z) and wp.isfinite(r0z)
        if wp.static(simplex_dimension >= 2):
            finite = finite and wp.isfinite(c1x) and wp.isfinite(r1x)
            finite = finite and wp.isfinite(c1y) and wp.isfinite(r1y)
            if wp.static(coordinate_dimension == 3):
                finite = finite and wp.isfinite(c1z) and wp.isfinite(r1z)
        if wp.static(simplex_dimension == 3):
            finite = finite and wp.isfinite(c2x) and wp.isfinite(r2x)
            finite = finite and wp.isfinite(c2y) and wp.isfinite(r2y)
            finite = finite and wp.isfinite(c2z) and wp.isfinite(r2z)
        if not finite:
            terms[batch, primitive] = type(zero)(wp.nan)
            return

        # Modified Gram--Schmidt gives E0 = L Q without forming E0 E0^T.
        # Solving L H = E then gives C = H H^T in the reference-orthonormal
        # simplex frame. This avoids the conditioning loss from a Gram inverse.
        l00 = wp.sqrt(r0x * r0x + r0y * r0y + r0z * r0z)
        if l00 <= zero or not wp.isfinite(l00):
            terms[batch, primitive] = type(zero)(wp.nan)
            return
        q0x = r0x / l00
        q0y = r0y / l00
        q0z = r0z / l00
        h0x = c0x / l00
        h0y = c0y / l00
        h0z = c0z / l00
        c00 = h0x * h0x + h0y * h0y + h0z * h0z
        trace_p = c00 - one
        deviatoric_p_squared = zero
        reference_scale = l00

        if wp.static(simplex_dimension >= 2):
            l10 = r1x * q0x + r1y * q0y + r1z * q0z
            u1x = r1x - l10 * q0x
            u1y = r1y - l10 * q0y
            u1z = r1z - l10 * q0z
            l11 = wp.sqrt(u1x * u1x + u1y * u1y + u1z * u1z)
            if l11 <= zero or not wp.isfinite(l11):
                terms[batch, primitive] = type(zero)(wp.nan)
                return
            q1x = u1x / l11
            q1y = u1y / l11
            q1z = u1z / l11
            h1x = (c1x - l10 * h0x) / l11
            h1y = (c1y - l10 * h0y) / l11
            h1z = (c1z - l10 * h0z) / l11
            c01 = h0x * h1x + h0y * h1y + h0z * h1z
            c11 = h1x * h1x + h1y * h1y + h1z * h1z
            p00 = c00 - one
            p11 = c11 - one
            trace_p = p00 + p11
            mean_trace_p = trace_p / type(zero)(2.0)
            d00 = p00 - mean_trace_p
            d11 = p11 - mean_trace_p
            deviatoric_p_squared = d00 * d00 + d11 * d11 + type(zero)(2.0) * c01 * c01
            reference_scale = l00 * l11

        if wp.static(simplex_dimension == 3):
            l20 = r2x * q0x + r2y * q0y + r2z * q0z
            u2x = r2x - l20 * q0x
            u2y = r2y - l20 * q0y
            u2z = r2z - l20 * q0z
            l21 = u2x * q1x + u2y * q1y + u2z * q1z
            u2x = u2x - l21 * q1x
            u2y = u2y - l21 * q1y
            u2z = u2z - l21 * q1z
            l22 = wp.sqrt(u2x * u2x + u2y * u2y + u2z * u2z)
            if l22 <= zero or not wp.isfinite(l22):
                terms[batch, primitive] = type(zero)(wp.nan)
                return
            h2x = (c2x - l20 * h0x - l21 * h1x) / l22
            h2y = (c2y - l20 * h0y - l21 * h1y) / l22
            h2z = (c2z - l20 * h0z - l21 * h1z) / l22
            c02 = h0x * h2x + h0y * h2y + h0z * h2z
            c12 = h1x * h2x + h1y * h2y + h1z * h2z
            c22 = h2x * h2x + h2y * h2y + h2z * h2z
            p00 = c00 - one
            p11 = c11 - one
            p22 = c22 - one
            trace_p = p00 + p11 + p22
            mean_trace_p = trace_p / type(zero)(3.0)
            d00 = p00 - mean_trace_p
            d11 = p11 - mean_trace_p
            d22 = p22 - mean_trace_p
            deviatoric_p_squared = (
                d00 * d00
                + d11 * d11
                + d22 * d22
                + type(zero)(2.0) * (c01 * c01 + c02 * c02 + c12 * c12)
            )
            reference_scale = l00 * l11 * l22

        if not wp.isfinite(reference_scale):
            terms[batch, primitive] = type(zero)(wp.nan)
            return
        reference_measure = reference_scale / type(zero)(factorial)
        if reference_measure <= zero or not wp.isfinite(reference_measure):
            terms[batch, primitive] = type(zero)(wp.nan)
            return
        volumetric_coefficient = (
            type(zero)(0.5) * lame_lambda
            + type(zero)(inverse_dimension) * shear_modulus
        )
        # ``P = C - I = 2 E``; the quarter factors map the manifestly
        # nonnegative deviatoric/volumetric split back to Green strain.
        terms[batch, primitive] = reference_measure * (
            type(zero)(0.25) * shear_modulus * deviatoric_p_squared
            + type(zero)(0.25) * volumetric_coefficient * trace_p * trace_p
        )

    return _name_kernel(
        kernel,
        f"simplex_stvk_d{coordinate_dimension}_m{simplex_dimension}",
    )


def _make_simplex_measure_kernel(coordinate_dimension: int, simplex_dimension: int):
    edge_determinant = _EDGE_DETERMINANT[coordinate_dimension]
    factorial = (1, 1, 2, 6)[simplex_dimension]

    def kernel(
        points: wp.array3d(dtype=Any),
        reference_points: wp.array3d(dtype=Any),
        simplices: wp.array2d(dtype=wp.int64),
        num_points: int,
        ratio: wp.array2d(dtype=Any),
        reference_measure: wp.array2d(dtype=Any),
    ):
        batch, primitive = wp.tid()
        vertex_0 = simplices[primitive, 0]
        vertex_1 = simplices[primitive, 1]
        vertex_2 = wp.int64(num_points)
        vertex_3 = wp.int64(num_points)
        valid = vertex_0 >= 0 and vertex_0 < num_points
        valid = valid and vertex_1 >= 0 and vertex_1 < num_points
        if wp.static(simplex_dimension >= 2):
            vertex_2 = simplices[primitive, 2]
            valid = valid and vertex_2 >= 0 and vertex_2 < num_points
        if wp.static(simplex_dimension == 3):
            vertex_3 = simplices[primitive, 3]
            valid = valid and vertex_3 >= 0 and vertex_3 < num_points
        if not valid:
            nan = type(ratio[batch, primitive])(wp.nan)
            ratio[batch, primitive] = nan
            reference_measure[batch, primitive] = nan
            return
        ordered_vertices = _sort_simplex_indices(vertex_0, vertex_1, vertex_2, vertex_3)
        vertex_0 = ordered_vertices[0]
        vertex_1 = ordered_vertices[1]
        vertex_2 = ordered_vertices[2]
        vertex_3 = ordered_vertices[3]

        if wp.static(simplex_dimension == coordinate_dimension):
            current_signed = edge_determinant(
                points,
                batch,
                vertex_0,
                vertex_1,
                vertex_2,
                vertex_3,
            )
            reference_signed = edge_determinant(
                reference_points,
                batch,
                vertex_0,
                vertex_1,
                vertex_2,
                vertex_3,
            )
            if (
                reference_signed == type(reference_signed)(0.0)
                or not wp.isfinite(reference_signed)
                or not wp.isfinite(current_signed)
            ):
                nan = type(reference_signed)(wp.nan)
                ratio[batch, primitive] = nan
                reference_measure[batch, primitive] = nan
                return
            signed_ratio = current_signed / reference_signed
            if not wp.isfinite(signed_ratio):
                nan = type(signed_ratio)(wp.nan)
                ratio[batch, primitive] = nan
                reference_measure[batch, primitive] = nan
                return
            ratio[batch, primitive] = signed_ratio
            reference_measure[batch, primitive] = wp.abs(reference_signed) / type(
                reference_signed
            )(factorial)
        else:
            zero = type(ratio[batch, primitive])(0.0)
            current_squared = zero
            reference_squared = zero
            finite = True
            if wp.static(simplex_dimension == 1):
                cx = points[batch, vertex_1, 0] - points[batch, vertex_0, 0]
                rx = (
                    reference_points[batch, vertex_1, 0]
                    - reference_points[batch, vertex_0, 0]
                )
                cy = zero
                cz = zero
                ry = zero
                rz = zero
                if wp.static(coordinate_dimension >= 2):
                    cy = points[batch, vertex_1, 1] - points[batch, vertex_0, 1]
                    ry = (
                        reference_points[batch, vertex_1, 1]
                        - reference_points[batch, vertex_0, 1]
                    )
                if wp.static(coordinate_dimension == 3):
                    cz = points[batch, vertex_1, 2] - points[batch, vertex_0, 2]
                    rz = (
                        reference_points[batch, vertex_1, 2]
                        - reference_points[batch, vertex_0, 2]
                    )
                finite = (
                    wp.isfinite(cx)
                    and wp.isfinite(cy)
                    and wp.isfinite(cz)
                    and wp.isfinite(rx)
                    and wp.isfinite(ry)
                    and wp.isfinite(rz)
                )
                current_squared = cx * cx + cy * cy + cz * cz
                reference_squared = rx * rx + ry * ry + rz * rz
            if wp.static(simplex_dimension == 2):
                c10 = points[batch, vertex_1, 0] - points[batch, vertex_0, 0]
                c11 = points[batch, vertex_1, 1] - points[batch, vertex_0, 1]
                c12 = points[batch, vertex_1, 2] - points[batch, vertex_0, 2]
                c20 = points[batch, vertex_2, 0] - points[batch, vertex_1, 0]
                c21 = points[batch, vertex_2, 1] - points[batch, vertex_1, 1]
                c22 = points[batch, vertex_2, 2] - points[batch, vertex_1, 2]
                r10 = (
                    reference_points[batch, vertex_1, 0]
                    - reference_points[batch, vertex_0, 0]
                )
                r11 = (
                    reference_points[batch, vertex_1, 1]
                    - reference_points[batch, vertex_0, 1]
                )
                r12 = (
                    reference_points[batch, vertex_1, 2]
                    - reference_points[batch, vertex_0, 2]
                )
                r20 = (
                    reference_points[batch, vertex_2, 0]
                    - reference_points[batch, vertex_1, 0]
                )
                r21 = (
                    reference_points[batch, vertex_2, 1]
                    - reference_points[batch, vertex_1, 1]
                )
                r22 = (
                    reference_points[batch, vertex_2, 2]
                    - reference_points[batch, vertex_1, 2]
                )
                current_cross_x = c11 * c22 - c12 * c21
                current_cross_y = c12 * c20 - c10 * c22
                current_cross_z = c10 * c21 - c11 * c20
                reference_cross_x = r11 * r22 - r12 * r21
                reference_cross_y = r12 * r20 - r10 * r22
                reference_cross_z = r10 * r21 - r11 * r20
                finite = (
                    wp.isfinite(c10)
                    and wp.isfinite(c11)
                    and wp.isfinite(c12)
                    and wp.isfinite(c20)
                    and wp.isfinite(c21)
                    and wp.isfinite(c22)
                    and wp.isfinite(r10)
                    and wp.isfinite(r11)
                    and wp.isfinite(r12)
                    and wp.isfinite(r20)
                    and wp.isfinite(r21)
                    and wp.isfinite(r22)
                )
                current_squared = (
                    current_cross_x * current_cross_x
                    + current_cross_y * current_cross_y
                    + current_cross_z * current_cross_z
                )
                reference_squared = (
                    reference_cross_x * reference_cross_x
                    + reference_cross_y * reference_cross_y
                    + reference_cross_z * reference_cross_z
                )

            if (
                not finite
                or reference_squared <= zero
                or not wp.isfinite(reference_squared)
                or not wp.isfinite(current_squared)
            ):
                nan = type(zero)(wp.nan)
                ratio[batch, primitive] = nan
                reference_measure[batch, primitive] = nan
                return
            reference_length = wp.sqrt(reference_squared)
            reference_measure[batch, primitive] = reference_length / type(zero)(
                factorial
            )
            if current_squared <= zero:
                # Intrinsic measure is nonsmooth at collapse. Avoid executing
                # sqrt(0) so the generated adjoint selects the zero subgradient.
                ratio[batch, primitive] = zero
            else:
                ratio[batch, primitive] = wp.sqrt(current_squared) / reference_length

    return _name_kernel(
        kernel,
        f"simplex_measure_d{coordinate_dimension}_m{simplex_dimension}",
    )


def _make_simplex_inversion_kernel(dimension: int):
    edge_determinant = _EDGE_DETERMINANT[dimension]
    factorial = (1, 1, 2, 6)[dimension]

    def kernel(
        points: wp.array3d(dtype=Any),
        reference_points: wp.array3d(dtype=Any),
        simplices: wp.array2d(dtype=wp.int64),
        num_points: int,
        minimum_jacobian: Any,
        terms: wp.array2d(dtype=Any),
    ):
        batch, primitive = wp.tid()
        vertex_0 = simplices[primitive, 0]
        vertex_1 = simplices[primitive, 1]
        vertex_2 = wp.int64(num_points)
        vertex_3 = wp.int64(num_points)
        valid = vertex_0 >= 0 and vertex_0 < num_points
        valid = valid and vertex_1 >= 0 and vertex_1 < num_points
        if wp.static(dimension >= 2):
            vertex_2 = simplices[primitive, 2]
            valid = valid and vertex_2 >= 0 and vertex_2 < num_points
        if wp.static(dimension == 3):
            vertex_3 = simplices[primitive, 3]
            valid = valid and vertex_3 >= 0 and vertex_3 < num_points
        if not valid:
            terms[batch, primitive] = type(terms[batch, primitive])(wp.nan)
            return
        ordered_vertices = _sort_simplex_indices(vertex_0, vertex_1, vertex_2, vertex_3)
        vertex_0 = ordered_vertices[0]
        vertex_1 = ordered_vertices[1]
        vertex_2 = ordered_vertices[2]
        vertex_3 = ordered_vertices[3]

        current_signed = edge_determinant(
            points, batch, vertex_0, vertex_1, vertex_2, vertex_3
        )
        reference_signed = edge_determinant(
            reference_points, batch, vertex_0, vertex_1, vertex_2, vertex_3
        )
        if (
            reference_signed == type(reference_signed)(0.0)
            or not wp.isfinite(reference_signed)
            or not wp.isfinite(current_signed)
        ):
            terms[batch, primitive] = type(reference_signed)(wp.nan)
            return
        reference_measure = wp.abs(reference_signed) / type(reference_signed)(factorial)
        jacobian = current_signed / reference_signed
        if not wp.isfinite(jacobian):
            terms[batch, primitive] = type(jacobian)(wp.nan)
            return
        violation = wp.max(minimum_jacobian - jacobian, type(jacobian)(0.0))
        terms[batch, primitive] = (
            type(jacobian)(0.5) * reference_measure * violation * violation
        )

    return _name_kernel(kernel, f"simplex_inversion_d{dimension}")


@wp.func
def _detached_scale(value: wp.float32):
    """Return a float32 scale without propagating its derivative."""

    return value


@wp.func_grad(_detached_scale)
def _detached_scale_grad_f32(value: wp.float32, adj_ret: wp.float32):
    pass


@wp.func
def _detached_scale(value: wp.float64):
    """Return a float64 scale without propagating its derivative."""

    return value


@wp.func_grad(_detached_scale)
def _detached_scale_grad_f64(value: wp.float64, adj_ret: wp.float64):
    pass


@wp.func
def _signed_dihedral_from_geometry(
    edge_x: Any,
    edge_y: Any,
    edge_z: Any,
    edge_length: Any,
    left_x: Any,
    left_y: Any,
    left_z: Any,
    left_length: Any,
    right_x: Any,
    right_y: Any,
    right_z: Any,
    right_length: Any,
):
    inverse_edge = type(edge_length)(1.0) / edge_length
    inverse_left = type(left_length)(1.0) / left_length
    inverse_right = type(right_length)(1.0) / right_length
    ux = edge_x * inverse_edge
    uy = edge_y * inverse_edge
    uz = edge_z * inverse_edge
    nlx = left_x * inverse_left
    nly = left_y * inverse_left
    nlz = left_z * inverse_left
    nrx = right_x * inverse_right
    nry = right_y * inverse_right
    nrz = right_z * inverse_right
    cross_x = nly * nrz - nlz * nry
    cross_y = nlz * nrx - nlx * nrz
    cross_z = nlx * nry - nly * nrx
    sine = ux * cross_x + uy * cross_y + uz * cross_z
    cosine = nlx * nrx + nly * nry + nlz * nrz
    return wp.atan2(sine, cosine)


@wp.kernel
def _hinge_bending(
    points: wp.array3d(dtype=Any),
    reference_points: wp.array3d(dtype=Any),
    hinges: wp.array2d(dtype=wp.int64),
    num_points: int,
    terms: wp.array2d(dtype=Any),
):
    batch, hinge = wp.tid()
    vertex_i = hinges[hinge, 0]
    vertex_j = hinges[hinge, 1]
    vertex_k = hinges[hinge, 2]
    vertex_l = hinges[hinge, 3]
    valid = vertex_i >= 0 and vertex_i < num_points
    valid = valid and vertex_j >= 0 and vertex_j < num_points
    valid = valid and vertex_k >= 0 and vertex_k < num_points
    valid = valid and vertex_l >= 0 and vertex_l < num_points
    if not valid:
        terms[batch, hinge] = type(terms[batch, hinge])(wp.nan)
        return

    current_edge_x = points[batch, vertex_j, 0] - points[batch, vertex_i, 0]
    current_edge_y = points[batch, vertex_j, 1] - points[batch, vertex_i, 1]
    current_edge_z = points[batch, vertex_j, 2] - points[batch, vertex_i, 2]
    current_k_x = points[batch, vertex_k, 0] - points[batch, vertex_i, 0]
    current_k_y = points[batch, vertex_k, 1] - points[batch, vertex_i, 1]
    current_k_z = points[batch, vertex_k, 2] - points[batch, vertex_i, 2]
    current_l_x = points[batch, vertex_l, 0] - points[batch, vertex_j, 0]
    current_l_y = points[batch, vertex_l, 1] - points[batch, vertex_j, 1]
    current_l_z = points[batch, vertex_l, 2] - points[batch, vertex_j, 2]
    # Hinge energy is invariant to a common coordinate scale. Normalize before
    # products and norms; _detached_scale keeps the original adjoint semantics.
    current_scale = _detached_scale(
        wp.max(
            wp.max(
                wp.max(wp.abs(current_edge_x), wp.abs(current_edge_y)),
                wp.max(wp.abs(current_edge_z), wp.abs(current_k_x)),
            ),
            wp.max(
                wp.max(wp.abs(current_k_y), wp.abs(current_k_z)),
                wp.max(
                    wp.max(wp.abs(current_l_x), wp.abs(current_l_y)),
                    wp.abs(current_l_z),
                ),
            ),
        )
    )
    if current_scale > type(current_scale)(0.0) and wp.isfinite(current_scale):
        current_edge_x = current_edge_x / current_scale
        current_edge_y = current_edge_y / current_scale
        current_edge_z = current_edge_z / current_scale
        current_k_x = current_k_x / current_scale
        current_k_y = current_k_y / current_scale
        current_k_z = current_k_z / current_scale
        current_l_x = current_l_x / current_scale
        current_l_y = current_l_y / current_scale
        current_l_z = current_l_z / current_scale
    current_left_x = current_edge_y * current_k_z - current_edge_z * current_k_y
    current_left_y = current_edge_z * current_k_x - current_edge_x * current_k_z
    current_left_z = current_edge_x * current_k_y - current_edge_y * current_k_x
    current_right_x = -current_edge_y * current_l_z + current_edge_z * current_l_y
    current_right_y = -current_edge_z * current_l_x + current_edge_x * current_l_z
    current_right_z = -current_edge_x * current_l_y + current_edge_y * current_l_x
    current_edge_length = wp.sqrt(
        current_edge_x * current_edge_x
        + current_edge_y * current_edge_y
        + current_edge_z * current_edge_z
    )
    current_left_area = wp.sqrt(
        current_left_x * current_left_x
        + current_left_y * current_left_y
        + current_left_z * current_left_z
    )
    current_right_area = wp.sqrt(
        current_right_x * current_right_x
        + current_right_y * current_right_y
        + current_right_z * current_right_z
    )

    reference_edge_x = (
        reference_points[batch, vertex_j, 0] - reference_points[batch, vertex_i, 0]
    )
    reference_edge_y = (
        reference_points[batch, vertex_j, 1] - reference_points[batch, vertex_i, 1]
    )
    reference_edge_z = (
        reference_points[batch, vertex_j, 2] - reference_points[batch, vertex_i, 2]
    )
    reference_k_x = (
        reference_points[batch, vertex_k, 0] - reference_points[batch, vertex_i, 0]
    )
    reference_k_y = (
        reference_points[batch, vertex_k, 1] - reference_points[batch, vertex_i, 1]
    )
    reference_k_z = (
        reference_points[batch, vertex_k, 2] - reference_points[batch, vertex_i, 2]
    )
    reference_l_x = (
        reference_points[batch, vertex_l, 0] - reference_points[batch, vertex_j, 0]
    )
    reference_l_y = (
        reference_points[batch, vertex_l, 1] - reference_points[batch, vertex_j, 1]
    )
    reference_l_z = (
        reference_points[batch, vertex_l, 2] - reference_points[batch, vertex_j, 2]
    )
    reference_scale = _detached_scale(
        wp.max(
            wp.max(
                wp.max(wp.abs(reference_edge_x), wp.abs(reference_edge_y)),
                wp.max(wp.abs(reference_edge_z), wp.abs(reference_k_x)),
            ),
            wp.max(
                wp.max(wp.abs(reference_k_y), wp.abs(reference_k_z)),
                wp.max(
                    wp.max(wp.abs(reference_l_x), wp.abs(reference_l_y)),
                    wp.abs(reference_l_z),
                ),
            ),
        )
    )
    if reference_scale > type(reference_scale)(0.0) and wp.isfinite(reference_scale):
        reference_edge_x = reference_edge_x / reference_scale
        reference_edge_y = reference_edge_y / reference_scale
        reference_edge_z = reference_edge_z / reference_scale
        reference_k_x = reference_k_x / reference_scale
        reference_k_y = reference_k_y / reference_scale
        reference_k_z = reference_k_z / reference_scale
        reference_l_x = reference_l_x / reference_scale
        reference_l_y = reference_l_y / reference_scale
        reference_l_z = reference_l_z / reference_scale
    reference_left_x = (
        reference_edge_y * reference_k_z - reference_edge_z * reference_k_y
    )
    reference_left_y = (
        reference_edge_z * reference_k_x - reference_edge_x * reference_k_z
    )
    reference_left_z = (
        reference_edge_x * reference_k_y - reference_edge_y * reference_k_x
    )
    reference_right_x = (
        -reference_edge_y * reference_l_z + reference_edge_z * reference_l_y
    )
    reference_right_y = (
        -reference_edge_z * reference_l_x + reference_edge_x * reference_l_z
    )
    reference_right_z = (
        -reference_edge_x * reference_l_y + reference_edge_y * reference_l_x
    )
    reference_edge_length = wp.sqrt(
        reference_edge_x * reference_edge_x
        + reference_edge_y * reference_edge_y
        + reference_edge_z * reference_edge_z
    )
    reference_left_area = wp.sqrt(
        reference_left_x * reference_left_x
        + reference_left_y * reference_left_y
        + reference_left_z * reference_left_z
    )
    reference_right_area = wp.sqrt(
        reference_right_x * reference_right_x
        + reference_right_y * reference_right_y
        + reference_right_z * reference_right_z
    )
    valid_geometry = (
        reference_edge_length > type(reference_edge_length)(0.0)
        and reference_left_area > type(reference_left_area)(0.0)
        and reference_right_area > type(reference_right_area)(0.0)
        and wp.isfinite(reference_edge_length)
        and wp.isfinite(reference_left_area)
        and wp.isfinite(reference_right_area)
        and current_left_area > type(current_left_area)(0.0)
        and current_right_area > type(current_right_area)(0.0)
    )
    if not valid_geometry:
        terms[batch, hinge] = type(reference_edge_length)(wp.nan)
        return

    angle = _signed_dihedral_from_geometry(
        current_edge_x,
        current_edge_y,
        current_edge_z,
        current_edge_length,
        current_left_x,
        current_left_y,
        current_left_z,
        current_left_area,
        current_right_x,
        current_right_y,
        current_right_z,
        current_right_area,
    )
    reference_angle = _signed_dihedral_from_geometry(
        reference_edge_x,
        reference_edge_y,
        reference_edge_z,
        reference_edge_length,
        reference_left_x,
        reference_left_y,
        reference_left_z,
        reference_left_area,
        reference_right_x,
        reference_right_y,
        reference_right_z,
        reference_right_area,
    )
    difference = angle - reference_angle
    wrapped = wp.atan2(wp.sin(difference), wp.cos(difference))
    reference_area_sum = type(reference_edge_length)(0.5) * (
        reference_left_area + reference_right_area
    )
    weight = reference_edge_length * reference_edge_length / reference_area_sum
    terms[batch, hinge] = type(weight)(0.5) * weight * wrapped * wrapped


@wp.kernel
def _closed_surface_volume_contributions(
    points: wp.array3d(dtype=Any),
    reference_points: wp.array3d(dtype=Any),
    triangles: wp.array2d(dtype=wp.int64),
    num_points: int,
    origin: wp.array2d(dtype=Any),
    reference_origin: wp.array2d(dtype=Any),
    current: wp.array2d(dtype=Any),
    reference: wp.array2d(dtype=Any),
):
    batch, triangle = wp.tid()
    vertex_0 = triangles[triangle, 0]
    vertex_1 = triangles[triangle, 1]
    vertex_2 = triangles[triangle, 2]
    valid = vertex_0 >= 0 and vertex_0 < num_points
    valid = valid and vertex_1 >= 0 and vertex_1 < num_points
    valid = valid and vertex_2 >= 0 and vertex_2 < num_points
    if not valid:
        nan = type(current[batch, triangle])(wp.nan)
        current[batch, triangle] = nan
        reference[batch, triangle] = nan
        return

    origin_x = origin[batch, 0]
    origin_y = origin[batch, 1]
    origin_z = origin[batch, 2]
    reference_origin_x = reference_origin[batch, 0]
    reference_origin_y = reference_origin[batch, 1]
    reference_origin_z = reference_origin[batch, 2]
    x0 = points[batch, vertex_0, 0] - origin_x
    y0 = points[batch, vertex_0, 1] - origin_y
    z0 = points[batch, vertex_0, 2] - origin_z
    x1 = points[batch, vertex_1, 0] - origin_x
    y1 = points[batch, vertex_1, 1] - origin_y
    z1 = points[batch, vertex_1, 2] - origin_z
    x2 = points[batch, vertex_2, 0] - origin_x
    y2 = points[batch, vertex_2, 1] - origin_y
    z2 = points[batch, vertex_2, 2] - origin_z
    reference_x0 = reference_points[batch, vertex_0, 0] - reference_origin_x
    reference_y0 = reference_points[batch, vertex_0, 1] - reference_origin_y
    reference_z0 = reference_points[batch, vertex_0, 2] - reference_origin_z
    reference_x1 = reference_points[batch, vertex_1, 0] - reference_origin_x
    reference_y1 = reference_points[batch, vertex_1, 1] - reference_origin_y
    reference_z1 = reference_points[batch, vertex_1, 2] - reference_origin_z
    reference_x2 = reference_points[batch, vertex_2, 0] - reference_origin_x
    reference_y2 = reference_points[batch, vertex_2, 1] - reference_origin_y
    reference_z2 = reference_points[batch, vertex_2, 2] - reference_origin_z
    current[batch, triangle] = (
        x0 * (y1 * z2 - z1 * y2) + y0 * (z1 * x2 - x1 * z2) + z0 * (x1 * y2 - y1 * x2)
    ) / type(x0)(6.0)
    reference[batch, triangle] = (
        reference_x0 * (reference_y1 * reference_z2 - reference_z1 * reference_y2)
        + reference_y0 * (reference_z1 * reference_x2 - reference_x1 * reference_z2)
        + reference_z0 * (reference_x1 * reference_y2 - reference_y1 * reference_x2)
    ) / type(reference_x0)(6.0)


def _overload_simplex_kernel(kernel, dtype, *, scalar_args=(), output_args=()):
    argument_types = {
        "points": wp.array3d(dtype=dtype),
        "reference_points": wp.array3d(dtype=dtype),
        "simplices": wp.array2d(dtype=wp.int64),
    }
    argument_types.update({name: dtype for name in scalar_args})
    argument_types.update({name: wp.array2d(dtype=dtype) for name in output_args})
    return wp.overload(kernel, argument_types)


_SIMPLEX_DIMENSIONS = tuple(
    (coordinate_dimension, simplex_dimension)
    for coordinate_dimension in (1, 2, 3)
    for simplex_dimension in range(1, coordinate_dimension + 1)
)
_STVK_BASE = {
    dimensions: _make_simplex_stvk_kernel(*dimensions)
    for dimensions in _SIMPLEX_DIMENSIONS
}
_MEASURE_BASE = {
    dimensions: _make_simplex_measure_kernel(*dimensions)
    for dimensions in _SIMPLEX_DIMENSIONS
}
_INVERSION_BASE = {
    dimension: _make_simplex_inversion_kernel(dimension) for dimension in (1, 2, 3)
}


def _simplex_precision_family(dtype):
    stvk = {
        dimensions: _overload_simplex_kernel(
            kernel,
            dtype,
            scalar_args=("lame_lambda", "shear_modulus"),
            output_args=("terms",),
        )
        for dimensions, kernel in _STVK_BASE.items()
    }
    measure = {
        dimensions: _overload_simplex_kernel(
            kernel,
            dtype,
            output_args=("ratio", "reference_measure"),
        )
        for dimensions, kernel in _MEASURE_BASE.items()
    }
    inversion = {
        (dimension, dimension): _overload_simplex_kernel(
            kernel,
            dtype,
            scalar_args=("minimum_jacobian",),
            output_args=("terms",),
        )
        for dimension, kernel in _INVERSION_BASE.items()
    }
    return stvk, measure, inversion


(
    SIMPLEX_STVK_F32,
    SIMPLEX_MEASURE_F32,
    SIMPLEX_INVERSION_F32,
) = _simplex_precision_family(wp.float32)
(
    SIMPLEX_STVK_F64,
    SIMPLEX_MEASURE_F64,
    SIMPLEX_INVERSION_F64,
) = _simplex_precision_family(wp.float64)

HINGE_BENDING_F32 = wp.overload(
    _hinge_bending,
    {
        "points": wp.array3d(dtype=wp.float32),
        "reference_points": wp.array3d(dtype=wp.float32),
        "terms": wp.array2d(dtype=wp.float32),
    },
)
HINGE_BENDING_F64 = wp.overload(
    _hinge_bending,
    {
        "points": wp.array3d(dtype=wp.float64),
        "reference_points": wp.array3d(dtype=wp.float64),
        "terms": wp.array2d(dtype=wp.float64),
    },
)
VOLUME_CONTRIBUTIONS_F32 = wp.overload(
    _closed_surface_volume_contributions,
    {
        "points": wp.array3d(dtype=wp.float32),
        "reference_points": wp.array3d(dtype=wp.float32),
        "origin": wp.array2d(dtype=wp.float32),
        "reference_origin": wp.array2d(dtype=wp.float32),
        "current": wp.array2d(dtype=wp.float32),
        "reference": wp.array2d(dtype=wp.float32),
    },
)
VOLUME_CONTRIBUTIONS_F64 = wp.overload(
    _closed_surface_volume_contributions,
    {
        "points": wp.array3d(dtype=wp.float64),
        "reference_points": wp.array3d(dtype=wp.float64),
        "origin": wp.array2d(dtype=wp.float64),
        "reference_origin": wp.array2d(dtype=wp.float64),
        "current": wp.array2d(dtype=wp.float64),
        "reference": wp.array2d(dtype=wp.float64),
    },
)


__all__ = [
    "HINGE_BENDING_F32",
    "HINGE_BENDING_F64",
    "SIMPLEX_INVERSION_F32",
    "SIMPLEX_INVERSION_F64",
    "SIMPLEX_MEASURE_F32",
    "SIMPLEX_MEASURE_F64",
    "SIMPLEX_STVK_F32",
    "SIMPLEX_STVK_F64",
    "VOLUME_CONTRIBUTIONS_F32",
    "VOLUME_CONTRIBUTIONS_F64",
]
