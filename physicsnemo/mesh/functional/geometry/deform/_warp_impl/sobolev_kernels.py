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

"""Warp kernels for uniform-mass P1 Sobolev deformation.

The kernels deliberately keep the global Helmholtz operator matrix-free. A
small dense stiffness matrix is stored per simplex, while applications scatter
one cell-row contribution per output degree of freedom.
"""

from typing import Any

import warp as wp


@wp.kernel
def assemble_segments(
    points: wp.array3d(dtype=Any),
    cells: wp.array2d(dtype=wp.int64),
    local_stiffness: wp.array4d(dtype=Any),
    stiffness_diagonal: wp.array2d(dtype=Any),
    mass_sum: wp.array1d(dtype=Any),
    invalid_geometry: wp.array1d(dtype=wp.int32),
    n_dims: int,
):
    """Assemble line-segment P1 stiffness matrices in any ambient dimension."""

    b, c = wp.tid()
    i0 = int(cells[c, 0])
    i1 = int(cells[c, 1])
    zero = type(points[b, i0, 0])(0.0)
    gram = zero
    for d in range(n_dims):
        edge = points[b, i1, d] - points[b, i0, d]
        gram += edge * edge

    if not wp.isfinite(gram) or gram <= zero:
        wp.atomic_max(invalid_geometry, b, 1)
        return

    measure = wp.sqrt(gram)
    value = measure / gram
    local_stiffness[b, c, 0, 0] = value
    local_stiffness[b, c, 0, 1] = -value
    local_stiffness[b, c, 1, 0] = -value
    local_stiffness[b, c, 1, 1] = value
    wp.atomic_add(stiffness_diagonal, b, i0, value)
    wp.atomic_add(stiffness_diagonal, b, i1, value)
    wp.atomic_add(mass_sum, b, measure)


@wp.kernel
def assemble_triangles(
    points: wp.array3d(dtype=Any),
    cells: wp.array2d(dtype=wp.int64),
    local_stiffness: wp.array4d(dtype=Any),
    stiffness_diagonal: wp.array2d(dtype=Any),
    mass_sum: wp.array1d(dtype=Any),
    invalid_geometry: wp.array1d(dtype=wp.int32),
    n_dims: int,
):
    """Assemble triangle P1 stiffness matrices in any ambient dimension."""

    b, c = wp.tid()
    i0 = int(cells[c, 0])
    i1 = int(cells[c, 1])
    i2 = int(cells[c, 2])
    zero = type(points[b, i0, 0])(0.0)
    g00 = zero
    g01 = zero
    g11 = zero
    for d in range(n_dims):
        e0 = points[b, i1, d] - points[b, i0, d]
        e1 = points[b, i2, d] - points[b, i0, d]
        g00 += e0 * e0
        g01 += e0 * e1
        g11 += e1 * e1

    determinant = g00 * g11 - g01 * g01
    if not wp.isfinite(determinant) or determinant <= zero:
        wp.atomic_max(invalid_geometry, b, 1)
        return

    measure = type(zero)(0.5) * wp.sqrt(determinant)
    q00 = g11 / determinant
    q01 = -g01 / determinant
    q11 = g00 / determinant

    k00 = measure * (q00 + type(zero)(2.0) * q01 + q11)
    k01 = -measure * (q00 + q01)
    k02 = -measure * (q01 + q11)
    k11 = measure * q00
    k12 = measure * q01
    k22 = measure * q11

    local_stiffness[b, c, 0, 0] = k00
    local_stiffness[b, c, 0, 1] = k01
    local_stiffness[b, c, 0, 2] = k02
    local_stiffness[b, c, 1, 0] = k01
    local_stiffness[b, c, 1, 1] = k11
    local_stiffness[b, c, 1, 2] = k12
    local_stiffness[b, c, 2, 0] = k02
    local_stiffness[b, c, 2, 1] = k12
    local_stiffness[b, c, 2, 2] = k22
    wp.atomic_add(stiffness_diagonal, b, i0, k00)
    wp.atomic_add(stiffness_diagonal, b, i1, k11)
    wp.atomic_add(stiffness_diagonal, b, i2, k22)
    wp.atomic_add(mass_sum, b, measure)


@wp.kernel
def assemble_tetrahedra(
    points: wp.array3d(dtype=Any),
    cells: wp.array2d(dtype=wp.int64),
    local_stiffness: wp.array4d(dtype=Any),
    stiffness_diagonal: wp.array2d(dtype=Any),
    mass_sum: wp.array1d(dtype=Any),
    invalid_geometry: wp.array1d(dtype=wp.int32),
    n_dims: int,
):
    """Assemble tetrahedron P1 stiffness matrices."""

    b, c = wp.tid()
    i0 = int(cells[c, 0])
    i1 = int(cells[c, 1])
    i2 = int(cells[c, 2])
    i3 = int(cells[c, 3])
    zero = type(points[b, i0, 0])(0.0)
    g00 = zero
    g01 = zero
    g02 = zero
    g11 = zero
    g12 = zero
    g22 = zero
    for d in range(n_dims):
        e0 = points[b, i1, d] - points[b, i0, d]
        e1 = points[b, i2, d] - points[b, i0, d]
        e2 = points[b, i3, d] - points[b, i0, d]
        g00 += e0 * e0
        g01 += e0 * e1
        g02 += e0 * e2
        g11 += e1 * e1
        g12 += e1 * e2
        g22 += e2 * e2

    c00 = g11 * g22 - g12 * g12
    c01 = g02 * g12 - g01 * g22
    c02 = g01 * g12 - g02 * g11
    c11 = g00 * g22 - g02 * g02
    c12 = g01 * g02 - g00 * g12
    c22 = g00 * g11 - g01 * g01
    determinant = g00 * c00 + g01 * c01 + g02 * c02
    if not wp.isfinite(determinant) or determinant <= zero:
        wp.atomic_max(invalid_geometry, b, 1)
        return

    measure = wp.sqrt(determinant) / type(zero)(6.0)
    q00 = c00 / determinant
    q01 = c01 / determinant
    q02 = c02 / determinant
    q11 = c11 / determinant
    q12 = c12 / determinant
    q22 = c22 / determinant

    row0 = q00 + q01 + q02
    row1 = q01 + q11 + q12
    row2 = q02 + q12 + q22
    k00 = measure * (row0 + row1 + row2)
    k01 = -measure * row0
    k02 = -measure * row1
    k03 = -measure * row2
    k11 = measure * q00
    k12 = measure * q01
    k13 = measure * q02
    k22 = measure * q11
    k23 = measure * q12
    k33 = measure * q22

    local_stiffness[b, c, 0, 0] = k00
    local_stiffness[b, c, 0, 1] = k01
    local_stiffness[b, c, 0, 2] = k02
    local_stiffness[b, c, 0, 3] = k03
    local_stiffness[b, c, 1, 0] = k01
    local_stiffness[b, c, 1, 1] = k11
    local_stiffness[b, c, 1, 2] = k12
    local_stiffness[b, c, 1, 3] = k13
    local_stiffness[b, c, 2, 0] = k02
    local_stiffness[b, c, 2, 1] = k12
    local_stiffness[b, c, 2, 2] = k22
    local_stiffness[b, c, 2, 3] = k23
    local_stiffness[b, c, 3, 0] = k03
    local_stiffness[b, c, 3, 1] = k13
    local_stiffness[b, c, 3, 2] = k23
    local_stiffness[b, c, 3, 3] = k33
    wp.atomic_add(stiffness_diagonal, b, i0, k00)
    wp.atomic_add(stiffness_diagonal, b, i1, k11)
    wp.atomic_add(stiffness_diagonal, b, i2, k22)
    wp.atomic_add(stiffness_diagonal, b, i3, k33)
    wp.atomic_add(mass_sum, b, measure)


@wp.kernel
def finalize_system(
    displacement: wp.array3d(dtype=Any),
    free_points: wp.array2d(dtype=wp.bool),
    mass_sum: wp.array1d(dtype=Any),
    stiffness_diagonal: wp.array2d(dtype=Any),
    connected_count: int,
    length_scale_squared: Any,
    use_displacement_initial: int,
    mass: wp.array2d(dtype=Any),
    right_hand_side: wp.array3d(dtype=Any),
    initial: wp.array3d(dtype=Any),
):
    """Finalize uniform mass, Jacobi diagonal, RHS, and initial iterate."""

    b, i, d = wp.tid()
    one = type(displacement[b, i, d])(1.0)
    zero = type(displacement[b, i, d])(0.0)
    mass_value = one
    if connected_count > 0:
        mass_value = mass_sum[b] / type(zero)(connected_count)
    if d == 0:
        mass[b, i] = mass_value
        if free_points[b, i]:
            stiffness_diagonal[b, i] = (
                mass_value + length_scale_squared * stiffness_diagonal[b, i]
            )
        else:
            stiffness_diagonal[b, i] = one

    if free_points[b, i]:
        right_hand_side[b, i, d] = mass_value * displacement[b, i, d]
        if use_displacement_initial != 0:
            initial[b, i, d] = displacement[b, i, d]
        else:
            initial[b, i, d] = zero
    else:
        right_hand_side[b, i, d] = zero
        initial[b, i, d] = zero


@wp.kernel
def initialize_adjoint(
    output_adjoint: wp.array3d(dtype=Any),
    free_points: wp.array2d(dtype=wp.bool),
    right_hand_side: wp.array3d(dtype=Any),
    initial: wp.array3d(dtype=Any),
):
    """Build the constrained adjoint right-hand side and zero initial guess."""

    b, i, d = wp.tid()
    zero = type(output_adjoint[b, i, d])(0.0)
    right_hand_side[b, i, d] = wp.where(
        free_points[b, i], output_adjoint[b, i, d], zero
    )
    initial[b, i, d] = zero


@wp.kernel
def helmholtz_base_matvec(
    x: wp.array1d(dtype=Any),
    y: wp.array1d(dtype=Any),
    mass: wp.array2d(dtype=Any),
    free_points: wp.array2d(dtype=wp.bool),
    n_points: int,
    n_dims: int,
    alpha: Any,
    beta: Any,
    z: wp.array1d(dtype=Any),
):
    """Write the mass/fixed part of ``z = alpha A x + beta y``."""

    index = wp.tid()
    point_linear = index // n_dims
    b = point_linear // n_points
    i = point_linear - b * n_points
    value = x[index]
    if free_points[b, i]:
        value = mass[b, i] * value
    result = alpha * value
    if beta != type(beta)(0):
        result += beta * y[index]
    z[index] = result


@wp.kernel
def helmholtz_stiffness_matvec(
    x: wp.array1d(dtype=Any),
    cells: wp.array2d(dtype=wp.int64),
    local_stiffness: wp.array4d(dtype=Any),
    free_points: wp.array2d(dtype=wp.bool),
    point_offsets: wp.array1d(dtype=wp.int64),
    point_incidence: wp.array1d(dtype=wp.int64),
    n_points: int,
    n_dims: int,
    n_cell_points: int,
    alpha_length_scale_squared: Any,
    z: wp.array1d(dtype=Any),
):
    """Gather the matrix-free stiffness part of a Helmholtz matvec."""

    b, i, d = wp.tid()
    if not free_points[b, i]:
        return
    index_i = (b * n_points + i) * n_dims + d
    value = type(x[index_i])(0.0)
    incidence_begin = int(point_offsets[i])
    incidence_end = int(point_offsets[i + 1])
    for incidence_index in range(incidence_begin, incidence_end):
        cell_entry = int(point_incidence[incidence_index])
        c = cell_entry // n_cell_points
        a = cell_entry - c * n_cell_points
        for j in range(n_cell_points):
            point_j = int(cells[c, j])
            if free_points[b, point_j]:
                index_j = (b * n_points + point_j) * n_dims + d
                value += local_stiffness[b, c, a, j] * x[index_j]
    z[index_i] += alpha_length_scale_squared * value


@wp.kernel
def jacobi_matvec(
    x: wp.array1d(dtype=Any),
    y: wp.array1d(dtype=Any),
    diagonal: wp.array2d(dtype=Any),
    n_points: int,
    n_dims: int,
    alpha: Any,
    beta: Any,
    z: wp.array1d(dtype=Any),
):
    """Apply the inverse Jacobi diagonal."""

    index = wp.tid()
    point_linear = index // n_dims
    b = point_linear // n_points
    i = point_linear - b * n_points
    result = alpha * x[index] / diagonal[b, i]
    if beta != type(beta)(0):
        result += beta * y[index]
    z[index] = result


@wp.kernel
def displacement_pullback(
    system_adjoint: wp.array3d(dtype=Any),
    mass: wp.array2d(dtype=Any),
    free_points: wp.array2d(dtype=wp.bool),
    displacement_adjoint: wp.array3d(dtype=Any),
):
    """Apply ``M`` to the implicit system adjoint."""

    b, i, d = wp.tid()
    zero = type(system_adjoint[b, i, d])(0.0)
    displacement_adjoint[b, i, d] = wp.where(
        free_points[b, i], mass[b, i] * system_adjoint[b, i, d], zero
    )


@wp.kernel
def accumulate_mass_geometry_coefficient(
    displacement: wp.array3d(dtype=Any),
    solution: wp.array3d(dtype=Any),
    system_adjoint: wp.array3d(dtype=Any),
    free_points: wp.array2d(dtype=wp.bool),
    coefficient: wp.array1d(dtype=Any),
):
    """Accumulate ``lambda . (d-u)`` for the uniform-mass geometry pullback."""

    b, i, d = wp.tid()
    if free_points[b, i]:
        wp.atomic_add(
            coefficient,
            b,
            system_adjoint[b, i, d] * (displacement[b, i, d] - solution[b, i, d]),
        )


@wp.kernel
def segment_geometry_pullback(
    points: wp.array3d(dtype=Any),
    cells: wp.array2d(dtype=wp.int64),
    solution: wp.array3d(dtype=Any),
    system_adjoint: wp.array3d(dtype=Any),
    mass_coefficient: wp.array1d(dtype=Any),
    connected_count: int,
    length_scale_squared: Any,
    n_dims: int,
    points_adjoint: wp.array3d(dtype=Any),
):
    """Pull back the implicit objective through segment geometry."""

    b, c = wp.tid()
    i0 = int(cells[c, 0])
    i1 = int(cells[c, 1])
    zero = type(points[b, i0, 0])(0.0)
    gram = zero
    for d in range(n_dims):
        edge = points[b, i1, d] - points[b, i0, d]
        gram += edge * edge
    measure = wp.sqrt(gram)
    inverse = type(zero)(1.0) / gram
    stiffness_pairing = zero
    symmetric_outer = zero
    for d in range(n_dims):
        a = system_adjoint[b, i1, d] - system_adjoint[b, i0, d]
        u = solution[b, i1, d] - solution[b, i0, d]
        p = inverse * a
        r = inverse * u
        stiffness_pairing += a * r
        symmetric_outer += type(zero)(2.0) * p * r
    mass_term = mass_coefficient[b] / type(zero)(connected_count)
    weight = measure * (
        (mass_term - length_scale_squared * stiffness_pairing) * inverse
        + length_scale_squared * symmetric_outer
    )
    for d in range(n_dims):
        edge = points[b, i1, d] - points[b, i0, d]
        value = weight * edge
        wp.atomic_add(points_adjoint, b, i1, d, value)
        wp.atomic_sub(points_adjoint, b, i0, d, value)


@wp.kernel
def triangle_geometry_pullback(
    points: wp.array3d(dtype=Any),
    cells: wp.array2d(dtype=wp.int64),
    solution: wp.array3d(dtype=Any),
    system_adjoint: wp.array3d(dtype=Any),
    mass_coefficient: wp.array1d(dtype=Any),
    connected_count: int,
    length_scale_squared: Any,
    n_dims: int,
    points_adjoint: wp.array3d(dtype=Any),
):
    """Pull back the implicit objective through triangle geometry."""

    b, c = wp.tid()
    i0 = int(cells[c, 0])
    i1 = int(cells[c, 1])
    i2 = int(cells[c, 2])
    zero = type(points[b, i0, 0])(0.0)
    g00 = zero
    g01 = zero
    g11 = zero
    for d in range(n_dims):
        e0 = points[b, i1, d] - points[b, i0, d]
        e1 = points[b, i2, d] - points[b, i0, d]
        g00 += e0 * e0
        g01 += e0 * e1
        g11 += e1 * e1
    determinant = g00 * g11 - g01 * g01
    measure = type(zero)(0.5) * wp.sqrt(determinant)
    q00 = g11 / determinant
    q01 = -g01 / determinant
    q11 = g00 / determinant

    stiffness_pairing = zero
    s00 = zero
    s01 = zero
    s11 = zero
    for d in range(n_dims):
        a0 = system_adjoint[b, i1, d] - system_adjoint[b, i0, d]
        a1 = system_adjoint[b, i2, d] - system_adjoint[b, i0, d]
        u0 = solution[b, i1, d] - solution[b, i0, d]
        u1 = solution[b, i2, d] - solution[b, i0, d]
        p0 = q00 * a0 + q01 * a1
        p1 = q01 * a0 + q11 * a1
        r0 = q00 * u0 + q01 * u1
        r1 = q01 * u0 + q11 * u1
        stiffness_pairing += a0 * r0 + a1 * r1
        s00 += type(zero)(2.0) * p0 * r0
        s01 += p0 * r1 + p1 * r0
        s11 += type(zero)(2.0) * p1 * r1

    mass_term = mass_coefficient[b] / type(zero)(connected_count)
    common = mass_term - length_scale_squared * stiffness_pairing
    w00 = measure * (common * q00 + length_scale_squared * s00)
    w01 = measure * (common * q01 + length_scale_squared * s01)
    w11 = measure * (common * q11 + length_scale_squared * s11)
    for d in range(n_dims):
        e0 = points[b, i1, d] - points[b, i0, d]
        e1 = points[b, i2, d] - points[b, i0, d]
        value0 = w00 * e0 + w01 * e1
        value1 = w01 * e0 + w11 * e1
        wp.atomic_add(points_adjoint, b, i1, d, value0)
        wp.atomic_add(points_adjoint, b, i2, d, value1)
        wp.atomic_sub(points_adjoint, b, i0, d, value0 + value1)


@wp.kernel
def tetrahedron_geometry_pullback(
    points: wp.array3d(dtype=Any),
    cells: wp.array2d(dtype=wp.int64),
    solution: wp.array3d(dtype=Any),
    system_adjoint: wp.array3d(dtype=Any),
    mass_coefficient: wp.array1d(dtype=Any),
    connected_count: int,
    length_scale_squared: Any,
    n_dims: int,
    points_adjoint: wp.array3d(dtype=Any),
):
    """Pull back the implicit objective through tetrahedron geometry."""

    b, c = wp.tid()
    i0 = int(cells[c, 0])
    i1 = int(cells[c, 1])
    i2 = int(cells[c, 2])
    i3 = int(cells[c, 3])
    zero = type(points[b, i0, 0])(0.0)
    g00 = zero
    g01 = zero
    g02 = zero
    g11 = zero
    g12 = zero
    g22 = zero
    for d in range(n_dims):
        e0 = points[b, i1, d] - points[b, i0, d]
        e1 = points[b, i2, d] - points[b, i0, d]
        e2 = points[b, i3, d] - points[b, i0, d]
        g00 += e0 * e0
        g01 += e0 * e1
        g02 += e0 * e2
        g11 += e1 * e1
        g12 += e1 * e2
        g22 += e2 * e2

    c00 = g11 * g22 - g12 * g12
    c01 = g02 * g12 - g01 * g22
    c02 = g01 * g12 - g02 * g11
    c11 = g00 * g22 - g02 * g02
    c12 = g01 * g02 - g00 * g12
    c22 = g00 * g11 - g01 * g01
    determinant = g00 * c00 + g01 * c01 + g02 * c02
    measure = wp.sqrt(determinant) / type(zero)(6.0)
    q00 = c00 / determinant
    q01 = c01 / determinant
    q02 = c02 / determinant
    q11 = c11 / determinant
    q12 = c12 / determinant
    q22 = c22 / determinant

    stiffness_pairing = zero
    s00 = zero
    s01 = zero
    s02 = zero
    s11 = zero
    s12 = zero
    s22 = zero
    for d in range(n_dims):
        a0 = system_adjoint[b, i1, d] - system_adjoint[b, i0, d]
        a1 = system_adjoint[b, i2, d] - system_adjoint[b, i0, d]
        a2 = system_adjoint[b, i3, d] - system_adjoint[b, i0, d]
        u0 = solution[b, i1, d] - solution[b, i0, d]
        u1 = solution[b, i2, d] - solution[b, i0, d]
        u2 = solution[b, i3, d] - solution[b, i0, d]
        p0 = q00 * a0 + q01 * a1 + q02 * a2
        p1 = q01 * a0 + q11 * a1 + q12 * a2
        p2 = q02 * a0 + q12 * a1 + q22 * a2
        r0 = q00 * u0 + q01 * u1 + q02 * u2
        r1 = q01 * u0 + q11 * u1 + q12 * u2
        r2 = q02 * u0 + q12 * u1 + q22 * u2
        stiffness_pairing += a0 * r0 + a1 * r1 + a2 * r2
        s00 += type(zero)(2.0) * p0 * r0
        s01 += p0 * r1 + p1 * r0
        s02 += p0 * r2 + p2 * r0
        s11 += type(zero)(2.0) * p1 * r1
        s12 += p1 * r2 + p2 * r1
        s22 += type(zero)(2.0) * p2 * r2

    mass_term = mass_coefficient[b] / type(zero)(connected_count)
    common = mass_term - length_scale_squared * stiffness_pairing
    w00 = measure * (common * q00 + length_scale_squared * s00)
    w01 = measure * (common * q01 + length_scale_squared * s01)
    w02 = measure * (common * q02 + length_scale_squared * s02)
    w11 = measure * (common * q11 + length_scale_squared * s11)
    w12 = measure * (common * q12 + length_scale_squared * s12)
    w22 = measure * (common * q22 + length_scale_squared * s22)
    for d in range(n_dims):
        e0 = points[b, i1, d] - points[b, i0, d]
        e1 = points[b, i2, d] - points[b, i0, d]
        e2 = points[b, i3, d] - points[b, i0, d]
        value0 = w00 * e0 + w01 * e1 + w02 * e2
        value1 = w01 * e0 + w11 * e1 + w12 * e2
        value2 = w02 * e0 + w12 * e1 + w22 * e2
        wp.atomic_add(points_adjoint, b, i1, d, value0)
        wp.atomic_add(points_adjoint, b, i2, d, value1)
        wp.atomic_add(points_adjoint, b, i3, d, value2)
        wp.atomic_sub(points_adjoint, b, i0, d, value0 + value1 + value2)


ASSEMBLY_KERNELS = {
    2: assemble_segments,
    3: assemble_triangles,
    4: assemble_tetrahedra,
}

GEOMETRY_PULLBACK_KERNELS = {
    2: segment_geometry_pullback,
    3: triangle_geometry_pullback,
    4: tetrahedron_geometry_pullback,
}


__all__ = [
    "ASSEMBLY_KERNELS",
    "GEOMETRY_PULLBACK_KERNELS",
    "accumulate_mass_geometry_coefficient",
    "displacement_pullback",
    "finalize_system",
    "helmholtz_base_matvec",
    "helmholtz_stiffness_matvec",
    "initialize_adjoint",
    "jacobi_matvec",
]
