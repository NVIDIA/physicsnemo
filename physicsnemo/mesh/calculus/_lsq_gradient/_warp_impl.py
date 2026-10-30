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

from __future__ import annotations

import math

import torch
import warp as wp

from physicsnemo.core.function_spec import FunctionSpec

from .utils import resolve_safe_epsilon, validate_inputs

### Warp runtime initialization for custom kernels.
wp.init()
wp.config.log_level = wp.LOG_WARNING


# Added to the diagonal of each normal matrix, so that the systems of degenerate
# neighborhoods stay stable in float32.
REG_EPS = 1.0e-6
_REG_EPS = wp.constant(REG_EPS)


@wp.kernel
def _mesh_lsq_gradient_1d_kernel(
    points: wp.array2d(dtype=wp.float32),
    values: wp.array(dtype=wp.float32),
    offsets: wp.array(dtype=wp.int32),
    indices: wp.array(dtype=wp.int32),
    weight_power: float,
    min_neighbors: int,
    reg_eps: float,
    dist_eps: float,
    gradients: wp.array2d(dtype=wp.float32),
    solved: wp.array(dtype=wp.int32),
):
    i = wp.tid()

    # Read the CSR neighbor segment for this entity.
    start = offsets[i]
    end = offsets[i + 1]
    count = end - start
    if count < min_neighbors:
        gradients[i, 0] = 0.0
        solved[i] = 0
        return

    # Gather center state and initialize normal-equation accumulators.
    px = points[i, 0]
    pval = values[i]

    m00 = float(reg_eps)
    b0 = float(0.0)

    # Accumulate A^T W A and A^T W b over neighbors.
    for p in range(start, end):
        n = indices[p]
        dx = points[n, 0] - px
        dphi = values[n] - pval

        dist2 = wp.max(dx * dx, dist_eps)
        w = wp.pow(dist2, -0.5 * weight_power)

        m00 = m00 + w * dx * dx
        b0 = b0 + w * dx * dphi

    # Solve the 1x1 normal equation with a numerical floor.
    gx = float(0.0)
    solved_i = int(0)
    if m00 > dist_eps:
        gx = b0 / m00
        solved_i = 1

    gradients[i, 0] = gx
    solved[i] = solved_i


@wp.func
def _point_2d(points: wp.array2d(dtype=wp.float32), i: int) -> wp.vec2:
    """Coordinates of entity ``i``."""
    return wp.vec2(points[i, 0], points[i, 1])


@wp.func
def _point_3d(points: wp.array2d(dtype=wp.float32), i: int) -> wp.vec3:
    """Coordinates of entity ``i``."""
    return wp.vec3(points[i, 0], points[i, 1], points[i, 2])


# Each entity's weighted least-squares system is solved by a QR factorization
# that takes one neighbor row at a time with Givens rotations. Unlike the normal
# equations, it does not square the condition number, so float32 keeps cells
# that are thousands of times thinner than they are wide. R^T R equals the
# normal matrix A^T W A + reg_eps I.
@wp.func
def _givens_2d(r: wp.mat22, z: wp.vec2, x: wp.vec2, y: float):
    """Rotate row ``x``, with right-hand side ``y``, into the triangle ``r``."""
    for k in range(2):
        a = r[k, k]
        b = x[k]
        h = wp.sqrt(a * a + b * b)
        if h > 0.0:
            c = a / h
            s = b / h
            for j in range(k, 2):
                rkj = r[k, j]
                r[k, j] = c * rkj + s * x[j]
                x[j] = c * x[j] - s * rkj
            zk = z[k]
            z[k] = c * zk + s * y
            y = c * y - s * zk
    return r, z


@wp.func
def _givens_3d(r: wp.mat33, z: wp.vec3, x: wp.vec3, y: float):
    """Rotate row ``x``, with right-hand side ``y``, into the triangle ``r``."""
    for k in range(3):
        a = r[k, k]
        b = x[k]
        h = wp.sqrt(a * a + b * b)
        if h > 0.0:
            c = a / h
            s = b / h
            for j in range(k, 3):
                rkj = r[k, j]
                r[k, j] = c * rkj + s * x[j]
                x[j] = c * x[j] - s * rkj
            zk = z[k]
            z[k] = c * zk + s * y
            y = c * y - s * zk
    return r, z


@wp.func
def _qr_2d(
    points: wp.array2d(dtype=wp.float32),
    values: wp.array(dtype=wp.float32),
    use_values: bool,
    indices: wp.array(dtype=wp.int32),
    start: int,
    end: int,
    i: int,
    weight_power: float,
    reg_eps: float,
    dist_eps: float,
):
    """Triangle ``r`` and rotated right-hand side ``z`` of entity ``i``."""
    root = wp.sqrt(reg_eps)
    r = wp.mat22(root, 0.0, 0.0, root)
    z = wp.vec2()
    center = _point_2d(points, i)
    for p in range(start, end):
        n = indices[p]
        d = _point_2d(points, n) - center
        root_w = wp.pow(wp.max(wp.dot(d, d), dist_eps), -0.25 * weight_power)
        y = float(0.0)
        if use_values:
            y = root_w * (values[n] - values[i])
        r, z = _givens_2d(r, z, root_w * d, y)
    return r, z


@wp.func
def _qr_3d(
    points: wp.array2d(dtype=wp.float32),
    values: wp.array(dtype=wp.float32),
    use_values: bool,
    indices: wp.array(dtype=wp.int32),
    start: int,
    end: int,
    i: int,
    weight_power: float,
    reg_eps: float,
    dist_eps: float,
):
    """Triangle ``r`` and rotated right-hand side ``z`` of entity ``i``."""
    root = wp.sqrt(reg_eps)
    r = wp.mat33(root, 0.0, 0.0, 0.0, root, 0.0, 0.0, 0.0, root)
    z = wp.vec3()
    center = _point_3d(points, i)
    for p in range(start, end):
        n = indices[p]
        d = _point_3d(points, n) - center
        root_w = wp.pow(wp.max(wp.dot(d, d), dist_eps), -0.25 * weight_power)
        y = float(0.0)
        if use_values:
            y = root_w * (values[n] - values[i])
        r, z = _givens_3d(r, z, root_w * d, y)
    return r, z


# A regularized system counts as singular by the determinant test of its normal
# matrix, which rejects the singular systems the regularization leaves (one
# neighbor, or collinear neighbors). An unregularized one comes from neighbors
# that span the space: as in PyTorch's least squares, it counts as singular
# where a diagonal entry of R falls below float32 precision times the number
# of rows, relative to the largest. Degenerate cells can still leave a vertex
# with neighbors that do not span the space; its system is then factored again
# with the regularization, which gives about PyTorch's minimum-norm solution.
@wp.func
def _solvable_2d(r: wp.mat22, rows: int, reg_eps: float, dist_eps: float) -> bool:
    """Whether the system with triangle ``r`` counts as solvable."""
    if reg_eps > 0.0:
        root_det = r[0, 0] * r[1, 1]
        m11 = r[0, 1] * r[0, 1] + r[1, 1] * r[1, 1]
        return root_det * root_det > 1.0e-6 * (r[0, 0] * r[0, 0] * m11 + dist_eps)
    tolerance = 1.1920929e-07 * float(wp.max(rows, 2))
    return wp.min(r[0, 0], r[1, 1]) > tolerance * wp.max(r[0, 0], r[1, 1])


@wp.func
def _solvable_3d(r: wp.mat33, rows: int, reg_eps: float, dist_eps: float) -> bool:
    """Whether the system with triangle ``r`` counts as solvable."""
    if reg_eps > 0.0:
        root_det = r[0, 0] * r[1, 1] * r[2, 2]
        trace = float(0.0)
        for k in range(3):
            for j in range(k, 3):
                trace = trace + r[k, j] * r[k, j]
        return root_det * root_det > 1.0e-8 * (trace * trace * trace + dist_eps)
    tolerance = 1.1920929e-07 * float(wp.max(rows, 3))
    smallest = wp.min(r[0, 0], wp.min(r[1, 1], r[2, 2]))
    largest = wp.max(r[0, 0], wp.max(r[1, 1], r[2, 2]))
    return smallest > tolerance * largest


@wp.func
def _factor_2d(
    points: wp.array2d(dtype=wp.float32),
    values: wp.array(dtype=wp.float32),
    use_values: bool,
    indices: wp.array(dtype=wp.int32),
    start: int,
    end: int,
    i: int,
    weight_power: float,
    reg_eps: float,
    dist_eps: float,
):
    """``r``, ``z``, and whether the system of entity ``i`` counts as solvable."""
    r, z = _qr_2d(
        points,
        values,
        use_values,
        indices,
        start,
        end,
        i,
        weight_power,
        reg_eps,
        dist_eps,
    )
    solvable = _solvable_2d(r, end - start, reg_eps, dist_eps)
    if not solvable and reg_eps == 0.0:
        r, z = _qr_2d(
            points,
            values,
            use_values,
            indices,
            start,
            end,
            i,
            weight_power,
            _REG_EPS,
            dist_eps,
        )
        solvable = _solvable_2d(r, end - start, _REG_EPS, dist_eps)
    return r, z, solvable


@wp.func
def _factor_3d(
    points: wp.array2d(dtype=wp.float32),
    values: wp.array(dtype=wp.float32),
    use_values: bool,
    indices: wp.array(dtype=wp.int32),
    start: int,
    end: int,
    i: int,
    weight_power: float,
    reg_eps: float,
    dist_eps: float,
):
    """``r``, ``z``, and whether the system of entity ``i`` counts as solvable."""
    r, z = _qr_3d(
        points,
        values,
        use_values,
        indices,
        start,
        end,
        i,
        weight_power,
        reg_eps,
        dist_eps,
    )
    solvable = _solvable_3d(r, end - start, reg_eps, dist_eps)
    if not solvable and reg_eps == 0.0:
        r, z = _qr_3d(
            points,
            values,
            use_values,
            indices,
            start,
            end,
            i,
            weight_power,
            _REG_EPS,
            dist_eps,
        )
        solvable = _solvable_3d(r, end - start, _REG_EPS, dist_eps)
    return r, z, solvable


@wp.func
def _back_2d(r: wp.mat22, z: wp.vec2) -> wp.vec2:
    """``g`` with ``r g = z``."""
    g1 = z[1] / r[1, 1]
    return wp.vec2((z[0] - r[0, 1] * g1) / r[0, 0], g1)


@wp.func
def _back_3d(r: wp.mat33, z: wp.vec3) -> wp.vec3:
    """``g`` with ``r g = z``."""
    g2 = z[2] / r[2, 2]
    g1 = (z[1] - r[1, 2] * g2) / r[1, 1]
    return wp.vec3((z[0] - r[0, 1] * g1 - r[0, 2] * g2) / r[0, 0], g1, g2)


@wp.func
def _normal_solve_2d(r: wp.mat22, v: wp.vec2) -> wp.vec2:
    """``(r^T r)^-1 v``, by substitution forward through ``r^T``, then back."""
    u0 = v[0] / r[0, 0]
    return _back_2d(r, wp.vec2(u0, (v[1] - r[0, 1] * u0) / r[1, 1]))


@wp.func
def _normal_solve_3d(r: wp.mat33, v: wp.vec3) -> wp.vec3:
    """``(r^T r)^-1 v``, by substitution forward through ``r^T``, then back."""
    u0 = v[0] / r[0, 0]
    u1 = (v[1] - r[0, 1] * u0) / r[1, 1]
    u2 = (v[2] - r[0, 2] * u0 - r[1, 2] * u1) / r[2, 2]
    return _back_3d(r, wp.vec3(u0, u1, u2))


@wp.kernel
def _mesh_lsq_gradient_2d_kernel(
    points: wp.array2d(dtype=wp.float32),
    values: wp.array(dtype=wp.float32),
    offsets: wp.array(dtype=wp.int32),
    indices: wp.array(dtype=wp.int32),
    weight_power: float,
    min_neighbors: int,
    reg_eps: float,
    dist_eps: float,
    gradients: wp.array2d(dtype=wp.float32),
    solved: wp.array(dtype=wp.int32),
):
    i = wp.tid()

    # Read the CSR neighbor segment for this entity.
    start = offsets[i]
    end = offsets[i + 1]
    if end - start < min_neighbors:
        gradients[i, 0] = 0.0
        gradients[i, 1] = 0.0
        solved[i] = 0
        return

    r, z, solvable = _factor_2d(
        points, values, True, indices, start, end, i, weight_power, reg_eps, dist_eps
    )
    g = wp.vec2()
    solved_i = int(0)
    if solvable:
        g = _back_2d(r, z)
        solved_i = 1

    gradients[i, 0] = g[0]
    gradients[i, 1] = g[1]
    solved[i] = solved_i


@wp.kernel
def _mesh_lsq_gradient_3d_kernel(
    points: wp.array2d(dtype=wp.float32),
    values: wp.array(dtype=wp.float32),
    offsets: wp.array(dtype=wp.int32),
    indices: wp.array(dtype=wp.int32),
    weight_power: float,
    min_neighbors: int,
    reg_eps: float,
    dist_eps: float,
    gradients: wp.array2d(dtype=wp.float32),
    solved: wp.array(dtype=wp.int32),
):
    i = wp.tid()

    # Read the CSR neighbor segment for this entity.
    start = offsets[i]
    end = offsets[i + 1]
    if end - start < min_neighbors:
        gradients[i, 0] = 0.0
        gradients[i, 1] = 0.0
        gradients[i, 2] = 0.0
        solved[i] = 0
        return

    r, z, solvable = _factor_3d(
        points, values, True, indices, start, end, i, weight_power, reg_eps, dist_eps
    )
    g = wp.vec3()
    solved_i = int(0)
    if solvable:
        g = _back_3d(r, z)
        solved_i = 1

    gradients[i, 0] = g[0]
    gradients[i, 1] = g[1]
    gradients[i, 2] = g[2]
    solved[i] = solved_i


@wp.kernel
def _mesh_lsq_gradient_1d_backward_kernel(
    points: wp.array2d(dtype=wp.float32),
    offsets: wp.array(dtype=wp.int32),
    indices: wp.array(dtype=wp.int32),
    grad_output: wp.array2d(dtype=wp.float32),
    weight_power: float,
    min_neighbors: int,
    reg_eps: float,
    dist_eps: float,
    grad_values: wp.array(dtype=wp.float32),
):
    i = wp.tid()

    start = offsets[i]
    end = offsets[i + 1]
    count = end - start
    if count < min_neighbors:
        return

    px = points[i, 0]
    m00 = float(reg_eps)
    for p in range(start, end):
        n = indices[p]
        dx = points[n, 0] - px
        dist2 = wp.max(dx * dx, dist_eps)
        w = wp.pow(dist2, -0.5 * weight_power)
        m00 = m00 + w * dx * dx

    p0 = float(0.0)
    if m00 > dist_eps:
        p0 = grad_output[i, 0] / m00

    self_contrib = float(0.0)
    for p in range(start, end):
        n = indices[p]
        dx = points[n, 0] - px
        dist2 = wp.max(dx * dx, dist_eps)
        w = wp.pow(dist2, -0.5 * weight_power)
        c = w * p0 * dx
        wp.atomic_add(grad_values, n, c)
        self_contrib = self_contrib - c

    wp.atomic_add(grad_values, i, self_contrib)


@wp.kernel
def _mesh_lsq_gradient_2d_backward_kernel(
    points: wp.array2d(dtype=wp.float32),
    offsets: wp.array(dtype=wp.int32),
    indices: wp.array(dtype=wp.int32),
    grad_output: wp.array2d(dtype=wp.float32),
    weight_power: float,
    min_neighbors: int,
    reg_eps: float,
    dist_eps: float,
    grad_values: wp.array(dtype=wp.float32),
):
    i = wp.tid()

    start = offsets[i]
    end = offsets[i + 1]
    if end - start < min_neighbors:
        return

    # The values of the factorization's right-hand side are not needed here.
    r, _, solvable = _factor_2d(
        points,
        grad_values,
        False,
        indices,
        start,
        end,
        i,
        weight_power,
        reg_eps,
        dist_eps,
    )
    if not solvable:
        return
    q = _normal_solve_2d(r, wp.vec2(grad_output[i, 0], grad_output[i, 1]))

    center = _point_2d(points, i)
    self_contrib = float(0.0)
    for p in range(start, end):
        n = indices[p]
        d = _point_2d(points, n) - center
        w = wp.pow(wp.max(wp.dot(d, d), dist_eps), -0.5 * weight_power)
        c = w * wp.dot(q, d)
        wp.atomic_add(grad_values, n, c)
        self_contrib = self_contrib - c

    wp.atomic_add(grad_values, i, self_contrib)


@wp.kernel
def _mesh_lsq_gradient_3d_backward_kernel(
    points: wp.array2d(dtype=wp.float32),
    offsets: wp.array(dtype=wp.int32),
    indices: wp.array(dtype=wp.int32),
    grad_output: wp.array2d(dtype=wp.float32),
    weight_power: float,
    min_neighbors: int,
    reg_eps: float,
    dist_eps: float,
    grad_values: wp.array(dtype=wp.float32),
):
    i = wp.tid()

    start = offsets[i]
    end = offsets[i + 1]
    if end - start < min_neighbors:
        return

    # The values of the factorization's right-hand side are not needed here.
    r, _, solvable = _factor_3d(
        points,
        grad_values,
        False,
        indices,
        start,
        end,
        i,
        weight_power,
        reg_eps,
        dist_eps,
    )
    if not solvable:
        return
    q = _normal_solve_3d(
        r, wp.vec3(grad_output[i, 0], grad_output[i, 1], grad_output[i, 2])
    )

    center = _point_3d(points, i)
    self_contrib = float(0.0)
    for p in range(start, end):
        n = indices[p]
        d = _point_3d(points, n) - center
        w = wp.pow(wp.max(wp.dot(d, d), dist_eps), -0.5 * weight_power)
        c = w * wp.dot(q, d)
        wp.atomic_add(grad_values, n, c)
        self_contrib = self_contrib - c

    wp.atomic_add(grad_values, i, self_contrib)


@wp.kernel
def _mesh_lsq_gradient_1d_backward_points_kernel(
    points: wp.array2d(dtype=wp.float32),
    values: wp.array(dtype=wp.float32),
    offsets: wp.array(dtype=wp.int32),
    indices: wp.array(dtype=wp.int32),
    grad_output: wp.array2d(dtype=wp.float32),
    weight_power: float,
    min_neighbors: int,
    reg_eps: float,
    dist_eps: float,
    grad_points: wp.array2d(dtype=wp.float32),
):
    i = wp.tid()

    start = offsets[i]
    end = offsets[i + 1]
    count = end - start
    if count < min_neighbors:
        return

    px = points[i, 0]
    vi = values[i]
    m00 = float(reg_eps)
    b0 = float(0.0)
    for p in range(start, end):
        n = indices[p]
        dx = points[n, 0] - px
        s = values[n] - vi
        dist2 = wp.max(dx * dx, dist_eps)
        w = wp.pow(dist2, -0.5 * weight_power)
        m00 = m00 + w * dx * dx
        b0 = b0 + w * dx * s

    if m00 <= dist_eps:
        return

    g0 = b0 / m00
    p0 = grad_output[i, 0] / m00
    for p in range(start, end):
        n = indices[p]
        dx = points[n, 0] - px
        s = values[n] - vi
        dist2 = wp.max(dx * dx, dist_eps)
        w = wp.pow(dist2, -0.5 * weight_power)

        dg = dx * g0
        pd = p0 * dx
        common = s - dg
        beta = float(0.0)  # the weight is constant below the distance floor
        if dx * dx > dist_eps:
            beta = (-weight_power / dist2) * w
        grad_d = beta * pd * common * dx + w * (common * p0 - pd * g0)

        wp.atomic_add(grad_points, n, 0, grad_d)
        wp.atomic_add(grad_points, i, 0, -grad_d)


@wp.kernel
def _mesh_lsq_gradient_2d_backward_points_kernel(
    points: wp.array2d(dtype=wp.float32),
    values: wp.array(dtype=wp.float32),
    offsets: wp.array(dtype=wp.int32),
    indices: wp.array(dtype=wp.int32),
    grad_output: wp.array2d(dtype=wp.float32),
    weight_power: float,
    min_neighbors: int,
    reg_eps: float,
    dist_eps: float,
    grad_points: wp.array2d(dtype=wp.float32),
):
    i = wp.tid()

    start = offsets[i]
    end = offsets[i + 1]
    if end - start < min_neighbors:
        return

    r, z, solvable = _factor_2d(
        points, values, True, indices, start, end, i, weight_power, reg_eps, dist_eps
    )
    if not solvable:
        return
    g = _back_2d(r, z)
    q = _normal_solve_2d(r, wp.vec2(grad_output[i, 0], grad_output[i, 1]))

    center = _point_2d(points, i)
    vi = values[i]
    for p in range(start, end):
        n = indices[p]
        d = _point_2d(points, n) - center
        dist2 = wp.max(wp.dot(d, d), dist_eps)
        w = wp.pow(dist2, -0.5 * weight_power)
        pd = wp.dot(d, q)
        common = values[n] - vi - wp.dot(d, g)
        beta = float(0.0)  # the weight is constant below the distance floor
        if wp.dot(d, d) > dist_eps:
            beta = (-weight_power / dist2) * w
        grad_d = (beta * pd * common) * d + w * (common * q - pd * g)

        wp.atomic_add(grad_points, n, 0, grad_d[0])
        wp.atomic_add(grad_points, n, 1, grad_d[1])
        wp.atomic_add(grad_points, i, 0, -grad_d[0])
        wp.atomic_add(grad_points, i, 1, -grad_d[1])


@wp.kernel
def _mesh_lsq_gradient_3d_backward_points_kernel(
    points: wp.array2d(dtype=wp.float32),
    values: wp.array(dtype=wp.float32),
    offsets: wp.array(dtype=wp.int32),
    indices: wp.array(dtype=wp.int32),
    grad_output: wp.array2d(dtype=wp.float32),
    weight_power: float,
    min_neighbors: int,
    reg_eps: float,
    dist_eps: float,
    grad_points: wp.array2d(dtype=wp.float32),
):
    i = wp.tid()

    start = offsets[i]
    end = offsets[i + 1]
    if end - start < min_neighbors:
        return

    r, z, solvable = _factor_3d(
        points, values, True, indices, start, end, i, weight_power, reg_eps, dist_eps
    )
    if not solvable:
        return
    g = _back_3d(r, z)
    q = _normal_solve_3d(
        r, wp.vec3(grad_output[i, 0], grad_output[i, 1], grad_output[i, 2])
    )

    center = _point_3d(points, i)
    vi = values[i]
    for p in range(start, end):
        n = indices[p]
        d = _point_3d(points, n) - center
        dist2 = wp.max(wp.dot(d, d), dist_eps)
        w = wp.pow(dist2, -0.5 * weight_power)
        pd = wp.dot(d, q)
        common = values[n] - vi - wp.dot(d, g)
        beta = float(0.0)  # the weight is constant below the distance floor
        if wp.dot(d, d) > dist_eps:
            beta = (-weight_power / dist2) * w
        grad_d = (beta * pd * common) * d + w * (common * q - pd * g)

        wp.atomic_add(grad_points, n, 0, grad_d[0])
        wp.atomic_add(grad_points, n, 1, grad_d[1])
        wp.atomic_add(grad_points, n, 2, grad_d[2])
        wp.atomic_add(grad_points, i, 0, -grad_d[0])
        wp.atomic_add(grad_points, i, 1, -grad_d[1])
        wp.atomic_add(grad_points, i, 2, -grad_d[2])


def _launch_forward(
    *,
    points_fp32: torch.Tensor,
    values_flat_fp32: torch.Tensor,
    offsets_i32: torch.Tensor,
    indices_i32: torch.Tensor,
    weight_power: float,
    min_neighbors: int,
    reg_eps: float,
    dist_eps: float,
    grads_components: torch.Tensor,
    solved: torch.Tensor,
    wp_device,
    wp_stream,
) -> None:
    ### Launch one LSQ forward kernel per value component.
    n_dims = points_fp32.shape[1]
    n_entities = points_fp32.shape[0]
    n_components = values_flat_fp32.shape[1]
    kernel = (
        _mesh_lsq_gradient_1d_kernel
        if n_dims == 1
        else _mesh_lsq_gradient_2d_kernel
        if n_dims == 2
        else _mesh_lsq_gradient_3d_kernel
    )

    wp_points = wp.from_torch(points_fp32, dtype=wp.float32)
    wp_offsets = wp.from_torch(offsets_i32, dtype=wp.int32)
    wp_indices = wp.from_torch(indices_i32, dtype=wp.int32)
    wp_solved = wp.from_torch(solved, dtype=wp.int32)

    with FunctionSpec.warp_stream_scope(wp_stream):
        for comp in range(n_components):
            wp.launch(
                kernel=kernel,
                dim=n_entities,
                inputs=[
                    wp_points,
                    wp.from_torch(
                        values_flat_fp32[:, comp].contiguous(), dtype=wp.float32
                    ),
                    wp_offsets,
                    wp_indices,
                    float(weight_power),
                    int(min_neighbors),
                    float(reg_eps),
                    float(dist_eps),
                    wp.from_torch(grads_components[comp], dtype=wp.float32),
                    wp_solved,
                ],
                device=wp_device,
                stream=wp_stream,
            )


def _launch_backward(
    *,
    points_fp32: torch.Tensor,
    offsets_i32: torch.Tensor,
    indices_i32: torch.Tensor,
    grad_output_components_fp32: torch.Tensor,
    weight_power: float,
    min_neighbors: int,
    reg_eps: float,
    dist_eps: float,
    grad_values_flat: torch.Tensor,
    wp_device,
    wp_stream,
) -> None:
    ### Launch one LSQ backward kernel per value component.
    n_dims = points_fp32.shape[1]
    n_entities = points_fp32.shape[0]
    n_components = grad_output_components_fp32.shape[0]
    kernel = (
        _mesh_lsq_gradient_1d_backward_kernel
        if n_dims == 1
        else _mesh_lsq_gradient_2d_backward_kernel
        if n_dims == 2
        else _mesh_lsq_gradient_3d_backward_kernel
    )

    wp_points = wp.from_torch(points_fp32, dtype=wp.float32)
    wp_offsets = wp.from_torch(offsets_i32, dtype=wp.int32)
    wp_indices = wp.from_torch(indices_i32, dtype=wp.int32)

    with FunctionSpec.warp_stream_scope(wp_stream):
        for comp in range(n_components):
            comp_grad_values = torch.zeros(
                (n_entities,),
                device=grad_values_flat.device,
                dtype=torch.float32,
            )
            wp.launch(
                kernel=kernel,
                dim=n_entities,
                inputs=[
                    wp_points,
                    wp_offsets,
                    wp_indices,
                    wp.from_torch(grad_output_components_fp32[comp], dtype=wp.float32),
                    float(weight_power),
                    int(min_neighbors),
                    float(reg_eps),
                    float(dist_eps),
                    wp.from_torch(comp_grad_values, dtype=wp.float32),
                ],
                device=wp_device,
                stream=wp_stream,
            )
            grad_values_flat[:, comp] = comp_grad_values


def _launch_backward_points(
    *,
    points_fp32: torch.Tensor,
    values_flat_fp32: torch.Tensor,
    offsets_i32: torch.Tensor,
    indices_i32: torch.Tensor,
    grad_output_components_fp32: torch.Tensor,
    weight_power: float,
    min_neighbors: int,
    reg_eps: float,
    dist_eps: float,
    grad_points: torch.Tensor,
    wp_device,
    wp_stream,
) -> None:
    """Launch explicit LSQ backward kernels for point gradients."""
    n_dims = points_fp32.shape[1]
    n_entities = points_fp32.shape[0]
    n_components = values_flat_fp32.shape[1]
    kernel = (
        _mesh_lsq_gradient_1d_backward_points_kernel
        if n_dims == 1
        else _mesh_lsq_gradient_2d_backward_points_kernel
        if n_dims == 2
        else _mesh_lsq_gradient_3d_backward_points_kernel
    )
    values_components = values_flat_fp32.transpose(0, 1).contiguous()

    with FunctionSpec.warp_stream_scope(wp_stream):
        wp_points = wp.from_torch(points_fp32, dtype=wp.float32)
        wp_offsets = wp.from_torch(offsets_i32, dtype=wp.int32)
        wp_indices = wp.from_torch(indices_i32, dtype=wp.int32)
        wp_grad_points = wp.from_torch(grad_points, dtype=wp.float32)
        for comp in range(n_components):
            wp.launch(
                kernel=kernel,
                dim=n_entities,
                inputs=[
                    wp_points,
                    wp.from_torch(values_components[comp], dtype=wp.float32),
                    wp_offsets,
                    wp_indices,
                    wp.from_torch(grad_output_components_fp32[comp], dtype=wp.float32),
                    float(weight_power),
                    int(min_neighbors),
                    float(reg_eps),
                    float(dist_eps),
                    wp_grad_points,
                ],
                device=wp_device,
                stream=wp_stream,
            )


@torch.library.custom_op("physicsnemo::mesh_lsq_gradient_warp_impl", mutates_args=())
def mesh_lsq_gradient_impl(
    points: torch.Tensor,
    values: torch.Tensor,
    neighbor_offsets: torch.Tensor,
    neighbor_indices: torch.Tensor,
    weight_power: float,
    min_neighbors: int,
    reg_eps: float,
    dist_eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute weighted LSQ gradients with Warp kernels.

    Also returns an ``int32`` mask of the entities whose system was solved; the
    others, with too few neighbors or a singular system, get zero gradients.
    """
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
        dtype=torch.int32, device=points.device
    ).contiguous()
    indices_i32 = neighbor_indices.to(
        dtype=torch.int32, device=points.device
    ).contiguous()

    n_entities = points_fp32.shape[0]
    n_dims = points_fp32.shape[1]
    value_shape = values.shape[1:]
    values_flat = values_fp32.reshape(n_entities, math.prod(value_shape))
    n_components = values_flat.shape[1]

    ### Store component-wise output as (C, N, dims) for contiguous warp writes.
    grads_components = torch.empty(
        (n_components, n_entities, n_dims),
        dtype=torch.float32,
        device=points.device,
    )
    solved = torch.zeros(n_entities, dtype=torch.int32, device=points.device)

    wp_device, wp_stream = FunctionSpec.warp_launch_context(points_fp32)
    _launch_forward(
        points_fp32=points_fp32,
        values_flat_fp32=values_flat,
        offsets_i32=offsets_i32,
        indices_i32=indices_i32,
        weight_power=float(weight_power),
        min_neighbors=int(min_neighbors),
        reg_eps=float(reg_eps),
        dist_eps=float(dist_eps),
        grads_components=grads_components,
        solved=solved,
        wp_device=wp_device,
        wp_stream=wp_stream,
    )

    # Contiguous, as the fake implementation reports; the permuted view is not.
    output = grads_components.permute(1, 2, 0).reshape(n_entities, n_dims, *value_shape)
    output = output.contiguous()
    if output.dtype != values.dtype:
        output = output.to(dtype=values.dtype)
    return output, solved


@mesh_lsq_gradient_impl.register_fake
def _mesh_lsq_gradient_impl_fake(
    points: torch.Tensor,
    values: torch.Tensor,
    neighbor_offsets: torch.Tensor,
    neighbor_indices: torch.Tensor,
    weight_power: float,
    min_neighbors: int,
    reg_eps: float,
    dist_eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fake tensor propagation for LSQ custom op."""
    _ = (
        neighbor_offsets,
        neighbor_indices,
        weight_power,
        min_neighbors,
        reg_eps,
        dist_eps,
    )
    gradients = torch.empty(
        (values.shape[0], points.shape[1], *values.shape[1:]),
        device=values.device,
        dtype=values.dtype,
    )
    solved = torch.empty(values.shape[0], device=values.device, dtype=torch.int32)
    return gradients, solved


def setup_mesh_lsq_gradient_context(
    ctx: torch.autograd.function.FunctionCtx, inputs: tuple, output: tuple
) -> None:
    """Store backward context for LSQ custom-op autograd."""
    (
        points,
        values,
        neighbor_offsets,
        neighbor_indices,
        weight_power,
        min_neighbors,
        reg_eps,
        dist_eps,
    ) = inputs
    _, solved = output
    ctx.save_for_backward(points, values, neighbor_offsets, neighbor_indices, solved)
    ctx.weight_power = float(weight_power)
    ctx.min_neighbors = int(min_neighbors)
    ctx.reg_eps = float(reg_eps)
    ctx.dist_eps = float(dist_eps)


@torch.library.custom_op(
    "physicsnemo::mesh_lsq_gradient_warp_backward_impl", mutates_args=()
)
def mesh_lsq_gradient_backward_impl(
    points: torch.Tensor,
    values: torch.Tensor,
    neighbor_offsets: torch.Tensor,
    neighbor_indices: torch.Tensor,
    grad_output: torch.Tensor,
    weight_power: float,
    min_neighbors: int,
    reg_eps: float,
    dist_eps: float,
    needs_points: bool,
    needs_values: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gradients of the LSQ op for ``points`` and ``values``, by Warp kernels.

    An op of its own, so that ``torch.compile`` can trace the backward. A
    gradient that is not needed comes back empty.
    """
    values_shape = values.shape
    n_entities = values_shape[0]
    points_fp32 = points.to(dtype=torch.float32).contiguous()
    values_flat_fp32 = (
        values.to(dtype=torch.float32)
        .reshape(n_entities, math.prod(values_shape[1:]))
        .contiguous()
    )
    offsets_i32 = neighbor_offsets.to(dtype=torch.int32, device=points.device)
    indices_i32 = neighbor_indices.to(dtype=torch.int32, device=points.device)
    offsets_i32, indices_i32 = offsets_i32.contiguous(), indices_i32.contiguous()
    grad_output_fp32 = grad_output.to(dtype=torch.float32).contiguous()
    value_shape = values_shape[1:]
    n_components = math.prod(value_shape)

    grad_output_components = grad_output_fp32.reshape(
        n_entities, grad_output_fp32.shape[1], n_components
    )
    grad_output_components = grad_output_components.permute(2, 0, 1).contiguous()
    grad_points = None
    grad_values_flat = None
    wp_device, wp_stream = FunctionSpec.warp_launch_context(grad_output_fp32)
    if needs_points:
        grad_points_fp32 = torch.zeros_like(points_fp32, dtype=torch.float32)
        _launch_backward_points(
            points_fp32=points_fp32,
            values_flat_fp32=values_flat_fp32,
            offsets_i32=offsets_i32,
            indices_i32=indices_i32,
            grad_output_components_fp32=grad_output_components,
            weight_power=weight_power,
            min_neighbors=min_neighbors,
            reg_eps=reg_eps,
            dist_eps=dist_eps,
            grad_points=grad_points_fp32,
            wp_device=wp_device,
            wp_stream=wp_stream,
        )
        grad_points = grad_points_fp32
        if grad_points.dtype != points.dtype:
            grad_points = grad_points.to(dtype=points.dtype)
    if needs_values:
        grad_values_flat = torch.empty(
            (n_entities, n_components),
            device=grad_output.device,
            dtype=torch.float32,
        )
        _launch_backward(
            points_fp32=points_fp32,
            offsets_i32=offsets_i32,
            indices_i32=indices_i32,
            grad_output_components_fp32=grad_output_components,
            weight_power=weight_power,
            min_neighbors=min_neighbors,
            reg_eps=reg_eps,
            dist_eps=dist_eps,
            grad_values_flat=grad_values_flat,
            wp_device=wp_device,
            wp_stream=wp_stream,
        )

    grad_values = None
    if needs_values and grad_values_flat is not None:
        grad_values = grad_values_flat.reshape(values_shape)
        if grad_values.dtype != values.dtype:
            grad_values = grad_values.to(dtype=values.dtype)
    return (
        grad_points if needs_points else points.new_empty(0),
        grad_values if needs_values else values.new_empty(0),
    )


@mesh_lsq_gradient_backward_impl.register_fake
def _mesh_lsq_gradient_backward_impl_fake(
    points: torch.Tensor,
    values: torch.Tensor,
    neighbor_offsets: torch.Tensor,
    neighbor_indices: torch.Tensor,
    grad_output: torch.Tensor,
    weight_power: float,
    min_neighbors: int,
    reg_eps: float,
    dist_eps: float,
    needs_points: bool,
    needs_values: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fake tensor propagation for the LSQ backward custom op."""
    _ = (neighbor_offsets, neighbor_indices, grad_output, weight_power)
    _ = (min_neighbors, reg_eps, dist_eps)
    return (
        torch.empty(points.shape, dtype=points.dtype, device=points.device)
        if needs_points
        else points.new_empty(0),
        torch.empty(values.shape, dtype=values.dtype, device=values.device)
        if needs_values
        else values.new_empty(0),
    )


def backward_mesh_lsq_gradient(
    ctx: torch.autograd.function.FunctionCtx,
    grad_output: torch.Tensor,
    grad_solved: torch.Tensor | None,
) -> tuple[
    torch.Tensor | None, torch.Tensor | None, None, None, None, None, None, None
]:
    """Backward pass for LSQ custom op.

    The Warp backward kernels are not differentiable. For higher-order autograd
    (``create_graph=True``) the gradients come from differentiating the
    equivalent PyTorch solve instead, without the entities that the Warp
    forward left unsolved, as the Warp kernels do.
    """
    _ = grad_solved  # the solved mask is not differentiable
    needs_points = ctx.needs_input_grad[0]
    needs_values = ctx.needs_input_grad[1]
    if grad_output is None or (not needs_points and not needs_values):
        return None, None, None, None, None, None, None, None

    points, values, neighbor_offsets, neighbor_indices, solved = ctx.saved_tensors
    if torch.is_grad_enabled():
        from ._torch_impl import mesh_lsq_gradient_torch

        # Fresh views, so that the gradient for ``points`` leaves out the path
        # through ``values`` when ``values`` was computed from ``points``.
        points, values = points.view_as(points), values.view_as(values)
        inputs = [
            t
            for t, needed in ((points, needs_points), (values, needs_values))
            if needed
        ]
        output = mesh_lsq_gradient_torch(
            points=points,
            values=values,
            neighbor_offsets=neighbor_offsets,
            neighbor_indices=neighbor_indices,
            weight_power=ctx.weight_power,
            min_neighbors=ctx.min_neighbors,
            safe_epsilon=ctx.dist_eps,
        )
        mask = solved.to(grad_output.dtype).view(-1, *[1] * (grad_output.dim() - 1))
        grads = iter(
            torch.autograd.grad(output, inputs, grad_output * mask, create_graph=True)
        )
        grad_points = next(grads) if needs_points else None
        grad_values = next(grads) if needs_values else None
        return grad_points, grad_values, None, None, None, None, None, None

    grad_points, grad_values = mesh_lsq_gradient_backward_impl(
        points,
        values,
        neighbor_offsets,
        neighbor_indices,
        grad_output,
        ctx.weight_power,
        ctx.min_neighbors,
        ctx.reg_eps,
        ctx.dist_eps,
        needs_points,
        needs_values,
    )
    return (
        grad_points if needs_points else None,
        grad_values if needs_values else None,
        None,
        None,
        None,
        None,
        None,
        None,
    )


mesh_lsq_gradient_impl.register_autograd(
    backward_mesh_lsq_gradient,
    setup_context=setup_mesh_lsq_gradient_context,
)


def mesh_lsq_gradient_warp(
    points: torch.Tensor,
    values: torch.Tensor,
    neighbor_offsets: torch.Tensor,
    neighbor_indices: torch.Tensor,
    weight_power: float = 2.0,
    min_neighbors: int = 0,
    reg_eps: float = REG_EPS,
    safe_epsilon: float | None = None,
) -> torch.Tensor:
    """Compute weighted LSQ mesh gradients with Warp kernels.

    ``reg_eps`` is added to the diagonal of each normal matrix, which keeps the
    systems of degenerate neighborhoods stable; a system then counts as
    singular by the determinant test of its normal matrix. With
    ``reg_eps=0``, for neighborhoods that span the space, it counts as singular
    by the rank of its QR factor, as in PyTorch, and is solved again with
    ``REG_EPS``.

    Notes
    -----
    Warp kernels compute in ``float32`` internally. Inputs in wider floating
    dtypes are accepted and cast to ``float32`` for compute. Float64 inputs are
    accepted, but derivative accuracy is limited to ``float32`` precision.
    Second derivatives differentiate the unregularized PyTorch solve.
    """
    dist_eps = resolve_safe_epsilon(safe_epsilon=safe_epsilon, dtype=torch.float32)
    gradients, _ = mesh_lsq_gradient_impl(
        points,
        values,
        neighbor_offsets,
        neighbor_indices,
        float(weight_power),
        int(min_neighbors),
        float(reg_eps),
        float(dist_eps),
    )
    return gradients
