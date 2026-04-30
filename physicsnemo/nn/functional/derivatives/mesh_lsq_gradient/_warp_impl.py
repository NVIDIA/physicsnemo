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

import torch
import warp as wp

from physicsnemo.core.function_spec import FunctionSpec

from .utils import resolve_safe_epsilon, validate_inputs

### Warp runtime initialization for custom kernels.
wp.init()
wp.config.quiet = True


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
):
    i = wp.tid()

    # Read the CSR neighbor segment for this entity.
    start = offsets[i]
    end = offsets[i + 1]
    count = end - start
    if count < min_neighbors:
        gradients[i, 0] = 0.0
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

        dist2 = dx * dx + dist_eps
        w = wp.pow(dist2, -0.5 * weight_power)

        m00 = m00 + w * dx * dx
        b0 = b0 + w * dx * dphi

    # Solve the 1x1 normal equation with a numerical floor.
    gx = float(0.0)
    if m00 > dist_eps:
        gx = b0 / m00

    gradients[i, 0] = gx


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
):
    i = wp.tid()

    # Read the CSR neighbor segment for this entity.
    start = offsets[i]
    end = offsets[i + 1]
    count = end - start
    if count < min_neighbors:
        gradients[i, 0] = 0.0
        gradients[i, 1] = 0.0
        return

    # Gather center state and initialize normal-equation accumulators.
    px = points[i, 0]
    py = points[i, 1]
    pval = values[i]

    m00 = float(reg_eps)
    m01 = float(0.0)
    m11 = float(reg_eps)
    b0 = float(0.0)
    b1 = float(0.0)

    # Accumulate A^T W A and A^T W b over neighbors.
    for p in range(start, end):
        n = indices[p]
        dx = points[n, 0] - px
        dy = points[n, 1] - py
        dphi = values[n] - pval

        dist2 = dx * dx + dy * dy + dist_eps
        w = wp.pow(dist2, -0.5 * weight_power)

        m00 = m00 + w * dx * dx
        m01 = m01 + w * dx * dy
        m11 = m11 + w * dy * dy

        b0 = b0 + w * dx * dphi
        b1 = b1 + w * dy * dphi

    # Solve the 2x2 system analytically with determinant-based conditioning.
    det = m00 * m11 - m01 * m01

    gx = float(0.0)
    gy = float(0.0)
    stability_scale = m00 * m11 + dist_eps
    if wp.abs(det) > 1.0e-6 * stability_scale:
        inv00 = m11 / det
        inv01 = -m01 / det
        inv11 = m00 / det
        gx = inv00 * b0 + inv01 * b1
        gy = inv01 * b0 + inv11 * b1

    gradients[i, 0] = gx
    gradients[i, 1] = gy


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
):
    i = wp.tid()

    # Read the CSR neighbor segment for this entity.
    start = offsets[i]
    end = offsets[i + 1]
    count = end - start
    if count < min_neighbors:
        gradients[i, 0] = 0.0
        gradients[i, 1] = 0.0
        gradients[i, 2] = 0.0
        return

    # Gather center state and initialize normal-equation accumulators.
    px = points[i, 0]
    py = points[i, 1]
    pz = points[i, 2]
    pval = values[i]

    m00 = float(reg_eps)
    m01 = float(0.0)
    m02 = float(0.0)
    m11 = float(reg_eps)
    m12 = float(0.0)
    m22 = float(reg_eps)

    b0 = float(0.0)
    b1 = float(0.0)
    b2 = float(0.0)

    # Accumulate A^T W A and A^T W b over neighbors.
    for p in range(start, end):
        n = indices[p]
        dx = points[n, 0] - px
        dy = points[n, 1] - py
        dz = points[n, 2] - pz
        dphi = values[n] - pval

        dist2 = dx * dx + dy * dy + dz * dz + dist_eps
        w = wp.pow(dist2, -0.5 * weight_power)

        m00 = m00 + w * dx * dx
        m01 = m01 + w * dx * dy
        m02 = m02 + w * dx * dz
        m11 = m11 + w * dy * dy
        m12 = m12 + w * dy * dz
        m22 = m22 + w * dz * dz

        b0 = b0 + w * dx * dphi
        b1 = b1 + w * dy * dphi
        b2 = b2 + w * dz * dphi

    # Build cofactors and solve the 3x3 system analytically.
    c00 = m11 * m22 - m12 * m12
    c01 = -(m01 * m22 - m12 * m02)
    c02 = m01 * m12 - m11 * m02
    c11 = m00 * m22 - m02 * m02
    c12 = -(m00 * m12 - m01 * m02)
    c22 = m00 * m11 - m01 * m01

    det = m00 * c00 + m01 * c01 + m02 * c02

    gx = float(0.0)
    gy = float(0.0)
    gz = float(0.0)
    trace = m00 + m11 + m22
    stability_scale = trace * trace * trace + dist_eps
    if wp.abs(det) > 1.0e-8 * stability_scale:
        inv_det = 1.0 / det
        inv00 = c00 * inv_det
        inv01 = c01 * inv_det
        inv02 = c02 * inv_det
        inv11 = c11 * inv_det
        inv12 = c12 * inv_det
        inv22 = c22 * inv_det

        gx = inv00 * b0 + inv01 * b1 + inv02 * b2
        gy = inv01 * b0 + inv11 * b1 + inv12 * b2
        gz = inv02 * b0 + inv12 * b1 + inv22 * b2

    gradients[i, 0] = gx
    gradients[i, 1] = gy
    gradients[i, 2] = gz


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
        dist2 = dx * dx + dist_eps
        w = wp.pow(dist2, -0.5 * weight_power)
        m00 = m00 + w * dx * dx

    p0 = float(0.0)
    if m00 > dist_eps:
        p0 = grad_output[i, 0] / m00

    self_contrib = float(0.0)
    for p in range(start, end):
        n = indices[p]
        dx = points[n, 0] - px
        dist2 = dx * dx + dist_eps
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
    count = end - start
    if count < min_neighbors:
        return

    px = points[i, 0]
    py = points[i, 1]
    m00 = float(reg_eps)
    m01 = float(0.0)
    m11 = float(reg_eps)
    for p in range(start, end):
        n = indices[p]
        dx = points[n, 0] - px
        dy = points[n, 1] - py
        dist2 = dx * dx + dy * dy + dist_eps
        w = wp.pow(dist2, -0.5 * weight_power)
        m00 = m00 + w * dx * dx
        m01 = m01 + w * dx * dy
        m11 = m11 + w * dy * dy

    p0 = float(0.0)
    p1 = float(0.0)
    det = m00 * m11 - m01 * m01
    stability_scale = m00 * m11 + dist_eps
    if wp.abs(det) > 1.0e-6 * stability_scale:
        inv00 = m11 / det
        inv01 = -m01 / det
        inv11 = m00 / det
        go0 = grad_output[i, 0]
        go1 = grad_output[i, 1]
        p0 = inv00 * go0 + inv01 * go1
        p1 = inv01 * go0 + inv11 * go1

    self_contrib = float(0.0)
    for p in range(start, end):
        n = indices[p]
        dx = points[n, 0] - px
        dy = points[n, 1] - py
        dist2 = dx * dx + dy * dy + dist_eps
        w = wp.pow(dist2, -0.5 * weight_power)
        c = w * (p0 * dx + p1 * dy)
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
    count = end - start
    if count < min_neighbors:
        return

    px = points[i, 0]
    py = points[i, 1]
    pz = points[i, 2]
    m00 = float(reg_eps)
    m01 = float(0.0)
    m02 = float(0.0)
    m11 = float(reg_eps)
    m12 = float(0.0)
    m22 = float(reg_eps)
    for p in range(start, end):
        n = indices[p]
        dx = points[n, 0] - px
        dy = points[n, 1] - py
        dz = points[n, 2] - pz
        dist2 = dx * dx + dy * dy + dz * dz + dist_eps
        w = wp.pow(dist2, -0.5 * weight_power)
        m00 = m00 + w * dx * dx
        m01 = m01 + w * dx * dy
        m02 = m02 + w * dx * dz
        m11 = m11 + w * dy * dy
        m12 = m12 + w * dy * dz
        m22 = m22 + w * dz * dz

    c00 = m11 * m22 - m12 * m12
    c01 = -(m01 * m22 - m12 * m02)
    c02 = m01 * m12 - m11 * m02
    c11 = m00 * m22 - m02 * m02
    c12 = -(m00 * m12 - m01 * m02)
    c22 = m00 * m11 - m01 * m01
    det = m00 * c00 + m01 * c01 + m02 * c02

    p0 = float(0.0)
    p1 = float(0.0)
    p2 = float(0.0)
    trace = m00 + m11 + m22
    stability_scale = trace * trace * trace + dist_eps
    if wp.abs(det) > 1.0e-8 * stability_scale:
        inv_det = 1.0 / det
        inv00 = c00 * inv_det
        inv01 = c01 * inv_det
        inv02 = c02 * inv_det
        inv11 = c11 * inv_det
        inv12 = c12 * inv_det
        inv22 = c22 * inv_det
        go0 = grad_output[i, 0]
        go1 = grad_output[i, 1]
        go2 = grad_output[i, 2]
        p0 = inv00 * go0 + inv01 * go1 + inv02 * go2
        p1 = inv01 * go0 + inv11 * go1 + inv12 * go2
        p2 = inv02 * go0 + inv12 * go1 + inv22 * go2

    self_contrib = float(0.0)
    for p in range(start, end):
        n = indices[p]
        dx = points[n, 0] - px
        dy = points[n, 1] - py
        dz = points[n, 2] - pz
        dist2 = dx * dx + dy * dy + dz * dz + dist_eps
        w = wp.pow(dist2, -0.5 * weight_power)
        c = w * (p0 * dx + p1 * dy + p2 * dz)
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
        dist2 = dx * dx + dist_eps
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
        dist2 = dx * dx + dist_eps
        w = wp.pow(dist2, -0.5 * weight_power)

        dg = dx * g0
        pd = p0 * dx
        common = s - dg
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
    count = end - start
    if count < min_neighbors:
        return

    px = points[i, 0]
    py = points[i, 1]
    vi = values[i]
    m00 = float(reg_eps)
    m01 = float(0.0)
    m11 = float(reg_eps)
    b0 = float(0.0)
    b1 = float(0.0)
    for p in range(start, end):
        n = indices[p]
        dx = points[n, 0] - px
        dy = points[n, 1] - py
        s = values[n] - vi
        dist2 = dx * dx + dy * dy + dist_eps
        w = wp.pow(dist2, -0.5 * weight_power)
        m00 = m00 + w * dx * dx
        m01 = m01 + w * dx * dy
        m11 = m11 + w * dy * dy
        b0 = b0 + w * dx * s
        b1 = b1 + w * dy * s

    det = m00 * m11 - m01 * m01
    stability_scale = m00 * m11 + dist_eps
    if wp.abs(det) <= 1.0e-6 * stability_scale:
        return

    inv00 = m11 / det
    inv01 = -m01 / det
    inv11 = m00 / det

    g0 = inv00 * b0 + inv01 * b1
    g1 = inv01 * b0 + inv11 * b1

    go0 = grad_output[i, 0]
    go1 = grad_output[i, 1]
    p0 = inv00 * go0 + inv01 * go1
    p1 = inv01 * go0 + inv11 * go1

    for p in range(start, end):
        n = indices[p]
        dx = points[n, 0] - px
        dy = points[n, 1] - py
        s = values[n] - vi
        dist2 = dx * dx + dy * dy + dist_eps
        w = wp.pow(dist2, -0.5 * weight_power)

        dg = dx * g0 + dy * g1
        pd = dx * p0 + dy * p1
        common = s - dg
        beta = (-weight_power / dist2) * w

        grad_dx = beta * pd * common * dx + w * (common * p0 - pd * g0)
        grad_dy = beta * pd * common * dy + w * (common * p1 - pd * g1)

        wp.atomic_add(grad_points, n, 0, grad_dx)
        wp.atomic_add(grad_points, n, 1, grad_dy)
        wp.atomic_add(grad_points, i, 0, -grad_dx)
        wp.atomic_add(grad_points, i, 1, -grad_dy)


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
    count = end - start
    if count < min_neighbors:
        return

    px = points[i, 0]
    py = points[i, 1]
    pz = points[i, 2]
    vi = values[i]

    m00 = float(reg_eps)
    m01 = float(0.0)
    m02 = float(0.0)
    m11 = float(reg_eps)
    m12 = float(0.0)
    m22 = float(reg_eps)
    b0 = float(0.0)
    b1 = float(0.0)
    b2 = float(0.0)
    for p in range(start, end):
        n = indices[p]
        dx = points[n, 0] - px
        dy = points[n, 1] - py
        dz = points[n, 2] - pz
        s = values[n] - vi
        dist2 = dx * dx + dy * dy + dz * dz + dist_eps
        w = wp.pow(dist2, -0.5 * weight_power)
        m00 = m00 + w * dx * dx
        m01 = m01 + w * dx * dy
        m02 = m02 + w * dx * dz
        m11 = m11 + w * dy * dy
        m12 = m12 + w * dy * dz
        m22 = m22 + w * dz * dz
        b0 = b0 + w * dx * s
        b1 = b1 + w * dy * s
        b2 = b2 + w * dz * s

    c00 = m11 * m22 - m12 * m12
    c01 = -(m01 * m22 - m12 * m02)
    c02 = m01 * m12 - m11 * m02
    c11 = m00 * m22 - m02 * m02
    c12 = -(m00 * m12 - m01 * m02)
    c22 = m00 * m11 - m01 * m01
    det = m00 * c00 + m01 * c01 + m02 * c02

    trace = m00 + m11 + m22
    stability_scale = trace * trace * trace + dist_eps
    if wp.abs(det) <= 1.0e-8 * stability_scale:
        return

    inv_det = 1.0 / det
    inv00 = c00 * inv_det
    inv01 = c01 * inv_det
    inv02 = c02 * inv_det
    inv11 = c11 * inv_det
    inv12 = c12 * inv_det
    inv22 = c22 * inv_det

    g0 = inv00 * b0 + inv01 * b1 + inv02 * b2
    g1 = inv01 * b0 + inv11 * b1 + inv12 * b2
    g2 = inv02 * b0 + inv12 * b1 + inv22 * b2

    go0 = grad_output[i, 0]
    go1 = grad_output[i, 1]
    go2 = grad_output[i, 2]
    p0 = inv00 * go0 + inv01 * go1 + inv02 * go2
    p1 = inv01 * go0 + inv11 * go1 + inv12 * go2
    p2 = inv02 * go0 + inv12 * go1 + inv22 * go2

    for p in range(start, end):
        n = indices[p]
        dx = points[n, 0] - px
        dy = points[n, 1] - py
        dz = points[n, 2] - pz
        s = values[n] - vi
        dist2 = dx * dx + dy * dy + dz * dz + dist_eps
        w = wp.pow(dist2, -0.5 * weight_power)

        dg = dx * g0 + dy * g1 + dz * g2
        pd = dx * p0 + dy * p1 + dz * p2
        common = s - dg
        beta = (-weight_power / dist2) * w

        grad_dx = beta * pd * common * dx + w * (common * p0 - pd * g0)
        grad_dy = beta * pd * common * dy + w * (common * p1 - pd * g1)
        grad_dz = beta * pd * common * dz + w * (common * p2 - pd * g2)

        wp.atomic_add(grad_points, n, 0, grad_dx)
        wp.atomic_add(grad_points, n, 1, grad_dy)
        wp.atomic_add(grad_points, n, 2, grad_dz)
        wp.atomic_add(grad_points, i, 0, -grad_dx)
        wp.atomic_add(grad_points, i, 1, -grad_dy)
        wp.atomic_add(grad_points, i, 2, -grad_dz)


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

    with wp.ScopedStream(wp_stream):
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

    with wp.ScopedStream(wp_stream):
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

    with wp.ScopedStream(wp_stream):
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
) -> torch.Tensor:
    """Compute weighted LSQ gradients with Warp kernels."""
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
    values_flat = values_fp32.reshape(n_entities, -1)
    n_components = values_flat.shape[1]

    ### Store component-wise output as (C, N, dims) for contiguous warp writes.
    grads_components = torch.empty(
        (n_components, n_entities, n_dims),
        dtype=torch.float32,
        device=points.device,
    )

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
        wp_device=wp_device,
        wp_stream=wp_stream,
    )

    output = grads_components.permute(1, 2, 0).reshape(n_entities, n_dims, *value_shape)
    if output.dtype != values.dtype:
        output = output.to(dtype=values.dtype)
    return output


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
) -> torch.Tensor:
    """Fake tensor propagation for LSQ custom op."""
    _ = (
        neighbor_offsets,
        neighbor_indices,
        weight_power,
        min_neighbors,
        reg_eps,
        dist_eps,
    )
    return torch.empty(
        (values.shape[0], points.shape[1], *values.shape[1:]),
        device=values.device,
        dtype=values.dtype,
    )


def setup_mesh_lsq_gradient_context(
    ctx: torch.autograd.function.FunctionCtx, inputs: tuple, output: torch.Tensor
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
    _ = output
    values_fp32 = values.to(dtype=torch.float32).contiguous()
    n_entities = values_fp32.shape[0]
    ctx.save_for_backward(
        points.to(dtype=torch.float32).contiguous(),
        values_fp32.reshape(n_entities, -1).contiguous(),
        neighbor_offsets.to(dtype=torch.int32, device=points.device).contiguous(),
        neighbor_indices.to(dtype=torch.int32, device=points.device).contiguous(),
    )
    ctx.points_dtype = points.dtype
    ctx.value_shape = values.shape
    ctx.values_dtype = values.dtype
    ctx.weight_power = float(weight_power)
    ctx.min_neighbors = int(min_neighbors)
    ctx.reg_eps = float(reg_eps)
    ctx.dist_eps = float(dist_eps)


def backward_mesh_lsq_gradient(
    ctx: torch.autograd.function.FunctionCtx,
    grad_output: torch.Tensor,
) -> tuple[
    torch.Tensor | None, torch.Tensor | None, None, None, None, None, None, None
]:
    """Backward pass for LSQ custom op."""
    needs_points = ctx.needs_input_grad[0]
    needs_values = ctx.needs_input_grad[1]
    if grad_output is None or (not needs_points and not needs_values):
        return None, None, None, None, None, None, None, None

    points_fp32, values_flat_fp32, offsets_i32, indices_i32 = ctx.saved_tensors
    grad_output_fp32 = grad_output.to(dtype=torch.float32).contiguous()
    values_shape = ctx.value_shape
    n_entities = values_shape[0]
    value_shape = values_shape[1:]
    n_components = int(torch.tensor(value_shape).prod().item()) if value_shape else 1

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
            weight_power=ctx.weight_power,
            min_neighbors=ctx.min_neighbors,
            reg_eps=ctx.reg_eps,
            dist_eps=ctx.dist_eps,
            grad_points=grad_points_fp32,
            wp_device=wp_device,
            wp_stream=wp_stream,
        )
        grad_points = grad_points_fp32
        if grad_points.dtype != ctx.points_dtype:
            grad_points = grad_points.to(dtype=ctx.points_dtype)
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
            weight_power=ctx.weight_power,
            min_neighbors=ctx.min_neighbors,
            reg_eps=ctx.reg_eps,
            dist_eps=ctx.dist_eps,
            grad_values_flat=grad_values_flat,
            wp_device=wp_device,
            wp_stream=wp_stream,
        )

    grad_values = None
    if needs_values and grad_values_flat is not None:
        grad_values = grad_values_flat.reshape(values_shape)
        if grad_values.dtype != ctx.values_dtype:
            grad_values = grad_values.to(dtype=ctx.values_dtype)
    return grad_points, grad_values, None, None, None, None, None, None


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
    reg_eps: float = 1.0e-6,
    safe_epsilon: float | None = None,
) -> torch.Tensor:
    """Compute weighted LSQ mesh gradients with Warp kernels.

    Notes
    -----
    Warp kernels compute in ``float32`` internally. Inputs in wider floating
    dtypes are accepted and cast to ``float32`` for compute. Float64 inputs are
    accepted, but derivative accuracy is limited to ``float32`` precision.
    """
    dist_eps = resolve_safe_epsilon(safe_epsilon=safe_epsilon, dtype=torch.float32)
    return mesh_lsq_gradient_impl(
        points,
        values,
        neighbor_offsets,
        neighbor_indices,
        float(weight_power),
        int(min_neighbors),
        float(reg_eps),
        float(dist_eps),
    )
