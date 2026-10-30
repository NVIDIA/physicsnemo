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
wp.config.log_level = wp.LOG_WARNING


# Added to the diagonal of each normal matrix, so that the systems of degenerate
# neighborhoods stay stable in float32.
REG_EPS = 1.0e-6


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

        dist2 = dx * dx + dist_eps
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
def _point_2d(points: wp.array2d(dtype=wp.float32), i: int) -> wp.vec2d:
    """Coordinates of entity ``i``, in float64."""
    return wp.vec2d(wp.float64(points[i, 0]), wp.float64(points[i, 1]))


@wp.func
def _point_3d(points: wp.array2d(dtype=wp.float32), i: int) -> wp.vec3d:
    """Coordinates of entity ``i``, in float64."""
    return wp.vec3d(
        wp.float64(points[i, 0]), wp.float64(points[i, 1]), wp.float64(points[i, 2])
    )


@wp.func
def _weight(dist2: wp.float64, weight_power: float, dist_eps: float) -> wp.float64:
    """Inverse-distance weight of a neighbor at squared distance ``dist2``."""
    return wp.pow(dist2 + wp.float64(dist_eps), wp.float64(-0.5 * weight_power))


@wp.func
def _normal_matrix_2d(
    points: wp.array2d(dtype=wp.float32),
    indices: wp.array(dtype=wp.int32),
    start: int,
    end: int,
    center: wp.vec2d,
    weight_power: float,
    reg_eps: float,
    dist_eps: float,
) -> wp.mat22d:
    """``A^T W A`` of one entity, plus ``reg_eps`` on the diagonal."""
    m = wp.diag(wp.vec2d(wp.float64(reg_eps), wp.float64(reg_eps)))
    for p in range(start, end):
        d = _point_2d(points, indices[p]) - center
        m = m + _weight(wp.dot(d, d), weight_power, dist_eps) * wp.outer(d, d)
    return m


@wp.func
def _normal_matrix_3d(
    points: wp.array2d(dtype=wp.float32),
    indices: wp.array(dtype=wp.int32),
    start: int,
    end: int,
    center: wp.vec3d,
    weight_power: float,
    reg_eps: float,
    dist_eps: float,
) -> wp.mat33d:
    """``A^T W A`` of one entity, plus ``reg_eps`` on the diagonal."""
    r = wp.float64(reg_eps)
    m = wp.diag(wp.vec3d(r, r, r))
    for p in range(start, end):
        d = _point_3d(points, indices[p]) - center
        m = m + _weight(wp.dot(d, d), weight_power, dist_eps) * wp.outer(d, d)
    return m


# The normal matrices are formed and solved in float64, in which the products
# of float32 coordinates are exact. A regularized matrix counts as singular by
# the test that rejects the singular systems the regularization leaves (one
# neighbor, or collinear neighbors). An unregularized one comes from neighbors
# that span the space, so only a determinant near float64 rounding, relative
# to the matrix's scale, counts as singular: cells far thinner than float32
# could solve keep their gradients.
@wp.func
def _inverse_2d(m: wp.mat22d, reg_eps: float, dist_eps: float) -> wp.mat22d:
    """Inverse of a normal matrix, or zero where it counts as singular."""
    scale = m[0, 0] * m[1, 1]
    tolerance = wp.float64(1.0e-13)
    if reg_eps > 0.0:
        scale = scale + wp.float64(dist_eps)
        tolerance = wp.float64(1.0e-6)
    if wp.abs(wp.determinant(m)) > tolerance * scale:
        return wp.inverse(m)
    return wp.mat22d()


@wp.func
def _inverse_3d(m: wp.mat33d, reg_eps: float, dist_eps: float) -> wp.mat33d:
    """Inverse of a normal matrix, or zero where it counts as singular."""
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    scale = trace * trace * trace
    tolerance = wp.float64(1.0e-13)
    if reg_eps > 0.0:
        scale = scale + wp.float64(dist_eps)
        tolerance = wp.float64(1.0e-8)
    if wp.abs(wp.determinant(m)) > tolerance * scale:
        return wp.inverse(m)
    return wp.mat33d()


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

    # Solve M g = A^T W b, both accumulated over the neighbors in float64.
    center = _point_2d(points, i)
    m = _normal_matrix_2d(
        points, indices, start, end, center, weight_power, reg_eps, dist_eps
    )
    inverse = _inverse_2d(m, reg_eps, dist_eps)
    value = wp.float64(values[i])
    b = wp.vec2d()
    for p in range(start, end):
        n = indices[p]
        d = _point_2d(points, n) - center
        dphi = wp.float64(values[n]) - value
        b = b + _weight(wp.dot(d, d), weight_power, dist_eps) * dphi * d
    g = inverse * b

    gradients[i, 0] = wp.float32(g[0])
    gradients[i, 1] = wp.float32(g[1])
    solved_i = int(0)
    if inverse[0, 0] > wp.float64(0.0):
        solved_i = 1
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

    # Solve M g = A^T W b, both accumulated over the neighbors in float64.
    center = _point_3d(points, i)
    m = _normal_matrix_3d(
        points, indices, start, end, center, weight_power, reg_eps, dist_eps
    )
    inverse = _inverse_3d(m, reg_eps, dist_eps)
    value = wp.float64(values[i])
    b = wp.vec3d()
    for p in range(start, end):
        n = indices[p]
        d = _point_3d(points, n) - center
        dphi = wp.float64(values[n]) - value
        b = b + _weight(wp.dot(d, d), weight_power, dist_eps) * dphi * d
    g = inverse * b

    gradients[i, 0] = wp.float32(g[0])
    gradients[i, 1] = wp.float32(g[1])
    gradients[i, 2] = wp.float32(g[2])
    solved_i = int(0)
    if inverse[0, 0] > wp.float64(0.0):
        solved_i = 1
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
    if end - start < min_neighbors:
        return

    center = _point_2d(points, i)
    m = _normal_matrix_2d(
        points, indices, start, end, center, weight_power, reg_eps, dist_eps
    )
    q = _inverse_2d(m, reg_eps, dist_eps) * wp.vec2d(
        wp.float64(grad_output[i, 0]), wp.float64(grad_output[i, 1])
    )

    self_contrib = wp.float64(0.0)
    for p in range(start, end):
        n = indices[p]
        d = _point_2d(points, n) - center
        c = _weight(wp.dot(d, d), weight_power, dist_eps) * wp.dot(q, d)
        wp.atomic_add(grad_values, n, wp.float32(c))
        self_contrib = self_contrib - c

    wp.atomic_add(grad_values, i, wp.float32(self_contrib))


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

    center = _point_3d(points, i)
    m = _normal_matrix_3d(
        points, indices, start, end, center, weight_power, reg_eps, dist_eps
    )
    q = _inverse_3d(m, reg_eps, dist_eps) * wp.vec3d(
        wp.float64(grad_output[i, 0]),
        wp.float64(grad_output[i, 1]),
        wp.float64(grad_output[i, 2]),
    )

    self_contrib = wp.float64(0.0)
    for p in range(start, end):
        n = indices[p]
        d = _point_3d(points, n) - center
        c = _weight(wp.dot(d, d), weight_power, dist_eps) * wp.dot(q, d)
        wp.atomic_add(grad_values, n, wp.float32(c))
        self_contrib = self_contrib - c

    wp.atomic_add(grad_values, i, wp.float32(self_contrib))


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
    if end - start < min_neighbors:
        return

    center = _point_2d(points, i)
    m = _normal_matrix_2d(
        points, indices, start, end, center, weight_power, reg_eps, dist_eps
    )
    inverse = _inverse_2d(m, reg_eps, dist_eps)
    if inverse[0, 0] <= wp.float64(0.0):
        return

    value = wp.float64(values[i])
    b = wp.vec2d()
    for p in range(start, end):
        n = indices[p]
        d = _point_2d(points, n) - center
        dphi = wp.float64(values[n]) - value
        b = b + _weight(wp.dot(d, d), weight_power, dist_eps) * dphi * d
    g = inverse * b
    q = inverse * wp.vec2d(wp.float64(grad_output[i, 0]), wp.float64(grad_output[i, 1]))

    power = wp.float64(weight_power)
    for p in range(start, end):
        n = indices[p]
        d = _point_2d(points, n) - center
        dist2 = wp.dot(d, d) + wp.float64(dist_eps)
        w = wp.pow(dist2, wp.float64(-0.5 * weight_power))
        pd = wp.dot(d, q)
        common = wp.float64(values[n]) - value - wp.dot(d, g)
        beta = (-power / dist2) * w
        grad_d = (beta * pd * common) * d + w * (common * q - pd * g)

        wp.atomic_add(grad_points, n, 0, wp.float32(grad_d[0]))
        wp.atomic_add(grad_points, n, 1, wp.float32(grad_d[1]))
        wp.atomic_add(grad_points, i, 0, -wp.float32(grad_d[0]))
        wp.atomic_add(grad_points, i, 1, -wp.float32(grad_d[1]))


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

    center = _point_3d(points, i)
    m = _normal_matrix_3d(
        points, indices, start, end, center, weight_power, reg_eps, dist_eps
    )
    inverse = _inverse_3d(m, reg_eps, dist_eps)
    if inverse[0, 0] <= wp.float64(0.0):
        return

    value = wp.float64(values[i])
    b = wp.vec3d()
    for p in range(start, end):
        n = indices[p]
        d = _point_3d(points, n) - center
        dphi = wp.float64(values[n]) - value
        b = b + _weight(wp.dot(d, d), weight_power, dist_eps) * dphi * d
    g = inverse * b
    q = inverse * wp.vec3d(
        wp.float64(grad_output[i, 0]),
        wp.float64(grad_output[i, 1]),
        wp.float64(grad_output[i, 2]),
    )

    power = wp.float64(weight_power)
    for p in range(start, end):
        n = indices[p]
        d = _point_3d(points, n) - center
        dist2 = wp.dot(d, d) + wp.float64(dist_eps)
        w = wp.pow(dist2, wp.float64(-0.5 * weight_power))
        pd = wp.dot(d, q)
        common = wp.float64(values[n]) - value - wp.dot(d, g)
        beta = (-power / dist2) * w
        grad_d = (beta * pd * common) * d + w * (common * q - pd * g)

        wp.atomic_add(grad_points, n, 0, wp.float32(grad_d[0]))
        wp.atomic_add(grad_points, n, 1, wp.float32(grad_d[1]))
        wp.atomic_add(grad_points, n, 2, wp.float32(grad_d[2]))
        wp.atomic_add(grad_points, i, 0, -wp.float32(grad_d[0]))
        wp.atomic_add(grad_points, i, 1, -wp.float32(grad_d[1]))
        wp.atomic_add(grad_points, i, 2, -wp.float32(grad_d[2]))


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
    values_flat = values_fp32.reshape(n_entities, -1)
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

    output = grads_components.permute(1, 2, 0).reshape(n_entities, n_dims, *value_shape)
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
        values.to(dtype=torch.float32).reshape(n_entities, -1).contiguous()
    )
    offsets_i32 = neighbor_offsets.to(dtype=torch.int32, device=points.device)
    indices_i32 = neighbor_indices.to(dtype=torch.int32, device=points.device)
    offsets_i32, indices_i32 = offsets_i32.contiguous(), indices_i32.contiguous()
    grad_output_fp32 = grad_output.to(dtype=torch.float32).contiguous()
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

    Notes
    -----
    Warp kernels compute in ``float32`` internally. Inputs in wider floating
    dtypes are accepted and cast to ``float32`` for compute. Float64 inputs are
    accepted, but derivative accuracy is limited to ``float32`` precision.
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
