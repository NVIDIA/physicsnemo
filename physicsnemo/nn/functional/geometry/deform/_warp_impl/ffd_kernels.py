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

"""Warp kernels for lattice free-form deformation.

Every kernel is written once as a generic Warp kernel (``typing.Any``
annotations, ``type(...)`` scalar constructors) and instantiated for float32
and float64 through :func:`warp.overload`, so the two precisions share one
numerical definition by construction.

All tensor-product bases run through one kernel family via a per-axis active
window: Bernstein activates every lattice node along an axis, the uniform
cubic B-spline activates the four nodes around the containing knot span, and
the interpolating bases activate the two nodes bounding the containing cell.
The kernels are dimension-generic. Per-axis lattice sizes arrive in
``resolution`` and coordinates are indexed componentwise.
"""

from typing import Any

import warp as wp

from .kernels import _precision_overload

BERNSTEIN_BASIS_ID = 0
BSPLINE_BASIS_ID = 1
LINEAR_BASIS_ID = 2
CUBIC_HERMITE_BASIS_ID = 3
QUINTIC_HERMITE_BASIS_ID = 4
# Keep common coarse lattices on the faster product path. At degree 64 the
# largest binomial coefficient is still many orders below the float32 limit.
_BERNSTEIN_DIRECT_MAX_DEGREE = 64


@wp.func
def _bernstein_value(index: int, degree: int, u: Any):
    """Evaluate the Bernstein polynomial ``B_{index, degree}`` at ``u``.

    Low degrees use the direct product. For higher degrees, evaluate the
    binomial probability mass in log space so a large binomial coefficient
    cannot overflow before the powers of ``u`` and ``1-u`` reduce it.
    """

    zero = type(u)(0.0)
    one = type(u)(1.0)
    if u == zero:
        if index == 0:
            return one
        return zero
    if u == one:
        if index == degree:
            return one
        return zero

    if degree > _BERNSTEIN_DIRECT_MAX_DEGREE:
        log_value = zero
        for k in range(index):
            log_value = log_value + wp.log(
                type(u)(degree - index + k + 1) / type(u)(k + 1)
            )
        if index > 0:
            log_value = log_value + type(u)(index) * wp.log(u)
        complement_degree = degree - index
        if complement_degree > 0:
            log_value = log_value + type(u)(complement_degree) * wp.log(one - u)
        return wp.exp(log_value)

    value = one
    for k in range(index):
        value = value * (type(u)(degree - index + k + 1) / type(u)(k + 1)) * u
    for _ in range(degree - index):
        value = value * (one - u)
    return value


@wp.func
def _bernstein_derivative(index: int, degree: int, u: Any):
    """Evaluate the Bernstein polynomial derivative ``B'_{index, degree}``."""

    zero = type(u)(0.0)
    lower = zero
    if index > 0:
        lower = _bernstein_value(index - 1, degree - 1, u)
    upper = zero
    if index < degree:
        upper = _bernstein_value(index, degree - 1, u)
    return type(u)(degree) * (lower - upper)


@wp.func
def _bspline_value(index: int, t: Any):
    """Evaluate one uniform cubic B-spline weight at cell parameter ``t``."""

    one = type(t)(1.0)
    sixth = one / type(t)(6.0)
    if index == 0:
        c = one - t
        return sixth * c * c * c
    if index == 1:
        return sixth * (type(t)(3.0) * t * t * t - type(t)(6.0) * t * t + type(t)(4.0))
    if index == 2:
        return sixth * (
            type(t)(-3.0) * t * t * t + type(t)(3.0) * t * t + type(t)(3.0) * t + one
        )
    return sixth * t * t * t


@wp.func
def _bspline_derivative(index: int, t: Any):
    """Evaluate one uniform cubic B-spline weight derivative."""

    half = type(t)(0.5)
    if index == 0:
        c = type(t)(1.0) - t
        return -half * c * c
    if index == 1:
        return half * t * (type(t)(3.0) * t - type(t)(4.0))
    if index == 2:
        return half * (-type(t)(3.0) * t * t + type(t)(2.0) * t + type(t)(1.0))
    return half * t * t


@wp.func
def _is_interpolating_basis(basis_id: int) -> bool:
    """Return whether ``basis_id`` selects a two-node cell basis."""

    return (
        basis_id == LINEAR_BASIS_ID
        or basis_id == CUBIC_HERMITE_BASIS_ID
        or basis_id == QUINTIC_HERMITE_BASIS_ID
    )


@wp.func
def _interpolating_blend(basis_id: int, t: Any):
    """Evaluate the upper-node blend function for a cell parameter ``t``."""

    if basis_id == CUBIC_HERMITE_BASIS_ID:
        return t * t * (type(t)(3.0) - type(t)(2.0) * t)
    if basis_id == QUINTIC_HERMITE_BASIS_ID:
        return t * t * t * (t * (type(t)(6.0) * t - type(t)(15.0)) + type(t)(10.0))
    return t


@wp.func
def _interpolating_blend_derivative(basis_id: int, t: Any):
    """Evaluate the cell-parameter derivative of the upper-node blend."""

    if basis_id == CUBIC_HERMITE_BASIS_ID:
        return type(t)(6.0) * t * (type(t)(1.0) - t)
    if basis_id == QUINTIC_HERMITE_BASIS_ID:
        complement = type(t)(1.0) - t
        return type(t)(30.0) * t * t * complement * complement
    return type(t)(1.0)


@wp.func
def _local_coordinate(
    points: wp.array3d(dtype=Any),
    origin: wp.array2d(dtype=Any),
    extent: wp.array2d(dtype=Any),
    b: int,
    i: int,
    d: int,
):
    return (points[b, i, d] - origin[b, d]) / extent[b, d]


@wp.func
def _inside_lattice(
    points: wp.array3d(dtype=Any),
    origin: wp.array2d(dtype=Any),
    extent: wp.array2d(dtype=Any),
    b: int,
    i: int,
    n_dims: int,
) -> bool:
    zero = type(points[b, i, 0])(0.0)
    one = type(zero)(1.0)
    outside = int(0)
    for d in range(n_dims):
        u = _local_coordinate(points, origin, extent, b, i, d)
        if u < zero or u > one or wp.isnan(u):
            outside = 1
    return outside == 0


@wp.func
def _window_count(basis_id: int, size: int) -> int:
    if basis_id == BSPLINE_BASIS_ID:
        return 4
    if _is_interpolating_basis(basis_id):
        return 2
    return size


@wp.func
def _window_start(basis_id: int, size: int, u: Any) -> int:
    if basis_id == BSPLINE_BASIS_ID:
        span = size - 3
        scaled = wp.clamp(u * type(u)(span), type(u)(0.0), type(u)(span - 1))
        return int(wp.floor(scaled))
    if _is_interpolating_basis(basis_id):
        cells = size - 1
        scaled = wp.clamp(u * type(u)(cells), type(u)(0.0), type(u)(cells - 1))
        return int(wp.floor(scaled))
    return 0


@wp.func
def _window_param(basis_id: int, size: int, start: int, u: Any):
    if basis_id == BSPLINE_BASIS_ID:
        return u * type(u)(size - 3) - type(u)(start)
    if _is_interpolating_basis(basis_id):
        return u * type(u)(size - 1) - type(u)(start)
    return u


@wp.func
def _window_value(basis_id: int, size: int, index: int, param: Any):
    if basis_id == BSPLINE_BASIS_ID:
        return _bspline_value(index, param)
    if _is_interpolating_basis(basis_id):
        upper = _interpolating_blend(basis_id, param)
        if index == 0:
            return type(param)(1.0) - upper
        return upper
    return _bernstein_value(index, size - 1, param)


@wp.func
def _window_derivative(basis_id: int, size: int, index: int, param: Any):
    if basis_id == BSPLINE_BASIS_ID:
        return _bspline_derivative(index, param)
    if _is_interpolating_basis(basis_id):
        derivative = _interpolating_blend_derivative(basis_id, param)
        if index == 0:
            return -derivative
        return derivative
    return _bernstein_derivative(index, size - 1, param)


@wp.func
def _param_gradient_scale(basis_id: int, size: int, u: Any):
    """Return the window-parameter derivative with respect to ``u``."""

    if basis_id == BSPLINE_BASIS_ID:
        return type(u)(size - 3)
    if _is_interpolating_basis(basis_id):
        return type(u)(size - 1)
    return type(u)(1.0)


@wp.func
def _window_weight(
    points: wp.array3d(dtype=Any),
    origin: wp.array2d(dtype=Any),
    extent: wp.array2d(dtype=Any),
    resolution: wp.array(dtype=wp.int32),
    basis_id: int,
    n_dims: int,
    window_total: int,
    b: int,
    i: int,
    m: int,
):
    """Tensor-product basis weight of window slot ``m`` for one query point."""

    weight = type(points[b, i, 0])(1.0)
    divisor = window_total
    for d in range(n_dims):
        size = resolution[d]
        count = _window_count(basis_id, size)
        divisor = divisor // count
        index = (m // divisor) % count
        u = _local_coordinate(points, origin, extent, b, i, d)
        start = _window_start(basis_id, size, u)
        param = _window_param(basis_id, size, start, u)
        weight = weight * _window_value(basis_id, size, index, param)
    return weight


@wp.func
def _window_weight_gradient(
    points: wp.array3d(dtype=Any),
    origin: wp.array2d(dtype=Any),
    extent: wp.array2d(dtype=Any),
    resolution: wp.array(dtype=wp.int32),
    basis_id: int,
    n_dims: int,
    window_total: int,
    b: int,
    i: int,
    m: int,
    target: int,
):
    """Window-weight derivative with respect to the local coordinate ``u_target``."""

    gradient = type(points[b, i, 0])(1.0)
    divisor = window_total
    for d in range(n_dims):
        size = resolution[d]
        count = _window_count(basis_id, size)
        divisor = divisor // count
        index = (m // divisor) % count
        u = _local_coordinate(points, origin, extent, b, i, d)
        start = _window_start(basis_id, size, u)
        param = _window_param(basis_id, size, start, u)
        if d == target:
            gradient = (
                gradient
                * _window_derivative(basis_id, size, index, param)
                * _param_gradient_scale(basis_id, size, u)
            )
        else:
            gradient = gradient * _window_value(basis_id, size, index, param)
    return gradient


@wp.func
def _window_lattice_index(
    points: wp.array3d(dtype=Any),
    origin: wp.array2d(dtype=Any),
    extent: wp.array2d(dtype=Any),
    resolution: wp.array(dtype=wp.int32),
    basis_id: int,
    n_dims: int,
    window_total: int,
    n_lattice: int,
    b: int,
    i: int,
    m: int,
) -> int:
    """Row-major flat lattice-node index addressed by window slot ``m``."""

    divisor = window_total
    stride = n_lattice
    flat = int(0)
    for d in range(n_dims):
        size = resolution[d]
        count = _window_count(basis_id, size)
        divisor = divisor // count
        index = (m // divisor) % count
        u = _local_coordinate(points, origin, extent, b, i, d)
        start = _window_start(basis_id, size, u)
        stride = stride // size
        flat = flat + (start + index) * stride
    return flat


@wp.kernel
def _ffd_forward(
    points: wp.array3d(dtype=Any),
    lattice_displacements: wp.array3d(dtype=Any),
    origin: wp.array2d(dtype=Any),
    extent: wp.array2d(dtype=Any),
    resolution: wp.array(dtype=wp.int32),
    basis_id: int,
    n_dims: int,
    n_lattice: int,
    window_total: int,
    field: wp.array3d(dtype=Any),
):
    """Interpolate a lattice free-form displacement field."""

    b, i = wp.tid()
    zero = type(points[b, i, 0])(0.0)
    for d in range(n_dims):
        field[b, i, d] = zero
    if not _inside_lattice(points, origin, extent, b, i, n_dims):
        return
    for m in range(window_total):
        weight = _window_weight(
            points, origin, extent, resolution, basis_id, n_dims, window_total, b, i, m
        )
        flat = _window_lattice_index(
            points,
            origin,
            extent,
            resolution,
            basis_id,
            n_dims,
            window_total,
            n_lattice,
            b,
            i,
            m,
        )
        for d in range(n_dims):
            field[b, i, d] = field[b, i, d] + weight * lattice_displacements[b, flat, d]


@wp.kernel
def _ffd_backward(
    points: wp.array3d(dtype=Any),
    origin: wp.array2d(dtype=Any),
    extent: wp.array2d(dtype=Any),
    resolution: wp.array(dtype=wp.int32),
    basis_id: int,
    n_dims: int,
    n_lattice: int,
    window_total: int,
    grad_field: wp.array3d(dtype=Any),
    grad_lattice_displacements: wp.array3d(dtype=Any),
):
    """Accumulate the lattice-displacement pullback over window slots."""

    b, i, m = wp.tid()
    if not _inside_lattice(points, origin, extent, b, i, n_dims):
        return
    weight = _window_weight(
        points, origin, extent, resolution, basis_id, n_dims, window_total, b, i, m
    )
    flat = _window_lattice_index(
        points,
        origin,
        extent,
        resolution,
        basis_id,
        n_dims,
        window_total,
        n_lattice,
        b,
        i,
        m,
    )
    for d in range(n_dims):
        wp.atomic_add(
            grad_lattice_displacements, b, flat, d, weight * grad_field[b, i, d]
        )


@wp.kernel
def _ffd_point_backward(
    points: wp.array3d(dtype=Any),
    lattice_displacements: wp.array3d(dtype=Any),
    origin: wp.array2d(dtype=Any),
    extent: wp.array2d(dtype=Any),
    resolution: wp.array(dtype=wp.int32),
    basis_id: int,
    n_dims: int,
    n_lattice: int,
    window_total: int,
    grad_field: wp.array3d(dtype=Any),
    grad_points: wp.array3d(dtype=Any),
):
    """Query-centric point pullback with no inter-point atomics.

    Outside points keep a zero field gradient: the identity term of the
    deformation is applied outside these kernels, so masked rows contribute
    nothing here.
    """

    b, i = wp.tid()
    zero = type(points[b, i, 0])(0.0)
    for d in range(n_dims):
        grad_points[b, i, d] = zero
    if not _inside_lattice(points, origin, extent, b, i, n_dims):
        return
    for m in range(window_total):
        flat = _window_lattice_index(
            points,
            origin,
            extent,
            resolution,
            basis_id,
            n_dims,
            window_total,
            n_lattice,
            b,
            i,
            m,
        )
        dot = zero
        for d in range(n_dims):
            dot = dot + grad_field[b, i, d] * lattice_displacements[b, flat, d]
        if dot != zero:
            for d in range(n_dims):
                gradient = _window_weight_gradient(
                    points,
                    origin,
                    extent,
                    resolution,
                    basis_id,
                    n_dims,
                    window_total,
                    b,
                    i,
                    m,
                    d,
                )
                grad_points[b, i, d] = (
                    grad_points[b, i, d] + dot * gradient / extent[b, d]
                )


_FORWARD_3D = ("points", "lattice_displacements", "field")
_FORWARD_2D = ("origin", "extent")
_BACKWARD_3D = ("points", "grad_field", "grad_lattice_displacements")
_BACKWARD_2D = ("origin", "extent")
_POINT_BACKWARD_3D = ("points", "lattice_displacements", "grad_field", "grad_points")
_POINT_BACKWARD_2D = ("origin", "extent")

ffd_forward_f32 = _precision_overload(
    _ffd_forward, wp.float32, _FORWARD_3D, _FORWARD_2D
)
ffd_forward_f64 = _precision_overload(
    _ffd_forward, wp.float64, _FORWARD_3D, _FORWARD_2D
)
ffd_backward_f32 = _precision_overload(
    _ffd_backward, wp.float32, _BACKWARD_3D, _BACKWARD_2D
)
ffd_backward_f64 = _precision_overload(
    _ffd_backward, wp.float64, _BACKWARD_3D, _BACKWARD_2D
)
ffd_point_backward_f32 = _precision_overload(
    _ffd_point_backward, wp.float32, _POINT_BACKWARD_3D, _POINT_BACKWARD_2D
)
ffd_point_backward_f64 = _precision_overload(
    _ffd_point_backward, wp.float64, _POINT_BACKWARD_3D, _POINT_BACKWARD_2D
)


__all__ = [
    "BERNSTEIN_BASIS_ID",
    "BSPLINE_BASIS_ID",
    "LINEAR_BASIS_ID",
    "CUBIC_HERMITE_BASIS_ID",
    "QUINTIC_HERMITE_BASIS_ID",
    "ffd_backward_f32",
    "ffd_backward_f64",
    "ffd_forward_f32",
    "ffd_forward_f64",
    "ffd_point_backward_f32",
    "ffd_point_backward_f64",
]
