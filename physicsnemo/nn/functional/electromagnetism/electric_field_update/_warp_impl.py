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

from typing import Sequence

import torch
import warp as wp

from physicsnemo.core.function_spec import FunctionSpec

from .utils import (
    _as_spacing_tensor,
    _normalize_material_field,
    _normalize_offset,
    _validate_common_inputs,
)

wp.config.quiet = True
wp.init()


# Compute periodic predecessor index for one axis.
@wp.func
def _periodic_prev(index: int, size: int) -> int:
    if index == 0:
        return size - 1
    return index - 1


# Average one scalar field for the Yee Ex location.
@wp.func
def _material_avg_x(
    field: wp.array3d(dtype=wp.float32),
    i: int,
    j: int,
    k: int,
    i_prev: int,
    j_prev: int,
    k_prev: int,
) -> wp.float32:
    _ = i_prev
    return 0.25 * (
        field[i, j, k]
        + field[i, j, k_prev]
        + field[i, j_prev, k]
        + field[i, j_prev, k_prev]
    )


# Average one scalar field for the Yee Ey location.
@wp.func
def _material_avg_y(
    field: wp.array3d(dtype=wp.float32),
    i: int,
    j: int,
    k: int,
    i_prev: int,
    j_prev: int,
    k_prev: int,
) -> wp.float32:
    _ = j_prev
    return 0.25 * (
        field[i, j, k]
        + field[i, j, k_prev]
        + field[i_prev, j, k]
        + field[i_prev, j, k_prev]
    )


# Average one scalar field for the Yee Ez location.
@wp.func
def _material_avg_z(
    field: wp.array3d(dtype=wp.float32),
    i: int,
    j: int,
    k: int,
    i_prev: int,
    j_prev: int,
    k_prev: int,
) -> wp.float32:
    _ = k_prev
    return 0.25 * (
        field[i, j, k]
        + field[i, j_prev, k]
        + field[i_prev, j, k]
        + field[i_prev, j_prev, k]
    )


# Sample impressed current with offset + clipping.
@wp.func
def _sample_current(
    current: wp.array4d(dtype=wp.float32),
    component: int,
    i: int,
    j: int,
    k: int,
    offset_x: int,
    offset_y: int,
    offset_z: int,
    size_x: int,
    size_y: int,
    size_z: int,
) -> wp.float32:
    src_i = i - offset_x
    src_j = j - offset_y
    src_k = k - offset_z

    if src_i < 0 or src_i >= size_x:
        return 0.0
    if src_j < 0 or src_j >= size_y:
        return 0.0
    if src_k < 0 or src_k >= size_z:
        return 0.0

    return current[component, src_i, src_j, src_k]


# Compute curl(H) at one electric-grid cell.
@wp.func
def _curl_h(
    magnetic_field: wp.array4d(dtype=wp.float32),
    i: int,
    j: int,
    k: int,
    i_prev: int,
    j_prev: int,
    k_prev: int,
) -> wp.vec3f:
    hx = magnetic_field[0, i, j, k]
    hy = magnetic_field[1, i, j, k]
    hz = magnetic_field[2, i, j, k]

    m_x_1_0_1 = magnetic_field[0, i, j_prev, k]
    m_x_1_1_0 = magnetic_field[0, i, j, k_prev]
    m_y_0_1_1 = magnetic_field[1, i_prev, j, k]
    m_y_1_1_0 = magnetic_field[1, i, j, k_prev]
    m_z_0_1_1 = magnetic_field[2, i_prev, j, k]
    m_z_1_0_1 = magnetic_field[2, i, j_prev, k]

    curl_h_x = (hz - m_z_1_0_1) - (hy - m_y_1_1_0)
    curl_h_y = (hx - m_x_1_1_0) - (hz - m_z_0_1_1)
    curl_h_z = (hy - m_y_0_1_1) - (hx - m_x_1_0_1)

    return wp.vec3f(curl_h_x, curl_h_y, curl_h_z)


# Compute update coefficients (c_ee, c_eh, c_ej) for one component.
@wp.func
def _coefficients(
    eps_value: wp.float32,
    sigma_value: wp.float32,
    dt: wp.float32,
    spacing_value: wp.float32,
) -> wp.vec3f:
    denom = 2.0 * eps_value + sigma_value * dt
    c_ee = (2.0 * eps_value - sigma_value * dt) / denom
    c_eh = (2.0 * dt) / (spacing_value * denom)
    c_ej = (-2.0 * dt) / denom
    return wp.vec3f(c_ee, c_eh, c_ej)


# Compute local derivative wrt averaged eps/sigma for one component update.
@wp.func
def _material_partials(
    eps_value: wp.float32,
    sigma_value: wp.float32,
    e_value: wp.float32,
    curl_value: wp.float32,
    current_value: wp.float32,
    dt: wp.float32,
    spacing_value: wp.float32,
) -> wp.vec2f:
    denom = 2.0 * eps_value + sigma_value * dt
    inv_denom2 = 1.0 / (denom * denom)

    d_c_ee_d_eps = 4.0 * sigma_value * dt * inv_denom2
    d_c_eh_d_eps = (-4.0 * dt / spacing_value) * inv_denom2
    d_c_ej_d_eps = 4.0 * dt * inv_denom2

    d_c_ee_d_sigma = -4.0 * eps_value * dt * inv_denom2
    d_c_eh_d_sigma = (-2.0 * dt * dt / spacing_value) * inv_denom2
    d_c_ej_d_sigma = 2.0 * dt * dt * inv_denom2

    d_update_d_eps = (
        d_c_ee_d_eps * e_value
        + d_c_eh_d_eps * curl_value
        + d_c_ej_d_eps * current_value
    )
    d_update_d_sigma = (
        d_c_ee_d_sigma * e_value
        + d_c_eh_d_sigma * curl_value
        + d_c_ej_d_sigma * current_value
    )

    return wp.vec2f(d_update_d_eps, d_update_d_sigma)


# Apply one component update from local values and coefficients.
@wp.func
def _update_component(
    e_value: wp.float32,
    curl_value: wp.float32,
    current_value: wp.float32,
    coeffs: wp.vec3f,
) -> wp.float32:
    return coeffs[0] * e_value + coeffs[1] * curl_value + coeffs[2] * current_value


# Resolve per-component material values at Yee locations.
@wp.func
def _material_components(
    field: wp.array3d(dtype=wp.float32),
    scalar_value: wp.float32,
    is_scalar: int,
    i: int,
    j: int,
    k: int,
    i_prev: int,
    j_prev: int,
    k_prev: int,
) -> wp.vec3f:
    if is_scalar == 1:
        return wp.vec3f(scalar_value, scalar_value, scalar_value)
    return wp.vec3f(
        _material_avg_x(field, i, j, k, i_prev, j_prev, k_prev),
        _material_avg_y(field, i, j, k, i_prev, j_prev, k_prev),
        _material_avg_z(field, i, j, k, i_prev, j_prev, k_prev),
    )


# Accumulate gradient contributions into magnetic field neighbors.
@wp.func
def _accumulate_grad_magnetic(
    grad_magnetic: wp.array4d(dtype=wp.float32),
    i: int,
    j: int,
    k: int,
    i_prev: int,
    j_prev: int,
    k_prev: int,
    fx: wp.float32,
    fy: wp.float32,
    fz: wp.float32,
) -> None:
    wp.atomic_add(grad_magnetic, 2, i, j, k, fx)
    wp.atomic_add(grad_magnetic, 2, i, j_prev, k, -fx)
    wp.atomic_add(grad_magnetic, 1, i, j, k, -fx)
    wp.atomic_add(grad_magnetic, 1, i, j, k_prev, fx)

    wp.atomic_add(grad_magnetic, 0, i, j, k, fy)
    wp.atomic_add(grad_magnetic, 0, i, j, k_prev, -fy)
    wp.atomic_add(grad_magnetic, 2, i, j, k, -fy)
    wp.atomic_add(grad_magnetic, 2, i_prev, j, k, fy)

    wp.atomic_add(grad_magnetic, 1, i, j, k, fz)
    wp.atomic_add(grad_magnetic, 1, i_prev, j, k, -fz)
    wp.atomic_add(grad_magnetic, 0, i, j, k, -fz)
    wp.atomic_add(grad_magnetic, 0, i, j_prev, k, fz)


# Scatter one Ex-style averaged gradient contribution to scalar field cells.
@wp.func
def _scatter_avg_x(
    grad_field: wp.array3d(dtype=wp.float32),
    grad_value: wp.float32,
    i: int,
    j: int,
    k: int,
    i_prev: int,
    j_prev: int,
    k_prev: int,
) -> None:
    _ = i_prev
    contrib = 0.25 * grad_value
    wp.atomic_add(grad_field, i, j, k, contrib)
    wp.atomic_add(grad_field, i, j, k_prev, contrib)
    wp.atomic_add(grad_field, i, j_prev, k, contrib)
    wp.atomic_add(grad_field, i, j_prev, k_prev, contrib)


# Scatter one Ey-style averaged gradient contribution to scalar field cells.
@wp.func
def _scatter_avg_y(
    grad_field: wp.array3d(dtype=wp.float32),
    grad_value: wp.float32,
    i: int,
    j: int,
    k: int,
    i_prev: int,
    j_prev: int,
    k_prev: int,
) -> None:
    _ = j_prev
    contrib = 0.25 * grad_value
    wp.atomic_add(grad_field, i, j, k, contrib)
    wp.atomic_add(grad_field, i, j, k_prev, contrib)
    wp.atomic_add(grad_field, i_prev, j, k, contrib)
    wp.atomic_add(grad_field, i_prev, j, k_prev, contrib)


# Scatter one Ez-style averaged gradient contribution to scalar field cells.
@wp.func
def _scatter_avg_z(
    grad_field: wp.array3d(dtype=wp.float32),
    grad_value: wp.float32,
    i: int,
    j: int,
    k: int,
    i_prev: int,
    j_prev: int,
    k_prev: int,
) -> None:
    _ = k_prev
    contrib = 0.25 * grad_value
    wp.atomic_add(grad_field, i, j, k, contrib)
    wp.atomic_add(grad_field, i, j_prev, k, contrib)
    wp.atomic_add(grad_field, i_prev, j, k, contrib)
    wp.atomic_add(grad_field, i_prev, j_prev, k, contrib)


# Forward kernel for scalar eps + scalar sigma_e.
@wp.kernel
def _electric_field_update_kernel_scalar_scalar(
    electric_field: wp.array4d(dtype=wp.float32),
    magnetic_field: wp.array4d(dtype=wp.float32),
    impressed_current: wp.array4d(dtype=wp.float32),
    output: wp.array4d(dtype=wp.float32),
    eps_scalar: wp.float32,
    sigma_scalar: wp.float32,
    dt: wp.float32,
    spacing: wp.array(dtype=wp.float32),
    offset_x: int,
    offset_y: int,
    offset_z: int,
    current_x: int,
    current_y: int,
    current_z: int,
):
    i, j, k = wp.tid()
    nx = electric_field.shape[1]
    ny = electric_field.shape[2]
    nz = electric_field.shape[3]

    i_prev = _periodic_prev(i, nx)
    j_prev = _periodic_prev(j, ny)
    k_prev = _periodic_prev(k, nz)

    curl = _curl_h(magnetic_field, i, j, k, i_prev, j_prev, k_prev)
    j_x = _sample_current(
        impressed_current,
        0,
        i,
        j,
        k,
        offset_x,
        offset_y,
        offset_z,
        current_x,
        current_y,
        current_z,
    )
    j_y = _sample_current(
        impressed_current,
        1,
        i,
        j,
        k,
        offset_x,
        offset_y,
        offset_z,
        current_x,
        current_y,
        current_z,
    )
    j_z = _sample_current(
        impressed_current,
        2,
        i,
        j,
        k,
        offset_x,
        offset_y,
        offset_z,
        current_x,
        current_y,
        current_z,
    )

    coeff_x = _coefficients(eps_scalar, sigma_scalar, dt, spacing[0])
    coeff_y = _coefficients(eps_scalar, sigma_scalar, dt, spacing[1])
    coeff_z = _coefficients(eps_scalar, sigma_scalar, dt, spacing[2])

    output[0, i, j, k] = _update_component(
        electric_field[0, i, j, k], curl[0], j_x, coeff_x
    )
    output[1, i, j, k] = _update_component(
        electric_field[1, i, j, k], curl[1], j_y, coeff_y
    )
    output[2, i, j, k] = _update_component(
        electric_field[2, i, j, k], curl[2], j_z, coeff_z
    )


# Forward kernel for scalar eps + spatial sigma_e.
@wp.kernel
def _electric_field_update_kernel_scalar_sigma_field(
    electric_field: wp.array4d(dtype=wp.float32),
    magnetic_field: wp.array4d(dtype=wp.float32),
    sigma_field: wp.array3d(dtype=wp.float32),
    impressed_current: wp.array4d(dtype=wp.float32),
    output: wp.array4d(dtype=wp.float32),
    eps_scalar: wp.float32,
    dt: wp.float32,
    spacing: wp.array(dtype=wp.float32),
    offset_x: int,
    offset_y: int,
    offset_z: int,
    current_x: int,
    current_y: int,
    current_z: int,
):
    i, j, k = wp.tid()
    nx = electric_field.shape[1]
    ny = electric_field.shape[2]
    nz = electric_field.shape[3]

    i_prev = _periodic_prev(i, nx)
    j_prev = _periodic_prev(j, ny)
    k_prev = _periodic_prev(k, nz)

    sigma_x = _material_avg_x(sigma_field, i, j, k, i_prev, j_prev, k_prev)
    sigma_y = _material_avg_y(sigma_field, i, j, k, i_prev, j_prev, k_prev)
    sigma_z = _material_avg_z(sigma_field, i, j, k, i_prev, j_prev, k_prev)

    curl = _curl_h(magnetic_field, i, j, k, i_prev, j_prev, k_prev)
    j_x = _sample_current(
        impressed_current,
        0,
        i,
        j,
        k,
        offset_x,
        offset_y,
        offset_z,
        current_x,
        current_y,
        current_z,
    )
    j_y = _sample_current(
        impressed_current,
        1,
        i,
        j,
        k,
        offset_x,
        offset_y,
        offset_z,
        current_x,
        current_y,
        current_z,
    )
    j_z = _sample_current(
        impressed_current,
        2,
        i,
        j,
        k,
        offset_x,
        offset_y,
        offset_z,
        current_x,
        current_y,
        current_z,
    )

    coeff_x = _coefficients(eps_scalar, sigma_x, dt, spacing[0])
    coeff_y = _coefficients(eps_scalar, sigma_y, dt, spacing[1])
    coeff_z = _coefficients(eps_scalar, sigma_z, dt, spacing[2])

    output[0, i, j, k] = _update_component(
        electric_field[0, i, j, k], curl[0], j_x, coeff_x
    )
    output[1, i, j, k] = _update_component(
        electric_field[1, i, j, k], curl[1], j_y, coeff_y
    )
    output[2, i, j, k] = _update_component(
        electric_field[2, i, j, k], curl[2], j_z, coeff_z
    )


# Forward kernel for spatial eps + scalar sigma_e.
@wp.kernel
def _electric_field_update_kernel_eps_field_scalar(
    electric_field: wp.array4d(dtype=wp.float32),
    magnetic_field: wp.array4d(dtype=wp.float32),
    eps_field: wp.array3d(dtype=wp.float32),
    impressed_current: wp.array4d(dtype=wp.float32),
    output: wp.array4d(dtype=wp.float32),
    sigma_scalar: wp.float32,
    dt: wp.float32,
    spacing: wp.array(dtype=wp.float32),
    offset_x: int,
    offset_y: int,
    offset_z: int,
    current_x: int,
    current_y: int,
    current_z: int,
):
    i, j, k = wp.tid()
    nx = electric_field.shape[1]
    ny = electric_field.shape[2]
    nz = electric_field.shape[3]

    i_prev = _periodic_prev(i, nx)
    j_prev = _periodic_prev(j, ny)
    k_prev = _periodic_prev(k, nz)

    eps_x = _material_avg_x(eps_field, i, j, k, i_prev, j_prev, k_prev)
    eps_y = _material_avg_y(eps_field, i, j, k, i_prev, j_prev, k_prev)
    eps_z = _material_avg_z(eps_field, i, j, k, i_prev, j_prev, k_prev)

    curl = _curl_h(magnetic_field, i, j, k, i_prev, j_prev, k_prev)
    j_x = _sample_current(
        impressed_current,
        0,
        i,
        j,
        k,
        offset_x,
        offset_y,
        offset_z,
        current_x,
        current_y,
        current_z,
    )
    j_y = _sample_current(
        impressed_current,
        1,
        i,
        j,
        k,
        offset_x,
        offset_y,
        offset_z,
        current_x,
        current_y,
        current_z,
    )
    j_z = _sample_current(
        impressed_current,
        2,
        i,
        j,
        k,
        offset_x,
        offset_y,
        offset_z,
        current_x,
        current_y,
        current_z,
    )

    coeff_x = _coefficients(eps_x, sigma_scalar, dt, spacing[0])
    coeff_y = _coefficients(eps_y, sigma_scalar, dt, spacing[1])
    coeff_z = _coefficients(eps_z, sigma_scalar, dt, spacing[2])

    output[0, i, j, k] = _update_component(
        electric_field[0, i, j, k], curl[0], j_x, coeff_x
    )
    output[1, i, j, k] = _update_component(
        electric_field[1, i, j, k], curl[1], j_y, coeff_y
    )
    output[2, i, j, k] = _update_component(
        electric_field[2, i, j, k], curl[2], j_z, coeff_z
    )


# Forward kernel for spatial eps + spatial sigma_e.
@wp.kernel
def _electric_field_update_kernel_eps_field_sigma_field(
    electric_field: wp.array4d(dtype=wp.float32),
    magnetic_field: wp.array4d(dtype=wp.float32),
    eps_field: wp.array3d(dtype=wp.float32),
    sigma_field: wp.array3d(dtype=wp.float32),
    impressed_current: wp.array4d(dtype=wp.float32),
    output: wp.array4d(dtype=wp.float32),
    dt: wp.float32,
    spacing: wp.array(dtype=wp.float32),
    offset_x: int,
    offset_y: int,
    offset_z: int,
    current_x: int,
    current_y: int,
    current_z: int,
):
    i, j, k = wp.tid()
    nx = electric_field.shape[1]
    ny = electric_field.shape[2]
    nz = electric_field.shape[3]

    i_prev = _periodic_prev(i, nx)
    j_prev = _periodic_prev(j, ny)
    k_prev = _periodic_prev(k, nz)

    eps_x = _material_avg_x(eps_field, i, j, k, i_prev, j_prev, k_prev)
    eps_y = _material_avg_y(eps_field, i, j, k, i_prev, j_prev, k_prev)
    eps_z = _material_avg_z(eps_field, i, j, k, i_prev, j_prev, k_prev)
    sigma_x = _material_avg_x(sigma_field, i, j, k, i_prev, j_prev, k_prev)
    sigma_y = _material_avg_y(sigma_field, i, j, k, i_prev, j_prev, k_prev)
    sigma_z = _material_avg_z(sigma_field, i, j, k, i_prev, j_prev, k_prev)

    curl = _curl_h(magnetic_field, i, j, k, i_prev, j_prev, k_prev)
    j_x = _sample_current(
        impressed_current,
        0,
        i,
        j,
        k,
        offset_x,
        offset_y,
        offset_z,
        current_x,
        current_y,
        current_z,
    )
    j_y = _sample_current(
        impressed_current,
        1,
        i,
        j,
        k,
        offset_x,
        offset_y,
        offset_z,
        current_x,
        current_y,
        current_z,
    )
    j_z = _sample_current(
        impressed_current,
        2,
        i,
        j,
        k,
        offset_x,
        offset_y,
        offset_z,
        current_x,
        current_y,
        current_z,
    )

    coeff_x = _coefficients(eps_x, sigma_x, dt, spacing[0])
    coeff_y = _coefficients(eps_y, sigma_y, dt, spacing[1])
    coeff_z = _coefficients(eps_z, sigma_z, dt, spacing[2])

    output[0, i, j, k] = _update_component(
        electric_field[0, i, j, k], curl[0], j_x, coeff_x
    )
    output[1, i, j, k] = _update_component(
        electric_field[1, i, j, k], curl[1], j_y, coeff_y
    )
    output[2, i, j, k] = _update_component(
        electric_field[2, i, j, k], curl[2], j_z, coeff_z
    )


# Backward kernel for E/H gradients with no impressed current.
@wp.kernel
def _electric_field_update_backward_kernel_fields_no_current(
    electric_field: wp.array4d(dtype=wp.float32),
    magnetic_field: wp.array4d(dtype=wp.float32),
    eps_field: wp.array3d(dtype=wp.float32),
    sigma_field: wp.array3d(dtype=wp.float32),
    grad_output: wp.array4d(dtype=wp.float32),
    grad_electric: wp.array4d(dtype=wp.float32),
    grad_magnetic: wp.array4d(dtype=wp.float32),
    eps_scalar: wp.float32,
    sigma_scalar: wp.float32,
    eps_is_scalar: int,
    sigma_is_scalar: int,
    dt: wp.float32,
    spacing: wp.array(dtype=wp.float32),
):
    i, j, k = wp.tid()
    nx = electric_field.shape[1]
    ny = electric_field.shape[2]
    nz = electric_field.shape[3]

    i_prev = _periodic_prev(i, nx)
    j_prev = _periodic_prev(j, ny)
    k_prev = _periodic_prev(k, nz)

    eps_components = _material_components(
        eps_field, eps_scalar, eps_is_scalar, i, j, k, i_prev, j_prev, k_prev
    )
    sigma_components = _material_components(
        sigma_field, sigma_scalar, sigma_is_scalar, i, j, k, i_prev, j_prev, k_prev
    )

    curl = _curl_h(magnetic_field, i, j, k, i_prev, j_prev, k_prev)

    coeff_x = _coefficients(eps_components[0], sigma_components[0], dt, spacing[0])
    coeff_y = _coefficients(eps_components[1], sigma_components[1], dt, spacing[1])
    coeff_z = _coefficients(eps_components[2], sigma_components[2], dt, spacing[2])

    go_x = grad_output[0, i, j, k]
    go_y = grad_output[1, i, j, k]
    go_z = grad_output[2, i, j, k]

    grad_electric[0, i, j, k] = go_x * coeff_x[0]
    grad_electric[1, i, j, k] = go_y * coeff_y[0]
    grad_electric[2, i, j, k] = go_z * coeff_z[0]

    fx = go_x * coeff_x[1]
    fy = go_y * coeff_y[1]
    fz = go_z * coeff_z[1]
    _accumulate_grad_magnetic(grad_magnetic, i, j, k, i_prev, j_prev, k_prev, fx, fy, fz)


# Backward kernel for E/H(+optional current) gradients with impressed current.
@wp.kernel
def _electric_field_update_backward_kernel_fields_with_current(
    electric_field: wp.array4d(dtype=wp.float32),
    magnetic_field: wp.array4d(dtype=wp.float32),
    eps_field: wp.array3d(dtype=wp.float32),
    sigma_field: wp.array3d(dtype=wp.float32),
    impressed_current: wp.array4d(dtype=wp.float32),
    grad_output: wp.array4d(dtype=wp.float32),
    grad_electric: wp.array4d(dtype=wp.float32),
    grad_magnetic: wp.array4d(dtype=wp.float32),
    grad_current: wp.array4d(dtype=wp.float32),
    eps_scalar: wp.float32,
    sigma_scalar: wp.float32,
    eps_is_scalar: int,
    sigma_is_scalar: int,
    dt: wp.float32,
    spacing: wp.array(dtype=wp.float32),
    offset_x: int,
    offset_y: int,
    offset_z: int,
    current_x: int,
    current_y: int,
    current_z: int,
    write_grad_current: int,
):
    i, j, k = wp.tid()
    nx = electric_field.shape[1]
    ny = electric_field.shape[2]
    nz = electric_field.shape[3]

    i_prev = _periodic_prev(i, nx)
    j_prev = _periodic_prev(j, ny)
    k_prev = _periodic_prev(k, nz)

    eps_components = _material_components(
        eps_field, eps_scalar, eps_is_scalar, i, j, k, i_prev, j_prev, k_prev
    )
    sigma_components = _material_components(
        sigma_field, sigma_scalar, sigma_is_scalar, i, j, k, i_prev, j_prev, k_prev
    )

    src_i = i - offset_x
    src_j = j - offset_y
    src_k = k - offset_z
    current_in_bounds = (
        src_i >= 0
        and src_i < current_x
        and src_j >= 0
        and src_j < current_y
        and src_k >= 0
        and src_k < current_z
    )

    coeff_x = _coefficients(eps_components[0], sigma_components[0], dt, spacing[0])
    coeff_y = _coefficients(eps_components[1], sigma_components[1], dt, spacing[1])
    coeff_z = _coefficients(eps_components[2], sigma_components[2], dt, spacing[2])

    go_x = grad_output[0, i, j, k]
    go_y = grad_output[1, i, j, k]
    go_z = grad_output[2, i, j, k]

    grad_electric[0, i, j, k] = go_x * coeff_x[0]
    grad_electric[1, i, j, k] = go_y * coeff_y[0]
    grad_electric[2, i, j, k] = go_z * coeff_z[0]

    fx = go_x * coeff_x[1]
    fy = go_y * coeff_y[1]
    fz = go_z * coeff_z[1]
    _accumulate_grad_magnetic(grad_magnetic, i, j, k, i_prev, j_prev, k_prev, fx, fy, fz)

    if write_grad_current == 1 and current_in_bounds:
        grad_current[0, src_i, src_j, src_k] = go_x * coeff_x[2]
        grad_current[1, src_i, src_j, src_k] = go_y * coeff_y[2]
        grad_current[2, src_i, src_j, src_k] = go_z * coeff_z[2]


# Backward kernel for full material gradients with no impressed current.
@wp.kernel
def _electric_field_update_backward_kernel_full_no_current(
    electric_field: wp.array4d(dtype=wp.float32),
    magnetic_field: wp.array4d(dtype=wp.float32),
    eps_field: wp.array3d(dtype=wp.float32),
    sigma_field: wp.array3d(dtype=wp.float32),
    grad_output: wp.array4d(dtype=wp.float32),
    grad_electric: wp.array4d(dtype=wp.float32),
    grad_magnetic: wp.array4d(dtype=wp.float32),
    grad_eps: wp.array3d(dtype=wp.float32),
    grad_sigma: wp.array3d(dtype=wp.float32),
    eps_scalar: wp.float32,
    sigma_scalar: wp.float32,
    eps_is_scalar: int,
    sigma_is_scalar: int,
    dt: wp.float32,
    spacing: wp.array(dtype=wp.float32),
    write_grad_eps: int,
    write_grad_sigma: int,
):
    i, j, k = wp.tid()
    nx = electric_field.shape[1]
    ny = electric_field.shape[2]
    nz = electric_field.shape[3]

    i_prev = _periodic_prev(i, nx)
    j_prev = _periodic_prev(j, ny)
    k_prev = _periodic_prev(k, nz)

    eps_components = _material_components(
        eps_field, eps_scalar, eps_is_scalar, i, j, k, i_prev, j_prev, k_prev
    )
    sigma_components = _material_components(
        sigma_field, sigma_scalar, sigma_is_scalar, i, j, k, i_prev, j_prev, k_prev
    )

    curl = _curl_h(magnetic_field, i, j, k, i_prev, j_prev, k_prev)
    coeff_x = _coefficients(eps_components[0], sigma_components[0], dt, spacing[0])
    coeff_y = _coefficients(eps_components[1], sigma_components[1], dt, spacing[1])
    coeff_z = _coefficients(eps_components[2], sigma_components[2], dt, spacing[2])

    go_x = grad_output[0, i, j, k]
    go_y = grad_output[1, i, j, k]
    go_z = grad_output[2, i, j, k]

    grad_electric[0, i, j, k] = go_x * coeff_x[0]
    grad_electric[1, i, j, k] = go_y * coeff_y[0]
    grad_electric[2, i, j, k] = go_z * coeff_z[0]

    fx = go_x * coeff_x[1]
    fy = go_y * coeff_y[1]
    fz = go_z * coeff_z[1]
    _accumulate_grad_magnetic(grad_magnetic, i, j, k, i_prev, j_prev, k_prev, fx, fy, fz)

    if write_grad_eps == 1 or write_grad_sigma == 1:
        partial_x = _material_partials(
            eps_components[0],
            sigma_components[0],
            electric_field[0, i, j, k],
            curl[0],
            0.0,
            dt,
            spacing[0],
        )
        partial_y = _material_partials(
            eps_components[1],
            sigma_components[1],
            electric_field[1, i, j, k],
            curl[1],
            0.0,
            dt,
            spacing[1],
        )
        partial_z = _material_partials(
            eps_components[2],
            sigma_components[2],
            electric_field[2, i, j, k],
            curl[2],
            0.0,
            dt,
            spacing[2],
        )

        if write_grad_eps == 1:
            _scatter_avg_x(
                grad_eps, go_x * partial_x[0], i, j, k, i_prev, j_prev, k_prev
            )
            _scatter_avg_y(
                grad_eps, go_y * partial_y[0], i, j, k, i_prev, j_prev, k_prev
            )
            _scatter_avg_z(
                grad_eps, go_z * partial_z[0], i, j, k, i_prev, j_prev, k_prev
            )

        if write_grad_sigma == 1:
            _scatter_avg_x(
                grad_sigma, go_x * partial_x[1], i, j, k, i_prev, j_prev, k_prev
            )
            _scatter_avg_y(
                grad_sigma, go_y * partial_y[1], i, j, k, i_prev, j_prev, k_prev
            )
            _scatter_avg_z(
                grad_sigma, go_z * partial_z[1], i, j, k, i_prev, j_prev, k_prev
            )


# Backward kernel for full material gradients with impressed current.
@wp.kernel
def _electric_field_update_backward_kernel_full_with_current(
    electric_field: wp.array4d(dtype=wp.float32),
    magnetic_field: wp.array4d(dtype=wp.float32),
    eps_field: wp.array3d(dtype=wp.float32),
    sigma_field: wp.array3d(dtype=wp.float32),
    impressed_current: wp.array4d(dtype=wp.float32),
    grad_output: wp.array4d(dtype=wp.float32),
    grad_electric: wp.array4d(dtype=wp.float32),
    grad_magnetic: wp.array4d(dtype=wp.float32),
    grad_eps: wp.array3d(dtype=wp.float32),
    grad_sigma: wp.array3d(dtype=wp.float32),
    grad_current: wp.array4d(dtype=wp.float32),
    eps_scalar: wp.float32,
    sigma_scalar: wp.float32,
    eps_is_scalar: int,
    sigma_is_scalar: int,
    dt: wp.float32,
    spacing: wp.array(dtype=wp.float32),
    offset_x: int,
    offset_y: int,
    offset_z: int,
    current_x: int,
    current_y: int,
    current_z: int,
    write_grad_eps: int,
    write_grad_sigma: int,
    write_grad_current: int,
):
    i, j, k = wp.tid()
    nx = electric_field.shape[1]
    ny = electric_field.shape[2]
    nz = electric_field.shape[3]

    i_prev = _periodic_prev(i, nx)
    j_prev = _periodic_prev(j, ny)
    k_prev = _periodic_prev(k, nz)

    eps_components = _material_components(
        eps_field, eps_scalar, eps_is_scalar, i, j, k, i_prev, j_prev, k_prev
    )
    sigma_components = _material_components(
        sigma_field, sigma_scalar, sigma_is_scalar, i, j, k, i_prev, j_prev, k_prev
    )

    curl = _curl_h(magnetic_field, i, j, k, i_prev, j_prev, k_prev)

    src_i = i - offset_x
    src_j = j - offset_y
    src_k = k - offset_z
    current_in_bounds = (
        src_i >= 0
        and src_i < current_x
        and src_j >= 0
        and src_j < current_y
        and src_k >= 0
        and src_k < current_z
    )

    j_x = 0.0
    j_y = 0.0
    j_z = 0.0
    if current_in_bounds:
        j_x = impressed_current[0, src_i, src_j, src_k]
        j_y = impressed_current[1, src_i, src_j, src_k]
        j_z = impressed_current[2, src_i, src_j, src_k]

    coeff_x = _coefficients(eps_components[0], sigma_components[0], dt, spacing[0])
    coeff_y = _coefficients(eps_components[1], sigma_components[1], dt, spacing[1])
    coeff_z = _coefficients(eps_components[2], sigma_components[2], dt, spacing[2])

    go_x = grad_output[0, i, j, k]
    go_y = grad_output[1, i, j, k]
    go_z = grad_output[2, i, j, k]

    grad_electric[0, i, j, k] = go_x * coeff_x[0]
    grad_electric[1, i, j, k] = go_y * coeff_y[0]
    grad_electric[2, i, j, k] = go_z * coeff_z[0]

    fx = go_x * coeff_x[1]
    fy = go_y * coeff_y[1]
    fz = go_z * coeff_z[1]
    _accumulate_grad_magnetic(grad_magnetic, i, j, k, i_prev, j_prev, k_prev, fx, fy, fz)

    if write_grad_current == 1 and current_in_bounds:
        grad_current[0, src_i, src_j, src_k] = go_x * coeff_x[2]
        grad_current[1, src_i, src_j, src_k] = go_y * coeff_y[2]
        grad_current[2, src_i, src_j, src_k] = go_z * coeff_z[2]

    if write_grad_eps == 1 or write_grad_sigma == 1:
        partial_x = _material_partials(
            eps_components[0],
            sigma_components[0],
            electric_field[0, i, j, k],
            curl[0],
            j_x,
            dt,
            spacing[0],
        )
        partial_y = _material_partials(
            eps_components[1],
            sigma_components[1],
            electric_field[1, i, j, k],
            curl[1],
            j_y,
            dt,
            spacing[1],
        )
        partial_z = _material_partials(
            eps_components[2],
            sigma_components[2],
            electric_field[2, i, j, k],
            curl[2],
            j_z,
            dt,
            spacing[2],
        )

        if write_grad_eps == 1:
            _scatter_avg_x(
                grad_eps, go_x * partial_x[0], i, j, k, i_prev, j_prev, k_prev
            )
            _scatter_avg_y(
                grad_eps, go_y * partial_y[0], i, j, k, i_prev, j_prev, k_prev
            )
            _scatter_avg_z(
                grad_eps, go_z * partial_z[0], i, j, k, i_prev, j_prev, k_prev
            )

        if write_grad_sigma == 1:
            _scatter_avg_x(
                grad_sigma, go_x * partial_x[1], i, j, k, i_prev, j_prev, k_prev
            )
            _scatter_avg_y(
                grad_sigma, go_y * partial_y[1], i, j, k, i_prev, j_prev, k_prev
            )
            _scatter_avg_z(
                grad_sigma, go_z * partial_z[1], i, j, k, i_prev, j_prev, k_prev
            )

# Launch one forward warp update for the selected material mode.
def _launch_warp_forward(
    electric_field: torch.Tensor,
    magnetic_field: torch.Tensor,
    eps: float | torch.Tensor,
    sigma_e: float | torch.Tensor,
    spacing: torch.Tensor,
    dt: float,
    impressed_current: torch.Tensor,
    impressed_current_offset: tuple[int, int, int],
    output: torch.Tensor,
) -> None:
    nx, ny, nz = electric_field.shape[1:]
    dim = (nx, ny, nz)

    wp_device, wp_stream = FunctionSpec.warp_launch_context(electric_field)

    wp_electric = wp.from_torch(electric_field.contiguous())
    wp_magnetic = wp.from_torch(magnetic_field.contiguous())
    wp_current = wp.from_torch(impressed_current.contiguous())
    wp_output = wp.from_torch(output, return_ctype=True)
    wp_spacing = wp.from_torch(spacing.contiguous())

    offset_x, offset_y, offset_z = impressed_current_offset
    current_x = int(impressed_current.shape[1])
    current_y = int(impressed_current.shape[2])
    current_z = int(impressed_current.shape[3])
    dt_value = float(dt)

    eps_is_scalar = isinstance(eps, (int, float))
    sigma_is_scalar = isinstance(sigma_e, (int, float))

    with wp.ScopedStream(wp_stream):
        if eps_is_scalar and sigma_is_scalar:
            wp.launch(
                kernel=_electric_field_update_kernel_scalar_scalar,
                dim=dim,
                inputs=[
                    wp_electric,
                    wp_magnetic,
                    wp_current,
                    wp_output,
                    float(eps),
                    float(sigma_e),
                    dt_value,
                    wp_spacing,
                    offset_x,
                    offset_y,
                    offset_z,
                    current_x,
                    current_y,
                    current_z,
                ],
                device=wp_device,
                stream=wp_stream,
            )
            return

        if eps_is_scalar and not sigma_is_scalar:
            sigma_field = _normalize_material_field(
                sigma_e,
                "sigma_e",
                (nx, ny, nz),
                electric_field.device,
            )
            wp_sigma = wp.from_torch(sigma_field.contiguous())
            wp.launch(
                kernel=_electric_field_update_kernel_scalar_sigma_field,
                dim=dim,
                inputs=[
                    wp_electric,
                    wp_magnetic,
                    wp_sigma,
                    wp_current,
                    wp_output,
                    float(eps),
                    dt_value,
                    wp_spacing,
                    offset_x,
                    offset_y,
                    offset_z,
                    current_x,
                    current_y,
                    current_z,
                ],
                device=wp_device,
                stream=wp_stream,
            )
            return

        if not eps_is_scalar and sigma_is_scalar:
            eps_field = _normalize_material_field(
                eps,
                "eps",
                (nx, ny, nz),
                electric_field.device,
            )
            wp_eps = wp.from_torch(eps_field.contiguous())
            wp.launch(
                kernel=_electric_field_update_kernel_eps_field_scalar,
                dim=dim,
                inputs=[
                    wp_electric,
                    wp_magnetic,
                    wp_eps,
                    wp_current,
                    wp_output,
                    float(sigma_e),
                    dt_value,
                    wp_spacing,
                    offset_x,
                    offset_y,
                    offset_z,
                    current_x,
                    current_y,
                    current_z,
                ],
                device=wp_device,
                stream=wp_stream,
            )
            return

        eps_field = _normalize_material_field(
            eps,
            "eps",
            (nx, ny, nz),
            electric_field.device,
        )
        sigma_field = _normalize_material_field(
            sigma_e,
            "sigma_e",
            (nx, ny, nz),
            electric_field.device,
        )
        wp_eps = wp.from_torch(eps_field.contiguous())
        wp_sigma = wp.from_torch(sigma_field.contiguous())
        wp.launch(
            kernel=_electric_field_update_kernel_eps_field_sigma_field,
            dim=dim,
            inputs=[
                wp_electric,
                wp_magnetic,
                wp_eps,
                wp_sigma,
                wp_current,
                wp_output,
                dt_value,
                wp_spacing,
                offset_x,
                offset_y,
                offset_z,
                current_x,
                current_y,
                current_z,
            ],
            device=wp_device,
            stream=wp_stream,
        )


# Launch one backward warp pass and return gradients for custom-op inputs.
def _launch_warp_backward(
    electric_field: torch.Tensor,
    magnetic_field: torch.Tensor,
    eps_field: torch.Tensor,
    sigma_e_field: torch.Tensor,
    impressed_current: torch.Tensor,
    grad_output: torch.Tensor,
    spacing: torch.Tensor,
    dt: float,
    eps_scalar: float,
    sigma_e_scalar: float,
    eps_is_scalar: bool,
    sigma_is_scalar: bool,
    impressed_current_offset: torch.Tensor,
    needs_input_grad: tuple[bool, ...],
) -> tuple[
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    need_grad_electric = needs_input_grad[0]
    need_grad_magnetic = needs_input_grad[1]
    need_grad_eps = (not eps_is_scalar) and needs_input_grad[2]
    need_grad_sigma = (not sigma_is_scalar) and needs_input_grad[3]
    need_grad_current = needs_input_grad[10] and impressed_current.numel() > 0

    if not any(
        (
            need_grad_electric,
            need_grad_magnetic,
            need_grad_eps,
            need_grad_sigma,
            need_grad_current,
        )
    ):
        return None, None, None, None, None

    device = electric_field.device
    spatial_shape = tuple(electric_field.shape[1:])
    dim = spatial_shape

    empty4 = torch.empty((3, 0, 0, 0), device=device, dtype=torch.float32)
    empty3 = torch.empty((0, 0, 0), device=device, dtype=torch.float32)

    # Materialize E/H gradient buffers unconditionally so specialized kernels
    # can write without per-element write guards.
    grad_electric = torch.zeros_like(electric_field, dtype=torch.float32)
    grad_magnetic = torch.zeros_like(magnetic_field, dtype=torch.float32)
    grad_eps = (
        torch.zeros_like(eps_field, dtype=torch.float32) if need_grad_eps else empty3
    )
    grad_sigma = (
        torch.zeros_like(sigma_e_field, dtype=torch.float32)
        if need_grad_sigma
        else empty3
    )
    grad_current = (
        torch.zeros_like(impressed_current, dtype=torch.float32)
        if need_grad_current
        else empty4
    )

    wp_device, wp_stream = FunctionSpec.warp_launch_context(electric_field)

    offset_x, offset_y, offset_z = tuple(
        int(v) for v in impressed_current_offset.detach().cpu().flatten().tolist()
    )
    use_current_input = int(impressed_current.numel() > 0)
    has_material_grads = need_grad_eps or need_grad_sigma

    wp_electric = wp.from_torch(electric_field.contiguous())
    wp_magnetic = wp.from_torch(magnetic_field.contiguous())
    wp_eps = wp.from_torch(eps_field.contiguous())
    wp_sigma = wp.from_torch(sigma_e_field.contiguous())
    wp_current = wp.from_torch(impressed_current.contiguous())
    wp_grad_output = wp.from_torch(grad_output.contiguous())
    wp_grad_electric = wp.from_torch(grad_electric, return_ctype=True)
    wp_grad_magnetic = wp.from_torch(grad_magnetic, return_ctype=True)
    wp_grad_eps = wp.from_torch(grad_eps, return_ctype=True)
    wp_grad_sigma = wp.from_torch(grad_sigma, return_ctype=True)
    wp_grad_current = wp.from_torch(grad_current, return_ctype=True)
    wp_spacing = wp.from_torch(spacing.contiguous())

    eps_scalar_value = float(eps_scalar)
    sigma_scalar_value = float(sigma_e_scalar)
    eps_is_scalar_flag = int(eps_is_scalar)
    sigma_is_scalar_flag = int(sigma_is_scalar)
    dt_value = float(dt)

    if has_material_grads and use_current_input == 1:
        wp.launch(
            kernel=_electric_field_update_backward_kernel_full_with_current,
            dim=dim,
            inputs=[
                wp_electric,
                wp_magnetic,
                wp_eps,
                wp_sigma,
                wp_current,
                wp_grad_output,
                wp_grad_electric,
                wp_grad_magnetic,
                wp_grad_eps,
                wp_grad_sigma,
                wp_grad_current,
                eps_scalar_value,
                sigma_scalar_value,
                eps_is_scalar_flag,
                sigma_is_scalar_flag,
                dt_value,
                wp_spacing,
                offset_x,
                offset_y,
                offset_z,
                int(impressed_current.shape[1]),
                int(impressed_current.shape[2]),
                int(impressed_current.shape[3]),
                int(need_grad_eps),
                int(need_grad_sigma),
                int(need_grad_current),
            ],
            device=wp_device,
            stream=wp_stream,
        )
    elif has_material_grads and use_current_input == 0:
        wp.launch(
            kernel=_electric_field_update_backward_kernel_full_no_current,
            dim=dim,
            inputs=[
                wp_electric,
                wp_magnetic,
                wp_eps,
                wp_sigma,
                wp_grad_output,
                wp_grad_electric,
                wp_grad_magnetic,
                wp_grad_eps,
                wp_grad_sigma,
                eps_scalar_value,
                sigma_scalar_value,
                eps_is_scalar_flag,
                sigma_is_scalar_flag,
                dt_value,
                wp_spacing,
                int(need_grad_eps),
                int(need_grad_sigma),
            ],
            device=wp_device,
            stream=wp_stream,
        )
    elif not has_material_grads and use_current_input == 1:
        wp.launch(
            kernel=_electric_field_update_backward_kernel_fields_with_current,
            dim=dim,
            inputs=[
                wp_electric,
                wp_magnetic,
                wp_eps,
                wp_sigma,
                wp_current,
                wp_grad_output,
                wp_grad_electric,
                wp_grad_magnetic,
                wp_grad_current,
                eps_scalar_value,
                sigma_scalar_value,
                eps_is_scalar_flag,
                sigma_is_scalar_flag,
                dt_value,
                wp_spacing,
                offset_x,
                offset_y,
                offset_z,
                int(impressed_current.shape[1]),
                int(impressed_current.shape[2]),
                int(impressed_current.shape[3]),
                int(need_grad_current),
            ],
            device=wp_device,
            stream=wp_stream,
        )
    else:
        wp.launch(
            kernel=_electric_field_update_backward_kernel_fields_no_current,
            dim=dim,
            inputs=[
                wp_electric,
                wp_magnetic,
                wp_eps,
                wp_sigma,
                wp_grad_output,
                wp_grad_electric,
                wp_grad_magnetic,
                eps_scalar_value,
                sigma_scalar_value,
                eps_is_scalar_flag,
                sigma_is_scalar_flag,
                dt_value,
                wp_spacing,
            ],
            device=wp_device,
            stream=wp_stream,
        )

    return (
        grad_electric if need_grad_electric else None,
        grad_magnetic if need_grad_magnetic else None,
        grad_eps if need_grad_eps else None,
        grad_sigma if need_grad_sigma else None,
        grad_current if need_grad_current else None,
    )


# Register out-of-place warp op with torch custom op.
@torch.library.custom_op("physicsnemo::electric_field_update_warp", mutates_args=())
def electric_field_update_impl(
    electric_field: torch.Tensor,
    magnetic_field: torch.Tensor,
    eps_field: torch.Tensor,
    sigma_e_field: torch.Tensor,
    eps_scalar: float,
    sigma_e_scalar: float,
    eps_is_scalar: bool,
    sigma_is_scalar: bool,
    spacing: torch.Tensor,
    dt: float,
    impressed_current: torch.Tensor,
    impressed_current_offset: torch.Tensor,
) -> torch.Tensor:
    output = torch.empty_like(electric_field)
    offset = tuple(
        int(v) for v in impressed_current_offset.detach().cpu().flatten().tolist()
    )
    eps_input: float | torch.Tensor = float(eps_scalar) if eps_is_scalar else eps_field
    sigma_input: float | torch.Tensor = (
        float(sigma_e_scalar) if sigma_is_scalar else sigma_e_field
    )
    _launch_warp_forward(
        electric_field=electric_field,
        magnetic_field=magnetic_field,
        eps=eps_input,
        sigma_e=sigma_input,
        spacing=spacing,
        dt=dt,
        impressed_current=impressed_current,
        impressed_current_offset=offset,
        output=output,
    )
    return output


# Provide fake-mode output metadata for torch compile.
@electric_field_update_impl.register_fake
def _(
    electric_field: torch.Tensor,
    magnetic_field: torch.Tensor,
    eps_field: torch.Tensor,
    sigma_e_field: torch.Tensor,
    eps_scalar: float,
    sigma_e_scalar: float,
    eps_is_scalar: bool,
    sigma_is_scalar: bool,
    spacing: torch.Tensor,
    dt: float,
    impressed_current: torch.Tensor,
    impressed_current_offset: torch.Tensor,
) -> torch.Tensor:
    _ = (
        magnetic_field,
        eps_field,
        sigma_e_field,
        eps_scalar,
        sigma_e_scalar,
        eps_is_scalar,
        sigma_is_scalar,
        spacing,
        dt,
        impressed_current,
        impressed_current_offset,
    )
    return torch.empty_like(electric_field)


# Save forward context used by autograd backward.
def setup_electric_field_update_context(
    ctx: torch.autograd.function.FunctionCtx,
    inputs: tuple,
    output: torch.Tensor,
) -> None:
    (
        electric_field,
        magnetic_field,
        eps_field,
        sigma_e_field,
        eps_scalar,
        sigma_e_scalar,
        eps_is_scalar,
        sigma_is_scalar,
        spacing,
        dt,
        impressed_current,
        impressed_current_offset,
    ) = inputs
    _ = output
    ctx.save_for_backward(
        electric_field,
        magnetic_field,
        eps_field,
        sigma_e_field,
        spacing,
        impressed_current,
        impressed_current_offset,
    )
    ctx.eps_scalar = float(eps_scalar)
    ctx.sigma_e_scalar = float(sigma_e_scalar)
    ctx.eps_is_scalar = bool(eps_is_scalar)
    ctx.sigma_is_scalar = bool(sigma_is_scalar)
    ctx.dt = float(dt)


# Warp-native backward for the custom op.
def backward_electric_field_update(
    ctx: torch.autograd.function.FunctionCtx,
    grad_output: torch.Tensor,
) -> tuple[
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    None,
    None,
    None,
    None,
    torch.Tensor | None,
    None,
    torch.Tensor | None,
    None,
]:
    (
        electric_field,
        magnetic_field,
        eps_field,
        sigma_e_field,
        spacing,
        impressed_current,
        impressed_current_offset,
    ) = ctx.saved_tensors

    if grad_output is None:
        return (None, None, None, None, None, None, None, None, None, None, None, None)

    (
        grad_electric,
        grad_magnetic,
        grad_eps,
        grad_sigma,
        grad_current,
    ) = _launch_warp_backward(
        electric_field=electric_field,
        magnetic_field=magnetic_field,
        eps_field=eps_field,
        sigma_e_field=sigma_e_field,
        impressed_current=impressed_current,
        grad_output=grad_output,
        spacing=spacing,
        dt=ctx.dt,
        eps_scalar=ctx.eps_scalar,
        sigma_e_scalar=ctx.sigma_e_scalar,
        eps_is_scalar=ctx.eps_is_scalar,
        sigma_is_scalar=ctx.sigma_is_scalar,
        impressed_current_offset=impressed_current_offset,
        needs_input_grad=ctx.needs_input_grad,
    )

    return (
        grad_electric,
        grad_magnetic,
        grad_eps,
        grad_sigma,
        None,
        None,
        None,
        None,
        None,
        None,
        grad_current,
        None,
    )


electric_field_update_impl.register_autograd(
    backward_electric_field_update,
    setup_context=setup_electric_field_update_context,
)


# Public warp entry point used by the FunctionSpec.
def electric_field_update_warp(
    electric_field: torch.Tensor,
    magnetic_field: torch.Tensor,
    eps: float | torch.Tensor,
    sigma_e: float | torch.Tensor,
    spacing: torch.Tensor | Sequence[float],
    dt: float,
    impressed_current: torch.Tensor | None = None,
    impressed_current_offset: torch.Tensor | Sequence[int] = (0, 0, 0),
    inplace: bool = False,
) -> torch.Tensor:
    _validate_common_inputs(
        electric_field,
        magnetic_field,
        eps,
        sigma_e,
        spacing,
        impressed_current,
        inplace,
    )

    if not electric_field.is_contiguous():
        raise ValueError("electric_field must be contiguous for the warp implementation")
    if not magnetic_field.is_contiguous():
        raise ValueError("magnetic_field must be contiguous for the warp implementation")
    if isinstance(eps, torch.Tensor) and not eps.is_contiguous():
        raise ValueError("eps tensor must be contiguous for the warp implementation")
    if isinstance(sigma_e, torch.Tensor) and not sigma_e.is_contiguous():
        raise ValueError("sigma_e tensor must be contiguous for the warp implementation")
    if isinstance(spacing, torch.Tensor) and not spacing.is_contiguous():
        raise ValueError("spacing tensor must be contiguous for the warp implementation")
    if impressed_current is not None and not impressed_current.is_contiguous():
        raise ValueError(
            "impressed_current must be contiguous for the warp implementation"
        )

    spacing_tensor = _as_spacing_tensor(
        spacing,
        device=electric_field.device,
        dtype=electric_field.dtype,
    ).contiguous()
    offset = _normalize_offset(impressed_current_offset)

    if impressed_current is None:
        impressed_current_tensor = torch.empty(
            (3, 0, 0, 0),
            device=electric_field.device,
            dtype=torch.float32,
        )
    else:
        impressed_current_tensor = impressed_current

    if inplace:
        _launch_warp_forward(
            electric_field=electric_field,
            magnetic_field=magnetic_field,
            eps=eps,
            sigma_e=sigma_e,
            spacing=spacing_tensor,
            dt=dt,
            impressed_current=impressed_current_tensor,
            impressed_current_offset=offset,
            output=electric_field,
        )
        return electric_field

    spatial_shape = tuple(electric_field.shape[1:])
    empty_material = torch.empty(
        (0, 0, 0),
        device=electric_field.device,
        dtype=torch.float32,
    )
    eps_is_scalar = isinstance(eps, (int, float))
    sigma_is_scalar = isinstance(sigma_e, (int, float))
    eps_field = (
        empty_material
        if eps_is_scalar
        else _normalize_material_field(
            eps, "eps", spatial_shape, electric_field.device
        ).contiguous()
    )
    sigma_field = (
        empty_material
        if sigma_is_scalar
        else _normalize_material_field(
            sigma_e, "sigma_e", spatial_shape, electric_field.device
        ).contiguous()
    )
    offset_tensor = torch.tensor(offset, device=electric_field.device, dtype=torch.int32)

    return electric_field_update_impl(
        electric_field,
        magnetic_field,
        eps_field,
        sigma_field,
        float(eps) if eps_is_scalar else 0.0,
        float(sigma_e) if sigma_is_scalar else 0.0,
        eps_is_scalar,
        sigma_is_scalar,
        spacing_tensor,
        float(dt),
        impressed_current_tensor,
        offset_tensor,
    )
