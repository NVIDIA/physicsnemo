# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
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

try:
    import warp as wp
except ImportError:
    print(
        """NVIDIA WARP is required for this datapipe. This package is under the 
NVIDIA Source Code License (NVSCL). To install use:

pip install warp-lang
"""
    )
    raise SystemExit(1)

from .indexing import (
    index_periodic_edges_batched_2d,
    index_vec2_periodic_edges_batched_2d,
)


@wp.func
def extrapolate_to_face_2d(
    f: float, f_dx: float, f_dy: float, dx: float
):  # pragma: no cover
    """Extrapolate cell values to edges of face

    Parameters
    ----------
    f : float
        Cell value
    f_dx : float
        X derivative of cell value
    f_dy : float
        Y derivative of cell value
    dx : float
        Cell size

    Returns
    -------
    wp.vec4
        (value on left x, value on right x, value left y, value right y)
    """
    f_xl = f - f_dx * (dx / 2.0)
    f_xr = f + f_dx * (dx / 2.0)
    f_yl = f - f_dy * (dx / 2.0)
    f_yr = f + f_dy * (dx / 2.0)
    return wp.vec4(f_xl, f_xr, f_yl, f_yr)


@wp.func
def apply_flux_2d(
    f: float,
    flux_f_xl_dx: float,
    flux_f_xr_dx: float,
    flux_f_yl_dy: float,
    flux_f_yr_dy: float,
    dx: float,
    dt: float,
):  # pragma: no cover
    """Apply flux to cell

    Parameters
    ----------
    f : float
        Cell value
    flux_f_xl_dx : float
        Left x flux
    flux_f_xr_dx : float
        Right x flux
    flux_f_yl_dy : float
        Left y flux
    flux_f_yr_dy : float
        Right y flux
    dx : float
        Cell size
    dt : float
        Time step size

    Returns
    -------
    float
        Cell value with added flux
    """
    f += -dt * dx * flux_f_xl_dx
    f += dt * dx * flux_f_xr_dx
    f += -dt * dx * flux_f_yl_dy
    f += dt * dx * flux_f_yr_dy
    return f


@wp.func
def apply_flux_vec2_2d(
    f: wp.vec2,
    flux_f_xl_dx: wp.vec2,
    flux_f_xr_dx: wp.vec2,
    flux_f_yl_dy: wp.vec2,
    flux_f_yr_dy: wp.vec2,
    dx: float,
    dt: float,
):  # pragma: no cover
    """Apply flux on cell with vector value

    Parameters
    ----------
    f : wp.vec2
        Cell vector value
    flux_f_xl_dx : wp.vec2
        Vector flux in x left
    flux_f_xr_dx : wp.vec2
        Vector flux in x right
    flux_f_yl_dy : wp.vec2
        Vector flux in y left
    flux_f_yr_dy : wp.vec2
        Vector flux in y right
    dx : float
        Cell size
    dt : float
        Time step size

    Returns
    -------
    wp.vec2
        Vector cell value with added flux
    """
    f += -dt * dx * flux_f_xl_dx
    f += dt * dx * flux_f_xr_dx
    f += -dt * dx * flux_f_yl_dy
    f += dt * dx * flux_f_yr_dy
    return f


@wp.func
def euler_flux_2d(
    rho_l: float,
    rho_r: float,
    vx_l: float,
    vx_r: float,
    vy_l: float,
    vy_r: float,
    p_l: float,
    p_r: float,
    gamma: float,
):  # pragma: no cover
    """Compute Euler flux

    Parameters
    ----------
    rho_l : float
        Density left
    rho_r : float
        Density right
    vx_l : float
        X velocity left
    vx_r : float
        X velocity right
    vy_l : float
        Y velocity left
    vy_r : float
        Y velocity right
    p_l : float
        Pressure left
    p_r : float
        Pressure right
    gamma : float
        Gas constant

    Returns
    -------
    wp.vec4
        Vector containing mass, momentum x, momentum y, and energy flux.
    """
    # get energies
    e_l = p_l / (gamma - 1.0) + 0.5 * rho_l * (vx_l * vx_l + vy_l * vy_l)
    e_r = p_r / (gamma - 1.0) + 0.5 * rho_r * (vx_r * vx_r + vy_r * vy_r)

    # averaged states
    rho_ave = 0.5 * (rho_l + rho_r)
    momx_ave = 0.5 * (rho_l * vx_l + rho_r * vx_r)
    momy_ave = 0.5 * (rho_l * vy_l + rho_r * vy_r)
    e_ave = 0.5 * (e_l + e_r)
    p_ave = (gamma - 1.0) * (
        e_ave - 0.5 * (momx_ave * momx_ave + momy_ave * momy_ave) / rho_ave
    )

    # compute fluxes
    flux_mass = momx_ave
    flux_momx = momx_ave * momx_ave / rho_ave + p_ave
    flux_momy = momx_ave * momy_ave / rho_ave
    flux_e = (e_ave + p_ave) * momx_ave / rho_ave

    # compute wavespeed
    c_l = wp.sqrt(gamma * p_l / rho_l) + wp.abs(vx_l)
    c_r = wp.sqrt(gamma * p_r / rho_r) + wp.abs(vx_r)
    c = wp.max(c_l, c_r)

    # add stabilizing diffusion term
    flux_mass -= c * 0.5 * (rho_l - rho_r)
    flux_momx -= c * 0.5 * (rho_l * vx_l - rho_r * vx_r)
    flux_momy -= c * 0.5 * (rho_l * vy_l - rho_r * vy_r)
    flux_e -= c * 0.5 * (e_l - e_r)

    return wp.vec4(flux_mass, flux_momx, flux_momy, flux_e)


@wp.kernel
def euler_primitive_to_conserved_batched_2d(
    rho: wp.array3d(dtype=float),
    vel: wp.array3d(dtype=wp.vec2),
    p: wp.array3d(dtype=float),
    mass: wp.array3d(dtype=float),
    mom: wp.array3d(dtype=wp.vec2),
    e: wp.array3d(dtype=float),
    gamma: float,
    vol: float,
    lx: int,
    ly: int,
):  # pragma: no cover
    """Primitive Euler to conserved values

    Parameters
    ----------
    rho : wp.array3d
        Density
    vel : wp.array3d
        Velocity
    p : wp.array3d
        Pressure
    mass : wp.array3d
        Mass
    mom : wp.array3d
        Momentum
    e : wp.array3d
        Energy
    gamma : float
        Gas constant
    vol : float
        Volume of cell
    lx : int
        Grid size x dim
    ly : int
        Grid size y dim
    """

    # get index
    b, i, j = wp.tid()

    # get conserve values
    rho_i_j = index_periodic_edges_batched_2d(rho, b, i, j, lx, ly)
    vel_i_j = index_vec2_periodic_edges_batched_2d(vel, b, i, j, lx, ly)
    p_i_j = index_periodic_edges_batched_2d(p, b, i, j, lx, ly)

    # get primitive values
    mass_i_j = rho_i_j * vol
    mom_i_j = vel_i_j * rho_i_j * vol
    e_i_j = (
        p_i_j / (gamma - 1.0)
        + 0.5 * rho_i_j * (vel_i_j[0] * vel_i_j[0] + vel_i_j[1] * vel_i_j[1])
    ) * vol

    # set values
    mass[b, i, j] = mass_i_j
    mom[b, i, j] = mom_i_j
    e[b, i, j] = e_i_j


@wp.kernel
def euler_conserved_to_primitive_batched_2d(
    mass: wp.array3d(dtype=float),
    mom: wp.array3d(dtype=wp.vec2),
    e: wp.array3d(dtype=float),
    rho: wp.array3d(dtype=float),
    vel: wp.array3d(dtype=wp.vec2),
    p: wp.array3d(dtype=float),
    gamma: float,
    vol: float,
    lx: int,
    ly: int,
):  # pragma: no cover
    """Conserved Euler to primitive value

    Parameters
    ----------
    mass : wp.array3d
        Mass
    mom : wp.array3d
        Momentum
    e : wp.array3d
        Energy
    rho : wp.array3d
        Density
    vel : wp.array3d
        Velocity
    p : wp.array3d
        Pressure
    gamma : float
        Gas constant
    vol : float
        Cell volume
    lx : int
        Grid size X dim
    ly : int
        Grid size Y dim
    """

    # get index
    b, i, j = wp.tid()

    # get conserve values
    mass_i_j = index_periodic_edges_batched_2d(mass, b, i, j, lx, ly)
    mom_i_j = index_vec2_periodic_edges_batched_2d(mom, b, i, j, lx, ly)
    e_i_j = index_periodic_edges_batched_2d(e, b, i, j, lx, ly)

    # get primitive values
    rho_i_j = mass_i_j / vol
    vel_i_j = mom_i_j / rho_i_j / vol
    p_i_j = (
        e_i_j / vol
        - 0.5 * rho_i_j * (vel_i_j[0] * vel_i_j[0] + vel_i_j[1] * vel_i_j[1])
    ) * (gamma - 1.0)

    # set values
    rho[b, i, j] = rho_i_j
    vel[b, i, j] = vel_i_j
    p[b, i, j] = p_i_j


@wp.kernel
def euler_extrapolation_batched_2d(
    rho: wp.array3d(dtype=float),
    vel: wp.array3d(dtype=wp.vec2),
    p: wp.array3d(dtype=float),
    rho_xl: wp.array3d(dtype=float),
    rho_xr: wp.array3d(dtype=float),
    rho_yl: wp.array3d(dtype=float),
    rho_yr: wp.array3d(dtype=float),
    vel_xl: wp.array3d(dtype=wp.vec2),
    vel_xr: wp.array3d(dtype=wp.vec2),
    vel_yl: wp.array3d(dtype=wp.vec2),
    vel_yr: wp.array3d(dtype=wp.vec2),
    p_xl: wp.array3d(dtype=float),
    p_xr: wp.array3d(dtype=float),
    p_yl: wp.array3d(dtype=float),
    p_yr: wp.array3d(dtype=float),
    gamma: float,
    dx: float,
    dt: float,
    lx: int,
    ly: int,
):  # pragma: no cover
    """Extrapolate Euler values to edges

    Parameters
    ----------
    rho : wp.array3d
        Density
    vel : wp.array3d
        Velocity
    p : wp.array3d
        Pressure
    rho_xl : wp.array3d
        Density x left
    rho_xr : wp.array3d
        Density x right
    rho_yl : wp.array3d
        Density y left
    rho_yr : wp.array3d
        Density y right
    vel_xl : wp.array3d
        Velocity x left
    vel_xr : wp.array3d
        Velocity x right
    vel_yl : wp.array3d
        Velocity y left
    vel_yr : wp.array3d
        Velocity y right
    p_xl : wp.array3d
        Pressure x left
    p_xr : wp.array3d
        Pressure x right
    p_yl : wp.array3d
        Pressure y left
    p_yr : wp.array3d
        Pressure y right
    gamma : float
        Gas constant
    dx : float
        Cell size
    dt : float
        Time step size
    lx : int
        Grid size x
    ly : int
        Grid size y
    """

    # get index
    b, i, j = wp.tid()

    # get rho stensil
    rho_1_1 = index_periodic_edges_batched_2d(rho, b, i, j, lx, ly)
    rho_2_1 = index_periodic_edges_batched_2d(rho, b, i + 1, j, lx, ly)
    rho_1_2 = index_periodic_edges_batched_2d(rho, b, i, j + 1, lx, ly)
    rho_0_1 = index_periodic_edges_batched_2d(rho, b, i - 1, j, lx, ly)
    rho_1_0 = index_periodic_edges_batched_2d(rho, b, i, j - 1, lx, ly)

    # get momentum stensil
    vel_1_1 = index_vec2_periodic_edges_batched_2d(vel, b, i, j, lx, ly)
    vel_2_1 = index_vec2_periodic_edges_batched_2d(vel, b, i + 1, j, lx, ly)
    vel_1_2 = index_vec2_periodic_edges_batched_2d(vel, b, i, j + 1, lx, ly)
    vel_0_1 = index_vec2_periodic_edges_batched_2d(vel, b, i - 1, j, lx, ly)
    vel_1_0 = index_vec2_periodic_edges_batched_2d(vel, b, i, j - 1, lx, ly)

    # get energy stensil
    p_1_1 = index_periodic_edges_batched_2d(p, b, i, j, lx, ly)
    p_2_1 = index_periodic_edges_batched_2d(p, b, i + 1, j, lx, ly)
    p_1_2 = index_periodic_edges_batched_2d(p, b, i, j + 1, lx, ly)
    p_0_1 = index_periodic_edges_batched_2d(p, b, i - 1, j, lx, ly)
    p_1_0 = index_periodic_edges_batched_2d(p, b, i, j - 1, lx, ly)

    # compute density grad
    rho_dx = (rho_2_1 - rho_0_1) / (2.0 * dx)
    rho_dy = (rho_1_2 - rho_1_0) / (2.0 * dx)

    # compute velocity grad
    vel_dx = (vel_2_1 - vel_0_1) / (2.0 * dx)
    vel_dy = (vel_1_2 - vel_1_0) / (2.0 * dx)

    # compute pressure grad
    p_dx = (p_2_1 - p_0_1) / (2.0 * dx)
    p_dy = (p_1_2 - p_1_0) / (2.0 * dx)

    # extrapolate half time step density
    rho_prime = rho_1_1 - 0.5 * dt * (
        vel_1_1[0] * rho_dx
        + rho_1_1 * vel_dx[0]
        + vel_1_1[1] * rho_dy
        + rho_1_1 * vel_dy[1]
    )
    vx_prime = vel_1_1[0] - 0.5 * dt * (
        vel_1_1[0] * vel_dx[0] + vel_1_1[1] * vel_dy[0] + (1.0 / rho_1_1) * p_dx
    )
    vy_prime = vel_1_1[1] - 0.5 * dt * (
        vel_1_1[0] * vel_dx[1] + vel_1_1[1] * vel_dy[1] + (1.0 / rho_1_1) * p_dy
    )
    p_prime = p_1_1 - 0.5 * dt * (
        gamma * p_1_1 * (vel_dx[0] + vel_dy[1]) + vel_1_1[0] * p_dx + vel_1_1[1] * p_dy
    )

    # extrapolate in space to face centers
    rho_space_extra = extrapolate_to_face_2d(rho_prime, rho_dx, rho_dy, dx)
    vx_space_extra = extrapolate_to_face_2d(vx_prime, vel_dx[0], vel_dy[0], dx)
    vy_space_extra = extrapolate_to_face_2d(vy_prime, vel_dx[1], vel_dy[1], dx)
    p_space_extra = extrapolate_to_face_2d(p_prime, p_dx, p_dy, dx)

    # store values
    rho_xl[b, i, j] = rho_space_extra[0]
    rho_xr[b, i, j] = rho_space_extra[1]
    rho_yl[b, i, j] = rho_space_extra[2]
    rho_yr[b, i, j] = rho_space_extra[3]
    vel_xl[b, i, j] = wp.vec2(vx_space_extra[0], vy_space_extra[0])
    vel_xr[b, i, j] = wp.vec2(vx_space_extra[1], vy_space_extra[1])
    vel_yl[b, i, j] = wp.vec2(vx_space_extra[2], vy_space_extra[2])
    vel_yr[b, i, j] = wp.vec2(vx_space_extra[3], vy_space_extra[3])
    p_xl[b, i, j] = p_space_extra[0]
    p_xr[b, i, j] = p_space_extra[1]
    p_yl[b, i, j] = p_space_extra[2]
    p_yr[b, i, j] = p_space_extra[3]


@wp.kernel
def euler_get_flux_batched_2d(
    rho_xl: wp.array3d(dtype=float),
    rho_xr: wp.array3d(dtype=float),
    rho_yl: wp.array3d(dtype=float),
    rho_yr: wp.array3d(dtype=float),
    vel_xl: wp.array3d(dtype=wp.vec2),
    vel_xr: wp.array3d(dtype=wp.vec2),
    vel_yl: wp.array3d(dtype=wp.vec2),
    vel_yr: wp.array3d(dtype=wp.vec2),
    p_xl: wp.array3d(dtype=float),
    p_xr: wp.array3d(dtype=float),
    p_yl: wp.array3d(dtype=float),
    p_yr: wp.array3d(dtype=float),
    mass_flux_x: wp.array3d(dtype=float),
    mass_flux_y: wp.array3d(dtype=float),
    mom_flux_x: wp.array3d(dtype=wp.vec2),
    mom_flux_y: wp.array3d(dtype=wp.vec2),
    e_flux_x: wp.array3d(dtype=float),
    e_flux_y: wp.array3d(dtype=float),
    gamma: float,
    lx: int,
    ly: int,
):  # pragma: no cover
    """Use extrapolated Euler values to compute fluxes

    Parameters
    ----------
    rho_xl : wp.array3d
        Density x left
    rho_xr : wp.array3d
        Density x right
    rho_yl : wp.array3d
        Density y left
    rho_yr : wp.array3d
        Density y right
    vel_xl : wp.array3d
        Velocity x left
    vel_xr : wp.array3d
        Velocity x right
    vel_yl : wp.array3d
        Velocity y left
    vel_yr : wp.array3d
        Velocity y right
    p_xl : wp.array3d
        Pressure x left
    p_xr : wp.array3d
        Pressure x right
    p_yl : wp.array3d
        Pressure y left
    p_yr : wp.array3d
        Pressure y right
    mass_flux_x : wp.array3d
        Mass flux x
    mass_flux_y : wp.array3d
        Mass flux y
    mom_flux_x : wp.array3d
        Momentum flux x
    mom_flux_y : wp.array3d
        Momentum flux y
    e_flux_x : wp.array3d
        Energy flux x
    e_flux_y : wp.array3d
        Energy flux y
    gamma : float
        Gas constant
    lx : int
        Grid size x
    ly : int
        Grid size y
    """

    # get index
    b, i, j = wp.tid()

    # get space extrapolation for faces
    rho_xl_1 = index_periodic_edges_batched_2d(rho_xl, b, i + 1, j, lx, ly)
    rho_xr_0 = index_periodic_edges_batched_2d(rho_xr, b, i, j, lx, ly)
    rho_yl_1 = index_periodic_edges_batched_2d(rho_yl, b, i, j + 1, lx, ly)
    rho_yr_0 = index_periodic_edges_batched_2d(rho_yr, b, i, j, lx, ly)
    vel_xl_1 = index_vec2_periodic_edges_batched_2d(vel_xl, b, i + 1, j, lx, ly)
    vel_xr_0 = index_vec2_periodic_edges_batched_2d(vel_xr, b, i, j, lx, ly)
    vel_yl_1 = index_vec2_periodic_edges_batched_2d(vel_yl, b, i, j + 1, lx, ly)
    vel_yr_0 = index_vec2_periodic_edges_batched_2d(vel_yr, b, i, j, lx, ly)
    p_xl_1 = index_periodic_edges_batched_2d(p_xl, b, i + 1, j, lx, ly)
    p_xr_0 = index_periodic_edges_batched_2d(p_xr, b, i, j, lx, ly)
    p_yl_1 = index_periodic_edges_batched_2d(p_yl, b, i, j + 1, lx, ly)
    p_yr_0 = index_periodic_edges_batched_2d(p_yr, b, i, j, lx, ly)

    # compute fluxes
    flux_x = euler_flux_2d(
        rho_xl_1,
        rho_xr_0,
        vel_xl_1[0],
        vel_xr_0[0],
        vel_xl_1[1],
        vel_xr_0[1],
        p_xl_1,
        p_xr_0,
        gamma,
    )
    flux_y = euler_flux_2d(
        rho_yl_1,
        rho_yr_0,
        vel_yl_1[1],
        vel_yr_0[1],
        vel_yl_1[0],
        vel_yr_0[0],
        p_yl_1,
        p_yr_0,
        gamma,
    )

    # set values
    mass_flux_x[b, i, j] = flux_x[0]
    mass_flux_y[b, i, j] = flux_y[0]
    mom_flux_x[b, i, j] = wp.vec2(flux_x[1], flux_x[2])
    mom_flux_y[b, i, j] = wp.vec2(flux_y[2], flux_y[1])
    e_flux_x[b, i, j] = flux_x[3]
    e_flux_y[b, i, j] = flux_y[3]


@wp.kernel
def euler_apply_flux_batched_2d(
    mass_flux_x: wp.array3d(dtype=float),
    mass_flux_y: wp.array3d(dtype=float),
    mom_flux_x: wp.array3d(dtype=wp.vec2),
    mom_flux_y: wp.array3d(dtype=wp.vec2),
    e_flux_x: wp.array3d(dtype=float),
    e_flux_y: wp.array3d(dtype=float),
    mass: wp.array3d(dtype=float),
    mom: wp.array3d(dtype=wp.vec2),
    e: wp.array3d(dtype=float),
    dx: float,
    dt: float,
    lx: int,
    ly: int,
):  # pragma: no cover
    """Apply fluxes to Euler values

    Parameters
    ----------
    mass_flux_x : wp.array3d
        Mass flux X
    mass_flux_y : wp.array3d
        Mass flux Y
    mom_flux_x : wp.array3d
        Momentum flux X
    mom_flux_y : wp.array3d
        Momentum flux Y
    e_flux_x : wp.array3d
        Energy flux X
    e_flux_y : wp.array3d
        Energy flux Y
    mass : wp.array3d
        Mass
    mom : wp.array3d
        Momentum
    e : wp.array3d
        Energy
    dx : float
        Cell size
    dt : float
        Time step size
    lx : int
        Grid size x
    ly : int
        Grid size y
    """

    # get index
    b, i, j = wp.tid()

    # get new mass
    mass_1 = index_periodic_edges_batched_2d(mass, b, i, j, lx, ly)
    mass_flux_x_1 = index_periodic_edges_batched_2d(mass_flux_x, b, i, j, lx, ly)
    mass_flux_x_0 = index_periodic_edges_batched_2d(mass_flux_x, b, i - 1, j, lx, ly)
    mass_flux_y_1 = index_periodic_edges_batched_2d(mass_flux_y, b, i, j, lx, ly)
    mass_flux_y_0 = index_periodic_edges_batched_2d(mass_flux_y, b, i, j - 1, lx, ly)
    new_mass = apply_flux_2d(
        mass_1, mass_flux_x_1, mass_flux_x_0, mass_flux_y_1, mass_flux_y_0, dx, dt
    )

    # get new mom
    mom_1 = index_vec2_periodic_edges_batched_2d(mom, b, i, j, lx, ly)
    mom_flux_x_1 = index_vec2_periodic_edges_batched_2d(mom_flux_x, b, i, j, lx, ly)
    mom_flux_x_0 = index_vec2_periodic_edges_batched_2d(mom_flux_x, b, i - 1, j, lx, ly)
    mom_flux_y_1 = index_vec2_periodic_edges_batched_2d(mom_flux_y, b, i, j, lx, ly)
    mom_flux_y_0 = index_vec2_periodic_edges_batched_2d(mom_flux_y, b, i, j - 1, lx, ly)
    new_mom = apply_flux_vec2_2d(
        mom_1, mom_flux_x_1, mom_flux_x_0, mom_flux_y_1, mom_flux_y_0, dx, dt
    )

    # get new energy
    e_1 = index_periodic_edges_batched_2d(e, b, i, j, lx, ly)
    e_flux_x_1 = index_periodic_edges_batched_2d(e_flux_x, b, i, j, lx, ly)
    e_flux_x_0 = index_periodic_edges_batched_2d(e_flux_x, b, i - 1, j, lx, ly)
    e_flux_y_1 = index_periodic_edges_batched_2d(e_flux_y, b, i, j, lx, ly)
    e_flux_y_0 = index_periodic_edges_batched_2d(e_flux_y, b, i, j - 1, lx, ly)
    new_e = apply_flux_2d(e_1, e_flux_x_1, e_flux_x_0, e_flux_y_1, e_flux_y_0, dx, dt)

    # set values
    mass[b, i, j] = new_mass
    mom[b, i, j] = new_mom
    e[b, i, j] = new_e


@wp.kernel
def initialize_kelvin_helmoltz_batched_2d(
    rho: wp.array3d(dtype=float),
    vel: wp.array3d(dtype=wp.vec2),
    p: wp.array3d(dtype=float),
    w: wp.array2d(dtype=float),
    sigma: float,
    lx: float,
    ly: float,
    nr_freq: int,
):  # pragma: no cover
    """Initialize state for Kelvin Helmoltz Instability

    Parameters
    ----------
    rho : wp.array3d
        Density
    vel : wp.array3d
        Velocity
    p : wp.array3d
        Pressure
    w : wp.array2d
        Perturbation frequency amplitude
    sigma : float
        Perturbation sigma
    vol : float
        Volume of cell
    gamma : float
        Gas constant
    lx : float
        Grid size x
    ly : float
        Grid size y
    nr_freq : int
        Number of frequencies in perturbation
    """

    # get cell coords
    b, i, j = wp.tid()
    x = wp.float(i) / wp.float(lx)
    y = wp.float(j) / wp.float(ly)

    # initial flow bands
    if wp.abs(y - 0.5) < 0.25:
        ux = 0.5
        r = 2.0
    else:
        ux = -0.5
        r = 1.0

    # perturbation
    uy = wp.float32(0.0)
    for f in range(nr_freq):
        ff = wp.float32(f + 1)
        uy += (
            ff
            * w[b, f]
            * wp.sin(4.0 * 3.14159 * x * ff)
            * (
                wp.exp(-(y - 0.25) * (y - 0.25) / (2.0 * sigma * sigma))
                + wp.exp(-(y - 0.75) * (y - 0.75) / (2.0 * sigma * sigma))
            )
        )
    u = wp.vec2(ux, uy)

    # set values
    rho[b, i, j] = r
    vel[b, i, j] = u
    p[b, i, j] = 2.5
