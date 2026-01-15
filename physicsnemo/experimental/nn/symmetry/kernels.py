# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
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


import warp as wp


@wp.func
def wp_silu(x: float) -> float:
    r"""Compute the SiLU activation on a single element.

    Parameters
    ----------
    x : float
        Value to evaluate SiLU(x) at.

    Returns
    float
        If the evaluated value is infinite, returns ``wp.nan``.
        Otherwise, return SiLU(x).
    """
    exp_x = wp.exp(-x)
    output = x / (x + x * exp_x)
    if wp.isinf(output):
        return wp.nan
    return output


@wp.func
def safe_acos(x: float) -> float:
    """Wrapped ``acos`` that is numerically stable.

    Clamps values of ``x`` to be within [-1, 1] before
    computing ``acos``.

    Parameters
    ----------
    x : float
        Value to clamp and compute with.

    Returns
    -------
    float
        Value of ``acos`` after clamping.

    """
    return wp.acos(wp.clamp(x, -1.0, 1.0))


# =============================================================================
# Wigner D-matrix kernels for l=0 to l=5 (Real part only)
# =============================================================================
# These functions compute the REAL PART of Wigner D-matrices D^l_{m,m'}(alpha, beta, gamma)
# using the convention: D = exp(+i*m*alpha) * d * exp(+i*m'*gamma)
# where d^l_{m,m'}(beta) is the real-valued small Wigner d-matrix.
#
# The real part is: Re(D) = d * cos(m*alpha + m'*gamma)
#
# Matrix indices go from m=l (row 0) to m=-l (row 2l), same for m' in columns.
# Output is stored in row-major order as a 1D array of floats.
#
# Note on usage:
# `wp.func` are not intended to be called directly, and instead will need
# to be wrapped in `wp.kernels` and launched appropriately.


def wigner_d_matrix_size(l: int) -> int:
    """Compute the number of elements in a Wigner D-matrix for angular degree l.

    The Wigner D-matrix for angular degree l is a (2l+1) x (2l+1) matrix,
    so the total number of elements is (2l+1)^2.

    Parameters
    ----------
    l : int
        Angular momentum quantum number (non-negative integer)

    Returns
    -------
    int
        The number of elements in the matrix: (2l+1)^2

    Examples
    --------
    >>> wigner_d_matrix_size(0)
    1
    >>> wigner_d_matrix_size(1)
    9
    >>> wigner_d_matrix_size(2)
    25
    >>> wigner_d_matrix_size(5)
    121
    """
    dim = 2 * l + 1
    return dim * dim


def wigner_d_matrix_shape(l: int) -> tuple[int, int]:
    """Compute the shape of a Wigner D-matrix for angular degree l.

    The Wigner D-matrix for angular degree l is a (2l+1) x (2l+1) matrix.

    Parameters
    ----------
    l : int
        Angular momentum quantum number (non-negative integer)

    Returns
    -------
    tuple[int, int]
        The shape of the matrix: (2l+1, 2l+1)

    Examples
    --------
    >>> wigner_d_matrix_shape(0)
    (1, 1)
    >>> wigner_d_matrix_shape(1)
    (3, 3)
    >>> wigner_d_matrix_shape(2)
    (5, 5)
    >>> wigner_d_matrix_shape(5)
    (11, 11)
    """
    dim = 2 * l + 1
    return (dim, dim)


@wp.func
def wigner_d_l0(
    alpha: float,
    beta: float,
    gamma: float,
    D: wp.array2d(dtype=float),
):
    """Compute the real part of Wigner D-matrix for l=0.

    The 1x1 matrix is stored in D.

    Parameters
    ----------
    alpha, beta, gamma : float
        Euler angles (z-y-z convention)
    D : wp.array2d(dtype=float)
        Output 2D array of shape (1, 1) to store the real part of matrix elements
    """
    # D^0_{0,0} = 1 (real)
    D[0, 0] = 1.0


@wp.func
def wigner_d_l1(
    alpha: float,
    beta: float,
    gamma: float,
    D: wp.array2d(dtype=float),
):
    """Compute the real part of Wigner D-matrix for l=1.

    The 3x3 matrix is stored in D.

    Parameters
    ----------
    alpha, beta, gamma : float
        Euler angles (z-y-z convention)
    D : wp.array2d(dtype=float)
        Output 2D array of shape (3, 3) to store the real part of matrix elements
    """
    # Precompute half-angle trig
    c = wp.cos(beta / 2.0)
    s = wp.sin(beta / 2.0)
    cb = wp.cos(beta)
    sb = wp.sin(beta)
    sqrt2 = wp.sqrt(2.0)

    # Small d-matrix elements (real-valued)
    d_1_1 = c * c
    d_1_0 = sqrt2 * sb / 2.0
    d_1_n1 = s * s
    d_0_1 = -sqrt2 * sb / 2.0
    d_0_0 = cb
    d_0_n1 = sqrt2 * sb / 2.0
    d_n1_1 = s * s
    d_n1_0 = -sqrt2 * sb / 2.0
    d_n1_n1 = c * c

    # Real part: d * cos(m*alpha + m'*gamma)
    # Row 0: m=1
    D[0, 0] = d_1_1 * wp.cos(alpha + gamma)  # m=1, m'=1
    D[0, 1] = d_1_0 * wp.cos(alpha)  # m=1, m'=0
    D[0, 2] = d_1_n1 * wp.cos(alpha - gamma)  # m=1, m'=-1

    # Row 1: m=0
    D[1, 0] = d_0_1 * wp.cos(gamma)  # m=0, m'=1
    D[1, 1] = d_0_0  # m=0, m'=0
    D[1, 2] = d_0_n1 * wp.cos(-gamma)  # m=0, m'=-1

    # Row 2: m=-1
    D[2, 0] = d_n1_1 * wp.cos(-alpha + gamma)  # m=-1, m'=1
    D[2, 1] = d_n1_0 * wp.cos(-alpha)  # m=-1, m'=0
    D[2, 2] = d_n1_n1 * wp.cos(-alpha - gamma)  # m=-1, m'=-1


@wp.func
def wigner_d_l2(
    alpha: float,
    beta: float,
    gamma: float,
    D: wp.array2d(dtype=float),
):
    """Compute the real part of Wigner D-matrix for l=2.

    The 5x5 matrix is stored in D.

    Parameters
    ----------
    alpha, beta, gamma : float
        Euler angles (z-y-z convention)
    D : wp.array2d(dtype=float)
        Output 2D array of shape (5, 5) to store the real part of matrix elements
    """
    # Precompute half-angle trig
    c = wp.cos(beta / 2.0)
    s = wp.sin(beta / 2.0)
    cb = wp.cos(beta)
    sb = wp.sin(beta)
    sqrt6 = wp.sqrt(6.0)
    c2 = c * c
    c3 = c2 * c
    c4 = c2 * c2
    s2 = s * s
    s3 = s2 * s
    s4 = s2 * s2
    sb2 = sb * sb
    cos2b = wp.cos(2.0 * beta)
    sin2b = wp.sin(2.0 * beta)

    # Small d-matrix elements (real-valued)
    d_2_2 = c4
    d_2_1 = 2.0 * s * c3
    d_2_0 = sqrt6 * sb2 / 4.0
    d_2_n1 = 2.0 * s3 * c
    d_2_n2 = s4
    d_1_2 = -2.0 * s * c3
    d_1_1 = cb / 2.0 + cos2b / 2.0
    d_1_0 = sqrt6 * sin2b / 4.0
    d_1_n1 = cb / 2.0 - cos2b / 2.0
    d_1_n2 = 2.0 * s3 * c
    d_0_2 = sqrt6 * sb2 / 4.0
    d_0_1 = -sqrt6 * sin2b / 4.0
    d_0_0 = 1.0 - 3.0 * sb2 / 2.0
    d_0_n1 = sqrt6 * sin2b / 4.0
    d_0_n2 = sqrt6 * sb2 / 4.0
    d_n1_2 = -2.0 * s3 * c
    d_n1_1 = cb / 2.0 - cos2b / 2.0
    d_n1_0 = -sqrt6 * sin2b / 4.0
    d_n1_n1 = cb / 2.0 + cos2b / 2.0
    d_n1_n2 = 2.0 * s * c3
    d_n2_2 = s4
    d_n2_1 = -2.0 * s3 * c
    d_n2_0 = sqrt6 * sb2 / 4.0
    d_n2_n1 = -2.0 * s * c3
    d_n2_n2 = c4

    # Real part: d * cos(m*alpha + m'*gamma)
    # Row 0: m=2
    D[0, 0] = d_2_2 * wp.cos(2.0 * alpha + 2.0 * gamma)
    D[0, 1] = d_2_1 * wp.cos(2.0 * alpha + gamma)
    D[0, 2] = d_2_0 * wp.cos(2.0 * alpha)
    D[0, 3] = d_2_n1 * wp.cos(2.0 * alpha - gamma)
    D[0, 4] = d_2_n2 * wp.cos(2.0 * alpha - 2.0 * gamma)

    # Row 1: m=1
    D[1, 0] = d_1_2 * wp.cos(alpha + 2.0 * gamma)
    D[1, 1] = d_1_1 * wp.cos(alpha + gamma)
    D[1, 2] = d_1_0 * wp.cos(alpha)
    D[1, 3] = d_1_n1 * wp.cos(alpha - gamma)
    D[1, 4] = d_1_n2 * wp.cos(alpha - 2.0 * gamma)

    # Row 2: m=0
    D[2, 0] = d_0_2 * wp.cos(2.0 * gamma)
    D[2, 1] = d_0_1 * wp.cos(gamma)
    D[2, 2] = d_0_0
    D[2, 3] = d_0_n1 * wp.cos(-gamma)
    D[2, 4] = d_0_n2 * wp.cos(-2.0 * gamma)

    # Row 3: m=-1
    D[3, 0] = d_n1_2 * wp.cos(-alpha + 2.0 * gamma)
    D[3, 1] = d_n1_1 * wp.cos(-alpha + gamma)
    D[3, 2] = d_n1_0 * wp.cos(-alpha)
    D[3, 3] = d_n1_n1 * wp.cos(-alpha - gamma)
    D[3, 4] = d_n1_n2 * wp.cos(-alpha - 2.0 * gamma)

    # Row 4: m=-2
    D[4, 0] = d_n2_2 * wp.cos(-2.0 * alpha + 2.0 * gamma)
    D[4, 1] = d_n2_1 * wp.cos(-2.0 * alpha + gamma)
    D[4, 2] = d_n2_0 * wp.cos(-2.0 * alpha)
    D[4, 3] = d_n2_n1 * wp.cos(-2.0 * alpha - gamma)
    D[4, 4] = d_n2_n2 * wp.cos(-2.0 * alpha - 2.0 * gamma)


@wp.func
def wigner_d_l3(
    alpha: float,
    beta: float,
    gamma: float,
    D: wp.array2d(dtype=float),
):
    """Compute the real part of Wigner D-matrix for l=3.

    The 7x7 matrix is stored in D.

    Parameters
    ----------
    alpha, beta, gamma : float
        Euler angles (z-y-z convention)
    D : wp.array2d(dtype=float)
        Output 2D array of shape (7, 7) to store the real part of matrix elements
    """
    # Precompute half-angle trig
    c = wp.cos(beta / 2.0)
    s = wp.sin(beta / 2.0)
    cb = wp.cos(beta)
    sb = wp.sin(beta)

    # Powers
    c2 = c * c
    c3 = c2 * c
    c4 = c2 * c2
    c6 = c4 * c2
    s2 = s * s
    s3 = s2 * s
    s4 = s2 * s2
    s5 = s4 * s
    s6 = s4 * s2
    sb2 = sb * sb

    # Trig multiples
    sin2b = wp.sin(2.0 * beta)
    sin3b = wp.sin(3.0 * beta)
    cos3b = wp.cos(3.0 * beta)
    cb_p1 = cb + 1.0  # (cos(beta) + 1)
    cb_p1_sq = cb_p1 * cb_p1

    # Square roots
    sqrt6 = wp.sqrt(6.0)
    sqrt10 = wp.sqrt(10.0)
    sqrt15 = wp.sqrt(15.0)
    sqrt5 = wp.sqrt(5.0)
    sqrt30 = wp.sqrt(30.0)
    sqrt3 = wp.sqrt(3.0)

    # Small d-matrix elements (real-valued)
    d_3_3 = c6
    d_3_2 = sqrt6 * cb_p1_sq * sb / 8.0
    d_3_1 = sqrt15 * s2 * c4
    d_3_0 = sqrt5 * (3.0 * sb - sin3b) / 16.0
    d_3_n1 = sqrt15 * s4 * c2
    d_3_n2 = sqrt6 * s5 * c
    d_3_n3 = s6
    d_2_3 = -sqrt6 * cb_p1_sq * sb / 8.0
    d_2_2 = (3.0 * cb - 2.0) * c4
    d_2_1 = sqrt10 * (3.0 * cb - 1.0) * s * c3 / 2.0
    d_2_0 = sqrt30 * (cb - cos3b) / 16.0
    d_2_n1 = sqrt10 * (3.0 * cb + 1.0) * s3 * c / 2.0
    d_2_n2 = (3.0 * cb + 2.0) * s4
    d_2_n3 = sqrt6 * s5 * c
    d_1_3 = sqrt15 * s2 * c4
    d_1_2 = sqrt10 * (sb - 4.0 * sin2b - 3.0 * sin3b) / 32.0
    d_1_1 = (15.0 * cb * cb - 10.0 * cb - 1.0) * c2 / 4.0
    d_1_0 = sqrt3 * (sb + 5.0 * sin3b) / 16.0
    d_1_n1 = (-15.0 * sb2 + 10.0 * cb + 14.0) * s2 / 4.0
    d_1_n2 = sqrt10 * (3.0 * cb + 1.0) * s3 * c / 2.0
    d_1_n3 = sqrt15 * s4 * c2
    d_0_3 = sqrt5 * (-3.0 * sb + sin3b) / 16.0
    d_0_2 = sqrt30 * (cb - cos3b) / 16.0
    d_0_1 = -sqrt3 * (sb + 5.0 * sin3b) / 16.0
    d_0_0 = 3.0 * cb / 8.0 + 5.0 * cos3b / 8.0
    d_0_n1 = sqrt3 * (sb + 5.0 * sin3b) / 16.0
    d_0_n2 = sqrt30 * (cb - cos3b) / 16.0
    d_0_n3 = sqrt5 * (3.0 * sb - sin3b) / 16.0
    d_n1_3 = sqrt15 * s4 * c2
    d_n1_2 = sqrt10 * (-sb - 4.0 * sin2b + 3.0 * sin3b) / 32.0
    d_n1_1 = (-15.0 * sb2 + 10.0 * cb + 14.0) * s2 / 4.0
    d_n1_0 = -sqrt3 * (sb + 5.0 * sin3b) / 16.0
    d_n1_n1 = (15.0 * cb * cb - 10.0 * cb - 1.0) * c2 / 4.0
    d_n1_n2 = sqrt10 * (3.0 * cb - 1.0) * s * c3 / 2.0
    d_n1_n3 = sqrt15 * s2 * c4
    d_n2_3 = -sqrt6 * s5 * c
    d_n2_2 = (3.0 * cb + 2.0) * s4
    d_n2_1 = sqrt10 * (-sb - 4.0 * sin2b + 3.0 * sin3b) / 32.0
    d_n2_0 = sqrt30 * (cb - cos3b) / 16.0
    d_n2_n1 = sqrt10 * (sb - 4.0 * sin2b - 3.0 * sin3b) / 32.0
    d_n2_n2 = (3.0 * cb - 2.0) * c4
    d_n2_n3 = sqrt6 * cb_p1_sq * sb / 8.0
    d_n3_3 = s6
    d_n3_2 = -sqrt6 * s5 * c
    d_n3_1 = sqrt15 * s4 * c2
    d_n3_0 = sqrt5 * (-3.0 * sb + sin3b) / 16.0
    d_n3_n1 = sqrt15 * s2 * c4
    d_n3_n2 = -sqrt6 * cb_p1_sq * sb / 8.0
    d_n3_n3 = c6

    # Real part: d * cos(m*alpha + m'*gamma)
    # Row 0: m=3
    D[0, 0] = d_3_3 * wp.cos(3.0 * alpha + 3.0 * gamma)
    D[0, 1] = d_3_2 * wp.cos(3.0 * alpha + 2.0 * gamma)
    D[0, 2] = d_3_1 * wp.cos(3.0 * alpha + gamma)
    D[0, 3] = d_3_0 * wp.cos(3.0 * alpha)
    D[0, 4] = d_3_n1 * wp.cos(3.0 * alpha - gamma)
    D[0, 5] = d_3_n2 * wp.cos(3.0 * alpha - 2.0 * gamma)
    D[0, 6] = d_3_n3 * wp.cos(3.0 * alpha - 3.0 * gamma)

    # Row 1: m=2
    D[1, 0] = d_2_3 * wp.cos(2.0 * alpha + 3.0 * gamma)
    D[1, 1] = d_2_2 * wp.cos(2.0 * alpha + 2.0 * gamma)
    D[1, 2] = d_2_1 * wp.cos(2.0 * alpha + gamma)
    D[1, 3] = d_2_0 * wp.cos(2.0 * alpha)
    D[1, 4] = d_2_n1 * wp.cos(2.0 * alpha - gamma)
    D[1, 5] = d_2_n2 * wp.cos(2.0 * alpha - 2.0 * gamma)
    D[1, 6] = d_2_n3 * wp.cos(2.0 * alpha - 3.0 * gamma)

    # Row 2: m=1
    D[2, 0] = d_1_3 * wp.cos(alpha + 3.0 * gamma)
    D[2, 1] = d_1_2 * wp.cos(alpha + 2.0 * gamma)
    D[2, 2] = d_1_1 * wp.cos(alpha + gamma)
    D[2, 3] = d_1_0 * wp.cos(alpha)
    D[2, 4] = d_1_n1 * wp.cos(alpha - gamma)
    D[2, 5] = d_1_n2 * wp.cos(alpha - 2.0 * gamma)
    D[2, 6] = d_1_n3 * wp.cos(alpha - 3.0 * gamma)

    # Row 3: m=0
    D[3, 0] = d_0_3 * wp.cos(3.0 * gamma)
    D[3, 1] = d_0_2 * wp.cos(2.0 * gamma)
    D[3, 2] = d_0_1 * wp.cos(gamma)
    D[3, 3] = d_0_0
    D[3, 4] = d_0_n1 * wp.cos(-gamma)
    D[3, 5] = d_0_n2 * wp.cos(-2.0 * gamma)
    D[3, 6] = d_0_n3 * wp.cos(-3.0 * gamma)

    # Row 4: m=-1
    D[4, 0] = d_n1_3 * wp.cos(-alpha + 3.0 * gamma)
    D[4, 1] = d_n1_2 * wp.cos(-alpha + 2.0 * gamma)
    D[4, 2] = d_n1_1 * wp.cos(-alpha + gamma)
    D[4, 3] = d_n1_0 * wp.cos(-alpha)
    D[4, 4] = d_n1_n1 * wp.cos(-alpha - gamma)
    D[4, 5] = d_n1_n2 * wp.cos(-alpha - 2.0 * gamma)
    D[4, 6] = d_n1_n3 * wp.cos(-alpha - 3.0 * gamma)

    # Row 5: m=-2
    D[5, 0] = d_n2_3 * wp.cos(-2.0 * alpha + 3.0 * gamma)
    D[5, 1] = d_n2_2 * wp.cos(-2.0 * alpha + 2.0 * gamma)
    D[5, 2] = d_n2_1 * wp.cos(-2.0 * alpha + gamma)
    D[5, 3] = d_n2_0 * wp.cos(-2.0 * alpha)
    D[5, 4] = d_n2_n1 * wp.cos(-2.0 * alpha - gamma)
    D[5, 5] = d_n2_n2 * wp.cos(-2.0 * alpha - 2.0 * gamma)
    D[5, 6] = d_n2_n3 * wp.cos(-2.0 * alpha - 3.0 * gamma)

    # Row 6: m=-3
    D[6, 0] = d_n3_3 * wp.cos(-3.0 * alpha + 3.0 * gamma)
    D[6, 1] = d_n3_2 * wp.cos(-3.0 * alpha + 2.0 * gamma)
    D[6, 2] = d_n3_1 * wp.cos(-3.0 * alpha + gamma)
    D[6, 3] = d_n3_0 * wp.cos(-3.0 * alpha)
    D[6, 4] = d_n3_n1 * wp.cos(-3.0 * alpha - gamma)
    D[6, 5] = d_n3_n2 * wp.cos(-3.0 * alpha - 2.0 * gamma)
    D[6, 6] = d_n3_n3 * wp.cos(-3.0 * alpha - 3.0 * gamma)


@wp.func
def wigner_d_l4(
    alpha: float,
    beta: float,
    gamma: float,
    D: wp.array2d(dtype=float),
):
    """Compute the real part of Wigner D-matrix for l=4.

    The 9x9 matrix is stored in D.

    Parameters
    ----------
    alpha, beta, gamma : float
        Euler angles (z-y-z convention)
    D : wp.array2d(dtype=float)
        Output 2D array of shape (9, 9) to store the real part of matrix elements
    """
    # Precompute half-angle trig
    c = wp.cos(beta / 2.0)
    s = wp.sin(beta / 2.0)
    cb = wp.cos(beta)
    sb = wp.sin(beta)

    # Powers
    c2 = c * c
    c3 = c2 * c
    c4 = c2 * c2
    c5 = c4 * c
    c6 = c4 * c2
    c8 = c4 * c4
    s2 = s * s
    s3 = s2 * s
    s4 = s2 * s2
    s5 = s4 * s
    s6 = s4 * s2
    s7 = s6 * s
    s8 = s4 * s4
    sb2 = sb * sb
    sb3 = sb2 * sb
    cb2 = cb * cb

    # Trig multiples
    sin2b = wp.sin(2.0 * beta)
    sin4b = wp.sin(4.0 * beta)
    cb_p1 = cb + 1.0
    cb_p1_cu = cb_p1 * cb_p1 * cb_p1
    cb_m1 = cb - 1.0
    cb_m1_sq = cb_m1 * cb_m1

    # Square roots
    sqrt2 = wp.sqrt(2.0)
    sqrt7 = wp.sqrt(7.0)
    sqrt14 = wp.sqrt(14.0)
    sqrt70 = wp.sqrt(70.0)
    sqrt35 = wp.sqrt(35.0)
    sqrt10 = wp.sqrt(10.0)
    sqrt5 = wp.sqrt(5.0)

    # Small d-matrix elements (real-valued)
    d_4_4 = c8
    d_4_3 = sqrt2 * cb_p1_cu * sb / 8.0
    d_4_2 = 2.0 * sqrt7 * s2 * c6
    d_4_1 = 2.0 * sqrt14 * s3 * c5
    d_4_0 = sqrt70 * s4 * c4
    d_4_n1 = 2.0 * sqrt14 * s5 * c3
    d_4_n2 = 2.0 * sqrt7 * s6 * c2
    d_4_n3 = 2.0 * sqrt2 * s7 * c
    d_4_n4 = s8
    d_3_4 = -sqrt2 * cb_p1_cu * sb / 8.0
    d_3_3 = (4.0 * cb - 3.0) * c6
    d_3_2 = sqrt14 * (-sb + sin2b) * (cb_p1) * (cb_p1) / 8.0
    d_3_1 = sqrt7 * (4.0 * cb - 1.0) * s2 * c4
    d_3_0 = sqrt35 * sb3 * cb / 4.0
    d_3_n1 = sqrt7 * (4.0 * cb + 1.0) * s4 * c2
    d_3_n2 = sqrt14 * (sb + sin2b) * cb_m1_sq / 8.0
    d_3_n3 = (4.0 * cb + 3.0) * s6
    d_3_n4 = 2.0 * sqrt2 * s7 * c
    d_2_4 = 2.0 * sqrt7 * s2 * c6
    d_2_3 = sqrt14 * (sb - sin2b) * (cb_p1) * (cb_p1) / 8.0
    d_2_2 = (7.0 * cb2 - 7.0 * cb + 1.0) * c4
    d_2_1 = sqrt2 * (14.0 * cb2 - 7.0 * cb - 1.0) * s * c3 / 2.0
    d_2_0 = sqrt10 * (6.0 - 7.0 * sb2) * s2 * c2 / 2.0
    d_2_n1 = sqrt2 * (-14.0 * sb2 + 7.0 * cb + 13.0) * s3 * c / 2.0
    d_2_n2 = (-7.0 * sb2 + 7.0 * cb + 8.0) * s4
    d_2_n3 = sqrt14 * (sb + sin2b) * cb_m1_sq / 8.0
    d_2_n4 = 2.0 * sqrt7 * s6 * c2
    d_1_4 = -2.0 * sqrt14 * s3 * c5
    d_1_3 = sqrt7 * (4.0 * cb - 1.0) * s2 * c4
    d_1_2 = sqrt2 * (-14.0 * cb2 + 7.0 * cb + 1.0) * s * c3 / 2.0
    d_1_1 = (-55.0 * s6 + 60.0 * s4 - 15.0 * s2 + c6) * c2
    d_1_0 = sqrt5 * (2.0 * sin2b + 7.0 * sin4b) / 32.0
    d_1_n1 = -46.0 * s8 + 75.0 * s6 - 30.0 * s4 + 10.0 * s2 * c6
    d_1_n2 = sqrt2 * (-14.0 * sb2 + 7.0 * cb + 13.0) * s3 * c / 2.0
    d_1_n3 = sqrt7 * (4.0 * cb + 1.0) * s4 * c2
    d_1_n4 = 2.0 * sqrt14 * s5 * c3
    d_0_4 = sqrt70 * s4 * c4
    d_0_3 = -sqrt35 * sb3 * cb / 4.0
    d_0_2 = sqrt10 * (6.0 - 7.0 * sb2) * s2 * c2 / 2.0
    d_0_1 = sqrt5 * (7.0 * sb2 - 4.0) * sin2b / 8.0
    d_0_0 = 53.0 * s8 - 88.0 * s6 + 36.0 * s4 + 17.0 * c8 - 16.0 * c6
    d_0_n1 = sqrt5 * (2.0 * sin2b + 7.0 * sin4b) / 32.0
    d_0_n2 = sqrt10 * (6.0 - 7.0 * sb2) * s2 * c2 / 2.0
    d_0_n3 = sqrt35 * sb3 * cb / 4.0
    d_0_n4 = sqrt70 * s4 * c4
    d_n1_4 = -2.0 * sqrt14 * s5 * c3
    d_n1_3 = sqrt7 * (4.0 * cb + 1.0) * s4 * c2
    d_n1_2 = sqrt2 * (14.0 * sb2 - 7.0 * cb - 13.0) * s3 * c / 2.0
    d_n1_1 = -46.0 * s8 + 75.0 * s6 - 30.0 * s4 + 10.0 * s2 * c6
    d_n1_0 = sqrt5 * (7.0 * sb2 - 4.0) * sin2b / 8.0
    d_n1_n1 = (-55.0 * s6 + 60.0 * s4 - 15.0 * s2 + c6) * c2
    d_n1_n2 = sqrt2 * (14.0 * cb2 - 7.0 * cb - 1.0) * s * c3 / 2.0
    d_n1_n3 = sqrt7 * (4.0 * cb - 1.0) * s2 * c4
    d_n1_n4 = 2.0 * sqrt14 * s3 * c5
    d_n2_4 = 2.0 * sqrt7 * s6 * c2
    d_n2_3 = -sqrt14 * (sb + sin2b) * cb_m1_sq / 8.0
    d_n2_2 = (-7.0 * sb2 + 7.0 * cb + 8.0) * s4
    d_n2_1 = sqrt2 * (14.0 * sb2 - 7.0 * cb - 13.0) * s3 * c / 2.0
    d_n2_0 = sqrt10 * (6.0 - 7.0 * sb2) * s2 * c2 / 2.0
    d_n2_n1 = sqrt2 * (-14.0 * cb2 + 7.0 * cb + 1.0) * s * c3 / 2.0
    d_n2_n2 = (7.0 * cb2 - 7.0 * cb + 1.0) * c4
    d_n2_n3 = sqrt14 * (-sb + sin2b) * (cb_p1) * (cb_p1) / 8.0
    d_n2_n4 = 2.0 * sqrt7 * s2 * c6
    d_n3_4 = -2.0 * sqrt2 * s7 * c
    d_n3_3 = (4.0 * cb + 3.0) * s6
    d_n3_2 = -sqrt14 * (sb + sin2b) * cb_m1_sq / 8.0
    d_n3_1 = sqrt7 * (4.0 * cb + 1.0) * s4 * c2
    d_n3_0 = -sqrt35 * sb3 * cb / 4.0
    d_n3_n1 = sqrt7 * (4.0 * cb - 1.0) * s2 * c4
    d_n3_n2 = sqrt14 * (sb - sin2b) * (cb_p1) * (cb_p1) / 8.0
    d_n3_n3 = (4.0 * cb - 3.0) * c6
    d_n3_n4 = sqrt2 * cb_p1_cu * sb / 8.0
    d_n4_4 = s8
    d_n4_3 = -2.0 * sqrt2 * s7 * c
    d_n4_2 = 2.0 * sqrt7 * s6 * c2
    d_n4_1 = -2.0 * sqrt14 * s5 * c3
    d_n4_0 = sqrt70 * s4 * c4
    d_n4_n1 = -2.0 * sqrt14 * s3 * c5
    d_n4_n2 = 2.0 * sqrt7 * s2 * c6
    d_n4_n3 = -sqrt2 * cb_p1_cu * sb / 8.0
    d_n4_n4 = c8

    # Real part: d * cos(m*alpha + m'*gamma)
    # Row 0: m=4
    D[0, 0] = d_4_4 * wp.cos(4.0 * alpha + 4.0 * gamma)
    D[0, 1] = d_4_3 * wp.cos(4.0 * alpha + 3.0 * gamma)
    D[0, 2] = d_4_2 * wp.cos(4.0 * alpha + 2.0 * gamma)
    D[0, 3] = d_4_1 * wp.cos(4.0 * alpha + gamma)
    D[0, 4] = d_4_0 * wp.cos(4.0 * alpha)
    D[0, 5] = d_4_n1 * wp.cos(4.0 * alpha - gamma)
    D[0, 6] = d_4_n2 * wp.cos(4.0 * alpha - 2.0 * gamma)
    D[0, 7] = d_4_n3 * wp.cos(4.0 * alpha - 3.0 * gamma)
    D[0, 8] = d_4_n4 * wp.cos(4.0 * alpha - 4.0 * gamma)

    # Row 1: m=3
    D[1, 0] = d_3_4 * wp.cos(3.0 * alpha + 4.0 * gamma)
    D[1, 1] = d_3_3 * wp.cos(3.0 * alpha + 3.0 * gamma)
    D[1, 2] = d_3_2 * wp.cos(3.0 * alpha + 2.0 * gamma)
    D[1, 3] = d_3_1 * wp.cos(3.0 * alpha + gamma)
    D[1, 4] = d_3_0 * wp.cos(3.0 * alpha)
    D[1, 5] = d_3_n1 * wp.cos(3.0 * alpha - gamma)
    D[1, 6] = d_3_n2 * wp.cos(3.0 * alpha - 2.0 * gamma)
    D[1, 7] = d_3_n3 * wp.cos(3.0 * alpha - 3.0 * gamma)
    D[1, 8] = d_3_n4 * wp.cos(3.0 * alpha - 4.0 * gamma)

    # Row 2: m=2
    D[2, 0] = d_2_4 * wp.cos(2.0 * alpha + 4.0 * gamma)
    D[2, 1] = d_2_3 * wp.cos(2.0 * alpha + 3.0 * gamma)
    D[2, 2] = d_2_2 * wp.cos(2.0 * alpha + 2.0 * gamma)
    D[2, 3] = d_2_1 * wp.cos(2.0 * alpha + gamma)
    D[2, 4] = d_2_0 * wp.cos(2.0 * alpha)
    D[2, 5] = d_2_n1 * wp.cos(2.0 * alpha - gamma)
    D[2, 6] = d_2_n2 * wp.cos(2.0 * alpha - 2.0 * gamma)
    D[2, 7] = d_2_n3 * wp.cos(2.0 * alpha - 3.0 * gamma)
    D[2, 8] = d_2_n4 * wp.cos(2.0 * alpha - 4.0 * gamma)

    # Row 3: m=1
    D[3, 0] = d_1_4 * wp.cos(alpha + 4.0 * gamma)
    D[3, 1] = d_1_3 * wp.cos(alpha + 3.0 * gamma)
    D[3, 2] = d_1_2 * wp.cos(alpha + 2.0 * gamma)
    D[3, 3] = d_1_1 * wp.cos(alpha + gamma)
    D[3, 4] = d_1_0 * wp.cos(alpha)
    D[3, 5] = d_1_n1 * wp.cos(alpha - gamma)
    D[3, 6] = d_1_n2 * wp.cos(alpha - 2.0 * gamma)
    D[3, 7] = d_1_n3 * wp.cos(alpha - 3.0 * gamma)
    D[3, 8] = d_1_n4 * wp.cos(alpha - 4.0 * gamma)

    # Row 4: m=0
    D[4, 0] = d_0_4 * wp.cos(4.0 * gamma)
    D[4, 1] = d_0_3 * wp.cos(3.0 * gamma)
    D[4, 2] = d_0_2 * wp.cos(2.0 * gamma)
    D[4, 3] = d_0_1 * wp.cos(gamma)
    D[4, 4] = d_0_0
    D[4, 5] = d_0_n1 * wp.cos(-gamma)
    D[4, 6] = d_0_n2 * wp.cos(-2.0 * gamma)
    D[4, 7] = d_0_n3 * wp.cos(-3.0 * gamma)
    D[4, 8] = d_0_n4 * wp.cos(-4.0 * gamma)

    # Row 5: m=-1
    D[5, 0] = d_n1_4 * wp.cos(-alpha + 4.0 * gamma)
    D[5, 1] = d_n1_3 * wp.cos(-alpha + 3.0 * gamma)
    D[5, 2] = d_n1_2 * wp.cos(-alpha + 2.0 * gamma)
    D[5, 3] = d_n1_1 * wp.cos(-alpha + gamma)
    D[5, 4] = d_n1_0 * wp.cos(-alpha)
    D[5, 5] = d_n1_n1 * wp.cos(-alpha - gamma)
    D[5, 6] = d_n1_n2 * wp.cos(-alpha - 2.0 * gamma)
    D[5, 7] = d_n1_n3 * wp.cos(-alpha - 3.0 * gamma)
    D[5, 8] = d_n1_n4 * wp.cos(-alpha - 4.0 * gamma)

    # Row 6: m=-2
    D[6, 0] = d_n2_4 * wp.cos(-2.0 * alpha + 4.0 * gamma)
    D[6, 1] = d_n2_3 * wp.cos(-2.0 * alpha + 3.0 * gamma)
    D[6, 2] = d_n2_2 * wp.cos(-2.0 * alpha + 2.0 * gamma)
    D[6, 3] = d_n2_1 * wp.cos(-2.0 * alpha + gamma)
    D[6, 4] = d_n2_0 * wp.cos(-2.0 * alpha)
    D[6, 5] = d_n2_n1 * wp.cos(-2.0 * alpha - gamma)
    D[6, 6] = d_n2_n2 * wp.cos(-2.0 * alpha - 2.0 * gamma)
    D[6, 7] = d_n2_n3 * wp.cos(-2.0 * alpha - 3.0 * gamma)
    D[6, 8] = d_n2_n4 * wp.cos(-2.0 * alpha - 4.0 * gamma)

    # Row 7: m=-3
    D[7, 0] = d_n3_4 * wp.cos(-3.0 * alpha + 4.0 * gamma)
    D[7, 1] = d_n3_3 * wp.cos(-3.0 * alpha + 3.0 * gamma)
    D[7, 2] = d_n3_2 * wp.cos(-3.0 * alpha + 2.0 * gamma)
    D[7, 3] = d_n3_1 * wp.cos(-3.0 * alpha + gamma)
    D[7, 4] = d_n3_0 * wp.cos(-3.0 * alpha)
    D[7, 5] = d_n3_n1 * wp.cos(-3.0 * alpha - gamma)
    D[7, 6] = d_n3_n2 * wp.cos(-3.0 * alpha - 2.0 * gamma)
    D[7, 7] = d_n3_n3 * wp.cos(-3.0 * alpha - 3.0 * gamma)
    D[7, 8] = d_n3_n4 * wp.cos(-3.0 * alpha - 4.0 * gamma)

    # Row 8: m=-4
    D[8, 0] = d_n4_4 * wp.cos(-4.0 * alpha + 4.0 * gamma)
    D[8, 1] = d_n4_3 * wp.cos(-4.0 * alpha + 3.0 * gamma)
    D[8, 2] = d_n4_2 * wp.cos(-4.0 * alpha + 2.0 * gamma)
    D[8, 3] = d_n4_1 * wp.cos(-4.0 * alpha + gamma)
    D[8, 4] = d_n4_0 * wp.cos(-4.0 * alpha)
    D[8, 5] = d_n4_n1 * wp.cos(-4.0 * alpha - gamma)
    D[8, 6] = d_n4_n2 * wp.cos(-4.0 * alpha - 2.0 * gamma)
    D[8, 7] = d_n4_n3 * wp.cos(-4.0 * alpha - 3.0 * gamma)
    D[8, 8] = d_n4_n4 * wp.cos(-4.0 * alpha - 4.0 * gamma)


@wp.func
def wigner_d_l5(
    alpha: float,
    beta: float,
    gamma: float,
    D: wp.array2d(dtype=float),
):
    """Compute the real part of Wigner D-matrix for l=5.

    The 11x11 matrix is stored in D.

    Parameters
    ----------
    alpha, beta, gamma : float
        Euler angles (z-y-z convention)
    D : wp.array2d(dtype=float)
        Output 2D array of shape (11, 11) to store the real part of matrix elements
    """
    # Precompute half-angle trig
    c = wp.cos(beta / 2.0)
    s = wp.sin(beta / 2.0)
    cb = wp.cos(beta)
    sb = wp.sin(beta)

    # Powers
    c2 = c * c
    c3 = c2 * c
    c4 = c2 * c2
    c5 = c4 * c
    c6 = c4 * c2
    c7 = c6 * c
    c8 = c4 * c4
    c10 = c8 * c2
    s2 = s * s
    s3 = s2 * s
    s4 = s2 * s2
    s5 = s4 * s
    s6 = s4 * s2
    s7 = s6 * s
    s8 = s4 * s4
    s9 = s8 * s
    s10 = s8 * s2
    sb2 = sb * sb
    cb2 = cb * cb
    cb4 = cb2 * cb2

    # Trig multiples
    sin2b = wp.sin(2.0 * beta)
    sin3b = wp.sin(3.0 * beta)
    sin5b = wp.sin(5.0 * beta)
    cos2b = wp.cos(2.0 * beta)
    cos3b = wp.cos(3.0 * beta)
    cb_p1 = cb + 1.0
    cb_p1_sq = cb_p1 * cb_p1
    cb_p1_4 = cb_p1_sq * cb_p1_sq

    # Square roots
    sqrt2 = wp.sqrt(2.0)
    sqrt3 = wp.sqrt(3.0)
    sqrt5 = wp.sqrt(5.0)
    sqrt6 = wp.sqrt(6.0)
    sqrt7 = wp.sqrt(7.0)
    sqrt10 = wp.sqrt(10.0)
    sqrt21 = wp.sqrt(21.0)
    sqrt30 = wp.sqrt(30.0)
    sqrt35 = wp.sqrt(35.0)
    sqrt42 = wp.sqrt(42.0)
    sqrt70 = wp.sqrt(70.0)
    sqrt210 = wp.sqrt(210.0)

    # Small d-matrix elements (real-valued)
    d_5_5 = c10
    d_5_4 = sqrt10 * cb_p1_4 * sb / 32.0
    d_5_3 = 3.0 * sqrt5 * s2 * c8
    d_5_2 = 2.0 * sqrt30 * s3 * c7
    d_5_1 = sqrt210 * s4 * c6
    d_5_0 = 6.0 * sqrt7 * s5 * c5
    d_5_n1 = sqrt210 * s6 * c4
    d_5_n2 = 2.0 * sqrt30 * s7 * c3
    d_5_n3 = 3.0 * sqrt5 * s8 * c2
    d_5_n4 = sqrt10 * s9 * c
    d_5_n5 = s10
    d_4_5 = -sqrt10 * cb_p1_4 * sb / 32.0
    d_4_4 = (5.0 * cb - 4.0) * c8
    d_4_3 = 3.0 * sqrt2 * (5.0 * cb - 3.0) * s * c7 / 2.0
    d_4_2 = sqrt3 * (10.0 * cb - 4.0) * s2 * c6
    d_4_1 = sqrt21 * (5.0 * cb - 1.0) * s3 * c5
    d_4_0 = 3.0 * sqrt70 * s4 * c4 * cb
    d_4_n1 = sqrt21 * (5.0 * cb + 1.0) * s5 * c3
    d_4_n2 = sqrt3 * (10.0 * cb + 4.0) * s6 * c2
    d_4_n3 = 3.0 * sqrt2 * (5.0 * cb + 3.0) * s7 * c / 2.0
    d_4_n4 = (5.0 * cb + 4.0) * s8
    d_4_n5 = sqrt10 * s9 * c
    d_3_5 = 3.0 * sqrt5 * s2 * c8
    d_3_4 = 3.0 * sqrt2 * (3.0 - 5.0 * cb) * s * c7 / 2.0
    d_3_3 = (-45.0 * sb2 - 54.0 * cb + 58.0) * c6 / 4.0
    d_3_2 = sqrt6 * cb_p1_sq * (19.0 * sb - 24.0 * sin2b + 15.0 * sin3b) / 64.0
    d_3_1 = (
        sqrt42 * cb_p1_sq * (-65.0 * cb + 42.0 * cos2b - 15.0 * cos3b + 38.0) / 128.0
    )
    d_3_0 = sqrt35 * (6.0 * sb + 13.0 * sin3b - 9.0 * sin5b) / 256.0
    d_3_n1 = sqrt42 * (-15.0 * sb2 + 6.0 * cb + 14.0) * s4 * c2 / 4.0
    d_3_n2 = sqrt6 * (-15.0 * sb2 + 12.0 * cb + 16.0) * s5 * c / 2.0
    d_3_n3 = (-45.0 * sb2 + 54.0 * cb + 58.0) * s6 / 4.0
    d_3_n4 = 3.0 * sqrt2 * (5.0 * cb + 3.0) * s7 * c / 2.0
    d_3_n5 = 3.0 * sqrt5 * s8 * c2
    d_2_5 = -2.0 * sqrt30 * s3 * c7
    d_2_4 = sqrt3 * (10.0 * cb - 4.0) * s2 * c6
    d_2_3 = sqrt6 * (15.0 * sb2 + 12.0 * cb - 16.0) * s * c5 / 2.0
    d_2_2 = (-119.0 * s6 + 105.0 * s4 - 21.0 * s2 + c6) * c4
    d_2_1 = 2.0 * sqrt7 * (-29.0 * s6 + 33.0 * s4 - 9.0 * s2 + c6) * s * c3
    d_2_0 = sqrt210 * (2.0 * cb + cos3b - 3.0 * wp.cos(5.0 * beta)) / 128.0
    d_2_n1 = 2.0 * sqrt7 * (-25.0 * s6 + 39.0 * s4 - 15.0 * s2 + 5.0 * c6) * s3 * c
    d_2_n2 = -85.0 * s10 + 147.0 * s8 - 63.0 * s6 + 35.0 * s4 * c6
    d_2_n3 = sqrt6 * (-15.0 * sb2 + 12.0 * cb + 16.0) * s5 * c / 2.0
    d_2_n4 = sqrt3 * (10.0 * cb + 4.0) * s6 * c2
    d_2_n5 = 2.0 * sqrt30 * s7 * c3
    d_1_5 = sqrt210 * s4 * c6
    d_1_4 = sqrt21 * (1.0 - 5.0 * cb) * s3 * c5
    d_1_3 = (
        sqrt42 * cb_p1_sq * (-65.0 * cb + 42.0 * cos2b - 15.0 * cos3b + 38.0) / 128.0
    )
    d_1_2 = 2.0 * sqrt7 * (29.0 * s6 - 33.0 * s4 + 9.0 * s2 - c6) * s * c3
    d_1_1 = (185.0 * s8 - 260.0 * s6 + 90.0 * s4 + 25.0 * c8 - 24.0 * c6) * c2
    d_1_0 = sqrt30 * (21.0 * cb4 - 14.0 * cb2 + 1.0) * sb / 16.0
    d_1_n1 = (115.0 * s8 - 204.0 * s6 + 90.0 * s4 + 95.0 * c8 - 80.0 * c6) * s2
    d_1_n2 = 2.0 * sqrt7 * (-25.0 * s6 + 39.0 * s4 - 15.0 * s2 + 5.0 * c6) * s3 * c
    d_1_n3 = sqrt42 * (-15.0 * sb2 + 6.0 * cb + 14.0) * s4 * c2 / 4.0
    d_1_n4 = sqrt21 * (5.0 * cb + 1.0) * s5 * c3
    d_1_n5 = sqrt210 * s6 * c4
    d_0_5 = -6.0 * sqrt7 * s5 * c5
    d_0_4 = 3.0 * sqrt70 * s4 * c4 * cb
    d_0_3 = sqrt35 * (-6.0 * sb - 13.0 * sin3b + 9.0 * sin5b) / 256.0
    d_0_2 = sqrt210 * (2.0 * cb + cos3b - 3.0 * wp.cos(5.0 * beta)) / 128.0
    d_0_1 = sqrt30 * (-21.0 * cb4 + 14.0 * cb2 - 1.0) * sb / 16.0
    d_0_0 = (63.0 * cb4 - 70.0 * cb2 + 15.0) * cb / 8.0
    d_0_n1 = sqrt30 * (21.0 * cb4 - 14.0 * cb2 + 1.0) * sb / 16.0
    d_0_n2 = sqrt210 * (2.0 * cb + cos3b - 3.0 * wp.cos(5.0 * beta)) / 128.0
    d_0_n3 = sqrt35 * (6.0 * sb + 13.0 * sin3b - 9.0 * sin5b) / 256.0
    d_0_n4 = 3.0 * sqrt70 * s4 * c4 * cb
    d_0_n5 = 6.0 * sqrt7 * s5 * c5
    d_n1_5 = sqrt210 * s6 * c4
    d_n1_4 = -sqrt21 * (5.0 * cb + 1.0) * s5 * c3
    d_n1_3 = sqrt42 * (-15.0 * sb2 + 6.0 * cb + 14.0) * s4 * c2 / 4.0
    d_n1_2 = 2.0 * sqrt7 * (25.0 * s6 - 39.0 * s4 + 15.0 * s2 - 5.0 * c6) * s3 * c
    d_n1_1 = (115.0 * s8 - 204.0 * s6 + 90.0 * s4 + 95.0 * c8 - 80.0 * c6) * s2
    d_n1_0 = sqrt30 * (-21.0 * cb4 + 14.0 * cb2 - 1.0) * sb / 16.0
    d_n1_n1 = (185.0 * s8 - 260.0 * s6 + 90.0 * s4 + 25.0 * c8 - 24.0 * c6) * c2
    d_n1_n2 = 2.0 * sqrt7 * (-29.0 * s6 + 33.0 * s4 - 9.0 * s2 + c6) * s * c3
    d_n1_n3 = (
        sqrt42 * cb_p1_sq * (-65.0 * cb + 42.0 * cos2b - 15.0 * cos3b + 38.0) / 128.0
    )
    d_n1_n4 = sqrt21 * (5.0 * cb - 1.0) * s3 * c5
    d_n1_n5 = sqrt210 * s4 * c6
    d_n2_5 = -2.0 * sqrt30 * s7 * c3
    d_n2_4 = sqrt3 * (10.0 * cb + 4.0) * s6 * c2
    d_n2_3 = sqrt6 * (15.0 * sb2 - 12.0 * cb - 16.0) * s5 * c / 2.0
    d_n2_2 = -85.0 * s10 + 147.0 * s8 - 63.0 * s6 + 35.0 * s4 * c6
    d_n2_1 = 2.0 * sqrt7 * (25.0 * s6 - 39.0 * s4 + 15.0 * s2 - 5.0 * c6) * s3 * c
    d_n2_0 = sqrt210 * (2.0 * cb + cos3b - 3.0 * wp.cos(5.0 * beta)) / 128.0
    d_n2_n1 = 2.0 * sqrt7 * (29.0 * s6 - 33.0 * s4 + 9.0 * s2 - c6) * s * c3
    d_n2_n2 = (-119.0 * s6 + 105.0 * s4 - 21.0 * s2 + c6) * c4
    d_n2_n3 = sqrt6 * cb_p1_sq * (19.0 * sb - 24.0 * sin2b + 15.0 * sin3b) / 64.0
    d_n2_n4 = sqrt3 * (10.0 * cb - 4.0) * s2 * c6
    d_n2_n5 = 2.0 * sqrt30 * s3 * c7
    d_n3_5 = 3.0 * sqrt5 * s8 * c2
    d_n3_4 = -3.0 * sqrt2 * (5.0 * cb + 3.0) * s7 * c / 2.0
    d_n3_3 = (-45.0 * sb2 + 54.0 * cb + 58.0) * s6 / 4.0
    d_n3_2 = sqrt6 * (15.0 * sb2 - 12.0 * cb - 16.0) * s5 * c / 2.0
    d_n3_1 = sqrt42 * (-15.0 * sb2 + 6.0 * cb + 14.0) * s4 * c2 / 4.0
    d_n3_0 = sqrt35 * (-6.0 * sb - 13.0 * sin3b + 9.0 * sin5b) / 256.0
    d_n3_n1 = (
        sqrt42 * cb_p1_sq * (-65.0 * cb + 42.0 * cos2b - 15.0 * cos3b + 38.0) / 128.0
    )
    d_n3_n2 = sqrt6 * (15.0 * sb2 + 12.0 * cb - 16.0) * s * c5 / 2.0
    d_n3_n3 = (-45.0 * sb2 - 54.0 * cb + 58.0) * c6 / 4.0
    d_n3_n4 = 3.0 * sqrt2 * (5.0 * cb - 3.0) * s * c7 / 2.0
    d_n3_n5 = 3.0 * sqrt5 * s2 * c8
    d_n4_5 = -sqrt10 * s9 * c
    d_n4_4 = (5.0 * cb + 4.0) * s8
    d_n4_3 = -3.0 * sqrt2 * (5.0 * cb + 3.0) * s7 * c / 2.0
    d_n4_2 = sqrt3 * (10.0 * cb + 4.0) * s6 * c2
    d_n4_1 = -sqrt21 * (5.0 * cb + 1.0) * s5 * c3
    d_n4_0 = 3.0 * sqrt70 * s4 * c4 * cb
    d_n4_n1 = sqrt21 * (1.0 - 5.0 * cb) * s3 * c5
    d_n4_n2 = sqrt3 * (10.0 * cb - 4.0) * s2 * c6
    d_n4_n3 = 3.0 * sqrt2 * (3.0 - 5.0 * cb) * s * c7 / 2.0
    d_n4_n4 = (5.0 * cb - 4.0) * c8
    d_n4_n5 = sqrt10 * cb_p1_4 * sb / 32.0
    d_n5_5 = s10
    d_n5_4 = -sqrt10 * s9 * c
    d_n5_3 = 3.0 * sqrt5 * s8 * c2
    d_n5_2 = -2.0 * sqrt30 * s7 * c3
    d_n5_1 = sqrt210 * s6 * c4
    d_n5_0 = -6.0 * sqrt7 * s5 * c5
    d_n5_n1 = sqrt210 * s4 * c6
    d_n5_n2 = -2.0 * sqrt30 * s3 * c7
    d_n5_n3 = 3.0 * sqrt5 * s2 * c8
    d_n5_n4 = -sqrt10 * cb_p1_4 * sb / 32.0
    d_n5_n5 = c10

    # Real part: d * cos(m*alpha + m'*gamma)
    # Row 0: m=5
    D[0, 0] = d_5_5 * wp.cos(5.0 * alpha + 5.0 * gamma)
    D[0, 1] = d_5_4 * wp.cos(5.0 * alpha + 4.0 * gamma)
    D[0, 2] = d_5_3 * wp.cos(5.0 * alpha + 3.0 * gamma)
    D[0, 3] = d_5_2 * wp.cos(5.0 * alpha + 2.0 * gamma)
    D[0, 4] = d_5_1 * wp.cos(5.0 * alpha + gamma)
    D[0, 5] = d_5_0 * wp.cos(5.0 * alpha)
    D[0, 6] = d_5_n1 * wp.cos(5.0 * alpha - gamma)
    D[0, 7] = d_5_n2 * wp.cos(5.0 * alpha - 2.0 * gamma)
    D[0, 8] = d_5_n3 * wp.cos(5.0 * alpha - 3.0 * gamma)
    D[0, 9] = d_5_n4 * wp.cos(5.0 * alpha - 4.0 * gamma)
    D[0, 10] = d_5_n5 * wp.cos(5.0 * alpha - 5.0 * gamma)

    # Row 1: m=4
    D[1, 0] = d_4_5 * wp.cos(4.0 * alpha + 5.0 * gamma)
    D[1, 1] = d_4_4 * wp.cos(4.0 * alpha + 4.0 * gamma)
    D[1, 2] = d_4_3 * wp.cos(4.0 * alpha + 3.0 * gamma)
    D[1, 3] = d_4_2 * wp.cos(4.0 * alpha + 2.0 * gamma)
    D[1, 4] = d_4_1 * wp.cos(4.0 * alpha + gamma)
    D[1, 5] = d_4_0 * wp.cos(4.0 * alpha)
    D[1, 6] = d_4_n1 * wp.cos(4.0 * alpha - gamma)
    D[1, 7] = d_4_n2 * wp.cos(4.0 * alpha - 2.0 * gamma)
    D[1, 8] = d_4_n3 * wp.cos(4.0 * alpha - 3.0 * gamma)
    D[1, 9] = d_4_n4 * wp.cos(4.0 * alpha - 4.0 * gamma)
    D[1, 10] = d_4_n5 * wp.cos(4.0 * alpha - 5.0 * gamma)

    # Row 2: m=3
    D[2, 0] = d_3_5 * wp.cos(3.0 * alpha + 5.0 * gamma)
    D[2, 1] = d_3_4 * wp.cos(3.0 * alpha + 4.0 * gamma)
    D[2, 2] = d_3_3 * wp.cos(3.0 * alpha + 3.0 * gamma)
    D[2, 3] = d_3_2 * wp.cos(3.0 * alpha + 2.0 * gamma)
    D[2, 4] = d_3_1 * wp.cos(3.0 * alpha + gamma)
    D[2, 5] = d_3_0 * wp.cos(3.0 * alpha)
    D[2, 6] = d_3_n1 * wp.cos(3.0 * alpha - gamma)
    D[2, 7] = d_3_n2 * wp.cos(3.0 * alpha - 2.0 * gamma)
    D[2, 8] = d_3_n3 * wp.cos(3.0 * alpha - 3.0 * gamma)
    D[2, 9] = d_3_n4 * wp.cos(3.0 * alpha - 4.0 * gamma)
    D[2, 10] = d_3_n5 * wp.cos(3.0 * alpha - 5.0 * gamma)

    # Row 3: m=2
    D[3, 0] = d_2_5 * wp.cos(2.0 * alpha + 5.0 * gamma)
    D[3, 1] = d_2_4 * wp.cos(2.0 * alpha + 4.0 * gamma)
    D[3, 2] = d_2_3 * wp.cos(2.0 * alpha + 3.0 * gamma)
    D[3, 3] = d_2_2 * wp.cos(2.0 * alpha + 2.0 * gamma)
    D[3, 4] = d_2_1 * wp.cos(2.0 * alpha + gamma)
    D[3, 5] = d_2_0 * wp.cos(2.0 * alpha)
    D[3, 6] = d_2_n1 * wp.cos(2.0 * alpha - gamma)
    D[3, 7] = d_2_n2 * wp.cos(2.0 * alpha - 2.0 * gamma)
    D[3, 8] = d_2_n3 * wp.cos(2.0 * alpha - 3.0 * gamma)
    D[3, 9] = d_2_n4 * wp.cos(2.0 * alpha - 4.0 * gamma)
    D[3, 10] = d_2_n5 * wp.cos(2.0 * alpha - 5.0 * gamma)

    # Row 4: m=1
    D[4, 0] = d_1_5 * wp.cos(alpha + 5.0 * gamma)
    D[4, 1] = d_1_4 * wp.cos(alpha + 4.0 * gamma)
    D[4, 2] = d_1_3 * wp.cos(alpha + 3.0 * gamma)
    D[4, 3] = d_1_2 * wp.cos(alpha + 2.0 * gamma)
    D[4, 4] = d_1_1 * wp.cos(alpha + gamma)
    D[4, 5] = d_1_0 * wp.cos(alpha)
    D[4, 6] = d_1_n1 * wp.cos(alpha - gamma)
    D[4, 7] = d_1_n2 * wp.cos(alpha - 2.0 * gamma)
    D[4, 8] = d_1_n3 * wp.cos(alpha - 3.0 * gamma)
    D[4, 9] = d_1_n4 * wp.cos(alpha - 4.0 * gamma)
    D[4, 10] = d_1_n5 * wp.cos(alpha - 5.0 * gamma)

    # Row 5: m=0
    D[5, 0] = d_0_5 * wp.cos(5.0 * gamma)
    D[5, 1] = d_0_4 * wp.cos(4.0 * gamma)
    D[5, 2] = d_0_3 * wp.cos(3.0 * gamma)
    D[5, 3] = d_0_2 * wp.cos(2.0 * gamma)
    D[5, 4] = d_0_1 * wp.cos(gamma)
    D[5, 5] = d_0_0
    D[5, 6] = d_0_n1 * wp.cos(-gamma)
    D[5, 7] = d_0_n2 * wp.cos(-2.0 * gamma)
    D[5, 8] = d_0_n3 * wp.cos(-3.0 * gamma)
    D[5, 9] = d_0_n4 * wp.cos(-4.0 * gamma)
    D[5, 10] = d_0_n5 * wp.cos(-5.0 * gamma)

    # Row 6: m=-1
    D[6, 0] = d_n1_5 * wp.cos(-alpha + 5.0 * gamma)
    D[6, 1] = d_n1_4 * wp.cos(-alpha + 4.0 * gamma)
    D[6, 2] = d_n1_3 * wp.cos(-alpha + 3.0 * gamma)
    D[6, 3] = d_n1_2 * wp.cos(-alpha + 2.0 * gamma)
    D[6, 4] = d_n1_1 * wp.cos(-alpha + gamma)
    D[6, 5] = d_n1_0 * wp.cos(-alpha)
    D[6, 6] = d_n1_n1 * wp.cos(-alpha - gamma)
    D[6, 7] = d_n1_n2 * wp.cos(-alpha - 2.0 * gamma)
    D[6, 8] = d_n1_n3 * wp.cos(-alpha - 3.0 * gamma)
    D[6, 9] = d_n1_n4 * wp.cos(-alpha - 4.0 * gamma)
    D[6, 10] = d_n1_n5 * wp.cos(-alpha - 5.0 * gamma)

    # Row 7: m=-2
    D[7, 0] = d_n2_5 * wp.cos(-2.0 * alpha + 5.0 * gamma)
    D[7, 1] = d_n2_4 * wp.cos(-2.0 * alpha + 4.0 * gamma)
    D[7, 2] = d_n2_3 * wp.cos(-2.0 * alpha + 3.0 * gamma)
    D[7, 3] = d_n2_2 * wp.cos(-2.0 * alpha + 2.0 * gamma)
    D[7, 4] = d_n2_1 * wp.cos(-2.0 * alpha + gamma)
    D[7, 5] = d_n2_0 * wp.cos(-2.0 * alpha)
    D[7, 6] = d_n2_n1 * wp.cos(-2.0 * alpha - gamma)
    D[7, 7] = d_n2_n2 * wp.cos(-2.0 * alpha - 2.0 * gamma)
    D[7, 8] = d_n2_n3 * wp.cos(-2.0 * alpha - 3.0 * gamma)
    D[7, 9] = d_n2_n4 * wp.cos(-2.0 * alpha - 4.0 * gamma)
    D[7, 10] = d_n2_n5 * wp.cos(-2.0 * alpha - 5.0 * gamma)

    # Row 8: m=-3
    D[8, 0] = d_n3_5 * wp.cos(-3.0 * alpha + 5.0 * gamma)
    D[8, 1] = d_n3_4 * wp.cos(-3.0 * alpha + 4.0 * gamma)
    D[8, 2] = d_n3_3 * wp.cos(-3.0 * alpha + 3.0 * gamma)
    D[8, 3] = d_n3_2 * wp.cos(-3.0 * alpha + 2.0 * gamma)
    D[8, 4] = d_n3_1 * wp.cos(-3.0 * alpha + gamma)
    D[8, 5] = d_n3_0 * wp.cos(-3.0 * alpha)
    D[8, 6] = d_n3_n1 * wp.cos(-3.0 * alpha - gamma)
    D[8, 7] = d_n3_n2 * wp.cos(-3.0 * alpha - 2.0 * gamma)
    D[8, 8] = d_n3_n3 * wp.cos(-3.0 * alpha - 3.0 * gamma)
    D[8, 9] = d_n3_n4 * wp.cos(-3.0 * alpha - 4.0 * gamma)
    D[8, 10] = d_n3_n5 * wp.cos(-3.0 * alpha - 5.0 * gamma)

    # Row 9: m=-4
    D[9, 0] = d_n4_5 * wp.cos(-4.0 * alpha + 5.0 * gamma)
    D[9, 1] = d_n4_4 * wp.cos(-4.0 * alpha + 4.0 * gamma)
    D[9, 2] = d_n4_3 * wp.cos(-4.0 * alpha + 3.0 * gamma)
    D[9, 3] = d_n4_2 * wp.cos(-4.0 * alpha + 2.0 * gamma)
    D[9, 4] = d_n4_1 * wp.cos(-4.0 * alpha + gamma)
    D[9, 5] = d_n4_0 * wp.cos(-4.0 * alpha)
    D[9, 6] = d_n4_n1 * wp.cos(-4.0 * alpha - gamma)
    D[9, 7] = d_n4_n2 * wp.cos(-4.0 * alpha - 2.0 * gamma)
    D[9, 8] = d_n4_n3 * wp.cos(-4.0 * alpha - 3.0 * gamma)
    D[9, 9] = d_n4_n4 * wp.cos(-4.0 * alpha - 4.0 * gamma)
    D[9, 10] = d_n4_n5 * wp.cos(-4.0 * alpha - 5.0 * gamma)

    # Row 10: m=-5
    D[10, 0] = d_n5_5 * wp.cos(-5.0 * alpha + 5.0 * gamma)
    D[10, 1] = d_n5_4 * wp.cos(-5.0 * alpha + 4.0 * gamma)
    D[10, 2] = d_n5_3 * wp.cos(-5.0 * alpha + 3.0 * gamma)
    D[10, 3] = d_n5_2 * wp.cos(-5.0 * alpha + 2.0 * gamma)
    D[10, 4] = d_n5_1 * wp.cos(-5.0 * alpha + gamma)
    D[10, 5] = d_n5_0 * wp.cos(-5.0 * alpha)
    D[10, 6] = d_n5_n1 * wp.cos(-5.0 * alpha - gamma)
    D[10, 7] = d_n5_n2 * wp.cos(-5.0 * alpha - 2.0 * gamma)
    D[10, 8] = d_n5_n3 * wp.cos(-5.0 * alpha - 3.0 * gamma)
    D[10, 9] = d_n5_n4 * wp.cos(-5.0 * alpha - 4.0 * gamma)
    D[10, 10] = d_n5_n5 * wp.cos(-5.0 * alpha - 5.0 * gamma)


@wp.kernel
def _compute_wigner_d(l: int, angles: wp.array(dtype=wp.vec3f)):
    idx = wp.tid()
    output = wp.zeros((2 * l + 1, 2 * l + 1), dtype=angles.dtype)
    alpha, beta, gamma = angles[idx, 0], angles[idx, 1], angles[idx, 2]
    if l == 0:
        wigner_d_l0(alpha, beta, gamma, output)
    elif l == 1:
        wigner_d_l1(alpha, beta, gamma, output)
    elif l == 2:
        wigner_d_l2(alpha, beta, gamma, output)
    elif l == 3:
        wigner_d_l3(alpha, beta, gamma, output)
    elif l == 4:
        wigner_d_l4(alpha, beta, gamma, output)
    else:
        wigner_d_l5(alpha, beta, gamma, output)
    return output
