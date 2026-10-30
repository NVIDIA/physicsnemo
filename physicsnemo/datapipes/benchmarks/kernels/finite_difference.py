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

from .indexing import index_clamped_edges_batched_2d, index_zero_edges_batched_2d


@wp.func
def darcy_point_update_batched_2d(
    darcy: wp.array3d(dtype=float),
    permeability: wp.array3d(dtype=float),
    b: int,
    gx: int,
    gy: int,
    source: float,
    lx: int,
    ly: int,
    gdx: float,
    mgrid_reduction_factor: int,
):  # pragma: no cover
    """Value at one grid point that zeroes its local Darcy residual.

    Solves ``k lap(u) + grad(k) . grad(u) + source = 0`` for the center value,
    holding the four neighbors (``mgrid_reduction_factor`` cells away) fixed.

    Parameters
    ----------
    darcy : wp.array3d
        Current Darcy solution
    permeability : wp.array3d
        Permeability field for Darcy equation
    b : int
        Batch index
    gx : int
        X index on the full grid
    gy : int
        Y index on the full grid
    source : float
        Source value for Darcy equation
    lx : int
        Length of domain in x dim
    ly : int
        Length of domain in y dim
    gdx : float
        Grid cell size at the current multi-grid level
    mgrid_reduction_factor : int
        Current multi-grid running at

    Returns
    -------
    float
        Updated center value
    """
    # compute darcy stensil
    d_0_1 = index_zero_edges_batched_2d(
        darcy, b, gx - mgrid_reduction_factor, gy, lx, ly
    )
    d_2_1 = index_zero_edges_batched_2d(
        darcy, b, gx + mgrid_reduction_factor, gy, lx, ly
    )
    d_1_0 = index_zero_edges_batched_2d(
        darcy, b, gx, gy - mgrid_reduction_factor, lx, ly
    )
    d_1_2 = index_zero_edges_batched_2d(
        darcy, b, gx, gy + mgrid_reduction_factor, lx, ly
    )

    # compute permeability stensil
    p_1_1 = index_clamped_edges_batched_2d(permeability, b, gx, gy, lx, ly)
    p_0_1 = index_clamped_edges_batched_2d(
        permeability, b, gx - mgrid_reduction_factor, gy, lx, ly
    )
    p_2_1 = index_clamped_edges_batched_2d(
        permeability, b, gx + mgrid_reduction_factor, gy, lx, ly
    )
    p_1_0 = index_clamped_edges_batched_2d(
        permeability, b, gx, gy - mgrid_reduction_factor, lx, ly
    )
    p_1_2 = index_clamped_edges_batched_2d(
        permeability, b, gx, gy + mgrid_reduction_factor, lx, ly
    )

    # compute terms
    dx_squared = gdx * gdx
    t_1 = p_1_1 * (d_0_1 + d_2_1 + d_1_0 + d_1_2) / dx_squared
    # product of two central differences, each over 2 * gdx
    t_2 = ((p_2_1 - p_0_1) * (d_2_1 - d_0_1)) / (4.0 * dx_squared)
    t_3 = ((p_1_2 - p_1_0) * (d_1_2 - d_1_0)) / (4.0 * dx_squared)

    return (t_1 + t_2 + t_3 + source) / (p_1_1 * 4.0 / dx_squared)


@wp.kernel
def darcy_mgrid_jacobi_iterative_batched_2d(
    darcy0: wp.array3d(dtype=float),
    darcy1: wp.array3d(dtype=float),
    permeability: wp.array3d(dtype=float),
    source: float,
    lx: int,
    ly: int,
    dx: float,
    mgrid_reduction_factor: int,
):  # pragma: no cover
    """Mult-grid jacobi step for Darcy equation.

    Parameters
    ----------
    darcy0 : wp.array3d
        Darcy solution previous step
    darcy1 : wp.array3d
        Darcy solution for next step
    permeability : wp.array3d
        Permeability field for Darcy equation
    source : float
        Source value for Darcy equation
    lx : int
        Length of domain in x dim
    ly : int
        Length of domain in y dim
    dx : float
        Grid cell size
    mgrid_reduction_factor : int
        Current multi-grid running at
    """

    # get index
    b, x, y = wp.tid()

    # update index from grid reduction factor
    gx = mgrid_reduction_factor * x + (mgrid_reduction_factor - 1)
    gy = mgrid_reduction_factor * y + (mgrid_reduction_factor - 1)
    gdx = dx * wp.float32(mgrid_reduction_factor)

    # buffers get swapped each iteration
    darcy1[b, gx, gy] = darcy_point_update_batched_2d(
        darcy0, permeability, b, gx, gy, source, lx, ly, gdx, mgrid_reduction_factor
    )


@wp.kernel
def darcy_mgrid_rbsor_batched_2d(
    darcy: wp.array3d(dtype=float),
    permeability: wp.array3d(dtype=float),
    source: float,
    lx: int,
    ly: int,
    dx: float,
    mgrid_reduction_factor: int,
    color: int,
    omega: float,
):  # pragma: no cover
    """Multi-grid red-black SOR half-sweep for Darcy equation.

    Updates, in place, the points of one checkerboard color on the current
    multi-grid level. The stencil only couples points of opposite color, so a
    half-sweep has no read/write races. A full iteration is a ``color=0`` launch
    followed by a ``color=1`` launch.

    Parameters
    ----------
    darcy : wp.array3d
        Darcy solution, updated in place
    permeability : wp.array3d
        Permeability field for Darcy equation
    source : float
        Source value for Darcy equation
    lx : int
        Length of domain in x dim
    ly : int
        Length of domain in y dim
    dx : float
        Grid cell size
    mgrid_reduction_factor : int
        Current multi-grid running at
    color : int
        Checkerboard color to update, 0 or 1, on the multi-grid level's indices
    omega : float
        Over-relaxation factor, between 1 and 2 for SOR
    """

    # get index
    b, x, y = wp.tid()
    if (x + y) % 2 != color:
        return

    # update index from grid reduction factor
    gx = mgrid_reduction_factor * x + (mgrid_reduction_factor - 1)
    gy = mgrid_reduction_factor * y + (mgrid_reduction_factor - 1)
    gdx = dx * wp.float32(mgrid_reduction_factor)

    d_star = darcy_point_update_batched_2d(
        darcy, permeability, b, gx, gy, source, lx, ly, gdx, mgrid_reduction_factor
    )
    darcy[b, gx, gy] = (1.0 - omega) * darcy[b, gx, gy] + omega * d_star


@wp.kernel
def mgrid_inf_residual_batched_2d(
    phi0: wp.array3d(dtype=float),
    phi1: wp.array3d(dtype=float),
    inf_res: wp.array(dtype=float),
    mgrid_reduction_factor: int,
):  # pragma: no cover
    """Infinity norm for checking multi-grid solutions.

    Parameters
    ----------
    phi0 : wp.array3d
        Previous solution
    phi1 : wp.array3d
        Current solution
    inf_res : wp.array
        Array to hold infinity norm value in. Non-finite differences are
        reported as ``1e30``, since NaN never wins an ``atomic_max``.
    mgrid_reduction_factor : int
        Current multi-grid running at
    """
    b, x, y = wp.tid()
    gx = mgrid_reduction_factor * x + (mgrid_reduction_factor - 1)
    gy = mgrid_reduction_factor * y + (mgrid_reduction_factor - 1)
    diff = wp.abs(phi0[b, gx, gy] - phi1[b, gx, gy])
    if not wp.isfinite(diff):
        diff = 1.0e30
    wp.atomic_max(inf_res, 0, diff)
