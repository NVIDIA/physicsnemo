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

from .indexing import index_clamped_edges_batched_2d, index_zero_edges_batched_2d


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

    # compute darcy stensil
    d_0_1 = index_zero_edges_batched_2d(
        darcy0, b, gx - mgrid_reduction_factor, gy, lx, ly
    )
    d_2_1 = index_zero_edges_batched_2d(
        darcy0, b, gx + mgrid_reduction_factor, gy, lx, ly
    )
    d_1_0 = index_zero_edges_batched_2d(
        darcy0, b, gx, gy - mgrid_reduction_factor, lx, ly
    )
    d_1_2 = index_zero_edges_batched_2d(
        darcy0, b, gx, gy + mgrid_reduction_factor, lx, ly
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
    t_2 = ((p_2_1 - p_0_1) * (d_2_1 - d_0_1)) / (2.0 * gdx)
    t_3 = ((p_1_2 - p_1_0) * (d_1_2 - d_1_0)) / (2.0 * gdx)

    # jacobi iterative method
    d_star = (t_1 + t_2 + t_3 + source) / (p_1_1 * 4.0 / dx_squared)

    # buffers get swapped each iteration
    darcy1[b, gx, gy] = d_star


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
        Array to hold infinity norm value in
    mgrid_reduction_factor : int
        Current multi-grid running at
    """
    b, x, y = wp.tid()
    gx = mgrid_reduction_factor * x + (mgrid_reduction_factor - 1)
    gy = mgrid_reduction_factor * y + (mgrid_reduction_factor - 1)
    wp.atomic_max(inf_res, 0, wp.abs(phi0[b, gx, gy] - phi1[b, gx, gy]))
