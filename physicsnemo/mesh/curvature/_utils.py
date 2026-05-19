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

"""Utility functions for curvature computations.

Provides helper functions for computing full angles in n-dimensions.
"""

import math


def compute_full_angle_n_sphere(n_manifold_dims: int) -> float:
    r"""Compute the full angle around a point in an n-dimensional manifold.

    This is the total solid angle / turning angle available at a point.

    For discrete differential geometry:

    - 1D curves: full turning angle is :math:`\pi` (can turn left or right
      from straight).
    - 2D surfaces: full angle is :math:`2\pi` (can look 360 degrees around
      a point).
    - 3D volumes: full solid angle is :math:`4\pi` (full sphere around a
      point).
    - nD: surface area of the unit :math:`(n-1)`-sphere.

    Parameters
    ----------
    n_manifold_dims : int
        Manifold dimension.

    Returns
    -------
    float
        Full angle for an :math:`n`-dimensional manifold:

        - 1D: :math:`\pi`.
        - 2D: :math:`2\pi`.
        - 3D: :math:`4\pi`.
        - nD: :math:`2 \pi^{n/2} / \Gamma(n/2)` for :math:`n \ge 2`.

    Examples
    --------
    >>> import math
    >>> assert abs(compute_full_angle_n_sphere(1) - math.pi) < 1e-10
    >>> assert abs(compute_full_angle_n_sphere(2) - 2 * math.pi) < 1e-10
    """

    ### Special case for 1D: turning angle is pi
    if n_manifold_dims == 1:
        return math.pi

    ### General case (n >= 2): surface area of (n-1)-sphere
    # Formula: 2 pi^(n/2) / Gamma(n/2)
    n = n_manifold_dims
    return 2 * math.pi ** (n / 2.0) / math.exp(math.lgamma(n / 2.0))
