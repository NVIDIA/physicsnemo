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
    """Compute the full angle around a point in an n-dimensional manifold.

    This is the total solid angle/turning angle available at a point.

    For discrete differential geometry:
    - 1D curves: Full turning angle is π (can turn left or right from straight)
    - 2D surfaces: Full angle is 2π (can look 360° around a point)
    - 3D volumes: Full solid angle is 4π (full sphere around a point)
    - nD: Surface area of unit (n-1)-sphere

    Parameters
    ----------
    n_manifold_dims : int
        Manifold dimension

    Returns
    -------
    float
        Full angle for n-dimensional manifold:
        - 1D: π
        - 2D: 2π
        - 3D: 4π
        - nD: 2π^(n/2) / Γ(n/2) for n ≥ 2

    Examples
    --------
        >>> import math
        >>> assert abs(compute_full_angle_n_sphere(1) - math.pi) < 1e-10  # π
        >>> assert abs(compute_full_angle_n_sphere(2) - 2*math.pi) < 1e-5  # 2π
    """

    ### Special case for 1D: turning angle is π
    if n_manifold_dims == 1:
        return math.pi

    ### General case (n ≥ 2): Surface area of (n-1)-sphere
    # Formula: 2π^(n/2) / Γ(n/2)
    n = n_manifold_dims
    return 2 * math.pi ** (n / 2.0) / math.exp(math.lgamma(n / 2.0))
