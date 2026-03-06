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

Provides helper functions for computing angles, full angles in n-dimensions,
and numerically stable geometric operations.
"""

import math

import torch


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


def stable_angle_between_vectors(v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
    """Compute angle between vectors using numerically stable atan2 formula.

    More stable than using acos(dot product) which suffers from numerical
    issues when vectors are nearly parallel or anti-parallel.

    Parameters
    ----------
    v1 : torch.Tensor
        First vector(s), shape (..., n_dims)
    v2 : torch.Tensor
        Second vector(s), shape (..., n_dims)

    Returns
    -------
    torch.Tensor
        Angle(s) in radians, shape (...)
        Range: [0, π]

    Formula:
        angle = atan2(||v1 × v2||, v1 · v2)

    For higher dimensions (>3), uses generalized cross product magnitude.
    """
    ### Compute dot product
    dot_product = (v1 * v2).sum(dim=-1)

    ### Compute cross product magnitude (generalized)
    # For 2D/3D: ||v1 × v2|| = ||v1|| * ||v2|| * sin(θ)
    # More generally: ||v1|| * ||v2|| * sin(θ) = sqrt(||v1||² * ||v2||² - (v1·v2)²)
    v1_norm = torch.linalg.vector_norm(v1, dim=-1)
    v2_norm = torch.linalg.vector_norm(v2, dim=-1)

    cross_magnitude_sq = v1_norm**2 * v2_norm**2 - dot_product**2
    # Clamp to avoid numerical issues with negative values near zero
    cross_magnitude_sq = torch.clamp(cross_magnitude_sq, min=0)
    cross_magnitude = torch.sqrt(cross_magnitude_sq)

    ### Compute angle using atan2 (stable)
    angle = torch.atan2(cross_magnitude, dot_product)

    return angle


def compute_triangle_angles(
    p0: torch.Tensor,
    p1: torch.Tensor,
    p2: torch.Tensor,
) -> torch.Tensor:
    """Compute the angle at p0 in triangle (p0, p1, p2) using stable formula.

    Uses atan2-based computation for numerical stability.

    Parameters
    ----------
    p0 : torch.Tensor
        Vertex at which to compute angle, shape (..., n_spatial_dims)
    p1 : torch.Tensor
        Second vertex, shape (..., n_spatial_dims)
    p2 : torch.Tensor
        Third vertex, shape (..., n_spatial_dims)

    Returns
    -------
    torch.Tensor
        Angle at p0 in radians, shape (...)

    Examples
    --------
        >>> # Right angle at origin
        >>> p0 = torch.tensor([0., 0.])
        >>> p1 = torch.tensor([1., 0.])
        >>> p2 = torch.tensor([0., 1.])
        >>> angle = compute_triangle_angles(p0, p1, p2)
        >>> assert torch.allclose(angle, torch.tensor(torch.pi / 2))
    """
    ### Compute edge vectors from p0
    edge1 = p1 - p0  # (..., n_spatial_dims)
    edge2 = p2 - p0  # (..., n_spatial_dims)

    ### Compute angle using stable formula
    angle = stable_angle_between_vectors(edge1, edge2)

    return angle
