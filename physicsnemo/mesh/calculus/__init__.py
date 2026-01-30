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

"""Discrete calculus operators for simplicial meshes.

This module implements discrete differential operators using both:
1. Discrete Exterior Calculus (DEC) - rigorous differential geometry framework
2. Weighted Least-Squares (LSQ) reconstruction - standard CFD approach

The DEC implementation follows Desbrun, Hirani, Leok, and Marsden's seminal work
on discrete exterior calculus (arXiv:math/0508341v2).

Key operators:
- Gradient: ∇φ (scalar → vector)
- Divergence: div(v) (vector → scalar)
- Curl: curl(v) (vector → vector, 3D only)
- Laplacian: Δφ (scalar → scalar, Laplace-Beltrami operator)

Both intrinsic (manifold tangent space) and extrinsic (ambient space) derivatives
are supported for manifolds embedded in higher-dimensional spaces.
"""

from physicsnemo.mesh.calculus.derivatives import (
    compute_cell_derivatives,
    compute_point_derivatives,
)

__all__ = [
    "compute_point_derivatives",
    "compute_cell_derivatives",
]
