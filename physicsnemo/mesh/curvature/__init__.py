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

"""Curvature computation for simplicial meshes.

This module provides discrete differential geometry tools for computing
intrinsic and extrinsic curvatures on n-dimensional simplicial manifolds.

Gaussian Curvature (Intrinsic):
- Angle defect method: K = (full_angle - Σ angles) / voronoi_area
- Works for any codimension (intrinsic property)
- Measures intrinsic geometry (Theorema Egregium)

Mean Curvature (Extrinsic):
- Cotangent Laplacian method: H = ||L @ points|| / (2 * voronoi_area)
- Requires codimension-1 (needs normal vectors)
- Measures extrinsic bending

Example:
    >>> from physicsnemo.mesh.curvature import gaussian_curvature_vertices, mean_curvature_vertices
    >>> from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral
    >>> mesh = sphere_icosahedral.load(subdivisions=2)
    >>>
    >>> # Compute Gaussian curvature
    >>> K = gaussian_curvature_vertices(mesh)
    >>>
    >>> # Compute mean curvature (codimension-1 only)
    >>> H = mean_curvature_vertices(mesh)
    >>>
    >>> # Or use Mesh properties:
    >>> K = mesh.gaussian_curvature_vertices
    >>> H = mesh.mean_curvature_vertices
"""

from physicsnemo.mesh.curvature.gaussian import (
    gaussian_curvature_cells,
    gaussian_curvature_vertices,
)
from physicsnemo.mesh.curvature.mean import mean_curvature_vertices
