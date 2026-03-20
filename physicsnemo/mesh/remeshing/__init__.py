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

"""Mesh remeshing and cell partitioning.

This module provides two complementary algorithms for mesh coarsening:

**Cell partitioning** (:func:`partition_cells`):
    Assigns each cell of a fine mesh to its nearest seed point (by Euclidean
    centroid distance) and accumulates area, normal, and centroid per cluster.
    This is a single-step discrete approximation to the restricted Voronoi
    diagram on the surface.  Pure PyTorch, no external dependencies.

**ACVD remeshing** (:func:`remesh`):
    Iterative Approximate Centroidal Voronoi Diagram clustering that creates
    a new mesh topology with approximately uniform cell distribution.
    Requires the ``pyacvd`` package.

``partition_cells`` is also a natural building block for a pure-PyTorch CVT
(Lloyd's algorithm iterates: partition -> move seeds to cluster centroids ->
repeat).

Example:
    >>> from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral
    >>> mesh = sphere_icosahedral.load(subdivisions=3)
    >>> # Remesh a triangle mesh to ~100 triangles
    >>> remeshed = remesh(mesh, n_clusters=100)
    >>> assert remeshed.n_cells > 0
"""

from physicsnemo.mesh.remeshing._partition import CellPartition, partition_cells
from physicsnemo.mesh.remeshing._remeshing import remesh
