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

"""Uniform mesh remeshing via clustering.

This module provides dimension-agnostic uniform remeshing based on the ACVD
(Approximate Centroidal Voronoi Diagram) clustering algorithm. It works for
arbitrary n-dimensional simplicial manifolds.

The algorithm:
1. Weights vertices by incident cell areas
2. Initializes clusters via area-based region growing
3. Removes spatially isolated cluster regions
4. Reconstructs a simplified mesh from cluster adjacency

The output mesh has approximately uniform cell distribution with ~0.5% non-manifold
edges (multiple faces sharing an edge), which is inherent to the face-mapping approach.

Current limitations:
- Energy minimization is disabled (made topology worse; needs investigation)
- Small percentage (~0.5-1%) of edges may be non-manifold with moderate cluster counts
- Higher cluster counts relative to mesh resolution produce better manifold quality

Example:
    >>> from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral
    >>> mesh = sphere_icosahedral.load(subdivisions=3)
    >>> # Remesh a triangle mesh to ~100 triangles
    >>> remeshed = remesh(mesh, n_clusters=100)
    >>> assert remeshed.n_cells > 0
"""

from physicsnemo.mesh.remeshing._remeshing import remesh
