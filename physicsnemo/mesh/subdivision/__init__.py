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

"""Mesh subdivision algorithms for simplicial meshes.

This module provides subdivision schemes for refining simplicial meshes:
- Linear: Simple midpoint subdivision (interpolating)
- Butterfly: Weighted stencil subdivision for smooth surfaces (interpolating)
- Loop: Valence-based subdivision with vertex repositioning (approximating)

All schemes work by:
1. Extracting edges from the mesh
2. Adding new vertices (at or near edge midpoints)
3. Splitting each n-simplex into 2^n child simplices
4. Interpolating/propagating data to new mesh

Example:
    >>> from physicsnemo.mesh.subdivision import subdivide_linear
    >>> from physicsnemo.mesh.primitives.basic import two_triangles_2d
    >>> mesh = two_triangles_2d.load()
    >>> subdivided = subdivide_linear(mesh)
    >>> assert subdivided.n_cells == mesh.n_cells * 4  # 2^2 for 2D
"""

from physicsnemo.mesh.subdivision.butterfly import subdivide_butterfly
from physicsnemo.mesh.subdivision.linear import subdivide_linear
from physicsnemo.mesh.subdivision.loop import subdivide_loop
