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

"""Example meshes for physicsnemo.mesh.

This module provides a comprehensive collection of canonical meshes organized by
category and dimensional configuration. All meshes are generated at runtime and
can be used for tutorials, testing, and experimentation.

Categories:
    - basic: Minimal test meshes (single cells, few cells)
    - curves: 1D manifolds in 1D, 2D, and 3D spaces
    - planar: 2D manifolds in 2D space (triangle meshes)
    - surfaces: 2D manifolds in 3D space (surface meshes)
    - volumes: 3D manifolds in 3D space (tetrahedral volumes)
    - procedural: Procedurally generated mesh variations
    - pyvista_datasets: Wrappers for PyVista example datasets (requires pyvista)

Usage:
    >>> from physicsnemo.mesh.primitives import surfaces






    >>> mesh = surfaces.sphere_uv.load(radius=1.0, theta_resolution=10, phi_resolution=10)
    >>> from physicsnemo.mesh.primitives import pyvista_datasets
    >>> mesh = pyvista_datasets.bunny.load()  # doctest: +SKIP
"""

from physicsnemo.mesh.primitives import (
    basic,
    curves,
    planar,
    procedural,
    pyvista_datasets,
    surfaces,
    volumes,
)
