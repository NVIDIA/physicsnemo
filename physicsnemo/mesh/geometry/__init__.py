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

"""Geometric primitives and computations for simplicial meshes.

This module contains fundamental geometric operations that are shared across
the codebase, including:
- Dual mesh (Voronoi/circumcentric) computations
- Circumcenter calculations
- Support volume computations (for DEC)
- Geometric utility functions

These are used by both DEC operators (calculus module) and differential
geometry computations (curvature module).
"""

from physicsnemo.mesh.geometry.dual_meshes import compute_dual_volumes_0
