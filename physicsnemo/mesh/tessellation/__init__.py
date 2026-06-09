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

"""Tessellation: decompose non-simplicial cells into a simplex mesh.

Exposes :func:`triangulate`, which converts a polygon soup (an ``Adjacency`` of
cell-to-vertex rings) into a triangle connectivity, staying correct for
non-convex polygons. It branches on manifold dimension and currently implements
the 2D (polygon -> triangle) case.
"""

from physicsnemo.mesh.tessellation.triangulate import triangulate

__all__ = ["triangulate"]
