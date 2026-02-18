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

"""2D manifolds in 3D space (surface meshes).

Includes spheres, cylinders, tori, platonic solids, and various parametric
surfaces embedded in 3D space.
"""

from physicsnemo.mesh.primitives.surfaces import (
    cone,
    cube_surface,
    cylinder,
    cylinder_open,
    disk,
    hemisphere,
    icosahedron_surface,
    mobius_strip,
    octahedron_surface,
    plane,
    sphere_icosahedral,
    sphere_uv,
    tetrahedron_surface,
    torus,
)
