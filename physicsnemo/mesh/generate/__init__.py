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

"""Mesh generation from implicit representations.

This module provides functions for generating meshes from scalar fields
and implicit functions. It supports isosurface extraction using marching
cubes and volume mesh generation for implicit domains in any spatial
dimension through ``mesh_implicit_domain``. It also includes a
differentiable geometry refit (``refit_mesh_to_implicit``) and
signed-distance building blocks.
"""

from physicsnemo.mesh.generate.implicit_domain import (
    mesh_implicit_domain,
    refit_mesh_to_implicit,
)
from physicsnemo.mesh.generate.implicit_functions import (
    project_to_zero_set,
    sdf_box,
    sdf_difference,
    sdf_intersection,
    sdf_polygon_2d,
    sdf_sphere,
    sdf_union,
)
from physicsnemo.mesh.generate.marching_cubes import marching_cubes
