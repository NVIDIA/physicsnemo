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

from __future__ import annotations

import numpy as np

from physicsnemo.mesh import Mesh as PhysicsNeMoMesh


def validate_mesh(mesh: PhysicsNeMoMesh, min_verts: int = 4) -> None:
    r"""
    Validate basic geometric integrity of a mesh.

    This function performs conservative checks to detect corrupted or
    degenerate geometries that could cause issues during feature extraction
    or downstream processing.

    Parameters
    ----------
    mesh : physicsnemo.mesh.Mesh
        Input triangular surface mesh to validate.
    min_verts : int, optional
        Minimum number of vertices required for a valid mesh. Defaults to 4.
        This ensures sufficient geometry for statistical feature extraction.

    Raises
    ------
    ValueError
        If ``mesh`` is not a :class:`physicsnemo.mesh.Mesh` instance.
    ValueError
        If the mesh contains fewer than ``min_verts`` vertices.
    ValueError
        If any vertex coordinates are non-finite (NaN or Inf).
    ValueError
        If the mesh surface area is non-positive.
    """
    if not isinstance(mesh, PhysicsNeMoMesh):
        raise ValueError("Object is not a physicsnemo.mesh.Mesh")

    if mesh.n_points < min_verts:
        raise ValueError(
            f"Too few vertices: {mesh.n_points} < {min_verts}"
        )

    # Check for non-finite vertex coordinates
    verts = mesh.points.cpu().numpy()
    if not np.isfinite(verts).all():
        raise ValueError("Non-finite vertex coordinates")

    # Check surface area
    # Some meshes may have degenerate triangles (zero area), which is acceptable
    # as long as there are some valid triangles with positive area
    cell_areas = mesh.cell_areas
    total_area = float(cell_areas.sum().item())
    
    if total_area <= 0:
        # Provide more detailed error message
        n_zero_area = int((cell_areas <= 0).sum().item())
        n_valid = mesh.n_cells - n_zero_area
        raise ValueError(
            f"Non-positive surface area: {total_area:.2e}. "
            f"Valid cells: {n_valid}/{mesh.n_cells}, "
            f"Degenerate cells: {n_zero_area}/{mesh.n_cells}"
        )
