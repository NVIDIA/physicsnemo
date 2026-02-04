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

r"""
Mesh validation utilities for geometry guardrails.

These checks aim to reject corrupted or degenerate geometries before
feature extraction. Validation is intentionally conservative to prevent
downstream numerical issues.
"""

from __future__ import annotations

import numpy as np
import trimesh


def validate_mesh(mesh: trimesh.Trimesh, min_verts: int = 50) -> None:
    r"""
    Validate basic geometric integrity of a mesh.

    This function performs conservative checks to detect corrupted or
    degenerate geometries that could cause issues during feature extraction
    or downstream processing.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Input triangular surface mesh to validate.
    min_verts : int, optional
        Minimum number of vertices required for a valid mesh. Defaults to 50.
        This ensures sufficient geometry for statistical feature extraction.

    Raises
    ------
    ValueError
        If ``mesh`` is not a :class:`trimesh.Trimesh` instance.
    ValueError
        If the mesh contains fewer than ``min_verts`` vertices.
    ValueError
        If any vertex coordinates are non-finite (NaN or Inf).
    ValueError
        If the mesh surface area is non-positive.

    Examples
    --------
    >>> import trimesh
    >>> import numpy as np
    >>> from physicsnemo.experimental.guardrails.geometry import validate_mesh
    >>> 
    >>> # Create a simple valid mesh (cube)
    >>> mesh = trimesh.creation.box()
    >>> validate_mesh(mesh)  # Should pass without error
    >>> 
    >>> # Create invalid mesh with too few vertices
    >>> vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    >>> faces = np.array([[0, 1, 2]])
    >>> invalid_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    >>> try:
    ...     validate_mesh(invalid_mesh)
    ... except ValueError as e:
    ...     print(f"Validation failed: {e}")
    Validation failed: Too few vertices

    Notes
    -----
    The validation checks are intentionally strict to prevent subtle issues
    during feature extraction and density modeling. Meshes that fail validation
    should be inspected and potentially repaired before use.

    See Also
    --------
    :func:`extract_features` : Extract geometric features from validated meshes.
    """
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("Object is not a Trimesh")

    if mesh.vertices.shape[0] < min_verts:
        raise ValueError(
            f"Too few vertices: {mesh.vertices.shape[0]} < {min_verts}"
        )

    if not np.isfinite(mesh.vertices).all():
        raise ValueError("Non-finite vertex coordinates")

    if mesh.area <= 0:
        raise ValueError("Non-positive surface area")
