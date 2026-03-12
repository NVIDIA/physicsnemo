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
# ruff: noqa: E402

"""Tests for geometry guardrail mesh validation module."""

import numpy as np
import pytest

pytest.importorskip("pyvista")


import pyvista as pv

from physicsnemo.experimental.guardrails.geometry import validate_mesh
from physicsnemo.mesh.io.io_pyvista import from_pyvista


def test_validate_mesh_valid():
    """Test validation of a valid mesh."""
    mesh = from_pyvista(pv.Cube())
    # Should not raise
    validate_mesh(mesh)


def test_validate_mesh_min_verts():
    """Test minimum vertices validation."""
    # Create simple mesh with few vertices
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    pv_mesh = pv.PolyData(vertices, faces=[3, 0, 1, 2])
    mesh = from_pyvista(pv_mesh)

    # Should fail with default min_verts=4
    with pytest.raises(ValueError, match="Too few vertices"):
        validate_mesh(mesh, min_verts=10)

    # Should pass with lower threshold
    validate_mesh(mesh, min_verts=3)


def test_validate_mesh_non_finite_vertices():
    """Test detection of non-finite vertex coordinates."""
    pv_mesh = pv.Cube()
    # Corrupt vertices with NaN
    points = pv_mesh.points.copy()
    points[0, 0] = np.nan
    pv_mesh.points = points
    mesh = from_pyvista(pv_mesh)

    with pytest.raises(ValueError, match="Non-finite"):
        validate_mesh(mesh)


def test_validate_mesh_zero_area():
    """Test detection of zero surface area."""
    # Create degenerate mesh (all vertices colinear)
    vertices = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], dtype=np.float32)
    # Create faces array for PyVista format [n, i0, i1, i2, n, i0, i1, i2]
    pv_faces = np.array([3, 0, 1, 2, 3, 1, 2, 3], dtype=np.int32)
    pv_mesh = pv.PolyData(vertices, faces=pv_faces)
    mesh = from_pyvista(pv_mesh)

    with pytest.raises(ValueError, match="Non-positive"):
        validate_mesh(mesh)


def test_validate_mesh_not_physicsnemo_mesh():
    """Test rejection of non-PhysicsNeMo mesh objects."""
    with pytest.raises(ValueError, match="Object is not a physicsnemo.mesh.Mesh"):
        validate_mesh("not a mesh")

    with pytest.raises(ValueError, match="Object is not a physicsnemo.mesh.Mesh"):
        validate_mesh(None)


@pytest.mark.parametrize("min_verts", [10, 50, 100])
def test_validate_mesh_min_verts_parametric(min_verts):
    """Test various minimum vertex thresholds."""
    # Create mesh with known vertex count
    mesh = from_pyvista(pv.Sphere())

    if mesh.n_points >= min_verts:
        validate_mesh(mesh, min_verts=min_verts)
    else:
        with pytest.raises(ValueError, match="Too few vertices"):
            validate_mesh(mesh, min_verts=min_verts)
