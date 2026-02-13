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

import tempfile
import numpy as np
import pyvista as pv
import pytest
from pathlib import Path

# Import functions from vtp_reader
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from vtp_reader import (
    load_vtp_file,
    extract_mesh_connectivity_from_polydata,
    build_edges_from_mesh_connectivity,
)


@pytest.fixture
def simple_vtp_file():
    """Create a simple VTP file for testing."""
    # Create a simple quad mesh (2x2 grid)
    points = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
        ],
        dtype=np.uint8,
    )

    # Single quad cell
    faces = np.array([4, 0, 1, 3, 2])  # quad with 4 vertices

    mesh = pv.PolyData(points, faces, force_float=False)

    # Add displacement fields for 3 timesteps
    mesh.point_data["displacement_t0.000"] = np.array(
        [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ],
        dtype=np.uint8,
    )

    mesh.point_data["displacement_t0.005"] = np.array(
        [
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
        ],
        dtype=np.uint8,
    )

    mesh.point_data["displacement_t0.010"] = np.array(
        [
            [2, 0, 0],
            [2, 0, 0],
            [2, 0, 0],
            [2, 0, 0],
        ],
        dtype=np.uint8,
    )

    # Add thickness as additional point data
    mesh.point_data["thickness"] = np.array([1, 1, 1, 1], dtype=np.uint8)

    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix=".vtp", delete=False) as f:
        temp_path = f.name

    mesh.save(temp_path)
    yield temp_path

    # Cleanup
    Path(temp_path).unlink(missing_ok=True)


def test_load_vtp_file_basic(simple_vtp_file):
    """Test basic VTP file loading."""
    pos_raw, mesh_connectivity, point_data_dict = load_vtp_file(simple_vtp_file)

    # Check positions shape: (timesteps, nodes, 3)
    assert pos_raw.shape == (3, 4, 3), f"Expected shape (3, 4, 3), got {pos_raw.shape}"

    # Check mesh connectivity
    assert len(mesh_connectivity) == 1, f"Expected 1 cell, got {len(mesh_connectivity)}"
    assert len(mesh_connectivity[0]) == 4, (
        f"Expected quad with 4 vertices, got {len(mesh_connectivity[0])}"
    )

    # Check point data dict contains thickness
    assert "thickness" in point_data_dict, "Thickness not found in point_data_dict"
    assert point_data_dict["thickness"].shape == (4,), (
        f"Expected thickness shape (4,), got {point_data_dict['thickness'].shape}"
    )


def test_load_vtp_file_displacements(simple_vtp_file):
    """Test that displacements are correctly applied."""
    pos_raw, _, _ = load_vtp_file(simple_vtp_file)

    # First timestep should be reference coords (displacement = 0)
    expected_t0 = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
        ]
    )
    np.testing.assert_array_almost_equal(pos_raw[0], expected_t0, decimal=5)

    # Second timestep should include displacement
    expected_t1 = expected_t0 + np.array([[1, 0, 0]] * 4)
    np.testing.assert_array_almost_equal(pos_raw[1], expected_t1, decimal=5)

    # Third timestep
    expected_t2 = expected_t0 + np.array([[2, 0, 0]] * 4)
    np.testing.assert_array_almost_equal(pos_raw[2], expected_t2, decimal=5)


def test_extract_mesh_connectivity():
    """Test mesh connectivity extraction from PolyData."""
    points = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
        ]
    )

    # Create a single quad
    faces = np.array([4, 0, 1, 2, 3])
    poly = pv.PolyData(points, faces, force_float=False)

    connectivity = extract_mesh_connectivity_from_polydata(poly)

    assert len(connectivity) == 1, f"Expected 1 cell, got {len(connectivity)}"
    assert len(connectivity[0]) == 4, f"Expected 4 vertices, got {len(connectivity[0])}"
    assert connectivity[0] == [0, 1, 2, 3], (
        f"Expected [0, 1, 2, 3], got {connectivity[0]}"
    )


def test_build_edges_from_mesh_connectivity():
    """Test edge building from mesh connectivity."""
    # Single quad: should produce 4 edges
    mesh_connectivity = [[0, 1, 2, 3]]
    edges = build_edges_from_mesh_connectivity(mesh_connectivity)

    expected_edges = {(0, 1), (1, 2), (2, 3), (0, 3)}
    assert edges == expected_edges, f"Expected {expected_edges}, got {edges}"


def test_point_data_extraction(simple_vtp_file):
    """Test that non-displacement point data is extracted correctly."""
    _, _, point_data_dict = load_vtp_file(simple_vtp_file)

    # Should have thickness
    assert "thickness" in point_data_dict, "Thickness not in point_data_dict"

    # Should NOT have displacement fields
    assert "displacement_t0.000" not in point_data_dict, (
        "Displacement fields should not be in point_data_dict"
    )
    assert "displacement_t0.005" not in point_data_dict, (
        "Displacement fields should not be in point_data_dict"
    )

    # Check thickness values
    expected_thickness = np.array([1, 1, 1, 1], dtype=np.uint8)
    np.testing.assert_array_almost_equal(
        point_data_dict["thickness"], expected_thickness, decimal=5
    )


def test_missing_displacement_fields():
    """Test that missing displacement fields raises appropriate error."""
    # Create VTP without displacement fields
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    faces = np.array([3, 0, 1, 2])
    mesh = pv.PolyData(points, faces, force_float=False)

    with tempfile.NamedTemporaryFile(suffix=".vtp", delete=False) as f:
        temp_path = f.name

    mesh.save(temp_path)

    try:
        with pytest.raises(ValueError, match="No displacement fields found"):
            load_vtp_file(temp_path)
    finally:
        Path(temp_path).unlink(missing_ok=True)


def test_empty_mesh_connectivity():
    """Test edge building with empty connectivity."""
    mesh_connectivity = []
    edges = build_edges_from_mesh_connectivity(mesh_connectivity)

    assert len(edges) == 0, f"Expected 0 edges for empty connectivity, got {len(edges)}"
