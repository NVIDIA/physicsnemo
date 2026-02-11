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

"""Tests for text rendering primitives."""

import pytest

# Skip this module if matplotlib is not available (text primitives require it)
pytest.importorskip("matplotlib")

from physicsnemo.mesh.primitives.text import (  # noqa: E402
    text_1d_2d,
    text_2d_2d,
    text_2d_3d,
    text_3d_3d,
)


def test_text_1d_2d():
    """Test 1D curve in 2D space text rendering."""
    mesh = text_1d_2d()

    assert mesh.n_manifold_dims == 1, "Should be 1D manifold"
    assert mesh.n_spatial_dims == 2, "Should be in 2D space"
    assert mesh.n_points > 0, "Should have points"
    assert mesh.n_cells > 0, "Should have cells (edges)"
    assert mesh.cells.shape[1] == 2, "Edges should have 2 vertices"


def test_text_2d_2d():
    """Test 2D surface in 2D space text rendering."""
    mesh = text_2d_2d()

    assert mesh.n_manifold_dims == 2, "Should be 2D manifold"
    assert mesh.n_spatial_dims == 2, "Should be in 2D space"
    assert mesh.n_points > 0, "Should have points"
    assert mesh.n_cells > 0, "Should have cells (triangles)"
    assert mesh.cells.shape[1] == 3, "Triangles should have 3 vertices"


def test_text_3d_3d():
    """Test 3D volume in 3D space text rendering."""
    mesh = text_3d_3d()

    assert mesh.n_manifold_dims == 3, "Should be 3D manifold"
    assert mesh.n_spatial_dims == 3, "Should be in 3D space"
    assert mesh.n_points > 0, "Should have points"
    assert mesh.n_cells > 0, "Should have cells (tetrahedra)"
    assert mesh.cells.shape[1] == 4, "Tetrahedra should have 4 vertices"


def test_text_2d_3d():
    """Test 2D surface in 3D space text rendering."""
    mesh = text_2d_3d()

    assert mesh.n_manifold_dims == 2, "Should be 2D manifold"
    assert mesh.n_spatial_dims == 3, "Should be in 3D space"
    assert mesh.n_points > 0, "Should have points"
    assert mesh.n_cells > 0, "Should have cells (triangles)"
    assert mesh.cells.shape[1] == 3, "Triangles should have 3 vertices"


def test_text_custom_text():
    """Test text rendering with custom text."""
    mesh = text_2d_2d(text="Test", font_size=10.0)

    assert mesh.n_manifold_dims == 2
    assert mesh.n_spatial_dims == 2
    assert mesh.n_points > 0
    assert mesh.n_cells > 0


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]
)
def test_text_device(device):
    """Test text rendering works on different devices."""
    mesh = text_2d_2d(device=device)

    assert mesh.points.device.type == device
    assert mesh.cells.device.type == device


def test_text_extrusion_height():
    """Test custom extrusion height."""
    mesh1 = text_3d_3d(extrusion_height=1.0)
    mesh2 = text_3d_3d(extrusion_height=3.0)

    # Different extrusion heights should produce different z-ranges
    z_range1 = mesh1.points[:, 2].max() - mesh1.points[:, 2].min()
    z_range2 = mesh2.points[:, 2].max() - mesh2.points[:, 2].min()

    assert z_range2 > z_range1, "Larger extrusion should produce larger z-range"


def test_text_max_segment_length():
    """Test that max_segment_length controls edge refinement."""
    mesh_coarse = text_1d_2d(max_segment_length=1.0)
    mesh_fine = text_1d_2d(max_segment_length=0.1)

    # Finer segmentation should have more points
    assert mesh_fine.n_points > mesh_coarse.n_points, (
        "Smaller max_segment_length should produce more points"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
