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

"""Tests to ensure facet extraction properly isolates cached properties.

This test module specifically addresses the bug where cached geometric properties
(like point normals) from parent meshes were incorrectly shared with facet meshes,
leading to invalid cached data for different mesh topologies.
"""

import pytest
import torch

from physicsnemo.mesh import Mesh


class TestCacheIsolation:
    """Test that facet meshes don't inherit cached properties from parent meshes."""

    def test_point_normals_not_inherited_by_facet_mesh(self):
        """Test that point normals from parent mesh don't contaminate facet mesh.

        This is a critical bug fix: point normals are only valid for the specific
        cell connectivity they were computed from. When extracting edges from triangles,
        the cached normals should not be inherited.
        """
        # Create triangle mesh in 3D (codimension-1, normals are valid)
        points = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 1.0, 0.0],
                [1.5, 1.0, 0.0],
            ]
        )
        cells = torch.tensor([[0, 1, 2], [1, 3, 2]])

        mesh = Mesh(points=points, cells=cells)

        # Compute point normals for triangle mesh
        triangle_normals = mesh.point_normals
        assert mesh._cache.get(("point", "normals"), None) is not None
        assert mesh.codimension == 1  # Valid for normals

        # Verify normals were correctly computed (should point in +z direction for this mesh)
        assert triangle_normals.shape == (4, 3), (
            "Point normals should have shape (n_points, 3)"
        )
        assert torch.all(torch.isfinite(triangle_normals)), "Normals should be finite"
        # All normals should be unit vectors
        norms = torch.norm(triangle_normals, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), (
            "Normals should be unit vectors"
        )

        # Extract edge mesh (codimension-2, normals are NOT valid)
        edge_mesh = mesh.get_facet_mesh(manifold_codimension=1)

        # Edge mesh should NOT have cached normals from parent
        assert edge_mesh._cache.get(("point", "normals"), None) is None, (
            "Cached point normals from parent mesh should not be in facet mesh"
        )

        # Attempting to access point_normals on edge mesh should raise ValueError
        with pytest.raises(ValueError, match="only defined for codimension-1"):
            _ = edge_mesh.point_normals

    def test_user_point_data_is_preserved(self):
        """Test that user-defined (non-cached) point data IS preserved in facet mesh."""
        points = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 1.0, 0.0],
            ]
        )
        cells = torch.tensor([[0, 1, 2]])

        # Add user-defined point data (not starting with "_")
        point_data = {
            "temperature": torch.tensor([100.0, 200.0, 150.0]),
            "velocity": torch.tensor(
                [[1.0, 0.0, 0.0], [1.0, 0.5, 0.0], [1.0, 0.25, 0.0]]
            ),
        }

        mesh = Mesh(points=points, cells=cells, point_data=point_data)

        # Compute some cached properties
        _ = mesh.point_normals  # Creates cache

        # Extract edge mesh
        edge_mesh = mesh.get_facet_mesh(manifold_codimension=1)

        # User data should be preserved
        assert "temperature" in edge_mesh.point_data
        assert "velocity" in edge_mesh.point_data
        assert torch.equal(
            edge_mesh.point_data["temperature"], point_data["temperature"]
        )
        assert torch.equal(edge_mesh.point_data["velocity"], point_data["velocity"])

        # Cached properties should NOT be preserved
        assert edge_mesh._cache.get(("point", "normals"), None) is None

    def test_multiple_cache_types_filtered(self):
        """Test that all cached properties are filtered from facet meshes."""
        points = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 1.0, 0.0],
            ]
        )
        cells = torch.tensor([[0, 1, 2]])

        mesh = Mesh(points=points, cells=cells)

        # Manually add various cached properties
        mesh._cache["point", "normals"] = torch.ones(3, 3)
        mesh._cache["point", "custom_cache"] = torch.zeros(3)
        mesh._cache["point", "another_property"] = torch.tensor([1.0, 2.0, 3.0])

        # Add non-cached property
        mesh.point_data["user_field"] = torch.tensor([10.0, 20.0, 30.0])

        # Extract facet mesh
        edge_mesh = mesh.get_facet_mesh(manifold_codimension=1)

        # All cached properties should be filtered
        assert edge_mesh._cache.get(("point", "normals"), None) is None
        assert edge_mesh._cache.get(("point", "custom_cache"), None) is None
        assert edge_mesh._cache.get(("point", "another_property"), None) is None

        # User field should be preserved
        assert "user_field" in edge_mesh.point_data
        assert torch.equal(
            edge_mesh.point_data["user_field"], mesh.point_data["user_field"]
        )

    @pytest.mark.parametrize("manifold_codimension", [1, 2])
    def test_cache_isolation_various_codimensions(self, manifold_codimension):
        """Test cache isolation works for different codimension extractions."""
        # Triangle mesh
        points = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 1.0, 0.0],
            ]
        )
        cells = torch.tensor([[0, 1, 2]])

        mesh = Mesh(points=points, cells=cells)

        # Add cached property
        _ = mesh.point_normals  # Creates cache
        assert mesh._cache.get(("point", "normals"), None) is not None

        # Extract facet mesh
        facet_mesh = mesh.get_facet_mesh(manifold_codimension=manifold_codimension)

        # Cached properties should always be filtered
        assert facet_mesh._cache.get(("point", "normals"), None) is None

    def test_empty_point_data(self):
        """Test that facet extraction works with empty point_data."""
        points = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 1.0, 0.0],
            ]
        )
        cells = torch.tensor([[0, 1, 2]])

        mesh = Mesh(points=points, cells=cells)

        # Don't add any data or compute any properties
        assert len(mesh.point_data.keys()) == 0

        # Extract facet mesh
        edge_mesh = mesh.get_facet_mesh(manifold_codimension=1)

        # Should work fine with empty point_data
        assert len(edge_mesh.point_data.keys()) == 0

    def test_cell_data_not_affected(self):
        """Test that cell_data aggregation still works correctly.

        Cell data has always been properly aggregated (not shared), so this
        test ensures our fix doesn't accidentally break that.
        """
        points = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 1.0, 0.0],
            ]
        )
        cells = torch.tensor([[0, 1, 2]])

        cell_data = {"pressure": torch.tensor([100.0])}
        mesh = Mesh(points=points, cells=cells, cell_data=cell_data)

        # Extract edge mesh
        edge_mesh = mesh.get_facet_mesh(
            manifold_codimension=1, data_source="cells", data_aggregation="mean"
        )

        # Cell data should be properly aggregated (not shared)
        assert "pressure" in edge_mesh.cell_data
        # Each of the 3 edges should have the same pressure value
        assert edge_mesh.cell_data["pressure"].shape == (3,)
        assert torch.allclose(
            edge_mesh.cell_data["pressure"], torch.tensor([100.0, 100.0, 100.0])
        )


class TestCacheConsistency:
    """Test that cached properties remain consistent across operations."""

    def test_parent_cache_unchanged_after_facet_extraction(self):
        """Test that extracting facets doesn't modify parent mesh caches."""
        points = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 1.0, 0.0],
            ]
        )
        cells = torch.tensor([[0, 1, 2]])

        mesh = Mesh(points=points, cells=cells)

        # Compute and cache point normals
        original_normals = mesh.point_normals.clone()
        assert mesh._cache.get(("point", "normals"), None) is not None

        # Extract facet mesh
        _ = mesh.get_facet_mesh(manifold_codimension=1)

        # Parent mesh caches should be unchanged
        assert mesh._cache.get(("point", "normals"), None) is not None
        assert torch.equal(mesh._cache["point", "normals"], original_normals)

    def test_independent_caches_after_extraction(self):
        """Test that parent and facet meshes maintain independent caches."""
        points = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 1.0, 0.0],
                [1.5, 1.0, 0.0],
            ]
        )
        cells = torch.tensor([[0, 1, 2], [1, 3, 2]])

        parent_mesh = Mesh(points=points, cells=cells)

        # Compute normals on parent (codimension-1, valid)
        parent_normals = parent_mesh.point_normals

        # Extract edge mesh (codimension-2)
        edge_mesh = parent_mesh.get_facet_mesh(manifold_codimension=1)

        # Add some user data to edge mesh point_data
        edge_mesh.point_data["custom_field"] = torch.ones(4)

        # Parent mesh should not have the custom field
        assert "custom_field" not in parent_mesh.point_data

        # Parent mesh should still have its cached normals
        assert parent_mesh._cache.get(("point", "normals"), None) is not None
        assert torch.equal(parent_mesh._cache["point", "normals"], parent_normals)
