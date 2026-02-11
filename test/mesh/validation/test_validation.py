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

"""Comprehensive tests for validation module.

Tests mesh validation, quality metrics computation, and mesh statistics
including edge cases and code path coverage.

This module consolidates tests from:
- Core validation tests (mesh validation, quality metrics, statistics)
- Edge case tests (code path coverage, special conditions)
"""

import pytest
import torch

from physicsnemo.mesh import Mesh
from physicsnemo.mesh.validation import (
    compute_mesh_statistics,
    compute_quality_metrics,
    validate_mesh,
)

###############################################################################
# Mesh Validation Tests
###############################################################################


class TestMeshValidation:
    """Tests for mesh validation."""

    def test_valid_mesh(self, device):
        """Test that valid mesh passes all checks."""
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, 1.0],
            ],
            dtype=torch.float32,
            device=device,
        )

        cells = torch.tensor([[0, 1, 2]], dtype=torch.long, device=device)

        mesh = Mesh(points=points, cells=cells)

        report = validate_mesh(mesh)

        assert report["valid"]
        assert report["n_degenerate_cells"] == 0
        assert report["n_out_of_bounds_cells"] == 0

    def test_out_of_bounds_indices(self, device):
        """Test detection of out-of-bounds cell indices."""
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, 1.0],
            ],
            dtype=torch.float32,
            device=device,
        )

        # Cell references non-existent vertex 10
        cells = torch.tensor([[0, 1, 10]], dtype=torch.long, device=device)

        mesh = Mesh(points=points, cells=cells)

        report = validate_mesh(mesh, check_out_of_bounds=True, raise_on_error=False)

        assert not report["valid"]
        assert report["n_out_of_bounds_cells"] == 1

    def test_degenerate_cells_detection(self, device):
        """Test detection of degenerate cells."""
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, 1.0],
                [2.0, 0.0],
            ],
            dtype=torch.float32,
            device=device,
        )

        # Second cell has duplicate vertex (degenerate)
        cells = torch.tensor(
            [
                [0, 1, 2],
                [1, 3, 1],  # Duplicate vertex 1
            ],
            dtype=torch.long,
            device=device,
        )

        mesh = Mesh(points=points, cells=cells)

        report = validate_mesh(mesh, check_degenerate_cells=True, raise_on_error=False)

        assert not report["valid"]
        assert report["n_degenerate_cells"] >= 1

    def test_duplicate_vertices_detection(self, device):
        """Test detection of duplicate vertices."""
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, 1.0],
                [0.0, 0.0],  # Exact duplicate of vertex 0
            ],
            dtype=torch.float32,
            device=device,
        )

        cells = torch.tensor([[0, 1, 2]], dtype=torch.long, device=device)

        mesh = Mesh(points=points, cells=cells)

        report = validate_mesh(
            mesh, check_duplicate_vertices=True, raise_on_error=False
        )

        assert not report["valid"]
        assert report["n_duplicate_vertices"] >= 1

    def test_raise_on_error(self, device):
        """Test that raise_on_error triggers exception."""
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, 1.0],
            ],
            dtype=torch.float32,
            device=device,
        )

        cells = torch.tensor([[0, 1, 10]], dtype=torch.long, device=device)

        mesh = Mesh(points=points, cells=cells)

        with pytest.raises(ValueError, match="out-of-bounds"):
            validate_mesh(mesh, check_out_of_bounds=True, raise_on_error=True)

    def test_manifoldness_check_2d(self, device):
        """Test manifoldness check for 2D meshes."""
        points = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 1.0, 0.0],
                [0.5, 0.5, 1.0],
            ],
            dtype=torch.float32,
            device=device,
        )

        # Two triangles sharing edge [0,1]
        cells = torch.tensor(
            [
                [0, 1, 2],
                [0, 1, 3],
            ],
            dtype=torch.long,
            device=device,
        )

        mesh = Mesh(points=points, cells=cells)

        report = validate_mesh(mesh, check_manifoldness=True)

        # Should be manifold (each edge shared by at most 2 faces)
        assert report["is_manifold"]
        assert report["n_non_manifold_edges"] == 0

    def test_empty_mesh_validation(self, device):
        """Test validation of empty mesh."""
        points = torch.zeros((0, 2), device=device)
        cells = torch.zeros((0, 3), dtype=torch.long, device=device)

        mesh = Mesh(points=points, cells=cells)

        report = validate_mesh(mesh)

        # Empty mesh should be valid
        assert report["valid"]


###############################################################################
# Quality Metrics Tests
###############################################################################


class TestQualityMetrics:
    """Tests for quality metrics computation."""

    def test_equilateral_triangle_quality(self, device):
        """Test that equilateral triangle has high quality score."""

        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, (3**0.5) / 2],
            ],
            dtype=torch.float32,
            device=device,
        )

        cells = torch.tensor([[0, 1, 2]], dtype=torch.long, device=device)

        mesh = Mesh(points=points, cells=cells)

        metrics = compute_quality_metrics(mesh)

        assert "quality_score" in metrics.keys()
        assert "aspect_ratio" in metrics.keys()
        assert "edge_length_ratio" in metrics.keys()

        # Equilateral triangle should have high quality
        quality = metrics["quality_score"][0]
        assert quality > 0.7  # High quality (formula gives ~0.75 for equilateral)

        # Edge length ratio should be close to 1.0
        edge_ratio = metrics["edge_length_ratio"][0]
        assert edge_ratio < 1.1  # Nearly equal edges

    def test_degenerate_triangle_quality(self, device):
        """Test that degenerate triangle has low quality score."""
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [10.0, 0.0],  # Nearly collinear
            ],
            dtype=torch.float32,
            device=device,
        )

        cells = torch.tensor([[0, 1, 2]], dtype=torch.long, device=device)

        mesh = Mesh(points=points, cells=cells)

        metrics = compute_quality_metrics(mesh)

        quality = metrics["quality_score"][0]

        # Very elongated triangle should have low quality
        assert quality < 0.3

    def test_quality_metrics_angles(self, device):
        """Test that angles are computed for triangles."""

        # Right triangle
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=torch.float32,
            device=device,
        )

        cells = torch.tensor([[0, 1, 2]], dtype=torch.long, device=device)

        mesh = Mesh(points=points, cells=cells)

        metrics = compute_quality_metrics(mesh)

        assert "min_angle" in metrics.keys()
        assert "max_angle" in metrics.keys()

        min_angle = metrics["min_angle"][0]
        max_angle = metrics["max_angle"][0]

        # Right triangle has angles: π/4, π/4, π/2
        assert min_angle > 0
        assert max_angle <= torch.pi

        # Max angle should be close to π/2
        assert torch.abs(max_angle - torch.pi / 2) < 0.1

    def test_empty_mesh_quality(self, device):
        """Test quality metrics on empty mesh."""
        points = torch.zeros((5, 2), device=device)
        cells = torch.zeros((0, 3), dtype=torch.long, device=device)

        mesh = Mesh(points=points, cells=cells)

        metrics = compute_quality_metrics(mesh)

        # Should return empty TensorDict
        assert len(metrics) == 0 or metrics.shape[0] == 0


###############################################################################
# Mesh Statistics Tests
###############################################################################


class TestMeshStatistics:
    """Tests for mesh statistics computation."""

    def test_basic_statistics(self, device):
        """Test basic mesh statistics."""
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, 1.0],
                [1.5, 0.5],
            ],
            dtype=torch.float32,
            device=device,
        )

        cells = torch.tensor(
            [
                [0, 1, 2],
                [1, 2, 3],
            ],
            dtype=torch.long,
            device=device,
        )

        mesh = Mesh(points=points, cells=cells)

        stats = compute_mesh_statistics(mesh)

        assert stats["n_points"] == 4
        assert stats["n_cells"] == 2
        assert stats["n_manifold_dims"] == 2
        assert stats["n_spatial_dims"] == 2
        assert stats["n_degenerate_cells"] == 0
        assert stats["n_isolated_vertices"] == 0

    def test_statistics_with_isolated(self, device):
        """Test statistics with isolated vertices."""
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, 1.0],
                [5.0, 5.0],  # Isolated
                [6.0, 6.0],  # Isolated
            ],
            dtype=torch.float32,
            device=device,
        )

        cells = torch.tensor([[0, 1, 2]], dtype=torch.long, device=device)

        mesh = Mesh(points=points, cells=cells)

        stats = compute_mesh_statistics(mesh)

        assert stats["n_isolated_vertices"] == 2

    def test_statistics_edge_lengths(self, device):
        """Test edge length statistics."""
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, 1.0],
            ],
            dtype=torch.float32,
            device=device,
        )

        cells = torch.tensor([[0, 1, 2]], dtype=torch.long, device=device)

        mesh = Mesh(points=points, cells=cells)

        stats = compute_mesh_statistics(mesh)

        assert "edge_length_stats" in stats
        min_len, mean_len, max_len, std_len = stats["edge_length_stats"]

        # All should be positive
        assert min_len > 0
        assert mean_len > 0
        assert max_len > 0

    def test_statistics_empty_mesh(self, device):
        """Test statistics on empty mesh."""
        points = torch.zeros((5, 2), device=device)
        cells = torch.zeros((0, 3), dtype=torch.long, device=device)

        mesh = Mesh(points=points, cells=cells)

        stats = compute_mesh_statistics(mesh)

        assert stats["n_cells"] == 0
        assert stats["n_isolated_vertices"] == 5


###############################################################################
# Mesh API Integration Tests
###############################################################################


class TestMeshAPIIntegration:
    """Test that Mesh class methods work correctly."""

    def test_mesh_validate_method(self, device):
        """Test mesh.validate() convenience method."""
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, 1.0],
            ],
            dtype=torch.float32,
            device=device,
        )

        cells = torch.tensor([[0, 1, 2]], dtype=torch.long, device=device)

        mesh = Mesh(points=points, cells=cells)

        report = mesh.validate()

        assert isinstance(report, dict)
        assert "valid" in report
        assert report["valid"]

    def test_mesh_quality_metrics_property(self, device):
        """Test mesh.quality_metrics property."""
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, 1.0],
            ],
            dtype=torch.float32,
            device=device,
        )

        cells = torch.tensor([[0, 1, 2]], dtype=torch.long, device=device)

        mesh = Mesh(points=points, cells=cells)

        metrics = mesh.quality_metrics

        assert "quality_score" in metrics.keys()
        assert metrics["quality_score"].shape == (1,)

    def test_mesh_statistics_property(self, device):
        """Test mesh.statistics property."""
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, 1.0],
            ],
            dtype=torch.float32,
            device=device,
        )

        cells = torch.tensor([[0, 1, 2]], dtype=torch.long, device=device)

        mesh = Mesh(points=points, cells=cells)

        stats = mesh.statistics

        assert isinstance(stats, dict)
        assert stats["n_points"] == 3
        assert stats["n_cells"] == 1

    def test_validation_with_all_checks(self, device):
        """Test validation with all checks enabled."""
        points = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 1.0, 0.0],
                [0.5, 0.5, 1.0],
            ],
            dtype=torch.float32,
            device=device,
        )

        cells = torch.tensor(
            [
                [0, 1, 2],
                [1, 2, 3],
            ],
            dtype=torch.long,
            device=device,
        )

        mesh = Mesh(points=points, cells=cells)

        report = mesh.validate(
            check_degenerate_cells=True,
            check_duplicate_vertices=True,
            check_out_of_bounds=True,
            check_manifoldness=True,
        )

        assert report["valid"]

    def test_validation_detects_negative_indices(self, device):
        """Test that negative cell indices are caught."""
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, 1.0],
            ],
            dtype=torch.float32,
            device=device,
        )

        cells = torch.tensor([[0, -1, 2]], dtype=torch.long, device=device)

        mesh = Mesh(points=points, cells=cells)

        report = validate_mesh(mesh, check_out_of_bounds=True, raise_on_error=False)

        assert not report["valid"]
        assert report["n_out_of_bounds_cells"] == 1


###############################################################################
# Quality Metrics Edge Cases
###############################################################################


class TestQualityMetricsEdgeCases:
    """Edge case tests for quality metrics."""

    def test_single_cell_quality(self, device):
        """Test quality metrics on single cell."""

        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, (3**0.5) / 2],
            ],
            dtype=torch.float32,
            device=device,
        )

        cells = torch.tensor([[0, 1, 2]], dtype=torch.long, device=device)

        mesh = Mesh(points=points, cells=cells)

        metrics = compute_quality_metrics(mesh)

        assert metrics.shape[0] == 1
        assert not torch.isnan(metrics["quality_score"][0])

    def test_multiple_cells_quality(self, device):
        """Test quality metrics on multiple cells."""
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, 1.0],
                [1.5, 0.5],
                [0.5, -0.5],
            ],
            dtype=torch.float32,
            device=device,
        )

        cells = torch.tensor(
            [
                [0, 1, 2],
                [1, 2, 3],
                [0, 1, 4],
            ],
            dtype=torch.long,
            device=device,
        )

        mesh = Mesh(points=points, cells=cells)

        metrics = compute_quality_metrics(mesh)

        assert metrics.shape[0] == 3
        assert torch.all(metrics["quality_score"] > 0)
        assert torch.all(metrics["quality_score"] <= 1.0)

    def test_3d_mesh_quality(self, device):
        """Test quality metrics on 3D tetrahedral mesh."""
        from physicsnemo.mesh.primitives.volumes import tetrahedron_volume

        # Use tetrahedron_volume primitive for a regular tetrahedron
        mesh = tetrahedron_volume.load(device=device)

        metrics = compute_quality_metrics(mesh)

        # Should compute metrics for tets including solid angles
        assert metrics.shape[0] == 1
        assert not torch.isnan(metrics["quality_score"][0])
        assert not torch.isnan(metrics["min_angle"][0])
        # Regular tet: all solid angles should be equal, so min == max
        assert torch.isclose(
            metrics["min_angle"][0], metrics["max_angle"][0], atol=1e-5
        )


###############################################################################
# Statistics Variations
###############################################################################


class TestStatisticsVariations:
    """Test statistics computation with various mesh configurations."""

    def test_statistics_include_quality(self, device):
        """Test that statistics include quality metrics."""
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, 1.0],
            ],
            dtype=torch.float32,
            device=device,
        )

        cells = torch.tensor([[0, 1, 2]], dtype=torch.long, device=device)

        mesh = Mesh(points=points, cells=cells)

        stats = compute_mesh_statistics(mesh)

        assert "cell_area_stats" in stats
        assert "quality_score_stats" in stats
        assert "aspect_ratio_stats" in stats

    def test_statistics_large_mesh(self, device):
        """Test statistics on a realistic mesh with many cells."""
        from physicsnemo.mesh.primitives.procedural import lumpy_sphere

        # Use lumpy_sphere (subdivisions=2 gives ~320 cells) for realistic mesh
        mesh = lumpy_sphere.load(subdivisions=2, device=device)

        stats = compute_mesh_statistics(mesh)

        # Lumpy sphere at subdivisions=2 has 320 triangles
        assert stats["n_cells"] >= 300
        assert stats["n_isolated_vertices"] == 0
        assert "cell_area_stats" in stats
        assert "quality_score_stats" in stats


###############################################################################
# Validation Code Path Tests
###############################################################################


class TestValidationCodePaths:
    """Tests for specific validation code paths."""

    def test_large_mesh_duplicate_check_works(self, device):
        """Test that duplicate check works efficiently for large meshes."""
        # Create mesh with >10K points
        n = 101
        x = torch.linspace(0, 1, n, device=device)
        y = torch.linspace(0, 1, n, device=device)
        xx, yy = torch.meshgrid(x, y, indexing="xy")

        points = torch.stack([xx.flatten(), yy.flatten()], dim=-1)

        # Create some triangles
        cells = torch.tensor([[0, 1, n]], dtype=torch.long, device=device)

        mesh = Mesh(points=points, cells=cells)

        # Duplicate check now works for all mesh sizes using vectorized spatial hashing
        report = validate_mesh(mesh, check_duplicate_vertices=True)

        # Should return actual count (0 since grid points are well-spaced)
        assert report["n_duplicate_vertices"] == 0

    def test_inverted_cells_3d(self, device):
        """Test detection of inverted cells in 3D."""
        # Regular tetrahedron
        points = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, (3**0.5) / 2, 0.0],
                [0.5, (3**0.5) / 6, ((2 / 3) ** 0.5)],
            ],
            dtype=torch.float32,
            device=device,
        )

        cells = torch.tensor(
            [
                [0, 1, 2, 3],  # Normal orientation
                [0, 2, 1, 3],  # Inverted (swapped 1 and 2)
            ],
            dtype=torch.long,
            device=device,
        )

        mesh = Mesh(points=points, cells=cells)

        report = validate_mesh(mesh, check_inverted_cells=True, raise_on_error=False)

        # Should detect one inverted cell
        assert report["n_inverted_cells"] >= 1
        assert not report["valid"]

    def test_non_manifold_edge_detection(self, device):
        """Test detection of non-manifold edges."""
        # Create T-junction (3 triangles meeting at one edge)
        points = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 1.0, 0.0],
                [0.5, -1.0, 0.0],
                [0.5, 0.0, 1.0],
            ],
            dtype=torch.float32,
            device=device,
        )

        # Three triangles sharing edge [0,1]
        cells = torch.tensor(
            [
                [0, 1, 2],
                [0, 1, 3],
                [0, 1, 4],
            ],
            dtype=torch.long,
            device=device,
        )

        mesh = Mesh(points=points, cells=cells)

        report = validate_mesh(mesh, check_manifoldness=True, raise_on_error=False)

        # Should detect non-manifold edge
        assert not report["is_manifold"]
        assert report["n_non_manifold_edges"] >= 1

    def test_validation_with_empty_cells(self, device):
        """Test validation on mesh with no cells."""
        points = torch.randn(5, 2, device=device)
        cells = torch.zeros((0, 3), dtype=torch.long, device=device)

        mesh = Mesh(points=points, cells=cells)

        report = validate_mesh(
            mesh,
            check_degenerate_cells=True,
            check_out_of_bounds=True,
            check_inverted_cells=True,
        )

        # Should be valid (no cells to have problems)
        assert report["valid"]
        assert report["n_degenerate_cells"] == 0
        assert report["n_out_of_bounds_cells"] == 0

    def test_inverted_check_not_applicable(self, device):
        """Test that inverted check returns -1 for non-volume meshes."""
        # 2D triangle in 3D (codimension 1)
        points = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 1.0, 0.0],
            ],
            dtype=torch.float32,
            device=device,
        )

        cells = torch.tensor([[0, 1, 2]], dtype=torch.long, device=device)

        mesh = Mesh(points=points, cells=cells)

        report = validate_mesh(mesh, check_inverted_cells=True)

        # Should return -1 (not applicable for codimension != 0)
        assert report["n_inverted_cells"] == -1 or report["n_inverted_cells"] == 0

    def test_manifoldness_not_applicable_non_2d(self, device):
        """Test that manifoldness check is only for 2D manifolds."""
        # 1D mesh (edges)
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [2.0, 0.0],
            ],
            dtype=torch.float32,
            device=device,
        )

        cells = torch.tensor(
            [
                [0, 1],
                [1, 2],
            ],
            dtype=torch.long,
            device=device,
        )

        mesh = Mesh(points=points, cells=cells)

        report = validate_mesh(mesh, check_manifoldness=True)

        # Should return None or -1 for non-2D manifolds
        assert (
            report.get("is_manifold") is None
            or report.get("n_non_manifold_edges") == -1
        )

    def test_validation_skips_geometry_after_out_of_bounds(self, device):
        """Test that validation short-circuits after finding out-of-bounds indices."""
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, 1.0],
            ],
            dtype=torch.float32,
            device=device,
        )

        # Invalid index
        cells = torch.tensor([[0, 1, 100]], dtype=torch.long, device=device)

        mesh = Mesh(points=points, cells=cells)

        # Should not crash even though area computation would fail
        report = validate_mesh(
            mesh,
            check_out_of_bounds=True,
            check_degenerate_cells=True,
            raise_on_error=False,
        )

        assert not report["valid"]
        assert report["n_out_of_bounds_cells"] == 1
        # Degenerate check should be skipped (no key or not computed)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
