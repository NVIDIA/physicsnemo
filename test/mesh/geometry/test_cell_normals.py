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

"""Tests for compute_cell_normals across all dimension-specific branches.

Each branch in ``physicsnemo.mesh.geometry._cell_normals`` is exercised
with known analytic answers:

- ``_normals_2d``      : d=2  (90-degree CCW rotation)
- ``_normals_3d``      : d=3  (cross product)
- ``_normals_general`` : d>=4 (signed minor determinants)
"""

import math

import pytest
import torch

from physicsnemo.mesh.geometry._cell_normals import compute_cell_normals

### Helpers ###


def _relative_vectors(vertices: list[list[float]]) -> torch.Tensor:
    """Build relative_vectors from a simplex vertex list.

    Args:
        vertices: List of vertex coordinates, where vertices[0] is the
            origin vertex and vertices[1:] are the remaining vertices.

    Returns:
        Tensor of shape ``(1, n_manifold_dims, n_spatial_dims)``.
    """
    pts = torch.tensor(vertices, dtype=torch.float64)
    return (pts[1:] - pts[0]).unsqueeze(0)


### Branch 1: _normals_2d (edges in 2D) ###


class TestNormals2D:
    """Tests for the d=2 branch (90-degree CCW rotation)."""

    def test_edge_along_x(self):
        """Edge along +x: normal is +y."""
        vecs = _relative_vectors([[0.0, 0.0], [1.0, 0.0]])
        result = compute_cell_normals(vecs)
        torch.testing.assert_close(
            result,
            torch.tensor([[0.0, 1.0]], dtype=torch.float64),
        )

    def test_edge_along_y(self):
        """Edge along +y: normal is -x."""
        vecs = _relative_vectors([[0.0, 0.0], [0.0, 1.0]])
        result = compute_cell_normals(vecs)
        torch.testing.assert_close(
            result,
            torch.tensor([[-1.0, 0.0]], dtype=torch.float64),
        )

    def test_diagonal_edge(self):
        """Edge along (1, 1): normal is (-1, 1) / sqrt(2)."""
        vecs = _relative_vectors([[0.0, 0.0], [1.0, 1.0]])
        result = compute_cell_normals(vecs)
        s = 1.0 / math.sqrt(2.0)
        torch.testing.assert_close(
            result,
            torch.tensor([[-s, s]], dtype=torch.float64),
            atol=1e-12,
            rtol=1e-12,
        )

    def test_multiple_edges(self):
        """Batch of two edges."""
        vecs = torch.tensor(
            [[[1.0, 0.0]], [[0.0, 1.0]]],
            dtype=torch.float64,
        )
        result = compute_cell_normals(vecs)
        expected = torch.tensor(
            [[0.0, 1.0], [-1.0, 0.0]],
            dtype=torch.float64,
        )
        torch.testing.assert_close(result, expected)


### Branch 2: _normals_3d (triangles in 3D, cross product) ###


class TestNormals3D:
    """Tests for the d=3 branch (cross product)."""

    def test_xy_plane_triangle(self):
        """Triangle in XY-plane: normal is +z."""
        vecs = _relative_vectors([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        result = compute_cell_normals(vecs)
        torch.testing.assert_close(
            result,
            torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64),
        )

    def test_xz_plane_triangle(self):
        """Triangle in XZ-plane: normal is -y (right-hand rule)."""
        vecs = _relative_vectors([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        result = compute_cell_normals(vecs)
        torch.testing.assert_close(
            result,
            torch.tensor([[0.0, -1.0, 0.0]], dtype=torch.float64),
        )

    def test_yz_plane_triangle(self):
        """Triangle in YZ-plane: normal is +x."""
        vecs = _relative_vectors([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        result = compute_cell_normals(vecs)
        torch.testing.assert_close(
            result,
            torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64),
        )

    def test_tilted_triangle(self):
        """Triangle with edges (1,0,0) and (0,1,1): normal = (0,-1,1)/sqrt(2)."""
        vecs = _relative_vectors([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 1.0]])
        result = compute_cell_normals(vecs)
        s = 1.0 / math.sqrt(2.0)
        torch.testing.assert_close(
            result,
            torch.tensor([[0.0, -s, s]], dtype=torch.float64),
            atol=1e-12,
            rtol=1e-12,
        )


### Branch 3: _normals_general (d >= 4) ###


class TestNormalsGeneral:
    """Tests for the d>=4 branch (signed minor determinants)."""

    def test_tetrahedron_facet_4d(self):
        """Tetrahedron facet in 4D with edges along first 3 axes.

        The 3 edge vectors span the (x, y, z) subspace, so the normal
        should point along the 4th axis: (0, 0, 0, +/-1).
        """
        vecs = _relative_vectors(
            [
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ]
        )
        result = compute_cell_normals(vecs)
        # Normal should be along w-axis; accept either sign
        assert torch.allclose(
            result.abs(),
            torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float64),
            atol=1e-12,
        )

    def test_4d_tilted_facet(self):
        """Facet in 4D with edges along x, y, and w axes.

        Spans (x, y, w) subspace, so normal is along z: (0, 0, +/-1, 0).
        """
        vecs = _relative_vectors(
            [
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        result = compute_cell_normals(vecs)
        assert torch.allclose(
            result.abs(),
            torch.tensor([[0.0, 0.0, 1.0, 0.0]], dtype=torch.float64),
            atol=1e-12,
        )


### Unit length ###


class TestUnitLength:
    """Verify all normals are unit length."""

    def test_random_triangles_3d(self):
        """Random non-degenerate triangles in 3D should have unit normals."""
        torch.manual_seed(42)
        vecs = torch.randn(100, 2, 3, dtype=torch.float64)
        result = compute_cell_normals(vecs)
        lengths = result.norm(dim=-1)
        torch.testing.assert_close(
            lengths,
            torch.ones(100, dtype=torch.float64),
            atol=1e-12,
            rtol=1e-12,
        )

    def test_random_edges_2d(self):
        """Random non-degenerate edges in 2D should have unit normals."""
        torch.manual_seed(42)
        vecs = torch.randn(100, 1, 2, dtype=torch.float64)
        result = compute_cell_normals(vecs)
        lengths = result.norm(dim=-1)
        torch.testing.assert_close(
            lengths,
            torch.ones(100, dtype=torch.float64),
            atol=1e-12,
            rtol=1e-12,
        )


### Degenerate cases ###


class TestDegenerateCases:
    """Tests for degenerate (zero-area) simplices."""

    def test_zero_length_edge_2d(self):
        """Zero-length edge should produce a zero normal (from F.normalize)."""
        vecs = torch.tensor([[[0.0, 0.0]]], dtype=torch.float64)
        result = compute_cell_normals(vecs)
        torch.testing.assert_close(
            result,
            torch.tensor([[0.0, 0.0]], dtype=torch.float64),
        )

    def test_collinear_triangle_3d(self):
        """Collinear edges produce a zero-area triangle with zero normal."""
        vecs = _relative_vectors([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        result = compute_cell_normals(vecs)
        torch.testing.assert_close(
            result,
            torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64),
        )


### Cross-branch consistency ###


class TestCrossBranchConsistency:
    """Verify that the specialized branches match the general formula."""

    def test_3d_cross_vs_general(self):
        """Cross-product (d=3) and signed-minor-det (general) should agree.

        Calls the internal functions directly on the same 3D input so we
        can compare the two code paths for a codimension-1 configuration.
        """
        from physicsnemo.mesh.geometry._cell_normals import (
            _normals_3d,
            _normals_general,
        )

        vecs = _relative_vectors([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        n_cross = _normals_3d(vecs)
        n_general = _normals_general(vecs)
        torch.testing.assert_close(n_cross, n_general, atol=1e-12, rtol=0)

    def test_2d_rotation_vs_general(self):
        """Rotation (d=2) and signed-minor-det (general) should agree."""
        from physicsnemo.mesh.geometry._cell_normals import (
            _normals_2d,
            _normals_general,
        )

        vecs = _relative_vectors([[0.0, 0.0], [3.0, 4.0]])
        n_rot = _normals_2d(vecs)
        n_general = _normals_general(vecs)
        torch.testing.assert_close(n_rot, n_general, atol=1e-12, rtol=0)


### Device parametrization ###


class TestDeviceParametrized:
    """Run representative cases on all available devices."""

    @pytest.mark.parametrize(
        "n_spatial_dims",
        [2, 3, 4],
        ids=["d2", "d3", "d4"],
    )
    def test_unit_normals(self, n_spatial_dims, device):
        """Non-degenerate codimension-1 simplices should produce unit normals."""
        n_manifold_dims = n_spatial_dims - 1
        # Build orthogonal unit edge vectors
        vecs = torch.zeros(
            1,
            n_manifold_dims,
            n_spatial_dims,
            device=device,
            dtype=torch.float64,
        )
        for i in range(n_manifold_dims):
            vecs[0, i, i] = 1.0
        result = compute_cell_normals(vecs)
        assert result.device.type == device
        lengths = result.norm(dim=-1)
        torch.testing.assert_close(
            lengths,
            torch.ones(1, device=device, dtype=torch.float64),
            atol=1e-12,
            rtol=1e-12,
        )
