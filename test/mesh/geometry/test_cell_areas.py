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

"""Tests for compute_cell_areas across all dimension-specific branches.

Each branch in ``physicsnemo.mesh.geometry._cell_areas`` is exercised with
known analytic answers:

- ``_edge_lengths``              : n_manifold_dims = 1
- ``_triangle_areas``            : n_manifold_dims = 2  (Lagrange identity)
- ``_tetrahedron_volumes_3d``    : n_manifold_dims = 3, n_spatial_dims = 3
- ``_tetrahedron_volumes_general``: n_manifold_dims = 3, n_spatial_dims > 3
- ``_gram_det_volumes``          : n_manifold_dims >= 4
"""

import math

import pytest
import torch

from physicsnemo.mesh.geometry._cell_areas import compute_cell_areas

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


### Branch 1: _edge_lengths (n_manifold_dims = 1) ###


class TestEdgeLengths:
    """Tests for the n=1 branch (vector norm)."""

    def test_edge_2d(self):
        """Unit edge along x-axis in 2D."""
        vecs = _relative_vectors([[0.0, 0.0], [3.0, 4.0]])
        result = compute_cell_areas(vecs)
        torch.testing.assert_close(result, torch.tensor([5.0], dtype=torch.float64))

    def test_edge_3d(self):
        """Edge in 3D: length = sqrt(1 + 4 + 9) = sqrt(14)."""
        vecs = _relative_vectors([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
        result = compute_cell_areas(vecs)
        torch.testing.assert_close(
            result,
            torch.tensor([math.sqrt(14.0)], dtype=torch.float64),
            atol=1e-12,
            rtol=1e-12,
        )

    def test_multiple_edges(self):
        """Batch of two edges."""
        vecs = torch.tensor(
            [[[1.0, 0.0]], [[0.0, 1.0]]],  # two edges in 2D
            dtype=torch.float64,
        )
        result = compute_cell_areas(vecs)
        torch.testing.assert_close(
            result,
            torch.tensor([1.0, 1.0], dtype=torch.float64),
        )


### Branch 2: _triangle_areas (n_manifold_dims = 2, Lagrange identity) ###


class TestTriangleAreas:
    """Tests for the n=2 branch (Lagrange identity)."""

    def test_right_triangle_2d(self):
        """Right triangle with legs 1 in 2D: area = 0.5."""
        vecs = _relative_vectors([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        result = compute_cell_areas(vecs)
        torch.testing.assert_close(result, torch.tensor([0.5], dtype=torch.float64))

    def test_right_triangle_3d(self):
        """Right triangle with legs 1 in 3D (in the xy-plane): area = 0.5."""
        vecs = _relative_vectors([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        result = compute_cell_areas(vecs)
        torch.testing.assert_close(result, torch.tensor([0.5], dtype=torch.float64))

    def test_equilateral_triangle_3d(self):
        """Equilateral triangle with side 2: area = sqrt(3)."""
        vecs = _relative_vectors(
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, math.sqrt(3.0), 0.0]]
        )
        result = compute_cell_areas(vecs)
        torch.testing.assert_close(
            result,
            torch.tensor([math.sqrt(3.0)], dtype=torch.float64),
            atol=1e-12,
            rtol=1e-12,
        )

    def test_triangle_4d(self):
        """Right triangle with legs 1 embedded in 4D: area = 0.5.

        Exercises the Lagrange identity with n_spatial_dims > 3.
        """
        vecs = _relative_vectors(
            [[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
        )
        result = compute_cell_areas(vecs)
        torch.testing.assert_close(result, torch.tensor([0.5], dtype=torch.float64))


### Branch 3: _tetrahedron_volumes_3d (n=3, d=3, scalar triple product) ###


class TestTetrahedronVolumes3D:
    """Tests for the n=3, d=3 branch (scalar triple product)."""

    def test_unit_tetrahedron(self):
        """Unit tetrahedron with orthogonal edges: volume = 1/6."""
        vecs = _relative_vectors(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        result = compute_cell_areas(vecs)
        torch.testing.assert_close(
            result,
            torch.tensor([1.0 / 6.0], dtype=torch.float64),
            atol=1e-12,
            rtol=1e-12,
        )

    def test_scaled_tetrahedron(self):
        """Tetrahedron with edge lengths 2: volume = 8/6 = 4/3."""
        vecs = _relative_vectors(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 2.0],
            ]
        )
        result = compute_cell_areas(vecs)
        torch.testing.assert_close(
            result,
            torch.tensor([4.0 / 3.0], dtype=torch.float64),
            atol=1e-12,
            rtol=1e-12,
        )


### Branch 4: _tetrahedron_volumes_general (n=3, d>3, Sarrus' rule) ###


class TestTetrahedronVolumesGeneral:
    """Tests for the n=3, d>3 branch (Sarrus' rule on 3x3 Gram matrix)."""

    def test_unit_tetrahedron_4d(self):
        """Unit tetrahedron with orthogonal edges embedded in 4D: volume = 1/6.

        Same geometry as the 3D case, with an extra zero coordinate.
        """
        vecs = _relative_vectors(
            [
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ]
        )
        result = compute_cell_areas(vecs)
        torch.testing.assert_close(
            result,
            torch.tensor([1.0 / 6.0], dtype=torch.float64),
            atol=1e-12,
            rtol=1e-12,
        )

    def test_tetrahedron_using_4th_dim(self):
        """Tetrahedron that actually extends into the 4th dimension."""
        vecs = _relative_vectors(
            [
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        result = compute_cell_areas(vecs)
        # Orthogonal edges of length 1 → same volume as 3D unit tet
        torch.testing.assert_close(
            result,
            torch.tensor([1.0 / 6.0], dtype=torch.float64),
            atol=1e-12,
            rtol=1e-12,
        )


### Branch 5: _gram_det_volumes (n_manifold_dims >= 4) ###


class TestGramDetVolumes:
    """Tests for the general fallback (n >= 4, Gram determinant)."""

    def test_4_simplex_in_4d(self):
        """Unit 4-simplex (orthogonal edges) in 4D: volume = 1/24."""
        vecs = _relative_vectors(
            [
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        result = compute_cell_areas(vecs)
        torch.testing.assert_close(
            result,
            torch.tensor([1.0 / 24.0], dtype=torch.float64),
            atol=1e-12,
            rtol=1e-12,
        )

    def test_5_simplex_in_5d(self):
        """Unit 5-simplex (orthogonal edges) in 5D: volume = 1/120."""
        n = 5
        origin = [0.0] * n
        vertices = [origin] + [
            [1.0 if j == i else 0.0 for j in range(n)] for i in range(n)
        ]
        vecs = _relative_vectors(vertices)
        result = compute_cell_areas(vecs)
        torch.testing.assert_close(
            result,
            torch.tensor([1.0 / math.factorial(n)], dtype=torch.float64),
            atol=1e-12,
            rtol=1e-12,
        )


### Degenerate cases ###


class TestDegenerateCases:
    """Tests for degenerate (zero-volume) simplices."""

    def test_collinear_triangle(self):
        """Collinear points form a zero-area triangle."""
        vecs = _relative_vectors([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        result = compute_cell_areas(vecs)
        torch.testing.assert_close(result, torch.tensor([0.0], dtype=torch.float64))

    def test_coplanar_tetrahedron(self):
        """Coplanar vertices form a zero-volume tetrahedron."""
        vecs = _relative_vectors(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.5, 0.5, 0.0],
            ]
        )
        result = compute_cell_areas(vecs)
        torch.testing.assert_close(
            result,
            torch.tensor([0.0], dtype=torch.float64),
            atol=1e-12,
            rtol=0,
        )


### Cross-branch consistency ###


class TestCrossBranchConsistency:
    """Verify that embedding the same geometry in higher dimensions gives
    the same volume, exercising different code paths."""

    def test_triangle_2d_vs_3d_vs_4d(self):
        """Same right triangle embedded in 2D, 3D, and 4D."""
        results = []
        for d in (2, 3, 4):
            v0 = [0.0] * d
            v1 = [1.0] + [0.0] * (d - 1)
            v2 = [0.0, 1.0] + [0.0] * (d - 2)
            results.append(compute_cell_areas(_relative_vectors([v0, v1, v2])))
        torch.testing.assert_close(results[0], results[1])
        torch.testing.assert_close(results[1], results[2])

    def test_tetrahedron_3d_vs_4d(self):
        """Same unit tetrahedron embedded in 3D (scalar triple product) and
        4D (Sarrus' rule) should give the same volume."""
        verts_3d = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
        verts_4d = [v + [0.0] for v in verts_3d]
        result_3d = compute_cell_areas(_relative_vectors(verts_3d))
        result_4d = compute_cell_areas(_relative_vectors(verts_4d))
        torch.testing.assert_close(result_3d, result_4d)


### Device parametrization ###


class TestDeviceParametrized:
    """Run representative cases on all available devices."""

    @pytest.mark.parametrize(
        ("n_manifold_dims", "n_spatial_dims"),
        [(1, 2), (1, 3), (2, 2), (2, 3), (3, 3), (3, 4)],
        ids=["edge-2d", "edge-3d", "tri-2d", "tri-3d", "tet-3d", "tet-4d"],
    )
    def test_positive_areas(self, n_manifold_dims, n_spatial_dims, device):
        """Non-degenerate simplices should have strictly positive volume."""
        # Build orthogonal unit edge vectors
        vecs = torch.zeros(1, n_manifold_dims, n_spatial_dims, device=device)
        for i in range(n_manifold_dims):
            vecs[0, i, i] = 1.0
        result = compute_cell_areas(vecs)
        assert result.device.type == device
        assert (result > 0).all()
