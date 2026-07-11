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

"""Tests for ``Mesh.cell_quadrature_weights`` and weighted integration.

Covers the property contract (ones fallback, set-method validation, storage in
``cell_data`` under the reserved key), that ``integrate`` /
``integrate_flux`` consume the effective measure
``cell_areas * cell_quadrature_weights``, that a Horvitz-Thompson-weighted
cell subsample yields unbiased integrals, and that weights survive
slicing and rigid/scaling transforms with the correct semantics.
"""

import math

import pytest
import torch

from physicsnemo.mesh import QUADRATURE_WEIGHTS_KEY, Mesh
from physicsnemo.mesh.calculus import integrate_moment
from physicsnemo.mesh.primitives.basic import two_triangles_2d


def make_triangle_strip(n_cells: int, widths: torch.Tensor | None = None) -> Mesh:
    """Planar 3D mesh of *n_cells* disjoint right triangles with known areas.

    Triangle ``i`` has vertices ``(x_i, 0, 0)``, ``(x_i + w_i, 0, 0)``,
    ``(x_i, 1, 0)`` and area ``w_i / 2``.  Vertices are not shared, so cell
    slicing keeps the area bookkeeping trivial.
    """
    if widths is None:
        widths = torch.ones(n_cells)
    x0 = torch.cat([torch.zeros(1), torch.cumsum(widths, dim=0)[:-1]])
    pts = []
    for i in range(n_cells):
        pts += [
            [float(x0[i]), 0.0, 0.0],
            [float(x0[i] + widths[i]), 0.0, 0.0],
            [float(x0[i]), 1.0, 0.0],
        ]
    return Mesh(
        points=torch.tensor(pts),
        cells=torch.arange(3 * n_cells).reshape(n_cells, 3),
    )


class TestCellQuadratureWeightsProperty:
    def test_defaults_to_ones(self):
        mesh = two_triangles_2d.load()
        w = mesh.cell_quadrature_weights
        assert w.shape == (mesh.n_cells,)
        torch.testing.assert_close(w, torch.ones(mesh.n_cells))
        ### The fallback must not materialize the reserved key.
        assert QUADRATURE_WEIGHTS_KEY not in mesh.cell_data.keys()

    def test_set_method_roundtrip_via_reserved_key(self):
        mesh = two_triangles_2d.load()
        mesh.set_cell_quadrature_weights(torch.tensor([2.0, 3.0]))
        assert QUADRATURE_WEIGHTS_KEY in mesh.cell_data.keys()
        torch.testing.assert_close(
            mesh.cell_quadrature_weights, torch.tensor([2.0, 3.0])
        )

    def test_set_method_rejects_wrong_shape(self):
        mesh = two_triangles_2d.load()
        with pytest.raises(ValueError, match="cell_quadrature_weights"):
            mesh.set_cell_quadrature_weights(torch.ones(mesh.n_cells + 1))

    def test_weights_survive_slice_cells(self):
        mesh = make_triangle_strip(6)
        mesh.set_cell_quadrature_weights(torch.arange(1.0, 7.0))
        sliced = mesh.slice_cells(torch.tensor([1, 4]))
        torch.testing.assert_close(
            sliced.cell_quadrature_weights, torch.tensor([2.0, 5.0])
        )


class TestWeightedIntegration:
    def test_integrate_cell_data_uses_effective_measure(self):
        mesh = make_triangle_strip(4)
        mesh.cell_data["f"] = torch.tensor([1.0, 2.0, 3.0, 4.0])
        unweighted = mesh.integrate("f")
        mesh.set_cell_quadrature_weights(torch.full((4,), 2.5))
        torch.testing.assert_close(mesh.integrate("f"), unweighted * 2.5)

    def test_integrate_point_data_uses_effective_measure(self):
        mesh = make_triangle_strip(3)
        mesh.point_data["T"] = torch.randn(mesh.n_points)
        unweighted = mesh.integrate("T", data_source="points")
        mesh.set_cell_quadrature_weights(torch.full((3,), 4.0))
        torch.testing.assert_close(
            mesh.integrate("T", data_source="points"), unweighted * 4.0
        )

    def test_integrate_flux_uses_effective_measure(self):
        mesh = make_triangle_strip(3)  # planar, normals +/- z
        mesh.cell_data["v"] = torch.randn(3, 3)
        unweighted = mesh.integrate_flux("v")
        mesh.set_cell_quadrature_weights(torch.full((3,), 3.0))
        torch.testing.assert_close(mesh.integrate_flux("v"), unweighted * 3.0)

    def test_integrate_moment_uses_effective_measure(self):
        mesh = make_triangle_strip(4)
        left = torch.randn(4, 2)
        right = torch.randn(4, 3)
        unweighted = integrate_moment(mesh, left, right)
        mesh.set_cell_quadrature_weights(torch.full((4,), 2.0))
        torch.testing.assert_close(
            integrate_moment(mesh, left, right), unweighted * 2.0
        )

    def test_no_weights_matches_geometric_measure(self):
        """Meshes without weights integrate against the bare geometric areas."""
        mesh = make_triangle_strip(5, widths=torch.rand(5) + 0.5)
        mesh.cell_data["f"] = torch.randn(5)
        expected = (mesh.cell_data["f"] * mesh.cell_areas).sum()
        torch.testing.assert_close(mesh.integrate("f"), expected)

    def test_ht_subsample_integral_unbiased_over_all_starts(self):
        """Mean over all cyclic-block subsamples equals the full integral."""
        n, k = 11, 4
        mesh = make_triangle_strip(n, widths=torch.rand(n) + 0.5)
        mesh.cell_data["f"] = torch.randn(n)
        full = mesh.integrate("f").to(torch.float64)

        estimates = []
        for start in range(n):
            idx = torch.arange(start, start + k) % n
            sub = mesh.slice_cells(idx)
            sub.set_cell_quadrature_weights(torch.full((k,), n / k))
            estimates.append(sub.integrate("f").to(torch.float64))
        torch.testing.assert_close(
            torch.stack(estimates).mean(), full, rtol=1e-5, atol=1e-6
        )


class TestWeightsUnderTransforms:
    def test_weights_invariant_under_rigid_and_scaling_transforms(self):
        """Dimensionless weights ride along; the effective measure tracks
        geometry through rotation/translation (invariant) and scaling
        (quadratic in the factor)."""
        n = 5
        mesh = make_triangle_strip(n, widths=torch.rand(n) + 0.5)
        mesh.cell_data["f"] = torch.ones(n)
        mesh.set_cell_quadrature_weights(torch.full((n,), 2.0))
        base_integral = mesh.integrate("f")

        moved = (
            mesh.rotate(math.pi / 3, axis="z", transform_cell_data=True)
            .translate(torch.tensor([1.0, -2.0, 0.5]))
            .scale(1.0 / 5.0, transform_cell_data=True)
        )

        torch.testing.assert_close(moved.cell_quadrature_weights, torch.full((n,), 2.0))
        torch.testing.assert_close(
            moved.integrate("f"), base_integral / 25.0, rtol=1e-5, atol=1e-7
        )
