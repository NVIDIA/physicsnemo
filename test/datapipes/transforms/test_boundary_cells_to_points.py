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

"""Tests for the BoundaryCellsToPoints transform."""

import pytest
import torch

from physicsnemo.datapipes.registry import COMPONENT_REGISTRY
from physicsnemo.datapipes.transforms.mesh import BoundaryCellsToPoints
from physicsnemo.mesh import DomainMesh, Mesh
from physicsnemo.mesh.calculus.measure import point_measures
from physicsnemo.mesh.primitives.basic import single_triangle_3d
from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral


def _domain() -> DomainMesh:
    """A sphere wall with a scalar and a vector cell field, and an inlet."""
    wall = sphere_icosahedral.load(subdivisions=2)
    n = wall.n_cells
    wall = wall.with_data(
        cell_data={
            "p": torch.arange(n, dtype=torch.float32),
            "tau": torch.randn(n, 3, generator=torch.Generator().manual_seed(0)),
            "other": torch.ones(n),
        }
    )
    return DomainMesh(
        interior=Mesh(points=torch.randn(7, 3)),
        boundaries={"wall": wall, "inlet": single_triangle_3d.load()},
        global_data={"U_inf": torch.tensor([1.0, 0.0, 0.0])},
    )


def _transform(**kwargs) -> BoundaryCellsToPoints:
    return BoundaryCellsToPoints(
        source="wall", name="wall_points", fields={"p": "p_w", "tau": "tau_w"}, **kwargs
    )


class TestBoundaryCellsToPoints:
    """BoundaryCellsToPoints moves cell fields onto a point-cloud boundary."""

    def test_fields_are_moved(self):
        """The fields leave the source and land on the new points, renamed."""
        domain = _domain()
        wall = domain.boundaries["wall"]
        out = _transform().apply_to_domain(domain)
        new = out.boundaries["wall_points"]
        assert torch.equal(new.point_data["p_w"], wall.cell_data["p"])
        assert torch.equal(new.point_data["tau_w"], wall.cell_data["tau"])
        assert set(out.boundaries["wall"].cell_data.keys()) == {"other"}
        assert torch.equal(out.boundaries["wall"].cells, wall.cells)

    def test_points_at_centroids_plus_offset(self):
        """Each point is its cell's centroid moved offset along the unit normal."""
        domain = _domain()
        wall = domain.boundaries["wall"]
        out = _transform(offset=0.01).apply_to_domain(domain)
        expected = wall.cell_centroids + 0.01 * wall.cell_normals
        torch.testing.assert_close(out.boundaries["wall_points"].points, expected)
        out = _transform().apply_to_domain(domain)
        assert torch.equal(out.boundaries["wall_points"].points, wall.cell_centroids)

    def test_measure_with_every_cell(self):
        """With every cell, each point carries its cell's area."""
        domain = _domain()
        out = _transform().apply_to_domain(domain)
        torch.testing.assert_close(
            point_measures(out.boundaries["wall_points"]),
            domain.boundaries["wall"].cell_areas,
        )

    def test_measure_unbiased_with_a_draw(self):
        """With n of N cells drawn, the measures sum to the area in expectation."""
        domain = _domain()
        area = domain.boundaries["wall"].cell_areas.sum()
        transform = _transform(n_cells=40)
        transform.set_generator(torch.Generator().manual_seed(0))
        sums = []
        for _ in range(400):
            new = transform.apply_to_domain(domain).boundaries["wall_points"]
            assert new.n_points == 40
            sums.append(point_measures(new).sum())
        torch.testing.assert_close(torch.stack(sums).mean(), area, rtol=0.01, atol=0)

    def test_draw_reproducible_with_a_seed(self):
        """The same seed draws the same cells; another seed draws others."""

        def draw(seed: int) -> torch.Tensor:
            transform = _transform(n_cells=40)
            transform.set_generator(torch.Generator().manual_seed(seed))
            return transform.apply_to_domain(_domain()).boundaries["wall_points"]

        assert torch.equal(draw(1).point_data["p_w"], draw(1).point_data["p_w"])
        assert not torch.equal(draw(1).point_data["p_w"], draw(2).point_data["p_w"])

    def test_other_boundaries_untouched(self):
        """The interior, the other boundaries and global data pass through."""
        domain = _domain()
        out = _transform(n_cells=40).apply_to_domain(domain)
        assert out.boundaries["inlet"] is domain.boundaries["inlet"]
        assert torch.equal(out.interior.points, domain.interior.points)
        assert torch.equal(out.global_data["U_inf"], domain.global_data["U_inf"])
        assert sorted(out.boundary_names) == ["inlet", "wall", "wall_points"]

    def test_bare_mesh_raises(self):
        """The transform needs a DomainMesh."""
        with pytest.raises(TypeError, match="DomainMesh"):
            _transform()(single_triangle_3d.load())

    def test_missing_source_or_field_raises(self):
        """An absent source boundary or field is an error."""
        domain = _domain()
        with pytest.raises(KeyError, match="nowhere"):
            BoundaryCellsToPoints(
                source="nowhere", name="x", fields={"p": "p_w"}
            ).apply_to_domain(domain)
        with pytest.raises(KeyError, match="missing"):
            BoundaryCellsToPoints(
                source="wall", name="x", fields={"missing": "m"}
            ).apply_to_domain(domain)

    def test_registered(self):
        """The transform is in the datapipe registry, for ``${dp:...}`` configs."""
        assert "BoundaryCellsToPoints" in COMPONENT_REGISTRY
