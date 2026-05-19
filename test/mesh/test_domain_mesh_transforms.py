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

"""Tests for DomainMesh transform passthrough methods."""

import math

import pytest
import torch

from physicsnemo.mesh import DomainMesh, Mesh
from physicsnemo.mesh.primitives.basic import (
    single_edge_2d,
    single_tetrahedron,
    single_triangle_2d,
    single_triangle_3d,
    two_tetrahedra,
)

### Fixtures


@pytest.fixture
def tet_domain():
    """DomainMesh: tet interior (m=3, s=3) with data, 2 tri boundaries, global_data."""
    interior = two_tetrahedra.load()
    interior.point_data["temperature"] = torch.randn(interior.n_points)
    interior.cell_data["pressure"] = torch.randn(interior.n_cells)
    wall = single_triangle_3d.load()
    wall.cell_data["wall_shear"] = torch.randn(wall.n_cells)
    inlet = single_triangle_3d.load()
    inlet.cell_data["mass_flux"] = torch.randn(inlet.n_cells)
    return DomainMesh(
        interior=interior,
        boundaries={"wall": wall, "inlet": inlet},
        global_data={"Re": torch.tensor(1e6), "AoA": torch.tensor(5.0)},
    )


@pytest.fixture
def no_boundary_domain():
    """DomainMesh: single tet interior with no boundaries or global data."""
    return DomainMesh(interior=single_tetrahedron.load())


def _open_tetrahedron_faces() -> dict[str, Mesh]:
    """4 triangular boundary patches forming a closed unit tetrahedron.

    Each face is a separate ``Mesh`` whose vertices are duplicated copies
    of the canonical tet vertices - this models the realistic case where
    boundary patches are meshed independently and only "share" vertices
    geometrically, not by index.
    """
    p = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    cells = torch.tensor([[0, 1, 2]])

    def face(*idx: int, offset: torch.Tensor | None = None) -> Mesh:
        pts = p[list(idx)].clone()
        if offset is not None:
            pts = pts + offset
        return Mesh(points=pts, cells=cells)

    return {
        "f0": face(0, 1, 2),
        "f1": face(0, 1, 3),
        "f2": face(0, 2, 3),
        "f3": face(1, 2, 3),
    }


### Properties


class TestProperties:
    """Tests for DomainMesh property accessors."""

    def test_n_boundaries_counts_entries(self, tet_domain):
        """n_boundaries reflects the actual number of boundary keys.

        Regression for the bug where ``len(self.boundaries)`` returned 0
        for a TensorDict with ``batch_size=[]`` regardless of how many
        named entries it carried.
        """
        assert tet_domain.n_boundaries == 2

    def test_boundary_names_sorted(self, tet_domain):
        """boundary_names returns keys in sorted order."""
        assert tet_domain.boundary_names == ["inlet", "wall"]


### apply_to_meshes


class TestApplyToMeshes:
    """Tests for DomainMesh.apply_to_meshes."""

    def test_applies_fn_to_interior(self, tet_domain):
        original_points = tet_domain.interior.points.clone()
        dm2 = tet_domain.apply_to_meshes(lambda m: m.translate([1, 0, 0]))
        expected = original_points + torch.tensor([1.0, 0.0, 0.0])
        assert torch.allclose(dm2.interior.points, expected)

    def test_applies_fn_to_all_boundaries(self, tet_domain):
        offset = torch.tensor([0.0, 0.0, 1.0])
        dm2 = tet_domain.apply_to_meshes(lambda m: m.translate([0, 0, 1]))
        for name in tet_domain.boundary_names:
            original = tet_domain.boundaries[name].points
            assert torch.allclose(dm2.boundaries[name].points, original + offset)

    def test_preserves_global_data(self, tet_domain):
        dm2 = tet_domain.apply_to_meshes(lambda m: m.translate([1, 1, 1]))
        assert torch.equal(dm2.global_data["Re"], tet_domain.global_data["Re"])
        assert torch.equal(dm2.global_data["AoA"], tet_domain.global_data["AoA"])

    def test_global_data_is_independent_copy(self, tet_domain):
        """Mutating transformed domain's global_data must not affect original."""
        original_re = tet_domain.global_data["Re"].clone()
        dm2 = tet_domain.apply_to_meshes(lambda m: m.translate([1, 0, 0]))
        dm2.global_data["Re"].fill_(0.0)
        assert torch.equal(tet_domain.global_data["Re"], original_re)

    def test_works_with_no_boundaries(self, no_boundary_domain):
        dm2 = no_boundary_domain.apply_to_meshes(lambda m: m.translate([1, 0, 0]))
        assert dm2.n_boundaries == 0
        assert dm2.interior.points[0, 0].item() == pytest.approx(1.0)

    def test_returns_domain_mesh(self, tet_domain):
        dm2 = tet_domain.apply_to_meshes(lambda m: m)
        assert isinstance(dm2, DomainMesh)

    def test_interior_only(self, tet_domain):
        """apply_to_meshes with boundaries=False should leave boundaries unchanged."""
        dm2 = tet_domain.apply_to_meshes(
            lambda m: m.translate([1, 0, 0]), boundaries=False
        )
        assert not torch.equal(dm2.interior.points, tet_domain.interior.points)
        for name in tet_domain.boundary_names:
            assert torch.equal(
                dm2.boundaries[name].points, tet_domain.boundaries[name].points
            )

    def test_boundaries_only(self, tet_domain):
        """apply_to_meshes with interior=False should leave interior unchanged."""
        dm2 = tet_domain.apply_to_meshes(
            lambda m: m.translate([1, 0, 0]), interior=False
        )
        assert torch.equal(dm2.interior.points, tet_domain.interior.points)
        for name in tet_domain.boundary_names:
            assert not torch.equal(
                dm2.boundaries[name].points, tet_domain.boundaries[name].points
            )


### Geometric Transforms


class TestTranslate:
    """Tests for DomainMesh.translate passthrough."""

    def test_shifts_all_points(self, tet_domain):
        offset = [2.0, -1.0, 0.5]
        dm2 = tet_domain.translate(offset)
        offset_t = torch.tensor(offset)
        assert torch.allclose(
            dm2.interior.points, tet_domain.interior.points + offset_t
        )
        for name in tet_domain.boundary_names:
            assert torch.allclose(
                dm2.boundaries[name].points,
                tet_domain.boundaries[name].points + offset_t,
            )

    def test_preserves_cells(self, tet_domain):
        dm2 = tet_domain.translate([1, 0, 0])
        assert torch.equal(dm2.interior.cells, tet_domain.interior.cells)

    def test_preserves_global_data(self, tet_domain):
        dm2 = tet_domain.translate([1, 0, 0])
        assert torch.equal(dm2.global_data["Re"], tet_domain.global_data["Re"])


class TestRotate:
    """Tests for DomainMesh.rotate passthrough."""

    def test_2d_rotation(self):
        """Rotate a 2D domain by 90 degrees; verify point (1,0) -> (0,1)."""
        dm = DomainMesh(
            interior=single_triangle_2d.load(),
            boundaries={"edge": single_edge_2d.load()},
        )
        dm2 = dm.rotate(angle=math.pi / 2)
        # Primitive point[1] = (1, 0) rotates to (0, 1)
        assert dm2.interior.points[1, 0].item() == pytest.approx(0.0, abs=1e-6)
        assert dm2.interior.points[1, 1].item() == pytest.approx(1.0, abs=1e-6)

    def test_roundtrip(self, tet_domain):
        """Rotating and un-rotating recovers original points."""
        dm2 = tet_domain.rotate(angle=math.pi / 4, axis="z")
        dm3 = dm2.rotate(angle=-math.pi / 4, axis="z")
        assert torch.allclose(
            dm3.interior.points, tet_domain.interior.points, atol=1e-6
        )


class TestScale:
    """Tests for DomainMesh.scale passthrough."""

    def test_uniform_scale(self, tet_domain):
        dm2 = tet_domain.scale(factor=2.0)
        assert torch.allclose(dm2.interior.points, tet_domain.interior.points * 2.0)
        for name in tet_domain.boundary_names:
            assert torch.allclose(
                dm2.boundaries[name].points,
                tet_domain.boundaries[name].points * 2.0,
            )

    def test_preserves_global_data(self, tet_domain):
        dm2 = tet_domain.scale(factor=3.0)
        assert torch.equal(dm2.global_data["Re"], tet_domain.global_data["Re"])


class TestTransform:
    """Tests for DomainMesh.transform passthrough."""

    def test_identity(self, tet_domain):
        dm2 = tet_domain.transform(torch.eye(3))
        assert torch.allclose(dm2.interior.points, tet_domain.interior.points)

    def test_scale_via_matrix(self, tet_domain):
        dm2 = tet_domain.transform(2.0 * torch.eye(3))
        assert torch.allclose(dm2.interior.points, tet_domain.interior.points * 2.0)


### Cleanup / Refinement


class TestClean:
    """Tests for DomainMesh.clean passthrough."""

    def test_cleans_all_meshes(self):
        """clean() merges duplicate points in interior; boundary unchanged."""
        # Interior with intentional duplicate points (no primitive has this)
        interior = Mesh(
            points=torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [1.0, 1.0]]),
            cells=torch.tensor([[0, 1, 3], [2, 1, 3]]),
        )
        boundary = single_edge_2d.load()
        dm = DomainMesh(interior=interior, boundaries={"edge": boundary})
        dm2 = dm.clean()
        assert dm2.interior.n_points < dm.interior.n_points
        assert dm2.boundaries["edge"].n_points == boundary.n_points

    def test_no_boundaries(self, no_boundary_domain):
        dm2 = no_boundary_domain.clean()
        assert isinstance(dm2, DomainMesh)
        assert dm2.n_boundaries == 0


class TestStripCaches:
    """Tests for DomainMesh.strip_caches passthrough."""

    def test_clears_cached_geometry(self):
        """Accessing cell_normals populates cache; strip_caches clears it."""
        dm = DomainMesh(interior=single_triangle_3d.load())
        _ = dm.interior.cell_normals
        assert "normals" in dm.interior._cache["cell"].keys()
        dm2 = dm.strip_caches()
        assert "normals" not in dm2.interior._cache["cell"].keys()


class TestSubdivide:
    """Tests for DomainMesh.subdivide passthrough."""

    def test_increases_cell_count(self):
        """Linear subdivision: tet -> 8 child tets, tri -> 4 child tris."""
        dm = DomainMesh(
            interior=single_tetrahedron.load(),
            boundaries={"wall": single_triangle_3d.load()},
        )
        dm2 = dm.subdivide(levels=1, filter="linear")
        assert dm2.interior.n_cells == 8
        assert dm2.boundaries["wall"].n_cells == 4


### Data Operations


class TestCellDataToPointData:
    """Tests for DomainMesh.cell_data_to_point_data passthrough."""

    def test_converts_all_meshes(self, tet_domain):
        dm2 = tet_domain.cell_data_to_point_data()
        assert "pressure" in dm2.interior.point_data.keys()
        assert "wall_shear" in dm2.boundaries["wall"].point_data.keys()
        assert "mass_flux" in dm2.boundaries["inlet"].point_data.keys()

    def test_preserves_original_cell_data(self, tet_domain):
        dm2 = tet_domain.cell_data_to_point_data()
        assert "pressure" in dm2.interior.cell_data.keys()


class TestPointDataToCellData:
    """Tests for DomainMesh.point_data_to_cell_data passthrough."""

    def test_converts_all_meshes(self, tet_domain):
        dm2 = tet_domain.point_data_to_cell_data()
        assert "temperature" in dm2.interior.cell_data.keys()


class TestComputePointDerivatives:
    """Tests for DomainMesh.compute_point_derivatives passthrough."""

    def test_gradient_keys_appear_in_interior(self, tet_domain):
        dm2 = tet_domain.compute_point_derivatives()
        assert "temperature_gradient" in dm2.interior.point_data.keys()

    def test_preserves_boundary_structure(self, tet_domain):
        dm2 = tet_domain.compute_point_derivatives()
        assert set(dm2.boundary_names) == set(tet_domain.boundary_names)


class TestComputeCellDerivatives:
    """Tests for DomainMesh.compute_cell_derivatives passthrough."""

    def test_gradient_keys_appear_in_interior(self):
        points = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.5, 0.5, 0.5],
            ],
            dtype=torch.float32,
        )
        cells = torch.tensor([[0, 1, 2, 4], [0, 1, 3, 4], [0, 2, 3, 4], [1, 2, 3, 4]])
        interior = Mesh(points=points, cells=cells)
        interior.cell_data["pressure"] = torch.randn(interior.n_cells)
        dm = DomainMesh(interior=interior)
        dm2 = dm.compute_cell_derivatives()
        assert "pressure_gradient" in dm2.interior.cell_data.keys()


### Validation


class TestValidate:
    """Tests for DomainMesh.validate passthrough."""

    def test_report_structure(self, tet_domain):
        report = tet_domain.validate()
        assert "interior" in report
        assert "boundaries" in report
        assert "valid" in report
        assert isinstance(report["valid"], bool)

    def test_report_contains_all_boundaries(self, tet_domain):
        report = tet_domain.validate()
        assert set(report["boundaries"].keys()) == {"wall", "inlet"}

    def test_no_boundaries(self, no_boundary_domain):
        report = no_boundary_domain.validate()
        assert report["boundaries"] == {}
        assert report["valid"] == report["interior"]["valid"]

    def test_invalid_mesh_propagates(self):
        """Out-of-bounds cell index causes valid=False."""
        # Intentionally invalid mesh (no primitive has out-of-bounds indices)
        interior = Mesh(
            points=torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
            cells=torch.tensor([[0, 1, 99]]),
        )
        dm = DomainMesh(interior=interior)
        report = dm.validate()
        assert not report["valid"]


### Boundary watertightness


class TestIsBoundaryWatertight:
    """Tests for DomainMesh.is_boundary_watertight."""

    def test_no_boundaries_returns_false(self, no_boundary_domain):
        """A domain with no boundaries cannot be watertight."""
        assert not no_boundary_domain.is_boundary_watertight()

    def test_closed_tet_is_watertight(self):
        """4 boundary triangles forming a closed tet are watertight."""
        dm = DomainMesh(
            interior=Mesh(points=torch.zeros((1, 3))),
            boundaries=_open_tetrahedron_faces(),
        )
        assert dm.is_boundary_watertight()

    def test_open_surface_is_not_watertight(self):
        """Removing one face must produce a non-watertight surface."""
        faces = _open_tetrahedron_faces()
        del faces["f3"]
        dm = DomainMesh(
            interior=Mesh(points=torch.zeros((1, 3))),
            boundaries=faces,
        )
        assert not dm.is_boundary_watertight()

    def test_default_tolerance_handles_float_noise(self):
        """Default tolerance must absorb realistic float32 round-off on shared vertices.

        Independently-meshed boundary patches share physical vertices that
        carry slightly different float values after any prior transform.
        The default tolerance must merge such near-duplicates so the
        surface is correctly classified as watertight.
        """
        eps = 1e-7  # Per-face perturbation; max pairwise distance ~2 * eps
        p = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )
        cells = torch.tensor([[0, 1, 2]])

        def face(idx: list[int], offset: torch.Tensor) -> Mesh:
            return Mesh(points=p[idx].clone() + offset, cells=cells)

        offsets = [
            torch.tensor([+eps, 0.0, 0.0]),
            torch.tensor([-eps, 0.0, 0.0]),
            torch.tensor([0.0, +eps, 0.0]),
            torch.tensor([0.0, -eps, 0.0]),
        ]
        face_defs = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
        dm = DomainMesh(
            interior=Mesh(points=torch.zeros((1, 3))),
            boundaries={
                f"f{i}": face(idx, off)
                for i, (idx, off) in enumerate(zip(face_defs, offsets, strict=True))
            },
        )

        ### Default tolerance (1e-6) comfortably absorbs the ~2e-7 mismatch
        assert dm.is_boundary_watertight()
        ### A too-tight explicit tolerance must NOT mask the noise -
        ### documents the failure mode the default protects against.
        assert not dm.is_boundary_watertight(tolerance=1e-12)

    def test_explicit_tolerance_passthrough(self):
        """An explicit tolerance is forwarded verbatim to Mesh.clean."""
        dm = DomainMesh(
            interior=Mesh(points=torch.zeros((1, 3))),
            boundaries=_open_tetrahedron_faces(),
        )
        ### Exact-match coincidence is below any positive tolerance, so any
        ### reasonable explicit value still reports watertight.
        assert dm.is_boundary_watertight(tolerance=1e-3)


### Boundary merging


class TestMergeBoundaries:
    """Tests for DomainMesh.merge_boundaries."""

    def test_no_boundaries_raises(self, no_boundary_domain):
        """Merging a domain with zero boundaries is an error."""
        with pytest.raises(ValueError, match="No boundary meshes"):
            no_boundary_domain.merge_boundaries()

    def test_default_strips_data(self, tet_domain):
        """Default merge produces a geometry-only mesh.

        ``tet_domain`` boundaries carry heterogeneous cell_data keys
        (``wall_shear`` vs ``mass_flux``). Stripping is the only way to
        merge such patches; the default does so silently.
        """
        merged = tet_domain.merge_boundaries()
        assert len(list(merged.point_data.keys())) == 0
        assert len(list(merged.cell_data.keys())) == 0
        ### Geometry is concatenated as expected
        expected_points = sum(
            tet_domain.boundaries[name].n_points for name in tet_domain.boundary_names
        )
        expected_cells = sum(
            tet_domain.boundaries[name].n_cells for name in tet_domain.boundary_names
        )
        assert merged.n_points == expected_points
        assert merged.n_cells == expected_cells

    def test_preserve_data_round_trips_homogeneous_fields(self):
        """preserve_data=True keeps fields that share keys across boundaries."""
        ### Two boundaries with the SAME cell_data key set
        b1 = single_triangle_3d.load()
        b1.cell_data["shear"] = torch.tensor([0.5])
        b2 = single_triangle_3d.load()
        b2.cell_data["shear"] = torch.tensor([1.5])
        dm = DomainMesh(
            interior=Mesh(points=torch.zeros((1, 3))),
            boundaries={"a": b1, "b": b2},
        )
        merged = dm.merge_boundaries(preserve_data=True)
        assert "shear" in merged.cell_data.keys()
        ### Merged values are concatenated in sorted-name order: a then b
        assert torch.allclose(merged.cell_data["shear"], torch.tensor([0.5, 1.5]))

    def test_preserve_data_raises_on_heterogeneous_keys(self, tet_domain):
        """preserve_data=True surfaces the underlying Mesh.merge mismatch."""
        with pytest.raises(ValueError):
            tet_domain.merge_boundaries(preserve_data=True)


### Visualization


@pytest.fixture
def mpl():
    """Headless matplotlib for visualization smoke tests; auto-closes figures."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    yield plt
    plt.close("all")


class TestDraw:
    """Smoke tests for DomainMesh.draw (matplotlib backend)."""

    def test_returns_canvas_with_boundaries(self, mpl, tet_domain):
        """draw() returns the matplotlib Axes with boundaries overlaid."""
        ax = tet_domain.draw(backend="matplotlib", show=False)
        ### tet_domain interior is 3D tetrahedral, so backend gives a 3D Axes
        import matplotlib.axes

        assert isinstance(ax, matplotlib.axes.Axes)

    def test_returns_canvas_no_boundaries(self, mpl, no_boundary_domain):
        """draw() works on a domain with no boundaries."""
        ax = no_boundary_domain.draw(backend="matplotlib", show=False)
        import matplotlib.axes

        assert isinstance(ax, matplotlib.axes.Axes)

    def test_reuses_supplied_axes(self, mpl, tet_domain):
        """draw(ax=existing) overlays on the supplied canvas."""
        fig = mpl.figure()
        ax_in = fig.add_subplot(111, projection="3d")
        ax_out = tet_domain.draw(backend="matplotlib", show=False, ax=ax_in)
        assert ax_out is ax_in

    def test_boundary_kwargs_override_defaults(self, mpl, tet_domain, monkeypatch):
        """boundary_kwargs values reach Mesh.draw and override auto defaults."""
        captured: list[dict] = []
        original_draw = Mesh.draw

        def spy(self, **kwargs):
            captured.append(kwargs)
            return original_draw(self, **kwargs)

        monkeypatch.setattr(Mesh, "draw", spy)

        tet_domain.draw(
            backend="matplotlib",
            show=False,
            boundary_kwargs={"alpha_cells": 0.7, "show_edges": True},
        )
        ### One call for interior + one per boundary
        assert len(captured) == 1 + tet_domain.n_boundaries
        boundary_calls = captured[1:]
        for call in boundary_calls:
            ### User overrides win against the auto defaults
            assert call["alpha_cells"] == 0.7
            assert call["show_edges"] is True
            ### Default that wasn't overridden survives
            assert call["alpha_points"] == 0


### Chaining


class TestChaining:
    """Tests for chaining multiple DomainMesh transforms."""

    def test_translate_scale_clean(self, tet_domain):
        dm2 = tet_domain.translate([1, 0, 0]).scale(2.0).clean(merge_points=False)
        assert isinstance(dm2, DomainMesh)
        expected = (tet_domain.interior.points + torch.tensor([1.0, 0.0, 0.0])) * 2.0
        assert torch.allclose(dm2.interior.points, expected)

    def test_chain_preserves_global_data(self, tet_domain):
        dm2 = tet_domain.translate([1, 0, 0]).rotate(0.1, axis="z").scale(0.5)
        assert torch.equal(dm2.global_data["Re"], tet_domain.global_data["Re"])

    def test_chain_with_no_boundaries(self, no_boundary_domain):
        dm2 = no_boundary_domain.translate([1, 0, 0]).scale(3.0)
        assert dm2.n_boundaries == 0
        expected = (
            no_boundary_domain.interior.points + torch.tensor([1.0, 0.0, 0.0])
        ) * 3.0
        assert torch.allclose(dm2.interior.points, expected)


### Domain-Level global_data Transformation


class TestDomainGlobalDataTransform:
    """Tests for domain-level global_data transformation via transform_global_data."""

    @pytest.fixture
    def domain_2d(self):
        """2D domain with a directional vector and a scalar in global_data."""
        return DomainMesh(
            interior=single_triangle_2d.load(),
            boundaries={"edge": single_edge_2d.load()},
            global_data={
                "velocity": torch.tensor([1.0, 0.0]),
                "Re": torch.tensor(1e6),
            },
        )

    @pytest.fixture
    def domain_3d(self):
        """3D domain with a directional vector and a scalar in global_data."""
        return DomainMesh(
            interior=single_triangle_3d.load(),
            global_data={
                "velocity": torch.tensor([1.0, 0.0, 0.0]),
                "Re": torch.tensor(1e6),
            },
        )

    def test_rotate_transforms_domain_velocity_2d(self, domain_2d):
        """90-degree CCW rotation: [1, 0] -> [0, 1]."""
        dm2 = domain_2d.rotate(angle=math.pi / 2, transform_global_data=True)
        assert dm2.global_data["velocity"][0].item() == pytest.approx(0.0, abs=1e-6)
        assert dm2.global_data["velocity"][1].item() == pytest.approx(1.0, abs=1e-6)

    def test_rotate_transforms_domain_velocity_3d(self, domain_3d):
        """90-degree rotation about z: [1, 0, 0] -> [0, 1, 0]."""
        dm2 = domain_3d.rotate(angle=math.pi / 2, axis="z", transform_global_data=True)
        assert dm2.global_data["velocity"][0].item() == pytest.approx(0.0, abs=1e-6)
        assert dm2.global_data["velocity"][1].item() == pytest.approx(1.0, abs=1e-6)
        assert dm2.global_data["velocity"][2].item() == pytest.approx(0.0, abs=1e-6)

    def test_rotate_preserves_domain_scalars(self, domain_2d):
        """Scalars in global_data are invariant under rotation."""
        dm2 = domain_2d.rotate(angle=math.pi / 2, transform_global_data=True)
        assert dm2.global_data["Re"].item() == pytest.approx(1e6)

    def test_rotate_default_preserves_domain_global_data(self, domain_2d):
        """Default transform_global_data=False leaves domain global_data unchanged."""
        dm2 = domain_2d.rotate(angle=math.pi / 2)
        assert torch.equal(
            dm2.global_data["velocity"], domain_2d.global_data["velocity"]
        )

    def test_scale_transforms_domain_velocity(self, domain_2d):
        """Uniform scale by 2: [1, 0] -> [2, 0]."""
        dm2 = domain_2d.scale(factor=2.0, transform_global_data=True)
        assert dm2.global_data["velocity"][0].item() == pytest.approx(2.0)
        assert dm2.global_data["velocity"][1].item() == pytest.approx(0.0)

    def test_scale_nonuniform_transforms_domain_velocity(self, domain_2d):
        """Non-uniform scale [3, 0.5]: [1, 0] -> [3, 0]."""
        dm2 = domain_2d.scale(
            factor=torch.tensor([3.0, 0.5]), transform_global_data=True
        )
        assert dm2.global_data["velocity"][0].item() == pytest.approx(3.0)
        assert dm2.global_data["velocity"][1].item() == pytest.approx(0.0)

    def test_transform_transforms_domain_velocity(self, domain_2d):
        """Apply 90-degree rotation matrix via transform(): [1, 0] -> [0, 1]."""
        R = torch.tensor([[0.0, -1.0], [1.0, 0.0]])
        dm2 = domain_2d.transform(matrix=R, transform_global_data=True)
        assert dm2.global_data["velocity"][0].item() == pytest.approx(0.0, abs=1e-6)
        assert dm2.global_data["velocity"][1].item() == pytest.approx(1.0, abs=1e-6)

    def test_transform_default_preserves_domain_global_data(self, domain_2d):
        """Default transform_global_data=False leaves domain global_data unchanged."""
        R = torch.tensor([[0.0, -1.0], [1.0, 0.0]])
        dm2 = domain_2d.transform(matrix=R)
        assert torch.equal(
            dm2.global_data["velocity"], domain_2d.global_data["velocity"]
        )

    def test_selective_domain_global_data(self, domain_2d):
        """Dict mask transforms only named keys, leaves others unchanged."""
        dm2 = domain_2d.rotate(
            angle=math.pi / 2, transform_global_data={"velocity": True}
        )
        assert dm2.global_data["velocity"][0].item() == pytest.approx(0.0, abs=1e-6)
        assert dm2.global_data["velocity"][1].item() == pytest.approx(1.0, abs=1e-6)
        assert dm2.global_data["Re"].item() == pytest.approx(1e6)

    def test_selective_skips_unmentioned_domain_keys(self, domain_2d):
        """Keys not in the mask dict are not transformed."""
        dm2 = domain_2d.rotate(angle=math.pi / 2, transform_global_data={"Re": False})
        assert torch.equal(
            dm2.global_data["velocity"], domain_2d.global_data["velocity"]
        )

    def test_rotate_with_center_transforms_domain_global_data(self, domain_2d):
        """Rotation about a center still transforms domain global_data correctly."""
        dm2 = domain_2d.rotate(
            angle=math.pi / 2, center=[1.0, 0.0], transform_global_data=True
        )
        assert dm2.global_data["velocity"][0].item() == pytest.approx(0.0, abs=1e-6)
        assert dm2.global_data["velocity"][1].item() == pytest.approx(1.0, abs=1e-6)
