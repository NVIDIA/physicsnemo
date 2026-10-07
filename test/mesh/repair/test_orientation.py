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

"""Tests for fix_orientation on simplicial meshes of every manifold dimension."""

import itertools
from collections import defaultdict

import pytest
import torch

from physicsnemo.mesh import Mesh
from physicsnemo.mesh.repair import fix_orientation, repair_mesh


def _permutation_parity(values: tuple[int, ...]) -> int:
    """Parity (0 even, 1 odd) of the permutation that sorts ``values``."""
    return sum(a > b for a, b in itertools.combinations(values, 2)) % 2


def _orientation_conflicts(cells: torch.Tensor) -> int:
    """Count facets of exactly two cells that both cells orient the same way.

    Independent reference: cell ``(v_0, ..., v_n)`` induces ``(-1)**i`` times
    the sorting parity of the remaining vertices on the facet without ``v_i``.
    A consistently oriented manifold induces opposite orientations on every
    interior facet, so the count is zero.
    """
    induced = defaultdict(list)
    for cell in cells.tolist():
        for i in range(len(cell)):
            facet = tuple(cell[:i] + cell[i + 1 :])
            induced[tuple(sorted(facet))].append((i + _permutation_parity(facet)) % 2)
    return sum(len(o) == 2 and o[0] == o[1] for o in induced.values())


def _reverse(mesh: Mesh, reversed_cells: torch.Tensor) -> Mesh:
    """Reverse the orientation of the selected cells by swapping their first two vertices."""
    cells = mesh.cells
    swapped = torch.cat([cells[:, 1:2], cells[:, :1], cells[:, 2:]], dim=1)
    cells = torch.where(reversed_cells[:, None], swapped, mesh.cells)
    return Mesh(points=mesh.points, cells=cells)


def _signed_volumes(mesh: Mesh) -> torch.Tensor:
    """Signed measure of full-dimensional simplices (triangles in 2D, tets in 3D)."""
    vertices = mesh.points[mesh.cells]
    return torch.linalg.det(vertices[:, 1:] - vertices[:, :1])


def _every_third(n: int, device) -> torch.Tensor:
    """Select every third cell, sparing cell 0 (each component keeps its first cell)."""
    return torch.arange(n, device=device) % 3 == 1


class TestRepairsWindings:
    """Scrambled windings are restored for every manifold dimension."""

    def test_closed_surfaces_in_3d(self, device):
        from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral

        sphere = sphere_icosahedral.load(subdivisions=3, device=device)
        two_spheres = Mesh.merge([sphere, sphere.translate([3.0, 0.0, 0.0])])
        local = torch.arange(two_spheres.n_cells, device=device) % sphere.n_cells
        reversed_cells = local % 3 == 1
        scrambled = _reverse(two_spheres, reversed_cells)

        oriented, stats = fix_orientation(scrambled)

        assert stats == {
            "n_faces_flipped": int(reversed_cells.sum()),
            "n_components": 2,
            "largest_component_size": sphere.n_cells,
            "n_non_orientable_components": 0,
        }
        torch.testing.assert_close(oriented.cell_normals, two_spheres.cell_normals)

    def test_sharp_creases(self, device):
        """Faces meeting at right angles are oriented by connectivity, not normals."""
        from physicsnemo.mesh.primitives.surfaces import cube_surface

        cube = cube_surface.load(device=device).subdivide(levels=2, filter="linear")
        cube = Mesh(points=cube.points, cells=cube.cells)
        scrambled = _reverse(cube, _every_third(cube.n_cells, device))
        assert _orientation_conflicts(scrambled.cells.cpu()) > 0

        oriented, stats = fix_orientation(scrambled)

        assert _orientation_conflicts(oriented.cells.cpu()) == 0
        assert stats["n_faces_flipped"] == int(_every_third(cube.n_cells, device).sum())
        # Outward like the first cell: the enclosed volume is positive.
        assert oriented.cell_normals.isfinite().all()
        volume = (oriented.cell_centroids * oriented.cell_normals).sum(-1)
        assert ((volume * oriented.cell_areas).sum() > 0).item()

    def test_planar_triangles(self, device):
        from physicsnemo.mesh.primitives.planar import structured_grid

        grid = structured_grid.load(n_x=9, n_y=7, device=device)
        generator = torch.Generator().manual_seed(0)
        reversed_cells = (torch.rand(grid.n_cells, generator=generator) < 0.4).to(
            device
        )
        reversed_cells[0] = False
        scrambled = _reverse(grid, reversed_cells)

        oriented, stats = fix_orientation(scrambled)

        assert stats["n_faces_flipped"] == int(reversed_cells.sum())
        signs = _signed_volumes(oriented).sign()
        assert (signs == signs[0]).all()

    def test_tetrahedra(self, device):
        from physicsnemo.mesh.primitives.volumes import cube_volume

        cube = cube_volume.load(subdivisions=3, device=device)
        # Make every tet positive first; the Kuhn children alternate in sign.
        positive = _signed_volumes(cube) > 0
        cube = _reverse(cube, ~positive)
        generator = torch.Generator().manual_seed(1)
        reversed_cells = (torch.rand(cube.n_cells, generator=generator) < 0.5).to(
            device
        )
        reversed_cells[0] = False
        scrambled = _reverse(cube, reversed_cells)

        oriented, stats = fix_orientation(scrambled)

        assert stats["n_components"] == 1
        assert stats["n_faces_flipped"] == int(reversed_cells.sum())
        assert (_signed_volumes(oriented) > 0).all()

    def test_segments(self, device):
        """A closed polyline gets one direction of travel."""
        n = 40
        theta = torch.linspace(0, 2 * torch.pi, n + 1, device=device)[:-1]
        points = torch.stack([theta.cos(), theta.sin(), 0 * theta], dim=1)
        index = torch.arange(n, device=device)
        loop = Mesh(points=points, cells=torch.stack([index, (index + 1) % n], dim=1))
        scrambled = _reverse(loop, _every_third(n, device))

        oriented, stats = fix_orientation(scrambled)

        assert stats["n_faces_flipped"] == int(_every_third(n, device).sum())
        torch.testing.assert_close(oriented.cells, loop.cells, rtol=0, atol=0)

    def test_matches_reference_on_random_reversals(self, device):
        """Agreement with a brute-force reference on a larger surface."""
        from physicsnemo.mesh.primitives.surfaces import torus

        mesh = torus.load(n_major=80, n_minor=40, device=device)
        generator = torch.Generator().manual_seed(2)
        reversed_cells = (torch.rand(mesh.n_cells, generator=generator) < 0.5).to(
            device
        )
        scrambled = _reverse(mesh, reversed_cells)

        oriented, stats = fix_orientation(scrambled)

        assert _orientation_conflicts(oriented.cells.cpu()) == 0
        assert stats["n_components"] == 1
        # The first cell keeps its orientation, so every cell follows the
        # original orientation, or every cell the reverse if cell 0 was reversed.
        sign = -1.0 if reversed_cells[0] else 1.0
        alignment = (oriented.cell_normals * mesh.cell_normals).sum(-1) * sign
        assert (alignment > 0.99).all()


class TestSpecialTopology:
    def test_non_orientable_surface_is_left_as_given(self, device):
        from physicsnemo.mesh.primitives.surfaces import mobius_strip

        strip = mobius_strip.load(device=device)

        oriented, stats = fix_orientation(strip)

        assert stats["n_non_orientable_components"] == 1
        assert stats["n_faces_flipped"] == 0
        torch.testing.assert_close(oriented.cells, strip.cells, rtol=0, atol=0)

    def test_non_manifold_edges_do_not_relate_cells(self, device):
        """Three triangles on one edge are oriented independently."""
        points = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 1.0, 0.0],
                [0.5, -1.0, 0.0],
                [0.5, 0.0, 1.0],
            ],
            device=device,
        )
        cells = torch.tensor([[0, 1, 2], [0, 1, 3], [0, 1, 4]], device=device)

        oriented, stats = fix_orientation(Mesh(points=points, cells=cells))

        assert stats["n_components"] == 3
        assert stats["n_faces_flipped"] == 0
        torch.testing.assert_close(oriented.cells, cells, rtol=0, atol=0)

    def test_degenerate_cells_are_left_as_given(self, device):
        points = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], device=device
        )
        # Cells 0 and 2 traverse their shared edge (1, 2) in the same direction
        cells = torch.tensor([[0, 1, 2], [3, 3, 0], [1, 2, 3]], device=device)

        oriented, stats = fix_orientation(Mesh(points=points, cells=cells))

        torch.testing.assert_close(oriented.cells[1], cells[1], rtol=0, atol=0)
        assert stats["n_faces_flipped"] == 1
        assert stats["n_components"] == 2
        assert _orientation_conflicts(oriented.cells[[0, 2]].cpu()) == 0

    def test_point_clouds_raise(self, device):
        mesh = Mesh(points=torch.rand(4, 3, device=device), cells=None)
        with pytest.raises(ValueError, match="at least two vertices"):
            fix_orientation(mesh)

    def test_empty_mesh(self, device):
        mesh = Mesh(
            points=torch.rand(3, 3, device=device),
            cells=torch.zeros((0, 3), dtype=torch.long, device=device),
        )
        oriented, stats = fix_orientation(mesh)
        assert oriented is mesh
        assert stats == {
            "n_faces_flipped": 0,
            "n_components": 0,
            "largest_component_size": 0,
            "n_non_orientable_components": 0,
        }

    def test_repair_pipeline_orients_volume_meshes(self, device):
        from physicsnemo.mesh.primitives.volumes import cube_volume

        cube = cube_volume.load(subdivisions=2, device=device)
        repaired, stats = repair_mesh(cube, merge_points=False, fix_orientation=True)

        assert "skipped" not in stats["orientation"]
        signs = _signed_volumes(repaired).sign()
        assert (signs == signs[0]).all()
