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

"""Comprehensive tests for mesh repair operations."""

import pytest
import torch

from physicsnemo.mesh import Mesh
from physicsnemo.mesh.repair import (
    fill_holes,
    remove_degenerate_cells,
    remove_isolated_points,
    repair_mesh,
    stitch_t_junctions,
)


def _grid_cube(ns: list[int], dtype=torch.float64) -> Mesh:
    """Welded unit cube with outward normals whose face ``k`` is an ``ns[k]`` grid.

    Faces with different ``ns`` meet at T-junction seams.
    """
    faces = [
        ((0, 0, 0), (0, 1, 0), (1, 0, 0)),
        ((0, 0, 1), (1, 0, 0), (0, 1, 0)),
        ((0, 0, 0), (1, 0, 0), (0, 0, 1)),
        ((0, 1, 0), (0, 0, 1), (1, 0, 0)),
        ((0, 0, 0), (0, 0, 1), (0, 1, 0)),
        ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
    ]
    points, cells = [], []
    for (o, u, v), n in zip(faces, ns):
        o, u, v = (torch.tensor(x, dtype=torch.float64) for x in (o, u, v))
        i, j = torch.meshgrid(torch.arange(n + 1), torch.arange(n + 1), indexing="xy")
        grid = o + (i.reshape(-1, 1) * u + j.reshape(-1, 1) * v) / n
        a = (torch.arange(n)[None, :] + (n + 1) * torch.arange(n)[:, None]).reshape(-1)
        b, c, d = a + 1, a + n + 2, a + n + 1
        quads = torch.cat([torch.stack([a, b, c], 1), torch.stack([a, c, d], 1)])
        cells.append(quads + sum(len(p) for p in points))
        points.append(grid)
    unique, inverse = torch.unique(torch.cat(points), dim=0, return_inverse=True)
    return Mesh(points=unique.to(dtype), cells=inverse[torch.cat(cells)])


def _edge_counts(cells: torch.Tensor) -> dict[str, int]:
    """Open, non-manifold, and repeated directed edges of a triangle mesh."""
    directed = torch.cat([cells[:, [0, 1]], cells[:, [1, 2]], cells[:, [2, 0]]])
    _, counts = torch.unique(directed.sort(dim=1).values, dim=0, return_counts=True)
    _, directed_counts = torch.unique(directed, dim=0, return_counts=True)
    return {
        "open": int((counts == 1).sum()),
        "nonmanifold": int((counts > 2).sum()),
        "same_dir": int((directed_counts > 1).sum()),
    }


def _signed_volume(mesh: Mesh) -> float:
    a, b, c = (mesh.points.double()[mesh.cells[:, k]] for k in range(3))
    return float((a * torch.linalg.cross(b, c)).sum() / 6)


@pytest.fixture
def device():
    """Test on CPU."""
    return "cpu"


class TestDuplicateRemoval:
    """Tests for duplicate vertex removal."""

    def test_remove_exact_duplicates(self, device):
        """Test removing exact duplicate vertices."""
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, 1.0],
                [0.0, 0.0],  # Exact duplicate of 0
            ],
            dtype=torch.float32,
            device=device,
        )

        cells = torch.tensor([[0, 1, 2]], dtype=torch.long, device=device)

        mesh = Mesh(points=points, cells=cells)

        mesh_clean = mesh.clean(tolerance=1e-10)

        assert mesh_clean.n_points == 3
        assert mesh_clean.n_cells == 1

    def test_remove_near_duplicates(self, device):
        """Test removing near-duplicate vertices within tolerance."""
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, 1.0],
                [0.0, 1e-7],  # Near duplicate of 0
            ],
            dtype=torch.float32,
            device=device,
        )

        cells = torch.tensor([[0, 1, 2]], dtype=torch.long, device=device)

        mesh = Mesh(points=points, cells=cells)

        mesh_clean = mesh.clean(tolerance=1e-6)

        assert mesh_clean.n_points == 3

    def test_no_duplicates(self, device):
        """Test mesh with no duplicates."""
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

        mesh_clean = mesh.clean()

        assert mesh_clean.n_points == 3
        assert torch.equal(mesh_clean.points, mesh.points)

    def test_multiple_duplicates(self, device):
        """Test removing multiple duplicate vertex groups."""
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, 1.0],
                [0.0, 0.0],  # Dup of 0
                [1.0, 0.0],  # Dup of 1
            ],
            dtype=torch.float32,
            device=device,
        )

        cells = torch.tensor([[0, 1, 2]], dtype=torch.long, device=device)

        mesh = Mesh(points=points, cells=cells)

        mesh_clean = mesh.clean()

        assert mesh_clean.n_points == 3

    def test_preserves_cell_connectivity(self, device):
        """Test that cell connectivity is correctly remapped."""
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, 1.0],
                [0.0, 0.0],  # Dup of 0
            ],
            dtype=torch.float32,
            device=device,
        )

        # Cell references duplicate
        cells = torch.tensor([[1, 2, 3]], dtype=torch.long, device=device)

        mesh = Mesh(points=points, cells=cells)

        mesh_clean = mesh.clean()

        # Verify cell still forms valid triangle
        assert mesh_clean.n_cells == 1

        # Should form a triangle
        area = mesh_clean.cell_areas[0]
        assert area > 0


class TestDegenerateRemoval:
    """Tests for degenerate cell removal."""

    def test_remove_zero_area_cells(self, device):
        """Test removing cells with zero area."""
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, 1.0],
                [2.0, 0.0],  # Collinear with 1, makes degenerate triangle
            ],
            dtype=torch.float32,
            device=device,
        )

        cells = torch.tensor(
            [
                [0, 1, 2],  # Good triangle
                [1, 3, 1],  # Degenerate (duplicate vertex)
            ],
            dtype=torch.long,
            device=device,
        )

        mesh = Mesh(points=points, cells=cells)

        mesh_clean, stats = remove_degenerate_cells(mesh)

        assert stats["n_duplicate_vertex_cells"] == 1
        assert mesh_clean.n_cells == 1

    def test_no_degenerates(self, device):
        """Test mesh with no degenerate cells."""
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

        mesh_clean, stats = remove_degenerate_cells(mesh)

        assert stats["n_zero_area_cells"] == 0
        assert stats["n_duplicate_vertex_cells"] == 0
        assert mesh_clean.n_cells == 1


class TestIsolatedRemoval:
    """Tests for isolated vertex removal."""

    def test_remove_single_isolated(self, device):
        """Test removing single isolated vertex."""
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, 1.0],
                [5.0, 5.0],  # Isolated
            ],
            dtype=torch.float32,
            device=device,
        )

        cells = torch.tensor([[0, 1, 2]], dtype=torch.long, device=device)

        mesh = Mesh(points=points, cells=cells)

        mesh_clean, stats = remove_isolated_points(mesh)

        assert stats["n_isolated_removed"] == 1
        assert mesh_clean.n_points == 3
        assert mesh_clean.n_cells == 1

    def test_remove_multiple_isolated(self, device):
        """Test removing multiple isolated vertices."""
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, 1.0],
                [5.0, 5.0],  # Isolated
                [6.0, 6.0],  # Isolated
                [7.0, 7.0],  # Isolated
            ],
            dtype=torch.float32,
            device=device,
        )

        cells = torch.tensor([[0, 1, 2]], dtype=torch.long, device=device)

        mesh = Mesh(points=points, cells=cells)

        mesh_clean, stats = remove_isolated_points(mesh)

        assert stats["n_isolated_removed"] == 3
        assert mesh_clean.n_points == 3

    def test_no_isolated(self, device):
        """Test mesh with no isolated vertices."""
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

        mesh_clean, stats = remove_isolated_points(mesh)

        assert stats["n_isolated_removed"] == 0
        assert mesh_clean.n_points == 3


class TestRepairPipeline:
    """Tests for comprehensive repair pipeline."""

    def test_pipeline_all_operations(self, device):
        """Test full pipeline with all problems."""
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, 1.0],
                [0.0, 0.0],  # Duplicate
                [5.0, 5.0],  # Isolated
            ],
            dtype=torch.float32,
            device=device,
        )

        cells = torch.tensor(
            [
                [0, 1, 2],  # Good
                [1, 1, 2],  # Degenerate
            ],
            dtype=torch.long,
            device=device,
        )

        mesh = Mesh(points=points, cells=cells)

        mesh_clean, all_stats = repair_mesh(
            mesh,
        )

        # Should have fixed all problems
        assert mesh_clean.n_points == 3
        assert mesh_clean.n_cells == 1

        # Verify individual stats
        assert "degenerates" in all_stats
        assert "merge_points" in all_stats
        assert "isolated" in all_stats

        assert all_stats["degenerates"]["n_cells_original"] == 2
        assert all_stats["degenerates"]["n_cells_final"] == 1

    def test_pipeline_clean_mesh_unchanged(self, device):
        """Test that clean mesh is unchanged by pipeline."""
        from physicsnemo.mesh.primitives.procedural import lumpy_sphere

        # Use lumpy_sphere - a complex, watertight mesh that should be clean
        mesh = lumpy_sphere.load(subdivisions=2, device=device)
        original_n_points = mesh.n_points
        original_n_cells = mesh.n_cells

        mesh_clean, stats = repair_mesh(mesh)

        # Should be unchanged
        assert mesh_clean.n_points == original_n_points
        assert mesh_clean.n_cells == original_n_cells
        assert stats["degenerates"]["n_zero_area_cells"] == 0
        assert stats["merge_points"]["n_duplicates_merged"] == 0
        assert stats["isolated"]["n_isolated_removed"] == 0

    def test_pipeline_preserves_data(self, device):
        """Test that repair preserves point and cell data."""
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, 1.0],
                [5.0, 5.0],  # Isolated
            ],
            dtype=torch.float32,
            device=device,
        )

        cells = torch.tensor([[0, 1, 2]], dtype=torch.long, device=device)

        mesh = Mesh(points=points, cells=cells)
        mesh.point_data["temperature"] = torch.tensor(
            [1.0, 2.0, 3.0, 999.0], device=device
        )
        mesh.cell_data["pressure"] = torch.tensor([100.0], device=device)

        mesh_clean, stats = repair_mesh(mesh, remove_isolated=True)

        # Data should be preserved for remaining points/cells
        assert "temperature" in mesh_clean.point_data
        assert "pressure" in mesh_clean.cell_data
        assert mesh_clean.point_data["temperature"].shape == (3,)
        assert mesh_clean.cell_data["pressure"].shape == (1,)

    def test_pipeline_removes_cells_collapsed_by_merge(self, device):
        """Cells that the point merge collapses are removed too."""
        from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral

        sphere = sphere_icosahedral.load(subdivisions=1, device=device)
        # Split the edge (a, b) of cell 0 at a point 3e-7 from a, below the merge
        # tolerance, and split both triangles on that edge there.
        a, b = sphere.cells[0, :2].tolist()
        on_edge = ((sphere.cells == a) | (sphere.cells == b)).sum(dim=1) == 2
        split = sphere.cells[on_edge]
        direction = sphere.points[b] - sphere.points[a]
        new_point = sphere.points[a] + 3e-7 * direction / direction.norm()
        n = sphere.n_points
        mesh = Mesh(
            points=torch.cat([sphere.points, new_point[None]]),
            cells=torch.cat(
                [
                    sphere.cells[~on_edge],
                    torch.where(split == b, n, split),
                    torch.where(split == a, n, split),
                ]
            ),
        )
        assert mesh.validate()["valid"] and mesh.is_watertight()

        repaired, stats = repair_mesh(mesh)

        # The merge folds the new point into a, giving back the sphere
        assert repaired.n_points == sphere.n_points
        assert repaired.n_cells == sphere.n_cells
        assert stats["degenerates"]["n_duplicate_vertex_cells"] == 2
        assert repaired.validate()["valid"]
        assert repaired.is_watertight()

        # A second repair changes nothing
        _, stats2 = repair_mesh(repaired)
        assert stats2["degenerates"]["n_cells_final"] == repaired.n_cells
        assert stats2["merge_points"]["n_duplicates_merged"] == 0

    @staticmethod
    def _seamed_cube(scale: float) -> Mesh:
        """Cube with T-junction seams, an L-shaped hole and a one-triangle fin.

        The top face is a 4 x 4 grid against 2 x 2 grids elsewhere; squares
        5, 6 and 9 of the top face are cut out; the fin stands outward on the
        diagonal of the first square of the face x = 1.
        """
        cube = _grid_cube([2, 4, 2, 2, 2, 2])
        hole = torch.tensor([5, 6, 9])
        keep = torch.ones(cube.n_cells, dtype=torch.bool)
        keep[torch.cat([8 + hole, 24 + hole])] = False
        a, _, c = cube.cells[64].tolist()
        apex = torch.tensor([[1.5, 0.25, 0.25]], dtype=torch.float64)
        return Mesh(
            points=torch.cat([cube.points, apex]) * scale,
            cells=torch.cat([cube.cells[keep], torch.tensor([[a, c, cube.n_points]])]),
        )

    @pytest.mark.parametrize("scale", [1e-3, 1.0, 1e3])
    def test_pipeline_closes_seams_holes_and_fins(self, scale):
        """A cube with T-junction seams, a hole, and a fin comes out closed,
        manifold, consistently wound and of volume ``scale**3``, with the same
        cells at any scale."""
        kwargs = dict(
            fix_orientation=True, fill_holes=True, split_nonmanifold_edges=True
        )
        repaired, stats = repair_mesh(self._seamed_cube(scale), **kwargs)
        assert _edge_counts(repaired.cells) == {
            "open": 0,
            "nonmanifold": 0,
            "same_dir": 0,
        }
        assert abs(_signed_volume(repaired)) == pytest.approx(scale**3)
        assert stats["nonmanifold_edges"]["n_cells_dropped"] == 1
        assert stats["holes"]["n_holes_filled"] == 1
        reference, _ = repair_mesh(self._seamed_cube(1.0), **kwargs)
        assert torch.equal(repaired.cells, reference.cells)

    @pytest.mark.parametrize("scale", [1e-3, 1.0, 1e3])
    def test_pipeline_removes_duplicates_made_by_merge(self, device, scale):
        """Two copies of a cube 1e-8 of its size apart give one clean cube, at
        any scale: the merge tolerance is relative to the size."""
        from physicsnemo.mesh.primitives.surfaces import cube_surface

        cube = cube_surface.load(device=device)
        points = cube.points.double() * scale
        mesh = Mesh(
            points=torch.cat([points, points + 1e-8 * scale]),
            cells=torch.cat([cube.cells, cube.cells + cube.n_points]),
        )
        repaired, stats = repair_mesh(mesh)
        assert torch.equal(repaired.cells, cube.cells)
        assert _edge_counts(repaired.cells)["nonmanifold"] == 0
        assert stats["duplicates"]["n_cells_final"] == 12

    def test_pipeline_closes_slit_of_removed_zero_area_cell(self, device):
        """Removing a zero-area cell opens a slit; stitching closes it."""
        from physicsnemo.mesh.primitives.surfaces import cube_surface

        cube = cube_surface.load(device=device)
        # Split the top face's triangle (1, 5, 7) at the midpoint m of its
        # diagonal (1, 7), and fill the gap to the other triangle (1, 7, 3)
        # with the zero-area triangle (1, m, 7).
        points = cube.points.double()
        m = cube.n_points
        cells = cube.cells[(cube.cells != torch.tensor([1, 5, 7])).any(dim=1)]
        mesh = Mesh(
            points=torch.cat([points, points[[1, 7]].mean(dim=0, keepdim=True)]),
            cells=torch.cat([cells, torch.tensor([[1, 5, m], [m, 5, 7], [1, m, 7]])]),
        )
        assert _edge_counts(mesh.cells)["open"] == 0
        repaired, stats = repair_mesh(mesh)
        assert _edge_counts(repaired.cells) == {
            "open": 0,
            "nonmanifold": 0,
            "same_dir": 0,
        }
        assert _signed_volume(repaired) == pytest.approx(1.0)
        assert stats["degenerates"]["n_zero_area_cells"] == 1
        assert stats["t_junctions"]["n_cells_split"] == 1

    def test_pipeline_clean_surface_unchanged(self, device):
        from physicsnemo.mesh.primitives.procedural import lumpy_sphere

        mesh = lumpy_sphere.load(subdivisions=2, device=device)
        repaired, stats = repair_mesh(mesh, split_nonmanifold_edges=True)
        assert torch.equal(repaired.points, mesh.points)
        assert torch.equal(repaired.cells, mesh.cells)
        assert stats["t_junctions"]["n_cells_split"] == 0
        assert stats["nonmanifold_edges"]["n_cells_dropped"] == 0

    def test_pipeline_keeps_plate_attached_along_an_edge(self, device):
        """By default nothing real is deleted: a zero-thickness plate on an
        edge of a cube stays, and goes only with split_nonmanifold_edges."""
        from physicsnemo.mesh.primitives.surfaces import cube_surface

        cube = cube_surface.load(device=device)
        # A plate on the cube's edge (0, 1), at x = y = -0.5
        plate_points = torch.tensor([[-1.0, -1.0, -0.5], [-1.0, -1.0, 0.5]])
        mesh = Mesh(
            points=torch.cat([cube.points, plate_points]),
            cells=torch.cat([cube.cells, torch.tensor([[0, 1, 9], [0, 9, 8]])]),
        )
        repaired, _ = repair_mesh(mesh)
        assert repaired.n_cells == 14
        dropped, _ = repair_mesh(mesh, split_nonmanifold_edges=True)
        assert dropped.n_cells == 12

    def test_pipeline_fills_slit_too_wide_to_stitch(self, device):
        """An explicit tolerance reaches hole filling too: a gap too wide for
        it to stitch is not a slit for fill_holes either, so it is filled."""
        from physicsnemo.mesh.primitives.surfaces import cube_surface

        cube = cube_surface.load(device=device)
        points = cube.points.double()
        # Split the top face's triangle (1, 5, 7) at a point m 1e-7 off its
        # diagonal (1, 7), towards 5, leaving a thin gap (1, m, 7)
        m = cube.n_points
        offset = torch.tensor([[1.0, -1.0, 0.0]], dtype=torch.float64) * 1e-7 / 2**0.5
        cells = cube.cells[(cube.cells != torch.tensor([1, 5, 7])).any(dim=1)]
        mesh = Mesh(
            points=torch.cat(
                [points, points[[1, 7]].mean(dim=0, keepdim=True) + offset]
            ),
            cells=torch.cat([cells, torch.tensor([[1, 5, m], [m, 5, 7]])]),
        )
        repaired, stats = repair_mesh(mesh, tolerance=1e-9, fill_holes=True)
        assert stats["t_junctions"]["n_cells_split"] == 0
        assert stats["holes"]["n_holes_filled"] == 1
        assert _edge_counts(repaired.cells)["open"] == 0

    def test_pipeline_merges_coincident_points(self, device):
        """Points that all coincide still merge: the relative tolerance
        has a floor."""
        mesh = Mesh(
            points=torch.ones(3, 3, device=device),
            cells=torch.tensor([[0, 1, 2]], device=device),
        )
        repaired, stats = repair_mesh(mesh)
        assert stats["merge_points"]["n_duplicates_merged"] == 2
        assert repaired.n_cells == 0


class TestHoleFilling:
    """Tests for hole filling."""

    def test_fill_simple_hole(self, device):
        """Test filling a simple boundary loop."""
        # Create mesh with hole (triangle with one missing face)
        points = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 1.0, 0.0],
                [1.5, 0.5, 0.0],
            ],
            dtype=torch.float32,
            device=device,
        )

        # Only one triangle - leaves edges as boundaries
        cells = torch.tensor([[0, 1, 2]], dtype=torch.long, device=device)

        mesh = Mesh(points=points, cells=cells)

        mesh_filled, stats = fill_holes(mesh)

        # Should add faces
        assert stats["n_holes_detected"] >= 1
        assert (
            mesh_filled.n_cells > mesh.n_cells or mesh_filled.n_points > mesh.n_points
        )

    def test_closed_mesh_no_holes(self, device):
        """Test that closed mesh is unchanged."""
        from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral

        # Use sphere_icosahedral - a complex watertight closed surface
        mesh = sphere_icosahedral.load(subdivisions=1, device=device)

        mesh_filled, stats = fill_holes(mesh)

        # Should find no holes
        assert stats["n_holes_filled"] == 0

    def test_fill_holes_extends_point_and_cell_data(self, device):
        """New centroid points and faces get NaN in float fields, 0 in others."""
        points = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            device=device,
        )
        cells = torch.tensor([[0, 1, 2], [0, 3, 1]], dtype=torch.long, device=device)
        mesh = Mesh(
            points=points,
            cells=cells,
            point_data={
                "T": torch.tensor([0.0, 1.0, 2.0, 3.0], device=device),
                "flow": {"v": torch.ones(4, 2, device=device)},
            },
            cell_data={"id": torch.tensor([7, 8], device=device)},
        )

        # Two faces of a tetrahedron bound one skew 4-edge hole: 1 centroid, 4 faces
        mesh_filled, stats = fill_holes(mesh)

        assert stats["n_points_added"] == 1 and stats["n_faces_added"] == 4
        nan = float("nan")
        torch.testing.assert_close(
            mesh_filled.point_data["T"],
            torch.tensor([0.0, 1.0, 2.0, 3.0, nan], device=device),
            equal_nan=True,
        )
        assert mesh_filled.point_data["flow", "v"].shape == (5, 2)
        assert mesh_filled.point_data["flow", "v"][4].isnan().all()
        assert torch.equal(
            mesh_filled.cell_data["id"], torch.tensor([7, 8, 0, 0, 0, 0], device=device)
        )

    def test_fill_wound_like_surface(self, device):
        """The fill follows the surface's winding, not the vertex labels."""
        cube = _grid_cube([2] * 6)
        top = (cube.points[cube.cells][:, :, 2] == 1).all(dim=1)
        generator = torch.Generator().manual_seed(0)
        for _ in range(20):
            perm = torch.randperm(cube.n_points, generator=generator)
            mesh = Mesh(
                points=cube.points[perm].to(device),
                cells=torch.argsort(perm)[cube.cells[~top]].to(device),
            )
            filled, stats = fill_holes(mesh, max_hole_edges=64)
            assert stats["n_holes_filled"] == 1
            assert _edge_counts(filled.cells) == {
                "open": 0,
                "nonmanifold": 0,
                "same_dir": 0,
            }
            assert _signed_volume(filled) == pytest.approx(1.0)

    def test_fan_fill_wound_like_surface(self, device):
        """A centroid fan over a non-planar hole follows the surface's winding."""
        cube = _grid_cube([2] * 6)
        corner = int((cube.points == 1).all(dim=1).nonzero())
        cut = (cube.cells == corner).any(dim=1)  # the hole bends around the corner
        generator = torch.Generator().manual_seed(0)
        for _ in range(20):
            perm = torch.randperm(cube.n_points, generator=generator)
            mesh = Mesh(
                points=cube.points[perm].to(device),
                cells=torch.argsort(perm)[cube.cells[~cut]].to(device),
            )
            filled, stats = fill_holes(mesh, max_hole_edges=64)
            assert stats["n_holes_filled"] == 1 and stats["n_points_added"] == 1
            assert _edge_counts(filled.cells) == {
                "open": 0,
                "nonmanifold": 0,
                "same_dir": 0,
            }
            # The cube less the cone from the fan's centroid over the cut cells
            a, b, c = (cube.points[cube.cells[cut][:, k]] for k in range(3))
            apex = filled.points[-1].cpu()
            cone = torch.linalg.det(torch.stack([a - apex, b - apex, c - apex], 1))
            assert _signed_volume(filled) == pytest.approx(1 - float(cone.sum()) / 6)

    def test_fill_holes_sharing_a_vertex(self, device):
        """Two holes touching at a vertex are both traced and filled."""
        points = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [-1.0, 0.0], [-1.0, -1.0]],
            device=device,
        )
        cells = torch.tensor([[0, 1, 2], [0, 3, 4]], device=device)
        filled, stats = fill_holes(Mesh(points=points, cells=cells))

        assert stats["n_holes_detected"] == 2 and stats["n_holes_filled"] == 2
        assert stats["n_edges_untraced"] == 0
        assert _edge_counts(filled.cells)["open"] == 0

    def test_zero_area_slit_left_open(self, device):
        """A slit (a vertex on the opposite side's edge) is never fanned."""
        mesh = _grid_cube([1] * 6)
        # Split one triangle of face z=1 at the midpoint of its diagonal (0, 0, 1)-(1, 1, 1)
        diagonal = torch.tensor([[0.0, 0.0, 1.0], [1.0, 1.0, 1.0]], dtype=torch.float64)
        b, c = (int((mesh.points == p).all(dim=1).nonzero()) for p in diagonal)
        k = int(
            ((mesh.cells == b).any(dim=1) & (mesh.cells == c).any(dim=1)).nonzero()[0]
        )
        tri = mesh.cells[k].roll(-int((mesh.cells[k] == b).nonzero()))  # (b, x, y)
        m = mesh.n_points
        if tri[1] == c:
            new = [[b, m, tri[2]], [m, c, tri[2]]]
        else:
            new = [[b, tri[1], m], [m, tri[1], c]]
        cells = torch.cat([mesh.cells[:k], mesh.cells[k + 1 :], torch.tensor(new)])
        points = torch.cat([mesh.points, diagonal.mean(dim=0, keepdim=True)])
        mesh = Mesh(points=points.to(device), cells=cells.to(device))

        filled, stats = fill_holes(mesh)

        assert stats["n_faces_added"] == 0 and filled.n_cells == mesh.n_cells
        assert stats["n_holes_detected"] == 1 and stats["n_holes_zero_area"] == 1
        stitched, _ = stitch_t_junctions(mesh)
        assert _edge_counts(stitched.cells) == {
            "open": 0,
            "nonmanifold": 0,
            "same_dir": 0,
        }

    def test_planar_nonconvex_hole_ear_clipped(self, device):
        """An L-shaped planar hole is filled without folds or overlap."""
        hole = [[0, 0], [4, 0], [4, 1], [1, 1], [1, 4], [0, 4]]
        outer = [[-1, -1], [5, -1], [5, 2], [2, 2], [2, 5], [-1, 5]]
        xy = torch.tensor(hole + outer, dtype=torch.float64)
        points = torch.cat([xy, torch.zeros(12, 1, dtype=torch.float64)], dim=1)
        i = torch.arange(6)
        j = (i + 1) % 6
        cells = torch.cat(
            [torch.stack([i, 6 + i, 6 + j], 1), torch.stack([i, 6 + j, j], 1)]
        )
        mesh = Mesh(points=points.to(device), cells=cells.to(device))

        filled, stats = fill_holes(mesh)

        assert stats["n_holes_filled"] == 2 and stats["n_points_added"] == 0
        new = filled.cells[mesh.n_cells :]
        p = filled.points
        z_area = (
            torch.linalg.cross(
                p[new[:, 1]] - p[new[:, 0]], p[new[:, 2]] - p[new[:, 0]]
            )[:, 2]
            / 2
        )
        # The inner L and the ring's outer boundary (wound the other way), each fully covered
        inner = (new < 6).all(dim=1)
        assert (z_area[inner] > 0).all() and float(
            z_area[inner].sum()
        ) == pytest.approx(7.0)
        assert (z_area[~inner] < 0).all() and float(
            z_area[~inner].sum()
        ) == pytest.approx(-27.0)


class TestStitchTJunctions:
    """Tests for T-junction stitching."""

    @pytest.mark.parametrize(
        "ns",
        [
            [1, 2, 1, 1, 1, 1],
            [2, 4, 2, 2, 2, 2],
            [2, 3, 2, 2, 2, 2],
            [1, 2, 3, 4, 5, 6],
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
    def test_mismatched_grids_closed(self, device, ns, dtype):
        """Cubes whose faces have different grids come out closed, manifold, and oriented."""
        mesh = _grid_cube(ns, dtype=dtype)
        if dtype == torch.float32:
            # Move the cube off the origin so that seam vertices round off their edges
            mesh = Mesh(
                points=mesh.points * 3.3 + torch.tensor([0.1, 0.2, 0.3]),
                cells=mesh.cells,
            )
        mesh = Mesh(points=mesh.points.to(device), cells=mesh.cells.to(device))
        assert _edge_counts(mesh.cells)["open"] > 0

        stitched, stats = stitch_t_junctions(mesh)

        assert stats["n_vertices_inserted"] > 0
        assert torch.equal(stitched.points, mesh.points)
        assert _edge_counts(stitched.cells) == {
            "open": 0,
            "nonmanifold": 0,
            "same_dir": 0,
        }
        scale = 1.0 if dtype == torch.float64 else 3.3**3
        assert _signed_volume(stitched) == pytest.approx(scale, rel=1e-5)

    def test_closed_mesh_unchanged(self, device):
        """A closed mesh comes back as is."""
        mesh = _grid_cube([2] * 6)
        stitched, stats = stitch_t_junctions(mesh)
        assert stitched is mesh
        assert stats == {"n_cells_split": 0, "n_vertices_inserted": 0, "n_rounds": 0}

    def test_cell_data_copied_from_parent(self, device):
        """Children of a split triangle copy its cell_data."""
        mesh = _grid_cube([1, 2, 1, 1, 1, 1])
        face = torch.arange(6).repeat_interleave(torch.tensor([2, 8, 2, 2, 2, 2]))
        mesh = Mesh(points=mesh.points, cells=mesh.cells, cell_data={"face": face})

        stitched, _ = stitch_t_junctions(mesh)

        # Each cell's face label matches its normal: the face it lies on
        expected = torch.tensor(
            [[0, 0, -1], [0, 0, 1], [0, -1, 0], [0, 1, 0], [-1, 0, 0], [1, 0, 0]]
        )
        torch.testing.assert_close(
            stitched.cell_normals, expected[stitched.cell_data["face"]].double()
        )
        assert stitched.n_cells > mesh.n_cells


class TestRepairIntegration:
    """Integration tests for repair operations."""

    def test_repair_sequence_order_matters(self, device):
        """Test that repair operations work correctly in sequence."""
        # Create mesh with multiple problems
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, 1.0],
                [0.0, 0.0],  # Duplicate
                [5.0, 5.0],  # Will become isolated after degenerate removal
            ],
            dtype=torch.float32,
            device=device,
        )

        cells = torch.tensor(
            [
                [0, 1, 2],  # Good triangle
                [3, 4, 4],  # Degenerate (duplicate vertex)
            ],
            dtype=torch.long,
            device=device,
        )

        mesh = Mesh(points=points, cells=cells)

        # Apply repairs in correct order
        mesh1, _ = remove_degenerate_cells(mesh)
        assert mesh1.n_cells == 1  # Removed degenerate

        mesh2 = mesh1.clean(remove_unused_points=False)
        assert mesh2.n_points == 4  # Merged duplicates

        mesh3, _ = remove_isolated_points(mesh2)
        assert mesh3.n_points == 3  # Removed isolated

        # Final mesh should be clean
        validation = mesh3.validate()
        assert validation["valid"]

    def test_idempotence(self, device):
        """Test that applying repair twice doesn't change result."""
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, 1.0],
                [0.0, 0.0],  # Duplicate
            ],
            dtype=torch.float32,
            device=device,
        )

        cells = torch.tensor([[0, 1, 2]], dtype=torch.long, device=device)

        mesh = Mesh(points=points, cells=cells)

        # Apply twice
        mesh1, stats1 = repair_mesh(mesh)
        mesh2, stats2 = repair_mesh(mesh1)

        # Second application should find no problems
        assert stats2["merge_points"]["n_duplicates_merged"] == 0
        assert stats2["degenerates"]["n_zero_area_cells"] == 0
        assert stats2["isolated"]["n_isolated_removed"] == 0

        # Meshes should be identical
        assert mesh1.n_points == mesh2.n_points
        assert mesh1.n_cells == mesh2.n_cells


def test_fix_orientation_component_size_not_overcounted():
    """Regression: the orientation BFS front must be deduplicated. A child face
    reached from two parents in one level was counted twice, making
    largest_component_size exceed n_cells on any mesh with cycles (every closed
    surface). Here a single closed connected surface must yield exactly one
    component covering all cells.
    """
    from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral
    from physicsnemo.mesh.repair.orientation import fix_orientation

    mesh = sphere_icosahedral.load(subdivisions=2)  # closed, connected, has cycles
    oriented, stats = fix_orientation(mesh)

    assert oriented.n_cells == mesh.n_cells
    assert stats["n_components"] == 1
    assert stats["largest_component_size"] == mesh.n_cells


def test_fix_orientation_preserves_data_and_invalidates_caches():
    """Rewinding indexed cells preserves fields without retaining stale caches."""
    from physicsnemo.mesh.repair.orientation import fix_orientation

    mesh = Mesh(
        points=torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        ),
        cells=torch.tensor(
            [
                [0, 1, 2],
                [0, 3, 2],  # Opposite winding from the first triangle.
            ]
        ),
        point_data={"point_id": torch.arange(4)},
        cell_data={"cell_id": torch.arange(2)},
        global_data={"case": torch.tensor(1)},
    )
    mesh._cache["point", "sentinel"] = torch.ones(mesh.n_points)
    mesh._cache["topology", "sentinel"] = torch.tensor(1)

    oriented, stats = fix_orientation(mesh)

    assert stats["n_faces_flipped"] == 1
    torch.testing.assert_close(oriented.point_data["point_id"], torch.arange(4))
    torch.testing.assert_close(oriented.cell_data["cell_id"], torch.arange(2))
    torch.testing.assert_close(oriented.global_data["case"], torch.tensor(1))
    assert (
        oriented.point_data["point_id"].data_ptr()
        != mesh.point_data["point_id"].data_ptr()
    )
    assert (
        oriented.cell_data["cell_id"].data_ptr() != mesh.cell_data["cell_id"].data_ptr()
    )
    assert not oriented._cache["cell"].keys()
    assert not oriented._cache["point"].keys()
    assert not oriented._cache["topology"].keys()
    torch.testing.assert_close(
        oriented.cell_normals[:, 2],
        torch.ones(oriented.n_cells),
    )
