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

"""Tests for splitting non-manifold edges and dropping fins."""

import pytest
import torch

from physicsnemo.mesh import Mesh
from physicsnemo.mesh.primitives.surfaces import cube_surface
from physicsnemo.mesh.repair import split_nonmanifold_edges
from physicsnemo.mesh.utilities._duplicate_detection import (
    vectorized_connected_components,
)

# Cube points 1 = (-, -, +) and 7 = (+, +, +) span a diagonal of the top face.
TOP_DIAGONAL = (1, 7)


def add_cells(mesh, new_points, new_cells):
    """The mesh with extra points (numbered after its own) and cells."""
    points = torch.cat([mesh.points, torch.tensor(new_points, dtype=torch.float32)])
    cells = torch.cat([mesh.cells, torch.tensor(new_cells, dtype=torch.int64)])
    return Mesh(points=points, cells=cells)


def assert_closed_and_oriented(cells):
    """Every directed edge appears once, and so does its reverse."""
    directed = torch.cat([cells[:, [0, 1]], cells[:, [1, 2]], cells[:, [2, 0]]])
    edges = set(map(tuple, directed.tolist()))
    assert len(edges) == directed.shape[0]
    assert edges == {(b, a) for a, b in edges}


def enclosed_volume(mesh):
    p = mesh.points[mesh.cells].double()
    return float(torch.linalg.det(p).sum() / 6)


def n_parts(cells):
    """Number of components connected through vertices."""
    pairs = torch.cat([cells[:, [0, 1]], cells[:, [1, 2]]])
    labels = vectorized_connected_components(pairs, int(cells.max()) + 1)
    return torch.unique(labels[cells.reshape(-1)]).numel()


class TestSplitNonmanifoldEdges:
    """Tests for split_nonmanifold_edges."""

    def test_closed_mesh_unchanged(self):
        cube = cube_surface.load()
        result, stats = split_nonmanifold_edges(cube)
        assert stats == {
            "n_edges_split": 0,
            "n_vertices_duplicated": 0,
            "n_cells_dropped": 0,
        }
        assert torch.equal(result.cells, cube.cells)
        assert torch.equal(result.points, cube.points)

    def test_one_triangle_fin_dropped(self):
        cube = cube_surface.load()
        mesh = add_cells(cube, [[0.0, 0.0, 1.0]], [[*TOP_DIAGONAL, 8]])
        result, stats = split_nonmanifold_edges(mesh)
        assert stats["n_cells_dropped"] == 1
        assert torch.equal(result.cells, cube.cells)
        assert enclosed_volume(result) == pytest.approx(1.0)

    def test_two_triangle_fin_dropped(self):
        """A quad fin: each of its triangles has only one open edge."""
        cube = cube_surface.load()
        mesh = add_cells(
            cube,
            [[-0.5, -0.5, 1.0], [0.5, 0.5, 1.0]],
            [[*TOP_DIAGONAL, 9], [TOP_DIAGONAL[0], 9, 8]],
        )
        result, stats = split_nonmanifold_edges(mesh)
        assert stats == {
            "n_edges_split": 0,
            "n_vertices_duplicated": 0,
            "n_cells_dropped": 2,
        }
        assert torch.equal(result.cells, cube.cells)
        assert_closed_and_oriented(result.cells)
        assert enclosed_volume(result) == pytest.approx(1.0)

    @pytest.mark.parametrize("inverted", [False, True])
    @pytest.mark.parametrize("height", [0.0, 1.0], ids=["inside", "outside"])
    @pytest.mark.parametrize("n_fin_cells", [1, 2])
    def test_fin_dropped_on_either_side(self, inverted, height, n_fin_cells):
        """A fin is dropped whichever side of the surface it lies on, and
        whichever way the surface is wound."""
        cube = cube_surface.load()
        if inverted:
            cube = Mesh(points=cube.points, cells=cube.cells[:, [0, 2, 1]])
        a, b = TOP_DIAGONAL
        if n_fin_cells == 1:
            mesh = add_cells(cube, [[0.0, 0.0, height]], [[a, b, 8]])
        else:
            mesh = add_cells(
                cube,
                [[-0.4, -0.4, height], [0.4, 0.4, height]],
                [[a, b, 9], [a, 9, 8]],
            )
        result, stats = split_nonmanifold_edges(mesh)
        assert stats["n_cells_dropped"] == n_fin_cells
        assert stats["n_vertices_duplicated"] == 0
        assert torch.equal(result.cells, cube.cells)
        assert_closed_and_oriented(result.cells)
        assert abs(enclosed_volume(result)) == pytest.approx(1.0)

    @pytest.mark.parametrize("first", [False, True])
    @pytest.mark.parametrize("inverted", [False, True])
    @pytest.mark.parametrize("height", [0.0, 1.0], ids=["inside", "outside"])
    def test_quad_pillow_dropped(self, first, inverted, height):
        """A quad and its reverse on one edge: the two copies pair with each
        other at the quad's diagonal, yet the pillow is dropped."""
        cube = cube_surface.load()
        if inverted:
            cube = Mesh(points=cube.points, cells=cube.cells[:, [0, 2, 1]])
        a, b = TOP_DIAGONAL
        quad = torch.tensor([[a, b, 9], [a, 9, 8]])
        pillow = torch.cat([quad, quad[:, [0, 2, 1]]])
        cells = [pillow, cube.cells] if first else [cube.cells, pillow]
        points = torch.tensor([[-0.4, -0.4, height], [0.4, 0.4, height]])
        mesh = Mesh(points=torch.cat([cube.points, points]), cells=torch.cat(cells))
        result, stats = split_nonmanifold_edges(mesh)
        assert stats["n_cells_dropped"] == 4
        assert torch.equal(result.cells, cube.cells)

    @pytest.mark.parametrize("sheet", ["square", "open box"])
    def test_doubled_open_sheet_kept(self, sheet):
        """A doubled open sheet has no surface without copies to keep, so at
        least one copy of each of its cells stays."""
        if sheet == "square":
            points = torch.tensor([[0.0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
            cells = torch.tensor([[0, 1, 2], [0, 2, 3]])
        else:
            cube = cube_surface.load()
            points = cube.points
            # The cube without its top face (points 1, 3, 5 and 7)
            cells = cube.cells[~(cube.cells % 2 == 1).all(dim=1)]
        result, stats = split_nonmanifold_edges(
            Mesh(points=points, cells=torch.cat([cells, cells]))
        )
        # Copied points map back to the point they copy
        source = (result.points[:, None] == points[None]).all(dim=-1).int().argmax(1)
        kept = source[result.cells].sort(dim=1).values
        assert set(map(tuple, kept.tolist())) == set(
            map(tuple, cells.sort(dim=1).values.tolist())
        )

    @pytest.mark.parametrize("first", [False, True])
    @pytest.mark.parametrize("face", [(0, 1), (2, 3), (10, 11)])
    def test_reversed_copy_of_a_face_dropped(self, face, first):
        """The copy of a two-triangle face goes; the surface's own stays."""
        cube = cube_surface.load()
        copy = cube.cells[list(face)][:, [0, 2, 1]]
        cells = [copy, cube.cells] if first else [cube.cells, copy]
        result, stats = split_nonmanifold_edges(
            Mesh(points=cube.points, cells=torch.cat(cells))
        )
        assert stats == {
            "n_edges_split": 0,
            "n_vertices_duplicated": 0,
            "n_cells_dropped": 2,
        }
        assert torch.equal(result.cells, cube.cells)

    @pytest.mark.parametrize("inverted", [False, True])
    def test_two_cubes_sharing_a_face_stay_closed(self, inverted):
        a = cube_surface.load()
        cells = a.cells[:, [0, 2, 1]] if inverted else a.cells
        # Points 0 to 3 of the second cube (its -X face) are points 4 to 7
        b_cells = cells + 8
        for k in range(4):
            b_cells[b_cells == 8 + k] = 4 + k
        mesh = Mesh(
            points=torch.cat([a.points, a.points + torch.tensor([1.0, 0.0, 0.0])]),
            cells=torch.cat([cells, b_cells]),
        )
        result, _ = split_nonmanifold_edges(mesh)
        assert_closed_and_oriented(result.cells)
        assert abs(enclosed_volume(result)) == pytest.approx(2.0)

    @pytest.mark.parametrize("winding", [[0, 1, 2], [0, 2, 1]])
    def test_duplicate_cell_dropped(self, winding):
        """A copy of a cell, wound either way, pairs nowhere and is dropped."""
        cube = cube_surface.load()
        mesh = Mesh(
            points=cube.points,
            cells=torch.cat([cube.cells, cube.cells[:1, winding]]),
        )
        result, stats = split_nonmanifold_edges(mesh)
        assert stats == {
            "n_edges_split": 0,
            "n_vertices_duplicated": 0,
            "n_cells_dropped": 1,
        }
        assert torch.equal(result.cells, cube.cells)

    @pytest.mark.parametrize("prepend", [False, True])
    def test_zero_area_cell_dropped_in_any_order(self, prepend):
        """A zero-area cell at an edge gives no reference for the angles."""
        cube = cube_surface.load()
        midpoint = cube.points[list(TOP_DIAGONAL)].mean(dim=0, keepdim=True)
        flat = torch.tensor([[*TOP_DIAGONAL, 8]])
        cells = [flat, cube.cells] if prepend else [cube.cells, flat]
        mesh = Mesh(points=torch.cat([cube.points, midpoint]), cells=torch.cat(cells))
        result, stats = split_nonmanifold_edges(mesh)
        assert stats["n_cells_dropped"] == 1
        assert torch.equal(result.cells, cube.cells)

    def test_fin_kept_detached_without_drop(self):
        cube = cube_surface.load()
        mesh = add_cells(cube, [[0.0, 0.0, 1.0]], [[*TOP_DIAGONAL, 8]])
        result, stats = split_nonmanifold_edges(mesh, drop_fins=False)
        assert stats == {
            "n_edges_split": 1,
            "n_vertices_duplicated": 2,
            "n_cells_dropped": 0,
        }
        assert torch.equal(result.cells[:12], cube.cells)
        assert set(result.cells[12].tolist()) == {8, 9, 10}
        assert torch.equal(result.points[[9, 10]], cube.points[list(TOP_DIAGONAL)])

    def test_pillow_dropped(self):
        """Two opposite triangles on one edge enclose no volume."""
        cube = cube_surface.load()
        mesh = add_cells(
            cube,
            [[0.0, 0.0, 1.0]],
            [[*TOP_DIAGONAL, 8], [*TOP_DIAGONAL[::-1], 8]],
        )
        result, stats = split_nonmanifold_edges(mesh)
        assert stats["n_cells_dropped"] == 2
        assert torch.equal(result.cells, cube.cells)

    def test_two_cubes_sharing_an_edge(self):
        """Each cube keeps its own copy of the shared edge."""
        a = cube_surface.load()
        b_points = a.points + torch.tensor([1.0, 1.0, 0.0])
        # Points 0 and 1 of the second cube are points 6 and 7 of the first.
        b_cells = a.cells + 8
        b_cells[b_cells == 8] = 6
        b_cells[b_cells == 9] = 7
        mesh = Mesh(
            points=torch.cat([a.points, b_points]),
            cells=torch.cat([a.cells, b_cells]),
            point_data={"id": torch.arange(16)},
            cell_data={"id": torch.arange(24)},
        )
        result, stats = split_nonmanifold_edges(mesh)
        assert stats == {
            "n_edges_split": 1,
            "n_vertices_duplicated": 2,
            "n_cells_dropped": 0,
        }
        assert_closed_and_oriented(result.cells)
        assert n_parts(result.cells) == 2
        assert enclosed_volume(result) == pytest.approx(2.0)
        assert torch.equal(result.points[16:], result.points[[6, 7]])
        assert result.point_data["id"][16:].tolist() == [6, 7]
        assert torch.equal(result.cell_data["id"], torch.arange(24))

    def test_two_cubes_sharing_an_edge_with_a_fin_first(self):
        """The cubes split apart although the dropped fin comes first."""
        a = cube_surface.load()
        b_cells = a.cells + 8
        b_cells[b_cells == 8] = 6
        b_cells[b_cells == 9] = 7
        points = torch.cat(
            [a.points, a.points + torch.tensor([1.0, 1.0, 0.0]), torch.eye(3)[:1]]
        )
        fin = torch.tensor([[6, 7, 16]])
        mesh = Mesh(points=points, cells=torch.cat([fin, a.cells, b_cells]))
        result, stats = split_nonmanifold_edges(mesh)
        assert stats == {
            "n_edges_split": 1,
            "n_vertices_duplicated": 2,
            "n_cells_dropped": 1,
        }
        assert_closed_and_oriented(result.cells)
        assert n_parts(result.cells) == 2
        assert enclosed_volume(result) == pytest.approx(2.0)

    def test_rejects_planar_mesh(self):
        from physicsnemo.mesh.primitives.basic import two_triangles_2d

        with pytest.raises(ValueError, match="triangle mesh in 3D"):
            split_nonmanifold_edges(two_triangles_2d.load())
