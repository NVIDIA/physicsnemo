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

"""Focused safety tests for linear generic VTK cell conversion."""

import inspect
import subprocess
import sys
import textwrap
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch

pv = pytest.importorskip("pyvista")
vtk = pytest.importorskip("vtk")

import physicsnemo.mesh.io.io_pyvista as io_pyvista  # noqa: E402
from physicsnemo.mesh.io.io_pyvista import from_pyvista  # noqa: E402


def _next_line_number() -> int:
    """Return the next source line in the caller for warning assertions."""
    frame = inspect.currentframe()
    if frame is None:
        raise RuntimeError("No current Python frame is available.")
    caller = frame.f_back
    if caller is None:
        raise RuntimeError("No caller Python frame is available.")
    return caller.f_lineno + 1


def _run_isolated_script(script: str) -> None:
    """Run a crash-sensitive validation script in one child process."""
    result = subprocess.run(  # noqa: S603 - interpreter/script are test constants
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _make_line_triangle_grid() -> "pv.UnstructuredGrid":
    """Build disjoint line and triangle cells with source-level data."""
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
        ]
    )
    grid = pv.UnstructuredGrid(
        np.array([2, 0, 1, 3, 2, 3, 4]),
        np.array([pv.CellType.LINE, pv.CellType.TRIANGLE]),
        points,
    )
    grid.point_data["point_id"] = np.arange(5, dtype=np.int32) + 100
    grid.point_data["vtkOriginalPointIds"] = np.arange(5, dtype=np.int64) + 1000
    grid.cell_data["kind"] = np.array([10, 20], dtype=np.int16)
    grid.field_data["case"] = np.array([7], dtype=np.int32)
    return grid


def _make_triangle_tetra_grid() -> "pv.UnstructuredGrid":
    """Build disjoint triangle and tetrahedron cells with parent labels."""
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [2.0, 0.0, 1.0],
        ]
    )
    grid = pv.UnstructuredGrid(
        np.array([3, 0, 1, 2, 4, 3, 4, 5, 6]),
        np.array([pv.CellType.TRIANGLE, pv.CellType.TETRA]),
        points,
    )
    grid.point_data["point_id"] = np.arange(7, dtype=np.int64)
    grid.cell_data["kind"] = np.array([20, 30], dtype=np.int32)
    return grid


def _make_two_quad_grid() -> "pv.UnstructuredGrid":
    """Build two disjoint quads for parent-provenance tests."""
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [3.0, 1.0, 0.0],
            [2.0, 1.0, 0.0],
        ]
    )
    return pv.UnstructuredGrid(
        np.array([4, 0, 1, 2, 3, 4, 4, 5, 6, 7]),
        np.array([pv.CellType.QUAD, pv.CellType.QUAD]),
        points,
    )


def _make_concave_l_prism() -> "pv.UnstructuredGrid":
    """Build a valid concave polyhedron that VTK cannot tetrahedralize exactly."""
    base_xy = np.array(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 1.0],
            [1.0, 1.0],
            [1.0, 2.0],
            [0.0, 2.0],
        ]
    )
    points = np.vstack(
        [
            np.column_stack([base_xy, np.zeros(6)]),
            np.column_stack([base_xy, np.ones(6)]),
        ]
    )
    base_triangles = [[0, 1, 3], [1, 2, 3], [0, 3, 5], [3, 4, 5]]
    faces = [
        *[triangle[::-1] for triangle in base_triangles],
        *[[point_id + 6 for point_id in triangle] for triangle in base_triangles],
        *[
            [index, (index + 1) % 6, (index + 1) % 6 + 6, index + 6]
            for index in range(6)
        ],
    ]
    polyhedron = [len(faces)]
    for face in faces:
        polyhedron.extend([len(face), *face])
    return pv.UnstructuredGrid(
        np.array([len(polyhedron), *polyhedron]),
        np.array([pv.CellType.POLYHEDRON]),
        points,
    )


def _make_bulk_polyhedra(
    n_polyhedra: int,
    malformed_parent_id: int | None = None,
) -> "pv.UnstructuredGrid":
    """Build alternating tetrahedral and pyramidal polyhedron parents."""
    tetra_points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    tetra_faces = [[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]]
    pyramid_points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.5, 1.0],
        ]
    )
    pyramid_faces = [
        [0, 3, 2, 1],
        [0, 1, 4],
        [1, 2, 4],
        [2, 3, 4],
        [3, 0, 4],
    ]

    points = []
    cells = []
    point_offset = 0
    for parent_id in range(n_polyhedra):
        base_points, base_faces = (
            (tetra_points, tetra_faces)
            if parent_id % 2 == 0
            else (pyramid_points, pyramid_faces)
        )
        points.append(base_points + np.array([2.0 * parent_id, 0.0, 0.0]))
        faces = [list(face) for face in base_faces]
        if parent_id == malformed_parent_id:
            faces.append(list(faces[0]))
        face_stream = [len(faces)]
        for face in faces:
            face_stream.extend(
                [len(face), *(point_offset + point_id for point_id in face)]
            )
        cells.extend([len(face_stream), *face_stream])
        point_offset += len(base_points)

    return pv.UnstructuredGrid(
        np.asarray(cells),
        np.full(n_polyhedra, pv.CellType.POLYHEDRON),
        np.vstack(points),
    )


def _vtk_parametric_points(cell_type: "pv.CellType") -> np.ndarray:
    """Return VTK's canonical nodes for a fixed-size cell."""
    generic_cell = vtk.vtkGenericCell()
    generic_cell.SetCellType(int(cell_type))
    representative = generic_cell.GetRepresentativeCell()
    n_points = representative.GetNumberOfPoints()
    return np.asarray(representative.GetParametricCoords()).reshape(n_points, 3).copy()


def _make_unsupported_grid(cell_type: "pv.CellType") -> "pv.UnstructuredGrid":
    """Build a representative unsupported topology with valid point IDs."""
    arities = {
        pv.CellType.QUADRATIC_EDGE: 3,
        pv.CellType.QUADRATIC_TETRA: 10,
        pv.CellType.LAGRANGE_QUADRILATERAL: 9,
        pv.CellType.BEZIER_CURVE: 3,
        pv.CellType.CONVEX_POINT_SET: 8,
    }
    n_points = arities[cell_type]
    points = np.arange(3 * n_points, dtype=float).reshape(n_points, 3)
    grid = pv.UnstructuredGrid(
        np.concatenate(([n_points], np.arange(n_points))),
        np.array([cell_type]),
        points,
    )
    grid.point_data["point_id"] = np.arange(n_points, dtype=np.int32)
    grid.cell_data["kind"] = np.array([9], dtype=np.int16)
    return grid


@pytest.mark.parametrize(
    "cell_type,points,expected_dim,expected_measure",
    [
        (
            pv.CellType.PIXEL,
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                    [0.0, 3.0, 0.0],
                    [2.0, 3.0, 0.0],
                ]
            ),
            2,
            6.0,
        ),
        (
            pv.CellType.TRIANGLE_STRIP,
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [1.0, 1.0, 0.0],
                ]
            ),
            2,
            1.0,
        ),
        (
            pv.CellType.PENTAGONAL_PRISM,
            _vtk_parametric_points(pv.CellType.PENTAGONAL_PRISM),
            3,
            None,
        ),
        (
            pv.CellType.HEXAGONAL_PRISM,
            _vtk_parametric_points(pv.CellType.HEXAGONAL_PRISM),
            3,
            None,
        ),
        (pv.CellType.EMPTY_CELL, np.empty((0, 3)), 0, None),
        (pv.CellType.VERTEX, np.array([[0.0, 0.0, 0.0]]), 0, None),
        (
            pv.CellType.POLY_VERTEX,
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            0,
            None,
        ),
        (
            pv.CellType.POLYGON,
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0],
                ]
            ),
            2,
            1.0,
        ),
        (
            pv.CellType.VOXEL,
            _vtk_parametric_points(pv.CellType.VOXEL),
            3,
            1.0,
        ),
        (
            pv.CellType.WEDGE,
            _vtk_parametric_points(pv.CellType.WEDGE),
            3,
            0.5,
        ),
        (
            pv.CellType.PYRAMID,
            _vtk_parametric_points(pv.CellType.PYRAMID),
            3,
            1.0 / 3.0,
        ),
    ],
    ids=[
        "pixel",
        "triangle-strip",
        "pentagonal-prism",
        "hexagonal-prism",
        "empty",
        "vertex",
        "poly-vertex",
        "polygon",
        "voxel",
        "wedge",
        "pyramid",
    ],
)
def test_allowlisted_linear_cells(cell_type, points, expected_dim, expected_measure):
    """Every allowlisted family converts with valid geometry and parent data."""
    grid = pv.UnstructuredGrid(
        np.concatenate(([len(points)], np.arange(len(points)))),
        np.array([cell_type]),
        points,
    )
    if expected_dim > 0:
        grid.cell_data["parent"] = np.array([5], dtype=np.int16)

    mesh = from_pyvista(grid, warn_on_lost_data=False)

    assert mesh.n_manifold_dims == expected_dim
    if expected_dim == 0:
        assert mesh.n_cells == 0
        return

    assert mesh.n_cells > 0
    assert torch.equal(
        mesh.cell_data["parent"],
        torch.full((mesh.n_cells,), 5, dtype=torch.int16),
    )
    assert int(mesh.cells.min()) >= 0
    assert int(mesh.cells.max()) < mesh.n_points
    assert bool((mesh.cell_areas > 0).all())
    if expected_measure is None:
        n_base_points = len(points) // 2
        base = points[:n_base_points, :2]
        base_area = 0.5 * abs(
            np.dot(base[:, 0], np.roll(base[:, 1], -1))
            - np.dot(base[:, 1], np.roll(base[:, 0], -1))
        )
        height = abs(
            points[n_base_points:, 2].mean() - points[:n_base_points, 2].mean()
        )
        expected_measure = base_area * height
    assert mesh.cell_areas.sum().item() == pytest.approx(expected_measure)


def test_adjacent_pixels_have_conforming_shared_edge():
    """Adjacent pixels produce matching triangles on their shared edge."""
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
        ]
    )
    grid = pv.UnstructuredGrid(
        np.array([4, 0, 1, 2, 3, 4, 1, 4, 3, 5]),
        np.array([pv.CellType.PIXEL, pv.CellType.PIXEL]),
        points,
    )

    mesh = from_pyvista(grid)

    assert mesh.n_cells == 4
    assert mesh.cell_areas.sum().item() == pytest.approx(2.0)
    edges = (
        torch.cat([mesh.cells[:, [0, 1]], mesh.cells[:, [1, 2]], mesh.cells[:, [2, 0]]])
        .sort(dim=1)
        .values
    )
    assert int(((edges == torch.tensor([1, 3])).all(dim=1)).sum()) == 2


def test_stacked_hexagonal_prisms_have_conforming_shared_face():
    """Stacked hexagonal prisms use identical shared-face triangulation."""
    representative = _vtk_parametric_points(pv.CellType.HEXAGONAL_PRISM)
    base_xy = representative[:6, :2]
    points = np.vstack(
        [np.column_stack([base_xy, np.full(6, z)]) for z in (0.0, 1.0, 2.0)]
    )
    grid = pv.UnstructuredGrid(
        np.concatenate(
            [
                [12],
                np.arange(12),
                [12],
                np.arange(6, 18),
            ]
        ),
        np.array([pv.CellType.HEXAGONAL_PRISM] * 2),
        points,
    )

    mesh = from_pyvista(grid)

    tetrahedra = mesh.cells
    facets = (
        torch.cat(
            [
                tetrahedra[:, [1, 2, 3]],
                tetrahedra[:, [0, 3, 2]],
                tetrahedra[:, [0, 1, 3]],
                tetrahedra[:, [0, 2, 1]],
            ]
        )
        .sort(dim=1)
        .values
    )
    middle_facets = facets[((facets >= 6) & (facets < 12)).all(dim=1)]
    _, counts = torch.unique(middle_facets, dim=0, return_counts=True)
    assert len(counts) > 0
    assert bool((counts == 2).all())


@pytest.mark.parametrize(
    "target_dim,expected_cells,expected_kind",
    [(1, [[0, 1]], [10]), (2, [[2, 3, 4]], [20])],
)
def test_mixed_linear_selection_preserves_source_contracts(
    target_dim, expected_cells, expected_kind
):
    """Explicit selection remaps IDs and keeps full point/global data."""
    grid = _make_line_triangle_grid()

    mesh = from_pyvista(
        grid,
        manifold_dim=target_dim,
        warn_on_lost_data=False,
    )

    assert torch.equal(mesh.points, torch.from_numpy(grid.points))
    assert torch.equal(mesh.cells, torch.tensor(expected_cells))
    assert torch.equal(
        mesh.point_data["point_id"], torch.from_numpy(grid.point_data["point_id"])
    )
    assert torch.equal(
        mesh.point_data["vtkOriginalPointIds"],
        torch.from_numpy(grid.point_data["vtkOriginalPointIds"]),
    )
    assert torch.equal(
        mesh.cell_data["kind"], torch.tensor(expected_kind, dtype=torch.int16)
    )
    assert torch.equal(mesh.global_data["case"], torch.tensor([7], dtype=torch.int32))


def test_dimension_selection_does_not_filter_user_attributes(monkeypatch):
    """Selection carries only synthetic IDs through the extraction filter."""
    grid = _make_line_triangle_grid()
    original_extract_cells = pv.UnstructuredGrid.extract_cells

    def inspect_extraction_source(mesh, indices, *args, **kwargs):
        assert list(mesh.point_data) == []
        assert list(mesh.cell_data) == []
        assert list(mesh.field_data) == []
        return original_extract_cells(mesh, indices, *args, **kwargs)

    monkeypatch.setattr(
        pv.UnstructuredGrid,
        "extract_cells",
        inspect_extraction_source,
    )

    mesh = from_pyvista(grid, manifold_dim=2, warn_on_lost_data=False)

    assert torch.equal(
        mesh.point_data["point_id"],
        torch.arange(100, 105, dtype=torch.int32),
    )
    assert torch.equal(mesh.cell_data["kind"], torch.tensor([20], dtype=torch.int16))
    assert torch.equal(mesh.global_data["case"], torch.tensor([7], dtype=torch.int32))


def test_vertex_polyline_parent_mapping_uses_polydata_cell_order():
    """Line parent indices begin after VTK PolyData vertex cells."""
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [3.0, 1.0, 0.0],
        ]
    )
    polyline = pv.PolyData(
        points,
        verts=np.array([1, 0]),
        lines=np.array([3, 1, 2, 3]),
    )
    polyline.cell_data["parent_id"] = np.array([111, 222], dtype=np.int16)

    with pytest.warns(
        UserWarning,
        match=r"native dimensions \[0\].*parent_id",
    ):
        mesh = from_pyvista(polyline)

    assert torch.equal(mesh.cells, torch.tensor([[1, 2], [2, 3]]))
    assert torch.equal(
        mesh.cell_data["parent_id"],
        torch.tensor([222, 222], dtype=torch.int16),
    )


def test_line_only_polyline_preserves_data_without_warning():
    """Same-dimension polyline splitting exactly replicates parent data."""
    points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 1.0, 0.0], [3.0, 1.0, 0.0]]
    )
    polyline = pv.PolyData(points, lines=np.array([4, 0, 1, 2, 3]))
    polyline.cell_data["line_id"] = np.array([7], dtype=np.int16)

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        mesh = from_pyvista(polyline)

    assert torch.equal(mesh.cells, torch.tensor([[0, 1], [1, 2], [2, 3]]))
    assert torch.equal(
        mesh.cell_data["line_id"],
        torch.tensor([7, 7, 7], dtype=torch.int16),
    )


def test_line_only_polydata_rejects_missing_surface_dimension_without_warning():
    """A missing PolyData surface target fails before warnings or empty tensors."""
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    polydata = pv.PolyData(points, lines=np.array([2, 0, 1]))
    polydata.cell_data["line_id"] = np.array([7], dtype=np.int16)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UserWarning)
        with pytest.raises(ValueError, match="no native surface cells"):
            from_pyvista(polydata, manifold_dim=2)
    relevant = [
        warning for warning in caught if issubclass(warning.category, UserWarning)
    ]
    assert relevant == []


def test_identity_line_parent_map_keeps_zero_copy_data_path(monkeypatch):
    """Two-point line-only data shares storage unless force_copy is requested."""
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    polydata = pv.PolyData(points, lines=np.array([2, 0, 1, 2, 1, 2]))
    polydata.cell_data["line_id"] = np.array([10, 20], dtype=np.int32)

    def fail_repeat(*args, **kwargs):
        raise AssertionError("identity line mapping allocated a repeated parent map")

    monkeypatch.setattr(io_pyvista.np, "repeat", fail_repeat)
    shared = from_pyvista(polydata)
    shared.cell_data["line_id"][0] = 99
    assert polydata.cell_data["line_id"][0] == 99

    polydata.cell_data["line_id"][0] = 10
    copied = from_pyvista(polydata, force_copy=True)
    copied.cell_data["line_id"][0] = 77
    assert polydata.cell_data["line_id"][0] == 10


def test_heterogeneous_polydata_line_mapping_preserves_data_and_input():
    """Heterogeneous lines map multidimensional data without mutating input."""
    points = np.column_stack([np.arange(12, dtype=float), np.zeros((12, 2))])
    polydata = pv.PolyData(
        points,
        verts=np.array([1, 0]),
        lines=np.array([2, 1, 2, 3, 3, 4, 5, 4, 6, 7, 8, 9]),
        faces=np.array([3, 1, 2, 10]),
        strips=np.array([3, 2, 10, 11]),
    )
    polydata.cell_data["parent_id"] = np.array(
        [111, 10, 20, 30, 400, 500], dtype=np.int32
    )
    polydata.cell_data["matrix"] = np.arange(12, dtype=np.int16).reshape(6, 2)
    labels = vtk.vtkStringArray()
    labels.SetName("labels")
    for value in ("vert", "line", "poly3", "poly4", "face", "strip"):
        labels.InsertNextValue(value)
    polydata.GetCellData().AddArray(labels)
    source_points = polydata.points.copy()
    source_ids = polydata.cell_data["parent_id"].copy()
    source_matrix = polydata.cell_data["matrix"].copy()

    mesh = from_pyvista(
        polydata,
        manifold_dim=1,
        warn_on_lost_data=False,
        force_copy=True,
    )

    assert torch.equal(
        mesh.cells,
        torch.tensor([[1, 2], [3, 4], [4, 5], [6, 7], [7, 8], [8, 9]]),
    )
    assert torch.equal(
        mesh.cell_data["parent_id"],
        torch.tensor([10, 20, 20, 30, 30, 30], dtype=torch.int32),
    )
    assert torch.equal(
        mesh.cell_data["matrix"],
        torch.from_numpy(source_matrix[[1, 2, 2, 3, 3, 3]]),
    )
    assert "labels" not in mesh.cell_data

    mesh.points[0, 0] = 99
    mesh.cell_data["parent_id"][0] = 99
    mesh.cell_data["matrix"][0, 0] = 99
    np.testing.assert_array_equal(polydata.points, source_points)
    np.testing.assert_array_equal(polydata.cell_data["parent_id"], source_ids)
    np.testing.assert_array_equal(polydata.cell_data["matrix"], source_matrix)
    assert list(polydata.cell_data["labels"]) == [
        "vert",
        "line",
        "poly3",
        "poly4",
        "face",
        "strip",
    ]


def test_implicit_vertices_triangle_maps_only_surface_parent_data():
    """Implicit VERTEX tuples are dropped while TRIANGLE data is retained."""
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    polydata = pv.PolyData(points)
    polydata.faces = np.array([3, 0, 1, 2])
    polydata.cell_data["parent_id"] = np.array([111, 112, 113, 222])

    with pytest.warns(
        UserWarning,
        match=r"native dimensions \[0\].*parent_id",
    ):
        mesh = from_pyvista(polydata)

    assert mesh.n_manifold_dims == 2
    assert torch.equal(mesh.cells, torch.tensor([[0, 1, 2]]))
    assert torch.equal(mesh.cell_data["parent_id"], torch.tensor([222]))
    assert mesh.cell_areas.sum().item() == pytest.approx(0.5)


def test_triangle_and_strip_surface_mapping_preserves_every_parent():
    """Triangle faces and strips are linearized independently on VTK 9.6."""
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [3.0, 1.0, 0.0],
        ]
    )
    polydata = pv.PolyData(
        points,
        faces=np.array([3, 0, 1, 2]),
        strips=np.array([4, 3, 4, 5, 6]),
    )
    polydata.cell_data["parent_id"] = np.array([10, 20], dtype=np.int16)

    assert polydata.is_all_triangles
    mesh = from_pyvista(polydata)

    assert torch.equal(
        mesh.cells,
        torch.tensor([[0, 1, 2], [3, 4, 5], [5, 4, 6]]),
    )
    assert torch.equal(
        mesh.cell_data["parent_id"],
        torch.tensor([10, 20, 20], dtype=torch.int16),
    )
    assert mesh.cell_areas.sum().item() == pytest.approx(1.5)


def test_polygon_strip_surface_mapping_replicates_exact_parents():
    """Quad, polygon, and strip children inherit their exact parent tuples."""
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [3.5, 0.5, 0.0],
            [2.5, 1.2, 0.0],
            [1.8, 0.6, 0.0],
            [4.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [4.0, 1.0, 0.0],
            [5.0, 1.0, 0.0],
        ]
    )
    polydata = pv.PolyData(
        points,
        faces=np.array([4, 0, 1, 2, 3, 5, 4, 5, 6, 7, 8]),
        strips=np.array([4, 9, 10, 11, 12]),
    )
    polydata.cell_data["parent_id"] = np.array([10, 20, 30], dtype=np.int16)
    polydata.cell_data["__physicsnemo_parent_cell_id"] = np.array(
        [100, 200, 300], dtype=np.int64
    )
    original_collision = polydata.cell_data["__physicsnemo_parent_cell_id"].copy()

    mesh = from_pyvista(polydata)

    assert torch.equal(
        mesh.cell_data["parent_id"],
        torch.tensor([10, 10, 20, 20, 20, 30, 30], dtype=torch.int16),
    )
    assert torch.equal(
        mesh.cell_data["__physicsnemo_parent_cell_id"],
        torch.tensor([100, 100, 200, 200, 200, 300, 300]),
    )
    np.testing.assert_array_equal(
        polydata.cell_data["__physicsnemo_parent_cell_id"],
        original_collision,
    )


@pytest.mark.parametrize("legacy", [False, True], ids=["current", "pyvista-0.46"])
@pytest.mark.parametrize("has_user_ids", [False, True], ids=["synthetic", "user"])
def test_extract_cells_preserves_user_original_ids(monkeypatch, legacy, has_user_ids):
    """Selection removes synthetic cell IDs and restores user collisions."""
    grid = _make_line_triangle_grid()
    if has_user_ids:
        grid.cell_data["vtkOriginalCellIds"] = np.array([1000, 2000])
    original_point_ids = grid.point_data["vtkOriginalPointIds"].copy()
    original_cell_keys = list(grid.cell_data)
    original_user_ids = (
        grid.cell_data["vtkOriginalCellIds"].copy() if has_user_ids else None
    )

    if legacy:
        original_extract_cells = pv.UnstructuredGrid.extract_cells

        def legacy_extract_cells(mesh, indices):
            """Simulate PyVista 0.46 extraction defaults."""
            return original_extract_cells(mesh, indices)

        monkeypatch.setattr(
            pv.UnstructuredGrid,
            "extract_cells",
            legacy_extract_cells,
        )

    mesh = from_pyvista(grid, manifold_dim=2, warn_on_lost_data=False)

    assert torch.equal(mesh.cells, torch.tensor([[2, 3, 4]]))
    if has_user_ids:
        assert torch.equal(mesh.cell_data["vtkOriginalCellIds"], torch.tensor([2000]))
    else:
        assert "vtkOriginalCellIds" not in mesh.cell_data
    np.testing.assert_array_equal(
        grid.point_data["vtkOriginalPointIds"], original_point_ids
    )
    assert list(grid.cell_data) == original_cell_keys
    if original_user_ids is not None:
        np.testing.assert_array_equal(
            grid.cell_data["vtkOriginalCellIds"], original_user_ids
        )


def test_mixed_selection_force_copy_detaches_all_data():
    """force_copy detaches mixed-selection geometry and attached data."""
    grid = _make_line_triangle_grid()
    source_points = grid.points.copy()
    source_point_data = grid.point_data["point_id"].copy()
    source_cell_data = grid.cell_data["kind"].copy()
    source_global_data = grid.field_data["case"].copy()

    mesh = from_pyvista(
        grid,
        manifold_dim=2,
        warn_on_lost_data=False,
        force_copy=True,
    )
    mesh.points[0, 0] = 99
    mesh.point_data["point_id"][0] = 99
    mesh.cell_data["kind"][0] = 99
    mesh.global_data["case"][0] = 99

    np.testing.assert_array_equal(grid.points, source_points)
    np.testing.assert_array_equal(grid.point_data["point_id"], source_point_data)
    np.testing.assert_array_equal(grid.cell_data["kind"], source_cell_data)
    np.testing.assert_array_equal(grid.field_data["case"], source_global_data)


def test_supported_parent_provenance_rejects_vanished_parent(monkeypatch):
    """A supported linear parent cannot disappear during triangulation."""
    grid = _make_two_quad_grid()
    grid.cell_data["__physicsnemo_parent_cell_id"] = np.array([100, 200])
    grid.cell_data["__physicsnemo_parent_cell_id_1"] = np.array([300, 400])
    original_collision_field = grid.cell_data["__physicsnemo_parent_cell_id"].copy()
    original_second_collision = grid.cell_data["__physicsnemo_parent_cell_id_1"].copy()
    original_triangulate = pv.UnstructuredGrid.triangulate

    def drop_second_parent(mesh, *args, **kwargs):
        triangulated = original_triangulate(mesh, *args, **kwargs)
        provenance_key = "__physicsnemo_parent_cell_id"
        assert list(triangulated.cell_data) == [provenance_key]
        keep = np.flatnonzero(triangulated.cell_data[provenance_key] == 0)
        return triangulated.extract_cells(keep)

    monkeypatch.setattr(pv.UnstructuredGrid, "triangulate", drop_second_parent)

    with pytest.raises(ValueError, match=r"parent 1.*QUAD"):
        from_pyvista(grid)
    np.testing.assert_array_equal(
        grid.cell_data["__physicsnemo_parent_cell_id"],
        original_collision_field,
    )
    np.testing.assert_array_equal(
        grid.cell_data["__physicsnemo_parent_cell_id_1"],
        original_second_collision,
    )


def test_triangulation_carries_only_required_provenance(monkeypatch):
    """Triangulation does not copy user attributes that reload from the source."""
    grid = _make_two_quad_grid()
    grid.point_data["point_id"] = np.arange(grid.n_points, dtype=np.int32)
    grid.cell_data["kind"] = np.array([10, 20], dtype=np.int16)
    grid.field_data["case"] = np.array([7], dtype=np.int32)
    original_triangulate = pv.UnstructuredGrid.triangulate

    def inspect_triangulation_source(mesh, *args, **kwargs):
        assert list(mesh.point_data) == []
        assert list(mesh.cell_data) == ["__physicsnemo_parent_cell_id"]
        assert list(mesh.field_data) == []
        return original_triangulate(mesh, *args, **kwargs)

    monkeypatch.setattr(
        pv.UnstructuredGrid,
        "triangulate",
        inspect_triangulation_source,
    )

    mesh = from_pyvista(grid)

    assert torch.equal(mesh.point_data["point_id"], torch.arange(8, dtype=torch.int32))
    assert torch.equal(
        mesh.cell_data["kind"],
        torch.tensor([10, 10, 20, 20], dtype=torch.int16),
    )
    assert torch.equal(mesh.global_data["case"], torch.tensor([7], dtype=torch.int32))


def test_parent_provenance_rejects_unknown_parent(monkeypatch):
    """Parent maps cannot index cells outside the selected source set."""
    grid = _make_two_quad_grid()
    grid.cell_data["kind"] = np.array([10, 20], dtype=np.int16)
    original_triangulate = pv.UnstructuredGrid.triangulate

    def corrupt_parent_id(mesh, *args, **kwargs):
        triangulated = original_triangulate(mesh, *args, **kwargs)
        triangulated.cell_data["__physicsnemo_parent_cell_id"][0] = -1
        return triangulated

    monkeypatch.setattr(pv.UnstructuredGrid, "triangulate", corrupt_parent_id)

    with pytest.raises(ValueError, match=r"unknown parent IDs.*-1"):
        from_pyvista(grid)


def test_malformed_inputs_raise_in_one_subprocess():
    """Malformed attributes and topology are checked in one isolated process."""
    script = textwrap.dedent(
        """
        import sys
        import numpy as np
        import pyvista as pv
        import vtk
        from physicsnemo.mesh.io.io_pyvista import from_pyvista

        attribute_cases = [
            ("point_data", 3, "vtkIntArray", 4),
            ("point_data", 5, "vtkStringArray", 4),
            ("cell_data", 0, "vtkIntArray", 1),
            ("cell_data", 2, "vtkStringArray", 1),
        ]
        failures = []
        for association, actual, array_class, expected in attribute_cases:
            print(
                f"CASE attribute:{association}:{actual}:{array_class}",
                flush=True,
            )
            points = np.array([
                [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0], [1.0, 1.0, 0.0],
            ])
            grid = pv.UnstructuredGrid(
                np.array([4, 0, 1, 2, 3]),
                np.array([pv.CellType.PIXEL]),
                points,
            )
            key = f"bad_{association}"
            array = getattr(vtk, array_class)()
            array.SetName(key)
            array.SetNumberOfTuples(actual)
            for index in range(actual):
                value = "value" if array_class == "vtkStringArray" else index
                array.SetValue(index, value)
            attributes = (
                grid.GetPointData()
                if association == "point_data"
                else grid.GetCellData()
            )
            attributes.AddArray(array)
            try:
                from_pyvista(grid)
            except ValueError as error:
                required = [
                    association, key, f"expected {expected}", f"got {actual}"
                ]
                if not all(part in str(error) for part in required):
                    failures.append((association, actual, str(error)))
            else:
                failures.append((association, actual, "no ValueError"))

        topology_cases = [
            ("PIXEL", 4, [0, 1, 2], ["PIXEL", "exactly 4", "got 3"]),
            ("PIXEL", 5, [0, 1, 2, 3, 4], ["PIXEL", "exactly 4", "got 5"]),
            ("PIXEL", 4, [0, 1, 2, -1], ["point ID -1", "PIXEL"]),
            ("PIXEL", 4, [0, 1, 2, 99], ["point ID 99", "PIXEL"]),
            ("LINE", 3, [0, 1, 2], ["LINE", "exactly 2", "got 3"]),
            ("POLY_LINE", 1, [0], ["POLY_LINE", "at least 2", "got 1"]),
            ("TRIANGLE", 3, [0, 1], ["TRIANGLE", "exactly 3", "got 2"]),
            ("TRIANGLE", 4, [0, 1, 2, 3], ["TRIANGLE", "exactly 3", "got 4"]),
            ("TRIANGLE", 3, [0, 1, 99], ["point ID 99", "TRIANGLE"]),
            ("TETRA", 4, [0, 1, 2], ["TETRA", "exactly 4", "got 3"]),
            ("TETRA", 5, [0, 1, 2, 3, 4], ["TETRA", "exactly 4", "got 5"]),
            ("TETRA", 4, [0, 1, 2, -1], ["point ID -1", "TETRA"]),
        ]
        for name, point_count, connectivity, required in topology_cases:
            print(f"CASE topology:{name}:{connectivity}", flush=True)
            points = np.zeros((point_count, 3))
            grid = pv.UnstructuredGrid(
                np.array([len(connectivity), *connectivity]),
                np.array([getattr(pv.CellType, name)]),
                points,
            )
            try:
                from_pyvista(grid)
            except ValueError as error:
                if not all(part in str(error) for part in required):
                    failures.append((name, connectivity, str(error)))
            else:
                failures.append((name, connectivity, "no ValueError"))

        marker = "CASE unstructured:invalid-offsets"
        print(marker, flush=True)
        structurally_invalid = pv.UnstructuredGrid(
            np.array([4, 0, 1, 2, 3]),
            np.array([pv.CellType.TETRA]),
            np.zeros((4, 3)),
        )
        np.asarray(
            structurally_invalid.GetCells().GetOffsetsArray()
        )[-1] += 1
        try:
            from_pyvista(structurally_invalid)
        except ValueError as error:
            if "UnstructuredGrid cells" not in str(error):
                failures.append((marker, str(error)))
        else:
            failures.append((marker, "no ValueError"))

        marker = "CASE unstructured:cell-type-count"
        print(marker, flush=True)
        invalid_cell_types = pv.UnstructuredGrid(
            np.array([4, 0, 1, 2, 3, 4, 4, 5, 6, 7]),
            np.array([pv.CellType.QUAD, pv.CellType.QUAD]),
            np.zeros((8, 3)),
        )
        invalid_cell_types.GetCellTypesArray().SetNumberOfTuples(1)
        try:
            from_pyvista(invalid_cell_types)
        except ValueError as error:
            required = ["cell types", "expected 2", "got 1"]
            if not all(part in str(error) for part in required):
                failures.append((marker, str(error)))
        else:
            failures.append((marker, "no ValueError"))

        marker = "CASE unstructured:multi-component-cell-types"
        print(marker, flush=True)
        invalid_cell_type_shape = pv.UnstructuredGrid(
            np.array([4, 0, 1, 2, 3]),
            np.array([pv.CellType.QUAD]),
            np.zeros((4, 3)),
        )
        vtk_cell_types = invalid_cell_type_shape.GetCellTypesArray()
        vtk_cell_types.SetNumberOfComponents(2)
        vtk_cell_types.SetNumberOfTuples(1)
        vtk_cell_types.SetComponent(0, 0, int(pv.CellType.QUAD))
        vtk_cell_types.SetComponent(0, 1, int(pv.CellType.QUAD))
        vtk_cell_types.Modified()
        invalid_cell_type_shape.Modified()
        try:
            from_pyvista(invalid_cell_type_shape)
        except ValueError as error:
            required = ["one-dimensional", "shape (1, 2)"]
            if not all(part in str(error) for part in required):
                failures.append((marker, str(error)))
        else:
            failures.append((marker, "no ValueError"))

        centroid_cases = [
            ("PIXEL", 4, [0, 1, 2], ["PIXEL", "exactly 4", "got 3"]),
            ("POLY_LINE", 1, [0], ["POLY_LINE", "at least 2", "got 1"]),
            ("PIXEL", 4, [0, 1, 2, 99], ["point ID 99", "PIXEL"]),
            (
                "QUADRATIC_TETRA",
                9,
                list(range(9)),
                ["QUADRATIC_TETRA", "exactly 10", "got 9"],
            ),
        ]
        for name, point_count, connectivity, required in centroid_cases:
            print(f"CASE centroid:{name}:{connectivity}", flush=True)
            points = np.zeros((point_count, 3))
            grid = pv.UnstructuredGrid(
                np.array([len(connectivity), *connectivity]),
                np.array([getattr(pv.CellType, name)]),
                points,
            )
            try:
                from_pyvista(grid, point_source="cell_centroids")
            except ValueError as error:
                if not all(part in str(error) for part in required):
                    failures.append((f"centroid-{name}", connectivity, str(error)))
            else:
                failures.append((f"centroid-{name}", connectivity, "no ValueError"))

        if failures:
            print(repr(failures), file=sys.stderr)
            raise SystemExit(2)
        """
    )
    _run_isolated_script(script)


def test_malformed_polydata_streams_raise_in_one_subprocess():
    """Every PolyData stream is validated in one crash-isolated process."""
    script = textwrap.dedent(
        """
        import sys
        import numpy as np
        import pyvista as pv
        import vtk
        from vtk.util.numpy_support import numpy_to_vtkIdTypeArray
        from physicsnemo.mesh.io.io_pyvista import from_pyvista

        specifications = {
            "verts": ("SetVerts", 1, 0),
            "lines": ("SetLines", 2, 1),
            "faces": ("SetPolys", 3, 2),
            "strips": ("SetStrips", 3, 2),
        }

        def make_polydata(association, raw):
            polydata = pv.PolyData(np.zeros((4, 3)))
            legacy = numpy_to_vtkIdTypeArray(
                np.asarray(raw, dtype=np.int64),
                deep=True,
            )
            cells = vtk.vtkCellArray()
            cells.ImportLegacyFormat(legacy)
            getattr(polydata, specifications[association][0])(cells)
            return polydata

        failures = []
        for association, (_, minimum, manifold_dim) in specifications.items():
            cases = [
                ("low", [minimum - 1, *range(max(minimum - 1, 0))]),
                ("negative", [minimum, -1, *range(1, minimum)]),
                ("oob", [minimum, 99, *range(1, minimum)]),
            ]
            for case_name, raw in cases:
                marker = f"CASE polydata:{association}:{case_name}"
                print(marker, flush=True)
                try:
                    from_pyvista(
                        make_polydata(association, raw),
                        manifold_dim=manifold_dim,
                        warn_on_lost_data=False,
                    )
                except ValueError as error:
                    if f"PolyData {association}" not in str(error):
                        failures.append((marker, str(error)))
                else:
                    failures.append((marker, "no ValueError"))

            valid = [minimum, *range(minimum)]
            marker = f"CASE polydata:{association}:valid"
            print(marker, flush=True)
            try:
                from_pyvista(
                    make_polydata(association, valid),
                    manifold_dim=manifold_dim,
                    warn_on_lost_data=False,
                )
            except Exception as error:
                failures.append((marker, repr(error)))

            marker = f"CASE polydata:{association}:invalid-offsets"
            print(marker, flush=True)
            structurally_invalid = make_polydata(association, valid)
            getters = {
                "verts": "GetVerts",
                "lines": "GetLines",
                "faces": "GetPolys",
                "strips": "GetStrips",
            }
            cell_array = getattr(structurally_invalid, getters[association])()
            np.asarray(cell_array.GetOffsetsArray())[-1] += 1
            try:
                from_pyvista(
                    structurally_invalid,
                    manifold_dim=manifold_dim,
                    warn_on_lost_data=False,
                )
            except ValueError as error:
                if f"PolyData {association}" not in str(error):
                    failures.append((marker, str(error)))
            else:
                failures.append((marker, "no ValueError"))

        if failures:
            print(repr(failures), file=sys.stderr)
            raise SystemExit(2)
        """
    )
    _run_isolated_script(script)


def test_malformed_polyhedron_auxiliary_arrays_raise_in_one_subprocess():
    """Malformed polyhedron faces/locations are checked in one isolated process."""
    script = textwrap.dedent(
        """
        import sys
        import numpy as np
        import pyvista as pv
        import vtk
        from physicsnemo.mesh.io.io_pyvista import from_pyvista
        from test.mesh.io.io_pyvista.test_from_pyvista_3d import (
            _make_pentagonal_prism,
        )

        base = _make_pentagonal_prism()

        def clone(cell_array):
            result = vtk.vtkCellArray()
            result.DeepCopy(cell_array)
            return result

        cases = {
            "face-offset": "POLYHEDRON faces",
            "face-id": "face point ID 99",
            "location-offset": "POLYHEDRON face locations",
            "location-id": "face-location reference 99",
            "parent-face-point-set": "connectivity and referenced faces",
        }
        failures = []
        for case_name, expected in cases.items():
            marker = f"CASE polyhedron:{case_name}"
            print(marker, flush=True)
            faces = clone(base.GetPolyhedronFaces())
            locations = clone(base.GetPolyhedronFaceLocations())
            cells = clone(base.GetCells())
            if case_name == "face-offset":
                np.asarray(faces.GetOffsetsArray())[-1] += 1
            elif case_name == "face-id":
                np.asarray(faces.GetConnectivityArray())[0] = 99
            elif case_name == "location-offset":
                np.asarray(locations.GetOffsetsArray())[-1] += 1
            elif case_name == "parent-face-point-set":
                np.asarray(cells.GetConnectivityArray())[0] = 9
            else:
                np.asarray(locations.GetConnectivityArray())[0] = 99

            vtk_grid = vtk.vtkUnstructuredGrid()
            vtk_grid.SetPoints(base.GetPoints())
            vtk_grid.SetPolyhedralCells(
                base.GetCellTypesArray(),
                cells,
                locations,
                faces,
            )
            try:
                from_pyvista(pv.wrap(vtk_grid))
            except ValueError as error:
                if expected not in str(error):
                    failures.append((marker, str(error)))
            else:
                failures.append((marker, "no ValueError"))

        marker = "CASE polyhedron:valid"
        print(marker, flush=True)
        try:
            mesh = from_pyvista(base)
            if mesh.n_cells == 0:
                failures.append((marker, "empty output"))
        except Exception as error:
            failures.append((marker, repr(error)))

        if failures:
            print(repr(failures), file=sys.stderr)
            raise SystemExit(2)
        """
    )
    _run_isolated_script(script)


def test_malformed_polyhedron_face_complexes_raise_on_every_path():
    """Common shell validation protects tetrahedron, centroid, and edge filters."""
    script = textwrap.dedent(
        """
        import sys
        import numpy as np
        import pyvista as pv
        import vtk
        from physicsnemo.mesh.io.io_pyvista import from_pyvista
        from test.mesh.io.io_pyvista.test_from_pyvista_3d import (
            _make_pentagonal_prism,
        )

        def make_polyhedron(points, faces):
            face_stream = [len(faces)]
            for face in faces:
                face_stream.extend([len(face), *face])
            return pv.UnstructuredGrid(
                np.array([len(face_stream), *face_stream]),
                np.array([pv.CellType.POLYHEDRON]),
                np.asarray(points, dtype=float),
            )

        def clone(cell_array):
            result = vtk.vtkCellArray()
            result.DeepCopy(cell_array)
            return result

        def make_parent_face_mismatch():
            base = _make_pentagonal_prism()
            faces = clone(base.GetPolyhedronFaces())
            locations = clone(base.GetPolyhedronFaceLocations())
            cells = clone(base.GetCells())
            np.asarray(cells.GetConnectivityArray())[0] = 9
            vtk_grid = vtk.vtkUnstructuredGrid()
            vtk_grid.SetPoints(base.GetPoints())
            vtk_grid.SetPolyhedralCells(
                base.GetCellTypesArray(),
                cells,
                locations,
                faces,
            )
            return pv.wrap(vtk_grid)

        tetra_points = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        tetra_faces = [
            [0, 2, 1],
            [0, 1, 3],
            [1, 2, 3],
            [2, 0, 3],
        ]
        duplicate_missing = make_polyhedron(
            tetra_points,
            [tetra_faces[0], tetra_faces[0], tetra_faces[1], tetra_faces[1]],
        )
        repeated_face_point = make_polyhedron(
            tetra_points,
            [[0, 1, 1], *tetra_faces[1:]],
        )

        pyramid_points = np.array([
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        open_shell = make_polyhedron(
            pyramid_points,
            [[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]],
        )

        two_shell_points = np.vstack(
            [tetra_points, tetra_points + np.array([3.0, 0.0, 0.0])]
        )
        two_shell_faces = [
            *tetra_faces,
            *[[point_id + 4 for point_id in face] for face in tetra_faces],
        ]
        disconnected_shells = make_polyhedron(two_shell_points, two_shell_faces)

        cases = [
            (
                "parent-face-point-set",
                make_parent_face_mismatch(),
                "connectivity and referenced faces",
            ),
            ("duplicate-missing", duplicate_missing, "duplicate faces"),
            ("repeated-face-point", repeated_face_point, "face repeats point ID"),
            ("open-shell", open_shell, "exactly 2 faces"),
            (
                "disconnected-shells",
                disconnected_shells,
                "one connected closed shell",
            ),
        ]
        conversions = [
            ("tetrahedralization", {}),
            (
                "centroids",
                {"manifold_dim": 0, "point_source": "cell_centroids"},
            ),
            ("edges", {"manifold_dim": 1}),
        ]

        failures = []
        for case_name, grid, expected in cases:
            for conversion_name, kwargs in conversions:
                marker = f"CASE {case_name}:{conversion_name}"
                print(marker, flush=True)
                try:
                    from_pyvista(grid, warn_on_lost_data=False, **kwargs)
                except ValueError as error:
                    if expected not in str(error):
                        failures.append((marker, str(error)))
                else:
                    failures.append((marker, "no ValueError"))

        if failures:
            print(repr(failures), file=sys.stderr)
            raise SystemExit(2)
        """
    )
    _run_isolated_script(script)


def test_unselected_polyhedron_face_complex_is_not_deeply_validated():
    """Explicit dimensional selection validates only topology it consumes."""
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
        ]
    )
    tetra_faces = [[0, 2, 1], [0, 1, 3]]
    malformed_face_stream = [
        4,
        *[value for face in tetra_faces for value in (len(face), *face)],
        *[value for face in tetra_faces for value in (len(face), *face)],
    ]
    grid = pv.UnstructuredGrid(
        np.array(
            [
                len(malformed_face_stream),
                *malformed_face_stream,
                3,
                4,
                5,
                6,
            ]
        ),
        np.array([pv.CellType.POLYHEDRON, pv.CellType.TRIANGLE]),
        points,
    )

    with pytest.raises(ValueError, match=r"index 0: duplicate faces"):
        from_pyvista(grid, manifold_dim=3, warn_on_lost_data=False)

    mesh = from_pyvista(grid, manifold_dim=2, warn_on_lost_data=False)

    assert torch.equal(mesh.cells, torch.tensor([[4, 5, 6]]))

    np.asarray(grid.GetPolyhedronFaces().GetConnectivityArray())[0] = 99
    with pytest.raises(ValueError, match="face point ID 99"):
        from_pyvista(grid, manifold_dim=2, warn_on_lost_data=False)


def test_bulk_valid_polyhedra_use_batched_face_complex_fast_path(monkeypatch):
    """Valid bulk topology avoids one scalar validation call per cell."""
    n_polyhedra = 128
    grid = _make_bulk_polyhedra(n_polyhedra)

    def unexpected_scalar_fallback(*args, **kwargs):
        raise AssertionError("valid bulk topology used the scalar diagnostic path")

    monkeypatch.setattr(
        io_pyvista,
        "_validate_polyhedron_face_complex",
        unexpected_scalar_fallback,
    )

    mesh = from_pyvista(
        grid,
        manifold_dim=0,
        point_source="cell_centroids",
        warn_on_lost_data=False,
    )

    assert mesh.n_points == n_polyhedra


def test_polyhedron_validation_batches_respect_face_point_budget():
    """Ragged parents split on gathered values as well as parent count."""
    grid = _make_bulk_polyhedra(5)
    face_locations = grid.GetPolyhedronFaceLocations()
    faces = grid.GetPolyhedronFaces()

    def batch_lists(max_face_points):
        return [
            batch.tolist()
            for batch in io_pyvista._polyhedron_validation_batches(
                np.arange(5, dtype=np.int64),
                np.asarray(face_locations.GetOffsetsArray()),
                np.asarray(face_locations.GetConnectivityArray()),
                np.asarray(faces.GetOffsetsArray()),
                max_cells=3,
                max_face_points=max_face_points,
            )
        ]

    assert batch_lists(28) == [[0, 1], [2, 3], [4]]
    assert batch_lists(8) == [[0], [1], [2], [3], [4]]


def test_batched_polyhedron_error_uses_parent_id_across_batch_boundary():
    """A diagnostic after the cell-count boundary retains its source ID."""
    malformed_parent_id = 4159
    grid = _make_bulk_polyhedra(
        malformed_parent_id + 1,
        malformed_parent_id=malformed_parent_id,
    )

    with pytest.raises(
        ValueError,
        match=rf"index {malformed_parent_id}: duplicate faces",
    ):
        from_pyvista(
            grid,
            manifold_dim=0,
            point_source="cell_centroids",
            warn_on_lost_data=False,
        )


def test_concave_polyhedron_rejected_before_tetrahedralization():
    """Concave polyhedra cannot enter VTK's shape-altering tetrahedralizer."""
    grid = _make_concave_l_prism()
    assert not bool(grid.GetCell(0).IsConvex())

    with pytest.raises(ValueError, match=r"non-convex.*POLYHEDRON.*index 0"):
        from_pyvista(grid)


def test_translated_float32_concave_polyhedron_remains_rejected():
    """Legacy-tolerance fallback cannot override VTK 9.6 concavity."""
    grid = _make_concave_l_prism()
    grid.points = (grid.points + 1e6).astype(np.float32)

    with pytest.raises(ValueError, match=r"non-convex.*POLYHEDRON.*index 0"):
        from_pyvista(grid)


@pytest.mark.parametrize(
    "dtype,scale,offset",
    [
        (np.float32, 1.0, 0.0),
        (np.float32, 1.0, 1e6),
        (np.float64, 1e-9, 0.0),
        (np.float64, 1.0, 1e6),
    ],
)
def test_convex_polyhedron_check_is_scale_and_precision_robust(
    dtype,
    scale,
    offset,
):
    """Valid transformed polyhedra survive the conservative convexity check."""
    from test.mesh.io.io_pyvista.test_from_pyvista_3d import (
        _make_pentagonal_prism,
    )

    angle = np.deg2rad(37.0)
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    grid = _make_pentagonal_prism()
    grid.points = ((grid.points @ rotation.T) * scale + offset).astype(dtype)

    mesh = from_pyvista(grid)

    assert mesh.n_cells > 0


@pytest.mark.parametrize(
    "cell_type",
    [
        pv.CellType.QUADRATIC_EDGE,
        pv.CellType.QUADRATIC_TETRA,
        pv.CellType.LAGRANGE_QUADRILATERAL,
        pv.CellType.BEZIER_CURVE,
        pv.CellType.CONVEX_POINT_SET,
    ],
)
def test_unsupported_topology_rejected_but_point_cloud_allowed(cell_type):
    """Unsupported topology rejects auto but permits unconnected point clouds."""
    grid = _make_unsupported_grid(cell_type)
    error_pattern = rf"{cell_type.name}.*globally conforming higher-order tessellation"

    with pytest.raises(ValueError, match=error_pattern):
        from_pyvista(grid)

    point_cloud = from_pyvista(
        grid,
        manifold_dim=0,
        warn_on_lost_data=False,
    )
    assert point_cloud.n_manifold_dims == 0
    assert torch.equal(point_cloud.points, torch.from_numpy(grid.points))
    assert torch.equal(
        point_cloud.point_data["point_id"],
        torch.from_numpy(grid.point_data["point_id"]),
    )


@pytest.mark.parametrize(
    "cell_type",
    [
        pv.CellType.QUADRATIC_EDGE,
        pv.CellType.QUADRATIC_TRIANGLE,
        pv.CellType.QUADRATIC_QUAD,
        pv.CellType.QUADRATIC_TETRA,
        pv.CellType.QUADRATIC_HEXAHEDRON,
        pv.CellType.QUADRATIC_WEDGE,
        pv.CellType.QUADRATIC_PYRAMID,
        pv.CellType.BIQUADRATIC_QUAD,
        pv.CellType.TRIQUADRATIC_HEXAHEDRON,
        pv.CellType.TRIQUADRATIC_PYRAMID,
        pv.CellType.QUADRATIC_LINEAR_QUAD,
        pv.CellType.QUADRATIC_LINEAR_WEDGE,
        pv.CellType.BIQUADRATIC_QUADRATIC_WEDGE,
        pv.CellType.BIQUADRATIC_QUADRATIC_HEXAHEDRON,
        pv.CellType.BIQUADRATIC_TRIANGLE,
        pv.CellType.CUBIC_LINE,
    ],
)
def test_fixed_higher_order_centroids_preserve_parent_data(cell_type):
    """Fixed-size nonlinear cells retain the pre-existing centroid path."""
    points = _vtk_parametric_points(cell_type)
    grid = pv.UnstructuredGrid(
        np.concatenate(([len(points)], np.arange(len(points)))),
        np.array([cell_type]),
        points,
    )
    grid.cell_data["kind"] = np.array([9], dtype=np.int16)

    mesh = from_pyvista(grid, point_source="cell_centroids")

    assert mesh.n_points == 1
    assert torch.equal(mesh.point_data["kind"], torch.tensor([9], dtype=torch.int16))


@pytest.mark.parametrize(
    "cell_type",
    [pv.CellType.QUADRATIC_EDGE, pv.CellType.CUBIC_LINE],
)
def test_higher_order_centroid_dual_graph_remains_rejected(cell_type):
    """Higher-order interpolation nodes cannot become false line endpoints."""
    points = _vtk_parametric_points(cell_type)
    grid = pv.UnstructuredGrid(
        np.concatenate(([len(points)], np.arange(len(points)))),
        np.array([cell_type]),
        points,
    )

    with pytest.raises(
        ValueError,
        match=rf"{cell_type.name}.*centroid filtering.*linear dual-graph",
    ):
        from_pyvista(
            grid,
            manifold_dim=1,
            point_source="cell_centroids",
        )


@pytest.mark.parametrize(
    "cell_type",
    [
        pv.CellType.LAGRANGE_QUADRILATERAL,
        pv.CellType.BEZIER_CURVE,
        pv.CellType.CONVEX_POINT_SET,
    ],
)
def test_unvalidated_centroid_families_remain_rejected(cell_type):
    """Generic and variable-order cells stay outside validated centroid scope."""
    grid = _make_unsupported_grid(cell_type)

    with pytest.raises(
        ValueError,
        match=rf"{cell_type.name}.*centroid filtering.*fixed-size",
    ):
        from_pyvista(grid, point_source="cell_centroids")


def test_explicit_point_cloud_does_not_inspect_supported_topology():
    """Vertex-only conversion ignores connectivity even for allowlisted cells."""
    grid = pv.UnstructuredGrid(
        np.array([4, 0, 1, 2, 99]),
        np.array([pv.CellType.PIXEL]),
        np.zeros((4, 3)),
    )
    grid.point_data["point_id"] = np.arange(4, dtype=np.int32)

    mesh = from_pyvista(
        grid,
        manifold_dim=0,
        warn_on_lost_data=False,
    )

    assert mesh.n_points == 4
    assert torch.equal(mesh.point_data["point_id"], torch.arange(4, dtype=torch.int32))


def test_unsupported_point_cloud_warns_when_parent_data_is_dropped():
    """Unsupported point-cloud conversion warns about discarded parent data."""
    grid = _make_unsupported_grid(pv.CellType.QUADRATIC_EDGE)

    with pytest.warns(UserWarning, match=r"cell_data.*kind") as caught:
        point_cloud = from_pyvista(grid, manifold_dim=0)
    assert point_cloud.n_points == grid.n_points
    assert "Use point_source='cell_centroids'" not in str(caught[0].message)


def test_zero_cell_schema_does_not_warn_about_lost_parent_data():
    """A zero-length cell-data schema contains no parent tuples to lose."""
    grid = pv.UnstructuredGrid(
        np.array([], dtype=np.int64),
        np.array([], dtype=np.uint8),
        np.empty((0, 3)),
    )
    grid.cell_data["kind"] = np.empty(0, dtype=np.int16)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UserWarning)
        from_pyvista(grid, manifold_dim=0)

    relevant = [
        warning
        for warning in caught
        if issubclass(warning.category, UserWarning)
        and "cell_data" in str(warning.message)
    ]
    assert relevant == []


def test_mixed_grid_with_unsupported_parent_rejects_before_selection():
    """Unsupported parents cannot be hidden by selecting a safe dimension."""
    unsupported = _make_unsupported_grid(pv.CellType.QUADRATIC_EDGE)
    tetra_points = np.array(
        [
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [2.0, 0.0, 1.0],
        ]
    )
    points = np.vstack([unsupported.points, tetra_points])
    grid = pv.UnstructuredGrid(
        np.array([3, 0, 1, 2, 4, 3, 4, 5, 6]),
        np.array([pv.CellType.QUADRATIC_EDGE, pv.CellType.TETRA]),
        points,
    )

    with pytest.raises(ValueError, match="QUADRATIC_EDGE.*globally conforming"):
        from_pyvista(grid, manifold_dim=3)


@pytest.mark.parametrize("mixed", [False, True], ids=["empty", "empty-triangle"])
def test_empty_cell_centroid_rejected_before_data_misalignment(mixed):
    """EMPTY_CELL cannot silently omit a parent center and its data tuple."""
    if mixed:
        points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        cells = np.array([0, 3, 0, 1, 2])
        cell_types = np.array([pv.CellType.EMPTY_CELL, pv.CellType.TRIANGLE])
        parent_data = np.array([0, 2], dtype=np.int16)
    else:
        points = np.empty((0, 3))
        cells = np.array([0])
        cell_types = np.array([pv.CellType.EMPTY_CELL])
        parent_data = np.array([0], dtype=np.int16)
    grid = pv.UnstructuredGrid(cells, cell_types, points)
    grid.cell_data["kind"] = parent_data

    with pytest.raises(
        ValueError,
        match=r"EMPTY_CELL.*cell centers.*cell_data alignment",
    ):
        from_pyvista(grid, point_source="cell_centroids")


def test_supported_generic_centroids_still_work():
    """Centroid mode remains available for supported linear generic cells."""
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [2.0, 2.0, 0.0],
        ]
    )
    grid = pv.UnstructuredGrid(
        np.array([4, 0, 1, 2, 3]),
        np.array([pv.CellType.PIXEL]),
        points,
    )
    grid.cell_data["kind"] = np.array([8], dtype=np.int16)

    mesh = from_pyvista(grid, point_source="cell_centroids")

    assert mesh.n_points == 1
    assert torch.equal(mesh.point_data["kind"], torch.tensor([8], dtype=torch.int16))


def test_failed_target_selection_emits_no_warning():
    """Selection failure occurs before any data-loss warning."""
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    grid = pv.UnstructuredGrid(
        np.array([3, 0, 1, 2]),
        np.array([pv.CellType.TRIANGLE]),
        points,
    )
    grid.cell_data["kind"] = np.array([20], dtype=np.int32)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UserWarning)
        with pytest.raises(ValueError, match="no cells with manifold dimension 3"):
            from_pyvista(grid, manifold_dim=3)
    relevant = [
        warning
        for warning in caught
        if issubclass(warning.category, UserWarning)
        and "cell_data" in str(warning.message)
    ]
    assert relevant == []


def test_warnings_resolve_to_user_callsite():
    """Vertex and centroid warnings point through the version-check wrapper."""
    vertex_grid = _make_triangle_tetra_grid()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UserWarning)
        vertex_call_line = _next_line_number()
        from_pyvista(vertex_grid)
    relevant = [
        warning
        for warning in caught
        if issubclass(warning.category, UserWarning)
        and "cell_data" in str(warning.message)
    ]
    assert len(relevant) == 1
    assert Path(relevant[0].filename) == Path(__file__)
    assert relevant[0].lineno == vertex_call_line

    centroid_grid = pv.UnstructuredGrid(
        np.array([3, 0, 1, 2]),
        np.array([pv.CellType.TRIANGLE]),
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
    )
    centroid_grid.point_data["temperature"] = np.array([1.0, 2.0, 3.0])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UserWarning)
        centroid_call_line = _next_line_number()
        from_pyvista(centroid_grid, point_source="cell_centroids")
    relevant = [
        warning
        for warning in caught
        if issubclass(warning.category, UserWarning)
        and "point_data" in str(warning.message)
    ]
    assert len(relevant) == 1
    assert Path(relevant[0].filename) == Path(__file__)
    assert relevant[0].lineno == centroid_call_line


def test_homogeneous_simplex_fast_path_skips_generic_allocations(monkeypatch):
    """Common simplex grids bypass dimension maps and selection arrays."""

    def fail_if_called(*args, **kwargs):
        raise AssertionError("common simplex grid used generic allocation path")

    monkeypatch.setattr(io_pyvista, "_unstructured_cell_dimensions", fail_if_called)
    monkeypatch.setattr(
        io_pyvista,
        "_select_and_linearize_unstructured_grid",
        fail_if_called,
    )
    cases = [
        (
            pv.CellType.LINE,
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            [0, 1],
        ),
        (
            pv.CellType.TRIANGLE,
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            [0, 1, 2],
        ),
        (
            pv.CellType.TETRA,
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
            [0, 1, 2, 3],
        ),
    ]
    for cell_type, points, connectivity in cases:
        grid = pv.UnstructuredGrid(
            np.array([len(connectivity), *connectivity]),
            np.array([cell_type]),
            points,
        )
        grid.cell_data["parent"] = np.array([7], dtype=np.int16)

        mesh = from_pyvista(grid)

        assert torch.equal(mesh.cells, torch.tensor([connectivity]))
        assert torch.equal(
            mesh.cell_data["parent"], torch.tensor([7], dtype=torch.int16)
        )
