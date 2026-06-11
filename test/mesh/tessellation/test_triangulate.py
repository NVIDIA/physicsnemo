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

"""Tests for physicsnemo.mesh.tessellation.triangulate."""

import numpy as np
import pytest
import torch

from physicsnemo.mesh import Mesh
from physicsnemo.mesh.neighbors import Adjacency
from physicsnemo.mesh.tessellation import triangulate

### Geometry fixtures in the z = 0 plane. The "dart" is non-convex: vertex 3 is
### reflex, so a vertex-0 fan emits overlapping triangles whose unsigned areas
### sum to 10 instead of the true area 6.
CONVEX_PENTAGON = np.array(
    [
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [2.5, 1.5, 0.0],
        [1.0, 2.5, 0.0],
        [-0.5, 1.5, 0.0],
    ]
)
DART = np.array([[0.0, 0.0, 0.0], [4.0, 2.0, 0.0], [0.0, 4.0, 0.0], [1.0, 2.0, 0.0]])
DART_TRUE_AREA = 6.0

### A non-convex pentagon whose reflex tip (vertex 3) lies *exactly* on the
### diagonal joining vertices 1 and 4 -- an edge of the candidate ear (4, 0, 1).
### A strict point-in-triangle test classifies that on-edge vertex as outside,
### clips the invalid ear, and emits overlapping triangles whose unsigned areas
### sum to 7.5 instead of the true 4.5. The boundary-inclusive ear test rejects
### that ear (regression for GH #1711).
ON_EDGE_PENTAGON = np.array(
    [
        [0.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
        [3.0, 2.0, 0.0],
        [1.5, 1.0, 0.0],
        [0.0, 2.0, 0.0],
    ]
)
ON_EDGE_PENTAGON_TRUE_AREA = 4.5


def _adjacency(*rings: list[int]) -> Adjacency:
    """Build a polygon Adjacency from explicit vertex-index rings."""
    counts = torch.tensor([len(r) for r in rings])
    offsets = torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)])
    indices = torch.tensor([v for ring in rings for v in ring])
    return Adjacency(offsets=offsets, indices=indices)


def _single(poly: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Triangulate one polygon; return (points, cells, parent_index)."""
    points = torch.as_tensor(poly, dtype=torch.float64)
    cells, parent_index = triangulate(points, _adjacency(list(range(poly.shape[0]))))
    return points, cells, parent_index


def _areas(points: torch.Tensor, cells: torch.Tensor) -> tuple[float, np.ndarray]:
    """Return (sum of unsigned triangle areas, summed signed vector area)."""
    mesh = Mesh(points=points.to(torch.float64), cells=cells.to(torch.long))
    scalar = float(mesh.cell_areas.sum())
    vector = (
        mesh.cell_normals.to(torch.float64)
        * mesh.cell_areas.to(torch.float64).unsqueeze(-1)
    ).sum(dim=0)
    return scalar, vector.numpy()


def _newell_vector_area(polygon: np.ndarray) -> np.ndarray:
    """Signed vector area of a planar polygon (triangulation-independent)."""
    polygon = np.asarray(polygon, dtype=np.float64)
    return 0.5 * np.cross(polygon, np.roll(polygon, -1, axis=0)).sum(axis=0)


def test_quad_fan_and_parent_index():
    """A convex quad fans into two triangles sharing one parent polygon."""
    points = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
    )
    cells, parent_index = triangulate(points, _adjacency([0, 1, 2, 3]))
    assert cells.tolist() == [[0, 1, 2], [0, 2, 3]]
    assert parent_index.tolist() == [0, 0]
    assert cells.dtype == torch.long
    assert parent_index.dtype == torch.long


def test_triangle_passthrough():
    """A polygon that is already a triangle is emitted unchanged."""
    points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    cells, parent_index = triangulate(points, _adjacency([0, 1, 2]))
    assert cells.tolist() == [[0, 1, 2]]
    assert parent_index.tolist() == [0]


def test_manifold_dim_other_than_2_raises():
    """Only the 2D (polygon -> triangle) case is implemented."""
    points = torch.zeros(4, 3)
    with pytest.raises(NotImplementedError, match="manifold_dim"):
        triangulate(points, _adjacency([0, 1, 2, 3]), manifold_dim=3)


def test_too_few_vertices_raises():
    """A polygon with fewer than three vertices is rejected."""
    points = torch.zeros(2, 3)
    with pytest.raises(ValueError, match="3 vertices"):
        triangulate(points, _adjacency([0, 1]))


def test_empty_input():
    """Zero polygons yields empty, well-shaped outputs."""
    empty = Adjacency(
        offsets=torch.zeros(1, dtype=torch.long),
        indices=torch.zeros(0, dtype=torch.long),
    )
    cells, parent_index = triangulate(torch.zeros(0, 3), empty)
    assert cells.shape == (0, 3)
    assert parent_index.shape == (0,)


def test_convex_polygon_areas_match_truth():
    """For a convex polygon, fan areas (scalar and vector) match ground truth."""
    points, cells, _ = _single(CONVEX_PENTAGON)
    scalar, vector = _areas(points, cells)
    truth = _newell_vector_area(CONVEX_PENTAGON)
    assert scalar == pytest.approx(float(np.linalg.norm(truth)), rel=1e-9)
    np.testing.assert_allclose(vector, truth, atol=1e-9)


def test_nonconvex_scalar_area_is_corrected():
    """The dart's ear-clip area is the true area; a bare fan over-counts."""
    points, cells, _ = _single(DART)
    scalar, vector = _areas(points, cells)
    assert scalar == pytest.approx(DART_TRUE_AREA, rel=1e-9)

    fan_cells, _ = triangulate(
        torch.as_tensor(DART, dtype=torch.float64),
        _adjacency([0, 1, 2, 3]),
        assume_convex=True,  # force the bare fan that this fix replaces
    )
    fan_scalar, fan_vector = _areas(
        torch.as_tensor(DART, dtype=torch.float64), fan_cells
    )
    assert fan_scalar == pytest.approx(10.0, rel=1e-9)  # the over-count this fixes

    # Signed vector area telescopes regardless of triangulation, so both agree.
    truth = _newell_vector_area(DART)
    np.testing.assert_allclose(vector, truth, atol=1e-9)
    np.testing.assert_allclose(fan_vector, truth, atol=1e-9)


def test_nonconvex_triangles_are_consistently_wound():
    """Every ear-clipped triangle normal agrees with the polygon normal."""
    points, cells, _ = _single(DART)
    mesh = Mesh(points=points, cells=cells)
    normal_hat = _newell_vector_area(DART)
    normal_hat = normal_hat / np.linalg.norm(normal_hat)
    dots = mesh.cell_normals.to(torch.float64).numpy() @ normal_hat
    assert np.all(dots > 0.0)


def test_nonconvex_boundary_vertex_area_is_corrected():
    """A reflex vertex lying exactly on a candidate ear's edge must block that
    ear instead of slipping through a strict interior test (regression for the
    boundary-vertex over-count, GH #1711).

    The bare fan happens to give the right area here -- its degenerate diagonal
    triangle has zero area -- so this specifically guards the ear-clip path,
    which over-counted to 7.5 before the boundary-inclusive containment fix.
    Random rigid transforms confirm the *relative* tolerance keeps the fix
    robust when the on-edge coincidence is only approximate in floating point.
    """
    points, cells, _ = _single(ON_EDGE_PENTAGON)
    scalar, vector = _areas(points, cells)
    assert scalar == pytest.approx(ON_EDGE_PENTAGON_TRUE_AREA, rel=1e-9)
    np.testing.assert_allclose(vector, _newell_vector_area(ON_EDGE_PENTAGON), atol=1e-9)

    rng = np.random.default_rng(0)
    for _ in range(16):
        rotation, _r = np.linalg.qr(rng.standard_normal((3, 3)))
        transformed = ON_EDGE_PENTAGON @ rotation.T + rng.uniform(-5.0, 5.0, 3)
        points, cells, _ = _single(transformed)
        scalar, _ = _areas(points, cells)
        assert scalar == pytest.approx(ON_EDGE_PENTAGON_TRUE_AREA, rel=1e-9)


def test_mixed_soup_parent_index_and_cell_data():
    """A mixed soup gives k-2 triangles per polygon and broadcasts cell data."""
    polys = [
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),  # triangle
        np.array(
            [[2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [3.0, 1.0, 0.0], [2.0, 1.0, 0.0]]
        ),  # quad
        DART + np.array([5.0, 0.0, 0.0]),  # non-convex
    ]
    points = torch.as_tensor(np.concatenate(polys), dtype=torch.float64)
    counts = [p.shape[0] for p in polys]
    rings, offset = [], 0
    for c in counts:
        rings.append(list(range(offset, offset + c)))
        offset += c

    cells, parent_index = triangulate(points, _adjacency(*rings))
    assert cells.shape[0] == sum(c - 2 for c in counts)  # 1 + 2 + 2 = 5
    assert parent_index.tolist() == [0, 1, 1, 2, 2]

    cell_values = torch.tensor([10.0, 20.0, 30.0])
    assert cell_values[parent_index].tolist() == [10.0, 20.0, 20.0, 30.0, 30.0]
    # The non-convex polygon's block is still force-correct after ear clipping.
    dart_cells = cells[parent_index == 2]
    dart_area, _ = _areas(points, dart_cells)
    assert dart_area == pytest.approx(DART_TRUE_AREA, rel=1e-9)


def test_from_polygons_constructor_dict_cell_data():
    """Mesh.from_polygons triangulates and broadcasts dict cell_data."""
    points = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
    )
    mesh = Mesh.from_polygons(
        points, _adjacency([0, 1, 2, 3]), cell_data={"p": torch.tensor([2.5])}
    )
    assert mesh.n_cells == 2
    assert mesh.n_manifold_dims == 2
    assert mesh.cell_data["p"].tolist() == [2.5, 2.5]


def test_from_polygons_constructor_tensordict_cell_data():
    """Mesh.from_polygons also accepts a TensorDict for cell_data."""
    from tensordict import TensorDict

    points = torch.as_tensor(DART, dtype=torch.float64)
    cell_data = TensorDict({"p": torch.tensor([7.0])}, batch_size=[1])
    mesh = Mesh.from_polygons(points, _adjacency([0, 1, 2, 3]), cell_data=cell_data)
    assert mesh.n_cells == 2
    assert mesh.cell_data["p"].tolist() == [7.0, 7.0]
    assert float(mesh.cell_areas.sum()) == pytest.approx(DART_TRUE_AREA, rel=1e-9)


def test_assume_convex_path_is_fullgraph_compilable():
    """The convex fan path traces under torch.compile with no graph break."""
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [3.0, 1.0, 0.0],
        ]
    )
    polygons = _adjacency([0, 1, 2, 3], [4, 5, 6])

    def fan(pts: torch.Tensor, polys: Adjacency) -> torch.Tensor:
        return triangulate(pts, polys, assume_convex=True)[0]

    compiled = torch.compile(fan, fullgraph=True)
    torch.testing.assert_close(compiled(points, polygons), fan(points, polygons))


def test_2d_input_matches_3d_embedding():
    """2D (n, 2) points are lifted to z = 0 and triangulate identically to 3D."""
    for poly in (CONVEX_PENTAGON, DART):
        pts3d = torch.as_tensor(poly, dtype=torch.float64)
        polys = _adjacency(list(range(poly.shape[0])))
        cells3d, parent3d = triangulate(pts3d, polys)
        cells2d, parent2d = triangulate(pts3d[:, :2].contiguous(), polys)
        assert torch.equal(cells2d, cells3d)
        assert torch.equal(parent2d, parent3d)

    # The non-convex dart is still area-corrected when fed as 2D points (area is
    # measured via the 3D embedding of the same cells to avoid a 2D normal).
    dart3d = torch.as_tensor(DART, dtype=torch.float64)
    cells2d, _ = triangulate(dart3d[:, :2].contiguous(), _adjacency([0, 1, 2, 3]))
    scalar, _ = _areas(dart3d, cells2d)
    assert scalar == pytest.approx(DART_TRUE_AREA, rel=1e-9)


@pytest.mark.parametrize("spatial_dim", [1, 4])
def test_non_2d_or_3d_points_raise(spatial_dim: int):
    """Point coordinates outside D in {2, 3} are rejected with a clear error."""
    points = torch.zeros(4, spatial_dim)
    with pytest.raises(ValueError, match="2-D or 3-D"):
        triangulate(points, _adjacency([0, 1, 2, 3]))


def test_out_of_range_index_raises():
    """A vertex index outside the points array is rejected with a clear error."""
    points = torch.zeros(4, 3)
    with pytest.raises(ValueError, match="reference vertex"):
        triangulate(points, _adjacency([0, 1, 2, 9]))  # index 9 >= 4 points


def test_negative_index_raises():
    """A negative vertex index is rejected; it would otherwise silently wrap
    around the points array (PyTorch gather semantics) and triangulate the
    wrong vertex instead of raising."""
    points = torch.zeros(4, 3)
    with pytest.raises(ValueError, match="non-negative"):
        triangulate(points, _adjacency([0, 1, 2, -1]))


def test_assume_convex_matches_default_on_convex_input():
    """On a convex polygon, the assume_convex fast path equals the default path."""
    points = torch.as_tensor(CONVEX_PENTAGON, dtype=torch.float64)
    polys = _adjacency(list(range(CONVEX_PENTAGON.shape[0])))
    cells_default, parent_default = triangulate(points, polys)
    cells_fast, parent_fast = triangulate(points, polys, assume_convex=True)
    assert torch.equal(cells_fast, cells_default)
    assert torch.equal(parent_fast, parent_default)


def test_nonconvex_area_correct_in_float32():
    """The non-convex area correction holds in float32, not just float64."""
    points = torch.as_tensor(DART, dtype=torch.float32)
    cells, _ = triangulate(points, _adjacency([0, 1, 2, 3]))
    scalar, _ = _areas(points, cells)  # _areas upcasts to float64 internally
    assert scalar == pytest.approx(DART_TRUE_AREA, rel=1e-6)


def test_cells_are_int64_even_with_int32_indices():
    """cells is always int64 (Mesh's expected dtype), regardless of index dtype."""
    points = torch.as_tensor(DART, dtype=torch.float64)  # dart -> ear-clip path
    polys = Adjacency(
        offsets=torch.tensor([0, 4], dtype=torch.int32),
        indices=torch.tensor([0, 1, 2, 3], dtype=torch.int32),
    )
    cells, parent_index = triangulate(points, polys)
    assert cells.dtype == torch.long
    assert parent_index.dtype == torch.long


class TestAgainstPyvista:
    """Cross-check against VTK's triangulation (the reference implementation)."""

    @staticmethod
    def _pyvista_area(points: np.ndarray) -> float:
        pv = pytest.importorskip("pyvista")
        from physicsnemo.mesh.io import from_pyvista

        k = points.shape[0]
        mesh = from_pyvista(
            pv.PolyData(points, np.hstack([[k, *range(k)]])), manifold_dim=2
        )
        return float(mesh.cell_areas.sum())

    def test_convex_matches_pyvista(self):
        points, cells, _ = _single(CONVEX_PENTAGON)
        scalar, _ = _areas(points, cells)
        assert scalar == pytest.approx(self._pyvista_area(CONVEX_PENTAGON), rel=1e-5)

    def test_nonconvex_matches_pyvista(self):
        points, cells, _ = _single(DART)
        scalar, _ = _areas(points, cells)
        assert scalar == pytest.approx(self._pyvista_area(DART), rel=1e-5)

    def test_ear_clip_batch_matches_pyvista(self):
        """A batch of non-convex polygons (same valence) is ear-clipped correctly."""
        # Several darts at different offsets -> one valence-4 group, m > 1.
        rng = np.random.default_rng(0)
        darts = [
            DART + np.array([dx, dy, 0.0]) for dx, dy in rng.uniform(0, 50, (6, 2))
        ]
        points = torch.as_tensor(np.concatenate(darts), dtype=torch.float64)
        rings = [list(range(4 * i, 4 * i + 4)) for i in range(len(darts))]

        cells, parent_index = triangulate(points, _adjacency(*rings))
        for i, dart in enumerate(darts):
            area, _ = _areas(points, cells[parent_index == i])
            assert area == pytest.approx(self._pyvista_area(dart), rel=1e-5)
