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

"""Tests for physicsnemo.mesh.tessellation.delaunay.

Covers the algorithmic guarantees layer by layer: the empty-circumcircle
property of the plain Bowyer-Watson triangulation, conformity of constrained
segment recovery, exterior/hole removal, Ruppert refinement quality bounds,
ODT smoothing invariants (bounds preserved, boundary untouched, deterministic),
bitwise determinism, structural validity (manifoldness, boundary/segment
agreement, Euler characteristic), adversarial geometry (cocircular inputs,
power-of-two scale exactness, far-from-origin domains, near-collinear
boundaries, high-aspect and reentrant domains, many holes), input validation,
and a production-shaped geometry (square cavity with a star hole, the
ns_cavity_star benchmark family's shape) at realistic resolution.
"""

import math

import numpy as np
import pytest
import torch

from physicsnemo.mesh.tessellation import polygon_interior_point
from physicsnemo.mesh.tessellation.delaunay import (
    _delaunay_triangulation,
    delaunay_mesh_2d,
)

### Geometry fixtures ---------------------------------------------------------


def _square_loop(n_per_edge: int, half: float = 1.0) -> np.ndarray:
    """Axis-aligned square [-half, half]^2 with n_per_edge points per side."""
    corners = np.array([[-half, -half], [half, -half], [half, half], [-half, half]])
    points = []
    for i in range(4):
        a, b = corners[i], corners[(i + 1) % 4]
        points.extend(a + t * (b - a) for t in np.arange(n_per_edge) / n_per_edge)
    return np.array(points)


def _star_loop(
    n: int, *, radius: float = 1.0, amplitude: float = 0.3, lobes: int = 5
) -> np.ndarray:
    """Star-deformed circle r(theta) = radius * (1 + amplitude cos(lobes theta))."""
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    r = radius * (1.0 + amplitude * np.cos(lobes * theta))
    return np.stack((r * np.cos(theta), r * np.sin(theta)), axis=1)


def _loop_segments(loops: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Concatenated loop vertices and the closed per-loop segment index pairs."""
    vertices = np.concatenate(loops, axis=0)
    segments = []
    offset = 0
    for loop in loops:
        n = loop.shape[0]
        first = np.arange(n) + offset
        segments.append(np.stack((first, np.roll(first, -1)), axis=1))
        offset += n
    return vertices, np.concatenate(segments, axis=0)


def _points_in_polygon(points: np.ndarray, loop: np.ndarray) -> np.ndarray:
    """Even-odd ray-crossing point-in-polygon test (vectorized, half-open)."""
    x, y = points[:, 0:1], points[:, 1:2]
    ax, ay = loop[:, 0][None], loop[:, 1][None]
    bx = np.roll(loop[:, 0], -1)[None]
    by = np.roll(loop[:, 1], -1)[None]
    straddles = (ay <= y) != (by <= y)
    crosses = x < ax + (y - ay) * (bx - ax) / (by - ay + (ay == by))
    return (straddles & crosses).sum(axis=1) % 2 == 1


def _triangle_geometry(
    points: torch.Tensor, triangles: torch.Tensor
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(signed areas, minimum angles in degrees, centroids) per triangle."""
    p = points.numpy()
    tri = triangles.numpy()
    a, b, c = p[tri[:, 0]], p[tri[:, 1]], p[tri[:, 2]]
    doubled = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (
        c[:, 0] - a[:, 0]
    )

    def angles(u, v):
        cosine = (u * v).sum(1) / (
            np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1)
        )
        return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

    minimum_angle = np.minimum(
        np.minimum(angles(b - a, c - a), angles(a - b, c - b)),
        angles(a - c, b - c),
    )
    return 0.5 * doubled, minimum_angle, (a + b + c) / 3.0


def _point_segment_distances(
    points: np.ndarray, starts: np.ndarray, ends: np.ndarray
) -> np.ndarray:
    """Distance from each point to its nearest segment among (starts, ends)."""
    vector = ends - starts  # (s, 2)
    length2 = np.maximum((vector**2).sum(axis=1), 1.0e-300)
    t = np.clip(
        ((points[:, None, :] - starts[None]) * vector[None]).sum(-1) / length2[None],
        0.0,
        1.0,
    )
    projections = starts[None] + t[..., None] * vector[None]
    return np.linalg.norm(points[:, None, :] - projections, axis=-1).min(axis=1)


def _assert_valid_triangulation(
    points: torch.Tensor,
    triangles: torch.Tensor,
    markers: torch.Tensor,
    segments: torch.Tensor,
    n_holes: int,
):
    """Structural soundness of a delaunay_mesh_2d output, independent of geometry.

    Checks positive orientation, that every vertex is used, edge-manifoldness
    (each directed edge at most once, each undirected edge in at most two
    triangles), that the topological boundary (edges of exactly one triangle)
    is *exactly* the returned constrained segments with the interior on their
    left, and the Euler characteristic V - E + F = 1 - n_holes of a disk with
    n_holes holes.
    """
    n_points = points.shape[0]
    tri = triangles.numpy()
    segs = segments.numpy()
    assert markers.shape == (n_points,)

    assert tri.min() >= 0 and tri.max() < n_points
    assert np.unique(tri).size == n_points, "orphan vertices in output"

    areas, _, _ = _triangle_geometry(points, triangles)
    assert (areas > 0.0).all(), "non-counterclockwise triangle"

    directed = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    directed_keys = directed[:, 0] * n_points + directed[:, 1]
    assert np.unique(directed_keys).size == directed_keys.size, (
        "a directed edge appears twice (overlapping or inverted triangles)"
    )
    undirected_keys = directed.min(axis=1) * n_points + directed.max(axis=1)
    unique_undirected, counts = np.unique(undirected_keys, return_counts=True)
    assert counts.max() <= 2, "an edge is shared by more than two triangles"

    boundary_keys = set(unique_undirected[counts == 1].tolist())
    boundary_directed = {
        (int(u), int(v))
        for (u, v), key in zip(directed, undirected_keys)
        if int(key) in boundary_keys
    }
    assert boundary_directed == {(int(u), int(v)) for u, v in segs}, (
        "constrained segments and one-triangle edges disagree (direction "
        "encodes interior-on-the-left)"
    )

    euler = n_points - unique_undirected.size + tri.shape[0]
    assert euler == 1 - n_holes, f"Euler characteristic {euler} != {1 - n_holes}"


### Plain Delaunay: empty-circumcircle property -------------------------------


@pytest.mark.parametrize("seed,n", [(0, 200), (1, 500), (2, 1000)])
def test_delaunay_property_on_pseudo_random_points(seed, n):
    """No point lies strictly inside any circumcircle (tol 1e-12, unit box)."""
    rng = np.random.default_rng(seed)
    points = rng.uniform(0.0, 1.0, (n, 2))
    triangles = _delaunay_triangulation(points)

    # A triangulation of the convex hull: positive orientation, correct count
    # (Euler: 2n - hull - 2 triangles), and total area equal to the hull's.
    a, b, c = (points[triangles[:, k]] for k in range(3))
    doubled = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (
        c[:, 0] - a[:, 0]
    )
    assert (doubled > 0.0).all()
    scipy_spatial = pytest.importorskip("scipy.spatial")  # not in CI env

    hull = scipy_spatial.ConvexHull(points)
    assert triangles.shape[0] == 2 * n - hull.vertices.shape[0] - 2
    assert 0.5 * doubled.sum() == pytest.approx(hull.volume, rel=1e-12)

    # Empty circumcircle, brute force against every point.
    denominator = 2.0 * doubled
    bl = ((b - a) ** 2).sum(1)
    cl = ((c - a) ** 2).sum(1)
    ux = a[:, 0] + ((c[:, 1] - a[:, 1]) * bl - (b[:, 1] - a[:, 1]) * cl) / denominator
    uy = a[:, 1] + ((b[:, 0] - a[:, 0]) * cl - (c[:, 0] - a[:, 0]) * bl) / denominator
    radius = np.sqrt((ux - a[:, 0]) ** 2 + (uy - a[:, 1]) ** 2)
    distances = np.sqrt(
        (points[:, 0][None] - ux[:, None]) ** 2
        + (points[:, 1][None] - uy[:, None]) ** 2
    )
    violation = (radius[:, None] - distances).max()
    assert violation <= 1.0e-12, f"circumcircle violated by {violation:.3e}"


### Constrained Delaunay: conformity ------------------------------------------


def _assert_segments_conforming(
    loops: list[np.ndarray],
    points: torch.Tensor,
    markers: torch.Tensor,
    boundary_segments: torch.Tensor,
    tolerance: float = 1.0e-9,
):
    """Every input segment is exactly tiled by output subsegments; markers hold."""
    vertices, input_segments = _loop_segments(loops)
    p = points.numpy()
    marks = markers.numpy()
    segs = boundary_segments.numpy()

    # Input vertices come first, bit-identically, and are boundary-marked.
    np.testing.assert_array_equal(p[: vertices.shape[0]], vertices)
    assert (marks[: vertices.shape[0]] == 1).all()

    # Subsegment endpoints are boundary-marked; nothing else is.
    assert (marks[segs.reshape(-1)] == 1).all()
    on_boundary = np.zeros(p.shape[0], dtype=bool)
    on_boundary[segs.reshape(-1)] = True
    assert (marks == on_boundary.astype(np.int64)).all()

    # Assign each subsegment to the input segment its endpoints lie on, and
    # verify the parameter intervals tile [0, 1] for every input segment.
    starts = vertices[input_segments[:, 0]]
    ends = vertices[input_segments[:, 1]]
    scale = np.abs(vertices).max()
    intervals: dict[int, list[tuple[float, float]]] = {
        i: [] for i in range(input_segments.shape[0])
    }
    for u, v in segs:
        midpoint = 0.5 * (p[u] + p[v])
        distances = _point_segment_distances(midpoint[None], starts, ends)
        parent = None
        for candidate in np.argsort(
            np.linalg.norm(0.5 * (starts + ends) - midpoint[None], axis=1)
        )[:8]:
            s, e = starts[candidate], ends[candidate]
            length2 = ((e - s) ** 2).sum()
            tu = ((p[u] - s) @ (e - s)) / length2
            tv = ((p[v] - s) @ (e - s)) / length2
            du = np.linalg.norm(p[u] - (s + tu * (e - s)))
            dv = np.linalg.norm(p[v] - (s + tv * (e - s)))
            if (
                max(du, dv) <= tolerance * scale
                and -1e-9 <= min(tu, tv)
                and max(tu, tv) <= 1.0 + 1e-9
            ):
                parent = int(candidate)
                intervals[parent].append((min(tu, tv), max(tu, tv)))
                break
        assert parent is not None, f"subsegment ({u}, {v}) lies on no input segment"
        assert distances[0] <= tolerance * scale
    for index, spans in intervals.items():
        assert spans, f"input segment {index} has no subsegments"
        spans.sort()
        assert spans[0][0] == pytest.approx(0.0, abs=1e-9)
        assert spans[-1][1] == pytest.approx(1.0, abs=1e-9)
        for (_, hi), (lo, _) in zip(spans[:-1], spans[1:]):
            assert lo == pytest.approx(hi, abs=1e-9), f"gap in segment {index}"


def test_cdt_conformity_without_refinement():
    """Pure CDT (no refinement): the input segments ARE the output segments."""
    loops = [_star_loop(48, amplitude=0.35), _star_loop(24, radius=0.3, lobes=3)]
    points, triangles, markers, segments = delaunay_mesh_2d(
        loops, max_area=None, min_angle_degrees=0.0
    )
    vertices, input_segments = _loop_segments(loops)
    assert points.shape[0] == vertices.shape[0]  # no Steiner points at all
    assert segments.shape[0] == input_segments.shape[0]
    directed = {(int(u), int(v)) for u, v in segments.numpy()}
    for a, b in input_segments:
        assert (a, b) in directed or (b, a) in directed
    _assert_segments_conforming(loops, points, markers, segments)


def test_cdt_conformity_with_refinement():
    """With refinement, every input segment is a chain of output subsegments."""
    loops = [_square_loop(6), _star_loop(16, radius=0.35)]
    points, triangles, markers, segments = delaunay_mesh_2d(
        loops, max_area=0.01, min_angle_degrees=30.0
    )
    assert points.shape[0] > sum(loop.shape[0] for loop in loops)
    assert segments.shape[0] > 6 * 4 + 16  # boundary refinement did split
    _assert_segments_conforming(loops, points, markers, segments)


### Hole and exterior removal --------------------------------------------------


def test_hole_and_exterior_removal():
    """No triangle centroid falls inside a hole or outside the outer loop."""
    outer = _star_loop(96, amplitude=0.4)
    hole_a = _star_loop(32, radius=0.25, lobes=3) + np.array([0.45, 0.0])
    hole_b = _star_loop(16, radius=0.12, amplitude=0.0) - np.array([0.45, 0.0])
    points, triangles, markers, segments = delaunay_mesh_2d(
        [outer, hole_a, hole_b], max_area=0.02
    )
    areas, _, centroids = _triangle_geometry(points, triangles)
    assert (areas > 0.0).all()
    assert _points_in_polygon(centroids, outer).all()
    assert not _points_in_polygon(centroids, hole_a).any()
    assert not _points_in_polygon(centroids, hole_b).any()
    # The mesh area equals the outer area minus the holes (polygon shoelace).

    def shoelace(loop):
        return 0.5 * abs(
            np.sum(loop[:, 0] * np.roll(loop[:, 1], -1))
            - np.sum(loop[:, 1] * np.roll(loop[:, 0], -1))
        )

    expected = shoelace(outer) - shoelace(hole_a) - shoelace(hole_b)
    assert areas.sum() == pytest.approx(expected, rel=1e-12)


### Ruppert refinement bounds ---------------------------------------------------


@pytest.mark.parametrize("min_angle", [20.0, 30.0, 33.0])
def test_refinement_quality_bounds(min_angle):
    """All angles >= bound - 0.5 deg; all areas <= max_area * (1 + 1e-9)."""
    max_area = math.sqrt(3.0) / 4.0 * 0.08**2
    loops = [_star_loop(128, amplitude=0.25), _star_loop(32, radius=0.3, lobes=3)]
    points, triangles, markers, segments = delaunay_mesh_2d(
        loops, max_area=max_area, min_angle_degrees=min_angle
    )
    areas, minimum_angles, _ = _triangle_geometry(points, triangles)
    assert (areas > 0.0).all()
    assert areas.max() <= max_area * (1.0 + 1.0e-9)
    assert minimum_angles.min() >= min_angle - 0.5

    # Every boundary subsegment (and thus every marker-1 vertex) lies on an
    # input segment, including midpoints inserted by encroachment splits.
    vertices, input_segments = _loop_segments(loops)
    boundary_vertices = points.numpy()[markers.numpy() == 1]
    distances = _point_segment_distances(
        boundary_vertices,
        vertices[input_segments[:, 0]],
        vertices[input_segments[:, 1]],
    )
    assert distances.max() <= 1.0e-9
    _assert_segments_conforming(loops, points, markers, segments)


def test_angle_only_refinement_without_area_bound():
    """max_area=None still refines away skinny CDT triangles."""
    loops = [_star_loop(64)]
    points, triangles, markers, segments = delaunay_mesh_2d(loops, max_area=None)
    _, minimum_angles, _ = _triangle_geometry(points, triangles)
    assert minimum_angles.min() >= 30.0 - 0.5


### Determinism -----------------------------------------------------------------


def test_bitwise_determinism():
    """Two identical calls return bitwise-identical tensors."""
    loops = [_star_loop(64, amplitude=0.3), _star_loop(24, radius=0.3)]
    first = delaunay_mesh_2d(loops, max_area=0.005, min_angle_degrees=30.0)
    second = delaunay_mesh_2d(
        [loop.copy() for loop in loops], max_area=0.005, min_angle_degrees=30.0
    )
    for tensor_a, tensor_b in zip(first, second):
        assert tensor_a.dtype == tensor_b.dtype
        assert torch.equal(tensor_a, tensor_b)


### Production-shaped geometry ----------------------------------------------------


def test_square_cavity_with_star_hole_at_production_resolution():
    """The ns_cavity_star shape (square cavity + star hole) meshes at h=0.05."""
    h = 0.05
    max_area = math.sqrt(3.0) / 4.0 * h * h
    outer = _square_loop(40)  # spacing 0.05 on a side of length 2
    hole = _star_loop(128, radius=0.35, amplitude=0.3, lobes=5)
    points, triangles, markers, segments = delaunay_mesh_2d(
        [outer, hole], max_area=max_area, min_angle_degrees=30.0
    )
    areas, minimum_angles, centroids = _triangle_geometry(points, triangles)
    assert (areas > 0.0).all()
    assert areas.max() <= max_area * (1.0 + 1.0e-9)
    assert minimum_angles.min() >= 29.5
    assert _points_in_polygon(centroids, outer).all()
    assert not _points_in_polygon(centroids, hole).any()

    # Sane counts: the domain area over the mean quality-triangle area brackets
    # the triangle count within loose constant factors.
    domain_area = 4.0 - np.pi * 0.35**2 * (1.0 + 0.3**2 / 2.0)
    assert (
        1.0 * domain_area / max_area
        <= triangles.shape[0]
        <= 4.0 * domain_area / max_area
    )
    assert torch.long == triangles.dtype == markers.dtype == segments.dtype
    assert points.dtype == torch.float64


### polygon_interior_point ---------------------------------------------------------


@pytest.mark.parametrize("winding", [1, -1])
def test_polygon_interior_point_star_and_square(winding):
    """The returned point is strictly inside, for both windings."""
    for loop in (
        _square_loop(1)[::winding].copy(),
        _star_loop(48, amplitude=0.45)[::winding].copy(),
        np.array([[0.0, 0.0], [4.0, 2.0], [0.0, 4.0], [1.0, 2.0]])[::winding].copy(),
    ):
        inside = polygon_interior_point(loop)
        assert inside.shape == (2,)
        assert inside.dtype == torch.float64
        assert _points_in_polygon(inside.numpy()[None], loop).all()
        distance = _point_segment_distances(
            inside.numpy()[None], loop, np.roll(loop, -1, axis=0)
        )
        assert distance[0] > 0.0


def test_polygon_interior_point_accepts_torch_input():
    loop = torch.tensor([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]])
    inside = polygon_interior_point(loop)
    assert 0.0 < float(inside[0]) < 2.0
    assert 0.0 < float(inside[1]) < 1.0


### Validation errors ---------------------------------------------------------------


def test_rejects_bad_arguments():
    square = _square_loop(1)
    with pytest.raises(ValueError, match="at least the outer boundary"):
        delaunay_mesh_2d([])
    with pytest.raises(ValueError, match=r"shape \(n >= 3, 2\)"):
        delaunay_mesh_2d([np.zeros((2, 2))])
    with pytest.raises(ValueError, match="non-finite"):
        delaunay_mesh_2d([np.array([[0.0, 0.0], [1.0, np.nan], [1.0, 1.0]])])
    with pytest.raises(ValueError, match="duplicate consecutive"):
        delaunay_mesh_2d([np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0]])])
    with pytest.raises(ValueError, match="duplicate vertex"):
        delaunay_mesh_2d([square, square])
    with pytest.raises(ValueError, match="max_area"):
        delaunay_mesh_2d([square], max_area=0.0)
    with pytest.raises(ValueError, match=r"min_angle_degrees must lie in \[0, 33\]"):
        delaunay_mesh_2d([square], min_angle_degrees=34.0)
    with pytest.raises(ValueError, match="smooth_iterations"):
        delaunay_mesh_2d([square], smooth_iterations=-1)
    with pytest.raises(ValueError, match="not inside the outer boundary"):
        delaunay_mesh_2d(
            [
                square,
                np.array([[0.0, 0.0], [3.0, 0.5], [0.1, 0.9]]),  # pokes outside
            ]
        )
    with pytest.raises(ValueError, match="inside hole loop"):
        delaunay_mesh_2d(
            [
                _square_loop(1, half=2.0),
                _square_loop(1, half=1.0),
                _square_loop(1, half=0.5),  # nested inside the first hole
            ]
        )
    with pytest.raises(ValueError, match="cross"):
        delaunay_mesh_2d(
            [
                _square_loop(1, half=2.0),
                # Two contained holes crossing each other in a plus shape:
                # containment validation passes, segment recovery must object.
                np.array([[-1.0, -0.1], [1.0, -0.1], [1.0, 0.1], [-1.0, 0.1]]),
                np.array([[-0.1, -1.0], [0.1, -1.0], [0.1, 1.0], [-0.1, 1.0]]),
            ]
        )
    with pytest.raises(ValueError, match="degenerate"):
        polygon_interior_point(np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]))


### Structural validity --------------------------------------------------------


@pytest.mark.parametrize(
    "loops_builder,n_holes,kwargs",
    [
        (lambda: [_star_loop(48)], 0, dict(max_area=0.02)),
        (lambda: [_star_loop(48)], 0, dict(max_area=None, min_angle_degrees=0.0)),
        (
            lambda: [_square_loop(10), _star_loop(24, radius=0.3)],
            1,
            dict(max_area=0.02),
        ),
        (
            lambda: [
                _star_loop(96, amplitude=0.4),
                _star_loop(32, radius=0.25, lobes=3) + np.array([0.45, 0.0]),
                _star_loop(16, radius=0.12, amplitude=0.0) - np.array([0.45, 0.0]),
            ],
            2,
            dict(max_area=0.02),
        ),
        (
            lambda: [_square_loop(10), _star_loop(24, radius=0.3)],
            1,
            dict(max_area=0.02, smooth_iterations=3),
        ),
    ],
)
def test_structural_validity(loops_builder, n_holes, kwargs):
    """Manifoldness, boundary/segment agreement, and Euler characteristic."""
    loops = loops_builder()
    points, triangles, markers, segments = delaunay_mesh_2d(loops, **kwargs)
    _assert_valid_triangulation(points, triangles, markers, segments, n_holes)


### Adversarial geometry --------------------------------------------------------


def test_cocircular_regular_polygon():
    """A regular polygon puts every vertex on one circle: incircle ties
    everywhere. Refinement must still terminate with valid quality output."""
    theta = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
    loops = [np.stack((np.cos(theta), np.sin(theta)), axis=1)]
    points, triangles, markers, segments = delaunay_mesh_2d(loops, max_area=0.05)
    _assert_valid_triangulation(points, triangles, markers, segments, 0)
    areas, minimum_angles, _ = _triangle_geometry(points, triangles)
    assert areas.max() <= 0.05 * (1.0 + 1e-9)
    assert minimum_angles.min() >= 29.5
    assert areas.sum() == pytest.approx(16.0 * np.sin(np.pi / 16.0), rel=1e-12)


def test_structured_grid_boundary_cocircularity():
    """A perfect lattice boundary breeds exactly-cocircular quads during
    refinement; the relative-epsilon tie handling must not cycle or corrupt."""
    loops = [_square_loop(16)]  # spacing 0.125, machine-exact lattice points
    max_area = math.sqrt(3.0) / 4.0 * 0.125**2
    points, triangles, markers, segments = delaunay_mesh_2d(loops, max_area=max_area)
    _assert_valid_triangulation(points, triangles, markers, segments, 0)
    areas, minimum_angles, _ = _triangle_geometry(points, triangles)
    assert minimum_angles.min() >= 29.5
    assert areas.sum() == pytest.approx(4.0, rel=1e-12)


def test_power_of_two_scaling_is_bitwise_exact():
    """Unit-box normalization promises scale invariance; for power-of-two
    scale factors every float operation is exact, so connectivity must be
    identical and coordinates must be exact multiples of the base output."""
    loops = [_star_loop(48), _star_loop(16, radius=0.3, lobes=3)]
    base = delaunay_mesh_2d(loops, max_area=0.01)
    for factor in [2.0**-30, 2.0**24]:
        scaled = delaunay_mesh_2d(
            [loop * factor for loop in loops], max_area=0.01 * factor * factor
        )
        assert torch.equal(scaled[1], base[1])
        assert torch.equal(scaled[2], base[2])
        assert torch.equal(scaled[3], base[3])
        assert torch.equal(scaled[0], base[0] * factor)


def test_far_from_origin_translation():
    """A domain 1e6 units from the origin still meets the quality bounds
    (normalization runs the predicates near the unit box regardless)."""
    offset = np.array([1.0e6, -1.0e6])
    loops = [_star_loop(48) + offset, _star_loop(16, radius=0.3, lobes=3) + offset]
    points, triangles, markers, segments = delaunay_mesh_2d(loops, max_area=0.01)
    _assert_valid_triangulation(points, triangles, markers, segments, 1)
    areas, minimum_angles, _ = _triangle_geometry(points, triangles)
    assert areas.max() <= 0.01 * (1.0 + 1e-9)
    assert minimum_angles.min() >= 29.5
    np.testing.assert_array_equal(points.numpy()[:64], np.concatenate(loops, axis=0))


def test_high_aspect_ratio_domain():
    """A 100:1 sliver rectangle refines to a valid quality mesh."""
    loops = [np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 1.0], [0.0, 1.0]])]
    points, triangles, markers, segments = delaunay_mesh_2d(loops, max_area=2.0)
    _assert_valid_triangulation(points, triangles, markers, segments, 0)
    areas, minimum_angles, _ = _triangle_geometry(points, triangles)
    assert minimum_angles.min() >= 29.5
    assert areas.sum() == pytest.approx(100.0, rel=1e-12)


def test_near_collinear_boundary_vertices():
    """Boundary vertices jittered ~1e-12 off a straight line (far below the
    predicate noise floor at unit scale) must not corrupt the triangulation."""
    x = np.linspace(0.0, 1.0, 21)
    jitter = 1.0e-12 * np.sin(np.arange(21.0))
    bottom = np.stack((x, jitter), axis=1)
    top = np.stack((x[::-1], 1.0 + jitter[::-1]), axis=1)
    loops = [np.concatenate([bottom, [[1.0, 0.5]], top, [[0.0, 0.5]]], axis=0)]
    points, triangles, markers, segments = delaunay_mesh_2d(loops, max_area=0.02)
    _assert_valid_triangulation(points, triangles, markers, segments, 0)
    areas, _, _ = _triangle_geometry(points, triangles)
    assert areas.sum() == pytest.approx(1.0, rel=1e-6)


def test_deep_notch_outer_loop():
    """A non-convex outer boundary with a deep reentrant notch."""
    loops = [
        np.array(
            [
                [0.0, 0.0],
                [4.0, 0.0],
                [4.0, 4.0],
                [2.2, 4.0],
                [2.2, 0.8],  # notch: down into the domain...
                [1.8, 0.8],
                [1.8, 4.0],  # ...and back up
                [0.0, 4.0],
            ]
        )
    ]
    points, triangles, markers, segments = delaunay_mesh_2d(loops, max_area=0.2)
    _assert_valid_triangulation(points, triangles, markers, segments, 0)
    areas, minimum_angles, centroids = _triangle_geometry(points, triangles)
    assert minimum_angles.min() >= 29.5
    assert _points_in_polygon(centroids, loops[0]).all()
    assert areas.sum() == pytest.approx(16.0 - 0.4 * 3.2, rel=1e-12)


def test_many_holes_grid():
    """A 3x3 grid of holes exercises parity removal on many components."""
    outer = _square_loop(8, half=2.0)
    holes = [
        _star_loop(12, radius=0.3, amplitude=0.0) + np.array([dx, dy])
        for dx in (-1.2, 0.0, 1.2)
        for dy in (-1.2, 0.0, 1.2)
    ]
    points, triangles, markers, segments = delaunay_mesh_2d(
        [outer, *holes], max_area=0.05
    )
    _assert_valid_triangulation(points, triangles, markers, segments, 9)
    areas, _, centroids = _triangle_geometry(points, triangles)
    for hole in holes:
        assert not _points_in_polygon(centroids, hole).any()
    hole_area = 9 * 0.5 * 12 * 0.3**2 * np.sin(2.0 * np.pi / 12.0)
    assert areas.sum() == pytest.approx(16.0 - hole_area, rel=1e-12)


def test_float32_and_gradient_tensor_inputs():
    """float32 / requires_grad torch inputs convert cleanly to float64."""
    loop = torch.tensor(_star_loop(24), dtype=torch.float32, requires_grad=True)
    points, triangles, markers, segments = delaunay_mesh_2d([loop], max_area=0.05)
    assert points.dtype == torch.float64
    assert not points.requires_grad
    np.testing.assert_array_equal(
        points.numpy()[:24], loop.detach().to(torch.float64).numpy()
    )
    _assert_valid_triangulation(points, triangles, markers, segments, 0)


def test_area_only_refinement():
    """min_angle_degrees=0 leaves angles unconstrained but bounds areas."""
    loops = [_star_loop(48)]
    points, triangles, markers, segments = delaunay_mesh_2d(
        loops, max_area=0.01, min_angle_degrees=0.0
    )
    _assert_valid_triangulation(points, triangles, markers, segments, 0)
    areas, _, _ = _triangle_geometry(points, triangles)
    assert areas.max() <= 0.01 * (1.0 + 1e-9)


### ODT smoothing ---------------------------------------------------------------


def test_smoothing_improves_quality_within_bounds():
    """Smoothing lifts the typical angle, never breaks either bound."""
    max_area = math.sqrt(3.0) / 4.0 * 0.05**2
    loops = [_square_loop(40), _star_loop(128, radius=0.35)]
    base = delaunay_mesh_2d(loops, max_area=max_area, min_angle_degrees=30.0)
    smoothed = delaunay_mesh_2d(
        loops, max_area=max_area, min_angle_degrees=30.0, smooth_iterations=3
    )
    base_areas, base_angles, _ = _triangle_geometry(base[0], base[1])
    areas, angles, _ = _triangle_geometry(smoothed[0], smoothed[1])
    assert angles.min() >= 30.0 - 0.5  # documented bound survives
    assert areas.max() <= max_area * (1.0 + 1e-9)  # area bound survives
    assert angles.mean() >= base_angles.mean() + 1.0  # typical quality up
    _assert_valid_triangulation(smoothed[0], smoothed[1], smoothed[2], smoothed[3], 1)


def test_smoothing_keeps_boundary_bit_identical_and_conforming():
    """Boundary vertices and conformity are untouched by smoothing."""
    loops = [_square_loop(6), _star_loop(16, radius=0.35)]
    n_input = sum(loop.shape[0] for loop in loops)
    smoothed = delaunay_mesh_2d(
        loops, max_area=0.01, min_angle_degrees=30.0, smooth_iterations=5
    )
    np.testing.assert_array_equal(
        smoothed[0].numpy()[:n_input], np.concatenate(loops, axis=0)
    )
    _assert_segments_conforming(loops, smoothed[0], smoothed[2], smoothed[3])


def test_smoothing_is_deterministic():
    """Smoothed runs are bitwise reproducible, like unsmoothed ones."""
    loops = [_star_loop(64, amplitude=0.3), _star_loop(24, radius=0.3)]
    first = delaunay_mesh_2d(loops, max_area=0.005, smooth_iterations=3)
    second = delaunay_mesh_2d(
        [loop.copy() for loop in loops], max_area=0.005, smooth_iterations=3
    )
    for tensor_a, tensor_b in zip(first, second):
        assert torch.equal(tensor_a, tensor_b)


def test_smoothing_is_noop_without_interior_vertices():
    """Pure CDT output has no interior vertices, so smoothing changes nothing."""
    loops = [_star_loop(48, amplitude=0.35)]
    base = delaunay_mesh_2d(loops, max_area=None, min_angle_degrees=0.0)
    smoothed = delaunay_mesh_2d(
        loops, max_area=None, min_angle_degrees=0.0, smooth_iterations=5
    )
    for tensor_a, tensor_b in zip(base, smoothed):
        assert torch.equal(tensor_a, tensor_b)


### polygon_interior_point: harder shapes ----------------------------------------


def test_polygon_interior_point_horseshoe():
    """A horseshoe's vertex-centroid falls in the void; the returned point
    must not."""
    outer_arc = 2.0 * np.stack(
        (
            np.cos(np.linspace(-2.6, 2.6, 40)),
            np.sin(np.linspace(-2.6, 2.6, 40)),
        ),
        axis=1,
    )
    inner_arc = 1.2 * np.stack(
        (
            np.cos(np.linspace(2.6, -2.6, 40)),
            np.sin(np.linspace(2.6, -2.6, 40)),
        ),
        axis=1,
    )
    loop = np.concatenate([outer_arc, inner_arc], axis=0)
    assert not _points_in_polygon(loop.mean(axis=0)[None], loop)[0]
    inside = polygon_interior_point(loop)
    assert _points_in_polygon(inside.numpy()[None], loop)[0]


@pytest.mark.parametrize("seed", range(8))
def test_polygon_interior_point_random_stars(seed):
    """Randomized star-shaped polygons (simple by construction), both windings."""
    rng = np.random.default_rng(seed)
    theta = np.sort(rng.uniform(0.0, 2.0 * np.pi, 30))
    radius = rng.uniform(0.2, 1.0, 30)
    loop = np.stack((radius * np.cos(theta), radius * np.sin(theta)), axis=1)
    if seed % 2:
        loop = loop[::-1].copy()
    inside = polygon_interior_point(loop)
    assert _points_in_polygon(inside.numpy()[None], loop)[0]
    distance = _point_segment_distances(
        inside.numpy()[None], loop, np.roll(loop, -1, axis=0)
    )
    assert distance[0] > 0.0


def test_polygon_interior_point_collinear_runs():
    """Long exactly-collinear vertex runs leave few true ears; still works."""
    loop = _square_loop(16)  # 60 of 64 vertices are collinear mid-edge points
    inside = polygon_interior_point(loop)
    assert _points_in_polygon(inside.numpy()[None], loop)[0]


def _shoelace(loop: torch.Tensor) -> float:
    x, y = loop[:, 0], loop[:, 1]
    return float(0.5 * (x * y.roll(-1) - x.roll(-1) * y).sum().abs())


def _meshed_area(points: torch.Tensor, triangles: torch.Tensor) -> float:
    tp = points[triangles]
    return float(
        (
            0.5
            * (
                (tp[:, 1, 0] - tp[:, 0, 0]) * (tp[:, 2, 1] - tp[:, 0, 1])
                - (tp[:, 2, 0] - tp[:, 0, 0]) * (tp[:, 1, 1] - tp[:, 0, 1])
            )
        )
        .abs()
        .sum()
    )


@pytest.mark.parametrize(
    "loop",
    [
        # Reflex-adjacent hexagon: segment recovery previously raised
        # "could not find a starting wedge" (reported by melo-gonzo).
        [
            [-0.6, -0.3],
            [-0.4, -0.7],
            [-0.1, -0.3],
            [0.1, -1.0],
            [0.3, -0.7],
            [0.5, -0.2],
        ],
        # Simple hexagon where the backward-wedge bug made the pipe walk
        # exit the hull and silently drop ~25% of the domain's area.
        [
            [0.97, 0.00],
            [0.32, 0.09],
            [0.43, 0.26],
            [0.23, 0.44],
            [0.43, -0.83],
            [0.84, -0.24],
        ],
    ],
    ids=["reflex-crash", "silent-coverage-loss"],
)
def test_segment_recovery_on_sparse_reflex_polygons(loop):
    pts = torch.tensor(loop, dtype=torch.float64)
    points, triangles, _, _ = delaunay_mesh_2d(
        [pts], max_area=None, min_angle_degrees=0.0
    )
    assert _meshed_area(points, triangles) == pytest.approx(_shoelace(pts), rel=1e-12)


def test_segment_recovery_random_star_polygons():
    """Property test (suggested by melo-gonzo): sparse random star polygons
    exercise real segment recovery -- densely sampled smooth boundaries make
    recovery a no-op, which is how the wedge-selection bug survived the
    original suite. Pure CDT (no refinement): meshed area must equal the
    shoelace area exactly."""
    g = torch.Generator().manual_seed(0)
    for trial in range(25):
        n = int(torch.randint(5, 15, (1,), generator=g))
        radii = 0.15 + 0.85 * torch.rand(n, generator=g, dtype=torch.float64)
        theta = (
            torch.arange(n, dtype=torch.float64) / n * 2 * math.pi
            + 0.3 * torch.rand(n, generator=g, dtype=torch.float64) / n
        )
        loop = torch.stack([radii * torch.cos(theta), radii * torch.sin(theta)], dim=1)
        points, triangles, _, _ = delaunay_mesh_2d(
            [loop], max_area=None, min_angle_degrees=0.0
        )
        assert _meshed_area(points, triangles) == pytest.approx(
            _shoelace(loop), rel=1e-12
        ), f"area mismatch on trial {trial}"
