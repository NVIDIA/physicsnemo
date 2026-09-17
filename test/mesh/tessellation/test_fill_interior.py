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

"""Tests for the Mesh-native fill_interior entry point.

The Ruppert/CDT engine itself is tested exhaustively in test_delaunay.py;
these tests cover the Mesh-native contract: loop extraction, nesting and
multi-component handling, exact vertex preservation, provenance data,
device/dtype round-trips, and input validation.
"""

import math

import pytest
import torch

from physicsnemo.mesh import Mesh
from physicsnemo.mesh.tessellation import fill_interior


def circle_mesh_parts(radius, n, offset, center=(0.0, 0.0)):
    t = torch.arange(n, dtype=torch.float64) / n * 2 * math.pi
    pts = torch.stack(
        [center[0] + radius * torch.cos(t), center[1] + radius * torch.sin(t)],
        dim=1,
    )
    idx = torch.arange(n)
    edges = torch.stack([idx, (idx + 1) % n], dim=1) + offset
    return pts, edges


def boundary_mesh(*parts, shuffle_edges=False, seed=0):
    pts = torch.cat([p for p, _ in parts])
    edges = torch.cat([e for _, e in parts])
    if shuffle_edges:
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(edges.shape[0], generator=g)
        edges = edges[perm]
        # Also flip a random half of the edges (orientation-free contract).
        flip = torch.rand(edges.shape[0], generator=g) < 0.5
        edges[flip] = edges[flip][:, [1, 0]]
    return Mesh(points=pts, cells=edges)


def min_angle_deg(mesh):
    p = mesh.points[mesh.cells].double()
    worst = 180.0
    for i in range(3):
        u = p[:, (i + 1) % 3] - p[:, i]
        v = p[:, (i + 2) % 3] - p[:, i]
        cos = (u * v).sum(-1) / (u.norm(dim=-1) * v.norm(dim=-1))
        ang = torch.rad2deg(torch.arccos(cos.clamp(-1, 1)))
        worst = min(worst, float(ang.min()))
    return worst


def signed_areas(mesh):
    p = mesh.points[mesh.cells].double()
    e1 = p[:, 1] - p[:, 0]
    e2 = p[:, 2] - p[:, 0]
    return 0.5 * (e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])


def test_disk_basic_contract():
    disk = boundary_mesh(circle_mesh_parts(1.0, 48, 0))
    filled = fill_interior(disk, max_cell_size=0.02)
    # Provenance is opt-in: no keys are claimed in point_data by default.
    assert len(filled.point_data.keys()) == 0
    assert filled.n_spatial_dims == 2 and filled.n_manifold_dims == 2
    # Exact vertex preservation, in input order.
    assert torch.equal(filled.points[:48], disk.points)
    # Positive orientation, guaranteed angle, size bound.
    assert bool((signed_areas(filled) > 0).all())
    assert min_angle_deg(filled) >= 30.0 - 1e-9
    assert float(signed_areas(filled).max()) <= 0.02 * (1 + 1e-12)
    # Area converges to the disk's.
    total = float(signed_areas(filled).sum())
    assert abs(total - math.pi) / math.pi < 0.01


def test_annulus_and_provenance():
    ring = boundary_mesh(circle_mesh_parts(1.0, 48, 0), circle_mesh_parts(0.4, 24, 48))
    filled = fill_interior(ring, max_cell_size=0.02, provenance=True)
    assert torch.equal(filled.points[:72], ring.points)
    total = float(signed_areas(filled).sum())
    exact = math.pi * (1.0 - 0.4**2)
    assert abs(total - exact) / exact < 0.02
    # Provenance: source_point maps inherited vertices back to the input.
    src = filled.point_data["source_point"]
    marker = filled.point_data["boundary_marker"]
    assert torch.equal(src[:72], torch.arange(72))
    assert bool((src[72:] == -1).all())
    assert bool((marker[:72] == 1).all())  # input verts are boundary
    inherited = src >= 0
    assert torch.equal(filled.points[inherited], ring.points[src[inherited]])


def test_edge_order_and_orientation_free():
    """Shuffled, randomly-flipped edges give the same domain."""
    a = boundary_mesh(circle_mesh_parts(1.0, 32, 0))
    b = boundary_mesh(circle_mesh_parts(1.0, 32, 0), shuffle_edges=True)
    fa = fill_interior(a, max_cell_size=0.05)
    fb = fill_interior(b, max_cell_size=0.05)
    assert abs(float(signed_areas(fa).sum()) - float(signed_areas(fb).sum())) < 1e-9


def test_multiple_components():
    two = boundary_mesh(
        circle_mesh_parts(0.5, 24, 0, center=(-1.0, 0.0)),
        circle_mesh_parts(0.5, 24, 24, center=(1.0, 0.0)),
    )
    filled = fill_interior(two, max_cell_size=0.02)
    total = float(signed_areas(filled).sum())
    exact = 2 * math.pi * 0.5**2
    assert abs(total - exact) / exact < 0.02
    assert torch.equal(filled.points[:48], two.points)


def test_island_inside_hole():
    """Nesting depth 2: disk, hole in it, island inside the hole."""
    m = boundary_mesh(
        circle_mesh_parts(1.0, 48, 0),
        circle_mesh_parts(0.6, 32, 48),
        circle_mesh_parts(0.25, 16, 80),
    )
    filled = fill_interior(m, max_cell_size=0.02)
    total = float(signed_areas(filled).sum())
    exact = math.pi * (1.0 - 0.6**2 + 0.25**2)
    assert abs(total - exact) / exact < 0.02


def test_dtype_device_roundtrip():
    disk32 = Mesh(
        points=boundary_mesh(circle_mesh_parts(1.0, 32, 0)).points.float(),
        cells=boundary_mesh(circle_mesh_parts(1.0, 32, 0)).cells,
    )
    filled = fill_interior(disk32, max_cell_size=0.05)
    assert filled.points.dtype == torch.float32
    # float32 -> float64 -> float32 round-trip is exact for the inputs.
    assert torch.equal(filled.points[:32], disk32.points)


def test_unreferenced_points_ignored():
    pts, edges = circle_mesh_parts(1.0, 32, 0)
    pts = torch.cat([pts, torch.tensor([[5.0, 5.0]], dtype=torch.float64)])
    filled = fill_interior(Mesh(points=pts, cells=edges), max_cell_size=0.05)
    assert float(filled.points[:, 0].max()) < 2.0  # stray point dropped


def test_open_curve_raises():
    pts, edges = circle_mesh_parts(1.0, 16, 0)
    with pytest.raises(ValueError, match="closed 1-manifold"):
        fill_interior(Mesh(points=pts, cells=edges[:-1]))


def test_t_junction_raises():
    pts, edges = circle_mesh_parts(1.0, 16, 0)
    pts = torch.cat([pts, torch.tensor([[2.0, 0.0]], dtype=torch.float64)])
    edges = torch.cat([edges, torch.tensor([[0, 16]])])
    with pytest.raises(ValueError, match="closed 1-manifold"):
        fill_interior(Mesh(points=pts, cells=edges))


def test_surface_input_raises_not_implemented():
    tet_surface = Mesh(
        points=torch.tensor(
            [[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
            dtype=torch.float64,
        ),
        cells=torch.tensor([[0, 2, 1], [0, 1, 3], [1, 2, 3], [0, 3, 2]]),
    )
    with pytest.raises(NotImplementedError, match="n=3"):
        fill_interior(tet_surface)


def test_wrong_codimension_raises():
    volume = Mesh(
        points=torch.tensor([[0.0, 0], [1, 0], [0, 1]], dtype=torch.float64),
        cells=torch.tensor([[0, 1, 2]]),
    )
    with pytest.raises(ValueError, match="codimension-one"):
        fill_interior(volume)


def test_determinism():
    m = boundary_mesh(circle_mesh_parts(1.0, 32, 0), circle_mesh_parts(0.4, 16, 32))
    a = fill_interior(m, max_cell_size=0.05)
    b = fill_interior(m, max_cell_size=0.05)
    assert torch.equal(a.points, b.points)
    assert torch.equal(a.cells, b.cells)


def test_smoothing_preserves_contract():
    disk = boundary_mesh(circle_mesh_parts(1.0, 32, 0))
    filled = fill_interior(disk, max_cell_size=0.05, smooth_iterations=3)
    assert torch.equal(filled.points[:32], disk.points)
    assert min_angle_deg(filled) >= 30.0 - 1e-9


def test_empty_boundary_raises():
    empty = Mesh(
        points=torch.zeros(3, 2, dtype=torch.float64),
        cells=torch.zeros(0, 2, dtype=torch.int64),
    )
    with pytest.raises(ValueError, match="no edges"):
        fill_interior(empty)


def test_crossing_loops_across_components_raise():
    """Two crossing rectangles (a plus sign) must not silently double-cover."""

    def rect(lo, hi, offset):
        pts = torch.tensor(
            [[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]],
            dtype=torch.float64,
        )
        i = torch.arange(4)
        return pts, torch.stack([i, (i + 1) % 4], dim=1) + offset

    p1, e1 = rect((-2.0, -0.2), (2.0, 0.2), 0)
    p2, e2 = rect((-0.2, -2.0), (0.2, 2.0), 4)
    plus = Mesh(points=torch.cat([p1, p2]), cells=torch.cat([e1, e2]))
    with pytest.raises(ValueError, match="cross"):
        fill_interior(plus, max_cell_size=0.1)


def test_coincident_duplicate_loops_raise():
    """Duplicated coincident loops leave no even-depth outer: clear error."""
    pts, edges = circle_mesh_parts(1.0, 16, 0)
    pts2, edges2 = circle_mesh_parts(1.0, 16, 16)
    dup = Mesh(points=torch.cat([pts, pts2]), cells=torch.cat([edges, edges2]))
    with pytest.raises(ValueError, match="touch or cross|even containment depth"):
        fill_interior(dup)


def test_device_roundtrip(device):
    """Output lands on the input's device (repo device parametrization)."""
    m = boundary_mesh(circle_mesh_parts(1.0, 24, 0)).to(device)
    filled = fill_interior(m, max_cell_size=0.1, provenance=True)
    assert filled.points.device.type == torch.device(device).type
    assert filled.point_data["boundary_marker"].device.type == (
        torch.device(device).type
    )


def test_boundary_facets_subdivide_input_segments():
    """The output boundary must lie ON the input polyline: every boundary
    edge's endpoints sit on one common input segment (subdivision only,
    never displacement), and total boundary length equals the perimeter."""
    ring = boundary_mesh(circle_mesh_parts(1.0, 24, 0), circle_mesh_parts(0.4, 12, 24))
    filled = fill_interior(ring, max_cell_size=0.02)
    # Output boundary edges = facets appearing once in the edge census.
    tris = filled.cells
    e = torch.cat([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    e, _ = torch.sort(e, dim=1)
    uniq, counts = torch.unique(e, dim=0, return_counts=True)
    bnd_edges = uniq[counts == 1]
    a = ring.points[ring.cells[:, 0]]  # input segment starts (S, 2)
    b = ring.points[ring.cells[:, 1]]
    ab = b - a
    ab2 = (ab * ab).sum(-1)

    def on_segment(q):  # (2,) -> (S,) bool: q lies on input segment s
        t = ((q - a) * ab).sum(-1) / ab2
        proj = a + t[:, None] * ab
        return (t > -1e-12) & (t < 1 + 1e-12) & ((q - proj).norm(dim=-1) < 1e-12)

    total = 0.0
    for u, v in bnd_edges.tolist():
        pu, pv = filled.points[u], filled.points[v]
        common = on_segment(pu) & on_segment(pv)
        assert bool(common.any()), "boundary edge not on any input segment"
        total += float((pu - pv).norm())
    perimeter = float(ab.norm(dim=-1).sum())
    assert abs(total - perimeter) < 1e-9 * perimeter


def test_square_annulus_and_island_nesting():
    """Rectilinear nesting regression (found in review by melo-gonzo): an
    outer square's largest-ear interior point can fall inside its own hole,
    so containment must be probed with boundary vertices, not interior
    points. 6x6 plate, centered 5x5 hole, then a 1x1 island in the hole."""
    points = torch.tensor(
        [
            [0.0, 0.0],
            [6.0, 0.0],
            [6.0, 6.0],
            [0.0, 6.0],
            [0.5, 0.5],
            [5.5, 0.5],
            [5.5, 5.5],
            [0.5, 5.5],
        ],
        dtype=torch.float64,
    )
    cells = torch.tensor(
        [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4]]
    )
    annulus = fill_interior(Mesh(points=points, cells=cells))
    area = float(signed_areas(annulus).sum())
    assert abs(area - (36.0 - 25.0)) < 1e-9  # nothing silently dropped

    island_pts = torch.tensor(
        [[2.5, 2.5], [3.5, 2.5], [3.5, 3.5], [2.5, 3.5]], dtype=torch.float64
    )
    island_cells = torch.tensor([[8, 9], [9, 10], [10, 11], [11, 8]])
    full = fill_interior(
        Mesh(
            points=torch.cat([points, island_pts]),
            cells=torch.cat([cells, island_cells]),
        )
    )
    area = float(signed_areas(full).sum())
    assert abs(area - (36.0 - 25.0 + 1.0)) < 1e-9


def test_sharp_corner_rejected_when_refining():
    """A 20-degree wedge voids Ruppert's termination guarantee: refinement
    chases itself into the apex, emitting thousands of sub-float32-area
    triangles with no error (found in review by melo-gonzo). Sharp corners
    must be rejected loudly when an angle bound is requested -- and still
    mesh fine as a pure CDT with the bound disabled."""
    wedge = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [math.cos(math.radians(20.0)), math.sin(math.radians(20.0))],
        ],
        dtype=torch.float64,
    )
    edges = torch.tensor([[0, 1], [1, 2], [2, 0]])
    m = Mesh(points=wedge, cells=edges)
    with pytest.raises(ValueError, match="60 degrees"):
        fill_interior(m, min_angle_degrees=30.0)
    cdt = fill_interior(m, min_angle_degrees=0.0)
    area = float(signed_areas(cdt).sum())
    exact = 0.5 * math.sin(math.radians(20.0))
    assert abs(area - exact) < 1e-12
