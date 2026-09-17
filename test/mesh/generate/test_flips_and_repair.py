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

"""Unit tests for the flip engine and topological repairs.

These test the *mechanisms* on hand-constructed configurations where the
correct behavior is known exactly -- the integration tests exercise them
inside the full generator.
"""

import math

import torch

from physicsnemo.mesh.generate._flips import flip_until_done
from physicsnemo.mesh.generate._repair import split_pinched_vertices
from physicsnemo.mesh.generate._simplex_ops import (
    _unique_rows,
    boundary_is_closed_manifold,
    signed_volumes,
    volume_length_quality,
)


def sorted_cells(cells):
    s, _ = torch.sort(cells, dim=1)
    return set(map(tuple, s.tolist()))


# ---------------------------------------------------------------------------
# 2D flips (2-2 edge flip)
# ---------------------------------------------------------------------------


def test_flip_improves_skinny_quad():
    """A kite triangulated on its long diagonal must flip to the short one.

    (A rectangle would NOT flip: its two diagonal triangulations are
    congruent, so the quality gain is zero -- itself a property worth
    knowing about the greedy criterion.)
    """
    points = torch.tensor(
        [[0.0, 0.0], [1.0, -0.15], [2.0, 0.0], [1.0, 0.15]],
        dtype=torch.float64,
    )
    # Long-diagonal (0-2) triangulation: two slivers.
    cells = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.int64)
    q_before = volume_length_quality(points, cells).min()
    new_cells, n = flip_until_done(
        points, cells, h=0.2, generator=torch.Generator().manual_seed(0)
    )
    assert n == 1
    q_after = volume_length_quality(points, new_cells).min()
    assert q_after > q_before
    # The flip must produce the short-diagonal triangulation.
    assert sorted_cells(new_cells) == {(0, 1, 3), (1, 2, 3)}


def test_reflex_quad_never_flips():
    """A reflex quad has one valid triangulation; flipping would change
    the covered region and must be rejected."""
    # Vertex 3 is INSIDE triangle (0,1,2): the pair covers a triangle.
    points = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0], [0.5, 0.3]], dtype=torch.float64
    )
    cells = torch.tensor([[0, 1, 3], [1, 2, 3]], dtype=torch.int64)
    new_cells, n = flip_until_done(
        points, cells, h=0.5, generator=torch.Generator().manual_seed(0)
    )
    assert n == 0
    assert sorted_cells(new_cells) == sorted_cells(cells)


def test_flip_reaches_better_diagonal_from_either_start():
    """Every improving 2-2 flip must fire regardless of WHICH diagonal the
    current triangulation uses.

    Nonregression for a target-class identification bug (found by
    adversarial review): in 2D both Radon classes have size k_share, so
    identifying the target by class SIZE resolved the tie to the negative
    SVD sign class -- whose sign is arbitrary -- and proposed the identity
    retriangulation (zero gain, rejected) whenever that class was the
    current diagonal. About half of all improving 2D flips were silently
    missed, deterministically per geometry, so affected quads NEVER
    improved no matter the seed.
    """
    gen = torch.Generator().manual_seed(0)
    exercised = 0
    for trial in range(300):
        g = torch.Generator().manual_seed(1000 + trial)
        # Convex-position quad: sorted angles on a random ellipse.
        ang = torch.sort(
            torch.rand(4, generator=g, dtype=torch.float64) * (2 * math.pi)
        ).values
        ab = 0.3 + torch.rand(2, generator=g, dtype=torch.float64)
        pts = torch.stack([ab[0] * torch.cos(ang), ab[1] * torch.sin(ang)], dim=1)
        tri_a = torch.tensor([[0, 1, 2], [0, 2, 3]])
        tri_b = torch.tensor([[0, 1, 3], [1, 2, 3]])
        if (signed_volumes(pts, tri_a) <= 0).any():
            continue
        if (signed_volumes(pts, tri_b) <= 0).any():
            continue
        qa = float(volume_length_quality(pts, tri_a).min())
        qb = float(volume_length_quality(pts, tri_b).min())
        if abs(qa - qb) < 1e-3:
            continue
        worse, better = (tri_a, tri_b) if qa < qb else (tri_b, tri_a)
        new_cells, n = flip_until_done(
            pts, worse, h=1.0, max_passes=5, generator=gen, q_focus=1.0
        )
        assert n == 1, "improving diagonal flip was not proposed"
        assert sorted_cells(new_cells) == sorted_cells(better)
        exercised += 1
    assert exercised > 150  # the harness must actually exercise flips


def test_flip_conserves_total_volume():
    torch.manual_seed(0)
    # Random convex quads: total area is invariant under any legal flip.
    for trial in range(20):
        g = torch.Generator().manual_seed(trial)
        base = torch.rand(4, 2, generator=g, dtype=torch.float64)
        # Order as a convex-position quad via angular sort around centroid.
        c = base.mean(dim=0)
        ang = torch.atan2(base[:, 1] - c[1], base[:, 0] - c[0])
        quad = base[torch.argsort(ang)]
        cells = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.int64)
        vol0 = signed_volumes(quad, cells).abs().sum()
        new_cells, _ = flip_until_done(
            quad, cells, h=1.0, generator=torch.Generator().manual_seed(0)
        )
        vol1 = signed_volumes(quad, new_cells).abs().sum()
        assert torch.isclose(vol0, vol1, rtol=1e-12)


# ---------------------------------------------------------------------------
# 3D flips (2-3 and 3-2)
# ---------------------------------------------------------------------------


def make_bipyramid():
    """Two tets sharing a face; the 2-3 flip is quality-improving when the
    shared face is large and the apexes are close to it."""
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],  # shared triangle in z=0
            [0.5, 0.35, 0.25],  # apex above
            [0.5, 0.35, -0.25],  # apex below
        ],
        dtype=torch.float64,
    )
    cells = torch.tensor([[0, 1, 2, 3], [0, 2, 1, 4]], dtype=torch.int64)
    return points, cells


def test_2_3_flip_fires_and_conserves():
    points, cells = make_bipyramid()
    assert bool((signed_volumes(points, cells) > 0).all())
    vol0 = signed_volumes(points, cells).abs().sum()
    new_cells, n = flip_until_done(
        points, cells, h=1.0, generator=torch.Generator().manual_seed(0)
    )
    assert n == 1
    assert new_cells.shape[0] == 3  # 2 -> 3
    assert bool((signed_volumes(points, new_cells) > 0).all())
    assert torch.isclose(
        signed_volumes(points, new_cells).abs().sum(), vol0, rtol=1e-12
    )
    # Every new tet contains the 3-4 edge (the new central edge).
    for row in new_cells.tolist():
        assert 3 in row and 4 in row


def test_3_2_flip_reverses():
    """Starting from the 3-tet configuration where the 2-tet one is better,
    the engine must find the 3-2 flip."""
    # Tall bipyramid: apexes far from the shared plane -> 2-tet config is
    # the good one; build the 3-tet config around the central edge (3,4).
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
            [0.5, 0.35, 0.9],
            [0.5, 0.35, -0.9],
        ],
        dtype=torch.float64,
    )
    three = torch.tensor([[0, 1, 3, 4], [1, 2, 3, 4], [2, 0, 3, 4]], dtype=torch.int64)
    from physicsnemo.mesh.generate._simplex_ops import orient_positive

    three = orient_positive(points, three)
    assert bool((signed_volumes(points, three) > 0).all())
    q_before = volume_length_quality(points, three).min()
    new_cells, n = flip_until_done(
        points, three, h=1.0, generator=torch.Generator().manual_seed(0)
    )
    assert n == 1
    assert new_cells.shape[0] == 2  # 3 -> 2
    assert volume_length_quality(points, new_cells).min() > q_before


def test_flip_pass_never_duplicates_cells():
    """Vertex-disjoint winners can never mint identical cells."""
    torch.manual_seed(0)
    from physicsnemo.mesh.generate._lattice import kuhn_lattice

    points, cells = kuhn_lattice([0, 0], [1, 1], 0.25)
    # Jitter interior points to create flip opportunities.
    g = torch.Generator().manual_seed(1)
    jitter = torch.rand(points.shape, generator=g, dtype=points.dtype) - 0.5
    interior = (points > 0.01).all(dim=1) & (points < 0.99).all(dim=1)
    points = points + 0.15 * 0.25 * jitter * interior[:, None]
    new_cells, _ = flip_until_done(
        points,
        cells,
        h=0.25,
        generator=torch.Generator().manual_seed(0),
        q_focus=0.99,
    )
    s, _ = torch.sort(new_cells, dim=1)
    _, _, counts = _unique_rows(s)
    assert int(counts.max()) == 1


# ---------------------------------------------------------------------------
# Pinch splitting
# ---------------------------------------------------------------------------


def test_bowtie_is_split():
    """Two triangles sharing exactly one vertex: the classic pinch."""
    points = torch.tensor(
        [
            [0.0, 0.0],  # the pinch vertex
            [-1.0, 0.5],
            [-1.0, -0.5],
            [1.0, 0.5],
            [1.0, -0.5],
        ],
        dtype=torch.float64,
    )
    cells = torch.tensor([[0, 1, 2], [0, 4, 3]], dtype=torch.int64)
    assert not boundary_is_closed_manifold(cells)
    new_points, new_cells, n_split = split_pinched_vertices(points, cells)
    assert n_split == 1
    assert new_points.shape[0] == 6
    assert boundary_is_closed_manifold(new_cells)
    # The duplicate starts geometrically coincident with the original.
    assert torch.equal(new_points[5], points[0])


def test_clean_mesh_is_untouched():
    from physicsnemo.mesh.generate._lattice import kuhn_lattice

    points, cells = kuhn_lattice([0, 0, 0], [1, 1, 1], 0.5)
    new_points, new_cells, n_split = split_pinched_vertices(points, cells)
    assert n_split == 0
    assert torch.equal(new_points, points)
    assert torch.equal(new_cells, cells)


def test_peel_followed_by_split_repairs_vertex_pinch():
    """Peeling a bridging sliver can expose a boundary pinched at a
    VERTEX, which the ridge-pairing manifoldness check inside the peel
    cannot see (every boundary ridge still pairs cleanly). The pipeline
    therefore follows every peel with a pinch split; this exercises that
    exact sequence on the minimal configuration. Found by adversarial
    audit: two tets sharing one vertex, bridged by a near-flat sliver.
    """
    from physicsnemo.mesh.generate._repair import peel_boundary_slivers

    pts = torch.tensor(
        [
            [0.0, 0.0, 0.0],  # the future pinch vertex
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.05, 0.001],  # near-flat bridge apex
        ],
        dtype=torch.float64,
    )
    cells = torch.tensor([[0, 1, 2, 3], [0, 5, 4, 6], [1, 2, 4, 7]], dtype=torch.int64)
    vol = signed_volumes(pts, cells)
    cells[vol < 0] = cells[vol < 0][:, [1, 0, 2, 3]]

    def phi(x):
        return torch.zeros(x.shape[:-1], dtype=x.dtype)

    p2, c2, n_peeled = peel_boundary_slivers(pts, cells, phi, h=1.0)
    assert n_peeled == 1
    # The peel output is vertex-pinched, yet facet-pairing blesses it...
    assert boundary_is_closed_manifold(c2)
    # ...so the follow-up split must find and repair it.
    p3, c3, n_split = split_pinched_vertices(p2, c2)
    assert n_split == 1
    assert set(c3[0].tolist()).isdisjoint(set(c3[1].tolist()))
