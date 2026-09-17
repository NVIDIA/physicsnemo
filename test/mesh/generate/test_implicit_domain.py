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

"""Tests for implicit-domain mesh generation.

Structure: validity invariants (the structural contract), quality bars
(regression floors from the research program's measured results), boundary
accuracy and the coverage guard, robustness/edge cases, determinism,
differentiable refit, and integration with the Mesh ecosystem.
"""

import math

import pytest
import torch

from physicsnemo.mesh.generate import (
    mesh_implicit_domain,
    refit_mesh_to_implicit,
    sdf_box,
    sdf_difference,
    sdf_sphere,
    sdf_union,
)
from physicsnemo.mesh.generate._simplex_ops import (
    boundary_is_closed_manifold,
    boundary_vertex_mask,
    signed_volumes,
    triangle_angles,
)


def assert_valid_volume_mesh(mesh):
    """The structural contract every generated mesh must satisfy."""
    assert mesh.n_manifold_dims == mesh.n_spatial_dims
    assert mesh.n_points > 0 and mesh.n_cells > 0
    assert mesh.cells.dtype == torch.int64
    assert mesh.cells.min() >= 0
    assert mesh.cells.max() < mesh.n_points
    vol = signed_volumes(mesh.points, mesh.cells)
    assert bool((vol > 0).all()), "inverted or degenerate cell"
    assert boundary_is_closed_manifold(mesh.cells), "boundary not watertight"
    # Every point is referenced (the generator compacts).
    assert torch.unique(mesh.cells.reshape(-1)).shape[0] == mesh.n_points


DISK_2D = sdf_sphere([0.0, 0.0], 0.7)
SHELL_3D = sdf_difference(sdf_sphere([0.0] * 3, 0.7), sdf_sphere([0.0] * 3, 0.3))


# ---------------------------------------------------------------------------
# Validity across dimensions, devices, and reconnection tiers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("reconnect", ["flips", "none"])
@pytest.mark.parametrize(
    "phi,bounds,h",
    [
        (DISK_2D, ([-1, -1], [1, 1]), 0.1),
        (SHELL_3D, ([-1] * 3, [1] * 3), 0.15),
    ],
    ids=["disk2d", "shell3d"],
)
def test_valid_mesh_all_tiers(phi, bounds, h, reconnect):
    # Validity is structural (gated updates); 25 iterations exercises every
    # code path without paying full-convergence cost in CI.
    mesh = mesh_implicit_domain(phi, bounds, h, reconnect=reconnect, iters=25)
    assert_valid_volume_mesh(mesh)


def test_valid_mesh_on_device(device):
    mesh = mesh_implicit_domain(DISK_2D, ([-1, -1], [1, 1]), 0.1, device=device)
    assert mesh.points.device.type == torch.device(device).type
    assert_valid_volume_mesh(mesh.to("cpu"))


@pytest.mark.slow
def test_valid_mesh_4d():
    mesh = mesh_implicit_domain(sdf_sphere([0.0] * 4, 0.7), ([-1] * 4, [1] * 4), 0.25)
    assert mesh.n_spatial_dims == 4
    assert_valid_volume_mesh(mesh)


def test_dtype_policy_and_override():
    mesh = mesh_implicit_domain(DISK_2D, ([-1, -1], [1, 1]), 0.15)
    assert mesh.points.dtype == torch.float64  # CPU default
    mesh32 = mesh_implicit_domain(
        DISK_2D, ([-1, -1], [1, 1]), 0.15, dtype=torch.float32
    )
    assert mesh32.points.dtype == torch.float32
    assert_valid_volume_mesh(mesh32)


# ---------------------------------------------------------------------------
# Quality bars (regression floors from the research program)
# ---------------------------------------------------------------------------


def test_2d_quality_floor():
    mesh, diag = mesh_implicit_domain(
        DISK_2D, ([-1, -1], [1, 1]), 0.06, full_output=True
    )
    angles = triangle_angles(mesh.points, mesh.cells)
    # Research program measured ~33 deg minimum on disks; 25 is the
    # regression floor, not the target.
    assert float(angles.min()) > 25.0
    assert diag["q_median"] > 0.85
    assert diag["sliver_fraction"] == 0.0


def test_3d_quality_floor():
    # h=0.1 gives ~4.5 cells across the shell thickness; coarser h makes
    # the geometry genuinely harder (measured 7.9% slivers at h=0.12), so
    # this test pays its 2 minutes to guard the headline quality claim.
    _, diag = mesh_implicit_domain(SHELL_3D, ([-1] * 3, [1] * 3), 0.1, full_output=True)
    # Measured 0.5% slivers at finer h on this geometry; 5% is the floor.
    assert diag["sliver_fraction"] < 0.05
    assert diag["q_median"] > 0.6


BITTEN = sdf_difference(sdf_sphere([0.0, 0.0], 0.7), sdf_box([0.2, -0.4], [1.0, 0.4]))
# The bite's corners: reentrant at x=0.2, convex tips where the bite walls
# meet the circle at x = sqrt(0.7^2 - 0.4^2).
_XTIP = math.sqrt(0.7**2 - 0.4**2)
BITTEN_CORNERS = torch.tensor(
    [[0.2, -0.4], [0.2, 0.4], [_XTIP, 0.4], [_XTIP, -0.4]],
    dtype=torch.float64,
)


def test_concave_domain_quality():
    """Concave features force topology changes; flips must deliver them.

    Disk with an inscribed square hole: four reentrant corners (concave)
    but no acute wedge tips (those are a resolution problem, tested in
    test_acute_wedge_tips_pinned_and_flagged).
    """
    holed = sdf_difference(
        sdf_sphere([0.0, 0.0], 0.7), sdf_box([-0.3, -0.3], [0.3, 0.3])
    )
    mesh, diag = mesh_implicit_domain(holed, ([-1, -1], [1, 1]), 0.06, full_output=True)
    assert_valid_volume_mesh(mesh)
    assert diag["flips"] > 0, "flips should fire on a concave domain"
    angles = triangle_angles(mesh.points, mesh.cells)
    # Assert the DISTRIBUTION, not the worst sample: on this concave
    # domain the absolute minimum angle was measured to swing 1.8-9.5 deg
    # with the flip RNG stream alone, while p1 is stable (~25 deg). The
    # variational mesher guarantees validity, not a worst-angle bound
    # (see the book's honesty chapter); validity is asserted above.
    assert float(angles.quantile(0.01)) > 20.0
    assert float(angles.quantile(0.05)) > 30.0


def test_corner_pinning_with_feature_points():
    """feature_points must interpolate corners exactly and close coverage.

    Domain: a pentagon (obtuse corners -- the case pinning can fully fix;
    acute wedge tips need h refinement instead, tested separately).
    """
    from physicsnemo.mesh.generate import sdf_polygon_2d

    ang = [2 * math.pi * k / 5 + math.pi / 2 for k in range(5)]
    corners = torch.tensor(
        [[0.7 * math.cos(a), 0.7 * math.sin(a)] for a in ang],
        dtype=torch.float64,
    )
    phi = sdf_polygon_2d(corners)
    mesh, diag = mesh_implicit_domain(
        phi,
        ([-1, -1], [1, 1]),
        0.08,
        feature_points=corners,
        full_output=True,
    )
    assert_valid_volume_mesh(mesh)
    assert diag["coverage_gap_h"] < 1.0
    d = (corners[:, None, :] - mesh.points[None, :, :]).norm(dim=-1).min(dim=1).values
    assert float(d.max()) < 1e-12, "feature points must appear as vertices"
    # And pinning genuinely helps: without it the corners are rounded.
    _, diag_plain = mesh_implicit_domain(
        phi,
        ([-1, -1], [1, 1]),
        0.08,
        max_coverage_gap_h=None,
        full_output=True,
    )
    assert diag["coverage_gap_h"] < diag_plain["coverage_gap_h"]


def test_acute_wedge_tips_pinned_and_flagged():
    """Acute wedge tips round off by ~2h without features -- the coverage
    guard must flag it -- and insertion-based pinning closes the gap."""
    _, diag_plain = mesh_implicit_domain(
        BITTEN,
        ([-1, -1], [1, 1]),
        0.06,
        max_coverage_gap_h=None,
        full_output=True,
    )
    assert diag_plain["coverage_gap_h"] > 1.5  # rounded tips, flagged
    _, diag_pinned = mesh_implicit_domain(
        BITTEN,
        ([-1, -1], [1, 1]),
        0.06,
        feature_points=BITTEN_CORNERS,
        max_coverage_gap_h=None,
        full_output=True,
    )
    assert diag_pinned["coverage_gap_h"] < 1.0


# ---------------------------------------------------------------------------
# Boundary accuracy and the coverage guard
# ---------------------------------------------------------------------------


def test_boundary_on_zero_set():
    mesh = mesh_implicit_domain(DISK_2D, ([-1, -1], [1, 1]), 0.08)
    bnd = boundary_vertex_mask(mesh.points, mesh.cells)
    assert bool(bnd.any())
    # Boundary vertices sit on phi=0 to well below h.
    assert float(DISK_2D(mesh.points[bnd]).abs().max()) < 0.05 * 0.08
    # Interior vertices are strictly inside.
    assert float(DISK_2D(mesh.points[~bnd]).max()) < 0.0


def test_volume_converges_to_analytic():
    mesh = mesh_implicit_domain(DISK_2D, ([-1, -1], [1, 1]), 0.05)
    vol = float(signed_volumes(mesh.points, mesh.cells).sum())
    exact = math.pi * 0.7**2
    # Chordal deficit is O(h^2); at h=0.05 on r=0.7 that's well under 1%.
    assert abs(vol - exact) / exact < 0.01


def test_coverage_guard_trips_on_sub_h_features():
    """A domain with features far below h must raise, not silently drop."""
    # Two disks joined by a channel much thinner than h; the channel is
    # long enough that its midpoint is several h from either disk.
    thin = sdf_union(
        sdf_sphere([-0.65, 0.0], 0.25),
        sdf_sphere([0.65, 0.0], 0.25),
        sdf_box([-0.65, -0.01], [0.65, 0.01]),
    )
    with pytest.raises(ValueError, match="coverage guard"):
        mesh_implicit_domain(thin, ([-1, -0.6], [1, 0.6]), 0.15)
    # ... and the loss is accepted explicitly with the guard disabled.
    mesh = mesh_implicit_domain(
        thin, ([-1, -0.6], [1, 0.6]), 0.15, max_coverage_gap_h=None
    )
    assert_valid_volume_mesh(mesh)


def test_coverage_reported_in_diagnostics():
    _, diag = mesh_implicit_domain(DISK_2D, ([-1, -1], [1, 1]), 0.1, full_output=True)
    assert diag["coverage_gap_h"] < 1.0


def _nan_field(x):
    """Disk SDF that is NaN for x > 0.2 -- the canonical out-of-range
    neural-field behavior (found by adversarial review: the guard used to
    drop the NaN probes and bless the half-missing disk)."""
    sdf = x.norm(dim=-1) - 0.7
    return torch.where(x[..., 0] > 0.2, torch.full_like(sdf, float("nan")), sdf)


def test_nan_phi_trips_guard():
    """Non-finite phi inside bounds must fail certification, not report
    the best possible coverage of whatever region stayed finite."""
    with pytest.raises(ValueError, match="could not be certified"):
        mesh_implicit_domain(_nan_field, ([-1, -1], [1, 1]), 0.1)


def test_nan_phi_best_effort_mesh_is_valid_and_finite():
    """With the guard disabled, the finite region meshes best-effort; the
    validity gate must treat NaN target volumes as bad, so no NaN
    coordinate can replace a valid vertex."""
    mesh, diag = mesh_implicit_domain(
        _nan_field, ([-1, -1], [1, 1]), 0.1, max_coverage_gap_h=None, full_output=True
    )
    assert_valid_volume_mesh(mesh)
    assert bool(torch.isfinite(mesh.points).all())
    assert math.isinf(diag["coverage_gap_h"])  # reported honestly


def test_step_phi_trips_guard():
    """A sign change with no autograd gradient (step function) cannot be
    certified; the no-gradient path must fail, not report perfect
    coverage for a staircase boundary."""

    def step(x):
        inside = x.norm(dim=-1) < 0.7
        return torch.where(
            inside, -torch.ones_like(x[..., 0]), torch.ones_like(x[..., 0])
        )

    with pytest.raises(ValueError, match="could not be certified"):
        mesh_implicit_domain(step, ([-1, -1], [1, 1]), 0.1)
    mesh = mesh_implicit_domain(step, ([-1, -1], [1, 1]), 0.1, max_coverage_gap_h=None)
    assert_valid_volume_mesh(mesh)
    vol = float(signed_volumes(mesh.points, mesh.cells).sum())
    assert abs(vol - math.pi * 0.49) < 0.2  # staircase disk, roughly right


def test_volume_cross_check_catches_covered_but_hollow_mesh():
    """The zero-set gap alone is blind to a mesh whose boundary covers the
    zero set but whose interior is missing; the Monte-Carlo volume
    cross-check must catch that."""
    from physicsnemo.mesh.generate.implicit_domain import _coverage_gap

    mesh = mesh_implicit_domain(DISK_2D, ([-1, -1], [1, 1]), 0.08)
    lo = torch.tensor([-1.0, -1.0], dtype=torch.float64)
    hi = torch.tensor([1.0, 1.0], dtype=torch.float64)
    full = _coverage_gap(DISK_2D, mesh.points, mesh.cells, (lo, hi), 0.08)
    assert full < 1.0
    # Hollow out the core: every zero-set probe still lands on covered
    # boundary, but a third of the domain is gone.
    ring = mesh.points[mesh.cells].mean(dim=1).norm(dim=-1) > 0.4
    hollow = _coverage_gap(DISK_2D, mesh.points, mesh.cells[ring], (lo, hi), 0.08)
    assert math.isinf(hollow)


# ---------------------------------------------------------------------------
# Robustness and edge cases
# ---------------------------------------------------------------------------


def test_level_set_input_not_a_distance():
    """Any level-set function with a usable gradient must work."""

    def blob(x):  # gradient magnitude far from 1
        r = x.norm(dim=-1)
        th = torch.atan2(x[..., 1], x[..., 0])
        return (r - (0.5 + 0.15 * torch.cos(5.0 * th))) * 3.7

    mesh = mesh_implicit_domain(blob, ([-0.8] * 2, [0.8] * 2), 0.06)
    assert_valid_volume_mesh(mesh)


def test_empty_domain_raises():
    with pytest.raises(ValueError, match="no lattice cell"):
        mesh_implicit_domain(sdf_sphere([0.0, 0.0], 0.01), ([-1, -1], [1, 1]), 0.5)


def test_domain_outside_bounds_raises():
    with pytest.raises(ValueError, match="no lattice cell"):
        mesh_implicit_domain(sdf_sphere([5.0, 5.0], 0.5), ([-1, -1], [1, 1]), 0.1)


def test_invalid_arguments_raise():
    with pytest.raises(ValueError, match="hi > lo"):
        mesh_implicit_domain(DISK_2D, ([1, 1], [-1, -1]), 0.1)
    with pytest.raises(ValueError, match="h must be positive"):
        mesh_implicit_domain(DISK_2D, ([-1, -1], [1, 1]), -0.1)
    with pytest.raises(ValueError, match="reconnect"):
        mesh_implicit_domain(DISK_2D, ([-1, -1], [1, 1]), 0.1, reconnect="qhull")
    with pytest.raises(ValueError, match="bounds"):
        mesh_implicit_domain(DISK_2D, ([[-1, -1]], [[1, 1]]), 0.1)


def test_sharp_corner_domain():
    """Corners (gradient kinks) must not produce NaN or invalid cells."""
    lshape = sdf_union(
        sdf_box([-0.8, -0.8], [0.1, 0.1]), sdf_box([-0.1, -0.1], [0.8, 0.8])
    )
    mesh = mesh_implicit_domain(lshape, ([-1, -1], [1, 1]), 0.08)
    assert_valid_volume_mesh(mesh)
    assert bool(torch.isfinite(mesh.points).all())


def test_disconnected_domain():
    two = sdf_union(sdf_sphere([-0.5, 0.0], 0.3), sdf_sphere([0.5, 0.0], 0.3))
    mesh = mesh_implicit_domain(two, ([-1, -0.6], [1, 0.6]), 0.07)
    assert_valid_volume_mesh(mesh)
    vol = float(signed_volumes(mesh.points, mesh.cells).sum())
    assert abs(vol - 2 * math.pi * 0.3**2) / (2 * math.pi * 0.3**2) < 0.03


def test_pinched_domain_produces_manifold_mesh():
    """Near-tangent union pinches the lattice; the split must repair it."""
    tangent = sdf_union(sdf_sphere([-0.35, 0.0], 0.35), sdf_sphere([0.35, 0.0], 0.35))
    mesh, diag = mesh_implicit_domain(
        tangent, ([-0.8, -0.45], [0.8, 0.45]), 0.09, full_output=True
    )
    assert_valid_volume_mesh(mesh)


def test_determinism_cpu():
    kwargs = dict(reconnect="flips", iters=25, seed=3)
    a = mesh_implicit_domain(DISK_2D, ([-1, -1], [1, 1]), 0.1, **kwargs)
    b = mesh_implicit_domain(DISK_2D, ([-1, -1], [1, 1]), 0.1, **kwargs)
    assert torch.equal(a.points, b.points)
    assert torch.equal(a.cells, b.cells)


def test_iteration_budget_is_respected():
    _, diag = mesh_implicit_domain(
        DISK_2D, ([-1, -1], [1, 1]), 0.1, iters=7, full_output=True
    )
    assert diag["iters_run"] <= 7


# ---------------------------------------------------------------------------
# Differentiable refit
# ---------------------------------------------------------------------------


def test_refit_gradient_matches_fd():
    base = mesh_implicit_domain(DISK_2D, ([-1, -1], [1, 1]), 0.08)
    r0 = 0.7

    def volume_at(r_t):
        refit = refit_mesh_to_implicit(base, lambda x: x.norm(dim=-1) - r_t)
        return signed_volumes(refit.points, refit.cells).sum()

    r = torch.tensor(r0, dtype=torch.float64, requires_grad=True)
    (grad_autograd,) = torch.autograd.grad(volume_at(r), r)
    eps = 1e-6
    with torch.no_grad():
        grad_fd = (
            volume_at(torch.tensor(r0 + eps, dtype=torch.float64))
            - volume_at(torch.tensor(r0 - eps, dtype=torch.float64))
        ) / (2 * eps)
    assert torch.isclose(grad_autograd, grad_fd, rtol=1e-6)
    # And the gradient is physically right: dV/dr = perimeter ~ 2 pi r.
    assert abs(float(grad_autograd) - 2 * math.pi * r0) / (2 * math.pi * r0) < 0.01


def test_refit_preserves_topology_and_validity():
    base = mesh_implicit_domain(DISK_2D, ([-1, -1], [1, 1]), 0.1)
    refit = refit_mesh_to_implicit(base, sdf_sphere([0.0, 0.0], 0.72))
    assert torch.equal(refit.cells, base.cells)
    assert bool((signed_volumes(refit.points, refit.cells) > 0).all())
    bnd = boundary_vertex_mask(refit.points, refit.cells)
    phi = sdf_sphere([0.0, 0.0], 0.72)
    assert float(phi(refit.points[bnd]).abs().max()) < 1e-3


# ---------------------------------------------------------------------------
# Integration with the Mesh ecosystem
# ---------------------------------------------------------------------------


def test_mesh_ecosystem_integration():
    mesh = mesh_implicit_domain(DISK_2D, ([-1, -1], [1, 1]), 0.1)
    # The library's own validator and quality metrics accept the output.
    assert isinstance(mesh.validate(), dict | object)
    metrics = mesh.quality_metrics
    assert "min_angle" in metrics or len(metrics.keys()) > 0


def test_diagnostics_contract():
    _, diag = mesh_implicit_domain(DISK_2D, ([-1, -1], [1, 1]), 0.12, full_output=True)
    for key in (
        "n_points",
        "n_cells",
        "all_volumes_positive",
        "boundary_closed_manifold",
        "q_min",
        "q_p01",
        "q_median",
        "sliver_fraction",
        "coverage_gap_h",
        "iters_run",
        "flips",
        "peeled",
        "pinch_splits",
        "time_optimize_s",
    ):
        assert key in diag, f"missing diagnostic: {key}"


def test_scaled_level_set_matches_sdf():
    """phi = c * sdf must behave like the sdf for any c (grad-normalized)."""
    for c in (1e-3, 1.0, 1e3):
        phi = lambda x, c=c: (x.norm(dim=-1) - 0.7) * c  # noqa: E731
        mesh = mesh_implicit_domain(phi, ([-1, -1], [1, 1]), 0.1)
        assert_valid_volume_mesh(mesh)
        vol = float(signed_volumes(mesh.points, mesh.cells).sum())
        assert abs(vol - math.pi * 0.49) / (math.pi * 0.49) < 0.02


def test_feature_point_on_interior_facet():
    """A feature landing exactly on a cell facet must split, not tent:
    tents over interior facets overlap existing cells (undetectable by
    volume or manifold diagnostics)."""
    mesh, diag = mesh_implicit_domain(
        sdf_box([-0.5, -0.5], [0.5, 0.5]),
        ([-0.7, -0.7], [0.7, 0.7]),
        0.35,
        feature_points=torch.tensor([[0.0, 0.0]], dtype=torch.float64),
        max_coverage_gap_h=None,
        full_output=True,
    )
    assert_valid_volume_mesh(mesh)
    d = mesh.points.norm(dim=-1).min()  # direct, not cdist (fp residue)
    assert float(d) < 1e-12
    # Coverage: total volume must not exceed the true square area (an
    # overlapping tent inflates it).
    vol = float(signed_volumes(mesh.points, mesh.cells).sum())
    assert vol <= 1.0 + 1e-9


def test_coverage_guard_survives_quantized_phi():
    """A noisy/quantized level set must not silently disable the guard."""
    q = 2e-5

    def phi(x):  # simulate float-quantized neural-field output
        raw = sdf_union(
            sdf_sphere([-0.65, 0.0], 0.25),
            sdf_sphere([0.65, 0.0], 0.25),
            sdf_box([-0.65, -0.01], [0.65, 0.01]),
        )(x)
        return torch.round(raw / q) * q

    with pytest.raises(ValueError, match="coverage guard"):
        mesh_implicit_domain(phi, ([-1, -0.6], [1, 0.6]), 0.15)


def test_constant_phi_fills_the_box():
    """phi with no dependence on x (e.g. a constant field) is a reasonable
    'domain fills the box' input and must not crash autograd (adversarial
    hardening round)."""
    mesh, diag = mesh_implicit_domain(
        lambda x: -torch.ones(x.shape[0], dtype=x.dtype),
        ([-1, -1], [1, 1]),
        0.25,
        full_output=True,
    )
    assert_valid_volume_mesh(mesh)
    assert diag["coverage_gap_h"] == 0.0
    vol = float(signed_volumes(mesh.points, mesh.cells).sum())
    # With no zero set anywhere, there is nothing to project boundary
    # vertices onto; the box is covered up to the erode margin (~1%).
    assert abs(vol - 4.0) / 4.0 < 0.02


def test_domain_clipped_by_bounds():
    """A domain larger than the bounding box meshes the box; the coverage
    guard must not count the out-of-box zero set against it."""
    mesh = mesh_implicit_domain(sdf_sphere([0.0, 0.0], 10.0), ([-1, -1], [1, 1]), 0.25)
    assert_valid_volume_mesh(mesh)
    vol = float(signed_volumes(mesh.points, mesh.cells).sum())
    assert abs(vol - 4.0) < 1e-9


def test_domain_touching_bounds_keeps_faces():
    """A domain clipped by the box must keep its boundary ON the faces.

    Found by adversarial review: face vertices used to Newton-project onto
    phi's interior zero set (the only one the raw field has) and the mesh
    silently detached from the faces -- this half-plane lost 13% of its
    area with the coverage guard blind to it. phi is now clipped by the
    box SDF and face vertices are pinned per-coordinate.
    """
    mesh = mesh_implicit_domain(lambda x: x[..., 1], ([-1, -1], [1, 1]), 0.1)
    assert_valid_volume_mesh(mesh)
    area = float(signed_volumes(mesh.points, mesh.cells).sum())
    assert abs(area - 2.0) < 0.03  # residual: corner rounding at (+-1, 0)
    assert float(mesh.points[:, 1].min()) == -1.0  # exactly on the bottom face
    # Boundary vertices survive on all three touched faces.
    for coord, val in ((0, -1.0), (0, 1.0), (1, -1.0)):
        assert int((mesh.points[:, coord] == val).sum()) >= 3


def test_external_flow_box_minus_obstacle_2d():
    """Box minus obstacle -- the farfield / external-flow case. The box
    supplies the farfield boundary, the obstacle the interior one; the
    box corners must be exact (per-coordinate pinning, not chamfered)."""
    obstacle = sdf_sphere([0.0, 0.0], 0.4)
    mesh, diag = mesh_implicit_domain(
        lambda x: -obstacle(x), ([-1, -1], [1, 1]), 0.08, full_output=True
    )
    assert_valid_volume_mesh(mesh)
    area = float(signed_volumes(mesh.points, mesh.cells).sum())
    expected = 4.0 - math.pi * 0.4**2
    assert abs(area - expected) / expected < 0.01
    assert diag["coverage_gap_h"] < 1.0
    pts = mesh.points
    for coord in (0, 1):
        for val in (-1.0, 1.0):
            assert int((pts[:, coord] == val).sum()) >= 5
    corners = torch.tensor(
        [[i, j] for i in (-1.0, 1.0) for j in (-1.0, 1.0)], dtype=pts.dtype
    )
    assert (
        float(
            (corners[:, None, :] - pts[None, :, :]).norm(dim=-1).min(dim=1).values.max()
        )
        == 0.0
    )


def test_external_flow_box_minus_obstacle_3d():
    """Same as 2D but in 3D, where the box's own EDGES would chamfer at
    the h scale if face vertices could migrate onto a single face."""
    obstacle = sdf_sphere([0.0, 0.0, 0.0], 0.4)
    mesh = mesh_implicit_domain(lambda x: -obstacle(x), ([-1] * 3, [1] * 3), 0.15)
    assert_valid_volume_mesh(mesh)
    vol = float(signed_volumes(mesh.points, mesh.cells).sum())
    expected = 8.0 - 4.0 / 3.0 * math.pi * 0.4**3
    assert abs(vol - expected) / expected < 0.02


def test_refit_with_bounds_keeps_face_vertices():
    """Refitting a clipped-domain mesh must not drag face vertices onto
    the obstacle; passing bounds clips phi exactly as generation did."""
    obstacle = sdf_sphere([0.0, 0.0], 0.4)
    base = mesh_implicit_domain(lambda x: -obstacle(x), ([-1, -1], [1, 1]), 0.1)
    on_face = (base.points.abs() == 1.0).any(dim=1)
    assert int(on_face.sum()) >= 8
    refit = refit_mesh_to_implicit(
        base, lambda x: 0.45 - x.norm(dim=-1), bounds=([-1, -1], [1, 1])
    )
    # Face vertices stay exactly put (zero Newton step on the box branch)...
    assert torch.equal(refit.points[on_face], base.points[on_face])
    # ... while the obstacle boundary moves to the new radius.
    bnd = boundary_vertex_mask(base.points, base.cells)
    r = refit.points[bnd & ~on_face].norm(dim=-1)
    assert float((r - 0.45).abs().max()) < 1e-9


def test_refit_warns_on_inversion():
    """Large shape changes through the ungated differentiable refit must
    warn instead of silently inverting cells."""
    base = mesh_implicit_domain(sdf_sphere([0.0, 0.0], 0.7), ([-1, -1], [1, 1]), 0.1)
    with pytest.warns(UserWarning, match="inverted"):
        refit_mesh_to_implicit(base, sdf_sphere([0.0, 0.0], 0.35))


def test_clustered_feature_points():
    """Five feature points within one h exercise sequential topological
    insertion in asymmetric neighborhoods. Found by adversarial fuzzing:
    the barycentric containing-cell solve used the transposed edge matrix,
    so insertions targeted cells that did not contain the feature and
    corrupted the facet structure (symmetric geometries masked it)."""
    fp = torch.tensor(
        [[0.0, 0.0], [0.02, 0.0], [0.0, 0.02], [-0.02, 0.0], [0.0, -0.02]],
        dtype=torch.float64,
    )
    mesh, diag = mesh_implicit_domain(
        sdf_sphere([0.0, 0.0], 0.7),
        ([-1, -1], [1, 1]),
        0.2,
        feature_points=fp,
        full_output=True,
        max_coverage_gap_h=None,
    )
    assert_valid_volume_mesh(mesh)
    d = (fp[:, None, :] - mesh.points[None, :, :]).norm(dim=-1).min(dim=1).values
    assert float(d.max()) < 1e-12


def test_feature_point_on_lattice_ridge_3d():
    """A 3D feature on a lattice EDGE (two zero barycentric coordinates)
    exercises the general closure-split: the earlier facet-only taxonomy
    rejected it as unresolvable (adversarial fuzzing, round 3)."""
    fp = torch.tensor([[0.1, 0.0, 0.0]], dtype=torch.float64)
    mesh, diag = mesh_implicit_domain(
        sdf_sphere([0.0] * 3, 0.7),
        ([-0.8] * 3, [0.8] * 3),
        0.4,
        feature_points=fp,
        full_output=True,
        max_coverage_gap_h=None,
    )
    assert_valid_volume_mesh(mesh)
    assert float((fp[:, None, :] - mesh.points[None, :, :]).norm(dim=-1).min()) < 1e-12
