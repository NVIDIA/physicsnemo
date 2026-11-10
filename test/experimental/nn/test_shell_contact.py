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
"""Shell contact functionals: pair search, features, depths, projection, wall clamp."""

import math

import pytest
import torch

from physicsnemo.experimental.nn.shell_contact import (
    CONTACT_FEATURE_DIM,
    KINEMATIC_CONTACT_FEATURE_DIM,
    ShellContactSearch,
    clamp_to_wall,
    contact_edge_features,
    node_maximum,
    project_shell_penetrations,
    shell_contact_edge_features,
    shell_contact_pairs,
    shell_penetration_depths,
    smooth_contact_cutoff,
    uses_fused_narrow_phase,
    wall_penetration_depths,
)

TORCH = dict(implementation="torch")


def _crossing_scene():
    """One triangle in z = 0; nodes 3 and 4 start 3 above it. Now node 3 is 1
    above (same side) and node 4 is 1 below (it crossed the shell)."""
    reference = torch.tensor(
        [[0.0, 0, 0], [10, 0, 0], [0, 10, 0], [2, 2, 3.0], [3, 2, 3.0]]
    )
    current = reference.clone()
    current[3, 2] = 1.0
    current[4, 2] = -1.0
    faces = torch.tensor([[0, 1, 2]])
    return reference, current, faces


def _slide_scene():
    """Node 3 was beside the triangle (closest point on the edge y = 0) and now
    lies under its interior on the other side of the plane."""
    reference = torch.tensor([[0.0, 0, 0], [10, 0, 0], [0, 10, 0], [5, -2, 0.5]])
    current = reference.clone()
    current[3] = torch.tensor([3.0, 3.0, -0.5])
    faces = torch.tensor([[0, 1, 2]])
    return reference, current, faces


def test_search_settings_are_validated_and_coerced():
    search = ShellContactSearch(activation_distance=3, implementation="torch")
    assert search.activation_distance == 3.0 and isinstance(
        search.activation_distance, float
    )
    assert search.search_band == 3.0 and not search.uses_reference
    with pytest.raises(ValueError, match="activation_distance"):
        ShellContactSearch(activation_distance=0.0)
    with pytest.raises(ValueError, match="implementation"):
        ShellContactSearch(implementation="numpy")
    with pytest.raises(ValueError, match="max_pairs"):
        ShellContactSearch(max_pairs=True)
    with pytest.raises(ValueError, match="side_symmetric"):
        ShellContactSearch(side_symmetric=True)
    with pytest.raises(ValueError, match="crossing_radius"):
        ShellContactSearch(side_from_reference=True, crossing_radius=1.0)
    with pytest.raises(ValueError, match="prediction_horizon=0"):
        ShellContactSearch(
            side_from_reference=True, crossing_radius=10.0, prediction_horizon=0.1
        )
    with pytest.raises(ValueError, match="clearance_approach_only"):
        ShellContactSearch(clearance_approach_only=True)
    swept = ShellContactSearch(side_from_reference=True, crossing_radius=12.0)
    assert swept.search_band == 12.0 and swept.uses_reference


def test_pairs_sign_crossed_nodes_against_the_reference():
    reference, current, faces = _crossing_scene()
    thickness = torch.zeros(5)
    plain = shell_contact_pairs(
        current, faces, thickness, search=ShellContactSearch(**TORCH)
    )
    assert plain.query.tolist() == [3, 4] and plain.face.tolist() == [0, 0]
    torch.testing.assert_close(plain.gap, torch.tensor([1.0, 1.0]))
    assert not plain.crossed.any() and plain.stats["crossed_pairs"] == 0
    assert plain.stats["active_pairs"] == 2 and plain.num_pairs == 2
    sided = shell_contact_pairs(
        current,
        faces,
        thickness,
        reference_positions=reference,
        search=ShellContactSearch(side_from_reference=True, **TORCH),
    )
    torch.testing.assert_close(sided.gap, torch.tensor([1.0, -1.0]))
    torch.testing.assert_close(sided.activation_gap, torch.tensor([1.0, -1.0]))
    assert sided.crossed.tolist() == [False, True]
    assert sided.stats["crossed_pairs"] == 1
    # The live geometry is unsigned and identical in both searches.
    torch.testing.assert_close(sided.distance, plain.distance)
    torch.testing.assert_close(sided.closest, torch.tensor([[2.0, 2, 0], [3, 2, 0]]))
    torch.testing.assert_close(sided.barycentric.sum(-1), torch.ones(2))
    with pytest.raises(ValueError, match="reference_positions"):
        shell_contact_pairs(
            current,
            faces,
            thickness,
            search=ShellContactSearch(side_from_reference=True, **TORCH),
        )


def test_features_flip_the_normal_of_crossed_pairs_and_carry_gradients():
    reference, current, faces = _crossing_scene()
    current.requires_grad_()
    thickness = torch.zeros(5)
    pairs = shell_contact_pairs(
        current,
        faces,
        thickness,
        reference_positions=reference,
        search=ShellContactSearch(side_from_reference=True, **TORCH),
    )
    features = shell_contact_edge_features(
        pairs, current, feature_scale=10.0, normal_epsilon=0.1
    )
    assert features.shape == (2, CONTACT_FEATURE_DIM)
    # Channels: displacement (3), distance, signed gap, normal (3).
    torch.testing.assert_close(features[:, 4], torch.tensor([0.1, -0.1]))
    torch.testing.assert_close(features[0, :3], torch.tensor([0.0, 0, -0.1]))
    torch.testing.assert_close(features[1, :3], torch.tensor([0.0, 0, 0.1]))
    expected_normal = 1.0 / math.sqrt(1.0 + 0.1**2)
    torch.testing.assert_close(features[0, 7], torch.tensor(-expected_normal))
    torch.testing.assert_close(features[1, 7], torch.tensor(-expected_normal))
    features.square().sum().backward()
    assert torch.isfinite(current.grad).all() and current.grad.abs().sum() > 0
    velocities = torch.randn(5, 3)
    kinematic = shell_contact_edge_features(
        pairs,
        current.detach(),
        feature_scale=10.0,
        normal_epsilon=0.1,
        velocities=velocities,
        velocity_scale=100.0,
    )
    assert kinematic.shape == (2, KINEMATIC_CONTACT_FEATURE_DIM)
    relative_velocity = (
        velocities[faces[0]] * pairs.barycentric.detach()[0, :3, None]
    ).sum(0) - velocities[3]
    torch.testing.assert_close(kinematic[0, 8:11], relative_velocity / 100.0)


def test_reference_clearance_drops_assembled_pairs():
    reference, current, faces = _crossing_scene()
    reference = reference.clone()
    reference[4, 2] = 30.0  # node 4 was far away: its pair is new contact
    thickness = torch.zeros(5)
    pairs = shell_contact_pairs(
        current,
        faces,
        thickness,
        reference_positions=reference,
        search=ShellContactSearch(
            side_from_reference=True, reference_clearance=5.0, **TORCH
        ),
    )
    assert pairs.query.tolist() == [4]
    assert pairs.stats["reference_excluded_pairs"] == 1
    assert pairs.stats["active_pairs"] == 1


def test_symmetric_side_test_keeps_slid_under_nodes_unsigned():
    reference, current, faces = _slide_scene()
    thickness = torch.zeros(4)
    one_sided = shell_contact_pairs(
        current,
        faces,
        thickness,
        reference_positions=reference,
        search=ShellContactSearch(side_from_reference=True, **TORCH),
    )
    symmetric = shell_contact_pairs(
        current,
        faces,
        thickness,
        reference_positions=reference,
        search=ShellContactSearch(
            side_from_reference=True, side_symmetric=True, **TORCH
        ),
    )
    assert one_sided.crossed.tolist() == [True]
    torch.testing.assert_close(one_sided.gap, torch.tensor([-0.5]))
    assert symmetric.crossed.tolist() == [False]
    torch.testing.assert_close(symmetric.gap, torch.tensor([0.5]))
    assert symmetric.stats["crossed_pairs"] == 0


def test_crossing_radius_discovers_deep_crossings_only():
    reference, current, faces = _crossing_scene()
    current = current.clone()
    current[3, 2] = 7.0  # approaching, beyond the band
    current[4, 2] = -7.0  # crossed, deeper than the band
    thickness = torch.zeros(5)
    banded = shell_contact_pairs(
        current,
        faces,
        thickness,
        reference_positions=reference,
        search=ShellContactSearch(
            activation_distance=5.0, side_from_reference=True, **TORCH
        ),
    )
    assert banded.num_pairs == 0
    swept = shell_contact_pairs(
        current,
        faces,
        thickness,
        reference_positions=reference,
        search=ShellContactSearch(
            activation_distance=5.0,
            side_from_reference=True,
            crossing_radius=10.0,
            **TORCH,
        ),
    )
    assert swept.query.tolist() == [4] and swept.crossed.tolist() == [True]
    torch.testing.assert_close(swept.gap, torch.tensor([-7.0]))
    assert swept.stats["sweep_pairs"] == 1 and swept.stats["crossed_pairs"] == 1


def test_clearance_approach_only_keeps_assembled_crossings():
    reference, current, faces = _crossing_scene()
    thickness = torch.zeros(5)
    strict = ShellContactSearch(
        side_from_reference=True, reference_clearance=5.0, **TORCH
    )
    lenient = ShellContactSearch(
        side_from_reference=True,
        reference_clearance=5.0,
        clearance_approach_only=True,
        **TORCH,
    )
    dropped = shell_contact_pairs(
        current, faces, thickness, reference_positions=reference, search=strict
    )
    kept = shell_contact_pairs(
        current, faces, thickness, reference_positions=reference, search=lenient
    )
    assert dropped.num_pairs == 0 and dropped.stats["reference_excluded_pairs"] == 2
    assert kept.query.tolist() == [4] and kept.crossed.tolist() == [True]
    assert kept.stats["clearance_crossings_kept"] == 1
    assert kept.stats["reference_excluded_pairs"] == 1


def test_exclusions_budget_and_empty_result():
    reference, current, faces = _crossing_scene()
    thickness = torch.zeros(5)
    excluded = torch.tensor([[3], [0]])
    pairs = shell_contact_pairs(
        current,
        faces,
        thickness,
        excluded_pairs=excluded,
        search=ShellContactSearch(**TORCH),
    )
    assert pairs.query.tolist() == [4]
    with pytest.raises(RuntimeError, match="active contact budget"):
        shell_contact_pairs(
            current, faces, thickness, search=ShellContactSearch(max_pairs=1, **TORCH)
        )
    empty = shell_contact_pairs(
        current,
        faces,
        thickness,
        search=ShellContactSearch(activation_distance=0.5, **TORCH),
    )
    assert empty.num_pairs == 0 and empty.gap.shape == (0,)
    assert empty.barycentric.shape == (0, 3) and empty.stats["active_pairs"] == 0
    envelope, crossing = shell_penetration_depths(empty, 5)
    assert torch.equal(envelope, torch.zeros(5)) and torch.equal(
        crossing, torch.zeros(5)
    )


def test_input_validation():
    reference, current, faces = _crossing_scene()
    search = ShellContactSearch(**TORCH)
    with pytest.raises(ValueError, match="thickness"):
        shell_contact_pairs(current, faces, torch.zeros(4), search=search)
    with pytest.raises(ValueError, match="nonnegative"):
        shell_contact_pairs(current, faces, -torch.ones(5), search=search)
    with pytest.raises(ValueError, match="connectivity"):
        shell_contact_pairs(
            current, torch.tensor([[0, 1, 9]]), torch.zeros(5), search=search
        )
    with pytest.raises(ValueError, match="velocities"):
        shell_contact_pairs(
            current,
            faces,
            torch.zeros(5),
            search=ShellContactSearch(prediction_horizon=0.1, **TORCH),
        )


def test_penetration_depths_follow_the_signed_gap():
    reference, current, faces = _crossing_scene()
    current = current.clone()
    current[3, 2] = 0.4  # overlaps the envelope from above
    current[4, 2] = -0.4  # crossed, 0.4 below the mid-surface
    thickness = torch.ones(5)  # half-thickness sum 1.0 for every pair
    pairs = shell_contact_pairs(
        current,
        faces,
        thickness,
        reference_positions=reference,
        search=ShellContactSearch(side_from_reference=True, **TORCH),
    )
    torch.testing.assert_close(pairs.half_thickness, torch.tensor([1.0, 1.0]))
    torch.testing.assert_close(pairs.gap, torch.tensor([-0.6, -1.4]))
    envelope, crossing = shell_penetration_depths(pairs, 5)
    torch.testing.assert_close(envelope, torch.tensor([0.0, 0, 0, 0.6, 1.4]))
    torch.testing.assert_close(crossing, torch.tensor([0.0, 0, 0, 0, 0.4]))
    torch.testing.assert_close(
        node_maximum(3, torch.tensor([1, 1, 2]), torch.tensor([0.5, 2.0, 1.0])),
        torch.tensor([0.0, 2.0, 1.0]),
    )


def test_wall_depths_and_clamp():
    positions = torch.tensor([[95.0, 0, 0], [100.5, 1, 0], [99.8, 2, 0]])
    thickness = torch.ones(3)
    envelope, crossing = wall_penetration_depths(positions, thickness, 100.0)
    torch.testing.assert_close(envelope, torch.tensor([0.0, 1.0, 0.3]))
    torch.testing.assert_close(crossing, torch.tensor([0.0, 0.5, 0.0]))
    clamped = clamp_to_wall(positions, 100.0)
    torch.testing.assert_close(clamped[:, 0], torch.tensor([95.0, 100.0, 99.8]))
    torch.testing.assert_close(clamped[:, 1:], positions[:, 1:])
    assert clamped.data_ptr() != positions.data_ptr()
    other = clamp_to_wall(positions, torch.tensor(96.0), sign=-1.0)
    torch.testing.assert_close(other[:, 0], torch.tensor([96.0, 100.5, 99.8]))
    along_y = clamp_to_wall(positions, 1.5, axis=1)
    torch.testing.assert_close(along_y[:, 1], torch.tensor([0.0, 1.0, 1.5]))
    with pytest.raises(ValueError, match="axis"):
        clamp_to_wall(positions, 1.0, axis=3)
    with pytest.raises(ValueError, match="sign"):
        wall_penetration_depths(positions, thickness, 1.0, sign=0.5)


def test_projection_moves_nodes_onto_the_envelope():
    reference, current, faces = _crossing_scene()
    current = current.clone()
    current[3, 2] = 0.6  # overlap without crossing: gap -0.4
    current[4, 2] = -0.5  # crossed: gap -1.5
    thickness = torch.ones(5)
    search = ShellContactSearch(side_from_reference=True, **TORCH)
    projected = project_shell_penetrations(
        current, faces, thickness, reference, search=search, iterations=1
    )
    # Both nodes end on the envelope (one half-thickness sum above the shell),
    # the crossed node pulled back through the shell to its original side.
    torch.testing.assert_close(projected[3], torch.tensor([2.0, 2.0, 1.0]))
    torch.testing.assert_close(projected[4], torch.tensor([3.0, 2.0, 1.0]))
    torch.testing.assert_close(projected[:3], current[:3])
    assert torch.equal(current[4], torch.tensor([3.0, 2.0, -0.5]))  # input untouched
    damped = project_shell_penetrations(
        current, faces, thickness, reference, search=search, iterations=1, damping=0.5
    )
    torch.testing.assert_close(damped[4, 2], torch.tensor(0.25))
    torch.testing.assert_close(damped[3, 2], torch.tensor(0.8))
    crossed_only = project_shell_penetrations(
        current,
        faces,
        thickness,
        reference,
        search=search,
        iterations=1,
        crossed_only=True,
    )
    torch.testing.assert_close(crossed_only[3], current[3])
    torch.testing.assert_close(crossed_only[4, 2], torch.tensor(1.0))
    tolerant = project_shell_penetrations(
        current, faces, thickness, reference, search=search, iterations=3, tolerance=2.0
    )
    torch.testing.assert_close(tolerant, current)
    assert torch.equal(
        project_shell_penetrations(
            current, faces, thickness, reference, search=search, iterations=0
        ),
        current,
    )
    # A second iteration finds nothing left to move.
    twice = project_shell_penetrations(
        current, faces, thickness, reference, search=search, iterations=2
    )
    torch.testing.assert_close(twice, projected)
    with pytest.raises(ValueError, match="side_from_reference"):
        project_shell_penetrations(
            current,
            faces,
            thickness,
            reference,
            search=ShellContactSearch(**TORCH),
            iterations=1,
        )
    with pytest.raises(ValueError, match="damping"):
        project_shell_penetrations(
            current,
            faces,
            thickness,
            reference,
            search=search,
            iterations=1,
            damping=0.0,
        )


def test_taper_and_feature_helpers():
    gaps = torch.tensor([-1.0, 0.0, 2.5, 5.0, 7.0])
    weights = smooth_contact_cutoff(gaps, 5.0)
    torch.testing.assert_close(weights, torch.tensor([1.0, 1.0, 0.5625, 0.0, 0.0]))
    with pytest.raises(ValueError, match="radius"):
        smooth_contact_cutoff(gaps, 0.0)
    relative = torch.tensor([[0.0, 0, 2.0]])
    features = contact_edge_features(
        relative, torch.tensor([1.0]), 10.0, normal_epsilon=0.1
    )
    assert features.shape == (1, CONTACT_FEATURE_DIM)
    torch.testing.assert_close(features[0, 3], torch.tensor(0.2))
    with pytest.raises(ValueError, match="feature_scale"):
        contact_edge_features(relative, torch.tensor([1.0]), 0.0)


def test_narrow_phase_selection():
    cpu = torch.zeros(1, 3)
    assert not uses_fused_narrow_phase(ShellContactSearch(**TORCH), cpu)
    assert not uses_fused_narrow_phase(ShellContactSearch(), cpu)
    assert uses_fused_narrow_phase(ShellContactSearch(narrow_phase="warp"), cpu)
    assert not uses_fused_narrow_phase(
        ShellContactSearch(narrow_phase="torch"), torch.zeros(1, 3, device="meta")
    )


def test_quad_facets_and_batches():
    # Two quads in z = 0 in two batch graphs; one probe node above each.
    plate = torch.tensor([[0.0, 0, 0], [4, 0, 0], [4, 4, 0], [0, 4, 0]])
    shift = torch.tensor([50.0, 50.0, 0.0])
    positions = torch.cat(
        (
            plate,
            torch.tensor([[2.0, 2, 1.0]]),
            plate + shift,
            torch.tensor([[52.0, 52, 2.0]]),
        )
    )
    faces = torch.tensor([[0, 1, 2, 3], [5, 6, 7, 8]])
    batch = torch.tensor([0] * 5 + [1] * 5)
    thickness = torch.zeros(10)
    pairs = shell_contact_pairs(
        positions, faces, thickness, batch=batch, search=ShellContactSearch(**TORCH)
    )
    assert pairs.query.tolist() == [4, 9] and pairs.face.tolist() == [0, 1]
    torch.testing.assert_close(pairs.gap, torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(pairs.barycentric, torch.full((2, 4), 0.25))
    reference = positions.clone()
    reference[4, 2] = reference[9, 2] = 3.0
    material = shell_contact_pairs(
        positions,
        faces,
        thickness,
        batch=batch,
        reference_positions=reference,
        search=ShellContactSearch(material_fan=True, **TORCH),
    )
    torch.testing.assert_close(material.gap, pairs.gap)
    # A planar quad whose vertices sit in both batch graphs is rejected.
    with pytest.raises(ValueError, match="batch"):
        shell_contact_pairs(
            positions,
            torch.tensor([[0, 1, 6, 5]]),
            thickness,
            batch=batch,
            search=ShellContactSearch(**TORCH),
        )


@pytest.mark.parametrize("settings", [TORCH, dict(implementation="warp")])
def test_reference_clearance_combines_with_a_prediction_horizon(settings):
    """The reference gap is static: a forecast horizon needs no reference velocity."""
    reference, current, faces = _crossing_scene()
    reference = reference.clone()
    reference[4, 2] = 30.0  # node 4 was far away: its pair is new contact
    thickness = torch.zeros(5)
    velocities = torch.zeros_like(current)
    velocities[3, 2] = -0.5  # node 3 keeps approaching the shell
    common = dict(side_from_reference=True, reference_clearance=5.0, **settings)
    static = shell_contact_pairs(
        current,
        faces,
        thickness,
        reference_positions=reference,
        search=ShellContactSearch(**common),
    )
    forecast = shell_contact_pairs(
        current,
        faces,
        thickness,
        reference_positions=reference,
        velocities=velocities,
        search=ShellContactSearch(prediction_horizon=1.0, **common),
    )
    for pairs in (static, forecast):
        assert pairs.query.tolist() == [4]
        assert pairs.stats["reference_excluded_pairs"] == 1
    torch.testing.assert_close(forecast.gap, static.gap)  # node 4 does not move


def test_warp_broad_phase_matches_the_torch_reference_on_cpu():
    """The Warp BVH broad phase (Torch narrow phase on CPU) reproduces the reference."""
    settings = dict(implementation="warp")
    reference, current, faces = _crossing_scene()
    plate = torch.tensor([[0.0, 0, 0], [4, 0, 0], [4, 4, 0], [0, 4, 0]])
    shift = torch.tensor([50.0, 50.0, 0.0])
    probes = torch.tensor([[2.0, 2, 1.0]]), torch.tensor([[52.0, 52, 2.0]])
    batched = torch.cat((plate, probes[0], plate + shift, probes[1]))
    scenes = [
        (
            (current, faces, torch.full((5,), 0.5)),
            dict(reference_positions=reference),
            dict(side_from_reference=True),
        ),
        (
            (batched, torch.tensor([[0, 1, 2, 3], [5, 6, 7, 8]]), torch.zeros(10)),
            dict(batch=torch.tensor([0] * 5 + [1] * 5)),
            {},
        ),
    ]
    for args, kwargs, options in scenes:
        expected = shell_contact_pairs(
            *args, **kwargs, search=ShellContactSearch(**options, **TORCH)
        )
        actual = shell_contact_pairs(
            *args, **kwargs, search=ShellContactSearch(**options, **settings)
        )
        assert actual.query.tolist() == expected.query.tolist()
        assert actual.face.tolist() == expected.face.tolist()
        assert actual.crossed.tolist() == expected.crossed.tolist()
        torch.testing.assert_close(actual.gap, expected.gap)
