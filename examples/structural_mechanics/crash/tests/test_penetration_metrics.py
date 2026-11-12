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

"""Penetration metrics on two plates with known depths (CPU, Torch contact backend)."""

import json
import os
import sys
from types import SimpleNamespace

import pytest
import torch

THIS_DIR = os.path.dirname(__file__)
CRASH_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
if CRASH_DIR not in sys.path:
    sys.path.insert(0, CRASH_DIR)

from datapipes.datapipe import SimSample  # noqa: E402
from metrics.penetration_metrics import PenetrationMetrics  # noqa: E402

SPACING = 10.0
GAP = 20.0  # reference distance between the plates (beyond the 10 mm clearance)
THICKNESS = 1.0
WALL = 28.0  # x = 28: the lower plate's x = 30 nodes and the upper plate's x = 35 nodes


def _plate(nx, ny, z, offset_xy, node_offset):
    xs, ys = torch.meshgrid(
        torch.arange(nx, dtype=torch.float32) * SPACING + offset_xy,
        torch.arange(ny, dtype=torch.float32) * SPACING + offset_xy,
        indexing="ij",
    )
    pos = torch.stack((xs.reshape(-1), ys.reshape(-1), torch.full((nx * ny,), z)), -1)
    faces = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            a = i * ny + j
            faces.append([a, a + ny, a + ny + 1, a + 1])
    return pos, torch.tensor(faces, dtype=torch.long) + node_offset


def _two_plates(upper_z):
    """Lower plate at z = 0 on the grid, upper plate shifted by half a cell in x and y.

    The half-cell shift puts every interior node of one plate over a facet
    centre of the other, so the symmetric side test certifies its crossing.
    """
    lower, faces_lower = _plate(4, 4, 0.0, 0.0, 0)
    upper, faces_upper = _plate(4, 4, upper_z, SPACING / 2, lower.shape[0])
    return torch.cat((lower, upper)), torch.cat((faces_lower, faces_upper))


def _graph(faces, num_nodes, with_contact=True):
    node = faces.reshape(-1)
    face = torch.arange(faces.shape[0]).repeat_interleave(4)
    graph = SimpleNamespace(num_nodes=num_nodes, rigid_wall=torch.tensor([WALL]))
    if with_contact:
        graph.contact_faces = faces
        graph.contact_surface_exclusions = torch.stack((node, face))
        graph.shell_thickness = torch.full((num_nodes,), THICKNESS)
    return graph


def _case(prediction_z, target_z, with_contact=True):
    """Sample, normalized prediction/target ``[N, 1, 3]`` and identity statistics."""
    reference, faces = _two_plates(GAP)
    predicted, _ = _two_plates(prediction_z)
    target, _ = _two_plates(target_z)
    graph = _graph(faces, reference.shape[0], with_contact)
    sample = SimSample(
        node_features={"coords": reference.clone()},
        node_target=target[:, None],
        graph=graph,
    )
    stats = {"node": {"pos_mean": torch.zeros(1, 3), "pos_std": torch.ones(1, 3)}}
    return sample, predicted[:, None], target[:, None], stats


def _metrics(**cfg):
    settings = dict(implementation="torch", wall_position_mm=WALL)
    settings.update(cfg)
    return PenetrationMetrics(settings, rollout_steps=1, device=torch.device("cpu"))


def test_known_depths_on_two_plates():
    """Upper plate through the lower one: crossing 2 mm (target) and 3 mm (prediction)."""
    sample, prediction, target, stats = _case(prediction_z=-3.0, target_z=-2.0)
    metrics = _metrics()
    metrics.accumulate(prediction, target, sample, stats)
    summary = metrics.summary(torch.ones(1))
    assert summary["complete"] and summary["failure_reasons"] == []
    assert summary["sidedness"] == "symmetric"
    # the ceiling: band plus the mean thickness of the validated design
    assert summary["crossing_depth_ceiling_mm"] == pytest.approx(5.0 + THICKNESS)

    crossing = summary["surface"]["crossing"]["per_region"]["all"]
    envelope = summary["surface"]["envelope"]["per_region"]["all"]
    # 9 interior nodes of each plate project onto the other plate's facets
    assert crossing["node_frames"] == 32
    assert crossing["penetrating_fraction"] == {
        "prediction": 18 / 32,
        "target": 18 / 32,
    }
    assert crossing["max_depth_mm"] == {"prediction": 3.0, "target": 2.0}
    assert crossing["spurious_count"] == 0 and crossing["missed_count"] == 0
    assert crossing["excess_max_mm"] == pytest.approx(1.0)
    assert crossing["excess_mean_mm"] == pytest.approx(18 / 32)
    # the envelope adds the half thicknesses of the pair
    assert envelope["max_depth_mm"] == {"prediction": 4.0, "target": 3.0}
    assert envelope["penetrating_fraction"]["target"] == 18 / 32
    # the absolute blocks read the envelope family beyond the 0.5 mm tolerance
    assert summary["prediction"]["max_depth_mm"] == pytest.approx(3.5)
    assert summary["target"]["max_depth_mm"] == pytest.approx(2.5)
    assert summary["prediction"]["crossed_pairs"] >= 18
    assert summary["pairs_per_time"]["crossed_pairs_target"][0] >= 18

    # the wall at x = 28: four lower nodes 2 mm past it, four upper nodes 7 mm
    wall = summary["wall"]["crossing"]["per_region"]["all"]
    assert wall["max_depth_mm"] == {"prediction": 7.0, "target": 7.0}
    assert wall["penetrating_fraction"] == {"prediction": 8 / 32, "target": 8 / 32}
    assert wall["excess_max_mm"] == 0.0
    wall_envelope = summary["wall"]["envelope"]["per_region"]["all"]
    assert wall_envelope["max_depth_mm"] == {"prediction": 7.5, "target": 7.5}
    assert "DISABLED" not in metrics.log_line(summary)
    json.dumps(summary)


def test_prediction_clear_of_contact_misses_every_target_crossing():
    """A prediction clear of contact misses every target crossing and adds none."""
    sample, prediction, target, stats = _case(prediction_z=GAP, target_z=-2.0)
    metrics = _metrics()
    metrics.accumulate(prediction, target, sample, stats)
    crossing = metrics.summary(torch.ones(1))["surface"]["crossing"]["per_region"][
        "all"
    ]
    assert crossing["penetrating_fraction"]["prediction"] == 0.0
    assert crossing["missed_count"] == 18 and crossing["missed_fraction"] == 1.0
    assert crossing["spurious_count"] == 0


def test_disabled_probe_writes_null_values_and_the_reason():
    """A failure keeps its reason and nulls every count and depth of the record."""
    sample, prediction, target, stats = _case(-3.0, -2.0, with_contact=False)
    metrics = _metrics()
    metrics.accumulate(prediction, target, sample, stats)
    assert metrics.disabled
    summary = metrics.summary(torch.ones(1))
    assert summary["complete"] is False
    assert len(summary["failure_reasons"]) == 1
    assert "contact faces" in summary["failure_reasons"][0]
    assert summary["prediction"]["max_depth_mm"] is None
    assert summary["prediction"]["crossed_pairs_per_time"] == [None]
    crossing = summary["surface"]["crossing"]["per_region"]["all"]
    assert crossing["spurious_count"] is None and crossing["max_depth_mm"] == {
        "prediction": None,
        "target": None,
    }
    assert summary["wall"] is None  # no frame was scored
    assert summary["tolerance_mm"] == 0.5  # the settings stay
    assert metrics.log_line(summary).startswith("DISABLED (")
    json.dumps(summary)
    # a later frame is ignored once the probe is disabled
    metrics.accumulate(prediction, target, sample, stats)
    assert len(metrics.failure_reasons) == 1
