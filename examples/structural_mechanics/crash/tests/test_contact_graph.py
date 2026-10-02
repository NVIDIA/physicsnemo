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

import os
import sys

import torch

THIS_DIR = os.path.dirname(__file__)
CRASH_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
if CRASH_DIR not in sys.path:
    sys.path.insert(0, CRASH_DIR)

from contact_graph import BumperCylinderContactEncoder  # noqa: E402


def test_bumper_cylinder_encoder_builds_obstacle_edges_with_signed_gap():
    """Verify bumper cylinder encoder builds obstacle edges with signed gap."""
    encoder = BumperCylinderContactEncoder(
        center_x=-170.0,
        radius=127.0,
        search_distance=200.0,
    )
    positions = torch.tensor(
        [
            [-40.0, 0.0, 0.0],  # 3 mm outside the cylinder
            [100.0, 0.0, 0.0],  # 143 mm outside, still in search range
            [300.0, 0.0, 0.0],  # outside the search range
        ]
    )

    graph = encoder(
        positions,
        center_y=torch.tensor(0.0),
        shell_thickness=torch.tensor([2.0, 4.0, 2.0]),
    )

    assert graph.edge_index.shape == (2, 2)
    torch.testing.assert_close(graph.edge_index[0], graph.edge_index[1])
    assert graph.obstacle_mask.all()
    torch.testing.assert_close(
        graph.edge_features[:, 4], torch.tensor([2.0 / 200.0, 141.0 / 200.0])
    )
    torch.testing.assert_close(
        graph.edge_features[:, 5:],
        torch.tensor([[-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]),
    )


def test_bumper_cylinder_encoder_uses_per_graph_center():
    """Verify bumper cylinder encoder uses per graph center."""
    encoder = BumperCylinderContactEncoder(search_distance=5.0)
    positions = torch.tensor(
        [
            [-40.0, 0.0, 0.0],
            [-40.0, 120.0, 0.0],
        ]
    )
    batch = torch.tensor([0, 1])

    graph = encoder(
        positions,
        center_y=torch.tensor([0.0, 120.0]),
        batch=batch,
    )

    assert graph.edge_index.shape[1] == 2
    torch.testing.assert_close(graph.edge_features[:, 4], torch.full((2,), 3.0 / 5.0))
