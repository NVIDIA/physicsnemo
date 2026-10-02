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
from types import SimpleNamespace

import numpy as np
import pytest
import torch

THIS_DIR = os.path.dirname(__file__)
CRASH_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
if CRASH_DIR not in sys.path:
    sys.path.insert(0, CRASH_DIR)

from datapipe import CrashBaseDataset, CrashGraphDataset, SimSample  # noqa: E402


def test_sample_device_transfer_does_not_move_cached_graph():
    """PyG Data.to mutates in place; the dataset-owned graph must stay on CPU."""
    Data = pytest.importorskip("torch_geometric.data").Data
    cached = Data(
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        edge_attr=torch.ones(2, 4),
        component_id=torch.zeros(2, dtype=torch.long),
        num_nodes=2,
    )
    original_edges = cached.edge_index
    for _ in range(4):
        sample = SimSample(
            {"coords": torch.zeros(2, 3)}, torch.zeros(2, 4, 5), graph=cached
        )
        # Meta exercises a real device transition without requiring CUDA.
        sample.to(torch.device("meta"))
        assert sample.graph is not cached
        assert sample.graph.edge_index.device.type == "meta"
        assert cached.edge_index is original_edges
        assert all(
            tensor.device.type == "cpu"
            for _, tensor in cached
            if torch.is_tensor(tensor)
        )


def test_sample_graph_transfer_preserves_values_and_metadata():
    """Verify sample graph transfer preserves values and metadata."""
    Data = pytest.importorskip("torch_geometric.data").Data
    cached = Data(
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        edge_attr=torch.ones(2, 4),
        num_nodes=2,
    )
    sample = SimSample(
        {"coords": torch.zeros(2, 3)}, torch.zeros(2, 4, 5), graph=cached
    )
    sample.to(torch.device("cpu"))
    assert sample.graph is not cached
    assert sample.graph.num_nodes == cached.num_nodes
    torch.testing.assert_close(sample.graph.edge_index, cached.edge_index)
    torch.testing.assert_close(sample.graph.edge_attr, cached.edge_attr)


def test_static_thickness_supports_vtp_point_data_record():
    """Verify static thickness supports VTP point data record."""
    thickness = np.array([1.8, 2.2], dtype=np.float32)

    loaded = CrashGraphDataset._get_static_feature(
        {"point_data": {"thickness": thickness}}, "thickness"
    )

    np.testing.assert_array_equal(loaded, thickness)


def test_edge_stats_weight_every_sample_equally():
    """Verify edge stats weight every sample equally."""
    dataset = CrashGraphDataset.__new__(CrashGraphDataset)
    dataset.num_samples = 2
    dataset.graphs = [
        SimpleNamespace(edge_attr=torch.ones(3, 4)),
        SimpleNamespace(edge_attr=torch.full((5, 4), 3.0)),
    ]

    stats = dataset._compute_edge_stats()

    torch.testing.assert_close(stats["edge_mean"], torch.full((4,), 2.0))
    torch.testing.assert_close(stats["edge_std"], torch.full((4,), 1.0))


def test_global_features_use_train_distribution_stats():
    """Verify global features use train distribution stats."""
    dataset = CrashGraphDataset.__new__(CrashGraphDataset)
    dataset.global_features_keys = ["velocity_x", "rwall_origin_y"]
    dataset.global_features = [
        {"velocity_x": -7.0, "rwall_origin_y": 0.0},
        {"velocity_x": -5.0, "rwall_origin_y": 120.0},
        {"velocity_x": -3.0, "rwall_origin_y": 240.0},
    ]

    dataset.global_stats = dataset._compute_global_stats()
    normalized = dataset._normalized_global_features(1)

    torch.testing.assert_close(
        dataset.global_stats["global_mean"], torch.tensor([-5.0, 120.0])
    )
    torch.testing.assert_close(torch.stack(list(normalized.values())), torch.zeros(2))


def test_graph_component_ids_are_contiguous():
    """Verify graph component ids are contiguous."""
    edge_index = torch.tensor(
        [
            [0, 1, 2, 3, 4],
            [1, 0, 3, 2, 4],
        ],
        dtype=torch.long,
    )

    component_ids = CrashGraphDataset.connected_component_ids(edge_index, 6)

    assert component_ids.tolist() == [0, 0, 1, 1, 2, 3]


def _window_dataset(sample_type: str) -> CrashBaseDataset:
    dataset = CrashBaseDataset.__new__(CrashBaseDataset)
    dataset.num_samples = 1
    dataset.num_steps = 6
    dataset.initial_history_steps = 2
    dataset.rollout_window_steps = 2
    dataset.sample_type = sample_type
    trajectory = torch.arange(6, dtype=torch.float32).view(6, 1, 1)
    dataset.mesh_pos_seq = [trajectory.expand(-1, 2, 3).clone()]
    dataset.node_features_data = [torch.ones(2, 1)]
    dataset.dynamic_targets = []
    dataset.target_series_data = [{}]
    return dataset


def test_two_frame_full_rollout_uses_second_frame_as_current_state():
    """Verify two frame full rollout uses second frame as current state."""
    dataset = _window_dataset("all_time_steps")

    inputs, target = dataset.build_xy(0, None)

    torch.testing.assert_close(inputs["previous_coords"], torch.zeros(2, 3))
    torch.testing.assert_close(inputs["coords"], torch.ones(2, 3))
    assert target.shape == (2, 4, 3)
    torch.testing.assert_close(target[:, 0], torch.full((2, 3), 2.0))
    torch.testing.assert_close(target[:, -1], torch.full((2, 3), 5.0))


def test_random_time_window_keeps_contiguous_two_frame_history(monkeypatch):
    """Verify random time window keeps contiguous two frame history."""
    dataset = _window_dataset("random_time_window")
    monkeypatch.setattr(torch, "randint", lambda *args, **kwargs: torch.tensor([2]))

    inputs, target = dataset.build_xy(0, None)

    torch.testing.assert_close(inputs["previous_coords"], torch.ones(2, 3))
    torch.testing.assert_close(inputs["coords"], torch.full((2, 3), 2.0))
    assert target.shape == (2, 2, 3)
    torch.testing.assert_close(target[:, 0], torch.full((2, 3), 3.0))
    torch.testing.assert_close(target[:, 1], torch.full((2, 3), 4.0))
