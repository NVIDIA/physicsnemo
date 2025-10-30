# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
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
from typing import Dict

import torch
import pytest


# Ensure we can import modules from the crash example directory
THIS_DIR = os.path.dirname(__file__)
CRASH_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
if CRASH_DIR not in sys.path:
    sys.path.insert(0, CRASH_DIR)

import rollout  # noqa: E402
from datapipe import SimSample  # noqa: E402


def make_sample(N: int = 5, T: int = 4, F: int = 2) -> SimSample:
    torch.manual_seed(0)
    coords = torch.randn(N, 3)
    features = torch.randn(N, F)
    # create ground-truth future positions flattened: [N, (T-1)*3]
    future = torch.randn(N, (T - 1) * 3)

    class DummyGraph:
        pass

    graph = DummyGraph()
    graph.edge_attr = torch.zeros(0, 1)

    node_inputs: Dict[str, torch.Tensor] = {"coords": coords, "features": features}
    return SimSample(node_features=node_inputs, node_target=future, graph=graph)


def make_data_stats() -> Dict[str, Dict[str, torch.Tensor]]:
    # Broadcastable stats: [1, 3]
    zeros = torch.zeros(1, 3)
    ones = torch.ones(1, 3)
    return {
        "node": {
            "norm_vel_mean": zeros,
            "norm_vel_std": ones,
            "norm_acc_mean": zeros,
            "norm_acc_std": ones,
        }
    }


@pytest.fixture(autouse=True)
def stub_parent_classes(monkeypatch):
    # Stub Transolver.__init__ and Transolver.forward
    def transolver_init(self, *args, **kwargs):
        torch.nn.Module.__init__(self)

    def transolver_forward(self, fx=None, embedding=None, time=None):
        # Match shapes expected downstream: return zeros like embedding
        assert embedding is not None
        return torch.zeros_like(embedding)

    monkeypatch.setattr(rollout.Transolver, "__init__", transolver_init, raising=True)
    monkeypatch.setattr(rollout.Transolver, "forward", transolver_forward, raising=True)

    # Stub MeshGraphNet.__init__ and MeshGraphNet.forward
    def mgn_init(self, *args, **kwargs):
        torch.nn.Module.__init__(self)

    def mgn_forward(self, node_features=None, edge_features=None, graph=None):
        # Return zeros acceleration with shape [N, 3]
        assert node_features is not None
        N = node_features.shape[0]
        return torch.zeros(N, 3, dtype=node_features.dtype, device=node_features.device)

    monkeypatch.setattr(rollout.MeshGraphNet, "__init__", mgn_init, raising=True)
    monkeypatch.setattr(rollout.MeshGraphNet, "forward", mgn_forward, raising=True)


def test_transolver_autoregressive_rollout_eval():
    N, T, F = 5, 4, 2
    sample = make_sample(N=N, T=T, F=F)
    stats = make_data_stats()

    model = rollout.TransolverAutoregressiveRolloutTraining(
        dt=5e-3, initial_vel=torch.zeros(1, 3), num_time_steps=T
    )
    model.eval()

    out = model.forward(sample=sample, data_stats=stats)
    assert out.shape == (T - 1, N, 3)


def test_transolver_time_conditional_rollout_eval():
    N, T, F = 6, 5, 3
    sample = make_sample(N=N, T=T, F=F)
    stats = make_data_stats()

    model = rollout.TransolverTimeConditionalRollout(num_time_steps=T)
    model.eval()

    out = model.forward(sample=sample, data_stats=stats)
    assert out.shape == (T - 1, N, 3)


def test_transolver_one_step_rollout_eval():
    N, T, F = 7, 6, 1
    sample = make_sample(N=N, T=T, F=F)
    stats = make_data_stats()

    model = rollout.TransolverOneStepRollout(
        dt=5e-3, initial_vel=torch.zeros(1, 3), num_time_steps=T
    )
    model.eval()

    out = model.forward(sample=sample, data_stats=stats)
    assert out.shape == (T - 1, N, 3)


def test_meshgraphnet_autoregressive_rollout_eval():
    N, T, F = 4, 4, 2
    sample = make_sample(N=N, T=T, F=F)
    stats = make_data_stats()

    model = rollout.MeshGraphNetAutoregressiveRolloutTraining(
        dt=5e-3, initial_vel=torch.zeros(1, 3), num_time_steps=T
    )
    model.eval()

    out = model.forward(sample=sample, data_stats=stats)
    assert out.shape == (T - 1, N, 3)


def test_meshgraphnet_time_conditional_rollout_eval():
    N, T, F = 3, 5, 4
    sample = make_sample(N=N, T=T, F=F)
    stats = make_data_stats()

    model = rollout.MeshGraphNetTimeConditionalRollout(num_time_steps=T)
    model.eval()

    out = model.forward(sample=sample, data_stats=stats)
    assert out.shape == (T - 1, N, 3)


def test_meshgraphnet_one_step_rollout_eval():
    N, T, F = 8, 3, 0
    # allow zero features
    torch.manual_seed(0)
    coords = torch.randn(N, 3)
    future = torch.randn(N, (T - 1) * 3)

    class DummyGraph:
        pass

    graph = DummyGraph()
    graph.edge_attr = torch.zeros(0, 1)

    node_inputs = {"coords": coords, "features": coords.new_zeros((N, 0))}
    sample = SimSample(node_features=node_inputs, node_target=future, graph=graph)
    stats = make_data_stats()

    model = rollout.MeshGraphNetOneStepRollout(
        dt=5e-3, initial_vel=torch.zeros(1, 3), num_time_steps=T
    )
    model.eval()

    out = model.forward(sample=sample, data_stats=stats)
    assert out.shape == (T - 1, N, 3)
