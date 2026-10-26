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

"""Active-learning acquisition uses generic scores without guardrail wrappers."""

import importlib.util
import sys
from pathlib import Path
from queue import Queue
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def strategies(monkeypatch):
    """Load the recipe and its actual local helpers without leaking import aliases."""
    folder = (
        Path(__file__).parents[2]
        / "examples/cfd/external_aerodynamics/active_learning_aero/src"
    )
    modules = {}
    for name in ("utils", "aero_physics", "strategies"):
        spec = importlib.util.spec_from_file_location(name, folder / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        modules[name] = module
    return modules["strategies"]


class Pool:
    """Known embeddings let acquisition ranking be checked independently."""

    vectors = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.1, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ]
    )

    def __init__(self, labeled):
        self.train_indices = torch.tensor(labeled, dtype=torch.int64)

    def unlabeled_indices(self):
        return torch.tensor([i for i in range(5) if i not in self.train_indices])

    def class_of(self, index):
        return "a" if index % 2 else "b"

    def get_by_flat_idx(self, index):
        return {
            "fx": torch.zeros(1, 1, 1),
            "embeddings": self.vectors[index].reshape(1, 1, 3),
        }


class Backbone(torch.nn.Module):
    """Expose the recipe's expected context shape with a known embedding."""

    def forward(self, *, local_embedding, **kwargs):
        return local_embedding, local_embedding.unsqueeze(1)


@pytest.mark.parametrize("k", [1, 10])
def test_latent_novelty_ranks_normalized_distances(strategies, device, k):
    pool = Pool([0, 1])
    strategy = strategies.LatentNoveltyQueryStrategy(max_samples=3, knn_k=k)
    strategy.attach(
        SimpleNamespace(
            training_pool=pool, learner=Backbone(), active_learning_step_idx=1
        )
    )
    queue = Queue()
    strategy.sample(
        queue, device=torch.device(device), embedding_reduction=torch.nn.Flatten(1)
    )
    reference, query = pool.vectors[:2], pool.vectors[2:]
    reference = reference / (reference.norm(dim=1, keepdim=True) + 1e-8)
    query = query / (query.norm(dim=1, keepdim=True) + 1e-8)
    expected = torch.cdist(query, reference).sort(dim=1).values[:, : min(k, 2)].mean(1)
    selected = strategy.selection_history[-1]["selected"]
    assert [queue.get_nowait() for _ in range(3)] == (
        expected.argsort(descending=True) + 2
    ).tolist()
    for row in selected:
        assert row["latent_novelty"] == pytest.approx(
            expected[row["flat_idx"] - 2].item(), abs=1e-6
        )
    assert strategy.selection_history[-1]["knn_k_effective"] == min(k, 2)
    assert strategy.selection_history[-1]["labeled_buffer_size"] == 2


def test_latent_novelty_cold_start_and_empty_pool(strategies):
    strategy = strategies.LatentNoveltyQueryStrategy(max_samples=2, cold_start_seed=4)
    strategy.attach(SimpleNamespace(training_pool=Pool([])))
    queue = Queue()
    strategy.sample(queue)
    record = strategy.selection_history[-1]
    assert record["cold_start_fallback"]
    assert {row["class"] for row in record["selected"]} == {"a", "b"}
    assert queue.qsize() == 2
    strategy.attach(SimpleNamespace(training_pool=Pool(list(range(5)))))
    strategy.sample(queue)
    assert len(strategy.selection_history) == 1
    assert queue.qsize() == 2
