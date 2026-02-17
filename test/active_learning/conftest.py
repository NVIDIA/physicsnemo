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
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator
from unittest.mock import MagicMock

import pytest
import torch

from physicsnemo.active_learning import protocols as p
from physicsnemo.active_learning._registry import registry
from physicsnemo.core import ModelRegistry, Module


# Fixture to clear model registry between tests to avoid naming conflicts
@pytest.fixture(autouse=True)
def clear_model_registry():
    """Clear and restore the model registry before and after each test"""
    model_registry = ModelRegistry()
    model_registry.__clear_registry__()
    yield
    model_registry.__restore_registry__()


# Mock classes for testing serialization
class MockQueryStrategy:
    """Mock query strategy for testing."""

    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        pass

    def attach(self, driver):
        """Attach strategy to driver (no-op for mock)."""
        pass


class MockLabelStrategy:
    """Mock label strategy for testing."""

    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        pass

    def attach(self, driver):
        """Attach strategy to driver (no-op for mock)."""
        pass


class MockMetrologyStrategy:
    """Mock metrology strategy for testing."""

    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        pass

    def attach(self, driver):
        """Attach strategy to driver (no-op for mock)."""
        pass


class MockTrainingLoop:
    """Mock training loop for testing."""

    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        pass


# Register mock classes
registry.register("MockQueryStrategy")(MockQueryStrategy)
registry.register("MockLabelStrategy")(MockLabelStrategy)
registry.register("MockMetrologyStrategy")(MockMetrologyStrategy)
registry.register("MockTrainingLoop")(MockTrainingLoop)


@dataclass
class MockDataStructure:
    """
    Essentially a stand-in that holds inputs, the target device,
    and for testing labeling workflows, an optional target.
    """

    inputs: torch.Tensor
    device: torch.device = torch.device("cpu")
    targets: torch.Tensor | None = None


class MockModule(Module):
    """A mock module that implements a linear layer and stands-in for a non-learner module."""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(64, 3)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass of the mock module"""
        return self.linear(input_tensor)


class MockLearnerModule(p.LearnerProtocol):
    """A mock learner module that implements a linear layer and a loss function"""

    def __init__(self):
        super().__init__()
        self.module = MockModule()
        self.loss_fn = torch.nn.MSELoss()

    def training_step(self, data: MockDataStructure, *args: Any, **kwargs: Any) -> None:
        """As this is a mock module, this is a no-op"""
        return None

    def validation_step(
        self, data: MockDataStructure, *args: Any, **kwargs: Any
    ) -> None:
        """As this is a mock module, this is a no-op"""
        return None

    def inference_step(
        self, data: MockDataStructure, *args: Any, **kwargs: Any
    ) -> None:
        """As this is a mock module, this is a no-op"""
        return None

    @property
    def parameters(self) -> Iterator[torch.Tensor]:
        """Returns an iterator over the parameters of the learner."""
        return self.module.parameters()

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass through the module."""
        return self.module.forward(*args, **kwargs)


@pytest.fixture(scope="function", autouse=True)
def learner_module() -> MockLearnerModule:
    """Mocks a learner module"""
    return MockLearnerModule()


@pytest.fixture(scope="function", autouse=True)
def mock_module() -> MockModule:
    """Mocks a module"""
    return MockModule()


@pytest.fixture(scope="function", autouse=True)
def mock_queue() -> p.AbstractQueue[MockDataStructure]:
    """Mocks a query queue with a single data entry"""
    mock = MagicMock(spec=p.AbstractQueue)
    mock.empty.return_value = False
    mock.get.return_value = MockDataStructure(
        inputs=torch.randn(16, 64),
        device=torch.device("cpu"),
    )
    mock.put.return_value = None
    return mock


@pytest.fixture(scope="function", autouse=True)
def mock_query_strategy() -> p.QueryStrategy:
    mock = MagicMock(spec=p.QueryStrategy)
    mock.sample.return_value = None
    mock._args = {
        "__name__": "MockQueryStrategy",
        "__module__": "test.active_learning.conftest",
        "__args__": {},
    }
    return mock


@pytest.fixture(scope="function", autouse=True)
def mock_label_strategy() -> p.LabelStrategy:
    mock = MagicMock(spec=p.LabelStrategy)
    mock.label.return_value = None
    mock._args = {
        "__name__": "MockLabelStrategy",
        "__module__": "test.active_learning.conftest",
        "__args__": {},
    }
    return mock


@pytest.fixture(scope="function", autouse=True)
def mock_metrology_strategy() -> p.MetrologyStrategy:
    mock = MagicMock(spec=p.MetrologyStrategy)
    mock.compute.return_value = None
    mock.__call__ = mock.compute
    mock.serialize_records.return_value = None
    mock.records = [
        None,
    ]
    mock._args = {
        "__name__": "MockMetrologyStrategy",
        "__module__": "test.active_learning.conftest",
        "__args__": {},
    }
    return mock


@pytest.fixture(scope="function", autouse=True)
def mock_training_loop() -> p.TrainingLoop:
    mock = MagicMock(spec=p.TrainingLoop)
    mock._args = {
        "__name__": "MockTrainingLoop",
        "__module__": "test.active_learning.conftest",
        "__args__": {},
    }
    return mock


@pytest.fixture(scope="function", autouse=True)
def mock_data_pool() -> p.DataPool[MockDataStructure]:
    """Mocks a data pool"""
    mock = MagicMock(spec=p.DataPool)
    mock.append.return_value = None
    mock.__getitem__.return_value = MockDataStructure(
        inputs=torch.randn(16, 64),
        device=torch.device("cpu"),
    )
    mock.__len__.return_value = 1
    mock.__iter__.return_value = iter(
        [
            MockDataStructure(
                inputs=torch.randn(16, 64),
                device=torch.device("cpu"),
            )
        ]
    )
    return mock
