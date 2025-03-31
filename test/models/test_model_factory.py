# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
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

from contextlib import contextmanager

import pytest
import torch

from physicsnemo.models import Module
from physicsnemo.registry import ModelRegistry


class MockModel(Module):
    def __init__(self, layer_size=16):
        super().__init__()
        self.layer_size = layer_size
        self.layer = torch.nn.Linear(layer_size, layer_size)

    def forward(self, x):
        return self.layer(x)


@contextmanager
def _retrieve_model(model: Module, name: str):
    """Registers model in the ModelRegistry and retrieves it."""

    registry = ModelRegistry()
    registry.register(model, name)
    yield registry.factory(name)

    registry.__clear_registry__()
    registry.__restore_registry__()


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_register_and_factory(device):
    # Register and retrieve the MockModel
    with _retrieve_model(MockModel, "mock_model") as RetrievedModel:
        # Check if the retrieved model is the same as the one registered
        assert RetrievedModel == MockModel

        # Check forward pass of the model.
        layer_size = 16
        invar = torch.randn(1, layer_size).to(device)
        model = RetrievedModel(layer_size=layer_size).to(device)
        outvar = model(invar)
        assert outvar.shape == invar.shape
        assert outvar.device == invar.device
