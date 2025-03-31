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

import random
from dataclasses import dataclass

import pytest
import torch

from physicsnemo.models.module import ModelMetaData, Module
from physicsnemo.registry import ModelRegistry

from . import common

registry = ModelRegistry()


class CustomModel(torch.nn.Module):
    """Custom User Model"""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)


@dataclass
class CustomMetaData(ModelMetaData):
    """Custom User Metadata for Model"""

    name: str = "FullyConnected"
    # Optimization
    jit: bool = True
    cuda_graphs: bool = True
    amp: bool = True
    torch_fx: bool = True
    # Inference
    onnx: bool = True
    onnx_runtime: bool = True
    # Physics informed
    func_torch: bool = True
    auto_grad: bool = True


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_from_torch_forward(device):
    """Test forward pass from PyTorch"""
    torch.manual_seed(0)

    # Construct CustomPhysicsNeMoModel
    CustomPhysicsNeMoModel = Module.from_torch(CustomModel, CustomMetaData())
    model = CustomPhysicsNeMoModel(in_features=32, out_features=8).to(device)

    bsize = 8
    invar = torch.randn(bsize, 32).to(device)
    model(invar)
    registry.__clear_registry__()
    registry.__restore_registry__()


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_from_torch_constructor(device):
    """Test constructor from PyTorch"""

    CustomPhysicsNeMoModel = Module.from_torch(CustomModel, CustomMetaData())
    model = CustomPhysicsNeMoModel(in_features=8, out_features=4).to(device)

    assert isinstance(model, Module)

    registry.__clear_registry__()
    registry.__restore_registry__()


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_from_torch_optims(device):
    """Test optimizations from PyTorch"""

    def setup_model():
        """Set up fresh model and inputs for each optim test"""
        # Construct CustomPhysicsNeMoModel
        CustomPhysicsNeMoModel = Module.from_torch(CustomModel, CustomMetaData())
        model = CustomPhysicsNeMoModel(in_features=32, out_features=8).to(device)

        bsize = random.randint(1, 16)
        invar = torch.randn(bsize, 32).to(device)
        return model, invar

    # Ideally always check graphs first
    model, invar = setup_model()
    assert common.validate_cuda_graphs(model, (invar,))
    registry.__clear_registry__()
    registry.__restore_registry__()
    # Check JIT
    model, invar = setup_model()
    assert common.validate_jit(model, (invar,))
    registry.__clear_registry__()
    registry.__restore_registry__()
    # Check AMP
    model, invar = setup_model()
    assert common.validate_amp(model, (invar,))
    registry.__clear_registry__()
    registry.__restore_registry__()
    # Check Combo
    model, invar = setup_model()
    assert common.validate_combo_optims(model, (invar,))
    registry.__clear_registry__()
    registry.__restore_registry__()


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_from_torch_checkpoint(device):
    """Test checkpoint save/load from PyTorch"""
    # Construct CustomPhysicsNeMoModel
    CustomPhysicsNeMoModel = Module.from_torch(CustomModel, CustomMetaData())
    model_1 = CustomPhysicsNeMoModel(in_features=4, out_features=4).to(device)

    model_2 = CustomPhysicsNeMoModel(in_features=4, out_features=4).to(device)

    bsize = random.randint(1, 16)
    invar = torch.randn(bsize, 4).to(device)
    assert common.validate_checkpoint(model_1, model_2, (invar,))
    registry.__clear_registry__()
    registry.__restore_registry__()


@common.check_ort_version()
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_from_torch_deploy(device):
    """Test deployment support from PyTorch"""
    # Construct CustomPhysicsNeMoModel
    CustomPhysicsNeMoModel = Module.from_torch(CustomModel, CustomMetaData())
    model = CustomPhysicsNeMoModel(in_features=4, out_features=4).to(device)

    bsize = random.randint(1, 4)
    invar = torch.randn(bsize, 4).to(device)
    assert common.validate_onnx_export(model, (invar,))
    assert common.validate_onnx_runtime(model, (invar,))
    registry.__clear_registry__()
    registry.__restore_registry__()
