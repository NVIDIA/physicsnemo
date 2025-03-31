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

import shutil
from typing import Callable

import pytest
import torch
import torch.nn as nn
from pytest_utils import import_or_fail

from physicsnemo.distributed import DistributedManager
from physicsnemo.models.mlp import FullyConnected


@pytest.fixture()
def checkpoint_folder() -> str:
    return "./checkpoints"


@pytest.fixture(params=["physicsnemo", "pytorch"])
def model_generator(request) -> Callable:
    # Create fully-connected NN generator function
    if request.param == "physicsnemo":

        def model(x):
            return FullyConnected(
                in_features=x,
                out_features=x,
                num_layers=2,
                layer_size=8,
            )

    else:

        def model(x):
            return nn.Sequential(
                nn.Linear(x, 8),
                nn.ReLU(),
                nn.Linear(8, x),
            )

    return model


@import_or_fail(["wandb", "mlflow"])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_model_checkpointing(
    device,
    model_generator,
    checkpoint_folder,
    pytestconfig,
    rtol: float = 1e-3,
    atol: float = 1e-3,
):
    """Test checkpointing util for model"""

    from physicsnemo.launch.utils import load_checkpoint, save_checkpoint

    # Initialize DistributedManager first since save_checkpoint instantiates it
    DistributedManager.initialize()

    mlp_model_1 = model_generator(8).to(device)
    mlp_model_2 = model_generator(4).to(device)

    input_1 = torch.randn(4, 8).to(device)
    input_2 = torch.randn(4, 4).to(device)

    output_1 = mlp_model_1(input_1)
    output_2 = mlp_model_2(input_2)
    # Save model weights to checkpoint
    save_checkpoint(
        checkpoint_folder,
        models=[mlp_model_1, mlp_model_2],
        metadata={"model_type": "MLP"},
    )

    # Load twin set of models for importing weights
    mlp_model_1 = model_generator(8).to(device)
    mlp_model_2 = model_generator(4).to(device)

    new_output_1 = mlp_model_1(input_1)
    new_output_2 = mlp_model_2(input_2)
    # Assert models are now different
    assert not torch.allclose(output_1, new_output_1, rtol, atol)
    assert not torch.allclose(output_2, new_output_2, rtol, atol)

    # Load model weights from checkpoint
    load_checkpoint(checkpoint_folder, models=[mlp_model_1, mlp_model_2], device=device)

    new_output_1 = mlp_model_1(input_1)
    new_output_2 = mlp_model_2(input_2)

    assert torch.allclose(output_1, new_output_1, rtol, atol)
    assert torch.allclose(output_2, new_output_2, rtol, atol)

    # Also load the model with metadata
    metadata_dict = {}
    epoch = load_checkpoint(
        checkpoint_folder,
        models=[mlp_model_1, mlp_model_2],
        metadata_dict=metadata_dict,
        device=device,
    )

    assert epoch == 0
    assert metadata_dict["model_type"] == "MLP"

    # Clean up
    shutil.rmtree(checkpoint_folder)
