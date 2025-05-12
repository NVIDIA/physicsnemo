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
# ruff: noqa: F401

import os
import shutil
from pathlib import Path
from typing import Callable

import fsspec
import pytest
import torch
import torch.nn as nn
from pytest_utils import import_or_fail

from physicsnemo.distributed import DistributedManager
from physicsnemo.models.mlp import FullyConnected

mock_aws = pytest.importorskip("moto.mock_aws")


@pytest.fixture(params=["./checkpoints", "msc://checkpoint-test/checkpoints"])
def checkpoint_folder(request) -> str:
    return request.param


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


@mock_aws
@import_or_fail(["wandb", "mlflow", "boto3"])
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

    import boto3
    from moto import mock_aws

    from physicsnemo.launch.utils import load_checkpoint, save_checkpoint

    # Set up the mock with IAM credentials for access. These should match those in
    # the MSC Config file (./msc_config_checkpoint.yaml).
    os.environ["AWS_ACCESS_KEY_ID"] = "access-key-id"
    # Credentials for testing only
    os.environ["AWS_SECRET_ACCESS_KEY"] = "secret-access-key"  # noqa: S105

    # Ensure default region is set to match the MSC Config file.
    os.environ["AWS_DEFAULT_REGION"] = "us-east-1"

    current_file = Path(__file__).resolve()
    current_dir = current_file.parent
    os.environ["MSC_CONFIG"] = f"{current_dir}/msc_config_checkpoint.yaml"

    # Create a bucket using the mock directly to ensure that MSC accesses the correct location.
    conn = boto3.resource("s3", region_name="us-east-1")
    conn.create_bucket(Bucket="checkpoint-test-bucket")

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

    loaded_output_1 = mlp_model_1(input_1)
    loaded_output_2 = mlp_model_2(input_2)

    assert torch.allclose(output_1, loaded_output_1, rtol, atol)
    assert torch.allclose(output_2, loaded_output_2, rtol, atol)

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

    # Clean up if writing to local file system (no need with object storage - files will disappear along with the mock).
    if fsspec.utils.get_protocol(checkpoint_folder) == "file":
        shutil.rmtree(checkpoint_folder)
    else:
        # if writing to object, the local cache must be cleared to allow multiple test runs
        local_cache = os.environ["HOME"] + "/.cache/physicsnemo"
        shutil.rmtree(local_cache)


def test_get_checkpoint_dir():
    from physicsnemo.launch.utils import get_checkpoint_dir

    assert get_checkpoint_dir(".", "model") == "./checkpoints_model"
    assert get_checkpoint_dir("./", "model") == "./checkpoints_model"
    assert (
        get_checkpoint_dir("/Users/auser", "model") == "/Users/auser/checkpoints_model"
    )
    assert (
        get_checkpoint_dir("/Users/auser/", "model") == "/Users/auser/checkpoints_model"
    )
    assert (
        get_checkpoint_dir("msc://test_profile/bucket", "model")
        == "msc://test_profile/bucket/checkpoints_model"
    )
    assert (
        get_checkpoint_dir("msc://test_profile/bucket/", "model")
        == "msc://test_profile/bucket/checkpoints_model"
    )
