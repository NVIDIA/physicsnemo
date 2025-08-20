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
# ruff: noqa: E402

import common
import pytest
import torch

from physicsnemo.experimental.models.dit import DiT

# --- Tests ---


@pytest.mark.parametrize("device", ["cuda:0"])
def test_dit_forward_accuracy(device):
    """Test DiT forward pass against a saved reference output."""
    torch.manual_seed(0)
    model = DiT(
        input_size=32,
        patch_size=4,
        in_channels=3,
        hidden_size=128,
        depth=2,
        num_heads=4,
    ).to(device)
    model.eval()  # Set to eval to avoid dropout randomness

    x = torch.randn(2, 3, 32, 32).to(device)
    t = torch.randint(0, 1000, (2,)).to(device)

    assert common.validate_forward_accuracy(
        model,
        (x, t, None),  # Inputs tuple for an unconditional model
        file_name="dit_unconditional_output.pth",
        atol=1e-3,
    )


@pytest.mark.parametrize("device", ["cuda:0"])
def test_dit_conditional_forward_accuracy(device):
    """Test conditional DiT forward pass against a saved reference output."""
    torch.manual_seed(0)
    model = DiT(
        input_size=32,
        patch_size=4,
        in_channels=3,
        hidden_size=128,
        depth=2,
        num_heads=4,
        condition_dim=128,
    ).to(device)
    model.eval()  # Set to eval to avoid dropout randomness

    x = torch.randn(2, 3, 32, 32).to(device)
    t = torch.randint(0, 1000, (2,)).to(device)
    condition = torch.randn(2, 128).to(device)

    assert common.validate_forward_accuracy(
        model,
        (x, t, condition),
        file_name="dit_conditional_output.pth",
        atol=1e-3,
    )


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_dit_constructor(device):
    """Test different DiT constructor options and shape consistency."""
    input_size = (16, 32)
    in_channels = 3
    out_channels = 5
    condition_dim = 128
    attention_backbone = "timm"
    layernorm_backbone = "torch"
    batch_size = 2

    model = DiT(
        input_size=input_size,
        patch_size=4,
        in_channels=in_channels,
        out_channels=out_channels,
        condition_dim=condition_dim,
        hidden_size=128,
        depth=2,
        attention_backbone=attention_backbone,
        layernorm_backbone=layernorm_backbone,
        num_heads=4,
    ).to(device)

    x = torch.randn(batch_size, in_channels, *input_size).to(device)
    t = torch.randint(0, 1000, (batch_size,)).to(device)
    condition = torch.randn(batch_size, condition_dim).to(device)

    output = model(x, t, condition)

    assert output.shape == (batch_size, out_channels, *input_size)


@pytest.mark.parametrize("device", ["cuda:0"])
def test_dit_checkpoint(device):
    """Test DiT checkpoint save/load."""
    model_1 = (
        DiT(
            input_size=(16, 16),
            patch_size=(4, 4),
            in_channels=3,
            out_channels=4,
            hidden_size=64,
            depth=1,
            num_heads=2,
            attention_backbone="timm",
        )
        .to(device)
        .eval()
    )
    model_2 = (
        DiT(
            input_size=(16, 16),
            patch_size=(4, 4),
            in_channels=3,
            out_channels=4,
            hidden_size=64,
            depth=1,
            num_heads=2,
            attention_backbone="timm",
        )
        .to(device)
        .eval()
    )

    # Change weights on one model to ensure they are different initially
    with torch.no_grad():
        model_2.proj_layer.output_projection.weight.data.add_(0.1)

    x = torch.randn(2, 3, 16, 16).to(device)
    t = torch.randint(0, 1000, (2,)).to(device)

    assert common.validate_checkpoint(model_1, model_2, (x, t, None))
