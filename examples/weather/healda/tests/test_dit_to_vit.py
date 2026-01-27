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
# ruff: noqa: S101
"""
Test DiT as_vit functionality.

When DiT is used without noise/label conditioning, AdaNorm parameters become
constant and can be baked into bias terms, removing the weight matrices.
"""

import models
import pytest
import torch
from scripts.convert_dit_to_vit import (
    bake_linear_layer,
    compute_constant_emb,
    convert_checkpoint_file,
)
from utils.checkpointing import Checkpoint

from physicsnemo.models.healda import DiT, ModelConfigV1


@pytest.fixture
def model_config():
    """Common model configuration for testing."""
    return {
        "num_attention_heads": 4,
        "attention_head_dim": 32,
        "in_channels": 2,
        "out_channels": 1,
        "num_layers": 2,
        "level_in": 4,
        "level_model": 3,
        "time_length": 1,
        "temporal_attention": False,
        "label_dim": 0,
    }


def test_bake_linear_layer(model_config):
    """Test that linear layer output is baked correctly."""
    dit = DiT(**model_config, as_vit=False)
    device = torch.device("cpu")
    dtype = torch.float32

    emb = compute_constant_emb(dit.noise_embed, device=device, dtype=dtype)

    # Get the first transformer block's norm1 linear
    block = dit.transformer_blocks[0]
    linear = block.norm1.linear

    baked = bake_linear_layer(linear, emb)

    # Should have shape [6 * dim] (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)
    dim = model_config["num_attention_heads"] * model_config["attention_head_dim"]
    assert baked.shape == (6 * dim,)

    # Should match direct computation
    with torch.no_grad():
        expected = linear(emb).squeeze(0)
    assert torch.allclose(baked, expected)


def test_dit_as_vit_init(model_config):
    """Test that DiT with as_vit=True initializes correctly (shapes and biases)."""
    dit = DiT(**model_config, as_vit=True)

    # noise_embed should be None
    assert dit.noise_embed is None
    assert dit.as_vit is True

    # proj_out_1 should be bias-only (emb_channels=0)
    assert dit.proj_out_1.weight.numel() == 0
    assert dit.proj_out_1.bias is not None

    # All AdaLayerNormZero layers should be bias-only
    for block in dit.transformer_blocks:
        assert block.norm1.linear.weight.numel() == 0
        assert block.norm1.linear.bias is not None


def test_convert_checkpoint_file_roundtrip(model_config, tmp_path):
    """Test full checkpoint conversion roundtrip."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    # Create model config
    config = ModelConfigV1(
        architecture="dit-test",
        condition_channels=model_config["in_channels"],
        out_channels=model_config["out_channels"],
        time_length=model_config["time_length"],
        label_dim=model_config["label_dim"],
    )

    # Create and save original DiT checkpoint
    dit = models.get_model(config)
    dit.to(device=device, dtype=dtype)
    dit.eval()

    input_path = tmp_path / "original.zip"
    with Checkpoint(str(input_path), mode="w") as ckpt:
        ckpt.write_model(dit)
        ckpt.write_model_config(config)

    # Convert checkpoint
    output_path = tmp_path / "converted.zip"
    convert_checkpoint_file(str(input_path), str(output_path), device=str(device))

    # Load converted checkpoint and verify
    with Checkpoint(str(output_path), mode="r") as ckpt:
        new_config = ckpt.read_model_config()
        new_dit = ckpt.read_model(map_location=device)

    # Verify config has as_vit=True
    assert new_config.as_vit is True

    # Verify model structure
    assert new_dit.noise_embed is None
    assert new_dit.as_vit is True
    assert new_dit.proj_out_1.weight.numel() == 0

    # Verify outputs match
    new_dit.to(device=device, dtype=dtype)
    new_dit.eval()
    batch_size = 2
    level_in = 6  # dit-test uses level_in=6
    nside = 2**level_in
    npix = 12 * nside * nside
    time_length = model_config["time_length"]

    hidden_states = torch.randn(
        batch_size,
        model_config["in_channels"],
        time_length,
        npix,
        device=device,
        dtype=dtype,
    )
    noise_labels = torch.zeros(batch_size, device=device, dtype=dtype)
    day_of_year = torch.zeros(batch_size, time_length, device=device, dtype=dtype)
    second_of_day = torch.zeros(batch_size, time_length, device=device, dtype=dtype)

    with torch.no_grad():
        dit_output = dit(
            hidden_states=hidden_states,
            noise_labels=noise_labels,
            class_labels=None,
            day_of_year=day_of_year,
            second_of_day=second_of_day,
        )
        vit_output = new_dit(
            hidden_states=hidden_states,
            day_of_year=day_of_year,
            second_of_day=second_of_day,
        )

    max_diff = (dit_output.out - vit_output.out).abs().max().item()
    assert max_diff < 1e-3, f"Output difference too large: {max_diff}"
