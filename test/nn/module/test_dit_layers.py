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

import pytest
import torch

from physicsnemo.nn.module.dit_layers import DiTBlock
from test import common
from test.conftest import requires_module

# --- DiTBlock tests ---


@torch.no_grad()
def test_ditblock_forward_accuracy_timm(device):
    if device == "cpu":
        pytest.skip("CUDA only")

    torch.manual_seed(0)
    hidden_size = 128
    num_heads = 4
    B, T = 2, 16

    block = (
        DiTBlock(
            hidden_size=hidden_size,
            num_heads=num_heads,
            attention_backend="timm",
            layernorm_backend="torch",
        )
        .to(device)
        .eval()
    )

    x = torch.randn(B, T, hidden_size, device=device)
    c = torch.randn(B, hidden_size, device=device)

    y = block(x, c)
    assert y.shape == (B, T, hidden_size)

    assert common.validate_tensor_accuracy(
        y,
        file_name="nn/module/data/ditblock_timm_output.pth",
    )


@torch.no_grad()
@requires_module(["natten"])
def test_ditblock_forward_accuracy_natten(device, pytestconfig):
    if device == "cpu":
        pytest.skip("natten not available on CPU")

    torch.manual_seed(0)
    hidden_size = 64
    num_heads = 4
    B, H, W = 2, 8, 8
    T = H * W

    block = (
        DiTBlock(
            hidden_size=hidden_size,
            num_heads=num_heads,
            attention_backend="natten2d",
            layernorm_backend="torch",
            attn_kernel=3,
        )
        .to(device)
        .eval()
    )

    x = torch.randn(B, T, hidden_size, device=device)
    c = torch.randn(B, hidden_size, device=device)

    y = block(x, c, attn_kwargs={"latent_hw": (H, W)})
    assert y.shape == (B, T, hidden_size)

    assert common.validate_tensor_accuracy(
        y,
        file_name="nn/module/data/ditblock_natten_output.pth",
    )


@torch.no_grad()
@requires_module(["transformer_engine"])
def test_ditblock_forward_accuracy_transformer_engine(device, pytestconfig):
    if device == "cpu":
        pytest.skip("Skipping DiT checkpoint test on CPU since TE is CUDA-only")

    torch.manual_seed(0)
    hidden_size = 128
    num_heads = 8
    B, T = 2, 32

    block = (
        DiTBlock(
            hidden_size=hidden_size,
            num_heads=num_heads,
            attention_backend="transformer_engine",
            layernorm_backend="torch",
        )
        .to(device)
        .eval()
    )

    x = torch.randn(B, T, hidden_size, device=device)
    c = torch.randn(B, hidden_size, device=device)

    y = block(x, c)
    assert y.shape == (B, T, hidden_size)

    assert common.validate_tensor_accuracy(
        y,
        file_name="nn/module/data/ditblock_te_output.pth",
    )


def test_ditblock_exceptions(device):
    hidden_size = 32
    num_heads = 4
    B, T = 2, 8
    block = (
        DiTBlock(
            hidden_size=hidden_size,
            num_heads=num_heads,
            attention_backend="timm",
            layernorm_backend="torch",
            intermediate_dropout=True,
        )
        .to(device)
        .train()
    )

    x = torch.randn(B, T, hidden_size, device=device)
    c = torch.randn(B, hidden_size, device=device)
    with pytest.raises(ValueError):
        _ = block(x, c, p_dropout=torch.tensor([0.5], device=device))

    try:
        import natten  # noqa: F401
    except Exception:
        pytest.skip("natten not available; skipping natten exception subtest")

    hidden_size = 64
    num_heads = 4
    B, T = 2, 64
    nat_block = DiTBlock(
        hidden_size=hidden_size,
        num_heads=num_heads,
        attention_backend="natten2d",
        layernorm_backend="torch",
        attn_kernel=3,
    ).to(device)

    x = torch.randn(B, T, hidden_size, device=device)
    c = torch.randn(B, hidden_size, device=device)
    with pytest.raises(TypeError):
        _ = nat_block(x, c)  # missing required attn_kwargs: latent_hw


def test_ditblock_intermediate_dropout_scalar_and_per_sample(device):
    torch.manual_seed(123)
    hidden_size = 64
    num_heads = 4
    B, T = 3, 16
    block = DiTBlock(
        hidden_size=hidden_size,
        num_heads=num_heads,
        attention_backend="timm",
        layernorm_backend="torch",
        intermediate_dropout=True,
    ).to(device)

    x = torch.randn(B, T, hidden_size, device=device)
    c = torch.randn(B, hidden_size, device=device)

    # Eval mode: dropout should be a no-op regardless of p_dropout
    block.eval()
    y_no = block(x, c, p_dropout=None)
    y_ps = block(x, c, p_dropout=0.7)
    assert torch.allclose(y_no, y_ps, atol=0.0)

    # Train mode: deterministic under fixed seed
    block.train()
    torch.manual_seed(999)
    y1 = block(x, c, p_dropout=0.5)
    torch.manual_seed(999)
    y2 = block(x, c, p_dropout=0.5)
    assert torch.allclose(y1, y2, atol=0.0)

    # Per-sample dropout requires p shaped [B]
    p = torch.tensor([0.1] * B, device=device)
    _ = block(x, c, p_dropout=p)  # should run
