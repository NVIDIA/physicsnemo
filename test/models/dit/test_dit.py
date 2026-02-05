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
# ruff: noqa: E402

from typing import Tuple

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from physicsnemo.experimental.models.dit import DiT
from physicsnemo.experimental.models.dit.layers import (
    DetokenizerModuleBase,
    DiTBlock,
    TokenizerModuleBase,
)
from test import common
from test.conftest import requires_module

# --- Tests ---


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
        layernorm_backend="torch",
        attention_backend="timm",
    ).to(device)
    model.eval()  # Set to eval to avoid dropout randomness

    x = torch.randn(2, 3, 32, 32).to(device)
    t = torch.randint(0, 1000, (2,)).to(device)

    assert common.validate_forward_accuracy(
        model,
        (x, t, None),  # Inputs tuple for an unconditional model
        file_name="models/dit/data/dit_unconditional_output.pth",
        atol=1e-3,
    )


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
        layernorm_backend="torch",
        attention_backend="timm",
    ).to(device)
    model.eval()  # Set to eval to avoid dropout randomness

    x = torch.randn(2, 3, 32, 32).to(device)
    t = torch.randint(0, 1000, (2,)).to(device)
    condition = torch.randn(2, 128).to(device)

    assert common.validate_forward_accuracy(
        model,
        (x, t, condition),
        file_name="models/dit/data/dit_conditional_output.pth",
        atol=1e-3,
    )


def test_dit_constructor(device):
    """Test different DiT constructor options and shape consistency."""
    input_size = (16, 32)
    in_channels = 3
    out_channels = 5
    condition_dim = 128
    attention_backend = "timm"
    layernorm_backend = "torch"
    batch_size = 2

    model = DiT(
        input_size=input_size,
        patch_size=4,
        in_channels=in_channels,
        out_channels=out_channels,
        condition_dim=condition_dim,
        hidden_size=128,
        depth=2,
        attention_backend=attention_backend,
        layernorm_backend=layernorm_backend,
        num_heads=4,
    ).to(device)

    x = torch.randn(batch_size, in_channels, *input_size).to(device)
    t = torch.randint(0, 1000, (batch_size,)).to(device)
    condition = torch.randn(batch_size, condition_dim).to(device)

    output = model(x, t, condition)

    assert output.shape == (batch_size, out_channels, *input_size)


class CustomTokenizer(TokenizerModuleBase):
    """Simple N C H W -> N L D mapping."""

    def __init__(self, in_channels, hidden_size, patch_size: int):
        super().__init__()
        self.proj = nn.Linear(in_channels, hidden_size)
        self.patch_size = patch_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.avg_pool2d(x, kernel_size=self.patch_size, stride=self.patch_size)
        x = x.flatten(2).permute(0, 2, 1)
        x = self.proj(x)
        print(x.shape)
        return x

    def initialize_weights(self):
        pass


class CustomDetokenizer(DetokenizerModuleBase):
    """Simple N L D -> N C H W mapping."""

    def __init__(
        self,
        out_channels: int,
        input_size: Tuple[int, int],
        hidden_size: int,
        patch_size: int,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.proj = nn.Conv2d(hidden_size, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2).reshape(
            -1,
            self.hidden_size,
            self.input_size[0] // self.patch_size,
            self.input_size[1] // self.patch_size,
        )
        x = F.interpolate(x, size=self.input_size, mode="nearest")
        x = self.proj(x)
        return x

    def initialize_weights(self):
        pass


@requires_module("transformer_engine")
@pytest.mark.parametrize(
    "tokenizer",
    [CustomTokenizer(in_channels=3, hidden_size=64, patch_size=4), "patch_embed_2d"],
)
@pytest.mark.parametrize(
    "detokenizer",
    [
        CustomDetokenizer(
            out_channels=4, input_size=(16, 16), hidden_size=64, patch_size=4
        ),
        "proj_reshape_2d",
    ],
)
def test_dit_checkpoint(device, tokenizer, detokenizer):
    """Test DiT checkpoint save/load with custom Modules"""

    if device == "cpu":
        pytest.skip("Skipping DiT checkpoint test on CPU since TE is CUDA-only")

    model_1 = (
        DiT(
            input_size=(16, 16),
            patch_size=(4, 4),
            in_channels=3,
            out_channels=4,
            hidden_size=64,
            depth=1,
            num_heads=2,
            layernorm_backend="torch",
            tokenizer=tokenizer,
            detokenizer=detokenizer,
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
            tokenizer=tokenizer,
            detokenizer=detokenizer,
            layernorm_backend="torch",
        )
        .to(device)
        .eval()
    )

    # Change weights on one model to ensure they are different initially
    with torch.no_grad():
        for param in model_2.parameters():
            param.add_(0.1)

    x = torch.randn(2, 3, 16, 16).to(device)
    t = torch.randint(0, 1000, (2,)).to(device)

    assert common.validate_checkpoint(model_1, model_2, (x, t, None))


# ---------- DiTBlock tests ----------


class _Meta:
    def __init__(self, name: str):
        self.name = name


class _DiTBlockWrapper(nn.Module):
    """Thin wrapper to adapt `DiTBlock` to the forward-accuracy helper.

    - Exposes `.meta.name` and `.device` expected by `common.validate_forward_accuracy`.
    - Fixes `attn_kwargs`/`p_dropout` so the wrapped forward is `(x, c) -> y`.
    """

    def __init__(
        self,
        block: DiTBlock,
        name: str,
        attn_kwargs: dict | None = None,
        p_dropout: float | torch.Tensor | None = None,
    ):
        super().__init__()
        self.block = block
        self.meta = _Meta(name)
        self._attn_kwargs = attn_kwargs or {}
        self._p_dropout = p_dropout

    @property
    def device(self):
        return next(self.block.parameters()).device

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return self.block(
            x, c, attn_kwargs=self._attn_kwargs, p_dropout=self._p_dropout
        )


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

    model = _DiTBlockWrapper(block, name="ditblock_timm", attn_kwargs=None)

    x = torch.randn(B, T, hidden_size, device=device)
    c = torch.randn(B, hidden_size, device=device)

    # Shape check folded into correctness test
    y = model(x, c)
    assert y.shape == (B, T, hidden_size)

    assert common.validate_forward_accuracy(
        model,
        (x, c),
        file_name="models/dit/data/ditblock_timm_output.pth",
    )


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

    model = _DiTBlockWrapper(
        block, name="ditblock_natten", attn_kwargs={"latent_hw": (H, W)}
    )

    x = torch.randn(B, T, hidden_size, device=device)
    c = torch.randn(B, hidden_size, device=device)

    # Shape check folded into correctness test
    y = model(x, c)
    assert y.shape == (B, T, hidden_size)

    assert common.validate_forward_accuracy(
        model,
        (x, c),
        file_name="models/dit/data/ditblock_natten_output.pth",
    )


@requires_module(["transformer_engine"])  # TE dependency
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

    model = _DiTBlockWrapper(block, name="ditblock_te")

    x = torch.randn(B, T, hidden_size, device=device)
    c = torch.randn(B, hidden_size, device=device)

    # Shape check folded into correctness test
    y = model(x, c)
    assert y.shape == (B, T, hidden_size)

    assert common.validate_forward_accuracy(
        model,
        (x, c),
        file_name="models/dit/data/ditblock_te_output.pth",
    )


def test_ditblock_exceptions(device):
    # Per-sample dropout mismatched shape should raise ValueError
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

    # NATTEN path missing latent_hw should raise TypeError (only if natten is installed)
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
