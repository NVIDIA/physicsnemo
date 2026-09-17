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

from physicsnemo.models.dit import DiT
from physicsnemo.nn.module.dit_layers import (
    DetokenizerModuleBase,
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


def test_dit_conv_detokenizer_forward_accuracy(device):
    """Non-regression for the ``proj_reshape_2d_conv`` detokenizer (ConvDetokenizer).

    The residual conv head is zero-initialized (so a fresh model is numerically
    identical to ``proj_reshape_2d``); we activate the final conv with a fixed
    seed so the smoothing path actually contributes, then compare against a saved
    reference. CPU-friendly: uses the timm attention backend.

    The reference holds a non-trivial fp32 conv output (unlike the all-zero
    references of the other DiT forward-accuracy tests, whose zero-initialized
    output projection makes the result device-independent). Two precautions keep
    the single reference valid on both CPU and CUDA:

    * All weights (including the activated conv head) are initialized while the
      model is still on CPU, *before* ``.to(device)``. Initializing after the
      move would draw the conv weights from the CUDA RNG, which produces a
      different sequence than the CPU RNG for the same seed, so the CUDA run
      would use different weights than the CPU-generated reference.
    * TF32 is disabled so the CUDA convolutions compute true fp32 and match the
      reference within 1e-3 instead of drifting by TF32's ~1e-3 relative error.
    """
    prev_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    prev_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
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
            detokenizer="proj_reshape_2d_conv",
            detokenizer_kwargs={
                "conv_layers": 2,
                "conv_hidden": 16,
                "conv_kernel": 3,
            },
        )  # built on CPU; moved to device below, after all init

        # Activate the zero-initialized residual head so the conv path
        # contributes to the output (otherwise the residual is exactly zero by
        # construction). Done on CPU so the RNG draw is device-independent.
        with torch.no_grad():
            convs = [m for m in model.detokenizer.conv_head if isinstance(m, nn.Conv2d)]
            torch.manual_seed(1)
            nn.init.normal_(convs[-1].weight, std=0.02)
            nn.init.normal_(convs[-1].bias, std=0.02)

        model = model.to(device)
        model.eval()  # Set to eval to avoid dropout randomness

        x = torch.randn(2, 3, 32, 32).to(device)
        t = torch.randint(0, 1000, (2,)).to(device)

        assert common.validate_forward_accuracy(
            model,
            (x, t, None),
            file_name="models/dit/data/dit_conv_detokenizer_output.pth",
            atol=1e-3,
        )
    finally:
        torch.backends.cudnn.allow_tf32 = prev_cudnn_tf32
        torch.backends.cuda.matmul.allow_tf32 = prev_matmul_tf32


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


def test_dit_rope_disables_pos_embed_and_warns(device):
    """Selecting the RoPE NATTEN backend forces pos_embed='none' on the patch
    tokenizer and warns if a conflicting value was explicitly requested.

    Construction does not run the NATTEN kernel, so this is CPU-friendly.
    """
    with pytest.warns(UserWarning, match="rotary"):
        model = DiT(
            input_size=(16, 16),
            patch_size=4,
            in_channels=3,
            hidden_size=64,
            depth=2,
            num_heads=4,
            attention_backend="natten2d_rope",
            attn_kwargs={"attn_kernel": 3},
            tokenizer_kwargs={"pos_embed": "learnable"},
        ).to(device)
    # Additive positional embedding is disabled.
    assert model.tokenizer.pos_embed == 0.0
    # The RoPE tables are owned by a single top-level provider (built once at the
    # latent grid size), not duplicated per block.
    head_dim = 64 // 4
    assert model.rope.rope_cos.shape == (4, 4, head_dim)
    assert not hasattr(model.blocks[0].attention, "rope_cos")
    # Exactly one owner of the tables across the whole model.
    n_owners = sum(1 for m in model.modules() if hasattr(m, "rope_cos"))
    assert n_owners == 1


def test_dit_pixel_mask_to_token_mask(device):
    """The pixel -> token mask reduction marks a patch invalid iff any pixel in
    it is invalid, flattened in the tokenizer's row-major (h, w) order. Pure
    host-side reduction, so it does not require a NATTEN kernel."""
    model = DiT(
        input_size=(16, 16),
        patch_size=4,
        in_channels=3,
        hidden_size=64,
        depth=2,
        num_heads=4,
        attention_backend="natten2d",
        attn_kwargs={"attn_kernel": 3},
        use_nan_mask_tokens=True,
    ).to(device)

    # No static buffer exists anymore; the mask is supplied dynamically.
    assert not hasattr(model, "invalid_token_mask")
    # Each NATTEN block allocated a learned mask token.
    assert model.blocks[0].attention.mask_token is not None

    # A patch is invalid if ANY pixel inside it is invalid. Batch of 2 with
    # distinct per-sample patterns exercises batch-variable masking.
    pixel_mask = torch.zeros(2, 1, 16, 16, dtype=torch.bool, device=device)
    pixel_mask[0, 0, 0:4, 0:4] = True  # sample 0: top-left patch (0, 0)
    pixel_mask[0, 0, 5, 5] = True  # sample 0: single pixel inside patch (1, 1)
    pixel_mask[1, 0, 8, 12] = True  # sample 1: patch (2, 3)

    token_mask = model._pixel_mask_to_token_mask(pixel_mask)
    assert token_mask.shape == (2, 16)  # (B, h_lat * w_lat)

    expected = torch.zeros(2, 4, 4, dtype=torch.bool, device=device)
    expected[0, 0, 0] = True
    expected[0, 1, 1] = True
    expected[1, 2, 3] = True
    assert torch.equal(token_mask, expected.reshape(2, 16))

    # The (B, H, W) form (no channel axis) is accepted and equivalent.
    token_mask_3d = model._pixel_mask_to_token_mask(pixel_mask.squeeze(1))
    assert torch.equal(token_mask_3d, token_mask)


def test_dit_invalid_mask_requires_feature(device):
    """Passing invalid_mask to a model built without use_nan_mask_tokens errors,
    rather than silently ignoring the mask."""
    plain = DiT(
        input_size=(16, 16),
        patch_size=4,
        in_channels=3,
        hidden_size=64,
        depth=2,
        num_heads=4,
        attention_backend="natten2d",
        attn_kwargs={"attn_kernel": 3},
    ).to(device)
    assert not hasattr(plain, "invalid_token_mask")

    x = torch.randn(2, 3, 16, 16, device=device)
    t = torch.randint(0, 1000, (2,), device=device)
    invalid_mask = torch.zeros(2, 1, 16, 16, dtype=torch.bool, device=device)
    with pytest.raises(ValueError):
        plain(x, t, invalid_mask=invalid_mask)


@requires_module(["natten"])
def test_dit_dynamic_invalid_mask_forward(device):
    """A dynamic invalid_mask replaces flagged tokens with the learned mask
    token, changing the forward output only where masked; an all-valid mask is a
    no-op equivalent to passing no mask."""
    if device == "cpu":
        pytest.skip("natten is CUDA-only")

    torch.manual_seed(0)
    model = (
        DiT(
            input_size=(16, 16),
            patch_size=4,
            in_channels=3,
            hidden_size=64,
            depth=2,
            num_heads=4,
            attention_backend="natten2d",
            attn_kwargs={"attn_kernel": 3},
            use_nan_mask_tokens=True,
            # Skip the DiT zero-init of the adaLN gates: with zeroed attention
            # gates the attention branch (where the mask token enters) is gated
            # off at init, so masking would not change the output.
            dit_initialization=False,
        )
        .to(device)
        .eval()
    )
    # Give the learned mask tokens a non-trivial value (init is zero).
    with torch.no_grad():
        for m in model.modules():
            if getattr(m, "mask_token", None) is not None:
                nn.init.normal_(m.mask_token)

    x = torch.randn(2, 3, 16, 16, device=device)
    t = torch.randint(0, 1000, (2,), device=device)

    with torch.no_grad():
        out_none = model(x, t)
        out_valid = model(
            x,
            t,
            invalid_mask=torch.zeros(2, 1, 16, 16, dtype=torch.bool, device=device),
        )
        # Mask the top-left patch of sample 0 only.
        invalid = torch.zeros(2, 1, 16, 16, dtype=torch.bool, device=device)
        invalid[0, 0, 0:4, 0:4] = True
        out_masked = model(x, t, invalid_mask=invalid)

    # An all-valid mask matches passing no mask at all.
    torch.testing.assert_close(out_none, out_valid)
    # Masking changes sample 0 but leaves sample 1 untouched.
    assert not torch.allclose(out_masked[0], out_none[0])
    torch.testing.assert_close(out_masked[1], out_none[1])


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


def test_dit_conv_detokenizer_checkpoint(device):
    """Checkpoint save/load with the ConvDetokenizer (proj_reshape_2d_conv).

    Adds the conv smoothing head's parameters to the checkpoint; verifies they
    round-trip. CPU-friendly (timm attention).
    """
    torch.manual_seed(0)

    def build():
        return (
            DiT(
                input_size=(16, 16),
                patch_size=(4, 4),
                in_channels=3,
                out_channels=4,
                hidden_size=64,
                depth=1,
                num_heads=2,
                layernorm_backend="torch",
                attention_backend="timm",
                detokenizer="proj_reshape_2d_conv",
                detokenizer_kwargs={"conv_layers": 2, "conv_hidden": 16},
            )
            .to(device)
            .eval()
        )

    model_1 = build()
    model_2 = build()
    with torch.no_grad():
        for param in model_2.parameters():
            param.add_(0.1)

    x = torch.randn(2, 3, 16, 16).to(device)
    t = torch.randint(0, 1000, (2,)).to(device)

    assert common.validate_checkpoint(model_1, model_2, (x, t, None))


@requires_module(["natten"])
def test_dit_rope_mask_token_checkpoint(device, tmp_path):
    """Checkpoint stability for the buffer-heavy RoPE + NaN-mask-token config.

    Every ``natten2d_rope`` block registers non-persistent
    ``rope_cos``/``rope_sin`` tables; they must stay out of the ``state_dict``
    yet be rebuilt deterministically so a save / load (and ``from_checkpoint``)
    reproduces the forward. The learned per-block ``mask_token`` parameters are
    ordinary (persistent) parameters and must round-trip. The invalid pattern is
    no longer a buffer (it is supplied dynamically via ``forward(invalid_mask=)``).
    """
    if device == "cpu":
        pytest.skip("natten is CUDA-only")

    def build():
        torch.manual_seed(0)
        return (
            DiT(
                input_size=(16, 16),
                patch_size=4,
                in_channels=3,
                out_channels=3,
                hidden_size=64,
                depth=2,
                num_heads=4,
                layernorm_backend="torch",
                attention_backend="natten2d_rope",
                attn_kwargs={"attn_kernel": 3},
                use_nan_mask_tokens=True,
            )
            .to(device)
            .eval()
        )

    model_1 = build()
    model_2 = build()

    # The deterministically-rebuilt tables are non-persistent; the invalid mask
    # is no longer a buffer at all.
    sd_keys = model_1.state_dict().keys()
    assert not any(k.endswith(("rope_cos", "rope_sin")) for k in sd_keys)
    assert not any(k.endswith("invalid_token_mask") for k in sd_keys)
    # The learned mask tokens are ordinary persistent parameters.
    assert any(k.endswith("mask_token") for k in sd_keys)

    with torch.no_grad():
        for param in model_2.parameters():
            param.add_(0.1)

    x = torch.randn(2, 3, 16, 16).to(device)
    t = torch.randint(0, 1000, (2,)).to(device)

    # Full save / load + from_checkpoint round-trip (no mask supplied -> all-valid).
    assert common.validate_checkpoint(model_1, model_2, (x, t, None))
