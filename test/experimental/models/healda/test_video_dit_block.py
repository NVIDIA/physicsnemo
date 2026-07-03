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

from physicsnemo.experimental.models.healda.obs_context import ObsContext
from physicsnemo.experimental.models.healda.attention_layers import (
    PixelCrossAttention,
)
from physicsnemo.experimental.models.healda.video_dit_block import VideoDiTBlock


def _build_tokenized_context(b, t, npix, obs_token_dim, device, max_count=4):
    """Build an already-tokenized ObsContext for ``b*t*npix`` pixels.

    VideoDiTBlock's cross-attention only reads ``tokens``/``cu_seqlens_k``/
    ``max_seqlen_k``; the raw per-observation fields (required on ObsContext,
    but otherwise unused here) are filled with unused placeholder data.
    """
    total_pixels = b * t * npix
    counts = torch.randint(0, max_count, (total_pixels,), device=device)
    cu = torch.zeros(total_pixels + 1, dtype=torch.int32, device=device)
    cu[1:] = torch.cumsum(counts, 0).to(torch.int32)
    n_tokens = int(cu[-1].item())
    tokens = torch.randn(n_tokens, obs_token_dim, device=device, requires_grad=True)
    return ObsContext(
        tokens=tokens,
        cu_seqlens_k=cu,
        max_seqlen_k=int(counts.max().item()) if total_pixels else 0,
        obs=torch.randn(n_tokens, device=device),
        float_metadata=torch.randn(n_tokens, 1, device=device),
        obs_type=torch.randint(0, 4, (n_tokens,), device=device),
        channel=torch.randint(0, 4, (n_tokens,), device=device),
        platform=torch.randint(0, 4, (n_tokens,), device=device),
    )


def test_plain_block_reduces_to_spatial_mlp_cpu():
    """With temporal/cross off the block is a spatial DiT block; runs on CPU."""
    torch.manual_seed(0)
    b, t, npix, c = 2, 3, 16, 64
    block = VideoDiTBlock(hidden_size=c, num_heads=4, condition_embed_dim=32)
    x = torch.randn(b, t, npix, c, requires_grad=True)
    emb = torch.randn(b, 32)
    out = block(x, emb)
    assert out.shape == (b, t, npix, c)
    out.float().pow(2).mean().backward()
    assert torch.isfinite(x.grad).all()


def test_temporal_block_cpu():
    """Temporal attention is pure torch and trains on CPU."""
    torch.manual_seed(0)
    b, t, npix, c = 2, 4, 16, 64
    block = VideoDiTBlock(
        hidden_size=c,
        num_heads=4,
        condition_embed_dim=32,
        temporal_attention=True,
        is_causal=True,
    )
    x = torch.randn(b, t, npix, c, requires_grad=True)
    emb = torch.randn(b, 32)
    out = block(x, emb)
    assert out.shape == (b, t, npix, c)
    out.float().pow(2).mean().backward()
    assert block.temporal_attention.qkv.weight.grad is not None
    assert torch.isfinite(x.grad).all()


def test_adaln_zero_init_toggle():
    """``adaln_zero_init`` actually zeros (or keeps) the modulation linear."""
    zeroed = VideoDiTBlock(
        hidden_size=64, num_heads=4, condition_embed_dim=32, adaln_zero_init=True
    )
    assert zeroed.norm1_modulation.modulation[-1].weight.abs().sum() == 0
    assert zeroed.norm1_modulation.modulation[-1].bias.abs().sum() == 0

    kept = VideoDiTBlock(
        hidden_size=64, num_heads=4, condition_embed_dim=32, adaln_zero_init=False
    )
    assert kept.norm1_modulation.modulation[-1].weight.abs().sum() > 0


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="triton cross-attn is CUDA-only"
)
def test_full_block_cuda():
    """Full block (spatial + cross-attn + temporal) forward/backward on CUDA."""
    torch.manual_seed(0)
    dev = "cuda"
    b, t, npix, c = 2, 3, 64, 256
    token_dim = 16

    def cross_attention():
        return PixelCrossAttention(
            hidden_size=c,
            token_dim=token_dim,
            n_q_heads=c // token_dim,
            n_kv_heads=1,
            d_head=token_dim,
            use_proj_bias=True,
        )

    block = VideoDiTBlock(
        hidden_size=c,
        num_heads=8,
        condition_embed_dim=128,
        temporal_attention=True,
        cross_attention=cross_attention,
        adaln_zero_init=False,  # non-zero gates so every branch gets grad
    ).to(dev)

    x = torch.randn(b, t, npix, c, device=dev, requires_grad=True)
    emb = torch.randn(b, 128, device=dev)
    context = _build_tokenized_context(b, t, npix, token_dim, dev)

    out = block(x, emb, cross_attention_context=context)
    assert out.shape == (b, t, npix, c)
    assert torch.isfinite(out).all()

    out.float().pow(2).mean().backward()
    for g in (
        x.grad,
        context.tokens.grad,
        block.cross_attention.q_proj.weight.grad,
        block.temporal_attention.qkv.weight.grad,
        next(block.attention.parameters()).grad,
    ):
        assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0
