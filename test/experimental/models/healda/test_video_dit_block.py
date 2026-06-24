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

from physicsnemo.experimental.models.healda.obs_packing import ObsCrossAttention
from physicsnemo.experimental.models.healda.video_dit_block import VideoDiTBlock


def _build_obs(b, t, npix, obs_token_dim, device, max_count=4):
    """Build a packed ragged obs bundle (tokens + meta) for ``b*t*npix`` pixels."""
    total_pixels = b * t * npix
    counts = torch.randint(0, max_count, (total_pixels,), device=device)
    cu = torch.zeros(total_pixels + 1, dtype=torch.int32, device=device)
    cu[1:] = torch.cumsum(counts, 0).to(torch.int32)
    n_tokens = int(cu[-1].item())
    tokens = torch.randn(n_tokens, obs_token_dim, device=device, requires_grad=True)
    obs = ObsCrossAttention(
        tokens=tokens,
        cu_seqlens_k=cu,
        max_seqlen_k=int(counts.max().item()) if total_pixels else 0,
    )
    return obs


def test_plain_block_reduces_to_spatial_mlp_cpu():
    """With temporal/obs off the block is a spatial DiT block; runs on CPU."""
    torch.manual_seed(0)
    b, t, npix, c = 2, 3, 16, 64
    block = VideoDiTBlock(hidden_size=c, num_heads=4, emb_channels=32)
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
        hidden_size=c, num_heads=4, emb_channels=32, temporal_attention=True
    )
    x = torch.randn(b, t, npix, c, requires_grad=True)
    emb = torch.randn(b, 32)
    out = block(x, emb, is_causal=True)
    assert out.shape == (b, t, npix, c)
    out.float().pow(2).mean().backward()
    assert block.temporal_attn.qkv.weight.grad is not None
    assert torch.isfinite(x.grad).all()


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="triton obs attn is CUDA-only"
)
def test_full_block_cuda():
    """Full block (spatial + obs cross-attn + temporal) forward/backward on CUDA."""
    torch.manual_seed(0)
    dev = "cuda"
    b, t, npix, c = 2, 3, 64, 256
    obs_token_dim = 16
    block = VideoDiTBlock(
        hidden_size=c,
        num_heads=8,
        emb_channels=128,
        temporal_attention=True,
        obs_cross_attention=True,
        obs_token_dim=obs_token_dim,
    ).to(dev)

    x = torch.randn(b, t, npix, c, device=dev, requires_grad=True)
    emb = torch.randn(b, 128, device=dev)
    obs = _build_obs(b, t, npix, obs_token_dim, dev)

    out = block(x, emb, obs=obs)
    assert out.shape == (b, t, npix, c)
    assert torch.isfinite(out).all()

    out.float().pow(2).mean().backward()
    for g in (
        x.grad,
        obs.tokens.grad,
        block.obs_attn.q_proj.weight.grad,
        block.temporal_attn.qkv.weight.grad,
        next(block.spatial_attn.parameters()).grad,
    ):
        assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0
