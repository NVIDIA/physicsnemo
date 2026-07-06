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
import torch.nn as nn

from physicsnemo.experimental.models.healda.attention_layers import (
    PixelCrossAttention,
)
from physicsnemo.experimental.models.healda.video_dit import VideoDiT, VideoDiTBlock
from physicsnemo.nn import RotaryEmbedding1DTables

NPIX = 48


class _MockTokenizer(nn.Module):
    """Grid-free tokenizer stub: per-pixel linear (B,C,T,X) -> (B,T,X,D).

    Stands in for a real grid tokenizer (e.g. HEALPixPatchTokenizer) so
    VideoDiT's spatial/temporal/cross-attention/conditioning wiring can be
    tested without an earth2grid dependency; ignores tokenizer_kwargs
    (e.g. calendar features) a real tokenizer would consume.
    """

    def __init__(self, in_channels: int, hidden_size: int):
        super().__init__()
        self.proj = nn.Linear(in_channels, hidden_size)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.proj(x.permute(0, 2, 3, 1))  # (B,C,T,X) -> (B,T,X,D)


class _MockDetokenizer(nn.Module):
    """Grid-free detokenizer stub: per-pixel linear (B,T,X,D) -> (B,C,T,X)."""

    def __init__(self, hidden_size: int, out_channels: int):
        super().__init__()
        self.proj = nn.Linear(hidden_size, out_channels)

    def forward(self, h: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return self.proj(h).permute(0, 3, 1, 2)  # (B,T,X,D) -> (B,C,T,X)


def _build_model(c, t, hidden, num_heads, device, **kwargs):
    emb_channels = 4 * hidden
    tokenizer = _MockTokenizer(c, hidden)
    detokenizer = _MockDetokenizer(hidden, c)
    return VideoDiT(
        tokenizer,
        detokenizer,
        hidden_size=hidden,
        num_heads=num_heads,
        num_layers=2,
        emb_channels=emb_channels,
        **kwargs,
    ).to(device)


def _calendar(b, t, device):
    sod = torch.rand(b, t, device=device) * 86400.0
    doy = torch.rand(b, t, device=device) * 365.0
    return {"second_of_day": sod, "day_of_year": doy}


def test_video_dit_cpu_temporal():
    """Grid-agnostic dense + temporal path (no cross-attention) on CPU."""
    torch.manual_seed(0)
    b, c, t, hidden = 2, 3, 2, 64
    model = _build_model(
        c, t, hidden, 4, "cpu", temporal_attention=True, is_causal=True
    )
    x = torch.randn(b, c, t, NPIX, requires_grad=True)
    out = model(x, torch.rand(b), tokenizer_kwargs=_calendar(b, t, "cpu"))
    assert out.shape == (b, c, t, NPIX)
    out.float().pow(2).mean().backward()
    assert torch.isfinite(x.grad).all()


def test_video_dit_drop_path_rates():
    """Explicit ``drop_path_rates`` are honored per block; bad length raises."""
    model = _build_model(3, 2, 64, 4, "cpu", drop_path_rates=[0.1, 0.2])
    assert [blk.drop_path.drop_prob for blk in model.blocks] == [0.1, 0.2]
    with pytest.raises(ValueError, match="drop_path_rates length"):
        _build_model(3, 2, 64, 4, "cpu", drop_path_rates=[0.1, 0.2, 0.3])


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="triton cross-attn is CUDA-only"
)
def test_video_dit_cuda_full():
    """Dense + temporal + injected cross-attention on CUDA."""
    torch.manual_seed(0)
    dev = "cuda"
    b, c, t, hidden, otd = 2, 3, 2, 256, 16

    def cross_attention():
        return PixelCrossAttention(
            hidden_size=hidden,
            token_dim=otd,
            n_q_heads=hidden // otd,
            n_kv_heads=1,
            d_head=otd,
            use_proj_bias=True,
        )

    model = _build_model(
        c,
        t,
        hidden,
        8,
        dev,
        temporal_attention=True,
        cross_attention=cross_attention,
        adaln_zero_init=False,  # non-zero gates so every branch gets grad
    )
    x = torch.randn(b, c, t, NPIX, device=dev, requires_grad=True)

    total_pixels = b * t * NPIX
    counts = torch.randint(0, 4, (total_pixels,), device=dev)
    cu = torch.zeros(total_pixels + 1, dtype=torch.int32, device=dev)
    cu[1:] = torch.cumsum(counts, 0).to(torch.int32)
    n_tokens = int(cu[-1])
    tokens = torch.randn(n_tokens, otd, device=dev, requires_grad=True)
    cross_attn_kwargs = {
        "tokens": tokens,
        "cu_seqlens_k": cu,
        "max_seqlen_k": int(counts.max()),
    }

    out = model(
        x,
        torch.rand(b, device=dev),
        cross_attn_kwargs=cross_attn_kwargs,
        tokenizer_kwargs=_calendar(b, t, dev),
    )
    assert out.shape == (b, c, t, NPIX)
    assert torch.isfinite(out).all()
    out.float().pow(2).mean().backward()
    assert x.grad.abs().sum() > 0 and tokens.grad.abs().sum() > 0


def _build_cross_attn_kwargs(b, t, npix, obs_token_dim, device, max_count=4):
    """Build PixelCrossAttention's forward kwargs for ``b*t*npix`` pixels."""
    total_pixels = b * t * npix
    counts = torch.randint(0, max_count, (total_pixels,), device=device)
    cu = torch.zeros(total_pixels + 1, dtype=torch.int32, device=device)
    cu[1:] = torch.cumsum(counts, 0).to(torch.int32)
    n_tokens = int(cu[-1].item())
    tokens = torch.randn(n_tokens, obs_token_dim, device=device, requires_grad=True)
    return {
        "tokens": tokens,
        "cu_seqlens_k": cu,
        "max_seqlen_k": int(counts.max().item()) if total_pixels else 0,
    }


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
    b, t, npix, c, num_heads = 2, 4, 16, 64, 4
    block = VideoDiTBlock(
        hidden_size=c,
        num_heads=num_heads,
        condition_embed_dim=32,
        temporal_attention=True,
        is_causal=True,
    )
    # VideoDiT normally owns a single RotaryEmbedding1DTables and passes its
    # tables into every block; build one here since the block is standalone.
    rope = RotaryEmbedding1DTables(head_dim=c // num_heads, max_seq_len=t)
    rope_cos, rope_sin = rope(seq_len=t)
    x = torch.randn(b, t, npix, c, requires_grad=True)
    emb = torch.randn(b, 32)
    out = block(
        x, emb, temporal_attn_kwargs={"rope_cos": rope_cos, "rope_sin": rope_sin}
    )
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

    rope = RotaryEmbedding1DTables(head_dim=c // 8, max_seq_len=t).to(dev)
    rope_cos, rope_sin = rope(seq_len=t)
    x = torch.randn(b, t, npix, c, device=dev, requires_grad=True)
    emb = torch.randn(b, 128, device=dev)
    cross_attn_kwargs = _build_cross_attn_kwargs(b, t, npix, token_dim, dev)

    out = block(
        x,
        emb,
        cross_attn_kwargs=cross_attn_kwargs,
        temporal_attn_kwargs={"rope_cos": rope_cos, "rope_sin": rope_sin},
    )
    assert out.shape == (b, t, npix, c)
    assert torch.isfinite(out).all()

    out.float().pow(2).mean().backward()
    for g in (
        x.grad,
        cross_attn_kwargs["tokens"].grad,
        block.cross_attention.q_proj.weight.grad,
        block.temporal_attention.qkv.weight.grad,
        next(block.attention.parameters()).grad,
    ):
        assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0
