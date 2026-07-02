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

pytest.importorskip("earth2grid")  # HEALPix tokenizer dependency

from physicsnemo.experimental.models.healda.obs_context import (  # noqa: E402
    ObsContext,
)
from physicsnemo.experimental.models.healda.pixel_cross_attention import (  # noqa: E402
    PixelCrossAttention,
)
from physicsnemo.experimental.models.healda.video_dit import VideoDiT  # noqa: E402
from physicsnemo.nn.module.hpx.tokenizer import (  # noqa: E402
    HEALPixPatchDetokenizer,
    HEALPixPatchTokenizer,
)

LEVEL_FINE = 2
LEVEL_COARSE = 1
NPIX = 12 * 4**LEVEL_FINE  # 192
NPIX_COARSE = 12 * 4**LEVEL_COARSE  # 48


def _build_model(c, t, hidden, num_heads, device, **kwargs):
    emb_channels = 4 * hidden
    tokenizer = HEALPixPatchTokenizer(
        in_channels=c,
        hidden_size=hidden,
        level_fine=LEVEL_FINE,
        level_coarse=LEVEL_COARSE,
        separate_time_axis=True,
    )
    detokenizer = HEALPixPatchDetokenizer(
        hidden_size=hidden,
        out_channels=c,
        level_coarse=LEVEL_COARSE,
        level_fine=LEVEL_FINE,
        condition_dim=emb_channels,
    )
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

    total_pixels = b * t * NPIX_COARSE
    counts = torch.randint(0, 4, (total_pixels,), device=dev)
    cu = torch.zeros(total_pixels + 1, dtype=torch.int32, device=dev)
    cu[1:] = torch.cumsum(counts, 0).to(torch.int32)
    n_tokens = int(cu[-1])
    tokens = torch.randn(n_tokens, otd, device=dev, requires_grad=True)
    # VideoDiT's cross-attention only reads tokens/cu_seqlens_k/max_seqlen_k; the
    # raw per-observation fields (required on ObsContext) are unused placeholders.
    context = ObsContext(
        tokens=tokens,
        cu_seqlens_k=cu,
        max_seqlen_k=int(counts.max()),
        obs=torch.randn(n_tokens, device=dev),
        float_metadata=torch.randn(n_tokens, 1, device=dev),
        obs_type=torch.randint(0, 4, (n_tokens,), device=dev),
        channel=torch.randint(0, 4, (n_tokens,), device=dev),
        platform=torch.randint(0, 4, (n_tokens,), device=dev),
    )

    out = model(
        x,
        torch.rand(b, device=dev),
        cross_attention_context=context,
        tokenizer_kwargs=_calendar(b, t, dev),
    )
    assert out.shape == (b, c, t, NPIX)
    assert torch.isfinite(out).all()
    out.float().pow(2).mean().backward()
    assert x.grad.abs().sum() > 0 and tokens.grad.abs().sum() > 0
