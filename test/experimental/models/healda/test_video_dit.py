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

from physicsnemo.experimental.models.healda.obs_packing import (  # noqa: E402
    ObsCrossAttention,
)
from physicsnemo.experimental.models.healda.video_dit import VideoDiT  # noqa: E402

LEVEL_FINE = 2
LEVEL_COARSE = 1
NPIX = 12 * 4**LEVEL_FINE  # 192
NPIX_COARSE = 12 * 4**LEVEL_COARSE  # 48


def _calendar(b, t, device):
    sod = torch.rand(b, t, device=device) * 86400.0
    doy = torch.rand(b, t, device=device) * 365.0
    return sod, doy


def test_video_dit_cpu_temporal():
    """Dense + temporal path (no obs) -- forward/backward shapes on CPU."""
    torch.manual_seed(0)
    b, c, t, hidden = 2, 3, 2, 64
    model = VideoDiT(
        in_channels=c,
        out_channels=c,
        level_fine=LEVEL_FINE,
        level_coarse=LEVEL_COARSE,
        time_length=t,
        hidden_size=hidden,
        num_heads=4,
        num_layers=2,
        temporal_attention=True,
    )
    x = torch.randn(b, c, t, NPIX, requires_grad=True)
    sod, doy = _calendar(b, t, "cpu")
    out = model(x, torch.rand(b), sod, doy, is_causal=True)
    assert out.shape == (b, c, t, NPIX)
    out.float().pow(2).mean().backward()
    assert torch.isfinite(x.grad).all()


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="triton obs attn is CUDA-only"
)
def test_video_dit_cuda_full():
    """Dense + temporal + observation cross-attention on CUDA."""
    torch.manual_seed(0)
    dev = "cuda"
    b, c, t, hidden, otd = 2, 3, 2, 256, 16
    model = VideoDiT(
        in_channels=c,
        out_channels=c,
        level_fine=LEVEL_FINE,
        level_coarse=LEVEL_COARSE,
        time_length=t,
        hidden_size=hidden,
        num_heads=8,
        num_layers=2,
        temporal_attention=True,
        obs_cross_attention=True,
        obs_token_dim=otd,
    ).to(dev)

    x = torch.randn(b, c, t, NPIX, device=dev, requires_grad=True)
    sod, doy = _calendar(b, t, dev)

    total_pixels = b * t * NPIX_COARSE
    counts = torch.randint(0, 4, (total_pixels,), device=dev)
    cu = torch.zeros(total_pixels + 1, dtype=torch.int32, device=dev)
    cu[1:] = torch.cumsum(counts, 0).to(torch.int32)
    tokens = torch.randn(int(cu[-1]), otd, device=dev, requires_grad=True)
    obs = ObsCrossAttention(
        tokens=tokens, cu_seqlens_k=cu, max_seqlen_k=int(counts.max())
    )

    out = model(x, torch.rand(b, device=dev), sod, doy, obs=obs)
    assert out.shape == (b, c, t, NPIX)
    assert torch.isfinite(out).all()
    out.float().pow(2).mean().backward()
    assert x.grad.abs().sum() > 0 and tokens.grad.abs().sum() > 0
