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
"""Construction + forward smoke tests for the VideoHealDA video+obs DA model."""

import pytest
import torch

pytest.importorskip("earth2grid")  # HEALPix tokenizer dependency

from physicsnemo.experimental.models.healda.obs_context import (  # noqa: E402
    prepare_obs_context,
)
from physicsnemo.experimental.models.healda.video_healda import (  # noqa: E402
    VideoHealDA,
)

LEVEL_FINE = 2
LEVEL_COARSE = 1
NPIX = 12 * 4**LEVEL_FINE  # 192
NPIX_COARSE = 12 * 4**LEVEL_COARSE  # 48


def _build_model(device):
    # q_per_kv = n_q_heads / n_kv_heads must be >= 16 for the cross-attn kernel.
    return VideoHealDA(
        in_channels=2,
        out_channels=3,
        hidden_size=64,
        num_layers=2,
        num_heads=2,
        level_in=LEVEL_FINE,
        level_model=LEVEL_COARSE,
        time_length=2,
        condition_embed_dim=32,
        noise_channels=32,
        obs_token_dim=16,
        obs_meta_dim=8,
        obs_type_embed_dim=4,
        channel_embed_dim=4,
        platform_embed_dim=4,
        obs_film_hidden_dim=32,
        pixel_attn_n_q_heads=32,
        pixel_attn_n_kv_heads=2,
        pixel_attn_head_dim=16,
    ).to(device)


def _calendar(b, t, device):
    sod = torch.rand(b, t, device=device) * 86400.0
    doy = torch.rand(b, t, device=device) * 365.0
    return sod, doy


def test_video_healda_spatial_qk_norm_parameter_free():
    """VideoHealDA's attn_kwargs engage affine-free RMSNorm q/k norm in spatial attn.

    Guards against the kwarg names being silently swallowed (which would leave
    qk-norm off): the q/k norm modules must exist (not Identity) and carry no
    learnable affine parameters.
    """
    model = _build_model("cpu")
    attn_op = model.dit.blocks[0].attention.attn_op
    for norm in (attn_op.q_norm, attn_op.k_norm):
        assert not isinstance(norm, torch.nn.Identity)
        assert list(norm.parameters()) == []


def test_video_healda_cpu_no_obs():
    """Full graph on CPU with an empty observation set (Triton-free obs path)."""
    torch.manual_seed(0)
    dev = "cpu"
    b, c, t = 2, 2, 2
    model = _build_model(dev)

    total_pixels = b * t * NPIX_COARSE
    x = torch.randn(b, c, t, NPIX, device=dev, requires_grad=True)
    sod, doy = _calendar(b, t, dev)
    empty_obs = torch.empty(0, device=dev)
    empty_ids = torch.empty(0, dtype=torch.int64, device=dev)

    obs_ctx = prepare_obs_context(
        obs=empty_obs,
        float_metadata=torch.empty(0, 8, device=dev),
        obs_type=empty_ids,
        channel=empty_ids,
        platform=empty_ids,
        flat_idx=torch.empty(0, dtype=torch.int32, device=dev),
        total_pixels=total_pixels,
    )
    out = model(x, torch.rand(b, device=dev), sod, doy, obs_ctx)

    assert out.shape == (b, 3, t, NPIX)
    out.float().pow(2).mean().backward()
    assert torch.isfinite(x.grad).all()

    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="obs cross-attn Triton kernel is CUDA-only"
)
def test_video_healda_cuda_full():
    """Full obs path (FiLM tokenizer + ragged cross-attn kernel) on CUDA."""
    pytest.importorskip("triton")
    torch.manual_seed(0)
    dev = "cuda"
    b, c, t = 1, 2, 2
    model = _build_model(dev)

    total_pixels = b * t * NPIX_COARSE
    # Scatter a handful of observations across a few pixels (pixel-sorted).
    counts = [0] * total_pixels
    counts[0] = 3
    counts[5] = 2
    counts[40] = 4
    nobs = sum(counts)

    x = torch.randn(b, c, t, NPIX, device=dev, requires_grad=True)
    sod, doy = _calendar(b, t, dev)
    obs = torch.randn(nobs, device=dev)
    float_metadata = torch.randn(nobs, 8, device=dev)
    obs_type = torch.randint(0, 4, (nobs,), device=dev)
    channel = torch.randint(0, 8, (nobs,), device=dev)
    platform = torch.randint(0, 4, (nobs,), device=dev)

    flat_idx = torch.tensor(
        [pix for pix, count in enumerate(counts) for _ in range(count)],
        dtype=torch.int32,
        device=dev,
    )
    obs_ctx = prepare_obs_context(
        obs=obs,
        float_metadata=float_metadata,
        obs_type=obs_type,
        channel=channel,
        platform=platform,
        flat_idx=flat_idx,
        total_pixels=total_pixels,
    )
    out = model(x, torch.rand(b, device=dev), sod, doy, obs_ctx)

    assert out.shape == (b, 3, t, NPIX)
    out.float().pow(2).mean().backward()
    assert torch.isfinite(x.grad).all()
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
