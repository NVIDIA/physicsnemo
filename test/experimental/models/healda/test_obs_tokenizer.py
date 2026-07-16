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
"""Fused FiLM observation tokenizer Triton kernel vs a PyTorch reference."""

import pytest
import torch

from physicsnemo.experimental.models.healda.obs_tokenizer import (
    ObsTokenizerFiLM,
)

# Small dims keep every kernel launch tiny and fast.
META_DIM = 12
OUT_DIM = 16
OBS_TYPE_EMBED_DIM = 4
CHANNEL_EMBED_DIM = 4
PLATFORM_EMBED_DIM = 4
N_EMBED = 32


def _film_reference(
    module: ObsTokenizerFiLM, obs, float_metadata, obs_type, channel, platform
):
    """Readable pure-PyTorch reference for the FiLM tokenizer math."""
    embed_vec = module.embed_table(obs_type)
    chan_emb = module.channel_embedding(channel)
    parts = [float_metadata, embed_vec, chan_emb]
    if module.use_platform_embedding:
        parts.append(module.platform_embedding(platform))
    conditioning = torch.cat(parts, dim=-1)
    ab = module.cond_mlp(conditioning)
    alpha, beta = ab.chunk(2, dim=-1)
    return alpha * obs.unsqueeze(-1) + beta


def _make_module(use_platform: bool, seed: int = 0) -> ObsTokenizerFiLM:
    torch.manual_seed(seed)
    return ObsTokenizerFiLM(
        meta_dim=META_DIM,
        out_dim=OUT_DIM,
        n_embed=N_EMBED,
        obs_type_embed_dim=OBS_TYPE_EMBED_DIM,
        channel_embed_dim=CHANNEL_EMBED_DIM,
        platform_embed_dim=PLATFORM_EMBED_DIM if use_platform else 0,
    )


def _make_inputs(n: int, use_platform: bool, device: str, seed: int = 1):
    gen = torch.Generator().manual_seed(seed)
    obs = torch.randn(n, generator=gen)
    float_metadata = torch.randn(n, META_DIM, generator=gen)
    obs_type = torch.randint(0, N_EMBED, (n,), generator=gen)
    channel = torch.randint(0, 64, (n,), generator=gen)
    platform = torch.randint(0, 64, (n,), generator=gen) if use_platform else None

    def to(x):
        return None if x is None else x.to(device)

    return to(obs), to(float_metadata), to(obs_type), to(channel), to(platform)


@pytest.mark.parametrize("use_platform", [False, True])
def test_film_tokenizer_cpu_reference_smoke(use_platform):
    # CPU has no triton, so forward() takes the pure-PyTorch reference branch.
    module = _make_module(use_platform)
    obs, float_metadata, obs_type, channel, platform = _make_inputs(
        17, use_platform, device="cpu"
    )

    out = module(obs, float_metadata, obs_type, channel, platform)
    assert out.shape == (17, OUT_DIM)
    assert torch.isfinite(out).all()

    # The module's reference branch must match the standalone reference math.
    ref = _film_reference(module, obs, float_metadata, obs_type, channel, platform)
    torch.testing.assert_close(out, ref)

    # Backward should reach every learned parameter.
    out.sum().backward()
    for name, p in module.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="fused FiLM tokenizer Triton kernel requires CUDA",
)
@pytest.mark.parametrize("use_platform", [False, True])
def test_film_tokenizer_triton_matches_reference(use_platform):
    pytest.importorskip("triton")
    from physicsnemo.experimental.models.healda.obs_tokenizer import (
        _fused_film_tokenizer_triton,
    )

    module = _make_module(use_platform).cuda()
    obs, float_metadata, obs_type, channel, platform = _make_inputs(
        130, use_platform, device="cuda"
    )

    # fp32 kernel path so the comparison against the fp32 reference is tight.
    triton_out = _fused_film_tokenizer_triton(
        obs,
        float_metadata,
        obs_type,
        channel,
        platform if use_platform else None,
        module.embed_table,
        module.channel_embedding,
        module.platform_embedding,
        module.cond_mlp[0],
        module.cond_mlp[1],
        module.cond_mlp[3],
        eps=module.cond_mlp[1].eps,
        force_fp32=True,
    )
    ref = _film_reference(module, obs, float_metadata, obs_type, channel, platform)

    assert triton_out.shape == ref.shape
    torch.testing.assert_close(triton_out, ref, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="fused FiLM tokenizer Triton kernel requires CUDA",
)
@pytest.mark.parametrize("use_platform", [False, True])
def test_film_tokenizer_module_cuda_forward_backward(use_platform):
    pytest.importorskip("triton")
    # Exercises the nn.Module wiring on the fused bf16 path: shapes, finiteness,
    # and full gradient coverage.
    module = _make_module(use_platform).cuda()
    obs, float_metadata, obs_type, channel, platform = _make_inputs(
        130, use_platform, device="cuda"
    )
    obs = obs.requires_grad_(True)

    out = module(obs, float_metadata, obs_type, channel, platform)
    assert out.shape == (130, OUT_DIM)
    assert torch.isfinite(out).all()

    out.sum().backward()
    for name, p in module.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
