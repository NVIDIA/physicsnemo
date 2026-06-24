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
"""Ragged pixel cross-attention Triton kernel vs a PyTorch GQA reference."""

import math

import pytest
import torch

triton = pytest.importorskip("triton")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="pixel cross-attention Triton kernel requires CUDA",
)

from physicsnemo.experimental.models.healda import (  # noqa: E402
    _pixel_attn_kernels as pcak,
)
from physicsnemo.experimental.models.healda.obs_packing import (  # noqa: E402
    build_pixel_group_map,
)
from physicsnemo.experimental.models.healda.pixel_cross_attention import (  # noqa: E402
    PixelCrossAttention,
    pixel_attention,
)

# Small power-of-two dims keep every kernel launch tiny and fast.
D_HEAD = 16
TOKEN_DIM = 32

# All layouts use q_per_kv=16, so only kv=1 and kv=2 kernels get compiled
# (kv=4 runs as two kv=2 phases). n_q is the minimum allowed for each kv count.
_HEAD_LAYOUTS = [(16, 1), (32, 2), (64, 4)]


@pytest.fixture(autouse=True)
def _single_autotune_config(monkeypatch):
    # Collapse the autotuner's config sweep to one config so each kernel compiles
    # once. num_warps/num_stages are functionally irrelevant to correctness.
    one = [triton.Config({"TILE_K": 32}, num_warps=4, num_stages=2)]
    for kernel in (pcak._pixel_attn_gqa_fwd, pcak._pixel_attn_gqa_bwd):
        monkeypatch.setattr(kernel, "configs", one, raising=False)
        if hasattr(kernel, "cache"):
            kernel.cache.clear()
    yield


def _cu_seqlens(counts):
    cu = torch.zeros(len(counts) + 1, dtype=torch.int32)
    if counts:
        cu[1:] = torch.tensor(counts, dtype=torch.int32).cumsum(0)
    return cu


def _ragged_gqa_reference(Q, tokens, W_k, W_v, cu, n_kv_heads, scale, B_v=None):
    """Readable per-pixel PyTorch reference for the ragged grouped-query attention."""
    n_pixels, n_q_heads, d_head = Q.shape
    q_per_kv = n_q_heads // n_kv_heads
    out = torch.zeros_like(Q)
    for p in range(n_pixels):
        start, end = int(cu[p]), int(cu[p + 1])
        if end == start:
            continue
        tok = tokens[start:end]
        K = (tok @ W_k.t()).view(-1, n_kv_heads, d_head)
        V = tok @ W_v.t()
        if B_v is not None:
            V = V + B_v
        V = V.view(-1, n_kv_heads, d_head)
        for h in range(n_q_heads):
            kv = h // q_per_kv
            scores = (K[:, kv] @ Q[p, h]) * scale
            weights = torch.softmax(scores, dim=0)
            out[p, h] = weights @ V[:, kv]
    return out


def _make_inputs(counts, n_q_heads, n_kv_heads, use_v_bias, seed=0):
    gen = torch.Generator().manual_seed(seed)
    kv_dim = n_kv_heads * D_HEAD
    Q = torch.randn(len(counts), n_q_heads, D_HEAD, generator=gen)
    tokens = torch.randn(sum(counts), TOKEN_DIM, generator=gen)
    W_k = torch.randn(kv_dim, TOKEN_DIM, generator=gen) * 0.1
    W_v = torch.randn(kv_dim, TOKEN_DIM, generator=gen) * 0.1
    B_v = torch.randn(kv_dim, generator=gen) * 0.1 if use_v_bias else None
    cu = _cu_seqlens(counts)
    max_seqlen_k = max(counts) if counts else 0

    def cuda(x):
        return None if x is None else x.cuda()

    return (
        cuda(Q),
        cuda(tokens),
        cuda(W_k),
        cuda(W_v),
        cuda(B_v),
        cu.cuda(),
        max_seqlen_k,
    )


def _assert_scale_close(actual, ref, rtol, name=""):
    # tl.dot accumulates in TF32 (not IEEE fp32), which -- plus a few near-zero
    # entries -- makes per-element relative error noisy, so validate against the
    # tensor's overall scale instead.
    scale = ref.abs().max().clamp_min(1e-6)
    max_abs_diff = (actual - ref).abs().max()
    assert max_abs_diff <= rtol * scale, (
        f"{name}: max_abs_diff={max_abs_diff.item():.3e} exceeds {rtol} * "
        f"scale ({scale.item():.3e})"
    )


@pytest.mark.parametrize("n_q_heads,n_kv_heads", _HEAD_LAYOUTS)
@pytest.mark.parametrize("use_v_bias", [False, True])
def test_pixel_attention_forward(n_q_heads, n_kv_heads, use_v_bias):
    # Mixed ragged groups: empty, singleton, and multi-token pixels.
    counts = [0, 1, 5, 0, 12, 3]
    Q, tokens, W_k, W_v, B_v, cu, max_seqlen_k = _make_inputs(
        counts, n_q_heads, n_kv_heads, use_v_bias
    )
    scale = 1.0 / math.sqrt(D_HEAD)

    out = pixel_attention(
        Q,
        tokens,
        W_k,
        W_v,
        cu,
        max_seqlen_k,
        n_kv_heads=n_kv_heads,
        scale=scale,
        B_v=B_v,
        force_fp32=True,
    )
    ref = _ragged_gqa_reference(Q, tokens, W_k, W_v, cu, n_kv_heads, scale, B_v=B_v)

    assert out.shape == ref.shape
    assert torch.count_nonzero(out[0]) == 0  # empty pixel -> zero output
    assert torch.count_nonzero(out[3]) == 0
    _assert_scale_close(out, ref, rtol=5e-3, name="forward")


def test_pixel_attention_packed_full_grid():
    # Packed full-grid layout: many pixels, almost all with zero observations.
    counts = [0] * 120
    for idx, c in [(5, 12), (37, 1), (90, 8)]:
        counts[idx] = c
    Q, tokens, W_k, W_v, B_v, cu, max_seqlen_k = _make_inputs(
        counts, 32, 2, use_v_bias=True, seed=1
    )
    scale = 1.0 / math.sqrt(D_HEAD)

    out = pixel_attention(
        Q,
        tokens,
        W_k,
        W_v,
        cu,
        max_seqlen_k,
        n_kv_heads=2,
        scale=scale,
        B_v=B_v,
        force_fp32=True,
    )
    ref = _ragged_gqa_reference(Q, tokens, W_k, W_v, cu, 2, scale, B_v=B_v)
    _assert_scale_close(out, ref, rtol=5e-3, name="packed_full_grid")
    assert torch.count_nonzero(out[90]) > 0
    assert torch.count_nonzero(out[0]) == 0


@pytest.mark.parametrize("n_q_heads,n_kv_heads", _HEAD_LAYOUTS)
def test_pixel_attention_backward(n_q_heads, n_kv_heads):
    counts = [0, 2, 9, 1, 6]
    Q, tokens, W_k, W_v, B_v, cu, max_seqlen_k = _make_inputs(
        counts, n_q_heads, n_kv_heads, use_v_bias=True, seed=3
    )
    scale = 1.0 / math.sqrt(D_HEAD)
    grad_out = torch.randn(Q.shape, generator=torch.Generator().manual_seed(7)).cuda()

    def grads_for(fn):
        leaves = {
            "Q": Q.clone().detach().requires_grad_(True),
            "tokens": tokens.clone().detach().requires_grad_(True),
            "W_k": W_k.clone().detach().requires_grad_(True),
            "W_v": W_v.clone().detach().requires_grad_(True),
            "B_v": B_v.clone().detach().requires_grad_(True),
        }
        out = fn(**leaves)
        (out * grad_out).sum().backward()
        return {k: v.grad for k, v in leaves.items()}

    triton_grads = grads_for(
        lambda Q, tokens, W_k, W_v, B_v: pixel_attention(
            Q,
            tokens,
            W_k,
            W_v,
            cu,
            max_seqlen_k,
            n_kv_heads=n_kv_heads,
            scale=scale,
            B_v=B_v,
            force_fp32=True,
        )
    )
    ref_grads = grads_for(
        lambda Q, tokens, W_k, W_v, B_v: _ragged_gqa_reference(
            Q, tokens, W_k, W_v, cu, n_kv_heads, scale, B_v=B_v
        )
    )
    for name in ref_grads:
        _assert_scale_close(
            triton_grads[name], ref_grads[name], rtol=2e-2, name=f"grad_{name}"
        )


@pytest.mark.parametrize("n_q_heads,n_kv_heads", _HEAD_LAYOUTS)
def test_pixel_attention_grouping_matches_ungrouped(n_q_heads, n_kv_heads):
    # Small-pixel grouping packs several pixels into one kernel program via a CSR
    # map but computes the identical math, so grouped output AND every gradient
    # must match the ungrouped (one-program-per-pixel) path bit-for-bit.
    counts = [3, 2, 4, 1, 0, 5, 2, 3, 30, 1, 2, 4, 0, 3, 2, 40, 1, 2]
    Q, tokens, W_k, W_v, B_v, cu, max_seqlen_k = _make_inputs(
        counts, n_q_heads, n_kv_heads, use_v_bias=True, seed=11
    )
    scale = 1.0 / math.sqrt(D_HEAD)
    group_map = build_pixel_group_map(cu)
    # Sanity: the map must actually group (fewer programs than nonzero pixels).
    n_nz = int((cu[1:] > cu[:-1]).sum())
    assert group_map.program_ptr.numel() - 1 < n_nz
    grad_out = torch.randn(Q.shape, generator=torch.Generator().manual_seed(13)).cuda()

    def grads_for(group_map):
        leaves = {
            "Q": Q.clone().detach().requires_grad_(True),
            "tokens": tokens.clone().detach().requires_grad_(True),
            "W_k": W_k.clone().detach().requires_grad_(True),
            "W_v": W_v.clone().detach().requires_grad_(True),
            "B_v": B_v.clone().detach().requires_grad_(True),
        }
        out = pixel_attention(
            leaves["Q"],
            leaves["tokens"],
            leaves["W_k"],
            leaves["W_v"],
            cu,
            max_seqlen_k,
            n_kv_heads=n_kv_heads,
            scale=scale,
            B_v=leaves["B_v"],
            force_fp32=True,
            group_map=group_map,
        )
        (out * grad_out).sum().backward()
        return out, {k: v.grad for k, v in leaves.items()}

    ungrouped_output, ungrouped_grads = grads_for(None)
    grouped_output, grouped_grads = grads_for(group_map)
    torch.testing.assert_close(grouped_output, ungrouped_output, rtol=0, atol=0)
    for name in ungrouped_grads:
        torch.testing.assert_close(
            grouped_grads[name], ungrouped_grads[name], rtol=0, atol=0
        )


def test_pixel_cross_attention_module_forward_backward():
    # Exercises the nn.Module wiring (q_proj/out_proj + reshapes) in the real
    # bf16 path. Smoke-checks shapes, finiteness, and full gradient coverage.
    torch.manual_seed(4)
    total_pixels = 20
    counts = [0, 3, 1, 0] * 5
    assert len(counts) == total_pixels
    module = PixelCrossAttention(
        token_dim=TOKEN_DIM, n_q_heads=32, n_kv_heads=2, d_head=D_HEAD
    ).cuda()

    gen = torch.Generator().manual_seed(5)
    hidden = (
        torch.randn(total_pixels, module.input_dim, generator=gen)
        .cuda()
        .requires_grad_(True)
    )
    tokens = (
        torch.randn(sum(counts), TOKEN_DIM, generator=gen).cuda().requires_grad_(True)
    )
    cu = _cu_seqlens(counts).cuda()

    out = module(hidden, tokens, total_pixels, cu, max(counts))
    assert out.shape == (total_pixels, module.output_dim)
    assert torch.isfinite(out).all()

    out.sum().backward()
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
    for name, p in module.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name


def test_pixel_cross_attention_empty_tokens_grad():
    # No observations anywhere (this path skips the kernel entirely): every
    # projection param must still get a (zero, finite) gradient so DDP stays in
    # lockstep across ranks.
    torch.manual_seed(6)
    total_pixels = 12
    module = PixelCrossAttention(
        token_dim=TOKEN_DIM, n_q_heads=32, n_kv_heads=2, d_head=D_HEAD
    ).cuda()
    hidden = torch.randn(total_pixels, module.input_dim).cuda()
    tokens = torch.zeros(0, TOKEN_DIM).cuda()
    cu = torch.zeros(total_pixels + 1, dtype=torch.int32).cuda()

    out = module(hidden, tokens, total_pixels, cu, 0)
    assert out.shape == (total_pixels, module.output_dim)
    out.sum().backward()
    for name, p in module.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name


def test_pixel_cross_attention_rejects_unsupported_configs():
    # Document the supported head layout: q_per_kv >= 16, n_kv_heads in {1,2,even},
    # n_q_heads divisible by n_kv_heads. These raise at construction, no kernel.
    with pytest.raises(ValueError, match="below Triton tl.dot minimum"):
        PixelCrossAttention(
            token_dim=TOKEN_DIM, n_q_heads=8, n_kv_heads=1, d_head=D_HEAD
        )
    with pytest.raises(ValueError, match="n_kv_heads"):
        PixelCrossAttention(
            token_dim=TOKEN_DIM, n_q_heads=64, n_kv_heads=3, d_head=D_HEAD
        )
    with pytest.raises(ValueError, match="divisible"):
        PixelCrossAttention(
            token_dim=TOKEN_DIM, n_q_heads=66, n_kv_heads=4, d_head=D_HEAD
        )
