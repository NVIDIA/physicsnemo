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
"""Ragged pixel cross-attention: Triton kernel and pure-PyTorch reference."""

import math

import pytest
import torch

from physicsnemo.core.version_check import OptionalImport
from physicsnemo.experimental.models.healda.obs_context import ObsContext
from physicsnemo.experimental.models.healda.pixel_cross_attention import (
    PixelCrossAttention,
    _pixel_attention_reference,
    build_pixel_group_map,
    pixel_attention,
)

triton = OptionalImport("triton")
# The Triton kernel needs triton + CUDA; the reference path runs anywhere.
requires_triton_cuda = pytest.mark.skipif(
    not (triton.available and torch.cuda.is_available()),
    reason="pixel cross-attention Triton kernel requires triton + CUDA",
)

_ragged_gqa_reference = _pixel_attention_reference

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
    if not (triton.available and torch.cuda.is_available()):
        yield
        return
    from physicsnemo.experimental.models.healda import _pixel_attn_kernels as kernels

    one = [triton.Config({"TILE_K": 32}, num_warps=4, num_stages=2)]
    for kernel in (kernels._pixel_attn_gqa_fwd, kernels._pixel_attn_gqa_bwd):
        monkeypatch.setattr(kernel, "configs", one, raising=False)
        if hasattr(kernel, "cache"):
            kernel.cache.clear()
    yield


def _cu_seqlens(counts):
    cu = torch.zeros(len(counts) + 1, dtype=torch.int32)
    if counts:
        cu[1:] = torch.tensor(counts, dtype=torch.int32).cumsum(0)
    return cu


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


@requires_triton_cuda
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


@requires_triton_cuda
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


@requires_triton_cuda
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


@requires_triton_cuda
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


@requires_triton_cuda
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

    context = ObsContext(tokens=tokens, cu_seqlens_k=cu, max_seqlen_k=max(counts))
    out = module(hidden.view(1, 1, total_pixels, module.input_dim), context)
    assert out.shape == (1, 1, total_pixels, module.output_dim)
    assert torch.isfinite(out).all()

    out.sum().backward()
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
    for name, p in module.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name


@requires_triton_cuda
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

    context = ObsContext(tokens=tokens, cu_seqlens_k=cu, max_seqlen_k=0)
    out = module(hidden.view(1, 1, total_pixels, module.input_dim), context)
    assert out.shape == (1, 1, total_pixels, module.output_dim)
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


def test_pixel_cross_attention_cpu_reference_forward_backward():
    # Pure-PyTorch reference path (no Triton/CUDA): forward shape + full grads.
    torch.manual_seed(0)
    counts = [2, 0, 3, 1, 4, 0]  # total_pixels = b*t*x = 1*2*3
    module = PixelCrossAttention(
        token_dim=TOKEN_DIM,
        n_q_heads=16,
        n_kv_heads=1,
        d_head=D_HEAD,
        use_proj_bias=True,
    )
    gen = torch.Generator().manual_seed(0)
    tokens = torch.randn(sum(counts), TOKEN_DIM, generator=gen, requires_grad=True)
    ctx = ObsContext(
        tokens=tokens, cu_seqlens_k=_cu_seqlens(counts), max_seqlen_k=max(counts)
    )
    hidden = torch.randn(1, 2, 3, module.input_dim, requires_grad=True)

    out = module(hidden, ctx)
    assert out.shape == (1, 2, 3, module.output_dim)
    assert torch.isfinite(out).all()

    out.pow(2).sum().backward()
    assert hidden.grad is not None
    assert tokens.grad is not None
    assert module.q_proj.weight.grad is not None


def test_pixel_cross_attention_cpu_all_empty_keeps_grads():
    # No observations: zero output, but every projection param still gets a grad.
    module = PixelCrossAttention(
        token_dim=TOKEN_DIM,
        n_q_heads=16,
        n_kv_heads=1,
        d_head=D_HEAD,
        use_proj_bias=True,
    )
    ctx = ObsContext(
        tokens=torch.zeros(0, TOKEN_DIM),
        cu_seqlens_k=torch.zeros(5, dtype=torch.int32),
        max_seqlen_k=0,
    )
    hidden = torch.randn(1, 1, 4, module.input_dim, requires_grad=True)

    out = module(hidden, ctx)
    assert out.shape == (1, 1, 4, module.output_dim)
    out.sum().backward()
    assert module.q_proj.weight.grad is not None
    assert module.out_proj.weight.grad is not None
