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
"""Ragged pixel cross-attention (Triton kernel + pure-PyTorch reference) and its
ObsContext packing utilities."""

import math

import pytest
import torch

from physicsnemo.core.version_check import OptionalImport
from physicsnemo.experimental.models.healda.attention_layers import (
    PixelCrossAttention,
    _pixel_attention_reference,
)
from physicsnemo.experimental.models.healda.obs_context import (
    build_pixel_group_map,
    counts_to_cu_seqlens,
    prepare_obs_context,
    sort_and_pack,
)

triton = OptionalImport("triton")
# The Triton kernel needs triton + CUDA; the reference path runs anywhere.
requires_triton_cuda = pytest.mark.skipif(
    not (triton.available and torch.cuda.is_available()),
    reason="pixel cross-attention Triton kernel requires triton + CUDA",
)

if triton.available:
    # kernels.pixel_attention imports triton eagerly, so only import it here.
    from physicsnemo.experimental.models.healda.kernels.pixel_attention import (
        pixel_attention,
    )

_ragged_gqa_reference = _pixel_attention_reference

# Small power-of-two dims keep every kernel launch tiny and fast.
D_HEAD = 16
TOKEN_DIM = 32

# All layouts use q_per_kv=16, so only kv=1 and kv=2 kernels get compiled
# (kv=4 runs as two kv=2 phases). n_q is the minimum allowed for each kv count.
_HEAD_LAYOUTS = [(16, 1), (32, 2), (64, 4)]


def _cross_attn_kwargs(tokens, counts, device=None, build_group_map=False):
    """Build PixelCrossAttention's forward kwargs for the given per-pixel counts."""
    cu = counts_to_cu_seqlens(torch.tensor(counts, dtype=torch.int64))
    if device is not None:
        cu = cu.to(device)
    kwargs = {
        "tokens": tokens,
        "cu_seqlens_k": cu,
        "max_seqlen_k": max(counts) if counts else 0,
    }
    if build_group_map:
        kwargs["group_map"] = build_pixel_group_map(cu)
    return kwargs


@pytest.fixture(autouse=True)
def _single_autotune_config(monkeypatch):
    # To reduce overhead from triton autotuning when running tests, collapse the autotuner's
    # config sweep to one config.
    if not (triton.available and torch.cuda.is_available()):
        yield
        return
    from physicsnemo.experimental.models.healda.kernels import (
        pixel_attention as pixel_attn_kernels,
    )

    one = [triton.Config({"TILE_K": 32}, num_warps=4, num_stages=2)]
    for kernel in (
        pixel_attn_kernels._pixel_attn_gqa_fwd,
        pixel_attn_kernels._pixel_attn_gqa_bwd,
    ):
        monkeypatch.setattr(kernel, "configs", one, raising=False)
        if hasattr(kernel, "cache"):
            kernel.cache.clear()
    yield


def _make_inputs(counts, n_q_heads, n_kv_heads, use_v_bias, seed=0):
    gen = torch.Generator().manual_seed(seed)
    kv_dim = n_kv_heads * D_HEAD
    Q = torch.randn(len(counts), n_q_heads, D_HEAD, generator=gen)
    tokens = torch.randn(sum(counts), TOKEN_DIM, generator=gen)
    W_k = torch.randn(kv_dim, TOKEN_DIM, generator=gen) * 0.1
    W_v = torch.randn(kv_dim, TOKEN_DIM, generator=gen) * 0.1
    B_v = torch.randn(kv_dim, generator=gen) * 0.1 if use_v_bias else None
    cu = counts_to_cu_seqlens(torch.tensor(counts, dtype=torch.int64))
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
def test_pixel_cross_attention_module_grouping_matches_ungrouped():
    # Small-pixel grouping must be transparent through the nn.Module wrapper:
    # passing a group_map only changes kernel launches, so module output AND
    # every parameter gradient must match the ungrouped path bit-for-bit.
    torch.manual_seed(9)
    counts = [3, 2, 4, 1, 0, 5, 2, 3, 30, 1, 2, 4, 0, 3, 2, 40, 1, 2]
    total_pixels = len(counts)
    module = PixelCrossAttention(
        hidden_size=64, token_dim=TOKEN_DIM, n_q_heads=32, n_kv_heads=2, d_head=D_HEAD
    ).cuda()

    gen = torch.Generator().manual_seed(10)
    hidden = torch.randn(1, 1, total_pixels, module.hidden_size, generator=gen).cuda()
    tokens = torch.randn(sum(counts), TOKEN_DIM, generator=gen).cuda()
    grouped_kwargs = _cross_attn_kwargs(
        tokens, counts, device="cuda", build_group_map=True
    )
    # Sanity: the map must actually group (fewer programs than nonzero pixels).
    n_nz = sum(1 for c in counts if c > 0)
    assert grouped_kwargs["group_map"].program_ptr.numel() - 1 < n_nz

    def out_and_grads(kwargs):
        module.zero_grad(set_to_none=True)
        h = hidden.clone().requires_grad_(True)
        out = module(h, **kwargs)
        out.sum().backward()
        grads = {name: p.grad.clone() for name, p in module.named_parameters()}
        return out, h.grad.clone(), grads

    ungrouped_kwargs = _cross_attn_kwargs(tokens, counts, device="cuda")
    ung_out, ung_hgrad, ung_grads = out_and_grads(ungrouped_kwargs)
    grp_out, grp_hgrad, grp_grads = out_and_grads(grouped_kwargs)

    torch.testing.assert_close(grp_out, ung_out, rtol=0, atol=0)
    torch.testing.assert_close(grp_hgrad, ung_hgrad, rtol=0, atol=0)
    for name in ung_grads:
        torch.testing.assert_close(grp_grads[name], ung_grads[name], rtol=0, atol=0)


@requires_triton_cuda
def test_pixel_cross_attention_module_forward_backward():
    # Exercises the nn.Module wiring (q_proj/out_proj + reshapes) in the real
    # bf16 path. Smoke-checks shapes, finiteness, and full gradient coverage.
    torch.manual_seed(4)
    total_pixels = 20
    counts = [0, 3, 1, 0] * 5
    assert len(counts) == total_pixels
    module = PixelCrossAttention(
        hidden_size=64, token_dim=TOKEN_DIM, n_q_heads=32, n_kv_heads=2, d_head=D_HEAD
    ).cuda()

    gen = torch.Generator().manual_seed(5)
    hidden = (
        torch.randn(total_pixels, module.hidden_size, generator=gen)
        .cuda()
        .requires_grad_(True)
    )
    tokens = (
        torch.randn(sum(counts), TOKEN_DIM, generator=gen).cuda().requires_grad_(True)
    )
    kwargs = _cross_attn_kwargs(tokens, counts, device="cuda")
    out = module(hidden.view(1, 1, total_pixels, module.hidden_size), **kwargs)
    assert out.shape == (1, 1, total_pixels, module.hidden_size)
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
        hidden_size=64, token_dim=TOKEN_DIM, n_q_heads=32, n_kv_heads=2, d_head=D_HEAD
    ).cuda()
    hidden = torch.randn(total_pixels, module.hidden_size).cuda()
    tokens = torch.zeros(0, TOKEN_DIM).cuda()
    kwargs = _cross_attn_kwargs(tokens, [0] * total_pixels, device="cuda")
    out = module(hidden.view(1, 1, total_pixels, module.hidden_size), **kwargs)
    assert out.shape == (1, 1, total_pixels, module.hidden_size)
    out.sum().backward()
    for name, p in module.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name


def test_pixel_cross_attention_rejects_unsupported_configs():
    # Document the supported head layout: q_per_kv >= 16, n_kv_heads in {1,2,even},
    # n_q_heads divisible by n_kv_heads. These raise at construction, no kernel.
    with pytest.raises(ValueError, match="below Triton tl.dot minimum"):
        PixelCrossAttention(
            hidden_size=64,
            token_dim=TOKEN_DIM,
            n_q_heads=8,
            n_kv_heads=1,
            d_head=D_HEAD,
        )
    with pytest.raises(ValueError, match="n_kv_heads"):
        PixelCrossAttention(
            hidden_size=64,
            token_dim=TOKEN_DIM,
            n_q_heads=64,
            n_kv_heads=3,
            d_head=D_HEAD,
        )
    with pytest.raises(ValueError, match="divisible"):
        PixelCrossAttention(
            hidden_size=64,
            token_dim=TOKEN_DIM,
            n_q_heads=66,
            n_kv_heads=4,
            d_head=D_HEAD,
        )


def test_pixel_cross_attention_cpu_reference_forward_backward():
    # Pure-PyTorch reference path (no Triton/CUDA): forward shape + full grads.
    torch.manual_seed(0)
    counts = [2, 0, 3, 1, 4, 0]  # total_pixels = b*t*x = 1*2*3
    module = PixelCrossAttention(
        hidden_size=64,
        token_dim=TOKEN_DIM,
        n_q_heads=16,
        n_kv_heads=1,
        d_head=D_HEAD,
        use_proj_bias=True,
    )
    gen = torch.Generator().manual_seed(0)
    tokens = torch.randn(sum(counts), TOKEN_DIM, generator=gen, requires_grad=True)
    kwargs = _cross_attn_kwargs(tokens, counts)
    hidden = torch.randn(1, 2, 3, module.hidden_size, requires_grad=True)

    out = module(hidden, **kwargs)
    assert out.shape == (1, 2, 3, module.hidden_size)
    assert torch.isfinite(out).all()

    out.pow(2).sum().backward()
    assert hidden.grad is not None
    assert tokens.grad is not None
    assert module.q_proj.weight.grad is not None


def test_pixel_cross_attention_hidden_size_differs_from_attn_dim():
    # hidden_size (residual width) != attn_dim (n_q_heads * d_head, the internal
    # attention width): q_proj maps up and out_proj maps back, so the layer stays
    # shape-preserving on the residual stream. Covers populated and all-empty
    # (kernel-skipped) paths on CPU.
    hidden_size = 24
    module = PixelCrossAttention(
        hidden_size=hidden_size,
        token_dim=TOKEN_DIM,
        n_q_heads=16,
        n_kv_heads=1,
        d_head=D_HEAD,
    )
    assert module.hidden_size == hidden_size
    assert module.attn_dim == 16 * D_HEAD
    assert module.hidden_size != module.attn_dim

    gen = torch.Generator().manual_seed(0)
    for counts in ([2, 0, 3, 1], [0, 0, 0, 0]):
        tokens = torch.randn(sum(counts), TOKEN_DIM, generator=gen)
        kwargs = _cross_attn_kwargs(tokens, counts)
        hidden = torch.randn(1, 1, len(counts), hidden_size, requires_grad=True)
        out = module(hidden, **kwargs)
        assert out.shape == (1, 1, len(counts), hidden_size)
        assert torch.isfinite(out).all()
        out.sum().backward()
        assert module.q_proj.weight.grad is not None
        assert module.out_proj.weight.grad is not None


def test_pixel_cross_attention_cpu_all_empty_keeps_grads():
    # No observations: zero output, but every projection param still gets a grad.
    module = PixelCrossAttention(
        hidden_size=64,
        token_dim=TOKEN_DIM,
        n_q_heads=16,
        n_kv_heads=1,
        d_head=D_HEAD,
        use_proj_bias=True,
    )
    tokens = torch.zeros(0, TOKEN_DIM)
    kwargs = _cross_attn_kwargs(tokens, [0, 0, 0, 0])
    hidden = torch.randn(1, 1, 4, module.hidden_size, requires_grad=True)

    out = module(hidden, **kwargs)
    assert out.shape == (1, 1, 4, module.hidden_size)
    out.sum().backward()
    assert module.q_proj.weight.grad is not None
    assert module.out_proj.weight.grad is not None


def test_counts_to_cu_seqlens():
    counts = torch.tensor([5, 0, 3, 4], dtype=torch.int64)
    cu = counts_to_cu_seqlens(counts)
    assert cu.dtype == torch.int32
    assert cu.tolist() == [0, 5, 5, 8, 12]


def test_sort_and_pack_groups_by_pixel():
    # Three pixels, observations interleaved; sorting must group each pixel's
    # source indices contiguously (order within a pixel is unconstrained).
    flat_idx = torch.tensor([2, 0, 1, 2, 0, 2], dtype=torch.int32)
    sorted_order, counts = sort_and_pack(flat_idx, total_pixels=3)
    assert counts.tolist() == [2, 1, 3]
    grouped = flat_idx[sorted_order.long()]
    assert grouped.tolist() == [0, 0, 1, 2, 2, 2]


def test_build_pixel_group_map_pairs_small_pixels():
    # Docstring example: counts [5, 0, 3, 4, 200], nonzero median 4, thresh 8 ->
    # large=[4], small=[0, 2, 3] -> programs [[4], [0, 2], [3]].
    cu = counts_to_cu_seqlens(torch.tensor([5, 0, 3, 4, 200], dtype=torch.int64))
    gm = build_pixel_group_map(cu)
    assert gm.program_ptr.tolist() == [0, 1, 3, 4]
    assert gm.program_pixels.tolist() == [4, 0, 2, 3]


def test_build_pixel_group_map_empty():
    cu = torch.zeros(6, dtype=torch.int32)
    gm = build_pixel_group_map(cu)
    assert gm.program_ptr.tolist() == [0]
    assert gm.program_pixels.numel() == 0


def test_prepare_obs_context_sorts_and_builds_group_map():
    obs = torch.tensor([1.0, 2.0, 3.0])
    float_metadata = torch.tensor([[1.0], [2.0], [3.0]])
    obs_type = torch.tensor([10, 20, 30])
    channel = torch.tensor([11, 21, 31])
    platform = torch.tensor([12, 22, 32])
    flat_idx = torch.tensor([2, 0, 2], dtype=torch.int32)

    context = prepare_obs_context(
        obs=obs,
        float_metadata=float_metadata,
        obs_type=obs_type,
        channel=channel,
        platform=platform,
        flat_idx=flat_idx,
        total_pixels=4,
    )

    assert torch.equal(context.obs, torch.tensor([2.0, 1.0, 3.0]))
    assert torch.equal(
        context.float_metadata.squeeze(-1), torch.tensor([2.0, 1.0, 3.0])
    )
    assert torch.equal(context.obs_type, torch.tensor([20, 10, 30]))
    assert torch.equal(
        context.cu_seqlens_k, torch.tensor([0, 1, 1, 3, 3], dtype=torch.int32)
    )
    assert context.max_seqlen_k == 2
    assert context.group_map is not None


def test_prepare_obs_context_empty_observations():
    context = prepare_obs_context(
        obs=torch.empty(0),
        float_metadata=torch.empty(0, 2),
        obs_type=torch.empty(0, dtype=torch.long),
        channel=torch.empty(0, dtype=torch.long),
        platform=torch.empty(0, dtype=torch.long),
        flat_idx=torch.empty(0, dtype=torch.int32),
        total_pixels=3,
    )

    assert context.obs.numel() == 0
    assert torch.equal(context.cu_seqlens_k, torch.zeros(4, dtype=torch.int32))
    assert context.max_seqlen_k == 0
    assert context.group_map is not None
    assert context.group_map.program_pixels.numel() == 0
