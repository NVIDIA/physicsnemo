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
"""
Standalone Triton-backed pixel cross-attention layer.

This module implements pixel/observation cross-attention.

1. Triton forward and backward kernels for ragged grouped-query attention.
2. ``torch.library.custom_op`` wrappers so the kernels participate in autograd and
   fake-tensor tracing for ``torch.compile``.
3. ``PixelCrossAttention``, an ``nn.Module`` that performs

       q_proj -> attention -> out_proj

   from pixel latents to the observation tokens assigned to each pixel.

The public ``pixel_attention()`` helper operates on a packed ragged layout:

* ``Q`` has shape ``[total_pixels, n_q_heads, d_head]``.
* ``tokens`` has shape ``[total_tokens, token_dim]`` and contains all
  observation tokens for all pixels concatenated together.
* ``cu_seqlens_k`` has shape ``[total_pixels + 1]`` and stores prefix sums that
  delimit which token rows belong to each pixel.
* ``W_k`` and ``W_v`` have shape ``[n_kv_heads * d_head, token_dim]``.

For a given pixel, the kernel projects only that pixel's token slice into keys
and values, applies grouped-query attention from that pixel's query heads, and
returns ``[n_q_heads, d_head]`` attention output for that pixel. The full module
just wraps that primitive with query and output projections.

Implementation notes
--------------------
* The kernel streams over the token dimension in tiles and keeps online softmax
  statistics, so it never materializes the full attention score matrix for a
  ragged pixel group.
* ``n_q_heads`` must be divisible by ``n_kv_heads``. The current Triton path
  also requires ``q_per_kv = n_q_heads / n_kv_heads >= 16`` because of the
  ``tl.dot`` tile shape used by the kernel.
* For ``n_kv_heads <= 2`` the module launches one grouped kernel per pixel. For
  larger even ``n_kv_heads`` it splits the head dimension into two-KV-head
  phases and concatenates the per-phase outputs back in the original order.
* K bias is accepted for API compatibility but dropped from the computation.
  For a fixed query, it adds the same scalar offset to every key logit, so
  softmax cancels it exactly and the analytic gradient should be zero. In
  practice, keeping it only introduced finite-precision noise, resulting in
  nonzero gradients on that bias. V bias is applied inside the Triton path.
* The Triton kernels assume packed contiguous tensors, so the Python wrappers
  materialize contiguous views before launch when needed.
* Autotuning buckets ``max_seqlen_k`` into coarse ranges so nearby ragged
  shapes can reuse the same cached launch configuration.

PyTorch integration
-------------------
``PixelCrossAttention`` keeps parameter ownership in standard ``nn.Linear``
modules. The Triton kernels consume the raw ``q_proj``/``k_proj``/``v_proj`` and
``out_proj`` tensors, while the custom-op registrations save the tensors needed
for backward so gradients flow back to the original module parameters through
ordinary PyTorch autograd.
"""

import hashlib
import math
import os

import torch
import torch.distributed as dist
import torch.nn as nn

from physicsnemo.core.version_check import OptionalImport
from physicsnemo.experimental.models.healda import triton_autotune_cache as tac

triton = OptionalImport("triton")
tl = OptionalImport("triton.language")


def _pixel_attn_kernels():
    from physicsnemo.experimental.models.healda import _pixel_attn_kernels as _k

    return _k


# Directory for the persisted Triton autotune cache. Overridable so deployments
# can redirect it to a fast/shared location; defaults under the user cache dir.
_AUTOTUNE_CACHE_DIR = os.environ.get(
    "PHYSICSNEMO_CACHE_DIR", os.path.expanduser("~/.cache/physicsnemo")
)


def _next_power_of_2(n):
    if n <= 0:
        return 1
    return 1 << (n - 1).bit_length()


_MAX_SEQLEN_K_BUCKETS = (64, 256, 1024, 4096)
_MAX_SEQLEN_K_BUCKET_OVERFLOW = 8192


def _bucket_max_seqlen_k(max_seqlen_k: int) -> int:
    # Bucket raw sequence lengths so nearby shapes share the same autotune cache.
    for upper_bound in _MAX_SEQLEN_K_BUCKETS:
        if max_seqlen_k <= upper_bound:
            return upper_bound
    return _MAX_SEQLEN_K_BUCKET_OVERFLOW


def _format_max_seqlen_k_bucket(bucket: int) -> str:
    if bucket == _MAX_SEQLEN_K_BUCKET_OVERFLOW:
        return ">4096"
    return f"<={bucket}"


_PRINTED_AUTOTUNE_CHOICES = set()
_AUTOTUNE_REPORTER = None


def set_pixel_attn_autotune_reporter(reporter):
    global _AUTOTUNE_REPORTER
    _AUTOTUNE_REPORTER = reporter


def _maybe_print_autotune_choice(
    kind,
    autotuner,
    q_per_kv,
    n_kv_heads,
    compute_dtype,
    max_seqlen_k,
    max_seqlen_k_bucket,
):
    if os.environ.get("HEALDA_PRINT_PIXEL_ATTN_AUTOTUNE") != "1":
        return
    best_config = getattr(autotuner, "best_config", None)
    if best_config is None:
        return
    key = (kind, q_per_kv, n_kv_heads, str(compute_dtype), int(max_seqlen_k_bucket))
    if key in _PRINTED_AUTOTUNE_CHOICES:
        return
    _PRINTED_AUTOTUNE_CHOICES.add(key)
    message = (
        "[pixel_attn autotune] "
        f"{kind} key=(Q_PER_KV={q_per_kv}, N_KV_HEADS={n_kv_heads}, "
        f"COMPUTE_DTYPE={compute_dtype}, "
        f"max_seqlen_k_bucket={_format_max_seqlen_k_bucket(max_seqlen_k_bucket)}) "
        f"raw_max_seqlen_k={max_seqlen_k} "
        f"best_config={best_config}"
    )
    if _AUTOTUNE_REPORTER is not None:
        _AUTOTUNE_REPORTER(message)
    else:
        print(message, flush=True)


# ─── Custom op registration ──────────────────────────────────────────
# The custom-op boundary lets PyTorch treat the Triton launch as a single op for
# autograd/fake tensor purposes while we keep the real launch logic in Python.


def _gqa_fwd_impl(
    Q,
    tokens,
    W_k,
    W_v,
    B_k,
    B_v,
    cu_seqlens_k,
    prog_ptr,
    prog_pix,
    scale,
    max_seqlen_k,
    q_per_kv,
    token_dim,
    n_kv_heads,
    use_v_bias,
    force_fp32=False,
):
    n_groups = cu_seqlens_k.shape[0] - 1
    # Empty CSR map => ungrouped: one program per pixel, kernel derives pixel =
    # program_id (GROUPED=False) and skips the per-program map loads.
    grouped = prog_pix.numel() > 0
    n_programs = (prog_ptr.shape[0] - 1) if grouped else n_groups
    n_q_heads = Q.shape[1]
    d_head = Q.shape[2]
    block_q = max(16, _next_power_of_2(q_per_kv))
    max_seqlen_k_bucket = _bucket_max_seqlen_k(int(max_seqlen_k))
    compute_dtype = tl.float32 if force_fp32 else tl.bfloat16
    # The Triton kernels below use flat pointer math for packed [group, head, d]
    # storage and do not take explicit tensor strides. Multi-phase q/head slices
    # are views with the original group stride, so materialize packed inputs here.
    Q = Q.contiguous()
    tokens = tokens.contiguous()
    W_k = W_k.contiguous()
    W_v = W_v.contiguous()
    B_k = B_k.contiguous()
    B_v = B_v.contiguous()
    Out = torch.zeros_like(Q)
    LSE = torch.empty(n_groups, n_q_heads, device=Q.device, dtype=torch.float32)
    _k = _pixel_attn_kernels()
    _k._pixel_attn_gqa_fwd[(n_programs,)](
        Q,
        tokens,
        W_k,
        W_v,
        B_k,
        B_v,
        Out,
        LSE,
        cu_seqlens_k,
        prog_ptr,
        prog_pix,
        scale,
        max_seqlen_k_bucket,
        n_groups,
        USE_V_BIAS=use_v_bias,
        Q_PER_KV=q_per_kv,
        BLOCK_Q=block_q,
        N_KV_HEADS=n_kv_heads,
        D_HEAD=d_head,
        TOKEN_DIM=token_dim,
        COMPUTE_DTYPE=compute_dtype,
        GROUPED=grouped,
    )
    _maybe_print_autotune_choice(
        "fwd",
        _k._pixel_attn_gqa_fwd,
        q_per_kv,
        n_kv_heads,
        compute_dtype,
        max_seqlen_k,
        max_seqlen_k_bucket,
    )
    return Out, LSE


def _gqa_bwd_impl(
    dOut,
    Q,
    tokens,
    W_k,
    W_v,
    B_k,
    B_v,
    Out,
    LSE,
    cu_seqlens_k,
    prog_ptr,
    prog_pix,
    scale,
    max_seqlen_k,
    q_per_kv,
    token_dim,
    n_kv_heads,
    use_v_bias,
    force_fp32=False,
):
    n_groups = cu_seqlens_k.shape[0] - 1
    grouped = prog_pix.numel() > 0
    n_programs = (prog_ptr.shape[0] - 1) if grouped else n_groups
    d_head = Q.shape[2]
    kv_dim = n_kv_heads * d_head
    block_q = max(16, _next_power_of_2(q_per_kv))
    max_seqlen_k_bucket = _bucket_max_seqlen_k(int(max_seqlen_k))
    compute_dtype = tl.float32 if force_fp32 else tl.bfloat16
    torch_compute_dtype = torch.float32 if force_fp32 else torch.bfloat16
    # Backward sees the original saved inputs from the custom op; for multi-phase
    # q/head slicing those can be non-contiguous views, which breaks the kernel's
    # flat indexing unless we repack them first.
    Q = Q.contiguous()
    tokens = tokens.contiguous()
    W_k = W_k.contiguous()
    W_v = W_v.contiguous()
    B_k = B_k.contiguous()
    B_v = B_v.contiguous()
    Out = Out.contiguous()
    LSE = LSE.contiguous()
    dOut = dOut.contiguous()
    dQ = torch.zeros_like(Q)
    d_tokens = torch.zeros_like(tokens)
    # HYBRID: kernel keeps in-kernel K/V recompute + in-kernel dtokens, but emits
    # per-token [dK | dV] rows so the weight grads are recovered with one dense
    # GEMM instead of two. Every token is written by exactly one non-empty pixel.
    dKV = torch.empty(
        tokens.shape[0], 2 * kv_dim, device=Q.device, dtype=torch_compute_dtype
    )
    _k = _pixel_attn_kernels()
    _k._pixel_attn_gqa_bwd[(n_programs,)](
        Q,
        tokens,
        W_k,
        W_v,
        B_k,
        B_v,
        Out,
        LSE,
        dOut,
        dQ,
        d_tokens,
        dKV,
        dKV,
        cu_seqlens_k,
        prog_ptr,
        prog_pix,
        scale,
        max_seqlen_k_bucket,
        n_groups,
        USE_V_BIAS=use_v_bias,
        Q_PER_KV=q_per_kv,
        BLOCK_Q=block_q,
        N_KV_HEADS=n_kv_heads,
        D_HEAD=d_head,
        TOKEN_DIM=token_dim,
        KV_DIM=kv_dim,
        COMPUTE_DTYPE=compute_dtype,
        GROUPED=grouped,
        COMBINED_DKV=True,
    )
    _maybe_print_autotune_choice(
        "bwd",
        _k._pixel_attn_gqa_bwd,
        q_per_kv,
        n_kv_heads,
        compute_dtype,
        max_seqlen_k,
        max_seqlen_k_bucket,
    )
    # Recover weight grads as one dense cuBLAS GEMM:
    # dKV rows are [dK | dV], so the result rows split back into [dW_k | dW_v].
    tokens_compute = tokens.to(torch_compute_dtype)
    dW_kv = (dKV.t() @ tokens_compute).to(torch.float32)
    dW_k = dW_kv[:kv_dim].clone()
    dW_v = dW_kv[kv_dim:].clone()
    if use_v_bias:
        # Accumulate in fp32 directly; do NOT materialize an fp32 copy of dV
        # (millions of rows) before reducing -- that HBM pass dominated the bwd.
        dB_v = dKV[:, kv_dim:].sum(dim=0, dtype=torch.float32)
    else:
        dB_v = torch.zeros(kv_dim, device=Q.device, dtype=torch.float32)
    dB_k = torch.zeros_like(B_k)
    dW_k = dW_k if W_k.dtype == dW_k.dtype else dW_k.to(W_k.dtype)
    dW_v = dW_v if W_v.dtype == dW_v.dtype else dW_v.to(W_v.dtype)
    if B_v.dtype != dB_v.dtype:
        dB_v = dB_v.to(B_v.dtype)
    return dQ, d_tokens, dW_k, dW_v, dB_k, dB_v


@torch.library.custom_op("healda::pixel_attn_fwd", mutates_args=())
def pixel_attn_fwd(
    Q: torch.Tensor,
    tokens: torch.Tensor,
    W_k: torch.Tensor,
    W_v: torch.Tensor,
    B_k: torch.Tensor,
    B_v: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    prog_ptr: torch.Tensor,
    prog_pix: torch.Tensor,
    scale: float,
    max_seqlen_k: int,
    q_per_kv: int,
    token_dim: int,
    n_kv_heads: int,
    use_v_bias: bool,
    force_fp32: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _gqa_fwd_impl(
        Q,
        tokens,
        W_k,
        W_v,
        B_k,
        B_v,
        cu_seqlens_k,
        prog_ptr,
        prog_pix,
        scale,
        max_seqlen_k,
        q_per_kv,
        token_dim,
        n_kv_heads,
        use_v_bias,
        force_fp32,
    )


@pixel_attn_fwd.register_fake
def _fake_fwd(
    Q,
    tokens,
    W_k,
    W_v,
    B_k,
    B_v,
    cu_seqlens_k,
    prog_ptr,
    prog_pix,
    scale,
    max_seqlen_k,
    q_per_kv,
    token_dim,
    n_kv_heads,
    use_v_bias,
    force_fp32,
):
    # Fake registrations mirror output metadata so torch.compile/export can trace
    # through the custom op without running the Triton kernel.
    n_groups, n_q_heads, d_head = Q.shape
    return Q.new_empty((n_groups, n_q_heads, d_head)), Q.new_empty(
        (n_groups, n_q_heads), dtype=torch.float32
    )


@torch.library.custom_op("healda::pixel_attn_bwd", mutates_args=())
def pixel_attn_bwd(
    dOut: torch.Tensor,
    Q: torch.Tensor,
    tokens: torch.Tensor,
    W_k: torch.Tensor,
    W_v: torch.Tensor,
    B_k: torch.Tensor,
    B_v: torch.Tensor,
    Out: torch.Tensor,
    LSE: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    prog_ptr: torch.Tensor,
    prog_pix: torch.Tensor,
    scale: float,
    max_seqlen_k: int,
    q_per_kv: int,
    token_dim: int,
    n_kv_heads: int,
    use_v_bias: bool,
    force_fp32: bool,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    return _gqa_bwd_impl(
        dOut,
        Q,
        tokens,
        W_k,
        W_v,
        B_k,
        B_v,
        Out,
        LSE,
        cu_seqlens_k,
        prog_ptr,
        prog_pix,
        scale,
        max_seqlen_k,
        q_per_kv,
        token_dim,
        n_kv_heads,
        use_v_bias,
        force_fp32,
    )


@pixel_attn_bwd.register_fake
def _fake_bwd(
    dOut,
    Q,
    tokens,
    W_k,
    W_v,
    B_k,
    B_v,
    Out,
    LSE,
    cu_seqlens_k,
    prog_ptr,
    prog_pix,
    scale,
    max_seqlen_k,
    q_per_kv,
    token_dim,
    n_kv_heads,
    use_v_bias,
    force_fp32,
):
    return (
        Q.new_empty(Q.shape),
        tokens.new_empty(tokens.shape),
        W_k.new_empty(W_k.shape),
        W_v.new_empty(W_v.shape),
        B_k.new_empty(B_k.shape),
        B_v.new_empty(B_v.shape),
    )


def _setup_context(ctx, inputs, output):
    (
        Q,
        tokens,
        W_k,
        W_v,
        B_k,
        B_v,
        cu_seqlens_k,
        prog_ptr,
        prog_pix,
        scale,
        max_seqlen_k,
        q_per_kv,
        token_dim,
        n_kv_heads,
        use_v_bias,
        force_fp32,
    ) = inputs
    Out, LSE = output
    # Save the packed tensors the Triton backward expects rather than rebuilding
    # projections during the autograd callback.
    ctx.save_for_backward(
        Q, tokens, W_k, W_v, B_k, B_v, Out, LSE, cu_seqlens_k, prog_ptr, prog_pix
    )
    ctx.scale = scale
    ctx.max_seqlen_k = max_seqlen_k
    ctx.q_per_kv = q_per_kv
    ctx.token_dim = token_dim
    ctx.n_kv_heads = n_kv_heads
    ctx.use_v_bias = use_v_bias
    ctx.force_fp32 = force_fp32


def _backward(ctx, grad_Out, grad_LSE):
    del grad_LSE
    (
        Q,
        tokens,
        W_k,
        W_v,
        B_k,
        B_v,
        Out,
        LSE,
        cu_seqlens_k,
        prog_ptr,
        prog_pix,
    ) = ctx.saved_tensors
    dQ, d_tokens, dW_k, dW_v, dB_k, dB_v = pixel_attn_bwd(
        grad_Out,
        Q,
        tokens,
        W_k,
        W_v,
        B_k,
        B_v,
        Out,
        LSE,
        cu_seqlens_k,
        prog_ptr,
        prog_pix,
        ctx.scale,
        ctx.max_seqlen_k,
        ctx.q_per_kv,
        ctx.token_dim,
        ctx.n_kv_heads,
        ctx.use_v_bias,
        ctx.force_fp32,
    )
    # One grad slot per fwd input: 6 real grads then None for
    # cu_seqlens_k, prog_ptr, prog_pix, scale, max_seqlen_k, q_per_kv,
    # token_dim, n_kv_heads, use_v_bias, force_fp32.
    return (
        dQ,
        d_tokens,
        dW_k,
        dW_v,
        dB_k,
        dB_v,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )


pixel_attn_fwd.register_autograd(_backward, setup_context=_setup_context)


def _pixel_attention_gqa(
    Q,
    tokens,
    W_k,
    W_v,
    B_k,
    B_v,
    cu_seqlens_k,
    prog_ptr,
    prog_pix,
    max_seqlen_k,
    n_kv_heads,
    scale,
    force_fp32=False,
):
    n_q_heads = Q.shape[1]
    q_per_kv = n_q_heads // n_kv_heads
    token_dim = tokens.shape[1]
    use_v_bias = B_v is not None
    if B_k is None:
        # The custom op has a fixed tensor schema, so use empty placeholders when
        # a bias is logically absent.
        B_k = W_k.new_empty((0,))
    if B_v is None:
        B_v = W_v.new_empty((0,))
    Q = Q.contiguous()
    tokens = tokens.contiguous()
    W_k = W_k.contiguous()
    W_v = W_v.contiguous()
    B_k = B_k.contiguous()
    B_v = B_v.contiguous()
    Out, _LSE = pixel_attn_fwd(
        Q,
        tokens,
        W_k,
        W_v,
        B_k,
        B_v,
        cu_seqlens_k,
        prog_ptr,
        prog_pix,
        scale,
        max_seqlen_k,
        q_per_kv,
        token_dim,
        n_kv_heads,
        use_v_bias,
        force_fp32,
    )
    return Out


def pixel_attention(
    Q,
    tokens,
    W_k,
    W_v,
    cu_seqlens_k,
    max_seqlen_k,
    n_kv_heads=1,
    scale=None,
    B_k=None,
    B_v=None,
    force_fp32=False,
    group_map=None,
):
    # First call: load this rank's persisted autotune configs + arm the
    # write-through (cheap, idempotent). Tuning then happens lazily on this real
    # batch and is saved for the next run.
    _ensure_autotune_cache()
    # ``group_map`` is an optional CSR map that packs multiple
    # small pixels into one kernel program (built once per batch in the dataloader,
    # carried on AttentionPacking). When absent we pass an empty map, which the
    # kernel treats as one-program-per-pixel with no CSR overhead (ungrouped path).
    if scale is None:
        scale = 1.0 / math.sqrt(Q.shape[-1])

    n_q_heads = Q.shape[1]
    if n_kv_heads < 1 or (n_kv_heads > 2 and n_kv_heads % 2 != 0):
        raise ValueError(
            f"pixel_attention requires n_kv_heads=1,2 or an even number, got {n_kv_heads}"
        )
    if n_q_heads % n_kv_heads != 0:
        raise ValueError(
            f"n_q_heads={n_q_heads} must be divisible by n_kv_heads={n_kv_heads}"
        )
    kv_dim = n_kv_heads * Q.shape[-1]
    token_dim = tokens.shape[1]
    if W_k.shape != (kv_dim, token_dim) or W_v.shape != (kv_dim, token_dim):
        raise ValueError(
            f"Expected W_k/W_v shape {(kv_dim, token_dim)}, "
            f"got W_k={tuple(W_k.shape)}, W_v={tuple(W_v.shape)}"
        )
    if B_v is not None and B_v.shape != (kv_dim,):
        raise ValueError(f"Expected B_v shape {(kv_dim,)}, got B_v={tuple(B_v.shape)}")
    # K bias only adds a per-query constant shift to the logits, which softmax
    # cancels exactly. Dropping it avoids carrying a mathematically redundant
    # term that can still pick up small finite-precision gradient noise.
    # Only V bias is material to the kernel path.
    B_k = None

    if group_map is None:
        prog_ptr = torch.empty(0, dtype=torch.int32, device=cu_seqlens_k.device)
        prog_pix = torch.empty(0, dtype=torch.int32, device=cu_seqlens_k.device)
    else:
        prog_ptr = group_map.program_ptr
        prog_pix = group_map.program_pixels

    if n_kv_heads <= 2:
        return _pixel_attention_gqa(
            Q,
            tokens,
            W_k,
            W_v,
            B_k,
            B_v,
            cu_seqlens_k,
            prog_ptr,
            prog_pix,
            max_seqlen_k,
            n_kv_heads,
            scale,
            force_fp32=force_fp32,
        )

    # For larger grouped-query layouts, run the same kernel in two-KV-head
    # phases and concatenate the head blocks back in the original order.
    n_phases = n_kv_heads // 2
    q_per_phase = n_q_heads // n_phases
    d_head = Q.shape[-1]
    kv_rows_per_phase = 2 * d_head
    outs = []
    for p in range(n_phases):
        q_slice = Q[:, p * q_per_phase : (p + 1) * q_per_phase]
        wk_slice = W_k[p * kv_rows_per_phase : (p + 1) * kv_rows_per_phase]
        wv_slice = W_v[p * kv_rows_per_phase : (p + 1) * kv_rows_per_phase]
        bv_slice = (
            None
            if B_v is None
            else B_v[p * kv_rows_per_phase : (p + 1) * kv_rows_per_phase]
        )
        outs.append(
            _pixel_attention_gqa(
                q_slice,
                tokens,
                wk_slice,
                wv_slice,
                None,
                bv_slice,
                cu_seqlens_k,
                prog_ptr,
                prog_pix,
                max_seqlen_k,
                2,
                scale,
                force_fp32=force_fp32,
            )
        )
    return torch.cat(outs, dim=1)


class PixelCrossAttention(nn.Module):
    """Cross-attention from pixel latents to packed per-pixel observation tokens.

    Forward expects:

    * ``hidden_states`` with shape ``[..., input_dim]`` containing one latent
      vector per pixel.
    * ``tokens`` with shape ``[total_tokens, token_dim]`` containing all
      observation tokens concatenated across pixels.
    * ``total_pixels`` equal to the flattened pixel count in ``hidden_states``.
    * ``cu_seqlens_k`` with shape ``[total_pixels + 1]`` storing prefix sums into
      ``tokens`` so pixel ``i`` attends to
      ``tokens[cu_seqlens_k[i]:cu_seqlens_k[i + 1]]``.
    * ``max_seqlen_k`` equal to the maximum per-pixel token count in that packed
      layout.

    The module reshapes ``hidden_states`` to ``[total_pixels, input_dim]``,
    applies ``q_proj``, runs ragged grouped-query attention over each pixel's
    token slice, applies ``out_proj``, and returns
    ``[total_pixels, output_dim]``.
    """

    def __init__(
        self,
        token_dim,
        n_q_heads,
        n_kv_heads,
        d_head,
        input_dim=None,
        output_dim=None,
        use_proj_bias=False,
    ):
        super().__init__()

        if n_kv_heads < 1 or (n_kv_heads > 2 and n_kv_heads % 2 != 0):
            raise ValueError(
                f"PixelCrossAttention requires n_kv_heads=1,2 or an even number, got {n_kv_heads}"
            )
        if n_q_heads % n_kv_heads != 0:
            raise ValueError(
                f"n_q_heads={n_q_heads} must be divisible by n_kv_heads={n_kv_heads}"
            )
        q_per_kv = n_q_heads // n_kv_heads
        if q_per_kv < 16:
            raise ValueError(
                f"n_q_heads/n_kv_heads={q_per_kv} < 16, below Triton tl.dot minimum. "
                f"For n_kv_heads={n_kv_heads}, need n_q_heads >= {n_kv_heads * 16}"
            )
        self.attn_dim = n_q_heads * d_head
        self.input_dim = self.attn_dim if input_dim is None else input_dim
        self.output_dim = self.attn_dim if output_dim is None else output_dim
        self.token_dim = token_dim
        self.n_q_heads = n_q_heads
        self.n_kv_heads = n_kv_heads
        self.d_head = d_head
        self.scale = 1.0 / math.sqrt(d_head)
        kv_dim = n_kv_heads * d_head
        self.q_proj = nn.Linear(self.input_dim, self.attn_dim, bias=use_proj_bias)
        self.k_proj = nn.Linear(token_dim, kv_dim, bias=False)
        self.v_proj = nn.Linear(token_dim, kv_dim, bias=use_proj_bias)
        self.out_proj = nn.Linear(self.attn_dim, self.output_dim, bias=use_proj_bias)

    def _forward_impl(
        self,
        hidden_states,
        tokens,
        total_pixels,
        cu_seqlens_k,
        max_seqlen_k,
        group_map=None,
    ):
        hidden_flat = hidden_states.reshape(total_pixels, self.input_dim)

        if tokens.shape[0] == 0:
            # Keep every projection parameter in the graph even when a batch has
            # no observations, so empty groups still produce gradients (prevents issues with DDP).
            token_dummy = tokens.sum() * 0
            q_dummy = self.q_proj.weight.sum() * 0
            if self.q_proj.bias is not None:
                q_dummy = q_dummy + self.q_proj.bias.sum() * 0
            kv_dummy = self.k_proj.weight.sum() * 0 + self.v_proj.weight.sum() * 0
            if self.v_proj.bias is not None:
                kv_dummy = kv_dummy + self.v_proj.bias.sum() * 0
            out = self.out_proj(hidden_flat.new_zeros((total_pixels, self.attn_dim)))
            return out + token_dummy + q_dummy + kv_dummy

        hidden_flat = self.q_proj(hidden_flat)
        Q = hidden_flat.view(total_pixels, self.n_q_heads, self.d_head)
        Q = Q.contiguous()

        attn_out = pixel_attention(
            Q,
            tokens,
            self.k_proj.weight,
            self.v_proj.weight,
            cu_seqlens_k,
            max_seqlen_k,
            n_kv_heads=self.n_kv_heads,
            scale=self.scale,
            B_k=self.k_proj.bias,
            B_v=self.v_proj.bias,
            group_map=group_map,
        )

        return self.out_proj(attn_out.reshape(total_pixels, self.attn_dim))

    def forward(
        self,
        hidden_states,
        tokens,
        total_pixels,
        cu_seqlens_k,
        max_seqlen_k,
        group_map=None,
    ):
        return self._forward_impl(
            hidden_states,
            tokens,
            total_pixels,
            cu_seqlens_k,
            max_seqlen_k,
            group_map=group_map,
        )


# ---------------------------------------------------------------------------
# Triton autotune config cache (startup optimization; does NOT change step-time
# math). Triton @autotune benchmarks every config the first time each shape-key is
# hit. We persist the chosen configs to a per-(GPU, rank) JSON and load them at
# setup so a fresh process reuses a prior run's tuning instead of re-benchmarking.
# Tuning itself stays LAZY on the real first batch (so the config always matches
# the real workload -- no synthetic data that might mistune), and a write-through
# saves each newly tuned config immediately. Each rank owns its file: no barrier /
# rank-0 coordination, and ranks tune the same buckets in parallel anyway. The file
# name embeds the GPU model + a source/Triton-version hash, so a kernel edit or a
# Triton upgrade transparently invalidates the cache (re-tunes from scratch).
# ---------------------------------------------------------------------------
def _autotuners():
    """Return the ``{name: triton.Autotuner}`` map for the kernel backend.

    Built lazily so the module imports without triton; only called from the
    autotune-cache paths, which run after the first (triton-backed) attention.
    """
    _k = _pixel_attn_kernels()
    return {
        "pixel_attn_gqa_fwd": _k._pixel_attn_gqa_fwd,
        "pixel_attn_gqa_bwd": _k._pixel_attn_gqa_bwd,
    }


_AUTOTUNE_CACHE_READY = False


def _autotune_cache_file():
    explicit = os.environ.get("HEALDA_PIXEL_ATTN_AUTOTUNE_CACHE")
    if explicit:
        return explicit
    with open(__file__, "rb") as f:
        digest = hashlib.sha1(f.read(), usedforsecurity=False).hexdigest()[:8]
    ver = getattr(triton, "__version__", "0")
    # Key by GPU model (a cache dir reused across GPU types never serves wrong-arch
    # configs) and by rank (each rank owns its file -> no write races, no barrier).
    gpu = (
        torch.cuda.get_device_name().replace(" ", "_")
        if torch.cuda.is_available()
        else "cpu"
    )
    rank = dist.get_rank() if (dist.is_available() and dist.is_initialized()) else 0
    name = f"pixel_cross_attention-{gpu}-{digest}-triton{ver}-rank{rank}.json"
    return os.path.join(_AUTOTUNE_CACHE_DIR, "triton_autotune", name)


def load_autotune_cache(path=None):
    return tac.load_caches(_autotuners(), path or _autotune_cache_file())


def save_autotune_cache(path=None):
    return tac.save_caches(_autotuners(), path or _autotune_cache_file())


def _install_writethrough(tuner, path):
    # Persist this rank's cache whenever the autotuner tunes a new key (i.e. the
    # first time a new shape/grid bucket is hit on the real workload), so the next
    # process loads it instead of re-benchmarking.
    if getattr(tuner, "_healda_writethrough", False):
        return
    tuner._healda_writethrough = True
    run = tuner.run

    def run_and_persist(*args, **kwargs):
        before = len(tuner.cache)
        out = run(*args, **kwargs)
        if len(tuner.cache) > before:
            save_autotune_cache(path)
        return out

    tuner.run = run_and_persist


def _ensure_autotune_cache():
    """Lazily wire up the per-rank autotune cache on the FIRST obs-attention call:
    load this rank's saved configs (so a fresh process reuses a prior run's tuning)
    and arm the write-through (so any config Triton tunes lazily on the real batch
    is persisted). Tuning itself stays Triton-lazy on the real workload, so the
    chosen config always matches production -- no synthetic data, no setup step.
    Idempotent and best-effort (a cache failure never blocks the kernel)."""
    global _AUTOTUNE_CACHE_READY
    if _AUTOTUNE_CACHE_READY:
        return
    _AUTOTUNE_CACHE_READY = True  # set first: best-effort, never retry per-call
    if os.environ.get("HEALDA_PIXEL_ATTN_PREWARM", "1") != "1":
        return
    path = _autotune_cache_file()
    load_autotune_cache(path)
    for tuner in _autotuners().values():
        _install_writethrough(tuner, path)
