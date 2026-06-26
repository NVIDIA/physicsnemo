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

"""FiLM-conditioned observation tokenizer with an optional fused Triton backend.

This module implements the FiLM (Feature-wise Linear Modulation) observation
tokenizer:

1. ``ObsTokenizerFiLM``, an ``nn.Module`` that maps each scalar observation to
   an ``out_dim``-vector token, modulated by per-observation metadata.
2. A pure-PyTorch reference path (the readable definition of the math) that
   runs everywhere, including CPU.
3. ``torch.library.custom_op`` wrappers around the fused Triton kernels in
   :mod:`._film_kernels`, so the kernels participate in autograd and
   fake-tensor tracing for ``torch.compile``.

The tokenizer computes::

    cond  = [float_meta, obs_type_emb, channel_emb, platform_emb?]
    h     = SiLU(LayerNorm(Linear1(cond)))
    alpha, beta = split(Linear2(h))
    out   = alpha * obs + beta                # broadcast scalar obs over out_dim

Dispatch
--------
``ObsTokenizerFiLM.forward`` runs the fused Triton kernel when the inputs are on
CUDA and triton is available; otherwise it falls back to the pure-PyTorch
reference, which produces identical results. Both paths share the same learned
parameters (the ``nn.Embedding``/``nn.Linear``/``nn.LayerNorm`` modules remain
the parameter owners), so checkpoints, optimizers, and autograd behave the same
regardless of backend.

Conditioning layout
-------------------
The conditioning vector is a plain concatenation::

    [meta, obs_emb, channel_emb, platform_emb?]

Any sensor-family (conv vs satellite) routing of the metadata is performed on
the data side, so this module only sees and consumes the final metadata vector.
"""

from dataclasses import dataclass
from typing import Optional

import torch
from jaxtyping import Float, Int

from physicsnemo.core.version_check import OptionalImport

triton = OptionalImport("triton")
tl = OptionalImport("triton.language")


GLOBAL_MAX_CHANNELS = 1024
GLOBAL_MAX_PLATFORM = 1024


def _next_pow2(x: int) -> int:
    return 1 << (x - 1).bit_length()


def _default_film_hidden_dim(out_dim: int) -> int:
    return out_dim * 2 if out_dim <= 64 else out_dim


# Presets chosen from sweep on H100. Need optimal performance with dynamic input
# sizes, so hard coded presets for nobs >O(1M).
@dataclass(frozen=True)
class _KernelPreset:
    BLOCK_M: int
    num_warps: int
    num_stages: int

    def as_config_dict(self) -> dict[str, int | None]:
        return {
            "num_warps": self.num_warps,
            "num_stages": self.num_stages,
            "num_ctas": 1,
            "maxnreg": None,
            "BLOCK_M": self.BLOCK_M,
        }


_FWD_PRESET = _KernelPreset(BLOCK_M=64, num_warps=4, num_stages=2)

_BWD_PRESET_A = _KernelPreset(BLOCK_M=64, num_warps=8, num_stages=1)
_BWD_PRESET_B = _KernelPreset(BLOCK_M=128, num_warps=8, num_stages=1)
_BWD_PRESET_C = _KernelPreset(BLOCK_M=128, num_warps=8, num_stages=2)


def _cond_dim(
    *, meta_dim: int, obs_embed_dim: int, chan_embed_dim: int, platform_embed_dim: int
) -> int:
    return meta_dim + obs_embed_dim + chan_embed_dim + platform_embed_dim


def _select_bwd_preset(
    *,
    meta_dim: int,
    obs_embed_dim: int,
    chan_embed_dim: int,
    platform_embed_dim: int,
) -> _KernelPreset:
    # Backward config selection is driven mostly by the padded conditioning width
    # because the persistent kernel keeps replay state and reduction accumulators
    # live across the tile loop.
    cond_dim = _cond_dim(
        meta_dim=meta_dim,
        obs_embed_dim=obs_embed_dim,
        chan_embed_dim=chan_embed_dim,
        platform_embed_dim=platform_embed_dim,
    )
    cond_pad = _next_pow2(max(cond_dim, 16))
    if cond_pad > 64:
        return _BWD_PRESET_A
    if cond_dim <= 46:
        return _BWD_PRESET_C
    if platform_embed_dim > 0:
        return _BWD_PRESET_B
    return _BWD_PRESET_A


def get_fused_film_launch_configs(
    *,
    meta_dim: int,
    obs_embed_dim: int,
    chan_embed_dim: int,
    platform_embed_dim: int,
) -> dict[str, dict[str, int | None]]:
    """Return the sweep-selected Triton launch config for this FiLM layout."""
    bwd_preset = _select_bwd_preset(
        meta_dim=meta_dim,
        obs_embed_dim=obs_embed_dim,
        chan_embed_dim=chan_embed_dim,
        platform_embed_dim=platform_embed_dim,
    )
    return {
        "fwd": _FWD_PRESET.as_config_dict(),
        "bwd": bwd_preset.as_config_dict(),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Python wrappers (custom ops for torch.compile)
# These wrappers are split into:
#   - Python launchers that allocate outputs/gradients,
#   - public/private custom ops for ``torch.compile`` compatibility,
#   - autograd glue that saves replay inputs for backward.
# ═══════════════════════════════════════════════════════════════════════════


def _launch_fused_film_fwd(
    obs: torch.Tensor,
    float_meta: torch.Tensor,
    obs_type_id: torch.Tensor,
    channel: torch.Tensor,
    platform: torch.Tensor,
    embed_weight: torch.Tensor,
    chan_embed_weight: torch.Tensor,
    platform_embed_weight: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor,
    ln_w: torch.Tensor,
    ln_b: torch.Tensor,
    w2: torch.Tensor,
    b2: torch.Tensor,
    eps: float,
    meta_dim: int,
    obs_embed_dim: int,
    chan_embed_dim: int,
    platform_embed_dim: int,
    out_dim: int,
    force_fp32: bool,
) -> torch.Tensor:
    from . import _film_kernels as kernels

    N = obs.shape[0]
    hidden = w1.shape[1]

    out_dtype = torch.float32 if force_fp32 else torch.bfloat16
    out = torch.empty(N, out_dim, device=obs.device, dtype=out_dtype)
    if N == 0:
        return out

    # Forward handles conditioning as a sum of segment-local loads/dots, so each
    # segment gets its own masked `tl.arange` extent. Using next-pow2 widths
    # keeps those tile shapes Triton-friendly without materializing full `cond`.
    meta_pad = _next_pow2(max(meta_dim, 16))
    obs_embed_pad = _next_pow2(max(obs_embed_dim, 16))
    chan_embed_pad = _next_pow2(max(chan_embed_dim, 16))
    platform_embed_pad = _next_pow2(max(platform_embed_dim, 16))
    out_pad = _next_pow2(max(out_dim, 16))
    grid = ((N + _FWD_PRESET.BLOCK_M - 1) // _FWD_PRESET.BLOCK_M,)
    kernels._fused_film_fwd[grid](
        obs,
        float_meta,
        obs_type_id,
        channel,
        platform,
        embed_weight,
        chan_embed_weight,
        platform_embed_weight,
        w1,
        b1,
        ln_w,
        ln_b,
        w2,
        b2,
        out,
        N,
        META_DIM=meta_dim,
        OBS_EMBED_DIM=obs_embed_dim,
        CHAN_EMBED_DIM=chan_embed_dim,
        PLATFORM_EMBED_DIM=platform_embed_dim,
        META_PAD=meta_pad,
        OBS_EMBED_PAD=obs_embed_pad,
        CHAN_EMBED_PAD=chan_embed_pad,
        PLATFORM_EMBED_PAD=platform_embed_pad,
        HIDDEN=hidden,
        OUT_DIM=out_dim,
        OUT_PAD=out_pad,
        EPS=eps,
        BLOCK_M=_FWD_PRESET.BLOCK_M,
        COMPUTE_DTYPE=tl.float32 if force_fp32 else tl.bfloat16,
        num_warps=_FWD_PRESET.num_warps,
        num_stages=_FWD_PRESET.num_stages,
    )
    return out


@torch.library.custom_op("healda::fused_film_fwd", mutates_args=())
def fused_film_fwd(
    obs: torch.Tensor,
    float_meta: torch.Tensor,
    obs_type_id: torch.Tensor,
    channel: torch.Tensor,
    platform: torch.Tensor,
    embed_weight: torch.Tensor,
    chan_embed_weight: torch.Tensor,
    platform_embed_weight: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor,
    ln_w: torch.Tensor,
    ln_b: torch.Tensor,
    w2: torch.Tensor,
    b2: torch.Tensor,
    eps: float,
    meta_dim: int,
    obs_embed_dim: int,
    chan_embed_dim: int,
    platform_embed_dim: int,
    out_dim: int,
    force_fp32: bool,
) -> torch.Tensor:
    """Forward custom op that launches the fused FiLM Triton kernel."""
    return _launch_fused_film_fwd(
        obs,
        float_meta,
        obs_type_id,
        channel,
        platform,
        embed_weight,
        chan_embed_weight,
        platform_embed_weight,
        w1,
        b1,
        ln_w,
        ln_b,
        w2,
        b2,
        eps,
        meta_dim,
        obs_embed_dim,
        chan_embed_dim,
        platform_embed_dim,
        out_dim,
        force_fp32,
    )


@fused_film_fwd.register_fake
def _fake_fused_film_fwd(
    obs,
    float_meta,
    obs_type_id,
    channel,
    platform,
    embed_weight,
    chan_embed_weight,
    platform_embed_weight,
    w1,
    b1,
    ln_w,
    ln_b,
    w2,
    b2,
    eps,
    meta_dim,
    obs_embed_dim,
    chan_embed_dim,
    platform_embed_dim,
    out_dim,
    force_fp32,
):
    N = obs.shape[0]
    return obs.new_empty(
        (N, out_dim), dtype=torch.float32 if force_fp32 else torch.bfloat16
    )


def _launch_fused_film_bwd(
    grad_out: torch.Tensor,
    obs: torch.Tensor,
    float_meta: torch.Tensor,
    obs_type_id: torch.Tensor,
    channel: torch.Tensor,
    platform: torch.Tensor,
    embed_weight: torch.Tensor,
    chan_embed_weight: torch.Tensor,
    platform_embed_weight: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor,
    ln_w: torch.Tensor,
    ln_b: torch.Tensor,
    w2: torch.Tensor,
    b2: torch.Tensor,
    eps: float,
    meta_dim: int,
    obs_embed_dim: int,
    chan_embed_dim: int,
    platform_embed_dim: int,
    out_dim: int,
    force_fp32: bool,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    from . import _film_kernels as kernels

    N = obs.shape[0]
    cond_dim = meta_dim + obs_embed_dim + chan_embed_dim + platform_embed_dim
    hidden = w1.shape[1]
    mlp_out_dim = 2 * out_dim
    grad_out = grad_out.contiguous()
    grad_embed = torch.zeros_like(embed_weight)
    grad_chan_embed = torch.zeros_like(chan_embed_weight)
    grad_platform_embed = torch.zeros_like(platform_embed_weight)

    out_pad = _next_pow2(max(out_dim, 16))
    obs_embed_pad = _next_pow2(max(obs_embed_dim, 16))
    chan_embed_pad = _next_pow2(max(chan_embed_dim, 16))
    platform_embed_pad = _next_pow2(max(platform_embed_dim, 16))

    dw1 = torch.zeros(cond_dim, hidden, device=obs.device, dtype=torch.float32)
    db1 = torch.zeros(hidden, device=obs.device, dtype=torch.float32)
    dln_w = torch.zeros(hidden, device=obs.device, dtype=torch.float32)
    dln_b = torch.zeros(hidden, device=obs.device, dtype=torch.float32)
    dw2 = torch.zeros(hidden, mlp_out_dim, device=obs.device, dtype=torch.float32)
    db2 = torch.zeros(mlp_out_dim, device=obs.device, dtype=torch.float32)

    if N == 0:
        return (
            grad_embed,
            grad_chan_embed,
            grad_platform_embed,
            dw1,
            db1,
            dln_w,
            dln_b,
            dw2,
            db2,
        )

    compute_dtype = tl.float32 if force_fp32 else tl.bfloat16
    num_sms = torch.cuda.get_device_properties(obs.device).multi_processor_count

    bwd_preset = _select_bwd_preset(
        meta_dim=meta_dim,
        obs_embed_dim=obs_embed_dim,
        chan_embed_dim=chan_embed_dim,
        platform_embed_dim=platform_embed_dim,
    )
    cond_pad = _next_pow2(max(cond_dim, 16))
    # Persistent launch: cap the grid at the SM count and let each CTA stride
    # over multiple row tiles.
    bwd_grid = (min(num_sms, (N + bwd_preset.BLOCK_M - 1) // bwd_preset.BLOCK_M),)

    kernels._fused_film_bwd[bwd_grid](
        grad_out,
        obs,
        float_meta,
        obs_type_id,
        channel,
        platform,
        embed_weight,
        chan_embed_weight,
        platform_embed_weight,
        w1,
        b1,
        ln_w,
        ln_b,
        w2,
        dw1,
        db1,
        dln_w,
        dln_b,
        dw2,
        db2,
        grad_embed,
        grad_chan_embed,
        grad_platform_embed,
        N,
        META_DIM=meta_dim,
        OBS_EMBED_DIM=obs_embed_dim,
        CHAN_EMBED_DIM=chan_embed_dim,
        PLATFORM_EMBED_DIM=platform_embed_dim,
        COND_DIM=cond_dim,
        HIDDEN=hidden,
        OUT_DIM=out_dim,
        COND_PAD=cond_pad,
        OUT_PAD=out_pad,
        OBS_EMBED_PAD=obs_embed_pad,
        CHAN_EMBED_PAD=chan_embed_pad,
        PLATFORM_EMBED_PAD=platform_embed_pad,
        EPS=eps,
        BLOCK_M=bwd_preset.BLOCK_M,
        COMPUTE_DTYPE=compute_dtype,
        num_warps=bwd_preset.num_warps,
        num_stages=bwd_preset.num_stages,
    )

    return (
        grad_embed,
        grad_chan_embed,
        grad_platform_embed,
        dw1,
        db1,
        dln_w,
        dln_b,
        dw2,
        db2,
    )


@torch.library.custom_op("healda::fused_film_bwd", mutates_args=())
def fused_film_bwd(
    grad_out: torch.Tensor,
    obs: torch.Tensor,
    float_meta: torch.Tensor,
    obs_type_id: torch.Tensor,
    channel: torch.Tensor,
    platform: torch.Tensor,
    embed_weight: torch.Tensor,
    chan_embed_weight: torch.Tensor,
    platform_embed_weight: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor,
    ln_w: torch.Tensor,
    ln_b: torch.Tensor,
    w2: torch.Tensor,
    b2: torch.Tensor,
    eps: float,
    meta_dim: int,
    obs_embed_dim: int,
    chan_embed_dim: int,
    platform_embed_dim: int,
    out_dim: int,
    force_fp32: bool,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Backward pass for the fused FiLM tokenizer.

    Rebuilds a padded conditioning matrix with pointer-gather, replays the
    FiLM MLP, and accumulates parameter and embedding gradients with a
    single persistent-CTA Triton kernel.

    Returns (grad_embed, grad_chan_embed, grad_platform_embed,
             dw1, db1, dln_w, dln_b, dw2, db2).
    """
    return _launch_fused_film_bwd(
        grad_out,
        obs,
        float_meta,
        obs_type_id,
        channel,
        platform,
        embed_weight,
        chan_embed_weight,
        platform_embed_weight,
        w1,
        b1,
        ln_w,
        ln_b,
        w2,
        b2,
        eps,
        meta_dim,
        obs_embed_dim,
        chan_embed_dim,
        platform_embed_dim,
        out_dim,
        force_fp32,
    )


@fused_film_bwd.register_fake
def _fake_fused_film_bwd(
    grad_out,
    obs,
    float_meta,
    obs_type_id,
    channel,
    platform,
    embed_weight,
    chan_embed_weight,
    platform_embed_weight,
    w1,
    b1,
    ln_w,
    ln_b,
    w2,
    b2,
    eps,
    meta_dim,
    obs_embed_dim,
    chan_embed_dim,
    platform_embed_dim,
    out_dim,
    force_fp32,
):
    cond_dim = meta_dim + obs_embed_dim + chan_embed_dim + platform_embed_dim
    hidden = w1.shape[1]
    mlp_out_dim = 2 * out_dim
    return (
        embed_weight.new_empty(embed_weight.shape),
        chan_embed_weight.new_empty(chan_embed_weight.shape),
        platform_embed_weight.new_empty(platform_embed_weight.shape),
        w1.new_empty((cond_dim, hidden), dtype=torch.float32),
        w1.new_empty((hidden,), dtype=torch.float32),
        w1.new_empty((hidden,), dtype=torch.float32),
        w1.new_empty((hidden,), dtype=torch.float32),
        w1.new_empty((hidden, mlp_out_dim), dtype=torch.float32),
        w1.new_empty((mlp_out_dim,), dtype=torch.float32),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Autograd glue
# ═══════════════════════════════════════════════════════════════════════════


def _setup_context(ctx, inputs, output):
    (
        obs,
        float_meta,
        obs_type_id,
        channel,
        platform,
        embed_weight,
        chan_embed_weight,
        platform_embed_weight,
        w1,
        b1,
        ln_w,
        ln_b,
        w2,
        b2,
        eps,
        meta_dim,
        obs_embed_dim,
        chan_embed_dim,
        platform_embed_dim,
        out_dim,
        force_fp32,
    ) = inputs
    # Save exactly the tensors needed to replay the FiLM computation in backward.
    ctx.save_for_backward(
        obs,
        float_meta,
        obs_type_id,
        channel,
        platform,
        embed_weight,
        chan_embed_weight,
        platform_embed_weight,
        w1,
        b1,
        ln_w,
        ln_b,
        w2,
        b2,
    )
    ctx.eps = eps
    ctx.meta_dim = meta_dim
    ctx.obs_embed_dim = obs_embed_dim
    ctx.chan_embed_dim = chan_embed_dim
    ctx.platform_embed_dim = platform_embed_dim
    ctx.out_dim = out_dim
    ctx.force_fp32 = force_fp32


def _backward(ctx, grad_out):
    (
        obs,
        float_meta,
        obs_type_id,
        channel,
        platform,
        embed_weight,
        chan_embed_weight,
        platform_embed_weight,
        w1,
        b1,
        ln_w,
        ln_b,
        w2,
        b2,
    ) = ctx.saved_tensors

    ge, gce, gpe, dw1, db1, dln_w, dln_b, dw2, db2 = fused_film_bwd(
        grad_out,
        obs,
        float_meta,
        obs_type_id,
        channel,
        platform,
        embed_weight,
        chan_embed_weight,
        platform_embed_weight,
        w1,
        b1,
        ln_w,
        ln_b,
        w2,
        b2,
        ctx.eps,
        ctx.meta_dim,
        ctx.obs_embed_dim,
        ctx.chan_embed_dim,
        ctx.platform_embed_dim,
        ctx.out_dim,
        ctx.force_fp32,
    )
    # The inputs are metadata, indices, and scalar observations; gradients are
    # exposed for the learned tables and dense FiLM MLP parameters only.
    return (
        None,
        None,
        None,
        None,
        None,  # obs, float_meta, obs_type_id, channel, platform
        ge,
        gce,
        gpe,  # embed, chan_embed, platform_embed
        dw1.to(w1.dtype),
        db1.to(b1.dtype),
        dln_w.to(ln_w.dtype),
        dln_b.to(ln_b.dtype),
        dw2.to(w2.dtype),
        db2.to(b2.dtype),
        None,  # eps
        None,  # meta_dim
        None,  # obs_embed_dim
        None,  # chan_embed_dim
        None,  # platform_embed_dim
        None,  # out_dim
        None,  # force_fp32
    )


fused_film_fwd.register_autograd(_backward, setup_context=_setup_context)


# ═══════════════════════════════════════════════════════════════════════════
# Triton public entry point
# ═══════════════════════════════════════════════════════════════════════════


def fused_film_tokenizer_triton(
    obs: torch.Tensor,
    float_meta: torch.Tensor,
    obs_type_id: torch.Tensor,
    channel: torch.Tensor,
    platform: Optional[torch.Tensor],
    embed_table: "torch.nn.Embedding",
    channel_embedding: "torch.nn.Embedding",
    platform_embedding: Optional["torch.nn.Embedding"],
    linear1: "torch.nn.Linear",
    layer_norm: "torch.nn.LayerNorm",
    linear2: "torch.nn.Linear",
    eps: float = 1e-5,
    force_fp32: bool = False,
) -> torch.Tensor:
    """Fused 2-layer FiLM observation tokenizer (Triton backend).

    Computes::

        cond  = [float_meta, obs_emb, channel_emb, platform_emb?]
        h     = SiLU(LayerNorm(Linear1(cond)))
        alpha, beta = split(Linear2(h))
        out   = alpha * obs + beta

    Any conv/sat metadata routing is expected to be baked into ``float_meta`` on
    the data side; this kernel consumes the metadata as-is.

    All MLP weights are kept in SRAM.  No activations are saved; the
    backward recomputes the forward from the conditioning vector.

    Parameters
    ----------
    obs : (N,) scalar observation values.
    float_meta : (N, meta_dim) per-observation metadata features.
    obs_type_id, channel : (N,) int32 embedding-table indices.
    platform : (N,) int32 or None.  Required when platform_embedding
        is provided.
    embed_table, channel_embedding : nn.Embedding lookups for obs type
        and channel.
    platform_embedding : nn.Embedding or None.
    linear1, layer_norm, linear2 : the cond-MLP layers.
        ``linear1.out_features`` must be a power of 2.
        ``linear2.out_features`` must equal ``2 * out_dim``.
    eps : LayerNorm epsilon.
    force_fp32 : run the kernel in fp32 instead of bf16.
    """
    meta_dim = float_meta.shape[1]

    obs_embed_dim = embed_table.weight.shape[1]
    chan_embed_dim = channel_embedding.weight.shape[1]

    if platform_embedding is None:
        platform_embed_dim = 0
        # Dummy weight: custom_op requires a tensor arg, but the kernel
        # never dereferences it (PLATFORM_EMBED_DIM=0 guards all loads).
        platform_embed_w = torch.empty((1, 1), device=obs.device, dtype=obs.dtype)
        if platform is None:
            # Likewise, platform pointer is passed but never loaded
            # when PLATFORM_EMBED_DIM=0.
            platform = obs_type_id
    else:
        platform_embed_dim = platform_embedding.weight.shape[1]
        platform_embed_w = platform_embedding.weight
        if platform is None:
            raise ValueError("platform required when platform_embedding is provided")

    hidden_dim = linear1.out_features
    if hidden_dim <= 0 or (hidden_dim & (hidden_dim - 1)) != 0:
        raise ValueError(f"hidden_dim must be power of 2, got {hidden_dim}")

    out_dim = linear2.out_features // 2
    # Triton kernels expect column-major logical access for the dense weights.
    w1_t = linear1.weight.t().contiguous()
    w2_t = linear2.weight.t().contiguous()

    return fused_film_fwd(
        obs,
        float_meta,
        obs_type_id,
        channel,
        platform,
        embed_table.weight,
        channel_embedding.weight,
        platform_embed_w,
        w1_t,
        linear1.bias,
        layer_norm.weight,
        layer_norm.bias,
        w2_t,
        linear2.bias,
        eps,
        meta_dim,
        obs_embed_dim,
        chan_embed_dim,
        platform_embed_dim,
        out_dim,
        force_fp32,
    )


# ═══════════════════════════════════════════════════════════════════════════
# FiLM tokenizer module
# ═══════════════════════════════════════════════════════════════════════════


class ObsTokenizerFiLM(torch.nn.Module):
    r"""FiLM-style observation tokenizer: map each scalar observation to a token.

    Each observation is a single scalar measurement (a brightness temperature, a
    PCA latent of one, a wind component, ...) plus metadata describing it
    (location/time features, which instrument channel, which platform). This
    module turns that into an ``out_dim``-vector token, one per observation.

    FiLM (Feature-wise Linear Modulation) keeps the raw measurement as the signal
    and lets the metadata *modulate* it with a per-feature scale and shift::

        conditioning = cat(metadata, obs_type_emb, channel_emb[, platform_emb])
        alpha, beta  = cond_mlp(conditioning).chunk(2)   # 2 * out_dim -> two out_dim vectors
        token        = alpha * obs + beta                # broadcast scalar obs over out_dim

    Rationale: empirically the model leans heavily on the raw measurement and
    largely ignores metadata, so giving metadata a competing slot in a shared
    feature space (the older concat tokenizer) wastes capacity. FiLM preserves the
    strong raw-obs signal while still letting metadata steer it via ``alpha``/``beta``.

    Embedding tables:
      - ``embed_table``: observation *type* embedding (which kind of obs).
      - ``channel_embedding``: instrument *channel* embedding.
      - ``platform_embedding`` (optional): satellite/platform embedding.
    All are looked up per observation and concatenated into the conditioning.

    Separate conv/sat first-linear head: conventional (in-situ / GNSS) and
    satellite observations have different metadata semantics. To let the first
    ``cond_mlp`` linear specialize per family without duplicating the whole MLP, the
    metadata is pre-expanded (on the data side) into
    ``[shared, sat_private, conv_private]`` with the off-family block zeroed -- so a
    single plain first linear effectively learns separate conv/sat weights. The
    expansion is the featurization layer's job; this module just consumes the wider
    metadata vector.

    When the inputs are on CUDA and triton is available, the whole tokenizer
    (embedding gather + conditioning + 2-layer MLP + FiLM) runs in a single
    Triton kernel; otherwise the pure-PyTorch reference path below runs and
    produces identical results.

    Parameters
    ----------
    meta_dim : int
        Dimension of float metadata features.
    out_dim : int
        Output token dimension.
    n_embed : int, optional, default=1024
        Size of the observation-type embedding table.
    nchannel : int, optional, default=1024
        Number of channels. TODO(polish): trim unused settings -- the channel
        embedding table is always sized ``GLOBAL_MAX_CHANNELS``, so this is unused.
    nplatform : int, optional, default=1024
        Number of platforms. TODO(polish): trim unused settings -- the platform
        embedding table is always sized ``GLOBAL_MAX_PLATFORM``, so this is unused.
    obs_type_embed_dim : int, optional, default=4
        Dimension of observation-type embeddings.
    channel_embed_dim : int, optional
        Dimension of channel embeddings. Defaults to ``obs_type_embed_dim``.
    platform_embed_dim : int, optional
        Dimension of platform embeddings. ``0``/``None`` disables platform
        embedding.
    use_fused_mlp : bool, optional, default=True
        Prefer the fused Triton backend when CUDA + triton are available.
    use_global_channel_platform_ids : bool, optional, default=False
        TODO(polish): trim unused settings -- channel/platform id-space selection
        is now the caller's responsibility (ids are passed to ``forward``), so
        this flag is unused here.
    hidden_dim : int, optional
        Hidden dimension of the conditioning MLP. Defaults to a heuristic on
        ``out_dim``; must be a power of 2 for the fused kernel.

    Forward
    -------
    obs : torch.Tensor
        Observation values with shape :math:`(N_{obs},)`.
    float_metadata : torch.Tensor
        Float metadata with shape :math:`(N_{obs}, M_{float})`.
    obs_type : torch.Tensor
        Observation-type ids with shape :math:`(N_{obs},)`.
    channel_ids : torch.Tensor
        Channel ids with shape :math:`(N_{obs},)`.
    platform_ids : torch.Tensor, optional
        Platform ids with shape :math:`(N_{obs},)`. Required when platform
        embedding is enabled.

    Outputs
    -------
    torch.Tensor
        Tokenized observation features of shape :math:`(N_{obs}, D_{out})`.
    """

    def __init__(
        self,
        meta_dim: int,
        out_dim: int,
        n_embed: int = 1024,
        nchannel: int = 1024,
        nplatform: int = 1024,
        obs_type_embed_dim: int = 4,
        channel_embed_dim: int | None = None,
        platform_embed_dim: int | None = None,
        use_fused_mlp: bool = True,
        use_global_channel_platform_ids: bool = False,
        hidden_dim: int | None = None,
    ):
        super().__init__()
        self.out_dim = out_dim
        if channel_embed_dim is None:
            channel_embed_dim = obs_type_embed_dim
        if platform_embed_dim is None:
            platform_embed_dim = 0
        self.use_platform_embedding = platform_embed_dim > 0
        self.obs_type_embed_dim = obs_type_embed_dim
        self.channel_embed_dim = channel_embed_dim
        self.platform_embed_dim = platform_embed_dim
        self.embed_table = torch.nn.Embedding(n_embed, obs_type_embed_dim)
        self.channel_embedding = torch.nn.Embedding(
            GLOBAL_MAX_CHANNELS, channel_embed_dim
        )
        self.platform_embedding = (
            torch.nn.Embedding(GLOBAL_MAX_PLATFORM, platform_embed_dim)
            if self.use_platform_embedding
            else None
        )
        # Prefer the fused Triton path, but fall back to the pure-PyTorch
        # reference at runtime when CUDA/triton are unavailable.
        self.use_fused_mlp = use_fused_mlp
        # TODO(polish): trim unused settings -- ids are now passed directly to
        # forward(), so id-space selection is the caller's responsibility.
        self.use_global_channel_platform_ids = use_global_channel_platform_ids
        if hidden_dim is None:
            hidden_dim = _default_film_hidden_dim(out_dim)
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        self.hidden_dim = hidden_dim

        cond_dim = (
            meta_dim + obs_type_embed_dim + channel_embed_dim + platform_embed_dim
        )

        self.cond_mlp = torch.nn.Sequential(
            torch.nn.Linear(cond_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, 2 * out_dim),
        )

    # --- Pure-PyTorch reference path -----------------------------------------
    # The helper below + the else-branch of forward() spell out the tokenizer
    # math step by step. On the fused path they are bypassed:
    # fused_film_tokenizer_triton fuses the embedding gathers, the 2-layer
    # conditioning MLP, and the final alpha*obs+beta FiLM into one Triton kernel
    # (no intermediate activations materialized), producing identical results.
    # The reference path is kept for readability, CPU runs, and correctness
    # checks.

    def _build_conditioning(
        self,
        float_metadata: torch.Tensor,
        obs_type: torch.Tensor,
        channel_ids: torch.Tensor,
        platform_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        embed_vec = self.embed_table(obs_type)
        chan_emb = self.channel_embedding(channel_ids)
        conditioning_parts = [float_metadata, embed_vec, chan_emb]
        if self.use_platform_embedding:
            if platform_ids is None:
                raise ValueError("platform embedding requires platform ids")
            conditioning_parts.append(self.platform_embedding(platform_ids))
        return torch.cat(conditioning_parts, dim=-1)

    def forward(
        self,
        obs: Float[torch.Tensor, "nobs"],
        float_metadata: Float[torch.Tensor, "nobs meta_dim"],
        obs_type: Int[torch.Tensor, "nobs"],
        channel_ids: Int[torch.Tensor, "nobs"],
        platform_ids: Int[torch.Tensor, "nobs"] | None = None,
    ) -> Float[torch.Tensor, "nobs out_dim"]:
        # Dispatch to the fused Triton kernel only on CUDA when triton is
        # installed; otherwise run the pure-PyTorch reference (e.g. on CPU).
        if self.use_fused_mlp and triton.available and obs.is_cuda:
            return fused_film_tokenizer_triton(
                obs,
                float_metadata,
                obs_type,
                channel_ids,
                platform_ids if self.use_platform_embedding else None,
                self.embed_table,
                self.channel_embedding,
                self.platform_embedding,
                self.cond_mlp[0],
                self.cond_mlp[1],
                self.cond_mlp[3],
                eps=self.cond_mlp[1].eps,
            )

        # Pure-PyTorch reference path (identical results to the fused kernel).
        conditioning = self._build_conditioning(
            float_metadata, obs_type, channel_ids, platform_ids
        )
        ab = self.cond_mlp(conditioning)
        alpha, beta = ab.chunk(2, dim=-1)
        return alpha * obs.unsqueeze(-1) + beta
