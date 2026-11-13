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
r"""Fused attention over static ``K``-slot neighborhoods.

:class:`~physicsnemo.experimental.models.deformer.local_attention.KNNAttentionBlock`
computes, per node :math:`n` and head :math:`h`,

.. math::

    p_{j} = \operatorname{softmax}_{j}\Big(\tfrac{q_n \cdot k_{i_{nj}}}{\sqrt{d_h}}
            + b_{nj}^{h}\Big),\qquad
    o_n = \sum_j p_j\, v_{i_{nj}},\qquad
    \bar e_n = \sum_j p_j\, e_{nj},

with masked slots at :math:`-\infty`. The reference implementation gathers
keys and values into :math:`[N, K, H, d_h]` tensors (and float32 copies of
them), which dominates both time and peak memory on crash meshes. This module
implements the same map as a :class:`torch.autograd.Function` whose forward
reads ``key[index[n, k]]`` and ``value[index[n, k]]`` on the fly, keeps an
online (numerically stable) softmax in registers, and whose backward produces
the gradients for the query, key, value, bias and edge latent with a scatter-add
into the key/value gradient rows. Only the per-(node, head) log-sum-exp and the
two outputs are saved for backward.

Two backends share one gradient derivation:

* ``"triton"``: fused CUDA kernels (Triton ships with CUDA builds of PyTorch).
  Key/value gradients are accumulated with float32 atomics, so their summation
  order, and therefore their last bits, vary between runs.
* ``"torch"``: a chunked pure-PyTorch implementation of the identical algorithm
  with explicit (manual) backward formulas. It runs on any device and is the
  reference for the kernel's backward math in the test-suite.

``backend="auto"`` selects Triton on CUDA tensors when it is importable and the
Torch backend otherwise.
"""

from __future__ import annotations

from typing import Literal

import torch
from jaxtyping import Bool, Float, Int

from physicsnemo.core.version_check import check_version_spec

# Triton ships with the GPU builds of PyTorch (distribution ``triton``;
# ``pytorch-triton`` for nightly, ``pytorch-triton-rocm`` for ROCm and
# ``triton-windows`` for Windows builds); CPU builds fall back to the Torch
# backend. Availability is read from the package metadata, so an installed but
# broken Triton raises at import instead of silently disabling the kernel.
TRITON_AVAILABLE = any(
    check_version_spec(name, "3.0.0", hard_fail=False)
    for name in ("triton", "pytorch-triton", "pytorch-triton-rocm", "triton-windows")
)

if TRITON_AVAILABLE:
    import triton
    import triton.language as tl
else:  # pragma: no cover - exercised only where Triton is absent
    triton = None
    tl = None

Backend = Literal["auto", "triton", "torch"]

__all__ = [
    "TRITON_AVAILABLE",
    "Backend",
    "knn_attention",
    "resolve_backend",
    "triton_available",
]


def triton_available(device: torch.device | str | None = None) -> bool:
    """Return whether the Triton backend can run on ``device`` (CUDA only)."""
    if triton is None:
        return False
    if device is None:
        return torch.cuda.is_available()
    return torch.device(device).type == "cuda"


def resolve_backend(backend: str, device: torch.device | str) -> str:
    """Map ``"auto"`` to a concrete backend and validate explicit requests."""
    if backend == "auto":
        return "triton" if triton_available(device) else "torch"
    if backend == "triton":
        if not triton_available(device):
            raise RuntimeError(
                "fused kNN attention backend 'triton' needs a CUDA tensor "
                "and an importable `triton`; use backend='torch' or 'auto'"
            )
        return backend
    if backend == "torch":
        return backend
    raise ValueError(f"backend must be 'auto', 'triton' or 'torch'; got {backend!r}")


# --------------------------------------------------------------------------- #
# Torch backend: chunked over nodes, explicit backward.                        #
# --------------------------------------------------------------------------- #

_CHUNK_BYTES = 1 << 28  # bound the gathered [chunk, K, D] float32 tensors


def _chunk_size(num_nodes: int, slots: int, width: int) -> int:
    per_node = max(1, slots * width * 4)
    return max(1, min(num_nodes, _CHUNK_BYTES // per_node))


def _compute_dtype(dtype: torch.dtype) -> torch.dtype:
    """float32 for half-precision inputs; float64 stays float64."""
    return torch.promote_types(dtype, torch.float32)


def _torch_forward(query, key, value, bias, edge, index, mask, scale):
    num_nodes, heads, head_dim = query.shape
    slots = index.shape[1]
    edge_dim = edge.shape[-1]
    compute = _compute_dtype(query.dtype)
    out = torch.empty(
        (num_nodes, heads, head_dim), dtype=value.dtype, device=query.device
    )
    edge_out = torch.empty(
        (num_nodes, heads, edge_dim), dtype=value.dtype, device=query.device
    )
    lse = torch.empty((num_nodes, heads), dtype=compute, device=query.device)
    tiny = torch.finfo(compute).tiny
    chunk = _chunk_size(num_nodes, slots, heads * head_dim)
    with torch.autocast(device_type=query.device.type, enabled=False):
        for start in range(0, num_nodes, chunk):
            rows = slice(start, min(start + chunk, num_nodes))
            idx = index[rows].reshape(-1)
            count = index[rows].shape[0]
            gathered_key = (
                key.index_select(0, idx).view(count, slots, heads, head_dim).to(compute)
            )
            logits = (
                torch.einsum("chd,ckhd->chk", query[rows].to(compute), gathered_key)
                * scale
            )
            logits = logits + bias[rows].to(compute).permute(0, 2, 1)
            logits = logits.masked_fill(~mask[rows].unsqueeze(1), float("-inf"))
            row_max = logits.amax(dim=-1, keepdim=True)
            row_max = torch.where(
                torch.isfinite(row_max), row_max, torch.zeros_like(row_max)
            )
            prob = torch.exp(logits - row_max)  # masked slots are exactly zero
            total = prob.sum(dim=-1, keepdim=True)
            has_valid = total > 0
            prob = torch.where(
                has_valid, prob / total.clamp_min(tiny), torch.zeros_like(prob)
            )
            lse[rows] = torch.where(
                has_valid,
                row_max + torch.log(total.clamp_min(tiny)),
                torch.full_like(row_max, float("inf")),
            ).squeeze(-1)
            gathered_value = (
                value.index_select(0, idx)
                .view(count, slots, heads, head_dim)
                .to(compute)
            )
            out[rows] = torch.einsum("chk,ckhd->chd", prob, gathered_value).to(
                out.dtype
            )
            edge_out[rows] = torch.einsum(
                "chk,cke->che", prob, edge[rows].to(compute)
            ).to(edge_out.dtype)
    return out, edge_out, lse


def _torch_backward(
    query,
    key,
    value,
    bias,
    edge,
    index,
    mask,
    scale,
    lse,
    out,
    edge_out,
    d_out,
    d_edge_out,
):
    num_nodes, heads, head_dim = query.shape
    slots = index.shape[1]
    compute = _compute_dtype(query.dtype)
    d_query = torch.empty_like(query)
    d_key = torch.zeros(
        (num_nodes, heads, head_dim), dtype=compute, device=query.device
    )
    d_value = torch.zeros_like(d_key)
    d_bias = torch.empty_like(bias)
    d_edge = torch.empty_like(edge)
    chunk = _chunk_size(num_nodes, slots, heads * head_dim)
    with torch.autocast(device_type=query.device.type, enabled=False):
        for start in range(0, num_nodes, chunk):
            rows = slice(start, min(start + chunk, num_nodes))
            idx = index[rows].reshape(-1)
            count = index[rows].shape[0]
            q = query[rows].to(compute)
            gathered_key = (
                key.index_select(0, idx).view(count, slots, heads, head_dim).to(compute)
            )
            gathered_value = (
                value.index_select(0, idx)
                .view(count, slots, heads, head_dim)
                .to(compute)
            )
            logits = torch.einsum("chd,ckhd->chk", q, gathered_key) * scale
            logits = logits + bias[rows].to(compute).permute(0, 2, 1)
            logits = logits.masked_fill(~mask[rows].unsqueeze(1), float("-inf"))
            prob = torch.exp(logits - lse[rows].unsqueeze(-1))  # 0 at masked slots
            do = d_out[rows].to(compute)
            deo = d_edge_out[rows].to(compute)
            e = edge[rows].to(compute)
            d_prob = torch.einsum("chd,ckhd->chk", do, gathered_value) + torch.einsum(
                "che,cke->chk", deo, e
            )
            # sum_k p_k dp_k == do.out + deo.edge_out, so no second pass over slots.
            delta = (do * out[rows].to(compute)).sum(-1) + (
                deo * edge_out[rows].to(compute)
            ).sum(-1)
            d_logits = prob * (d_prob - delta.unsqueeze(-1))
            d_bias[rows] = d_logits.permute(0, 2, 1).to(d_bias.dtype)
            d_edge[rows] = torch.einsum("chk,che->cke", prob, deo).to(d_edge.dtype)
            d_query[rows] = (
                torch.einsum("chk,ckhd->chd", d_logits, gathered_key) * scale
            ).to(d_query.dtype)
            d_key.index_add_(
                0,
                idx,
                (torch.einsum("chk,chd->ckhd", d_logits, q) * scale).reshape(
                    -1, heads, head_dim
                ),
            )
            d_value.index_add_(
                0,
                idx,
                torch.einsum("chk,chd->ckhd", prob, do).reshape(-1, heads, head_dim),
            )
    return d_query, d_key.to(key.dtype), d_value.to(value.dtype), d_bias, d_edge


# --------------------------------------------------------------------------- #
# Triton backend.                                                              #
# --------------------------------------------------------------------------- #

if triton is not None:
    _FWD_CONFIGS = [
        triton.Config({"BLOCK_N": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_N": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_N": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_N": 128}, num_warps=4, num_stages=2),
    ]
    _BWD_CONFIGS = [
        triton.Config({"BLOCK_N": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_N": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_N": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_N": 16}, num_warps=4, num_stages=2),
    ]

    @triton.autotune(configs=_FWD_CONFIGS, key=["K", "H", "DH_PAD", "E_PAD"])
    @triton.jit
    def _knn_attention_fwd_kernel(
        Q,
        KEY,
        VAL,
        BIAS,
        EDGE,
        INDEX,
        VALID,
        OUT,
        EOUT,
        LSE,
        N,
        K,
        scale,
        stride_qn,
        stride_qh,
        stride_kn,
        stride_kh,
        stride_vn,
        stride_vh,
        stride_bn,
        stride_bk,
        stride_en,
        stride_ek,
        stride_on,
        stride_oh,
        stride_eon,
        stride_eoh,
        H: tl.constexpr,
        DH: tl.constexpr,
        DH_PAD: tl.constexpr,
        E: tl.constexpr,
        E_PAD: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid = tl.program_id(0)
        head = tl.program_id(1)
        offs_n = (pid * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
        node_ok = offs_n < N
        offs_d = tl.arange(0, DH_PAD)
        d_ok = offs_d < DH
        offs_e = tl.arange(0, E_PAD)
        e_ok = offs_e < E
        nd_ok = node_ok[:, None] & d_ok[None, :]
        ne_ok = node_ok[:, None] & e_ok[None, :]

        q = tl.load(
            Q + offs_n[:, None] * stride_qn + head * stride_qh + offs_d[None, :],
            mask=nd_ok,
            other=0.0,
        ).to(tl.float32)
        running_max = tl.full([BLOCK_N], -1.0e30, tl.float32)
        running_sum = tl.zeros([BLOCK_N], tl.float32)
        acc = tl.zeros([BLOCK_N, DH_PAD], tl.float32)
        edge_acc = tl.zeros([BLOCK_N, E_PAD], tl.float32)
        for slot in range(0, K):
            valid = tl.load(VALID + offs_n * K + slot, mask=node_ok, other=0) != 0
            # A masked slot may carry any ID: read a valid row instead.
            neighbor = tl.load(INDEX + offs_n * K + slot, mask=node_ok, other=0)
            neighbor = tl.where(valid, neighbor, 0)
            key = tl.load(
                KEY
                + neighbor[:, None] * stride_kn
                + head * stride_kh
                + offs_d[None, :],
                mask=nd_ok,
                other=0.0,
            ).to(tl.float32)
            # ``scale`` is a Python float that Inductor types as fp64 under
            # torch.compile; pin the product to fp32 so the loop state keeps
            # one type.
            logit = (tl.sum(q * key, axis=1) * scale).to(tl.float32)
            logit = logit + tl.load(
                BIAS + offs_n * stride_bn + slot * stride_bk + head,
                mask=node_ok,
                other=0.0,
            ).to(tl.float32)
            logit = tl.where(valid, logit, float("-inf"))
            new_max = tl.maximum(running_max, logit)
            rescale = tl.exp(running_max - new_max)
            prob = tl.exp(logit - new_max)
            running_sum = running_sum * rescale + prob
            value = tl.load(
                VAL
                + neighbor[:, None] * stride_vn
                + head * stride_vh
                + offs_d[None, :],
                mask=nd_ok,
                other=0.0,
            ).to(tl.float32)
            acc = acc * rescale[:, None] + prob[:, None] * value
            edge = tl.load(
                EDGE + offs_n[:, None] * stride_en + slot * stride_ek + offs_e[None, :],
                mask=ne_ok,
                other=0.0,
            ).to(tl.float32)
            edge_acc = edge_acc * rescale[:, None] + prob[:, None] * edge
            running_max = new_max
        has_valid = running_sum > 0.0
        inv = tl.where(has_valid, 1.0 / running_sum, 0.0)
        lse = tl.where(has_valid, running_max + tl.log(running_sum), float("inf"))
        tl.store(
            OUT + offs_n[:, None] * stride_on + head * stride_oh + offs_d[None, :],
            (acc * inv[:, None]).to(OUT.dtype.element_ty),
            mask=nd_ok,
        )
        tl.store(
            EOUT + offs_n[:, None] * stride_eon + head * stride_eoh + offs_e[None, :],
            (edge_acc * inv[:, None]).to(EOUT.dtype.element_ty),
            mask=ne_ok,
        )
        tl.store(LSE + offs_n * H + head, lse, mask=node_ok)

    @triton.autotune(
        configs=_BWD_CONFIGS,
        key=["K", "H", "DH_PAD", "E_PAD"],
        reset_to_zero=["DKEY", "DVAL"],
    )
    @triton.jit
    def _knn_attention_bwd_kernel(
        Q,
        KEY,
        VAL,
        BIAS,
        EDGE,
        INDEX,
        VALID,
        OUT,
        EOUT,
        LSE,
        DOUT,
        DEOUT,
        DQ,
        DKEY,
        DVAL,
        DBIAS,
        PROB,
        N,
        K,
        scale,
        stride_qn,
        stride_qh,
        stride_kn,
        stride_kh,
        stride_vn,
        stride_vh,
        stride_bn,
        stride_bk,
        stride_en,
        stride_ek,
        stride_on,
        stride_oh,
        stride_eon,
        stride_eoh,
        stride_don,
        stride_doh,
        stride_deon,
        stride_deoh,
        stride_dqn,
        stride_dqh,
        stride_dkn,
        stride_dkh,
        stride_dvn,
        stride_dvh,
        H: tl.constexpr,
        DH: tl.constexpr,
        DH_PAD: tl.constexpr,
        E: tl.constexpr,
        E_PAD: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid = tl.program_id(0)
        head = tl.program_id(1)
        offs_n = (pid * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
        node_ok = offs_n < N
        offs_d = tl.arange(0, DH_PAD)
        d_ok = offs_d < DH
        offs_e = tl.arange(0, E_PAD)
        e_ok = offs_e < E
        nd_ok = node_ok[:, None] & d_ok[None, :]
        ne_ok = node_ok[:, None] & e_ok[None, :]

        q = tl.load(
            Q + offs_n[:, None] * stride_qn + head * stride_qh + offs_d[None, :],
            mask=nd_ok,
            other=0.0,
        ).to(tl.float32)
        do = tl.load(
            DOUT + offs_n[:, None] * stride_don + head * stride_doh + offs_d[None, :],
            mask=nd_ok,
            other=0.0,
        ).to(tl.float32)
        deo = tl.load(
            DEOUT
            + offs_n[:, None] * stride_deon
            + head * stride_deoh
            + offs_e[None, :],
            mask=ne_ok,
            other=0.0,
        ).to(tl.float32)
        out = tl.load(
            OUT + offs_n[:, None] * stride_on + head * stride_oh + offs_d[None, :],
            mask=nd_ok,
            other=0.0,
        ).to(tl.float32)
        eout = tl.load(
            EOUT + offs_n[:, None] * stride_eon + head * stride_eoh + offs_e[None, :],
            mask=ne_ok,
            other=0.0,
        ).to(tl.float32)
        # sum_j p_j dp_j == do.o + deo.ebar, so the softmax Jacobian needs no
        # second pass over the slots.
        delta = tl.sum(do * out, axis=1) + tl.sum(deo * eout, axis=1)
        lse = tl.load(LSE + offs_n * H + head, mask=node_ok, other=float("inf"))
        dq = tl.zeros([BLOCK_N, DH_PAD], tl.float32)
        for slot in range(0, K):
            valid = tl.load(VALID + offs_n * K + slot, mask=node_ok, other=0) != 0
            # A masked slot may carry any ID: read a valid row instead.
            neighbor = tl.load(INDEX + offs_n * K + slot, mask=node_ok, other=0)
            neighbor = tl.where(valid, neighbor, 0)
            key = tl.load(
                KEY
                + neighbor[:, None] * stride_kn
                + head * stride_kh
                + offs_d[None, :],
                mask=nd_ok,
                other=0.0,
            ).to(tl.float32)
            value = tl.load(
                VAL
                + neighbor[:, None] * stride_vn
                + head * stride_vh
                + offs_d[None, :],
                mask=nd_ok,
                other=0.0,
            ).to(tl.float32)
            edge = tl.load(
                EDGE + offs_n[:, None] * stride_en + slot * stride_ek + offs_e[None, :],
                mask=ne_ok,
                other=0.0,
            ).to(tl.float32)
            # ``scale`` is a Python float that Inductor types as fp64 under
            # torch.compile; pin the product to fp32 so the loop state keeps
            # one type.
            logit = (tl.sum(q * key, axis=1) * scale).to(tl.float32)
            logit = logit + tl.load(
                BIAS + offs_n * stride_bn + slot * stride_bk + head,
                mask=node_ok,
                other=0.0,
            ).to(tl.float32)
            logit = tl.where(valid, logit, float("-inf"))
            prob = tl.exp(logit - lse)
            dprob = tl.sum(do * value, axis=1) + tl.sum(deo * edge, axis=1)
            dlogit = prob * (dprob - delta)
            tl.store(DBIAS + (offs_n * K + slot) * H + head, dlogit, mask=node_ok)
            tl.store(PROB + (offs_n * K + slot) * H + head, prob, mask=node_ok)
            dq += dlogit[:, None] * key
            contrib_ok = (valid & node_ok)[:, None] & d_ok[None, :]
            tl.atomic_add(
                DKEY
                + neighbor[:, None] * stride_dkn
                + head * stride_dkh
                + offs_d[None, :],
                (dlogit * scale).to(tl.float32)[:, None] * q,
                mask=contrib_ok,
                sem="relaxed",
            )
            tl.atomic_add(
                DVAL
                + neighbor[:, None] * stride_dvn
                + head * stride_dvh
                + offs_d[None, :],
                prob[:, None] * do,
                mask=contrib_ok,
                sem="relaxed",
            )
        tl.store(
            DQ + offs_n[:, None] * stride_dqn + head * stride_dqh + offs_d[None, :],
            (dq * scale).to(tl.float32).to(DQ.dtype.element_ty),
            mask=nd_ok,
        )


def _next_pow2(value: int) -> int:
    return 1 << max(0, (value - 1).bit_length())


def _row_major(tensor: torch.Tensor) -> torch.Tensor:
    """Return ``tensor`` with a unit stride on its last dimension."""
    return tensor if tensor.stride(-1) == 1 else tensor.contiguous()


def _triton_forward(query, key, value, bias, edge, index, valid, scale):
    num_nodes, heads, head_dim = query.shape
    slots = index.shape[1]
    edge_dim = edge.shape[-1]
    query, key, value = _row_major(query), _row_major(key), _row_major(value)
    bias, edge = _row_major(bias), _row_major(edge)
    out = torch.empty(
        (num_nodes, heads, head_dim), dtype=value.dtype, device=query.device
    )
    edge_out = torch.empty(
        (num_nodes, heads, edge_dim), dtype=value.dtype, device=query.device
    )
    lse = torch.empty((num_nodes, heads), dtype=torch.float32, device=query.device)
    if num_nodes == 0:
        return out, edge_out, lse

    def grid(meta):
        return (triton.cdiv(num_nodes, meta["BLOCK_N"]), heads)

    # Triton launches on the current device; the model may live elsewhere.
    with torch.cuda.device(query.device):
        _knn_attention_fwd_kernel[grid](
            query,
            key,
            value,
            bias,
            edge,
            index,
            valid,
            out,
            edge_out,
            lse,
            num_nodes,
            slots,
            scale,
            query.stride(0),
            query.stride(1),
            key.stride(0),
            key.stride(1),
            value.stride(0),
            value.stride(1),
            bias.stride(0),
            bias.stride(1),
            edge.stride(0),
            edge.stride(1),
            out.stride(0),
            out.stride(1),
            edge_out.stride(0),
            edge_out.stride(1),
            H=heads,
            DH=head_dim,
            DH_PAD=_next_pow2(head_dim),
            E=edge_dim,
            E_PAD=_next_pow2(edge_dim),
        )
    return out, edge_out, lse


def _triton_backward(
    query,
    key,
    value,
    bias,
    edge,
    index,
    valid,
    scale,
    lse,
    out,
    edge_out,
    d_out,
    d_edge_out,
):
    num_nodes, heads, head_dim = query.shape
    slots = index.shape[1]
    edge_dim = edge.shape[-1]
    query, key, value = _row_major(query), _row_major(key), _row_major(value)
    bias, edge = _row_major(bias), _row_major(edge)
    out, edge_out = _row_major(out), _row_major(edge_out)
    d_out, d_edge_out = _row_major(d_out), _row_major(d_edge_out)
    d_query = torch.empty(
        (num_nodes, heads, head_dim), dtype=query.dtype, device=query.device
    )
    d_key = torch.zeros(
        (num_nodes, heads, head_dim), dtype=torch.float32, device=query.device
    )
    d_value = torch.zeros_like(d_key)
    d_logits = torch.empty(
        (num_nodes, slots, heads), dtype=torch.float32, device=query.device
    )
    prob = torch.empty_like(d_logits)
    if num_nodes == 0:
        return (
            d_query,
            d_key.to(key.dtype),
            d_value.to(value.dtype),
            d_logits.to(bias.dtype),
            torch.empty_like(edge),
        )

    def grid(meta):
        return (triton.cdiv(num_nodes, meta["BLOCK_N"]), heads)

    with torch.cuda.device(query.device):
        _knn_attention_bwd_kernel[grid](
            query,
            key,
            value,
            bias,
            edge,
            index,
            valid,
            out,
            edge_out,
            lse,
            d_out,
            d_edge_out,
            d_query,
            d_key,
            d_value,
            d_logits,
            prob,
            num_nodes,
            slots,
            scale,
            query.stride(0),
            query.stride(1),
            key.stride(0),
            key.stride(1),
            value.stride(0),
            value.stride(1),
            bias.stride(0),
            bias.stride(1),
            edge.stride(0),
            edge.stride(1),
            out.stride(0),
            out.stride(1),
            edge_out.stride(0),
            edge_out.stride(1),
            d_out.stride(0),
            d_out.stride(1),
            d_edge_out.stride(0),
            d_edge_out.stride(1),
            d_query.stride(0),
            d_query.stride(1),
            d_key.stride(0),
            d_key.stride(1),
            d_value.stride(0),
            d_value.stride(1),
            H=heads,
            DH=head_dim,
            DH_PAD=_next_pow2(head_dim),
            E=edge_dim,
            E_PAD=_next_pow2(edge_dim),
        )
    # d e_{nk} = sum_h p_{nhk} d ebar_{nh}: a batched [K, H] x [H, E] product,
    # deterministic and far cheaper than cross-head atomics into [N, K, E].
    d_edge = torch.bmm(prob, d_edge_out.float()).to(edge.dtype)
    return (
        d_query,
        d_key.to(key.dtype),
        d_value.to(value.dtype),
        d_logits.to(bias.dtype),
        d_edge,
    )


# --------------------------------------------------------------------------- #
# Autograd function and public entry point.                                    #
# --------------------------------------------------------------------------- #


class _KNNAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value, bias, edge, index, mask, scale, backend):
        if backend == "triton":
            index = index.contiguous()
            valid = mask.to(torch.int8).contiguous()
            out, edge_out, lse = _triton_forward(
                query, key, value, bias, edge, index, valid, scale
            )
        else:
            valid = mask
            # The gather reference dereferences every slot: keep masked IDs in
            # range so a sentinel there is as harmless as in the kernel.
            index = torch.where(mask, index, torch.zeros_like(index))
            out, edge_out, lse = _torch_forward(
                query, key, value, bias, edge, index, mask, scale
            )
        ctx.save_for_backward(
            query, key, value, bias, edge, index, valid, lse, out, edge_out
        )
        ctx.scale = scale
        ctx.backend = backend
        # The row log-sum-exp is exposed as a non-differentiable statistic (the
        # backward below only propagates gradients through ``out``/``edge_out``).
        ctx.mark_non_differentiable(lse)
        return out, edge_out, lse

    @staticmethod
    def backward(ctx, d_out, d_edge_out, d_lse):
        query, key, value, bias, edge, index, valid, lse, out, edge_out = (
            ctx.saved_tensors
        )
        if d_out is None:
            d_out = torch.zeros_like(out)
        if d_edge_out is None:
            d_edge_out = torch.zeros_like(edge_out)
        kernel = _triton_backward if ctx.backend == "triton" else _torch_backward
        d_query, d_key, d_value, d_bias, d_edge = kernel(
            query,
            key,
            value,
            bias,
            edge,
            index,
            valid,
            ctx.scale,
            lse,
            out,
            edge_out,
            d_out,
            d_edge_out,
        )
        return d_query, d_key, d_value, d_bias, d_edge, None, None, None, None


def knn_attention(
    query: Float[torch.Tensor, "num_nodes heads head_dim"],
    key: Float[torch.Tensor, "num_nodes heads head_dim"],
    value: Float[torch.Tensor, "num_nodes heads head_dim"],
    bias: Float[torch.Tensor, "num_nodes slots heads"],
    edge_latent: Float[torch.Tensor, "num_nodes slots edge_dim"],
    index: Int[torch.Tensor, "num_nodes slots"],
    mask: Bool[torch.Tensor, "num_nodes slots"],
    scale: float,
    backend: Backend = "auto",
    return_lse: bool = False,
) -> (
    tuple[
        Float[torch.Tensor, "num_nodes heads head_dim"],
        Float[torch.Tensor, "num_nodes heads edge_dim"],
    ]
    | tuple[
        Float[torch.Tensor, "num_nodes heads head_dim"],
        Float[torch.Tensor, "num_nodes heads edge_dim"],
        Float[torch.Tensor, "num_nodes heads"],
    ]
):
    r"""Masked softmax attention over static neighbor slots without gathered tensors.

    Parameters
    ----------
    query, key, value : torch.Tensor
        Per-node projections of shape :math:`(N, H, d_h)`; ``key``/``value`` may
        be strided views (for example halves of one ``[N, 2D]`` projection).
    bias : torch.Tensor
        Per-slot, per-head logit bias of shape :math:`(N, K, H)`.
    edge_latent : torch.Tensor
        Encoded edge latent of shape :math:`(N, K, D_e)`; aggregated with the
        attention weights.
    index : torch.Tensor
        Neighbor IDs of shape :math:`(N, K)` (``int64``).
    mask : torch.Tensor
        Valid-slot mask of shape :math:`(N, K)` (``bool``). Masked slots get zero
        weight; a node with no valid slot produces zeros (matching the reference's
        ``nan_to_num``).
    scale : float
        Logit scale, normally :math:`1/\sqrt{d_h}`.
    backend : {"auto", "triton", "torch"}, optional
        Kernel backend; ``"auto"`` uses Triton on CUDA when available.
    return_lse : bool, optional
        Also return the per-node, per-head log-sum-exp of the masked logits,
        shape :math:`(N, H)`, float32, detached (no gradient flows through it).
        Default is ``False``.

    Returns
    -------
    tuple[torch.Tensor, ...]
        ``(sum_j p_j value[index[n, j]], sum_j p_j edge_latent[n, j])`` with shapes
        :math:`(N, H, d_h)` and :math:`(N, H, D_e)` in ``value``'s dtype, plus the
        log-sum-exp when ``return_lse`` is set. Logits, softmax and accumulation
        are float32 internally.
    """
    if query.ndim != 3 or key.shape != query.shape or value.shape != query.shape:
        raise ValueError("query, key and value must share the shape [N, H, head_dim]")
    num_nodes, heads, _ = query.shape
    if index.ndim != 2 or index.shape[0] != num_nodes or index.dtype != torch.long:
        raise ValueError(
            f"index must be int64 [{num_nodes}, K]; got {tuple(index.shape)} {index.dtype}"
        )
    if mask.shape != index.shape or mask.dtype != torch.bool:
        raise ValueError("mask must be bool with the index shape")
    slots = index.shape[1]
    if bias.shape != (num_nodes, slots, heads):
        raise ValueError(
            f"bias must have shape [{num_nodes}, {slots}, {heads}]; got {tuple(bias.shape)}"
        )
    if edge_latent.ndim != 3 or edge_latent.shape[:2] != (num_nodes, slots):
        raise ValueError(
            f"edge_latent must have shape [{num_nodes}, {slots}, D_e]; got {tuple(edge_latent.shape)}"
        )
    for name, tensor in (
        ("key", key),
        ("value", value),
        ("bias", bias),
        ("edge_latent", edge_latent),
        ("index", index),
        ("mask", mask),
    ):
        if tensor.device != query.device:
            raise ValueError(f"{name} must be on the query's device")
    if not torch.compiler.is_compiling() and index.numel():
        # The kernels do pointer arithmetic on the IDs of valid slots: a stale
        # ID would read garbage and the backward atomics would write past the
        # end of d_key/d_value. Masked slots are never dereferenced.
        lowest, highest = torch.aminmax(index.masked_fill(~mask, 0))
        if lowest < 0 or highest >= num_nodes:
            raise ValueError(
                f"index holds neighbor IDs outside [0, {num_nodes}) on valid slots"
            )
    resolved = resolve_backend(backend, query.device)
    out, edge_out, lse = _KNNAttention.apply(
        query, key, value, bias, edge_latent, index, mask, float(scale), resolved
    )
    if return_lse:
        return out, edge_out, lse
    return out, edge_out
