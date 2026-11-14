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

"""ISLA's backbone: the slice block and the pieces it is made of.

:class:`_SliceBlock` is one pre-LN layer of measure-weighted soft-slice attention and
MLP. :class:`_RoutingLogits` produces its point-to-slice logits, and :class:`_QueryTokens`
holds the prediction-point tokens' own parameters. :func:`_relational_invariants` and
:func:`_geo_region` form the per-layer geometry region, which the fused Triton kernel of
:mod:`physicsnemo.experimental.models.isla._geo_kernel` replaces exactly. Private to
:mod:`physicsnemo.experimental.models.isla`."""

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from torch.utils.checkpoint import checkpoint

GeoKernel = Literal["eager", "fused"]
"""Geometry-region backend: the PyTorch reference region or the exact fused Triton kernel."""

#: The default bound of the cosine routing logits (ISLA's ``routing_temperature``):
#: each block's point-to-slice logit is this times the cosine between the token state
#: and a slice prototype.
ROUTING_TEMPERATURE = 10.0

#: The default expansion ratio of the per-token and per-slice MLPs (ISLA's
#: ``mlp_ratio``).
MLP_RATIO = 4


def _softmax_over_points(x: Float[torch.Tensor, "batch tokens slices"]):
    """Softmax over the point dimension (dim 1) of a (B, N, S) tensor.

    A transpose makes the softmax reduce over a contiguous last dimension, which
    PyTorch serves with its row-parallel kernel. A softmax over a *middle* dimension
    goes to the "spatial" kernel instead, which parallelizes over the B*S columns
    only and walks the N points serially, so its time grows superlinearly in the
    token count. The arithmetic is that of ``torch.softmax(x, dim=1)`` up to
    summation order. In float32, the row-parallel kernel's error against a float64
    reference is ~1e-7, the middle-dimension kernel's ~1e-4.
    """
    return torch.softmax(x.transpose(1, 2), dim=-1).transpose(1, 2)


def _relational_invariants(
    r: Float[torch.Tensor, "batch tokens dim"],
    n_hat: Float[torch.Tensor, "batch tokens dim"],
    g_hat: Float[torch.Tensor, "batch tokens vectors dim"],
    z_pos: Float[torch.Tensor, "batch slices dim"],
    m_s: Float[torch.Tensor, "batch slices dim"],
    eps: float,
    vecs: Float[torch.Tensor, "batch tokens extra dim"] | None = None,
    tens: Float[torch.Tensor, "batch tokens tensors dim dim"] | None = None,
) -> Float[torch.Tensor, "batch tokens slices geo"]:
    """The point-anchor invariants: distance and its log, the unit relative
    vector dotted with each global direction (K terms), with the point normal
    and with the anchor normal, the point normal dotted with the anchor normal,
    then the unit relative vector dotted with each of the token's extra vectors
    (its rank-1 inputs and the vectors derived from its rank-2 inputs) and the
    quadratic form of each rank-2 input in the unit relative vector. Width
    ``5 + K + P' + Q`` (:func:`_geo_width`). Every term depends only on the anchors'
    positions relative to the point, so no frame origin enters."""
    z = z_pos[:, None, :, :]
    m = m_s[:, None, :, :]
    rel = r[:, :, None, :] - z  # (B, N, S, d)
    dist = rel.norm(dim=-1, keepdim=True).clamp_min(eps)
    rel_hat = rel / dist
    n_exp = n_hat[:, :, None, :]
    feats = [
        dist,
        torch.log(dist),
        ### Broadcast multiply-and-sum, not einsum: einsum is autocast to bf16 and
        ### the geometry must stay in the input precision.
        (rel_hat[..., None, :] * g_hat[:, :, None, :, :]).sum(-1),  # (B, N, S, K)
        (rel_hat * n_exp).sum(-1, keepdim=True),
        (rel_hat * m).sum(-1, keepdim=True),
        (n_exp * m).sum(-1, keepdim=True),
    ]
    if vecs is not None and vecs.shape[2]:
        ### The token's vector inputs, contracted with the anchor direction like the
        ### global directions above.
        feats.append((rel_hat[..., None, :] * vecs[:, :, None, :, :]).sum(-1))
    if tens is not None and tens.shape[2]:
        ### rho^T T rho per rank-2 input, by einsum with autocast disabled (the geometry
        ### stays in the input precision). A broadcast product would materialize a
        ### (B, N, S, Q, d, d) intermediate.
        with torch.autocast(rel_hat.device.type, enabled=False):
            t_rho = torch.einsum("bnqij,bnsj->bnsqi", tens, rel_hat)  # (B, N, S, Q, d)
            feats.append(torch.einsum("bnsqi,bnsi->bnsq", t_rho, rel_hat))
    return torch.cat(feats, dim=-1)


def _geo_region(
    lin: nn.Module,
    logits_pre,
    r,
    n_hat,
    g_hat,
    z_pos,
    m_s,
    eps: float,
    vecs=None,
    tens=None,
):
    """One recompute region per layer: the per-slice routing bias from the
    invariants and the invariants pooled over slices by the resulting
    point->slice mix. Returns (bias (B,N,S), mix (B,N,S), pooled (B,N,geo)); the
    (B,N,S,geo) invariants and their (B,N,S,d) intermediates never leave the
    region, so under checkpointing they are rebuilt in backward, not stored."""
    geo = _relational_invariants(r, n_hat, g_hat, z_pos, m_s, eps, vecs, tens)
    bias = lin(geo).squeeze(-1)
    mix = torch.softmax(logits_pre + bias, dim=-1)  # normalized over slices
    return bias, mix, torch.einsum("bns,bnsg->bng", mix, geo)


def _fused_vectors(g_hat, vecs):
    """The fused kernel's per-point vector set: the K global directions followed by
    the token's rank-1 vectors (if any), (B, N, K + P', d)."""
    if vecs is None or vecs.shape[2] == 0:
        return g_hat
    return torch.cat([g_hat, vecs], dim=2)


def _fused_geo_region():
    """The Triton-fused dense geometry region (geo_kernel="fused"); imported lazily so
    that the module loads without triton and the eager path never touches it."""
    from physicsnemo.experimental.models.isla._geo_kernel import fused_geo_region

    return fused_geo_region


def _geo_width(n_global_vectors: int, n_vecs: int = 0, n_tens: int = 0) -> int:
    """Width of _relational_invariants: dist, log dist, K rel.g_k, rel.n, rel.m,
    n.m (5 + K), then P' rel.v and Q rho^T T rho."""
    return 5 + int(n_global_vectors) + int(n_vecs) + int(n_tens)


class _RoutingLogits(nn.Module):
    """Point-to-slice routing logits, ``(B, N, hidden) -> (B, N, S)``.

    Each logit is ``temperature`` ``T`` (default :data:`ROUTING_TEMPERATURE`) times
    the cosine between the token state and a per-slice prototype. Every logit thus
    lies in ``[-T, T]``, and no block can route every point to one slice however its
    weights grow.

    The prototypes are synthesized per sample from learned seeds (the rows of
    ``weight``) and the sample's own token states. Each seed attends over the token
    states with a measure-weighted softmax: ``log_w`` enters as a log-space bias, so
    the summary is a quadrature of the tokens and invariant to a uniform rescale of
    the weights. The attended value term is centered across the slices and bounded
    in scale before it is added to the seeds:
    ``P = seeds + Delta_c * b / (rms(Delta_c) + b)``, with ``b = g rms(seeds)``,
    ``Delta_c = Delta - mean_S(Delta)``, ``rms`` over all entries (per sample for
    ``Delta_c``) and ``g`` one learned scalar (init 0.1).

    The seed-independent component of the value term is common to every prototype.
    It carries no routing information under cosine logits and, when it dominates,
    collapses the prototypes onto one direction; centering removes it exactly. For
    ``g > 0`` the bound saturates smoothly. A small centered term (near-uniform
    attention, roundoff) passes through unchanged, and a large one is held to an rms
    of at most ``g rms(seeds)``. The sample-dependent part thus never exceeds a
    fraction ``g`` of the seeds' spread, and is never amplified to it. The gain is
    not constrained: a negative one has a pole where ``rms(Delta_c) = -g rms(seeds)``.

    Token states are invariant features, so the prototypes are invariant and the
    contracts hold. A token cloud whose attention is the same for every seed leaves
    the prototypes equal to the seeds. Each block costs ``2 hidden^2 + 1``
    parameters beyond the seeds (the key and value maps and the gain).
    """

    def __init__(self, hidden: int, n_slices: int, temperature: float = ROUTING_TEMPERATURE) -> None:
        super().__init__()
        self.temperature = float(temperature)
        self.weight = nn.Linear(hidden, n_slices, bias=False).weight
        self.k_proj = nn.Linear(hidden, hidden, bias=False)
        self.v_proj = nn.Linear(hidden, hidden, bias=False)
        self.scale = 1.0 / math.sqrt(hidden)
        self.gain = nn.Parameter(torch.tensor(0.1))

    def prototypes(self, x: torch.Tensor, log_w: torch.Tensor) -> torch.Tensor:
        """Per-sample prototypes (B, S, hidden) from the seeds and the token states."""
        seeds = self.weight.to(x.dtype)  # (S, H)
        att = torch.einsum("sh,bnh->bsn", seeds, self.k_proj(x))  # (B, S, N)
        ### Scale, measure term and softmax in at least float32: eager then rounds as
        ### the compiled path does (only the GEMM output is bf16 under autocast).
        att = att.to(torch.promote_types(att.dtype, torch.float32)) * self.scale
        att = att + log_w[..., 0][:, None, :].to(att.dtype)
        att = torch.softmax(att, dim=-1).to(x.dtype)
        delta = torch.einsum("bsn,bnh->bsh", att, self.v_proj(x))  # (B, S, H)
        delta = delta - delta.mean(dim=1, keepdim=True)
        bound = self.gain.to(x.dtype) * seeds.square().mean().sqrt()  # g rms(seeds), scalar
        ### Floor the mean square before the root. When every token state is identical
        ### (a two-cell planar probe whose only block-0 feature, n.g, is the same for
        ### both), the centered term is exactly zero and sqrt'(0) is infinite. The floor
        ### makes the gradient there zero instead of NaN, and is 1e-15 below any
        ### float32 value the term takes otherwise.
        rms_delta = delta.square().mean(dim=(1, 2), keepdim=True).clamp_min(1e-30).sqrt()  # (B, 1, 1)
        return seeds[None] + delta * (bound / (rms_delta + bound))

    def logits(self, x: torch.Tensor, protos: torch.Tensor) -> torch.Tensor:
        """Bounded cosine logits (B, N, S) of token states against given prototypes."""

        def unit(t):
            ### float16 cannot hold F.normalize's eps (1e-12 underflows to 0, so a padding
            ### token's zero state gives 0/0). Normalize in float32, as CUDA autocast does,
            ### and return float16 rows to the einsum.
            if t.dtype == torch.float16:
                return F.normalize(t.float(), dim=-1).to(t.dtype)
            return F.normalize(t, dim=-1)

        ### Upcast the cosine to at least float32 before the product with T, so the
        ### logits, their bias sum and their gradient accumulation are float32, as on
        ### the compiled path.
        cos = torch.einsum("bnh,bsh->bns", unit(x), unit(protos))
        return self.temperature * cos.to(torch.promote_types(cos.dtype, torch.float32))

    def forward(self, x: torch.Tensor, log_w: torch.Tensor) -> torch.Tensor:
        return self.logits(x, self.prototypes(x, log_w))


class _QueryTokens(nn.Module):
    """The query tokens' own parameters, in the interior task.

    Their input layer ``seed`` maps their seed to the token width: the K terms
    ``n_q.g_k``, the global scalars, and the built-in signed distance to the
    boundary surface as ``[s, asinh(s / 1e-6)]``, ``s = d / L``. As with a boundary
    condition's input layer, the model's shared GELU and linear layer follow. The
    tokens are passive (the query-independent form): they carry no routing weight,
    so they own no routing parameter.
    """

    def __init__(self, hidden: int, n_seed: int) -> None:
        super().__init__()
        self.seed = nn.Linear(n_seed, hidden)


class _SliceBlock(nn.Module):
    """One pre-LN layer of measure-weighted soft-slice attention + MLP."""

    def __init__(
        self,
        hidden: int,
        n_slices: int,
        n_global_vectors: int,
        n_block_vecs: int = 0,
        n_block_tens: int = 0,
        routing_temperature: float = ROUTING_TEMPERATURE,
        mlp_ratio: int = MLP_RATIO,
    ) -> None:
        super().__init__()
        self.n_geo = _geo_width(n_global_vectors, n_block_vecs, n_block_tens)
        self.norm_assign = nn.LayerNorm(hidden)
        self.assign = _RoutingLogits(hidden, n_slices, routing_temperature)
        ### Relational geometry: per-slice equivariant anchors and point-anchor
        ### invariants. The anchors are many local, data-adaptive reference points
        ### in place of a global frame, smooth by construction.
        self.geo_width = hidden // 2
        ### No bias: a constant added to every slice's logit cancels in both the
        ### softmax over slices and the one over points, so it would get no gradient.
        self.geo_logit = nn.Linear(self.n_geo, 1, bias=False)
        self.geo_feat = nn.Linear(self.n_geo, self.geo_width)
        self.slice_mlp = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, mlp_ratio * hidden),
            nn.GELU(),
            nn.Linear(mlp_ratio * hidden, hidden),
        )
        self.broadcast = nn.Linear(2 * hidden + self.geo_width, hidden)
        self.norm_mlp = nn.LayerNorm(hidden)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, mlp_ratio * hidden),
            nn.GELU(),
            nn.Linear(mlp_ratio * hidden, hidden),
        )

    def forward(
        self,
        h: Float[torch.Tensor, "batch tokens hidden"],
        log_w: Float[torch.Tensor, "batch tokens 1"],
        r: Float[torch.Tensor, "batch tokens dim"],
        n_hat: Float[torch.Tensor, "batch tokens dim"],
        g_hat: Float[torch.Tensor, "batch tokens vectors dim"],
        eps: float,
        vecs=None,
        tens=None,
        geo_kernel: GeoKernel = "eager",
        slices: bool = False,
    ) -> Float[torch.Tensor, "batch tokens hidden"]:
        """The block over every token; with ``slices`` also what a passive token reads
        of it, ``(prototypes, z_pos, m_s, z)``, for :meth:`read`."""
        ### Soft assignment of points to slices; measure weights enter as a
        ### log-space bias so slice states are quadrature-weighted means.
        h_n = self.norm_assign(h)
        protos = self.assign.prototypes(h_n, log_w)
        logits = self.assign.logits(h_n, protos)  # (B, N, S)
        a = _softmax_over_points(logits + log_w)  # normalized over points
        ### Equivariant anchors: each slice's weighted mean position and mean normal
        ### direction, so an anchor has an orientation.
        z_pos = torch.einsum("bns,bnc->bsc", a, r)  # (B, S, d)
        ### The mean normal of a near-uniform slice on a closed boundary nearly cancels,
        ### so its direction must not be bf16 noise: autocast off (a, n_hat at least float32).
        ### A slice that cancels to roundoff (every token alike on a closed boundary) has
        ### no direction: the floor keeps that roundoff small instead of normalizing it.
        with torch.autocast(a.device.type, enabled=False):
            m_s = torch.einsum("bns,bnc->bsc", a, n_hat)
            m_s = m_s / m_s.norm(dim=-1, keepdim=True).clamp_min(1e-3)
        ### Geometry refines the routing and the readback. Pooling the invariants over
        ### slices before the projection equals projecting then pooling (the
        ### projection is affine and point_mix sums to one over slices), and saves a
        ### (B, N, geo) activation instead of (B, N, S, hidden/2).
        bias, point_mix, pooled = self._geo(logits, r, n_hat, g_hat, z_pos, m_s, eps, vecs, tens, geo_kernel)
        logits = logits + bias
        a = _softmax_over_points(logits + log_w)
        ### The slice sublayer reads the normalized state, as a pre-LN sublayer does:
        ### the pooling and the broadcast both read norm_assign(h), so the raw h
        ### enters the block only in the residual identity and norm_mlp.
        z = torch.einsum("bns,bnh->bsh", a, h_n)  # slice states
        z = z + self.slice_mlp(z)
        h = self._update(h, h_n, point_mix, pooled, z)
        return (h, (protos, z_pos, m_s, z)) if slices else h

    def read(self, h, slices, r, n_hat, g_hat, eps: float, vecs=None, tens=None, geo_kernel: GeoKernel = "eager"):
        """The block over passive tokens (log weight -inf), given the ``slices`` the
        other tokens formed (``forward(..., slices=True)``). Passive tokens enter a
        block only through their own routing and readback, so this is the arithmetic
        :meth:`forward` gives them."""
        protos, z_pos, m_s, z = slices
        h_n = self.norm_assign(h)
        logits = self.assign.logits(h_n, protos)
        _, point_mix, pooled = self._geo(logits, r, n_hat, g_hat, z_pos, m_s, eps, vecs, tens, geo_kernel)
        return self._update(h, h_n, point_mix, pooled, z)

    def _geo(self, logits, r, n_hat, g_hat, z_pos, m_s, eps, vecs, tens, geo_kernel):
        """The geometry region, ``(bias, point_mix, pooled)``. ``geo_kernel`` names its
        implementation. ``"eager"`` checkpoints :func:`_geo_region`. ``"fused"`` does
        the same arithmetic (invariants -> routing bias -> point->slice mix -> pooled
        invariants) in one Triton kernel per direction, with the recompute built in,
        and never materializes a (B,N,S,.) tensor (``_geo_kernel.py``)."""
        if geo_kernel == "fused":
            ### The fused kernel takes the global directions and the token's rank-1
            ### vectors as one per-point vector set (at most four in all, no rank-2
            ### inputs; the model's resolution checks it).
            bias, point_mix, pooled = _fused_geo_region()(
                self.geo_logit,
                logits,
                r,
                n_hat,
                _fused_vectors(g_hat, vecs),
                z_pos,
                m_s,
                eps,
                n_global=g_hat.shape[2],
            )
        else:
            ### Activation recompute: the per-slice geometry tensors (rel (B,N,S,d),
            ### rel_hat, dist and two bf16 copies of the (B,N,S,geo) invariants)
            ### dominate the saved activations. Backward rebuilds them from (r, n_hat,
            ### g_hat, z_pos, m_s) instead of storing them; the forward is bitwise the
            ### stored-activation forward (same ops, same order).
            bias, point_mix, pooled = checkpoint(
                _geo_region,
                self.geo_logit,
                logits,
                r,
                n_hat,
                g_hat,
                z_pos,
                m_s,
                eps,
                vecs,
                tens,
                use_reentrant=False,
            )
        return bias, point_mix, pooled

    def _update(self, h, h_n, point_mix, pooled, z):
        """Each token's readback of the slice states ``z`` and the MLP sublayer."""
        back = torch.einsum("bns,bsh->bnh", point_mix, z)
        geo_pool = self.geo_feat(pooled)
        h = h + self.broadcast(torch.cat([h_n, back, geo_pool], dim=-1))
        return h + self.mlp(self.norm_mlp(h))
