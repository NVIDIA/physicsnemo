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

import io

import pytest
import torch
import torch.nn as nn

from physicsnemo.nn import (
    LocalPointTransformerBlock,
    LocalTokenCrossAttentionBlock,
    ResidualMLP,
)
from physicsnemo.nn.module.point_transformer_attention import _dilated_knn


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _self_block(dim=32, num_heads=4, neighbor_k=6, dilation=1, conditioning_dim=None):
    return LocalPointTransformerBlock(
        dim=dim,
        num_heads=num_heads,
        neighbor_k=neighbor_k,
        dilation=dilation,
        mlp_ratio=2,
        dropout=0.0,
        knn_chunk_size=128,
        conditioning_dim=conditioning_dim,
    )


def _cross_block(dim=32, num_heads=4, neighbor_k=6, conditioning_dim=None):
    return LocalTokenCrossAttentionBlock(
        dim=dim,
        num_heads=num_heads,
        neighbor_k=neighbor_k,
        mlp_ratio=2,
        dropout=0.0,
        knn_chunk_size=128,
        conditioning_dim=conditioning_dim,
    )


def _ref_dilated_knn(query, key, k, dilation):
    """Independent brute-force reference for the dilated k-NN."""
    d = torch.cdist(query.float(), key.float())
    order = torch.argsort(d, dim=1, stable=True)
    k_wide = min(k * dilation, key.shape[0])
    wide = order[:, :k_wide]
    if dilation > 1:
        wide = wide[:, ::dilation]
    out_k = max(1, k_wide // dilation)
    return wide[:, :out_k]


# --------------------------------------------------------------------------- #
# forward shape / basic behaviour
# --------------------------------------------------------------------------- #
def test_self_block_forward_shape(device):
    block = _self_block().to(device).eval()
    feats = torch.randn(40, 32, device=device)
    coords = torch.randn(40, 3, device=device)
    out = block(feats, coords)
    assert out.shape == (40, 32)
    assert torch.isfinite(out).all()


def test_cross_block_forward_shape(device):
    block = _cross_block().to(device).eval()
    qf = torch.randn(25, 32, device=device)
    qc = torch.randn(25, 3, device=device)
    cf = torch.randn(18, 32, device=device)
    cc = torch.randn(18, 3, device=device)
    out = block(qf, qc, cf, cc)
    assert out.shape == (25, 32)
    assert torch.isfinite(out).all()


def test_residual_mlp_forward_shape(device):
    mlp = ResidualMLP(dim=16, mlp_ratio=4, dropout=0.0).to(device).eval()
    x = torch.randn(7, 16, device=device)
    assert mlp(x).shape == (7, 16)


@pytest.mark.parametrize("dilation", [1, 2, 3])
def test_self_block_dilation_runs(device, dilation):
    block = _self_block(dilation=dilation).to(device).eval()
    feats = torch.randn(50, 32, device=device)
    coords = torch.randn(50, 3, device=device)
    assert block(feats, coords).shape == (50, 32)


# --------------------------------------------------------------------------- #
# conditioning (AdaLN)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("block_kind", ["self", "cross"])
def test_conditioning_path(device, block_kind):
    cond_dim = 5
    if block_kind == "self":
        block = _self_block(conditioning_dim=cond_dim).to(device).eval()
        feats = torch.randn(30, 32, device=device)
        coords = torch.randn(30, 3, device=device)
        cond = torch.randn(cond_dim, device=device)
        out = block(feats, coords, cond=cond)
        assert out.shape == (30, 32)
        with pytest.raises(ValueError):
            block(feats, coords, cond=None)
    else:
        block = _cross_block(conditioning_dim=cond_dim).to(device).eval()
        qf = torch.randn(20, 32, device=device)
        qc = torch.randn(20, 3, device=device)
        cf = torch.randn(15, 32, device=device)
        cc = torch.randn(15, 3, device=device)
        cond = torch.randn(cond_dim, device=device)
        out = block(qf, qc, cf, cc, cond=cond)
        assert out.shape == (20, 32)
        with pytest.raises(ValueError):
            block(qf, qc, cf, cc, cond=None)


def test_adaln_zero_init_is_identity_on_attention(device):
    # With AdaLN the conditioning MLP is zero-initialized at its last layer, so
    # at init shift=scale=gate=0 -> the conditioned forward equals the
    # unconditioned forward of the same weights.
    torch.manual_seed(0)
    cond_dim = 4
    block = _self_block(conditioning_dim=cond_dim).to(device).eval()
    feats = torch.randn(20, 32, device=device)
    coords = torch.randn(20, 3, device=device)
    cond = torch.randn(cond_dim, device=device)
    # zero-init means the cond MLP outputs zeros regardless of cond at init
    out_a = block(feats, coords, cond=cond)
    out_b = block(feats, coords, cond=torch.zeros(cond_dim, device=device))
    torch.testing.assert_close(out_a, out_b)


# --------------------------------------------------------------------------- #
# edge cases
# --------------------------------------------------------------------------- #
def test_self_block_single_point_skips_attention(device):
    block = _self_block().to(device).eval()
    feats = torch.randn(1, 32, device=device)
    coords = torch.randn(1, 3, device=device)
    # falls back to FFN-only; must not invoke the k-NN / attention path
    assert block(feats, coords).shape == (1, 32)


def test_cross_block_zero_tokens_is_noop(device):
    block = _cross_block().to(device).eval()
    qf = torch.randn(10, 32, device=device)
    qc = torch.randn(10, 3, device=device)
    empty_f = torch.randn(0, 32, device=device)
    empty_c = torch.randn(0, 3, device=device)
    out = block(qf, qc, empty_f, empty_c)
    torch.testing.assert_close(out, qf)


def test_dim_not_divisible_by_heads_raises():
    with pytest.raises(ValueError, match="divisible"):
        _self_block(dim=30, num_heads=4)
    with pytest.raises(ValueError, match="divisible"):
        _cross_block(dim=30, num_heads=4)


def test_batch_ids_isolate_neighbors(device):
    # Two clouds packed into one tensor and *spatially overlapping*, so a
    # query in cloud A would naturally pick cloud-B neighbors if unmasked.
    # With batch_ids the result for cloud A must not change when cloud B's
    # features are perturbed -- the neighbor mask forbids cross-cloud
    # attention. (Overlap is what makes this exercise the mask: with disjoint
    # clouds A's neighbors would already all be in A.)
    torch.manual_seed(0)
    block = _self_block(neighbor_k=4).to(device).eval()
    na, nb = 12, 12
    coords_a = torch.randn(na, 3, device=device)
    coords_b = torch.randn(nb, 3, device=device)  # same region -> overlapping
    coords = torch.cat([coords_a, coords_b], 0)
    feats = torch.randn(na + nb, 32, device=device)
    batch_ids = torch.cat(
        [torch.zeros(na, dtype=torch.long), torch.ones(nb, dtype=torch.long)]
    ).to(device)

    out1 = block(feats, coords, batch_ids=batch_ids)
    feats2 = feats.clone()
    feats2[na:] += 5.0  # perturb only cloud B
    out2 = block(feats2, coords, batch_ids=batch_ids)
    # cloud A output unchanged
    torch.testing.assert_close(out1[:na], out2[:na])


# --------------------------------------------------------------------------- #
# k-NN equivalence (the controlled swap to physicsnemo.nn.functional.knn)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("dilation", [1, 2, 3])
@pytest.mark.parametrize("k", [1, 4, 8])
def test_dilated_knn_matches_reference(device, k, dilation):
    torch.manual_seed(123)
    # well-separated -> tie-free, so neighbor order is deterministic
    key = torch.rand(64, 3, device=device) * 100.0
    query = torch.rand(20, 3, device=device) * 100.0
    got = _dilated_knn(query_coords=query, key_coords=key, k=k, dilation=dilation)
    ref = _ref_dilated_knn(query, key, k, dilation)
    assert got.shape == ref.shape
    torch.testing.assert_close(got, ref.to(got.dtype))


def test_dilated_knn_self_includes_self(device):
    # When query and key are the same points, each point's nearest neighbor is
    # itself (distance 0) -> first column equals arange.
    pts = torch.rand(30, 3, device=device) * 100.0
    idx = _dilated_knn(query_coords=pts, key_coords=pts, k=5, dilation=1)
    self_idx = torch.arange(30, device=device)
    torch.testing.assert_close(idx[:, 0], self_idx)


def test_dilated_knn_k_clamped_to_keys(device):
    key = torch.rand(5, 3, device=device)
    query = torch.rand(8, 3, device=device)
    idx = _dilated_knn(query_coords=query, key_coords=key, k=10, dilation=1)
    assert idx.shape[1] == 5  # cannot exceed number of keys
    assert (idx >= 0).all() and (idx < 5).all()


# --------------------------------------------------------------------------- #
# sqrt-scaling removal: the absorption identity that makes it lossless
# --------------------------------------------------------------------------- #
def test_sqrt_scaling_absorption_identity():
    # The block drops the `/ sqrt(head_dim)` divisor that scaled-dot-product
    # attention uses, because here the score is a *learned* MLP output, not an
    # inner product. Dividing that output by a constant is exactly absorbable
    # into the final Linear's weights, so it changes no representable function.
    # This documents/guards that reasoning.
    torch.manual_seed(0)
    head_dim = 16
    s = head_dim**0.5
    lin = nn.Linear(32, 8)
    x = torch.randn(5, 32)
    scaled = lin(x) / s

    lin_absorbed = nn.Linear(32, 8)
    with torch.no_grad():
        lin_absorbed.weight.copy_(lin.weight / s)
        lin_absorbed.bias.copy_(lin.bias / s)
    torch.testing.assert_close(scaled, lin_absorbed(x))


# --------------------------------------------------------------------------- #
# gradients + state dict
# --------------------------------------------------------------------------- #
def test_self_block_backward(device):
    block = _self_block().to(device).train()
    feats = torch.randn(20, 32, device=device, requires_grad=True)
    coords = torch.randn(20, 3, device=device)
    block(feats, coords).sum().backward()
    assert feats.grad is not None
    assert torch.isfinite(feats.grad).all()


def test_state_dict_roundtrip(device):
    torch.manual_seed(0)
    block = _self_block(conditioning_dim=4).to(device).eval()
    feats = torch.randn(20, 32, device=device)
    coords = torch.randn(20, 3, device=device)
    cond = torch.randn(4, device=device)
    ref = block(feats, coords, cond=cond)

    buf = io.BytesIO()
    torch.save(block.state_dict(), buf)
    buf.seek(0)
    block2 = _self_block(conditioning_dim=4).to(device).eval()
    block2.load_state_dict(torch.load(buf, weights_only=True))
    torch.testing.assert_close(block2(feats, coords, cond=cond), ref)
