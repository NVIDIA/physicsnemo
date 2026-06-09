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
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from physicsnemo.core import Module
from physicsnemo.nn import (
    AdaLNResidualMLP,
    LocalPointTransformerBlock,
    LocalTokenCrossAttentionBlock,
)
from physicsnemo.nn.module.point_transformer_attention import _dilated_knn


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _self_block(
    dim=32,
    num_heads=4,
    neighbor_k=6,
    dilation=1,
    conditioning_dim=None,
    adaln_zero=False,
):
    return LocalPointTransformerBlock(
        dim=dim,
        num_heads=num_heads,
        neighbor_k=neighbor_k,
        dilation=dilation,
        mlp_ratio=2,
        dropout=0.0,
        conditioning_dim=conditioning_dim,
        adaln_zero=adaln_zero,
    )


def _cross_block(
    dim=32, num_heads=4, neighbor_k=6, conditioning_dim=None, adaln_zero=False
):
    return LocalTokenCrossAttentionBlock(
        dim=dim,
        num_heads=num_heads,
        neighbor_k=neighbor_k,
        mlp_ratio=2,
        dropout=0.0,
        conditioning_dim=conditioning_dim,
        adaln_zero=adaln_zero,
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
@pytest.mark.parametrize("dilation", [1, 2, 3])
def test_self_block_forward(device, dilation):
    # forward smoke + finite output across dilation regimes (dilation == 1 is
    # the plain-k-NN path; > 1 exercises the wider-then-strided path)
    block = _self_block(dilation=dilation).to(device).eval()
    feats = torch.randn(40, 32, device=device)
    coords = torch.randn(40, 3, device=device)
    out = block(feats, coords)
    assert out.shape == (40, 32)
    assert torch.isfinite(out).all()


def test_cross_block_forward(device):
    block = _cross_block().to(device).eval()
    qf = torch.randn(25, 32, device=device)
    qc = torch.randn(25, 3, device=device)
    cf = torch.randn(18, 32, device=device)
    cc = torch.randn(18, 3, device=device)
    out = block(qf, qc, cf, cc)
    assert out.shape == (25, 32)
    assert torch.isfinite(out).all()


def test_residual_mlp_forward(device):
    mlp = AdaLNResidualMLP(dim=16, mlp_ratio=4, dropout=0.0).to(device).eval()
    x = torch.randn(7, 16, device=device)
    out = mlp(x)
    assert out.shape == (7, 16)
    assert torch.isfinite(out).all()


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


def _activate_conditioning(block):
    # The AdaLN conditioning MLP is zero-initialized (identity at init); perturb
    # its final layer so conditioning actually modulates the output, for tests
    # that assert the conditioning has an effect.
    with torch.no_grad():
        last = block.conditioning.layers[-1]
        last.weight.normal_(0.0, 0.1)
        last.bias.normal_(0.0, 0.1)


@pytest.mark.parametrize("block_kind", ["self", "cross"])
def test_adaln_zero_gating_runs(device, block_kind):
    # Exercise the adaln_zero=True gating branch (out * gate), distinct from the
    # default out * (1 + gate).
    cond_dim = 4
    cond = torch.randn(cond_dim, device=device)
    if block_kind == "self":
        block = (
            _self_block(conditioning_dim=cond_dim, adaln_zero=True).to(device).eval()
        )
        out = block(
            torch.randn(20, 32, device=device),
            torch.randn(20, 3, device=device),
            cond=cond,
        )
    else:
        block = (
            _cross_block(conditioning_dim=cond_dim, adaln_zero=True).to(device).eval()
        )
        out = block(
            torch.randn(18, 32, device=device),
            torch.randn(18, 3, device=device),
            torch.randn(12, 32, device=device),
            torch.randn(12, 3, device=device),
            cond=cond,
        )
    assert torch.isfinite(out).all()


def test_cross_block_context_cond_modulates(device):
    # context_cond modulates the key/value side; changing it must change output.
    cond_dim = 4
    block = _cross_block(conditioning_dim=cond_dim).to(device).eval()
    _activate_conditioning(block)
    qf = torch.randn(18, 32, device=device)
    qc = torch.randn(18, 3, device=device)
    cf = torch.randn(12, 32, device=device)
    cc = torch.randn(12, 3, device=device)
    cond = torch.randn(cond_dim, device=device)
    out_a = block(
        qf, qc, cf, cc, cond=cond, context_cond=torch.randn(cond_dim, device=device)
    )
    out_b = block(
        qf, qc, cf, cc, cond=cond, context_cond=torch.randn(cond_dim, device=device)
    )
    assert out_a.shape == (18, 32)
    assert not torch.allclose(out_a, out_b)


def test_cross_block_per_query_cond_without_context_cond(device):
    # Regression: per-query cond with context_cond=None must not crash even when
    # N_q != N_c. The KV side reduces the per-query cond to a single global
    # vector so it broadcasts against the N_c context tokens.
    cond_dim = 4
    block = _cross_block(conditioning_dim=cond_dim).to(device).eval()
    nq, nc = 18, 12  # N_q != N_c
    cond = torch.randn(nq, cond_dim, device=device)  # per-query conditioning
    out = block(
        torch.randn(nq, 32, device=device),
        torch.randn(nq, 3, device=device),
        torch.randn(nc, 32, device=device),
        torch.randn(nc, 3, device=device),
        cond=cond,
    )
    assert out.shape == (nq, 32)
    assert torch.isfinite(out).all()


def test_cross_block_batch_ids_isolate_neighbors(device):
    # Cross-block neighbor mask: batch-0 queries must be unaffected by batch-1
    # context tokens, even though coords overlap (so unmasked they'd be picked).
    torch.manual_seed(0)
    block = _cross_block(neighbor_k=4).to(device).eval()
    nq, nc = 10, 10
    qf = torch.randn(nq, 32, device=device)
    qc = torch.randn(nq, 3, device=device)
    cf = torch.randn(nc, 32, device=device)
    cc = torch.randn(nc, 3, device=device)
    query_batch_ids = torch.zeros(nq, dtype=torch.long, device=device)
    context_batch_ids = torch.cat(
        [
            torch.zeros(nc // 2, dtype=torch.long),
            torch.ones(nc - nc // 2, dtype=torch.long),
        ]
    ).to(device)
    out1 = block(
        qf,
        qc,
        cf,
        cc,
        query_batch_ids=query_batch_ids,
        context_batch_ids=context_batch_ids,
    )
    cf2 = cf.clone()
    cf2[nc // 2 :] += 5.0  # perturb only batch-1 context
    out2 = block(
        qf,
        qc,
        cf2,
        cc,
        query_batch_ids=query_batch_ids,
        context_batch_ids=context_batch_ids,
    )
    torch.testing.assert_close(out1, out2)


@pytest.mark.parametrize(
    "block_cls, kwargs, expected",
    [
        (
            LocalPointTransformerBlock,
            dict(
                dim=32,
                num_heads=4,
                neighbor_k=6,
                dilation=2,
                mlp_ratio=2,
                dropout=0.0,
                coord_dim=3,
            ),
            dict(
                dim=32, num_heads=4, head_dim=8, neighbor_k=6, dilation=2, coord_dim=3
            ),
        ),
        (
            LocalTokenCrossAttentionBlock,
            dict(
                dim=64, num_heads=8, neighbor_k=8, mlp_ratio=4, dropout=0.0, coord_dim=2
            ),
            dict(dim=64, num_heads=8, head_dim=8, neighbor_k=8, coord_dim=2),
        ),
    ],
)
def test_constructor_attributes(block_cls, kwargs, expected):
    # MOD-008a: constructor/attribute coverage across >= 2 configurations.
    block = block_cls(**kwargs)
    for name, value in expected.items():
        assert getattr(block, name) == value


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


@pytest.mark.parametrize("coord_dim", [2, 3, 6])
def test_non_3d_coords(device, coord_dim):
    # coord_dim generalizes the layer beyond 3D point clouds (e.g. 2D meshes,
    # 6-DoF poses); pos_proj adapts to the coordinate dimensionality.
    block = (
        LocalPointTransformerBlock(
            dim=32,
            num_heads=4,
            neighbor_k=6,
            dilation=1,
            mlp_ratio=2,
            dropout=0.0,
            coord_dim=coord_dim,
        )
        .to(device)
        .eval()
    )
    feats = torch.randn(20, 32, device=device)
    coords = torch.randn(20, coord_dim, device=device)
    out = block(feats, coords)
    assert out.shape == (20, 32)
    # wrong coord dim must fail fast
    with pytest.raises(ValueError, match="coords"):
        block(feats, torch.randn(20, coord_dim + 1, device=device))


def test_forward_shape_validation_raises(device):
    # MOD-005: forward validates tensor shapes at the API boundary.
    block = _self_block(dim=32).to(device).eval()
    coords = torch.randn(10, 3, device=device)
    with pytest.raises(ValueError, match="features"):
        block(torch.randn(10, 16, device=device), coords)  # wrong feature dim
    with pytest.raises(ValueError, match="coords"):
        block(torch.randn(10, 32, device=device), torch.randn(10, 2, device=device))
    with pytest.raises(ValueError, match="share"):
        block(torch.randn(10, 32, device=device), torch.randn(8, 3, device=device))


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


_DATA_DIR = Path(__file__).parent / "data"

# (checkpoint stem, expected attrs, call(model, inputs) -> output). Reference
# artifacts are produced by data/generate_point_transformer_attention_references.py
# (seeded, CPU). Re-run that script to regenerate after an intentional change.
_REFERENCE_CASES = [
    (
        "local_point_transformer_block_v1",
        {"dim": 16, "num_heads": 2},
        lambda m, d: m(d["feats"], d["coords"], cond=d["cond"]),
    ),
    (
        "local_token_cross_attention_block_v1",
        {"dim": 16, "num_heads": 2},
        lambda m, d: m(d["qf"], d["qc"], d["cf"], d["cc"], cond=d["cond"]),
    ),
]


@pytest.mark.parametrize(
    "stem, attrs, call", _REFERENCE_CASES, ids=[c[0] for c in _REFERENCE_CASES]
)
def test_reference_regression(device, stem, attrs, call):
    # MOD-008b + MOD-008c: load each block from a committed ``.mdlus`` via
    # ``Module.from_checkpoint`` and check public attributes and forward output
    # values against committed reference data.
    model = Module.from_checkpoint(str(_DATA_DIR / f"{stem}.mdlus")).to(device).eval()
    for name, expected in attrs.items():
        assert getattr(model, name) == expected
    ref = torch.load(_DATA_DIR / f"{stem}.pth", weights_only=True)
    inputs = {k: v.to(device) for k, v in ref.items() if k != "out"}
    with torch.no_grad():
        out = call(model, inputs)
    # atol/rtol loosened from the 1e-5 default to absorb the TE-vs-torch
    # LayerNorm kernel divergence when the test runs on CUDA (reference is
    # generated on CPU / torch LayerNorm); a real regression fails far wider.
    torch.testing.assert_close(out, ref["out"].to(device), atol=1e-4, rtol=1e-4)
