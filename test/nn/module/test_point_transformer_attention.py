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

import pytest
import torch
import torch.nn as nn

from physicsnemo.nn import (
    AdaLNResidualMLP,
    LocalPointTransformerBlock,
    LocalTokenCrossAttentionBlock,
)
from physicsnemo.nn.module import layer_norm as _layer_norm
from physicsnemo.nn.module.point_transformer_attention import _dilated_knn
from test.common import validate_checkpoint, validate_forward_accuracy

# The blocks normalize with the TE-aware ``LayerNorm``, which resolves once at
# import to Transformer Engine's CUDA-only LayerNorm whenever TE + CUDA are
# available. It cannot run on CPU tensors, so on a TE-enabled GPU node the
# ``device="cpu"`` parametrization is skipped (mirroring physicsnemo's own
# ``test_layer_norm`` handling). On CPU-only environments TE is unavailable,
# ``LayerNorm`` falls back to ``torch.nn.LayerNorm`` and CPU cases run.
_TE_LAYERNORM_CUDA_ONLY = not issubclass(_layer_norm.LayerNorm, torch.nn.LayerNorm)


@pytest.fixture(autouse=True)
def _skip_cpu_under_te_layernorm(request):
    if (
        _TE_LAYERNORM_CUDA_ONLY
        and "device" in request.fixturenames
        and request.getfixturevalue("device") == "cpu"
    ):
        pytest.skip("TE LayerNorm backend is CUDA-only; CPU case not applicable")


# --------------------------------------------------------------------------- #
# builders
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


def _activate_conditioning(block):
    # The AdaLN conditioning MLP is zero-initialized (identity at init); perturb
    # its final layer so conditioning actually modulates the output, for tests
    # that assert the conditioning has an effect.
    with torch.no_grad():
        last = block.conditioning.layers[-1]
        last.weight.normal_(0.0, 0.1)
        last.bias.normal_(0.0, 0.1)


# --------------------------------------------------------------------------- #
# LocalPointTransformerBlock
# --------------------------------------------------------------------------- #
class TestLocalPointTransformerBlock:
    @pytest.mark.parametrize("dim, num_heads", [(32, 4), (64, 8)])
    @pytest.mark.parametrize("dilation", [1, 2, 3])
    def test_output_shape_and_finite(self, device, dim, num_heads, dilation):
        # dilation == 1 is the plain k-NN path; > 1 exercises the wider-then-
        # strided receptive field.
        block = _self_block(dim=dim, num_heads=num_heads, dilation=dilation)
        block = block.to(device).eval()
        feats = torch.randn(40, dim, device=device)
        coords = torch.randn(40, 3, device=device)
        out = block(feats, coords)
        assert out.shape == (40, dim)
        assert torch.isfinite(out).all()

    @pytest.mark.parametrize("coord_dim", [2, 3, 6])
    def test_coordinate_dimensionality(self, device, coord_dim):
        # coord_dim generalizes beyond 3D point clouds (2D meshes, 6-DoF poses).
        block = LocalPointTransformerBlock(
            dim=32,
            num_heads=4,
            neighbor_k=6,
            dilation=1,
            mlp_ratio=2,
            dropout=0.0,
            coord_dim=coord_dim,
        )
        block = block.to(device).eval()
        feats = torch.randn(20, 32, device=device)
        out = block(feats, torch.randn(20, coord_dim, device=device))
        assert out.shape == (20, 32)
        with pytest.raises(ValueError, match="coords"):
            block(feats, torch.randn(20, coord_dim + 1, device=device))

    def test_single_point_skips_attention(self, device):
        # <= 1 point: the attention sublayer is skipped, only the FFN runs.
        block = _self_block().to(device).eval()
        out = block(torch.randn(1, 32, device=device), torch.randn(1, 3, device=device))
        assert out.shape == (1, 32)

    def test_gradient_flow(self, device):
        block = _self_block().to(device).train()
        feats = torch.randn(20, 32, device=device, requires_grad=True)
        block(feats, torch.randn(20, 3, device=device)).sum().backward()
        assert feats.grad is not None
        assert torch.isfinite(feats.grad).all()

    def test_dropout_training_vs_eval(self, device):
        # Dropout is active in train mode (stochastic) and disabled in eval.
        block = LocalPointTransformerBlock(
            dim=32,
            num_heads=4,
            neighbor_k=6,
            dilation=1,
            mlp_ratio=2,
            dropout=0.5,
        ).to(device)
        feats = torch.randn(30, 32, device=device)
        coords = torch.randn(30, 3, device=device)
        block.train()
        assert not torch.allclose(block(feats, coords), block(feats, coords))
        block.eval()
        torch.testing.assert_close(block(feats, coords), block(feats, coords))

    def test_conditioning_zero_init_is_identity(self, device):
        # Zero-initialized AdaLN => conditioned forward equals the unconditioned
        # forward at initialization, regardless of cond.
        torch.manual_seed(0)
        block = _self_block(conditioning_dim=4).to(device).eval()
        feats = torch.randn(20, 32, device=device)
        coords = torch.randn(20, 3, device=device)
        out_a = block(feats, coords, cond=torch.randn(4, device=device))
        out_b = block(feats, coords, cond=torch.zeros(4, device=device))
        torch.testing.assert_close(out_a, out_b)

    @pytest.mark.parametrize("adaln_zero", [False, True])
    def test_active_conditioning_modulates(self, device, adaln_zero):
        # Once the conditioning MLP is non-trivial, different cond changes the
        # output; covers both gating variants (1 + gate vs gate).
        block = _self_block(conditioning_dim=4, adaln_zero=adaln_zero).to(device).eval()
        _activate_conditioning(block)
        feats = torch.randn(20, 32, device=device)
        coords = torch.randn(20, 3, device=device)
        out_a = block(feats, coords, cond=torch.randn(4, device=device))
        out_b = block(feats, coords, cond=torch.randn(4, device=device))
        assert torch.isfinite(out_a).all()
        assert not torch.allclose(out_a, out_b)

    def test_conditioning_requires_cond(self, device):
        block = _self_block(conditioning_dim=4).to(device).eval()
        with pytest.raises(ValueError, match="conditioning"):
            block(torch.randn(8, 32, device=device), torch.randn(8, 3, device=device))

    def test_batch_ids_isolate_neighbors(self, device):
        # Spatially overlapping clouds: with batch_ids, cloud-0 output must be
        # unaffected by perturbing cloud-1 features (the neighbor mask forbids
        # cross-cloud attention).
        torch.manual_seed(0)
        block = _self_block(neighbor_k=4).to(device).eval()
        na = nb = 12
        coords = torch.randn(na + nb, 3, device=device)
        feats = torch.randn(na + nb, 32, device=device)
        batch_ids = torch.cat(
            [torch.zeros(na, dtype=torch.long), torch.ones(nb, dtype=torch.long)]
        ).to(device)
        out1 = block(feats, coords, batch_ids=batch_ids)
        feats2 = feats.clone()
        feats2[na:] += 5.0
        out2 = block(feats2, coords, batch_ids=batch_ids)
        torch.testing.assert_close(out1[:na], out2[:na])

    def test_invalid_input_dims(self, device):
        # MOD-005: forward validates tensor shapes at the API boundary.
        block = _self_block(dim=32).to(device).eval()
        coords = torch.randn(10, 3, device=device)
        with pytest.raises(ValueError, match="features"):
            block(torch.randn(10, 16, device=device), coords)
        with pytest.raises(ValueError, match="coords"):
            block(torch.randn(10, 32, device=device), torch.randn(10, 2, device=device))
        with pytest.raises(ValueError, match="share"):
            block(torch.randn(10, 32, device=device), torch.randn(8, 3, device=device))

    def test_dim_not_divisible_by_heads_raises(self):
        with pytest.raises(ValueError, match="divisible"):
            _self_block(dim=30, num_heads=4)

    @pytest.mark.parametrize(
        "kwargs, expected",
        [
            (
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
                    dim=32,
                    num_heads=4,
                    head_dim=8,
                    neighbor_k=6,
                    dilation=2,
                    coord_dim=3,
                ),
            ),
            (
                dict(
                    dim=48,
                    num_heads=6,
                    neighbor_k=8,
                    dilation=1,
                    mlp_ratio=4,
                    dropout=0.0,
                    coord_dim=2,
                ),
                dict(
                    dim=48,
                    num_heads=6,
                    head_dim=8,
                    neighbor_k=8,
                    dilation=1,
                    coord_dim=2,
                ),
            ),
        ],
    )
    def test_constructor_attributes(self, kwargs, expected):
        # MOD-008a: constructor/attribute coverage across >= 2 configurations.
        block = LocalPointTransformerBlock(**kwargs)
        for name, value in expected.items():
            assert getattr(block, name) == value

    def test_forward_accuracy(self, device):
        # MOD-008b: non-regression against committed reference output.
        torch.manual_seed(0)
        block = _self_block(conditioning_dim=4).to(device).eval()
        # Generate on CPU so CPU and CUDA cases consume the same RNG stream
        # and can compare against one shared reference output.
        feats = torch.randn(24, 32).to(device)
        coords = torch.randn(24, 3).to(device)
        cond = torch.randn(4).to(device)
        assert validate_forward_accuracy(
            block,
            (feats, coords, cond),
            file_name="nn/module/data/local_point_transformer_block_output.pth",
            rtol=1e-3,
            atol=1e-3,
        )

    def test_checkpoint(self, device):
        # MOD-008c: save/load round-trip preserves the forward pass.
        torch.manual_seed(0)
        block_1 = _self_block(conditioning_dim=4).to(device)
        block_2 = _self_block(conditioning_dim=4).to(device)
        feats = torch.randn(24, 32, device=device)
        coords = torch.randn(24, 3, device=device)
        cond = torch.randn(4, device=device)
        assert validate_checkpoint(block_1, block_2, (feats, coords, cond))


# --------------------------------------------------------------------------- #
# LocalTokenCrossAttentionBlock
# --------------------------------------------------------------------------- #
class TestLocalTokenCrossAttentionBlock:
    @pytest.mark.parametrize("dim, num_heads", [(32, 4), (64, 8)])
    def test_output_shape_and_finite(self, device, dim, num_heads):
        block = _cross_block(dim=dim, num_heads=num_heads).to(device).eval()
        qf = torch.randn(25, dim, device=device)
        qc = torch.randn(25, 3, device=device)
        cf = torch.randn(18, dim, device=device)
        cc = torch.randn(18, 3, device=device)
        out = block(qf, qc, cf, cc)
        assert out.shape == (25, dim)
        assert torch.isfinite(out).all()

    def test_zero_tokens_is_noop(self, device):
        block = _cross_block().to(device).eval()
        qf = torch.randn(10, 32, device=device)
        qc = torch.randn(10, 3, device=device)
        out = block(
            qf, qc, torch.randn(0, 32, device=device), torch.randn(0, 3, device=device)
        )
        torch.testing.assert_close(out, qf)

    def test_gradient_flow(self, device):
        block = _cross_block().to(device).train()
        qf = torch.randn(18, 32, device=device, requires_grad=True)
        qc = torch.randn(18, 3, device=device)
        cf = torch.randn(12, 32, device=device)
        cc = torch.randn(12, 3, device=device)
        block(qf, qc, cf, cc).sum().backward()
        assert qf.grad is not None
        assert torch.isfinite(qf.grad).all()

    def test_context_cond_modulates(self, device):
        # The key/value side responds to context_cond independently of cond.
        block = _cross_block(conditioning_dim=4).to(device).eval()
        _activate_conditioning(block)
        qf = torch.randn(18, 32, device=device)
        qc = torch.randn(18, 3, device=device)
        cf = torch.randn(12, 32, device=device)
        cc = torch.randn(12, 3, device=device)
        cond = torch.randn(4, device=device)
        out_a = block(
            qf, qc, cf, cc, cond=cond, context_cond=torch.randn(4, device=device)
        )
        out_b = block(
            qf, qc, cf, cc, cond=cond, context_cond=torch.randn(4, device=device)
        )
        assert not torch.allclose(out_a, out_b)

    def test_per_query_cond_without_context_cond(self, device):
        # Regression: per-query cond with context_cond=None must not crash when
        # N_q != N_c (the KV side reduces it to a single global vector).
        block = _cross_block(conditioning_dim=4).to(device).eval()
        nq, nc = 18, 12
        out = block(
            torch.randn(nq, 32, device=device),
            torch.randn(nq, 3, device=device),
            torch.randn(nc, 32, device=device),
            torch.randn(nc, 3, device=device),
            cond=torch.randn(nq, 4, device=device),
        )
        assert out.shape == (nq, 32)
        assert torch.isfinite(out).all()

    def test_batch_ids_isolate_neighbors(self, device):
        # Batch-0 queries must be unaffected by batch-1 context tokens despite
        # overlapping coordinates.
        torch.manual_seed(0)
        block = _cross_block(neighbor_k=4).to(device).eval()
        nq = nc = 10
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
        cf2[nc // 2 :] += 5.0
        out2 = block(
            qf,
            qc,
            cf2,
            cc,
            query_batch_ids=query_batch_ids,
            context_batch_ids=context_batch_ids,
        )
        torch.testing.assert_close(out1, out2)

    def test_invalid_input_dims(self, device):
        block = _cross_block(dim=32).to(device).eval()
        qc = torch.randn(10, 3, device=device)
        cf = torch.randn(8, 32, device=device)
        cc = torch.randn(8, 3, device=device)
        with pytest.raises(ValueError, match="query_features"):
            block(torch.randn(10, 16, device=device), qc, cf, cc)
        with pytest.raises(ValueError, match="context_coords"):
            block(
                torch.randn(10, 32, device=device),
                qc,
                cf,
                torch.randn(8, 2, device=device),
            )

    @pytest.mark.parametrize(
        "kwargs, expected",
        [
            (
                dict(
                    dim=64,
                    num_heads=8,
                    neighbor_k=8,
                    mlp_ratio=4,
                    dropout=0.0,
                    coord_dim=2,
                ),
                dict(dim=64, num_heads=8, head_dim=8, neighbor_k=8, coord_dim=2),
            ),
            (
                dict(
                    dim=32,
                    num_heads=4,
                    neighbor_k=6,
                    mlp_ratio=2,
                    dropout=0.0,
                    coord_dim=3,
                ),
                dict(dim=32, num_heads=4, head_dim=8, neighbor_k=6, coord_dim=3),
            ),
        ],
    )
    def test_constructor_attributes(self, kwargs, expected):
        block = LocalTokenCrossAttentionBlock(**kwargs)
        for name, value in expected.items():
            assert getattr(block, name) == value

    def test_forward_accuracy(self, device):
        torch.manual_seed(0)
        block = _cross_block(conditioning_dim=4).to(device).eval()
        # Generate on CPU so CPU and CUDA cases consume the same RNG stream
        # and can compare against one shared reference output.
        qf = torch.randn(20, 32).to(device)
        qc = torch.randn(20, 3).to(device)
        cf = torch.randn(14, 32).to(device)
        cc = torch.randn(14, 3).to(device)
        cond = torch.randn(4).to(device)
        assert validate_forward_accuracy(
            block,
            (qf, qc, cf, cc, cond),
            file_name="nn/module/data/local_token_cross_attention_block_output.pth",
            rtol=1e-3,
            atol=1e-3,
        )

    def test_checkpoint(self, device):
        torch.manual_seed(0)
        block_1 = _cross_block(conditioning_dim=4).to(device)
        block_2 = _cross_block(conditioning_dim=4).to(device)
        qf = torch.randn(20, 32, device=device)
        qc = torch.randn(20, 3, device=device)
        cf = torch.randn(14, 32, device=device)
        cc = torch.randn(14, 3, device=device)
        cond = torch.randn(4, device=device)
        assert validate_checkpoint(block_1, block_2, (qf, qc, cf, cc, cond))


# --------------------------------------------------------------------------- #
# AdaLNResidualMLP
# --------------------------------------------------------------------------- #
class TestAdaLNResidualMLP:
    @pytest.mark.parametrize("shape", [(7, 16), (2, 5, 16)])
    def test_output_shape(self, device, shape):
        mlp = AdaLNResidualMLP(dim=16, mlp_ratio=4, dropout=0.0).to(device).eval()
        out = mlp(torch.randn(*shape, device=device))
        assert out.shape == shape
        assert torch.isfinite(out).all()

    def test_conditioning_zero_init_is_identity(self, device):
        torch.manual_seed(0)
        mlp = AdaLNResidualMLP(dim=16, mlp_ratio=4, dropout=0.0, conditioning_dim=4)
        mlp = mlp.to(device).eval()
        x = torch.randn(8, 16, device=device)
        out_a = mlp(x, cond=torch.randn(4, device=device))
        out_b = mlp(x, cond=torch.zeros(4, device=device))
        torch.testing.assert_close(out_a, out_b)

    def test_conditioning_requires_cond(self, device):
        mlp = AdaLNResidualMLP(dim=16, mlp_ratio=4, dropout=0.0, conditioning_dim=4)
        with pytest.raises(ValueError, match="conditioning"):
            mlp.to(device).eval()(torch.randn(8, 16, device=device))


# --------------------------------------------------------------------------- #
# _dilated_knn (the kNN backing the blocks)
# --------------------------------------------------------------------------- #
class TestDilatedKnn:
    @pytest.mark.parametrize("dilation", [1, 2, 3])
    @pytest.mark.parametrize("k", [1, 4, 8])
    def test_selected_distances_match_reference(self, device, k, dilation):
        # _dilated_knn must select neighbors at the correct (dilation-strided)
        # distance ranks. Validate via the *distances* of the selected
        # neighbors, not raw indices: backends (torch / scipy / cuML) order
        # equal- and near-equal-distance neighbors differently, so exact index
        # equality is not a stable cross-backend invariant -- cf.
        # ``KNN.compare_forward``, which compares distances rather than indices.
        torch.manual_seed(123)
        key = torch.rand(64, 3, device=device) * 100.0
        query = torch.rand(20, 3, device=device) * 100.0
        got = _dilated_knn(query_coords=query, key_coords=key, k=k, dilation=dilation)

        dist = torch.cdist(query.float(), key.float())
        sorted_dist = dist.sort(dim=1).values
        k_wide = min(k * dilation, key.shape[0])
        expected = sorted_dist[:, :k_wide]
        if dilation > 1:
            expected = expected[:, ::dilation]
        expected = expected[:, : max(1, k_wide // dilation)]

        got_dist = torch.gather(dist, 1, got).sort(dim=1).values
        assert got_dist.shape == expected.shape
        torch.testing.assert_close(got_dist, expected, rtol=1e-3, atol=1e-2)

    def test_self_includes_self(self, device):
        pts = torch.rand(30, 3, device=device) * 100.0
        idx = _dilated_knn(query_coords=pts, key_coords=pts, k=5, dilation=1)
        torch.testing.assert_close(idx[:, 0], torch.arange(30, device=device))

    def test_k_clamped_to_key_count(self, device):
        idx = _dilated_knn(
            query_coords=torch.rand(8, 3, device=device),
            key_coords=torch.rand(5, 3, device=device),
            k=10,
            dilation=1,
        )
        assert idx.shape[1] == 5
        assert (idx >= 0).all() and (idx < 5).all()


def test_sqrt_scaling_absorption_identity():
    # The blocks omit the `/ sqrt(head_dim)` divisor that scaled-dot-product
    # attention uses, because the score here is a learned MLP output, not an
    # inner product: dividing it by a constant is exactly absorbable into the
    # final Linear's weights. This guards that justification.
    torch.manual_seed(0)
    head_dim = 16
    s = head_dim**0.5
    lin = nn.Linear(32, 8)
    x = torch.randn(5, 32)
    absorbed = nn.Linear(32, 8)
    with torch.no_grad():
        absorbed.weight.copy_(lin.weight / s)
        absorbed.bias.copy_(lin.bias / s)
    torch.testing.assert_close(lin(x) / s, absorbed(x))
