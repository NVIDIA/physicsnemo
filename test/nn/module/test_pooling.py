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

"""Tests for physicsnemo.nn.module.pooling (AttentionPooling, MeanPooling)."""

import pytest
import torch

from physicsnemo.nn.module.pooling import AttentionPooling, MeanPooling

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(params=[1, 4], ids=["batch1", "batch4"])
def batch_size(request):
    return request.param


@pytest.fixture(params=[100, 500], ids=["pts100", "pts500"])
def n_points(request):
    return request.param


@pytest.fixture(params=[64, 256], ids=["feat64", "feat256"])
def feat_dim(request):
    return request.param


@pytest.fixture(params=[16, 32], ids=["emb16", "emb32"])
def embed_dim(request):
    return request.param


# ---------------------------------------------------------------------------
# AttentionPooling
# ---------------------------------------------------------------------------


class TestAttentionPooling:
    """Tests for the AttentionPooling module."""

    def test_output_shape(self, batch_size, n_points, feat_dim, embed_dim):
        pool = AttentionPooling(feat_dim=feat_dim, embed_dim=embed_dim)
        x = torch.randn(batch_size, n_points, feat_dim)
        out = pool(x)
        assert out.shape == (batch_size, embed_dim)

    def test_gradient_flow(self):
        pool = AttentionPooling(feat_dim=64, embed_dim=16)
        x = torch.randn(2, 50, 64, requires_grad=True)
        out = pool(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_normalize_unit_norm(self):
        pool = AttentionPooling(
            feat_dim=64,
            embed_dim=16,
            normalize=True,
            target_scale=1.0,
        )
        x = torch.randn(4, 100, 64)
        out = pool(x)
        norms = out.norm(dim=-1)
        torch.testing.assert_close(
            norms,
            torch.ones(4),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_normalize_custom_scale(self):
        scale = 2.5
        pool = AttentionPooling(
            feat_dim=64,
            embed_dim=16,
            normalize=True,
            target_scale=scale,
        )
        x = torch.randn(4, 100, 64)
        out = pool(x)
        norms = out.norm(dim=-1)
        torch.testing.assert_close(
            norms,
            torch.full((4,), scale),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_no_normalize(self):
        pool = AttentionPooling(feat_dim=64, embed_dim=16, normalize=False)
        x = torch.randn(4, 100, 64)
        out = pool(x)
        norms = out.norm(dim=-1)
        assert not torch.allclose(norms, torch.ones(4), atol=0.1)

    def test_spectral_norm_applied(self):
        pool = AttentionPooling(
            feat_dim=64,
            embed_dim=16,
            spectral_norm=True,
        )
        has_sn = False
        for module in pool.modules():
            if hasattr(module, "parametrizations"):
                has_sn = True
                break
        assert has_sn, "spectral_norm=True should add parametrizations"

    def test_deterministic_eval(self):
        pool = AttentionPooling(feat_dim=64, embed_dim=16)
        pool.eval()
        x = torch.randn(2, 50, 64)
        out1 = pool(x)
        out2 = pool(x)
        torch.testing.assert_close(out1, out2)


# ---------------------------------------------------------------------------
# MeanPooling
# ---------------------------------------------------------------------------


class TestMeanPooling:
    """Tests for the MeanPooling module."""

    def test_output_shape(self, batch_size, n_points, feat_dim, embed_dim):
        pool = MeanPooling(feat_dim=feat_dim, embed_dim=embed_dim)
        x = torch.randn(batch_size, n_points, feat_dim)
        out = pool(x)
        assert out.shape == (batch_size, embed_dim)

    def test_gradient_flow(self):
        pool = MeanPooling(feat_dim=64, embed_dim=16)
        x = torch.randn(2, 50, 64, requires_grad=True)
        out = pool(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_normalize_unit_norm(self):
        pool = MeanPooling(
            feat_dim=64,
            embed_dim=16,
            normalize=True,
            target_scale=1.0,
        )
        x = torch.randn(4, 100, 64)
        out = pool(x)
        norms = out.norm(dim=-1)
        torch.testing.assert_close(
            norms,
            torch.ones(4),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_spectral_norm_applied(self):
        pool = MeanPooling(feat_dim=64, embed_dim=16, spectral_norm=True)
        has_sn = False
        for module in pool.modules():
            if hasattr(module, "parametrizations"):
                has_sn = True
                break
        assert has_sn

    def test_single_point_equals_projection(self):
        """With a single point, mean pooling should equal projector(point)."""
        pool = MeanPooling(feat_dim=32, embed_dim=8, normalize=False)
        pool.eval()
        x = torch.randn(1, 1, 32)
        out = pool(x)
        expected = pool.projector(x.squeeze(1))
        torch.testing.assert_close(out, expected)
