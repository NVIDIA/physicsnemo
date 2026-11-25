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

"""Tests for the GP training helpers and the head-side training API."""

import pytest
import torch
import torch.nn as nn

from physicsnemo.experimental.uq.gp import (
    kmeans_medoids,
    latent_distance_penalty,
    linear_warmup,
    subsample_points,
)

# ---------------------------------------------------------------------------
# Pure helpers (no gpytorch needed)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "epoch, start, end, expected",
    [
        (0, 10, 20, 0.0),
        (15, 10, 20, 0.5),
        (20, 10, 20, 1.0),
        (99, 10, 20, 1.0),
        (0, 10, 10, 1.0),
        (0, 20, 10, 1.0),
    ],
)
def test_linear_warmup(epoch, start, end, expected):
    """0 before the window, linear inside, 1 after; empty window disables it."""
    assert linear_warmup(epoch, start, end) == pytest.approx(expected)


@pytest.mark.parametrize("n_points", [None, 0, 100, 500])
def test_subsample_points_passthrough(n_points):
    """None, 0 or n >= N return the inputs untouched."""
    f, t = torch.randn(2, 100, 8), torch.randn(2, 100, 3)
    f2, t2 = subsample_points(f, t, n_points)
    assert f2 is f and t2 is t


def test_subsample_points_keeps_pairs():
    """The same points are kept for features and targets in every sample."""
    torch.manual_seed(0)
    f = torch.arange(2 * 50, dtype=torch.float32).reshape(2, 50, 1).expand(2, 50, 4)
    t = f[..., :1] * 10
    f2, t2 = subsample_points(f, t, 7)
    assert f2.shape == (2, 7, 4) and t2.shape == (2, 7, 1)
    torch.testing.assert_close(t2, f2[..., :1] * 10)


def test_latent_distance_penalty_collapse_vs_spread():
    """Collapsed features are penalized; features that track the target are not."""
    torch.manual_seed(0)
    target = torch.randn(2, 64, 1)
    spread = torch.cat([target * 5, torch.zeros(2, 64, 3)], dim=-1)
    collapsed = torch.randn(2, 64, 4) * 1e-3
    pen_spread = latent_distance_penalty(spread, target, 512, 1.0)
    pen_collapsed = latent_distance_penalty(collapsed, target, 512, 1.0)
    assert pen_spread.item() == pytest.approx(0.0, abs=1e-5)
    assert pen_collapsed.item() > 0.1


def test_latent_distance_penalty_grad_and_degenerate():
    """Gradient reaches the features; too few points or pairs give zero."""
    feats = torch.randn(1, 32, 4, requires_grad=True)
    latent_distance_penalty(feats * 1e-3, torch.randn(1, 32, 1), 64, 1.0).backward()
    assert feats.grad is not None and feats.grad.abs().sum() > 0
    assert (
        latent_distance_penalty(torch.randn(1, 1, 4), torch.randn(1, 1, 1), 8, 1.0) == 0
    )
    assert (
        latent_distance_penalty(torch.randn(1, 9, 4), torch.randn(1, 9, 1), 0, 1.0) == 0
    )


def test_kmeans_medoids_covers_small_clusters():
    """A uniform draw would almost never hit the two small clusters; k-means does."""
    g = torch.Generator().manual_seed(0)
    big = torch.randn(2000, 3, generator=g)
    small_a = torch.randn(20, 3, generator=g) * 0.05 + 10.0
    small_b = torch.randn(20, 3, generator=g) * 0.05 - 10.0
    x = torch.cat([big, small_a, small_b])
    idx = kmeans_medoids(x, 8, generator=torch.Generator().manual_seed(1))
    assert idx.numel() == 8 and idx.unique().numel() == 8
    picked = x[idx]
    assert (picked[:, 0] > 5).any() and (picked[:, 0] < -5).any()


def test_kmeans_medoids_distinct_with_duplicates_and_small_n():
    """Exactly k distinct indices even with duplicate points; k >= N returns all."""
    x = torch.cat([torch.zeros(50, 2), torch.ones(3, 2)])
    idx = kmeans_medoids(x, 10, generator=torch.Generator().manual_seed(0))
    assert idx.unique().numel() == 10
    torch.testing.assert_close(kmeans_medoids(x[:5], 9), torch.arange(5))


# ---------------------------------------------------------------------------
# Head-side API (needs gpytorch)
# ---------------------------------------------------------------------------


def _field_head(**kw):
    pytest.importorskip("gpytorch")
    from physicsnemo.experimental.uq.gp import FieldVariationalGPHead

    args = dict(
        input_dim=16,
        n_train=1000,
        num_tasks=2,
        n_inducing=8,
        mlp_hidden=[16, 4],
        feature_norm="l2_radial",
        noise_mlp_hidden=[8],
    )
    args.update(kw)
    return FieldVariationalGPHead(**args)


@pytest.mark.parametrize("method", ["kmeans", "random"])
def test_seed_inducing_points(method):
    """Seeding installs n_inducing points, resets q(u) and restores train mode."""
    torch.manual_seed(0)
    head = _field_head()
    head.train()
    base = head.gp_layer.variational_strategy.base_variational_strategy
    base._variational_distribution.variational_mean.data.fill_(3.0)
    before = base.inducing_points.detach().clone()

    feats = torch.randn(4, 200, 16) * 7 + 2  # off the BN's initial stats
    head.seed_inducing_points(
        feats, method=method, generator=torch.Generator().manual_seed(0)
    )

    assert head.training
    assert not torch.equal(base.inducing_points, before)
    assert base.inducing_points.shape == before.shape
    assert torch.all(base._variational_distribution.variational_mean == 0)
    # The radial BatchNorm was re-estimated on the candidates' DKL norms.
    norms = head.feature_extractor(feats.reshape(-1, 16)).norm(dim=-1)
    torch.testing.assert_close(
        head._radial_bn.running_mean, norms.mean().reshape(1), rtol=1e-4, atol=1e-4
    )
    head.predict(feats[:1])  # still usable


def test_seed_inducing_points_errors():
    """Unknown methods and too few candidates are rejected."""
    head = _field_head()
    with pytest.raises(ValueError, match="method"):
        head.seed_inducing_points(torch.randn(50, 16), method="bogus")
    with pytest.raises(ValueError, match="at least 8"):
        head.seed_inducing_points(torch.randn(5, 16))


def test_set_num_data_scales_kl():
    """num_data reaches the ELBO and changes the KL weight."""
    torch.manual_seed(0)
    head = _field_head(noise_mlp_hidden=None)
    feats, tgt = torch.randn(64, 16), torch.randn(64, 2)
    head.set_num_data(10)
    assert head.mll.num_data == 10 and head._num_data == 10
    _, small_n = head.forward_and_loss(feats, tgt)
    head.set_num_data(10**6)
    _, large_n = head.forward_and_loss(feats, tgt)
    assert small_n.item() > large_n.item()  # stronger KL with fewer data
    head.set_num_data(0)
    assert head.mll.num_data == 1


def test_param_groups_partition():
    """Every parameter appears exactly once; networks go to gp_nn."""
    head = _field_head()
    groups = head.param_groups(gp_lr=1e-2, nn_lr=1e-3)
    assert [g["name"] for g in groups] == ["gp", "gp_nn"]
    ids = [id(p) for g in groups for p in g["params"]]
    assert len(ids) == len(set(ids)) == len(list(head.parameters()))
    nn_ids = {id(p) for p in head.feature_extractor.parameters()} | {
        id(p) for p in head.noise_head.parameters()
    }
    assert {id(p) for p in groups[1]["params"]} == nn_ids
    assert id(head.log_base_noise) in {id(p) for p in groups[0]["params"]}
    torch.optim.AdamW(groups)  # accepted by torch


def test_param_groups_without_networks():
    """No DKL and no noise MLP: a single gp group."""
    groups = _field_head(mlp_hidden=None, noise_mlp_hidden=None).param_groups()
    assert [g["name"] for g in groups] == ["gp"]


def test_dtype_cast_keeps_gp_double():
    """A parent-level float32 cast leaves the GP internals in float64."""
    torch.manual_seed(0)
    head = _field_head()
    parent = nn.Module()
    parent.head = head
    parent.to(torch.float32)
    assert all(p.dtype == torch.float64 for p in head.gp_layer.parameters())
    assert all(p.dtype == torch.float64 for p in head.noise_head.parameters())
    assert head.log_base_noise.dtype == torch.float64
    assert all(p.dtype == torch.float32 for p in head.feature_extractor.parameters())
    mean, loss = head.forward_and_loss(torch.randn(32, 16), torch.randn(32, 2))
    assert torch.isfinite(loss) and mean.dtype == torch.float32


def test_scalar_head_dtype_and_num_data():
    """VariationalGPHead: float64 survives a cast, and set_num_data works."""
    pytest.importorskip("gpytorch")
    from physicsnemo.experimental.uq.gp import VariationalGPHead

    head = VariationalGPHead(input_dim=4, n_train=100, n_inducing=4)
    head.to(torch.float32)
    assert all(p.dtype == torch.float64 for p in head.gp_layer.parameters())
    head.set_num_data(7)
    assert head.mll.num_data == 7
    head.predict(torch.randn(3, 4))
