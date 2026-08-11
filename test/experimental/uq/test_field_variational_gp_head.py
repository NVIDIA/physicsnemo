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

"""Tests for FieldVariationalGPHead, the per-point multitask variational GP head."""

import pytest
import torch
import torch.nn as nn

pytest.importorskip("gpytorch", reason="FieldVariationalGPHead requires gpytorch")

from physicsnemo.experimental.uq import (  # noqa: E402
    FieldVariationalGPHead,
    FieldVariationalGPPrediction,
)

INPUT_DIM = 32
NUM_TASKS = 4
N_TRAIN = 8192


def make_head(device, **overrides):
    """Build a small head; kwargs override the defaults."""
    kwargs = dict(
        input_dim=INPUT_DIM,
        num_tasks=NUM_TASKS,
        n_inducing=16,
        n_train=N_TRAIN,
        mlp_hidden=[16, 8],
    )
    kwargs.update(overrides)
    return FieldVariationalGPHead(**kwargs).to(device)


@pytest.mark.parametrize("lead_shape", [(2, 64), (64,), (2, 3, 16)])
def test_predict_shapes(device, lead_shape):
    """predict() preserves arbitrary leading dims and appends num_tasks."""
    torch.manual_seed(0)
    head = make_head(device)
    feats = torch.randn(*lead_shape, INPUT_DIM, device=device)

    pred = head.predict(feats)

    assert isinstance(pred, FieldVariationalGPPrediction)
    expected = (*lead_shape, NUM_TASKS)
    for name in ("mean", "variance", "lower", "upper", "epistemic_variance"):
        tensor = getattr(pred, name)
        assert tensor.shape == expected, f"{name} has shape {tensor.shape}"
        assert torch.isfinite(tensor).all(), f"{name} is not finite"


def test_predict_variance_and_interval_are_consistent(device):
    """Variance is positive, contains the epistemic part, and brackets the mean."""
    torch.manual_seed(0)
    head = make_head(device)
    feats = torch.randn(2, 64, INPUT_DIM, device=device)

    pred = head.predict(feats)

    assert (pred.variance > 0).all()
    assert (pred.epistemic_variance > 0).all()
    # Total variance = epistemic + observation noise, so it cannot be smaller.
    assert (pred.variance >= pred.epistemic_variance - 1e-9).all()
    assert (pred.lower < pred.mean).all()
    assert (pred.upper > pred.mean).all()


def test_head_is_backbone_agnostic(device):
    """Any module emitting (..., input_dim) features can drive the head.

    Uses a plain nn.Linear as the "backbone" to pin the contract: the head needs
    only a feature tensor, and gradients must flow back into that module so it
    can be trained jointly.
    """
    torch.manual_seed(0)
    backbone = nn.Linear(7, INPUT_DIM).to(device)
    head = make_head(device)

    raw = torch.randn(2, 48, 7, device=device)
    target = torch.randn(2, 48, NUM_TASKS, device=device)

    feats = backbone(raw)
    mean, neg_elbo = head.forward_and_loss(feats, target)
    assert mean.shape == (2, 48, NUM_TASKS)

    neg_elbo.backward()
    assert backbone.weight.grad is not None
    assert torch.isfinite(backbone.weight.grad).all()
    assert backbone.weight.grad.abs().sum() > 0, "no gradient reached the backbone"


def test_loss_trains_head_parameters(device):
    """A few optimizer steps reduce the loss and update GP parameters."""
    torch.manual_seed(0)
    head = make_head(device)
    feats = torch.randn(2, 64, INPUT_DIM, device=device)
    target = torch.randn(2, 64, NUM_TASKS, device=device)

    before = head.gp_layer.variational_strategy.base_variational_strategy._variational_distribution.variational_mean.detach().clone()
    opt = torch.optim.Adam(head.parameters(), lr=1e-2)

    losses = []
    for _ in range(5):
        opt.zero_grad()
        loss = head.loss(feats, target)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    assert all(torch.isfinite(torch.tensor(x)) for x in losses)
    assert losses[-1] < losses[0], f"loss did not decrease: {losses}"
    after = head.gp_layer.variational_strategy.base_variational_strategy._variational_distribution.variational_mean
    assert not torch.allclose(before, after)


def test_l2_radial_adds_one_kernel_dimension(device):
    """'l2_radial' appends the standardized feature magnitude as an ARD dim."""
    plain = make_head(device, feature_norm="none")
    radial = make_head(device, feature_norm="l2_radial")

    assert plain.gp_input_dim == 8
    assert radial.gp_input_dim == 9
    # The magnitude is standardized by a non-affine BatchNorm, which contributes
    # running buffers (and therefore state_dict keys) the plain variant lacks.
    assert plain._radial_bn is None
    assert radial._radial_bn is not None
    assert "_radial_bn.running_mean" in radial.state_dict()


def test_invalid_feature_norm_is_rejected():
    """Modes dropped after the exploratory phase must fail loudly, not silently."""
    for bad in ("l2", "layernorm", "batchnorm"):
        with pytest.raises(ValueError, match="feature_norm"):
            FieldVariationalGPHead(
                input_dim=INPUT_DIM, n_train=N_TRAIN, feature_norm=bad
            )


def test_n_train_is_required():
    """n_train sets the ELBO normalizer, so it cannot be defaulted silently."""
    with pytest.raises(TypeError, match="n_train"):
        FieldVariationalGPHead(input_dim=INPUT_DIM)


def test_tuning_knobs_are_keyword_only():
    """Only input_dim is positional, so a config block cannot be mis-ordered."""
    with pytest.raises(TypeError):
        FieldVariationalGPHead(INPUT_DIM, N_TRAIN)


@pytest.mark.parametrize("matern_nu", [0.5, 1.5, 2.5])
def test_matern_order_is_configurable(device, matern_nu):
    """The kernel's smoothness order is exposed, and reaches the kernel."""
    head = make_head(device, matern_nu=matern_nu)
    assert head.gp_layer.covar_module.base_kernel.nu == matern_nu
    pred = head.predict(torch.randn(1, 32, INPUT_DIM, device=device))
    assert torch.isfinite(pred.variance).all()


@pytest.mark.parametrize("feature_norm", ["none", "l2_radial"])
def test_state_dict_round_trip_is_strict_and_exact(device, feature_norm):
    """A head rebuilt from config loads its own state_dict with strict=True.

    This is the checkpoint-compatibility contract: reconstructing the head from
    the same hyperparameters must yield an identical parameter layout, and the
    reloaded head must reproduce predictions exactly.
    """
    torch.manual_seed(0)
    head = make_head(device, feature_norm=feature_norm)
    feats = torch.randn(2, 32, INPUT_DIM, device=device)
    expected = head.predict(feats)

    clone = make_head(device, feature_norm=feature_norm)
    missing, unexpected = clone.load_state_dict(head.state_dict(), strict=True)
    assert not missing and not unexpected

    reloaded = clone.predict(feats)
    torch.testing.assert_close(reloaded.mean, expected.mean, rtol=0, atol=0)
    torch.testing.assert_close(reloaded.variance, expected.variance, rtol=0, atol=0)


def test_heteroscedastic_noise_is_input_dependent(device):
    """The noise head makes the aleatoric variance vary per point.

    With homoscedastic noise the total-minus-epistemic gap is one constant per
    channel, so the total std ranks points exactly like the epistemic std. That
    is the property the noise MLP exists to break.
    """
    torch.manual_seed(0)
    feats = torch.randn(1, 128, INPUT_DIM, device=device)

    homosced = make_head(device)
    assert homosced.heteroscedastic is False
    assert homosced.noise_head is None
    gap = homosced.predict(feats).variance - homosced.predict(feats).epistemic_variance
    # One value per channel, identical across every point.
    for task in range(NUM_TASKS):
        assert torch.allclose(gap[0, :, task], gap[0, 0, task], atol=1e-6)

    hetero = make_head(device, noise_mlp_hidden=[8, 8])
    assert hetero.heteroscedastic is True
    # Perturb the zero-initialized output layer so the modulation is not exactly 1x.
    with torch.no_grad():
        for param in hetero.noise_head[-1].parameters():
            param.add_(torch.randn_like(param) * 0.5)

    pred = hetero.predict(feats)
    het_gap = pred.variance - pred.epistemic_variance
    assert (het_gap > 0).all()
    assert het_gap[0, :, 0].std() > 0, "noise did not vary across points"


def test_noise_std_range_clamps_the_noise(device):
    """noise_std_range is a hard clamp guarding the 1/sigma^2 ELBO weighting."""
    torch.manual_seed(0)
    lo, hi = 0.5, 0.6
    head = make_head(device, noise_mlp_hidden=[8, 8], noise_std_range=(lo, hi))
    # Drive the noise MLP hard in both directions; the clamp must still hold.
    with torch.no_grad():
        head.log_base_noise.add_(10.0)
        for param in head.noise_head[-1].parameters():
            param.add_(torch.randn_like(param) * 5.0)

    feats = torch.randn(1, 64, INPUT_DIM, device=device)
    pred = head.predict(feats)
    noise_var = pred.variance - pred.epistemic_variance

    assert (noise_var >= lo**2 - 1e-6).all()
    assert (noise_var <= hi**2 + 1e-6).all()


def test_set_inducing_points_accepts_shared_and_per_task(device):
    """Inducing points can be re-seeded from collected features."""
    torch.manual_seed(0)
    head = make_head(device, n_inducing=16)
    strategy = head.gp_layer.variational_strategy.base_variational_strategy

    shared = torch.randn(16, INPUT_DIM, device=device)
    head.set_inducing_points(shared)
    assert strategy.inducing_points.shape == (NUM_TASKS, 16, head.gp_input_dim)
    # A shared seed is broadcast, so every task starts from the same locations.
    assert torch.allclose(strategy.inducing_points[0], strategy.inducing_points[1])

    per_task = torch.randn(NUM_TASKS, 16, INPUT_DIM, device=device)
    head.set_inducing_points(per_task)
    assert strategy.inducing_points.shape == (NUM_TASKS, 16, head.gp_input_dim)
    assert not torch.allclose(strategy.inducing_points[0], strategy.inducing_points[1])


def test_transform_features_matches_the_kernel_input(device):
    """The public transform returns exactly what the kernel sees.

    The trainer relies on this to compute auxiliary losses in kernel space and
    pass the result back via ``pretransformed=True`` without re-running the
    transform (which would update the BatchNorm statistics twice per step).
    """
    torch.manual_seed(0)
    head = make_head(device, feature_norm="l2_radial")
    head.eval()  # freeze the BatchNorm so the two calls are comparable
    feats = torch.randn(2, 32, INPUT_DIM, device=device)
    target = torch.randn(2, 32, NUM_TASKS, device=device)

    gp_feats = head.transform_features(feats)
    assert gp_feats.shape == (2, 32, head.gp_input_dim)

    direct, _ = head.forward_and_loss(feats, target)
    pre, _ = head.forward_and_loss(gp_feats, target, pretransformed=True)
    torch.testing.assert_close(direct, pre)


def test_predict_restores_training_mode(device):
    """predict() must not leave the module in eval mode."""
    head = make_head(device)
    head.train()
    head.predict(torch.randn(1, 16, INPUT_DIM, device=device))
    assert head.training is True
