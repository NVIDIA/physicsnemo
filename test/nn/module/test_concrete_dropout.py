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

"""Tests for the concrete_dropout module.

This module tests the ConcreteDropout implementation used for learned per-layer
dropout rates via the concrete relaxation (Gal, Hron & Kendall, 2017), as well
as the helper functions collect_concrete_dropout_losses and
get_concrete_dropout_rates.
"""

import pytest
import torch

from physicsnemo.nn import (
    ConcreteDropout,
    collect_concrete_dropout_losses,
    get_concrete_dropout_rates,
)

# =============================================================================
# ConcreteDropout Basic Tests
# =============================================================================


class TestConcreteDropoutBasic:
    """Basic functionality tests for ConcreteDropout."""

    @pytest.mark.parametrize("batch_size", [1, 4, 16])
    @pytest.mark.parametrize("features", [8, 64, 256])
    def test_output_shape_2d(self, device, batch_size, features):
        """Test that output shape matches input shape for 2D input."""
        cd = ConcreteDropout(in_features=features).to(device)
        cd.train()
        x = torch.randn(batch_size, features, device=device)
        out = cd(x)

        assert out.shape == x.shape

    @pytest.mark.parametrize("batch_size", [1, 4])
    @pytest.mark.parametrize("tokens", [10, 100])
    @pytest.mark.parametrize("features", [32, 128])
    def test_output_shape_3d(self, device, batch_size, tokens, features):
        """Test that output shape matches input shape for 3D input."""
        cd = ConcreteDropout(in_features=features).to(device)
        cd.train()
        x = torch.randn(batch_size, tokens, features, device=device)
        out = cd(x)

        assert out.shape == x.shape

    def test_output_shape_4d(self, device):
        """Test with 4D input (e.g., batch, heads, tokens, dim)."""
        cd = ConcreteDropout(in_features=16).to(device)
        cd.train()
        x = torch.randn(2, 4, 100, 16, device=device)
        out = cd(x)

        assert out.shape == x.shape

    def test_output_dtype_preserved(self, device):
        """Test that output dtype matches input dtype."""
        cd = ConcreteDropout(in_features=32).to(device)
        cd.train()
        x = torch.randn(4, 32, device=device)
        out = cd(x)

        assert out.dtype == x.dtype

    def test_output_device_preserved(self, device):
        """Test that output is on the same device as input."""
        cd = ConcreteDropout(in_features=32).to(device)
        cd.train()
        x = torch.randn(4, 32, device=device)
        out = cd(x)

        assert out.device == x.device


# =============================================================================
# Train / Eval Mode Tests
# =============================================================================


class TestConcreteDropoutModes:
    """Tests for train vs eval mode behavior."""

    def test_eval_mode_is_identity(self, device):
        """Test that eval mode returns input unchanged."""
        cd = ConcreteDropout(in_features=32).to(device)
        cd.eval()
        x = torch.randn(4, 32, device=device)
        out = cd(x)

        assert torch.equal(out, x)

    def test_eval_mode_deterministic(self, device):
        """Test that eval mode produces identical outputs across calls."""
        cd = ConcreteDropout(in_features=32).to(device)
        cd.eval()
        x = torch.randn(4, 32, device=device)

        out1 = cd(x)
        out2 = cd(x)

        assert torch.equal(out1, out2)

    def test_train_mode_modifies_input(self, device):
        """Test that train mode produces output different from input."""
        cd = ConcreteDropout(in_features=32, init_p=0.5).to(device)
        cd.train()
        x = torch.randn(8, 32, device=device)
        out = cd(x)

        assert not torch.equal(out, x)

    def test_mode_switching(self, device):
        """Test switching between train and eval modes."""
        cd = ConcreteDropout(in_features=32).to(device)
        x = torch.randn(4, 32, device=device)

        cd.eval()
        out_eval = cd(x)
        assert torch.equal(out_eval, x)

        cd.train()
        out_train = cd(x)
        assert not torch.equal(out_train, x)

        cd.eval()
        out_eval2 = cd(x)
        assert torch.equal(out_eval2, x)


# =============================================================================
# Stochasticity Tests
# =============================================================================


class TestConcreteDropoutStochasticity:
    """Tests for stochastic behavior."""

    def test_outputs_differ_across_calls(self, device):
        """Test that train-mode outputs are stochastic."""
        cd = ConcreteDropout(in_features=64, init_p=0.3).to(device)
        cd.train()
        x = torch.randn(8, 64, device=device)

        out1 = cd(x)
        out2 = cd(x)

        assert not torch.allclose(out1, out2)

    def test_deterministic_with_seed(self, device):
        """Test that outputs are reproducible with same seed."""
        cd = ConcreteDropout(in_features=64).to(device)
        cd.train()
        x = torch.randn(8, 64, device=device)

        torch.manual_seed(42)
        out1 = cd(x)

        torch.manual_seed(42)
        out2 = cd(x)

        assert torch.allclose(out1, out2)


# =============================================================================
# Dropout Probability Tests
# =============================================================================


class TestConcreteDropoutProbability:
    """Tests for the learned dropout probability."""

    def test_p_property_matches_sigmoid_of_logit(self, device):
        """Test that p = sigmoid(p_logit)."""
        cd = ConcreteDropout(in_features=32).to(device)
        expected_p = torch.sigmoid(cd.p_logit)

        assert torch.allclose(cd.p, expected_p)

    @pytest.mark.parametrize("init_p", [0.01, 0.1, 0.3, 0.5, 0.7, 0.99])
    def test_init_p_respected(self, device, init_p):
        """Test that initial p matches the requested value."""
        cd = ConcreteDropout(in_features=32, init_p=init_p).to(device)

        assert torch.allclose(cd.p, torch.tensor(init_p, device=device), atol=1e-5)

    def test_init_p_clamped_low(self, device):
        """Test that init_p below 0.01 is clamped."""
        cd = ConcreteDropout(in_features=32, init_p=0.0).to(device)

        assert cd.p.item() >= 0.01 - 1e-6

    def test_init_p_clamped_high(self, device):
        """Test that init_p above 0.99 is clamped."""
        cd = ConcreteDropout(in_features=32, init_p=1.0).to(device)

        assert cd.p.item() <= 0.99 + 1e-6

    def test_p_is_learnable(self, device):
        """Test that p_logit is a learnable parameter."""
        cd = ConcreteDropout(in_features=32).to(device)

        assert isinstance(cd.p_logit, torch.nn.Parameter)
        assert cd.p_logit.requires_grad

    def test_p_bounded_after_gradient_step(self, device):
        """Test that p stays in (0, 1) after gradient updates."""
        cd = ConcreteDropout(in_features=32).to(device)
        cd.train()

        x = torch.randn(8, 32, device=device)
        optimizer = torch.optim.SGD(cd.parameters(), lr=100.0)

        for _ in range(10):
            out = cd(x)
            loss = out.sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        assert cd.p.item() > 0.0
        assert cd.p.item() < 1.0


# =============================================================================
# Regularization Loss Tests
# =============================================================================


class TestConcreteDropoutRegularizationLoss:
    """Tests for the regularization loss."""

    def test_reg_loss_is_scalar(self, device):
        """Test that regularization loss is a scalar tensor."""
        cd = ConcreteDropout(in_features=32).to(device)
        loss = cd.regularization_loss()

        assert loss.dim() == 0

    def test_reg_loss_is_negative(self, device):
        """Test that entropy regularization is negative (or zero)."""
        cd = ConcreteDropout(in_features=32, init_p=0.3).to(device)
        loss = cd.regularization_loss()

        # Bernoulli entropy: p*log(p) + (1-p)*log(1-p) is always <= 0
        assert loss.item() <= 0.0

    @pytest.mark.parametrize("init_p", [0.1, 0.3, 0.5, 0.7, 0.9])
    def test_reg_loss_matches_bernoulli_entropy(self, device, init_p):
        """Test that loss matches the Bernoulli entropy formula."""
        cd = ConcreteDropout(in_features=32, init_p=init_p).to(device)
        loss = cd.regularization_loss()

        p = torch.tensor(init_p, device=device)
        eps = 1e-7
        expected = p * torch.log(p + eps) + (1.0 - p) * torch.log(1.0 - p + eps)

        assert torch.allclose(loss, expected, atol=1e-5)

    def test_reg_loss_maximally_negative_at_half(self, device):
        """Test that entropy is most negative at p=0.5."""
        cd_half = ConcreteDropout(in_features=32, init_p=0.5).to(device)
        cd_low = ConcreteDropout(in_features=32, init_p=0.1).to(device)
        cd_high = ConcreteDropout(in_features=32, init_p=0.9).to(device)

        loss_half = cd_half.regularization_loss().item()
        loss_low = cd_low.regularization_loss().item()
        loss_high = cd_high.regularization_loss().item()

        assert loss_half < loss_low
        assert loss_half < loss_high

    def test_reg_loss_has_grad(self, device):
        """Test that regularization loss has gradient w.r.t. p_logit."""
        cd = ConcreteDropout(in_features=32).to(device)
        loss = cd.regularization_loss()
        loss.backward()

        assert cd.p_logit.grad is not None
        assert not torch.isnan(cd.p_logit.grad)

    def test_reg_loss_on_correct_device(self, device):
        """Test that regularization loss is on the same device as the module."""
        cd = ConcreteDropout(in_features=32).to(device)
        loss = cd.regularization_loss()

        assert str(loss.device) == device


# =============================================================================
# Temperature Tests
# =============================================================================


class TestConcreteDropoutTemperature:
    """Tests for temperature parameter behavior."""

    def test_low_temperature_produces_sharper_masks(self, device):
        """Test that lower temperature produces more binary-like masks."""
        x = torch.randn(64, 128, device=device)

        cd_low = ConcreteDropout(in_features=128, init_p=0.5, temperature=0.01).to(
            device
        )
        cd_high = ConcreteDropout(in_features=128, init_p=0.5, temperature=1.0).to(
            device
        )
        cd_low.train()
        cd_high.train()

        torch.manual_seed(42)
        out_low = cd_low(x)
        torch.manual_seed(42)
        out_high = cd_high(x)

        # Lower temperature -> output values closer to 0 or 2*x (binary mask)
        # Higher temperature -> output values more spread out
        # Measure variance of output / input ratio
        ratio_low = (out_low / (x + 1e-8)).std()
        ratio_high = (out_high / (x + 1e-8)).std()

        assert ratio_low > ratio_high

    def test_default_temperature(self, device):
        """Test that default temperature is 0.1."""
        cd = ConcreteDropout(in_features=32).to(device)

        assert cd.temperature == 0.1


# =============================================================================
# Gradient Flow Tests
# =============================================================================


class TestConcreteDropoutGradients:
    """Tests for gradient flow through ConcreteDropout."""

    def test_gradient_flow_to_input(self, device):
        """Test that gradients flow through to the input."""
        cd = ConcreteDropout(in_features=32).to(device)
        cd.train()
        x = torch.randn(4, 32, device=device, requires_grad=True)

        out = cd(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_gradient_not_nan(self, device):
        """Test that gradients are not NaN."""
        cd = ConcreteDropout(in_features=32).to(device)
        cd.train()
        x = torch.randn(4, 32, device=device, requires_grad=True)

        out = cd(x)
        loss = out.sum()
        loss.backward()

        assert not torch.any(torch.isnan(x.grad))

    def test_gradient_to_p_logit(self, device):
        """Test that gradients flow to the learnable dropout logit."""
        cd = ConcreteDropout(in_features=32).to(device)
        cd.train()
        x = torch.randn(4, 32, device=device)

        out = cd(x)
        loss = out.sum()
        loss.backward()

        assert cd.p_logit.grad is not None
        assert not torch.isnan(cd.p_logit.grad)

    def test_combined_loss_gradient(self, device):
        """Test gradient flow with data loss + regularization loss."""
        cd = ConcreteDropout(in_features=32).to(device)
        cd.train()
        x = torch.randn(4, 32, device=device, requires_grad=True)

        out = cd(x)
        data_loss = out.sum()
        reg_loss = cd.regularization_loss()
        total_loss = data_loss + 1e-4 * reg_loss
        total_loss.backward()

        assert x.grad is not None
        assert cd.p_logit.grad is not None
        assert not torch.any(torch.isnan(x.grad))
        assert not torch.isnan(cd.p_logit.grad)


# =============================================================================
# Numerical Stability Tests
# =============================================================================


class TestConcreteDropoutNumericalStability:
    """Tests for numerical stability."""

    def test_no_nan_output(self, device):
        """Test that no NaN values are produced."""
        cd = ConcreteDropout(in_features=32).to(device)
        cd.train()
        x = torch.randn(8, 32, device=device)
        out = cd(x)

        assert not torch.any(torch.isnan(out))

    def test_no_inf_output(self, device):
        """Test that no infinite values are produced."""
        cd = ConcreteDropout(in_features=32).to(device)
        cd.train()
        x = torch.randn(8, 32, device=device)
        out = cd(x)

        assert not torch.any(torch.isinf(out))

    def test_large_input_values(self, device):
        """Test with very large input values."""
        cd = ConcreteDropout(in_features=32).to(device)
        cd.train()
        x = torch.randn(8, 32, device=device) * 1000
        out = cd(x)

        assert torch.all(torch.isfinite(out))

    def test_small_input_values(self, device):
        """Test with very small input values."""
        cd = ConcreteDropout(in_features=32).to(device)
        cd.train()
        x = torch.randn(8, 32, device=device) * 1e-6
        out = cd(x)

        assert torch.all(torch.isfinite(out))

    def test_zero_input(self, device):
        """Test with zero input."""
        cd = ConcreteDropout(in_features=32).to(device)
        cd.train()
        x = torch.zeros(8, 32, device=device)
        out = cd(x)

        assert torch.all(torch.isfinite(out))
        assert torch.allclose(out, torch.zeros_like(out))

    def test_extreme_init_p_low(self, device):
        """Test numerical stability with very low init_p."""
        cd = ConcreteDropout(in_features=32, init_p=0.01).to(device)
        cd.train()
        x = torch.randn(8, 32, device=device)
        out = cd(x)

        assert torch.all(torch.isfinite(out))
        assert torch.all(torch.isfinite(cd.regularization_loss()))

    def test_extreme_init_p_high(self, device):
        """Test numerical stability with very high init_p."""
        cd = ConcreteDropout(in_features=32, init_p=0.99).to(device)
        cd.train()
        x = torch.randn(8, 32, device=device)
        out = cd(x)

        assert torch.all(torch.isfinite(out))
        assert torch.all(torch.isfinite(cd.regularization_loss()))

    def test_reg_loss_finite_for_all_p(self, device):
        """Test that regularization loss is finite across range of p values."""
        for init_p in [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]:
            cd = ConcreteDropout(in_features=32, init_p=init_p).to(device)
            loss = cd.regularization_loss()
            assert torch.isfinite(loss), f"Non-finite loss at init_p={init_p}"


# =============================================================================
# Module Integration Tests
# =============================================================================


class TestConcreteDropoutModule:
    """Tests for nn.Module integration."""

    def test_in_sequential(self, device):
        """Test ConcreteDropout in nn.Sequential."""
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 20),
            ConcreteDropout(in_features=20),
        ).to(device)
        model.train()

        x = torch.randn(4, 10, device=device)
        out = model(x)

        assert out.shape == (4, 20)

    def test_state_dict_save_load(self, device):
        """Test that p_logit is saved and loaded via state_dict."""
        cd1 = ConcreteDropout(in_features=32, init_p=0.3).to(device)
        cd2 = ConcreteDropout(in_features=32, init_p=0.7).to(device)

        assert not torch.allclose(cd1.p, cd2.p)

        cd2.load_state_dict(cd1.state_dict())

        assert torch.allclose(cd1.p, cd2.p)

    def test_state_dict_contains_p_logit(self, device):
        """Test that state_dict contains the p_logit parameter."""
        cd = ConcreteDropout(in_features=32).to(device)
        state = cd.state_dict()

        assert "p_logit" in state

    def test_parameters_count(self, device):
        """Test that there is exactly one learnable parameter."""
        cd = ConcreteDropout(in_features=32).to(device)
        params = list(cd.parameters())

        assert len(params) == 1
        assert params[0] is cd.p_logit

    def test_extra_repr(self, device):
        """Test that extra_repr contains expected info."""
        cd = ConcreteDropout(in_features=64, init_p=0.2).to(device)
        repr_str = cd.extra_repr()

        assert "in_features=64" in repr_str
        assert "p=" in repr_str

    def test_to_device(self, device):
        """Test moving module to device."""
        cd = ConcreteDropout(in_features=32)
        cd = cd.to(device)

        assert str(cd.p_logit.device) == device


# =============================================================================
# collect_concrete_dropout_losses Tests
# =============================================================================


class TestCollectConcreteDropoutLosses:
    """Tests for the collect_concrete_dropout_losses helper."""

    def test_no_concrete_dropout_returns_zero(self, device):
        """Test that a model with no ConcreteDropout returns zero loss."""
        model = torch.nn.Linear(10, 10).to(device)
        loss = collect_concrete_dropout_losses(model)

        assert loss.item() == 0.0

    def test_single_layer(self, device):
        """Test collection from a single ConcreteDropout layer."""
        cd = ConcreteDropout(in_features=32, init_p=0.3).to(device)
        model = torch.nn.Sequential(torch.nn.Linear(32, 32), cd).to(device)

        collected = collect_concrete_dropout_losses(model)
        direct = cd.regularization_loss()

        assert torch.allclose(collected, direct)

    def test_multiple_layers(self, device):
        """Test that losses are summed across multiple layers."""
        cd1 = ConcreteDropout(in_features=32, init_p=0.2)
        cd2 = ConcreteDropout(in_features=32, init_p=0.5)
        cd3 = ConcreteDropout(in_features=32, init_p=0.8)
        model = torch.nn.Sequential(cd1, cd2, cd3).to(device)

        collected = collect_concrete_dropout_losses(model)
        expected = (
            cd1.regularization_loss()
            + cd2.regularization_loss()
            + cd3.regularization_loss()
        )

        assert torch.allclose(collected, expected)

    def test_collected_loss_has_grad(self, device):
        """Test that collected loss supports backpropagation."""
        cd = ConcreteDropout(in_features=32).to(device)
        model = torch.nn.Sequential(cd).to(device)

        loss = collect_concrete_dropout_losses(model)
        loss.backward()

        assert cd.p_logit.grad is not None

    def test_collected_loss_on_correct_device(self, device):
        """Test that collected loss is on the correct device."""
        cd = ConcreteDropout(in_features=32).to(device)
        model = torch.nn.Sequential(cd).to(device)

        loss = collect_concrete_dropout_losses(model)

        assert str(loss.device) == device


# =============================================================================
# get_concrete_dropout_rates Tests
# =============================================================================


class TestGetConcreteDropoutRates:
    """Tests for the get_concrete_dropout_rates helper."""

    def test_no_concrete_dropout_returns_empty(self, device):
        """Test that a model with no ConcreteDropout returns empty dict."""
        model = torch.nn.Linear(10, 10).to(device)
        rates = get_concrete_dropout_rates(model)

        assert rates == {}

    def test_single_layer(self, device):
        """Test rate extraction from a single layer."""
        cd = ConcreteDropout(in_features=32, init_p=0.3).to(device)
        model = torch.nn.Sequential(cd).to(device)
        rates = get_concrete_dropout_rates(model)

        assert len(rates) == 1
        rate = list(rates.values())[0]
        assert abs(rate - 0.3) < 1e-4

    def test_multiple_layers(self, device):
        """Test rate extraction from multiple layers."""
        cd1 = ConcreteDropout(in_features=32, init_p=0.2)
        cd2 = ConcreteDropout(in_features=32, init_p=0.5)
        model = torch.nn.Sequential(cd1, cd2).to(device)
        rates = get_concrete_dropout_rates(model)

        assert len(rates) == 2
        values = sorted(rates.values())
        assert abs(values[0] - 0.2) < 1e-4
        assert abs(values[1] - 0.5) < 1e-4

    def test_rates_are_floats(self, device):
        """Test that extracted rates are Python floats."""
        cd = ConcreteDropout(in_features=32).to(device)
        model = torch.nn.Sequential(cd).to(device)
        rates = get_concrete_dropout_rates(model)

        for rate in rates.values():
            assert isinstance(rate, float)

    def test_rates_bounded(self, device):
        """Test that all rates are in (0, 1)."""
        cd1 = ConcreteDropout(in_features=32, init_p=0.01)
        cd2 = ConcreteDropout(in_features=32, init_p=0.99)
        model = torch.nn.Sequential(cd1, cd2).to(device)
        rates = get_concrete_dropout_rates(model)

        for rate in rates.values():
            assert 0.0 < rate < 1.0

    def test_nested_model(self, device):
        """Test rate extraction from a nested model structure."""
        inner = torch.nn.Sequential(
            ConcreteDropout(in_features=32, init_p=0.3),
            torch.nn.Linear(32, 32),
        )
        outer = torch.nn.Sequential(
            inner,
            ConcreteDropout(in_features=32, init_p=0.6),
        ).to(device)

        rates = get_concrete_dropout_rates(outer)

        assert len(rates) == 2
