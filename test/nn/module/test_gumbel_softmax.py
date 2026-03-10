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

"""Tests for the gumbel_softmax function.

This module tests the Gumbel-Softmax implementation used in Transolver++
for differentiable categorical sampling.
"""

import pytest
import torch

from physicsnemo.nn import gumbel_softmax


class TestGumbelSoftmaxBasic:
    """Basic functionality tests for gumbel_softmax."""

    @pytest.mark.parametrize("batch_size", [1, 4, 16])
    @pytest.mark.parametrize("num_categories", [2, 10, 64])
    def test_output_shape(self, device, batch_size, num_categories):
        """Test that output shape matches input shape."""
        logits = torch.randn(batch_size, num_categories, device=device)
        output = gumbel_softmax(logits)

        assert output.shape == logits.shape

    def test_output_sums_to_one(self, device):
        """Test that output is a valid probability distribution (sums to 1)."""
        logits = torch.randn(8, 10, device=device)
        output = gumbel_softmax(logits)

        # Each row should sum to approximately 1
        row_sums = output.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    def test_output_non_negative(self, device):
        """Test that all output values are non-negative."""
        logits = torch.randn(8, 10, device=device)
        output = gumbel_softmax(logits)

        assert torch.all(output >= 0)

    def test_output_bounded(self, device):
        """Test that output values are bounded in [0, 1]."""
        logits = torch.randn(8, 10, device=device)
        output = gumbel_softmax(logits)

        assert torch.all(output >= 0)
        assert torch.all(output <= 1)


class TestGumbelSoftmaxTemperature:
    """Tests for temperature parameter behavior."""

    def test_low_temperature_sharpens_distribution(self, device):
        """Test that lower temperature produces sharper distributions."""
        logits = torch.randn(8, 10, device=device)

        output_high_temp = gumbel_softmax(logits, tau=5.0)
        output_low_temp = gumbel_softmax(logits, tau=0.1)

        # Lower temperature should have higher max values (sharper)
        # We compare the average max across batch
        high_temp_max = output_high_temp.max(dim=-1).values.mean()
        low_temp_max = output_low_temp.max(dim=-1).values.mean()

        assert low_temp_max > high_temp_max

    def test_high_temperature_smooths_distribution(self, device):
        """Test that higher temperature produces more uniform distributions."""
        # Use fixed logits with clear preference
        logits = torch.zeros(8, 10, device=device)
        logits[:, 0] = 5.0  # Strong preference for first category

        output_low_temp = gumbel_softmax(logits, tau=0.5)
        output_high_temp = gumbel_softmax(logits, tau=10.0)

        # High temperature should have lower max (more uniform)
        low_temp_max = output_low_temp.max(dim=-1).values.mean()
        high_temp_max = output_high_temp.max(dim=-1).values.mean()

        assert high_temp_max < low_temp_max

    def test_temperature_tensor_input(self, device):
        """Test that temperature can be a tensor."""
        logits = torch.randn(8, 10, device=device)
        tau = torch.tensor(1.0, device=device)

        output = gumbel_softmax(logits, tau=tau)

        assert output.shape == logits.shape
        assert torch.allclose(
            output.sum(dim=-1), torch.ones(8, device=device), atol=1e-5
        )

    def test_per_element_temperature(self, device):
        """Test that temperature can vary per element."""
        logits = torch.randn(8, 10, device=device)
        # Different temperature for each batch element
        tau = torch.linspace(0.5, 2.0, 8, device=device).unsqueeze(-1)

        output = gumbel_softmax(logits, tau=tau)

        assert output.shape == logits.shape
        assert torch.allclose(
            output.sum(dim=-1), torch.ones(8, device=device), atol=1e-5
        )


class TestGumbelSoftmaxStochasticity:
    """Tests for stochastic behavior of gumbel_softmax."""

    def test_outputs_differ_across_calls(self, device):
        """Test that outputs are stochastic (differ across calls)."""
        logits = torch.randn(8, 10, device=device)

        output1 = gumbel_softmax(logits, tau=1.0)
        output2 = gumbel_softmax(logits, tau=1.0)

        # Outputs should differ due to Gumbel noise
        assert not torch.allclose(output1, output2)

    def test_deterministic_with_seed(self, device):
        """Test that outputs are reproducible with same seed."""
        logits = torch.randn(8, 10, device=device)

        torch.manual_seed(42)
        output1 = gumbel_softmax(logits, tau=1.0)

        torch.manual_seed(42)
        output2 = gumbel_softmax(logits, tau=1.0)

        assert torch.allclose(output1, output2)


class TestGumbelSoftmaxGradients:
    """Tests for gradient flow through gumbel_softmax."""

    def test_gradient_flow(self, device):
        """Test that gradients flow through the function."""
        logits = torch.randn(8, 10, device=device, requires_grad=True)
        output = gumbel_softmax(logits, tau=1.0)
        loss = output.sum()
        loss.backward()

        assert logits.grad is not None
        assert logits.grad.shape == logits.shape

    def test_gradient_not_nan(self, device):
        """Test that gradients are not NaN."""
        logits = torch.randn(8, 10, device=device, requires_grad=True)
        output = gumbel_softmax(logits, tau=1.0)
        loss = output.sum()
        loss.backward()

        assert logits.grad is not None
        assert not torch.any(torch.isnan(logits.grad))

    def test_gradient_with_varying_temperature(self, device):
        """Test gradient flow with varying temperature."""
        logits = torch.randn(8, 10, device=device, requires_grad=True)
        tau = torch.tensor(0.5, device=device, requires_grad=True)

        output = gumbel_softmax(logits, tau=tau)
        loss = output.sum()
        loss.backward()

        assert logits.grad is not None
        # Note: tau gradient may not flow depending on implementation


class TestGumbelSoftmaxEdgeCases:
    """Tests for edge cases and special inputs."""

    def test_very_large_logits(self, device):
        """Test handling of very large logit values."""
        logits = torch.randn(8, 10, device=device) * 100
        output = gumbel_softmax(logits, tau=1.0)

        # Should still be valid probabilities
        assert torch.all(torch.isfinite(output))
        assert torch.allclose(
            output.sum(dim=-1), torch.ones(8, device=device), atol=1e-4
        )

    def test_very_small_logits(self, device):
        """Test handling of very small logit values."""
        logits = torch.randn(8, 10, device=device) * 1e-6
        output = gumbel_softmax(logits, tau=1.0)

        assert torch.all(torch.isfinite(output))
        assert torch.allclose(
            output.sum(dim=-1), torch.ones(8, device=device), atol=1e-5
        )

    def test_uniform_logits(self, device):
        """Test with uniform logits (all same value)."""
        logits = torch.ones(8, 10, device=device)
        output = gumbel_softmax(logits, tau=1.0)

        # Should still produce valid distribution
        assert torch.all(torch.isfinite(output))
        assert torch.allclose(
            output.sum(dim=-1), torch.ones(8, device=device), atol=1e-5
        )

    def test_binary_classification(self, device):
        """Test with only 2 categories (binary case)."""
        logits = torch.randn(8, 2, device=device)
        output = gumbel_softmax(logits, tau=1.0)

        assert output.shape == (8, 2)
        assert torch.allclose(
            output.sum(dim=-1), torch.ones(8, device=device), atol=1e-5
        )

    def test_single_category(self, device):
        """Test with single category edge case."""
        logits = torch.randn(8, 1, device=device)
        output = gumbel_softmax(logits, tau=1.0)

        # Single category should always be 1.0
        assert output.shape == (8, 1)
        assert torch.allclose(output, torch.ones_like(output), atol=1e-5)


class TestGumbelSoftmaxHigherDimensions:
    """Tests for higher-dimensional inputs."""

    def test_3d_input(self, device):
        """Test with 3D input tensor."""
        logits = torch.randn(4, 8, 10, device=device)
        output = gumbel_softmax(logits, tau=1.0)

        assert output.shape == logits.shape
        # Last dimension should sum to 1
        assert torch.allclose(
            output.sum(dim=-1), torch.ones(4, 8, device=device), atol=1e-5
        )

    def test_4d_input(self, device):
        """Test with 4D input tensor (e.g., batch, heads, tokens, categories)."""
        logits = torch.randn(2, 4, 100, 16, device=device)
        output = gumbel_softmax(logits, tau=1.0)

        assert output.shape == logits.shape
        # Last dimension should sum to 1
        assert torch.allclose(
            output.sum(dim=-1), torch.ones(2, 4, 100, device=device), atol=1e-5
        )

    def test_broadcast_temperature_with_higher_dims(self, device):
        """Test temperature broadcasting with higher-dimensional inputs."""
        logits = torch.randn(2, 4, 8, 10, device=device)
        # Temperature that broadcasts
        tau = torch.ones(2, 1, 1, 1, device=device) * 0.5

        output = gumbel_softmax(logits, tau=tau)

        assert output.shape == logits.shape
        assert torch.allclose(
            output.sum(dim=-1), torch.ones(2, 4, 8, device=device), atol=1e-5
        )


class TestGumbelSoftmaxNumericalStability:
    """Tests for numerical stability."""

    def test_no_nan_with_extreme_values(self, device):
        """Test that no NaN values are produced with extreme inputs."""
        # Very negative logits
        logits_neg = torch.full((8, 10), -100.0, device=device)
        output_neg = gumbel_softmax(logits_neg, tau=1.0)
        assert not torch.any(torch.isnan(output_neg))

        # Very positive logits
        logits_pos = torch.full((8, 10), 100.0, device=device)
        output_pos = gumbel_softmax(logits_pos, tau=1.0)
        assert not torch.any(torch.isnan(output_pos))

    def test_no_inf_values(self, device):
        """Test that no infinite values are produced."""
        logits = torch.randn(8, 10, device=device) * 50
        output = gumbel_softmax(logits, tau=0.1)

        assert not torch.any(torch.isinf(output))

    def test_very_small_temperature(self, device):
        """Test behavior with very small temperature."""
        logits = torch.randn(8, 10, device=device)
        output = gumbel_softmax(logits, tau=0.01)

        # Should still be valid (though very peaked)
        assert torch.all(torch.isfinite(output))
        assert torch.all(output >= 0)


# =============================================================================
# GumbelSoftmax Module Tests
# =============================================================================


class TestGumbelSoftmaxModule:
    """Tests for the GumbelSoftmax nn.Module class."""

    def test_basic_forward(self, device):
        """Test basic forward pass of GumbelSoftmax module."""
        from physicsnemo.nn import GumbelSoftmax

        gs = GumbelSoftmax(tau=1.0).to(device)
        logits = torch.randn(8, 10, device=device)
        output = gs(logits)

        assert output.shape == logits.shape
        assert torch.allclose(
            output.sum(dim=-1), torch.ones(8, device=device), atol=1e-5
        )

    def test_custom_temperature(self, device):
        """Test GumbelSoftmax with custom temperature."""
        from physicsnemo.nn import GumbelSoftmax

        gs = GumbelSoftmax(tau=0.5).to(device)
        logits = torch.randn(8, 10, device=device)
        output = gs(logits)

        assert output.shape == logits.shape
        assert torch.all(output >= 0)
        assert torch.all(output <= 1)

    def test_learnable_temperature(self, device):
        """Test GumbelSoftmax with learnable temperature parameter."""
        from physicsnemo.nn import GumbelSoftmax

        gs = GumbelSoftmax(tau=1.0, learnable=True).to(device)

        # Temperature should be a learnable parameter
        assert isinstance(gs.tau, torch.nn.Parameter)
        assert gs.tau.requires_grad

        logits = torch.randn(8, 10, device=device)
        output = gs(logits)
        loss = output.sum()
        loss.backward()

        # Gradient should flow to temperature
        assert gs.tau.grad is not None

    def test_non_learnable_temperature(self, device):
        """Test GumbelSoftmax with non-learnable temperature (buffer)."""
        from physicsnemo.nn import GumbelSoftmax

        gs = GumbelSoftmax(tau=1.0, learnable=False).to(device)

        # Temperature should be a buffer, not a parameter
        assert not isinstance(gs.tau, torch.nn.Parameter)
        assert "tau" in dict(gs.named_buffers())

    def test_module_in_sequential(self, device):
        """Test GumbelSoftmax can be used in nn.Sequential."""
        from physicsnemo.nn import GumbelSoftmax

        model = torch.nn.Sequential(
            torch.nn.Linear(10, 20),
            GumbelSoftmax(tau=0.5),
        ).to(device)

        x = torch.randn(4, 10, device=device)
        output = model(x)

        assert output.shape == (4, 20)
        assert torch.allclose(
            output.sum(dim=-1), torch.ones(4, device=device), atol=1e-5
        )

    def test_module_state_dict(self, device):
        """Test that temperature is saved in state dict."""
        from physicsnemo.nn import GumbelSoftmax

        gs = GumbelSoftmax(tau=0.75, learnable=False).to(device)
        state_dict = gs.state_dict()

        assert "tau" in state_dict
        assert state_dict["tau"].item() == 0.75

    def test_module_load_state_dict(self, device):
        """Test loading state dict with different temperature."""
        from physicsnemo.nn import GumbelSoftmax

        gs1 = GumbelSoftmax(tau=0.5).to(device)
        gs2 = GumbelSoftmax(tau=1.0).to(device)

        gs2.load_state_dict(gs1.state_dict())

        assert gs2.tau.item() == 0.5

    def test_gradient_flow_through_module(self, device):
        """Test gradient flow through the module."""
        from physicsnemo.nn import GumbelSoftmax

        gs = GumbelSoftmax(tau=1.0).to(device)
        logits = torch.randn(8, 10, device=device, requires_grad=True)

        output = gs(logits)
        loss = output.sum()
        loss.backward()

        assert logits.grad is not None
        assert not torch.any(torch.isnan(logits.grad))

    def test_eval_mode(self, device):
        """Test module behavior in eval mode (should still be stochastic)."""
        from physicsnemo.nn import GumbelSoftmax

        gs = GumbelSoftmax(tau=1.0).to(device)
        gs.eval()

        logits = torch.randn(8, 10, device=device)

        # Even in eval mode, outputs should differ due to Gumbel noise
        output1 = gs(logits)
        output2 = gs(logits)

        assert not torch.allclose(output1, output2)

    def test_equivalence_with_function(self, device):
        """Test that module produces equivalent results to function."""
        from physicsnemo.nn import GumbelSoftmax

        tau = 0.5
        gs = GumbelSoftmax(tau=tau).to(device)
        logits = torch.randn(8, 10, device=device)

        # With same seed, should produce same results
        torch.manual_seed(42)
        output_module = gs(logits)

        torch.manual_seed(42)
        output_function = gumbel_softmax(logits, tau=tau)

        assert torch.allclose(output_module, output_function)
