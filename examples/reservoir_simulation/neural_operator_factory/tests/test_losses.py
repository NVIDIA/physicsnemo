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

"""Comprehensive unit tests for loss functions."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.losses import SimpleRelativeL2Loss, UnifiedLoss, get_loss_function

# ---------------------------------------------------------------------------
# Helpers: build inputs with grid-width channels (NOF convention)
# ---------------------------------------------------------------------------


def _make_2d_inputs(B, H, W, T, C, dx=None, dy=None):
    """Create (B, H, W, T, C) inputs with grid widths in last 3 channels."""
    inputs = torch.randn(B, H, W, T, C)
    if dx is None:
        dx = torch.ones(W)
    if dy is None:
        dy = torch.ones(H)
    dt = torch.linspace(0, 30, T)
    inputs[..., -3] = dx.view(1, 1, W, 1).expand(B, H, W, T)
    inputs[..., -2] = dy.view(1, H, 1, 1).expand(B, H, W, T)
    inputs[..., -1] = dt.view(1, 1, 1, T).expand(B, H, W, T)
    return inputs


def _make_3d_inputs(B, X, Y, Z, T, C, dx=None, dy=None, dz=None):
    """Create (B, X, Y, Z, T, C) inputs with grid widths in last 4 channels."""
    inputs = torch.randn(B, X, Y, Z, T, C)
    if dx is None:
        dx = torch.ones(X)
    if dy is None:
        dy = torch.ones(Y)
    if dz is None:
        dz = torch.ones(Z)
    dt = torch.linspace(0, 30, T)
    inputs[..., -4] = dx.view(1, X, 1, 1, 1).expand(B, X, Y, Z, T)
    inputs[..., -3] = dy.view(1, 1, Y, 1, 1).expand(B, X, Y, Z, T)
    inputs[..., -2] = dz.view(1, 1, 1, Z, 1).expand(B, X, Y, Z, T)
    inputs[..., -1] = dt.view(1, 1, 1, 1, T).expand(B, X, Y, Z, T)
    return inputs


# ===================================================================
# SimpleRelativeL2Loss
# ===================================================================


class TestSimpleRelativeL2Loss:
    """Tests for SimpleRelativeL2Loss."""

    def test_zero_for_identical(self):
        """Verify relative L2 loss is zero for identical pred and target."""
        target = torch.randn(2, 8, 16, 4)
        loss = SimpleRelativeL2Loss()(target.clone(), target)
        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)

    def test_positive_for_different(self):
        """Verify relative L2 loss is positive for different pred and target."""
        target = torch.randn(2, 8, 16, 4)
        pred = target + 0.1 * torch.randn_like(target)
        assert SimpleRelativeL2Loss()(pred, target) > 0

    def test_scale_invariance(self):
        """Verify relative L2 loss is scale-invariant."""
        target = torch.randn(2, 8, 16, 4) + 2
        pred = target + 0.1 * torch.randn_like(target)
        fn = SimpleRelativeL2Loss()
        assert torch.isclose(fn(pred, target), fn(pred * 5, target * 5), rtol=1e-4)

    def test_epsilon_prevents_nan(self):
        """Verify epsilon prevents NaN when target is all zeros."""
        target = torch.zeros(2, 4, 4, 2)
        pred = torch.ones(2, 4, 4, 2)
        loss = SimpleRelativeL2Loss(eps=1e-8)(pred, target)
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)


# ===================================================================
# UnifiedLoss — Data Losses
# ===================================================================


class TestDataLosses:
    """Tests for DataLosses."""

    @pytest.mark.parametrize("loss_type", ["mse", "l1", "relative_l2", "huber"])
    def test_all_types_run(self, loss_type):
        """Verify each loss type runs and produces a positive, finite loss."""
        pred = torch.randn(2, 8, 16, 4)
        target = torch.randn(2, 8, 16, 4)
        fn = UnifiedLoss(types=[loss_type], weights=[1.0])
        loss = fn(pred, target)
        assert loss > 0
        assert not torch.isnan(loss)

    def test_mse_value(self):
        """Verify MSE loss computes correct value."""
        target = torch.zeros(1, 4, 4, 2)
        pred = torch.ones(1, 4, 4, 2)
        loss = UnifiedLoss(types=["mse"])(pred, target)
        assert torch.isclose(loss, torch.tensor(1.0))

    def test_l1_value(self):
        """Verify L1 loss computes correct value."""
        target = torch.zeros(1, 4, 4, 2)
        pred = torch.full_like(target, 2.0)
        loss = UnifiedLoss(types=["l1"])(pred, target)
        assert torch.isclose(loss, torch.tensor(2.0))

    def test_huber_equals_mse_for_small_errors(self):
        """Verify Huber loss approximates MSE for small errors."""
        target = torch.randn(2, 8, 8, 4)
        pred = target + 0.01 * torch.randn_like(target)
        mse_loss = UnifiedLoss(types=["mse"])(pred, target)
        huber_loss = UnifiedLoss(types=["huber"], huber_delta=1.0)(pred, target)
        assert torch.isclose(mse_loss, huber_loss * 2, rtol=0.1)

    def test_relative_l2_epsilon_zero_target(self):
        """Verify relative L2 with epsilon handles zero target without NaN."""
        target = torch.zeros(2, 4, 4, 2)
        pred = torch.ones(2, 4, 4, 2)
        loss = UnifiedLoss(types=["relative_l2"], eps=1e-6)(pred, target)
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_multiple_losses_combined(self):
        """Verify combined losses with zero weight reduce to single loss."""
        pred = torch.randn(2, 8, 8, 4)
        target = torch.randn(2, 8, 8, 4)
        fn_single = UnifiedLoss(types=["mse"], weights=[1.0])
        fn_multi = UnifiedLoss(types=["mse", "l1"], weights=[1.0, 0.0])
        assert torch.isclose(fn_single(pred, target), fn_multi(pred, target))

    def test_invalid_type_raises(self):
        """Verify invalid loss type raises ValueError."""
        with pytest.raises(ValueError, match="Loss type must be"):
            UnifiedLoss(types=["invalid"])

    def test_invalid_reduction_raises(self):
        """Verify invalid reduction mode raises ValueError."""
        with pytest.raises(ValueError, match="reduction"):
            UnifiedLoss(reduction="invalid")

    def test_mismatched_lengths_raises(self):
        """Verify mismatched types/weights lengths raises ValueError."""
        with pytest.raises(ValueError, match="same length"):
            UnifiedLoss(types=["mse", "l1"], weights=[1.0])

    def test_gradient_flow(self):
        """Verify gradients flow through MSE loss to prediction tensor."""
        target = torch.randn(2, 8, 8, 4)
        pred = torch.randn(2, 8, 8, 4, requires_grad=True)
        loss = UnifiedLoss(types=["mse"])(pred, target)
        loss.backward()
        assert pred.grad is not None


# ===================================================================
# UnifiedLoss — Masking
# ===================================================================


class TestMasking:
    """Tests for Masking."""

    def test_mse_mask_only_active(self):
        """MSE should average only over active cells."""
        B, H, W, T = 1, 4, 4, 2
        target = torch.zeros(B, H, W, T)
        pred = torch.ones(B, H, W, T)
        mask = torch.zeros(H, W, dtype=torch.bool)
        mask[0, 0] = True  # only one cell active

        fn = UnifiedLoss(types=["mse"])
        loss = fn(pred, target, spatial_mask=mask)
        assert torch.isclose(loss, torch.tensor(1.0))

    def test_l1_mask_only_active(self):
        """Verify L1 loss averages only over active masked cells."""
        B, H, W, T = 1, 4, 4, 2
        target = torch.zeros(B, H, W, T)
        pred = torch.full((B, H, W, T), 3.0)
        mask = torch.zeros(H, W, dtype=torch.bool)
        mask[0, 0] = True

        loss = UnifiedLoss(types=["l1"])(pred, target, spatial_mask=mask)
        assert torch.isclose(loss, torch.tensor(3.0))

    def test_mask_zeros_dont_leak_2d(self):
        """Error only in masked-out region => loss = 0."""
        B, H, W, T = 1, 8, 8, 4
        target = torch.randn(B, H, W, T)
        pred = target.clone()
        pred[:, :4, :, :] += 10.0

        mask = torch.zeros(H, W, dtype=torch.bool)
        mask[4:, :] = True

        loss = UnifiedLoss(types=["mse"])(pred, target, spatial_mask=mask)
        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)

    def test_mask_3d(self):
        """Verify 3D spatial masking excludes inactive cells from loss."""
        B, X, Y, Z, T = 1, 4, 4, 2, 3
        target = torch.randn(B, X, Y, Z, T)
        pred = target.clone()
        pred[:, :2, :, :, :] += 10.0

        mask = torch.zeros(X, Y, Z, dtype=torch.bool)
        mask[2:, :, :] = True

        loss = UnifiedLoss(types=["mse"])(pred, target, spatial_mask=mask)
        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


# ===================================================================
# UnifiedLoss — Derivative Loss (2D)
# ===================================================================


class TestDerivativeLoss2D:
    """Tests for DerivativeLoss2D."""

    def test_derivative_dx_runs(self):
        """Verify dx derivative loss runs and produces a positive value."""
        B, H, W, T, C = 2, 8, 16, 4, 12
        inputs = _make_2d_inputs(B, H, W, T, C)
        pred = torch.randn(B, H, W, T)
        target = torch.randn(B, H, W, T)
        fn = UnifiedLoss(
            types=["mse"],
            derivative_config={"enabled": True, "weight": 0.5, "dims": ["dx"]},
        )
        loss = fn(pred, target, inputs)
        assert loss > 0
        assert not torch.isnan(loss)

    def test_derivative_dy_runs(self):
        """Verify dy derivative loss runs and produces a positive value."""
        B, H, W, T, C = 2, 8, 16, 4, 12
        inputs = _make_2d_inputs(B, H, W, T, C)
        pred = torch.randn(B, H, W, T)
        target = torch.randn(B, H, W, T)
        fn = UnifiedLoss(
            types=["mse"],
            derivative_config={"enabled": True, "weight": 0.5, "dims": ["dy"]},
        )
        loss = fn(pred, target, inputs)
        assert loss > 0

    def test_derivative_both_dims(self):
        """Verify derivative loss with both dx and dy dimensions."""
        B, H, W, T, C = 2, 8, 16, 4, 12
        inputs = _make_2d_inputs(B, H, W, T, C)
        pred = torch.randn(B, H, W, T)
        target = torch.randn(B, H, W, T)
        fn = UnifiedLoss(
            types=["mse"],
            derivative_config={"enabled": True, "weight": 0.5, "dims": ["dx", "dy"]},
        )
        loss = fn(pred, target, inputs)
        assert loss > 0

    def test_derivative_zero_for_identical(self):
        """Verify derivative loss is zero when pred equals target."""
        B, H, W, T, C = 1, 8, 16, 4, 12
        inputs = _make_2d_inputs(B, H, W, T, C)
        target = torch.randn(B, H, W, T)
        pred = target.clone()
        fn = UnifiedLoss(
            types=["mse"],
            derivative_config={"enabled": True, "weight": 1.0, "dims": ["dx"]},
        )
        loss = fn(pred, target, inputs)
        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)

    def test_derivative_gradient_flow(self):
        """Verify gradients flow through derivative loss to prediction."""
        B, H, W, T, C = 2, 8, 16, 4, 12
        inputs = _make_2d_inputs(B, H, W, T, C)
        target = torch.randn(B, H, W, T)
        pred = torch.randn(B, H, W, T, requires_grad=True)
        fn = UnifiedLoss(
            types=["mse"],
            derivative_config={"enabled": True, "weight": 0.5, "dims": ["dx"]},
        )
        loss = fn(pred, target, inputs)
        loss.backward()
        assert pred.grad is not None

    def test_derivative_non_uniform_grid(self):
        """Verify derivative uses non-uniform spacing correctly."""
        B, H, W, T, C = 1, 4, 6, 2, 12
        dx = torch.tensor([10.0, 20.0, 30.0, 15.0, 25.0, 10.0])
        dy = torch.tensor([5.0, 10.0, 5.0, 10.0])
        inputs = _make_2d_inputs(B, H, W, T, C, dx=dx, dy=dy)

        target = torch.zeros(B, H, W, T)
        pred = torch.zeros(B, H, W, T)
        pred[0, :, :, 0] = torch.arange(W).float().unsqueeze(0).expand(H, W)

        fn = UnifiedLoss(
            types=["mse"],
            derivative_config={"enabled": True, "weight": 1.0, "dims": ["dx"]},
        )
        loss = fn(pred, target, inputs)
        assert loss > 0
        assert not torch.isnan(loss)

    def test_requires_inputs(self):
        """Verify derivative loss raises ValueError when inputs are missing."""
        fn = UnifiedLoss(
            types=["mse"],
            derivative_config={"enabled": True, "weight": 0.5, "dims": ["dx"]},
        )
        with pytest.raises(ValueError, match="inputs required"):
            fn(torch.randn(2, 8, 8, 4), torch.randn(2, 8, 8, 4))


# ===================================================================
# UnifiedLoss — Derivative Loss (3D)
# ===================================================================


class TestDerivativeLoss3D:
    """Tests for DerivativeLoss3D."""

    def test_derivative_all_3d_dims(self):
        """Verify derivative loss with all three 3D spatial dimensions."""
        B, X, Y, Z, T, C = 2, 6, 8, 4, 3, 11
        inputs = _make_3d_inputs(B, X, Y, Z, T, C)
        pred = torch.randn(B, X, Y, Z, T)
        target = torch.randn(B, X, Y, Z, T)
        fn = UnifiedLoss(
            types=["mse"],
            derivative_config={
                "enabled": True,
                "weight": 0.5,
                "dims": ["dx", "dy", "dz"],
            },
        )
        loss = fn(pred, target, inputs)
        assert loss > 0
        assert not torch.isnan(loss)

    def test_derivative_single_dim_3d(self):
        """Verify each individual 3D derivative dimension works."""
        B, X, Y, Z, T, C = 1, 6, 8, 4, 2, 11
        inputs = _make_3d_inputs(B, X, Y, Z, T, C)
        pred = torch.randn(B, X, Y, Z, T)
        target = torch.randn(B, X, Y, Z, T)
        for dim in ["dx", "dy", "dz"]:
            fn = UnifiedLoss(
                types=["mse"],
                derivative_config={"enabled": True, "weight": 0.5, "dims": [dim]},
            )
            loss = fn(pred, target, inputs)
            assert loss > 0, f"Failed for {dim}"

    def test_invalid_dim_for_2d_raises(self):
        """Verify dz derivative on 2D data raises ValueError."""
        B, H, W, T, C = 1, 8, 16, 4, 12
        inputs = _make_2d_inputs(B, H, W, T, C)
        fn = UnifiedLoss(
            types=["mse"],
            derivative_config={"enabled": True, "weight": 0.5, "dims": ["dz"]},
        )
        with pytest.raises(ValueError, match="not valid for 2D"):
            fn(torch.randn(B, H, W, T), torch.randn(B, H, W, T), inputs)


class TestDerivativeWithMask:
    """Tests for derivative loss with spatial masking (Norne-like sparse grids)."""

    def test_derivative_with_sparse_mask_no_nan(self):
        """Derivative loss must not produce NaN on grids with many inactive cells."""
        B, X, Y, Z, T, C = 1, 10, 12, 4, 3, 11
        dx = torch.tensor([10.0, 20.0, 15.0, 10.0, 25.0, 30.0, 10.0, 20.0, 15.0, 10.0])
        dy = torch.ones(Y) * 5.0
        dz = torch.ones(Z) * 8.0
        inputs = _make_3d_inputs(B, X, Y, Z, T, C, dx=dx, dy=dy, dz=dz)
        target = torch.randn(B, X, Y, Z, T)
        pred = target + 0.1 * torch.randn_like(target)

        # Sparse mask: only 30% active (similar to Norne's 39%)
        mask = torch.zeros(X, Y, Z, dtype=torch.bool)
        mask[2:5, 3:8, :] = True

        fn = UnifiedLoss(
            types=["relative_l2"],
            derivative_config={"enabled": True, "weight": 0.5, "dims": ["dx"]},
        )
        loss = fn(pred, target, inputs, spatial_mask=mask)
        assert not torch.isnan(loss), "Loss is NaN with sparse mask"
        assert not torch.isinf(loss), "Loss is Inf with sparse mask"

    def test_derivative_all_dims_with_mask(self):
        """All 3D derivative directions work with masking."""
        B, X, Y, Z, T, C = 1, 8, 10, 4, 2, 11
        inputs = _make_3d_inputs(B, X, Y, Z, T, C)
        target = torch.randn(B, X, Y, Z, T)
        pred = target + 0.05 * torch.randn_like(target)

        mask = torch.zeros(X, Y, Z, dtype=torch.bool)
        mask[1:6, 2:8, 1:3] = True

        fn = UnifiedLoss(
            types=["relative_l2"],
            derivative_config={
                "enabled": True,
                "weight": 0.5,
                "dims": ["dx", "dy", "dz"],
            },
        )
        loss = fn(pred, target, inputs, spatial_mask=mask)
        assert not torch.isnan(loss)

    def test_derivative_auto_detects_inactive_no_mask(self):
        """Derivative loss auto-detects inactive cells even without spatial_mask."""
        B, X, Y, Z, T, C = 1, 8, 10, 4, 3, 11
        inputs = _make_3d_inputs(B, X, Y, Z, T, C)
        target = torch.randn(B, X, Y, Z, T)
        # Make some cells inactive (zero across all timesteps)
        target[:, :3, :, :, :] = 0.0
        pred = target + 0.1 * torch.randn_like(target)
        pred[:, :3, :, :, :] = 0.0  # pred also zero there

        fn = UnifiedLoss(
            types=["relative_l2"],
            derivative_config={"enabled": True, "weight": 0.5, "dims": ["dx"]},
        )
        # No spatial_mask passed — auto-detection should handle it
        loss = fn(pred, target, inputs, spatial_mask=None)
        assert not torch.isnan(loss), "NaN with auto-detected inactive cells"
        assert not torch.isinf(loss)

    def test_derivative_all_active_no_mask(self):
        """When all cells are active and no mask, derivative works normally."""
        B, H, W, T, C = 1, 8, 16, 4, 12
        inputs = _make_2d_inputs(B, H, W, T, C)
        target = torch.randn(B, H, W, T) + 1.0  # no zeros
        pred = target + 0.05 * torch.randn_like(target)

        fn = UnifiedLoss(
            types=["mse"],
            derivative_config={"enabled": True, "weight": 0.5, "dims": ["dx", "dy"]},
        )
        loss = fn(pred, target, inputs, spatial_mask=None)
        assert not torch.isnan(loss)
        assert loss > 0

    def test_derivative_mask_excludes_boundary_artifacts(self):
        """Error at inactive cells should not affect derivative loss."""
        B, H, W, T, C = 1, 8, 12, 2, 12
        inputs = _make_2d_inputs(B, H, W, T, C)
        target = torch.randn(B, H, W, T)
        pred = target.clone()
        # Large error only in masked-out region
        pred[:, :3, :, :] += 100.0

        mask = torch.zeros(H, W, dtype=torch.bool)
        mask[4:, :] = True  # only bottom half active

        fn = UnifiedLoss(
            types=["mse"],
            derivative_config={"enabled": True, "weight": 1.0, "dims": ["dx", "dy"]},
        )
        loss = fn(pred, target, inputs, spatial_mask=mask)
        # Data loss should be ~0, derivative loss should be ~0
        # because pred=target in the active region
        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-4), (
            f"Expected ~0, got {loss.item()}"
        )


# ===================================================================
# get_loss_function factory
# ===================================================================


class TestFactory:
    """Tests for Factory."""

    def test_defaults(self):
        """Verify default factory produces UnifiedLoss with relative_l2."""
        fn = get_loss_function({})
        assert isinstance(fn, UnifiedLoss)
        assert fn.loss_types == ["relative_l2"]

    def test_with_physics(self):
        """Verify factory registers physics losses from config."""
        cfg = {
            "types": ["mse"],
            "weights": [1.0],
            "physics": {"mass_conservation": {"enabled": True, "weight": 0.5}},
        }
        fn = get_loss_function(cfg, variable="saturation")
        assert "mass_conservation" in fn._physics_losses

    def test_new_derivative_config(self):
        """Verify factory applies derivative config from loss config."""
        cfg = {
            "types": ["mse"],
            "weights": [1.0],
            "derivative": {"enabled": True, "weight": 0.3, "dims": ["dx", "dy"]},
        }
        fn = get_loss_function(cfg)
        assert fn._deriv_enabled is True
        assert fn._deriv_weight == 0.3
        assert fn._deriv_dims == ["dx", "dy"]

    def test_pressure_warning(self):
        """Verify mass conservation on pressure variable emits a warning."""
        cfg = {
            "types": ["mse"],
            "weights": [1.0],
            "physics": {"mass_conservation": {"enabled": True, "weight": 1.0}},
        }
        with pytest.warns(UserWarning, match="pressure"):
            get_loss_function(cfg, variable="pressure")


# ===================================================================
# AR window compatibility
# ===================================================================


class TestARCompatibility:
    """Tests for ARCompatibility."""

    def test_single_timestep_2d(self):
        """Verify loss with single-timestep 2D AR window."""
        B, H, W, C = 2, 8, 16, 12
        pred = torch.randn(B, H, W, 1)
        target = torch.randn(B, H, W, 1)
        inputs = _make_2d_inputs(B, H, W, 1, C)
        fn = UnifiedLoss(
            types=["relative_l2"],
            derivative_config={"enabled": True, "weight": 0.5, "dims": ["dx"]},
        )
        loss = fn(pred, target, inputs)
        assert not torch.isnan(loss)

    def test_small_window_3d(self):
        """Verify loss with small 3D AR window (K=3)."""
        B, X, Y, Z, C = 2, 6, 8, 4, 11
        K = 3
        pred = torch.randn(B, X, Y, Z, K)
        target = torch.randn(B, X, Y, Z, K)
        inputs = _make_3d_inputs(B, X, Y, Z, K, C)
        fn = UnifiedLoss(types=["mse"])
        loss = fn(pred, target, inputs)
        assert not torch.isnan(loss)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# ===================================================================
# Derivative loss with all metric types
# ===================================================================


class TestDerivativeAllMetrics:
    """Verify derivative loss works with every loss metric + masking."""

    @pytest.mark.parametrize("metric", ["mse", "l1", "relative_l2", "huber"])
    def test_derivative_metric_2d(self, metric):
        """All metrics produce finite loss for 2D derivative."""
        B, H, W, T, C = 1, 8, 16, 4, 12
        inputs = _make_2d_inputs(B, H, W, T, C)
        target = torch.randn(B, H, W, T) + 1.0
        pred = target + 0.05 * torch.randn_like(target)

        fn = UnifiedLoss(
            types=["mse"],
            derivative_config={
                "enabled": True,
                "weight": 0.5,
                "dims": ["dx"],
                "metric": metric,
            },
        )
        loss = fn(pred, target, inputs)
        assert not torch.isnan(loss), f"NaN with metric={metric}"
        assert loss > 0

    @pytest.mark.parametrize("metric", ["mse", "l1", "relative_l2", "huber"])
    def test_derivative_metric_with_mask(self, metric):
        """All metrics work with derivative + sparse mask."""
        B, X, Y, Z, T, C = 1, 8, 10, 4, 3, 11
        inputs = _make_3d_inputs(B, X, Y, Z, T, C)
        target = torch.randn(B, X, Y, Z, T) + 1.0
        pred = target + 0.05 * torch.randn_like(target)

        mask = torch.zeros(X, Y, Z, dtype=torch.bool)
        mask[2:6, 2:8, 1:3] = True

        fn = UnifiedLoss(
            types=["mse"],
            derivative_config={
                "enabled": True,
                "weight": 0.5,
                "dims": ["dx"],
                "metric": metric,
            },
        )
        loss = fn(pred, target, inputs, spatial_mask=mask)
        assert not torch.isnan(loss), f"NaN with metric={metric} + mask"


# ===================================================================
# Edge cases
# ===================================================================


class TestLossEdgeCases:
    """Edge cases: minimum grids, single batch, small AR windows."""

    def test_minimum_grid_for_derivative_2d(self):
        """3 cells along derivative axis = minimum for central difference."""
        B, H, W, T, C = 1, 3, 3, 2, 12
        inputs = _make_2d_inputs(B, H, W, T, C)
        target = torch.randn(B, H, W, T) + 1.0
        pred = target + 0.1

        fn = UnifiedLoss(
            types=["mse"],
            derivative_config={"enabled": True, "weight": 0.5, "dims": ["dx", "dy"]},
        )
        loss = fn(pred, target, inputs)
        assert not torch.isnan(loss)
        assert loss > 0

    def test_minimum_grid_for_derivative_3d(self):
        """3x3x3 grid with all 3 derivative dims."""
        B, X, Y, Z, T, C = 1, 3, 3, 3, 2, 11
        inputs = _make_3d_inputs(B, X, Y, Z, T, C)
        target = torch.randn(B, X, Y, Z, T) + 1.0
        pred = target + 0.1

        fn = UnifiedLoss(
            types=["mse"],
            derivative_config={
                "enabled": True,
                "weight": 0.5,
                "dims": ["dx", "dy", "dz"],
            },
        )
        loss = fn(pred, target, inputs)
        assert not torch.isnan(loss)

    def test_single_batch(self):
        """B=1 with all loss components."""
        B, H, W, T, C = 1, 8, 8, 4, 12
        inputs = _make_2d_inputs(B, H, W, T, C)
        target = torch.randn(B, H, W, T) + 1.0
        pred = target + 0.05 * torch.randn_like(target)

        from training.physics_losses import MassConservationLoss

        fn = UnifiedLoss(
            types=["relative_l2"],
            derivative_config={"enabled": True, "weight": 0.3, "dims": ["dx"]},
            physics_losses={"mc": (MassConservationLoss(), 0.5)},
        )
        loss = fn(pred, target, inputs)
        assert not torch.isnan(loss)

    def test_k_equals_1_ar_window(self):
        """K=1 (single output timestep) with derivative and physics."""
        B, X, Y, Z, C = 1, 6, 8, 4, 11
        inputs = _make_3d_inputs(B, X, Y, Z, 1, C)
        target = torch.randn(B, X, Y, Z, 1) + 1.0
        pred = target + 0.1

        from training.physics_losses import MassConservationLoss

        fn = UnifiedLoss(
            types=["mse"],
            derivative_config={"enabled": True, "weight": 0.5, "dims": ["dx"]},
            physics_losses={"mc": (MassConservationLoss(), 1.0)},
        )
        loss = fn(pred, target, inputs)
        assert not torch.isnan(loss)

    def test_norne_regression(self):
        """Regression test: Norne-like grid (39% active, normalized widths)
        with relative_l2 derivative should NOT produce NaN."""
        B, X, Y, Z, K, C = 2, 46, 112, 22, 3, 11
        pred = torch.randn(B, X, Y, Z, K)
        target = torch.randn(B, X, Y, Z, K)
        inputs = torch.zeros(B, X, Y, Z, 1, C)

        mask = torch.zeros(X, Y, Z, dtype=torch.bool)
        mask[5:35, 10:100, 2:18] = True

        gx = torch.linspace(0, 1, X).view(X, 1, 1).expand(X, Y, Z) * mask.float()
        gy = torch.linspace(0, 1, Y).view(1, Y, 1).expand(X, Y, Z) * mask.float()
        gz = torch.linspace(0, 1, Z).view(1, 1, Z).expand(X, Y, Z) * mask.float()
        inputs[..., 0, -4] = gx.unsqueeze(0).expand(B, -1, -1, -1)
        inputs[..., 0, -3] = gy.unsqueeze(0).expand(B, -1, -1, -1)
        inputs[..., 0, -2] = gz.unsqueeze(0).expand(B, -1, -1, -1)

        pred[:, ~mask, :] = 0.0
        target[:, ~mask, :] = 0.0

        fn = UnifiedLoss(
            types=["relative_l2"],
            derivative_config={"enabled": True, "weight": 0.5, "dims": ["dx"]},
        )
        loss = fn(pred, target, inputs, spatial_mask=mask)
        assert not torch.isnan(loss), "Norne regression: NaN detected"
        assert not torch.isinf(loss), "Norne regression: Inf detected"
