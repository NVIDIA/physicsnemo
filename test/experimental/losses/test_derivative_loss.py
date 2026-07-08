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

"""Tests for the generic spatial-derivative regularization loss."""

import functools

import pytest
import torch
import torch.nn.functional as F

from physicsnemo.experimental.losses import (
    SpatialDerivativeLoss,
    cell_centre_distance,
    central_difference,
)
from physicsnemo.metrics.general.mse import mse


# Small, self-contained comparison metrics used to exercise dependency injection
# (the loss accepts any ``metric(pred, target, dim=...)`` callable).
def _l1(pred, target, dim=None):
    return torch.mean(torch.abs(pred - target), dim=dim)


def _relative_l2(pred, target, dim=None, eps=1e-8):
    diff = torch.linalg.vector_norm(pred - target, dim=dim)
    denom = torch.linalg.vector_norm(target, dim=dim)
    return diff / (denom + eps)


def _huber(pred, target, delta=1.0, dim=None):
    return torch.mean(
        F.huber_loss(pred, target, reduction="none", delta=delta), dim=dim
    )


METRICS = {
    "mse": mse,
    "l1": _l1,
    "relative_l2": _relative_l2,
    "huber": _huber,
}


# ---------------------------------------------------------------------------
# Finite-difference helpers
# ---------------------------------------------------------------------------


class TestCellCentreDistance:
    def test_uniform_grid(self, device):
        widths = torch.tensor([2.0, 2.0, 2.0, 2.0], device=device)
        d = cell_centre_distance(widths)
        # d[i] = w[i]/2 + w[i+1] + w[i+2]/2 = 1 + 2 + 1 = 4
        assert d.shape == (2,)
        assert torch.allclose(d, torch.tensor([4.0, 4.0], device=device))

    def test_non_uniform(self, device):
        widths = torch.tensor([10.0, 20.0, 30.0, 15.0, 25.0, 10.0], device=device)
        d = cell_centre_distance(widths)
        assert d.shape == (4,)
        # d[0] = 10/2 + 20 + 30/2 = 40 ; d[1] = 20/2 + 30 + 15/2 = 47.5
        assert torch.isclose(d[0], torch.tensor(40.0, device=device))
        assert torch.isclose(d[1], torch.tensor(47.5, device=device))

    def test_min_spacing_floor(self, device):
        widths = torch.zeros(4, device=device)
        d = cell_centre_distance(widths, min_spacing=1e-3)
        assert torch.all(d >= 1e-3)


class TestCentralDifference:
    def test_linear_field(self, device):
        # Derivative of a linear field f(w)=w on a uniform grid is 1.
        field = (
            torch.arange(6, dtype=torch.float, device=device)
            .view(1, 1, 6, 1)
            .expand(1, 4, 6, 2)
        )
        spacing = cell_centre_distance(torch.ones(6, device=device))
        deriv = central_difference(field, axis=2, spacing=spacing)
        assert deriv.shape == (1, 4, 4, 2)
        assert torch.allclose(deriv, torch.ones_like(deriv))

    def test_non_uniform_spacing(self, device):
        field = torch.tensor([0.0, 1.0, 3.0, 6.0, 10.0, 15.0], device=device).view(
            1, 1, 6, 1
        )
        widths = torch.tensor([10.0, 20.0, 30.0, 15.0, 25.0, 10.0], device=device)
        spacing = cell_centre_distance(widths)
        deriv = central_difference(field, axis=2, spacing=spacing)
        # d[0] = (f[2]-f[0])/40 = 3/40 ; d[1] = (f[3]-f[1])/47.5 = 5/47.5
        assert torch.isclose(deriv[0, 0, 0, 0], torch.tensor(3.0 / 40.0, device=device))
        assert torch.isclose(deriv[0, 0, 1, 0], torch.tensor(5.0 / 47.5, device=device))

    def test_3d_axis(self, device):
        field = torch.randn(1, 6, 4, 3, 2, device=device)
        spacing = cell_centre_distance(torch.ones(6, device=device))
        deriv = central_difference(field, axis=1, spacing=spacing)
        assert deriv.shape == (1, 4, 4, 3, 2)


# ---------------------------------------------------------------------------
# SpatialDerivativeLoss
# ---------------------------------------------------------------------------


def _uniform_widths(shape, axes, device):
    return {ax: torch.ones(shape[ax], device=device) for ax in axes}


class TestSpatialDerivativeLoss:
    def test_constructor_defaults(self):
        loss = SpatialDerivativeLoss()
        assert loss.metric is mse
        assert loss.min_spacing == 1e-6

    def test_zero_for_identical_2d(self, device):
        loss = SpatialDerivativeLoss()
        target = torch.randn(1, 8, 16, 4, device=device)
        widths = _uniform_widths((1, 8, 16, 4), (1, 2), device)
        value = loss(target.clone(), target, widths)
        assert torch.isclose(value, torch.tensor(0.0, device=device), atol=1e-6)

    def test_zero_for_identical_3d(self, device):
        loss = SpatialDerivativeLoss()
        target = torch.randn(1, 6, 8, 4, 3, device=device)
        widths = _uniform_widths((1, 6, 8, 4, 3), (1, 2, 3), device)
        value = loss(target.clone(), target, widths)
        assert torch.isclose(value, torch.tensor(0.0, device=device), atol=1e-6)

    @pytest.mark.parametrize("metric_name", list(METRICS))
    def test_all_metrics_2d(self, metric_name, device):
        loss = SpatialDerivativeLoss(metric=METRICS[metric_name])
        target = torch.randn(2, 8, 16, 4, device=device) + 1.0
        pred = target + 0.05 * torch.randn_like(target)
        widths = _uniform_widths((2, 8, 16, 4), (1, 2), device)
        value = loss(pred, target, widths)
        assert torch.isfinite(value)
        assert value > 0

    def test_single_axis(self, device):
        loss = SpatialDerivativeLoss()
        pred = torch.randn(2, 8, 16, 4, device=device)
        target = torch.randn(2, 8, 16, 4, device=device)
        for ax in (1, 2):
            value = loss(pred, target, {ax: torch.ones(pred.shape[ax], device=device)})
            assert torch.isfinite(value)

    def test_non_uniform_grid(self, device):
        loss = SpatialDerivativeLoss()
        dx = torch.tensor([10.0, 20.0, 30.0, 15.0, 25.0, 10.0], device=device)
        target = torch.zeros(1, 4, 6, 2, device=device)
        pred = torch.zeros(1, 4, 6, 2, device=device)
        pred[0, :, :, 0] = torch.arange(6, device=device).float()
        value = loss(pred, target, {2: dx})
        assert torch.isfinite(value)
        assert value > 0

    def test_gradient_flow(self, device):
        loss = SpatialDerivativeLoss()
        target = torch.randn(2, 8, 16, 4, device=device)
        pred = torch.randn(2, 8, 16, 4, device=device, requires_grad=True)
        widths = _uniform_widths((2, 8, 16, 4), (1, 2), device)
        loss(pred, target, widths).backward()
        assert pred.grad is not None
        assert torch.isfinite(pred.grad).all()

    def test_mask_excludes_boundary_artifacts(self, device):
        # Large error only in the masked-out region -> loss ~ 0.
        loss = SpatialDerivativeLoss()
        target = torch.randn(1, 8, 12, 2, device=device)
        pred = target.clone()
        pred[:, :3, :, :] += 100.0
        mask = torch.zeros(8, 12, dtype=torch.bool, device=device)
        mask[4:, :] = True
        widths = _uniform_widths((1, 8, 12, 2), (1, 2), device)
        value = loss(pred, target, widths, mask=mask)
        assert torch.isclose(value, torch.tensor(0.0, device=device), atol=1e-4)

    def test_per_sample_mask_finite(self, device):
        # A per-sample mask with different active regions per sample is accepted
        # and produces a finite loss.
        loss = SpatialDerivativeLoss()
        target = torch.randn(2, 8, 10, 3, device=device) + 1.0
        pred = target + 0.05 * torch.randn_like(target)
        mask = torch.zeros(2, 8, 10, dtype=torch.bool, device=device)
        mask[0, 1:6, 2:8] = True
        mask[1, 2:7, 1:6] = True
        widths = _uniform_widths((2, 8, 10, 3), (1, 2), device)
        value = loss(pred, target, widths, mask=mask)
        assert torch.isfinite(value)

    def test_per_sample_mask_is_not_global(self, device):
        # A per-sample mask must not leak across samples: a large error confined
        # to masked-out rows of sample 0 (which are unperturbed and active for
        # sample 1) must leave the loss at zero.
        loss = SpatialDerivativeLoss()
        target = torch.randn(2, 8, 10, 2, device=device)
        pred = target.clone()
        pred[0, :3, :, :] += 100.0  # error in rows 0..2 of sample 0 only
        mask = torch.ones(2, 8, 10, dtype=torch.bool, device=device)
        mask[0, :4, :] = False  # sample 0: rows 0..3 inactive (cover error+stencil)
        widths = _uniform_widths((2, 8, 10, 2), (1, 2), device)
        value = loss(pred, target, widths, mask=mask)
        assert torch.isclose(value, torch.tensor(0.0, device=device), atol=1e-4)

    def test_sparse_mask_no_nan(self, device):
        # Norne-like sparse grid with relative_l2 must not produce NaN/Inf.
        loss = SpatialDerivativeLoss(metric=_relative_l2)
        dx = torch.tensor(
            [10.0, 20.0, 15.0, 10.0, 25.0, 30.0, 10.0, 20.0, 15.0, 10.0],
            device=device,
        )
        target = torch.randn(1, 10, 12, 3, device=device)
        pred = target + 0.1 * torch.randn_like(target)
        mask = torch.zeros(10, 12, dtype=torch.bool, device=device)
        mask[2:5, 3:8] = True
        value = loss(pred, target, {1: dx}, mask=mask)
        assert torch.isfinite(value)

    def test_single_timestep(self, device):
        loss = SpatialDerivativeLoss(metric=_relative_l2)
        pred = torch.randn(2, 8, 16, 1, device=device)
        target = torch.randn(2, 8, 16, 1, device=device)
        widths = _uniform_widths((2, 8, 16, 1), (1, 2), device)
        assert torch.isfinite(loss(pred, target, widths))

    def test_minimum_grid(self, device):
        # 3 cells along an axis is the minimum for a central difference.
        loss = SpatialDerivativeLoss()
        target = torch.randn(1, 3, 3, 2, device=device) + 1.0
        pred = target + 0.1
        widths = _uniform_widths((1, 3, 3, 2), (1, 2), device)
        value = loss(pred, target, widths)
        assert torch.isfinite(value)

    def test_custom_metric_eps(self, device):
        # A metric with a custom keyword can be injected via functools.partial.
        loss = SpatialDerivativeLoss(metric=functools.partial(_relative_l2, eps=1e-3))
        pred = torch.randn(1, 6, 6, 2, device=device)
        target = torch.zeros(1, 6, 6, 2, device=device)
        value = loss(pred, target, _uniform_widths((1, 6, 6, 2), (1, 2), device))
        assert torch.isfinite(value)

    def test_shape_mismatch_raises(self, device):
        loss = SpatialDerivativeLoss()
        pred = torch.randn(1, 8, 8, 2, device=device)
        target = torch.randn(1, 8, 8, 3, device=device)
        with pytest.raises(ValueError, match="same shape"):
            loss(pred, target, {1: torch.ones(8, device=device)})

    def test_empty_widths_raises(self, device):
        loss = SpatialDerivativeLoss()
        pred = torch.randn(1, 8, 8, 2, device=device)
        with pytest.raises(ValueError, match="at least one axis"):
            loss(pred, pred, {})

    def test_fully_masked_returns_zero(self, device):
        # No active cell anywhere -> no contributing axis -> zero (not NaN).
        loss = SpatialDerivativeLoss()
        pred = torch.randn(1, 8, 10, 2, device=device)
        target = torch.randn(1, 8, 10, 2, device=device)
        mask = torch.zeros(8, 10, dtype=torch.bool, device=device)
        widths = _uniform_widths((1, 8, 10, 2), (1, 2), device)
        value = loss(pred, target, widths, mask=mask)
        assert torch.isfinite(value)
        assert torch.isclose(value, torch.tensor(0.0, device=device))

    def test_masked_axis_excluded_from_average(self, device):
        # A mask that leaves no valid stencil along axis 2 (only two active
        # columns) but valid samples along axis 1. The fully-masked axis must be
        # excluded from the average, so differentiating {1, 2} equals {1} alone.
        loss = SpatialDerivativeLoss()
        pred = torch.randn(1, 6, 6, 2, device=device)
        target = torch.randn(1, 6, 6, 2, device=device)
        mask = torch.zeros(6, 6, dtype=torch.bool, device=device)
        mask[:, 2:4] = True  # two active columns -> no 3-wide window along W
        both = loss(
            pred, target, _uniform_widths((1, 6, 6, 2), (1, 2), device), mask=mask
        )
        only_axis1 = loss(
            pred, target, _uniform_widths((1, 6, 6, 2), (1,), device), mask=mask
        )
        assert torch.isclose(both, only_axis1)

    @pytest.mark.parametrize("bad_axis", [0, 3])
    def test_invalid_axis_key_raises(self, bad_axis, device):
        # axis 0 is the batch axis and axis 3 is the trailing feature/time axis;
        # neither is a valid spatial axis for a (B, H, W, T) field.
        loss = SpatialDerivativeLoss()
        pred = torch.randn(1, 8, 8, 2, device=device)
        with pytest.raises(ValueError, match="spatial axes"):
            loss(
                pred, pred, {bad_axis: torch.ones(pred.shape[bad_axis], device=device)}
            )

    def test_widths_length_mismatch_raises(self, device):
        # cell widths must span the full axis (length == pred.shape[axis]).
        loss = SpatialDerivativeLoss()
        pred = torch.randn(1, 8, 8, 2, device=device)
        with pytest.raises(ValueError, match="must be a 1-D tensor of length"):
            loss(pred, pred, {1: torch.ones(5, device=device)})


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
