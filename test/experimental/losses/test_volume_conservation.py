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

"""Tests for the generic volume-conservation enforcement loss.

Uses analytically-verifiable dummy reservoirs with known cell volumes, passed
explicitly to the (grid-convention-agnostic) loss.
"""

import functools

import pytest
import torch
import torch.nn.functional as F

from physicsnemo.experimental.losses import VolumeConservationLoss
from physicsnemo.metrics.general.mse import mse


# Small, self-contained comparison metrics used to exercise dependency injection
# (the loss accepts any ``metric(pred, target, dim=...)`` callable). Each reduces
# over the trailing (time) axis so the loss can average over the batch.
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


# 2D reservoir: cell widths -> volumes (areas); total area = 30 * 110 = 3300.
DX_2D = torch.tensor([10.0, 20.0, 30.0, 15.0, 25.0, 10.0])
DY_2D = torch.tensor([5.0, 10.0, 5.0, 10.0])
# 3D reservoir: total volume = 45 * 30 * 20 = 27000.
DX_3D = torch.tensor([10.0, 20.0, 15.0])
DY_3D = torch.tensor([5.0, 10.0, 5.0, 10.0])
DZ_3D = torch.tensor([8.0, 12.0])

METRICS = {"mse": mse, "l1": _l1, "relative_l2": _relative_l2, "huber": _huber}


def _volumes_2d(device):
    return (DY_2D.to(device).unsqueeze(1) * DX_2D.to(device).unsqueeze(0)).contiguous()


def _volumes_3d(device):
    return (
        DX_3D.to(device)[:, None, None]
        * DY_3D.to(device)[None, :, None]
        * DZ_3D.to(device)[None, None, :]
    ).contiguous()


class TestConstructor:
    def test_defaults(self):
        loss = VolumeConservationLoss()
        assert loss.metric is mse


class TestVolumeConservation2D:
    def test_zero_for_identical(self, device):
        vol = _volumes_2d(device)
        target = torch.ones(1, 4, 6, 3, device=device) * 0.5
        loss = VolumeConservationLoss()(target.clone(), target, vol)
        assert torch.isclose(loss, torch.tensor(0.0, device=device), atol=1e-6)

    def test_known_imbalance_relative_l2(self, device):
        vol = _volumes_2d(device)
        target = torch.ones(1, 4, 6, 1, device=device) * 0.5  # total mass 1650
        pred = target.clone()
        pred[0, 0, 0, 0] += 1.0  # cell volume 50 -> total 1700
        loss = VolumeConservationLoss(metric=_relative_l2)(pred, target, vol)
        assert abs(loss.item() - 50.0 / 1650.0) < 1e-4

    def test_known_imbalance_mse(self, device):
        vol = _volumes_2d(device)
        target = torch.ones(1, 4, 6, 1, device=device) * 0.5
        pred = target.clone()
        pred[0, 0, 0, 0] += 1.0
        loss = VolumeConservationLoss(metric=mse)(pred, target, vol)
        assert abs(loss.item() - 2500.0) < 1.0  # (1650 - 1700)^2

    def test_known_imbalance_l1(self, device):
        vol = _volumes_2d(device)
        target = torch.ones(1, 4, 6, 1, device=device) * 0.5
        pred = target.clone()
        pred[0, 0, 0, 0] += 1.0
        loss = VolumeConservationLoss(metric=_l1)(pred, target, vol)
        assert abs(loss.item() - 50.0) < 0.1

    @pytest.mark.parametrize("metric_name", list(METRICS))
    def test_all_metrics_run(self, metric_name, device):
        vol = _volumes_2d(device)
        target = torch.ones(1, 4, 6, 3, device=device) * 0.5
        pred = target + 0.05 * torch.randn_like(target)
        loss = VolumeConservationLoss(metric=METRICS[metric_name])(pred, target, vol)
        assert torch.isfinite(loss)
        assert loss > 0

    def test_volume_weighting_vs_uniform(self, device):
        vol = _volumes_2d(device)
        target = torch.ones(1, 4, 6, 1, device=device) * 0.5
        pred_small = target.clone()
        pred_small[0, 0, 0, 0] += 1.0  # small cell (volume 50)
        pred_large = target.clone()
        pred_large[0, 1, 2, 0] += 1.0  # large cell (volume 300)

        fn = VolumeConservationLoss(metric=_relative_l2)
        assert fn(pred_large, target, vol) > fn(pred_small, target, vol)

        # Without volume weighting both perturbations add +1 to the sum.
        assert torch.isclose(fn(pred_small, target, None), fn(pred_large, target, None))

    def test_multi_timestep(self, device):
        vol = _volumes_2d(device)
        target = torch.ones(1, 4, 6, 3, device=device) * 0.5
        pred = target.clone()
        pred[0, 0, 0, 0] += 1.0
        fn = VolumeConservationLoss(metric=_relative_l2)
        loss_t0 = fn(pred, target, vol)
        pred[0, 0, 0, 1] += 1.0
        loss_t01 = fn(pred, target, vol)
        assert loss_t01 > loss_t0

    def test_mask_excludes_perturbed_cell(self, device):
        vol = _volumes_2d(device)
        target = torch.ones(1, 4, 6, 1, device=device) * 0.5
        pred = target.clone()
        pred[0, 0, 0, 0] += 10.0
        mask = torch.ones(4, 6, dtype=torch.bool, device=device)
        mask[0, 0] = False
        loss = VolumeConservationLoss(metric=_relative_l2)(pred, target, vol, mask=mask)
        assert torch.isclose(loss, torch.tensor(0.0, device=device), atol=1e-5)


class TestVolumeConservation3D:
    def test_zero_for_identical(self, device):
        vol = _volumes_3d(device)
        target = torch.ones(1, 3, 4, 2, 2, device=device) * 0.3
        loss = VolumeConservationLoss()(target.clone(), target, vol)
        assert torch.isclose(loss, torch.tensor(0.0, device=device), atol=1e-6)

    def test_known_imbalance(self, device):
        vol = _volumes_3d(device)
        target = torch.ones(1, 3, 4, 2, 1, device=device) * 0.3  # total mass 8100
        pred = target.clone()
        pred[0, 0, 0, 0, 0] += 1.0  # cell volume 400 -> total 8500
        loss = VolumeConservationLoss(metric=_relative_l2)(pred, target, vol)
        assert abs(loss.item() - 400.0 / 8100.0) < 1e-3

    def test_gradient_flow(self, device):
        vol = _volumes_3d(device)
        target = torch.ones(1, 3, 4, 2, 2, device=device) * 0.3
        pred = torch.randn_like(target).requires_grad_(True)
        VolumeConservationLoss(metric=_relative_l2)(pred, target, vol).backward()
        assert pred.grad is not None
        assert torch.isfinite(pred.grad).all()

    def test_scale_invariance(self, device):
        vol = _volumes_3d(device)
        target = torch.ones(1, 3, 4, 2, 2, device=device) * 0.3
        pred = target + 0.1 * torch.randn_like(target)
        fn = VolumeConservationLoss(metric=_relative_l2)
        assert torch.isclose(
            fn(pred, target, vol), fn(pred * 5, target * 5, vol), rtol=1e-4
        )


class TestEdgeCases:
    def test_uniform_default_volumes(self, device):
        target = torch.ones(2, 4, 6, 3, device=device) * 0.5
        pred = target + 0.05 * torch.randn_like(target)
        loss = VolumeConservationLoss(metric=_relative_l2)(pred, target)
        assert torch.isfinite(loss)

    def test_single_timestep(self, device):
        vol = _volumes_2d(device)
        target = torch.ones(1, 4, 6, 1, device=device) * 0.5
        pred = target.clone()
        pred[0, 0, 0, 0] += 1.0
        loss = VolumeConservationLoss(metric=_relative_l2)(pred, target, vol)
        assert torch.isfinite(loss)
        assert loss > 0

    def test_custom_metric_partial(self, device):
        vol = _volumes_2d(device)
        target = torch.ones(1, 4, 6, 3, device=device) * 0.5
        pred = target + 0.05 * torch.randn_like(target)
        fn = VolumeConservationLoss(metric=functools.partial(_huber, delta=0.5))
        assert torch.isfinite(fn(pred, target, vol))

    def test_shape_mismatch_raises(self, device):
        pred = torch.randn(1, 4, 6, 2, device=device)
        target = torch.randn(1, 4, 6, 3, device=device)
        with pytest.raises(ValueError, match="same shape"):
            VolumeConservationLoss()(pred, target)

    def test_cell_volumes_shape_mismatch_raises(self, device):
        pred = torch.randn(1, 4, 6, 2, device=device)
        bad_vol = torch.ones(4, 5, device=device)  # spatial is (4, 6)
        with pytest.raises(ValueError, match="cell_volumes must have shape"):
            VolumeConservationLoss()(pred, pred, bad_vol)

    def test_mask_shape_mismatch_raises(self, device):
        pred = torch.randn(2, 4, 6, 2, device=device)
        bad_mask = torch.ones(4, 5, dtype=torch.bool, device=device)
        with pytest.raises(ValueError, match="mask must have shape"):
            VolumeConservationLoss()(pred, pred, mask=bad_mask)

    def test_per_sample_mask_accepted(self, device):
        # A (B, *spatial) per-sample mask is valid and produces a finite loss.
        pred = torch.ones(2, 4, 6, 3, device=device) * 0.5
        target = pred.clone()
        mask = torch.ones(2, 4, 6, dtype=torch.bool, device=device)
        value = VolumeConservationLoss()(pred, target, mask=mask)
        assert torch.isfinite(value)

    def test_per_sample_mask_is_not_global(self, device):
        # A per-sample mask must not leak across samples: a perturbation in a cell
        # that is masked-out for sample 0 (but active for the unperturbed sample 1)
        # must leave the loss at zero.
        vol = _volumes_2d(device)
        target = torch.ones(2, 4, 6, 1, device=device) * 0.5
        pred = target.clone()
        pred[0, 0, 0, 0] += 1.0  # perturb sample 0 only, cell (0, 0)
        mask = torch.ones(2, 4, 6, dtype=torch.bool, device=device)
        mask[0, 0, 0] = False  # exclude that cell for sample 0 only
        loss = VolumeConservationLoss(metric=mse)(pred, target, vol, mask=mask)
        assert torch.isclose(loss, torch.tensor(0.0, device=device), atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
