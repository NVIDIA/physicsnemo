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

"""Comprehensive tests for physics losses with analytically-verifiable dummy reservoirs."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.losses import UnifiedLoss, get_loss_function
from training.physics_losses import (
    MassConservationLoss,
    _extract_grid_widths,
    build_physics_losses,
    cell_centre_distance,
    central_difference,
    compute_cell_volumes_from_widths,
    extract_grid_widths_for_axis,
)

# ===================================================================
# Dummy Reservoir Fixtures
# ===================================================================
# 2D reservoir: H=4, W=6, T=3, C=12
#   dx = [10, 20, 30, 15, 25, 10] metres  (varies along W, channel -3)
#   dy = [5, 10, 5, 10] metres            (varies along H, channel -2)
#   Total area = sum_h sum_w (dy[h] * dx[w])
#              = (5+10+5+10) * (10+20+30+15+25+10) = 30 * 110 = 3300 m^2
#
# 3D reservoir: X=3, Y=4, Z=2, T=2, C=11
#   dx = [10, 20, 15] metres     (varies along X, channel -4)
#   dy = [5, 10, 5, 10] metres   (varies along Y, channel -3)
#   dz = [8, 12] metres          (varies along Z, channel -2)
#   Total volume = 45 * 30 * 20 = 27000 m^3

DX_2D = torch.tensor([10.0, 20.0, 30.0, 15.0, 25.0, 10.0])
DY_2D = torch.tensor([5.0, 10.0, 5.0, 10.0])
TOTAL_AREA_2D = DY_2D.sum().item() * DX_2D.sum().item()  # 3300

DX_3D = torch.tensor([10.0, 20.0, 15.0])
DY_3D = torch.tensor([5.0, 10.0, 5.0, 10.0])
DZ_3D = torch.tensor([8.0, 12.0])
TOTAL_VOL_3D = DX_3D.sum().item() * DY_3D.sum().item() * DZ_3D.sum().item()  # 27000


def _build_2d_reservoir(B=1, T=3, saturation_fn=None):
    """Build a 2D reservoir with known grid widths and saturation field.

    Returns (inputs, target, pred) where:
    - inputs: (B, 4, 6, T, 12)
    - target: (B, 4, 6, T)  saturation field
    - pred:   (B, 4, 6, T)  prediction (= target by default)
    """
    H, W, C = 4, 6, 12
    inputs = torch.zeros(B, H, W, T, C)
    inputs[..., -3] = DX_2D.view(1, 1, W, 1).expand(B, H, W, T)
    inputs[..., -2] = DY_2D.view(1, H, 1, 1).expand(B, H, W, T)
    inputs[..., -1] = torch.linspace(0, 10, T).view(1, 1, 1, T).expand(B, H, W, T)

    if saturation_fn is not None:
        target = saturation_fn(B, H, W, T)
    else:
        target = torch.ones(B, H, W, T) * 0.5

    return inputs, target, target.clone()


def _build_3d_reservoir(B=1, T=2, saturation_fn=None):
    """Build a 3D reservoir with known grid widths.

    Returns (inputs, target, pred).
    """
    X, Y, Z, C = 3, 4, 2, 11
    inputs = torch.zeros(B, X, Y, Z, T, C)
    inputs[..., -4] = DX_3D.view(1, X, 1, 1, 1).expand(B, X, Y, Z, T)
    inputs[..., -3] = DY_3D.view(1, 1, Y, 1, 1).expand(B, X, Y, Z, T)
    inputs[..., -2] = DZ_3D.view(1, 1, 1, Z, 1).expand(B, X, Y, Z, T)
    inputs[..., -1] = torch.linspace(0, 10, T).view(1, 1, 1, 1, T).expand(B, X, Y, Z, T)

    if saturation_fn is not None:
        target = saturation_fn(B, X, Y, Z, T)
    else:
        target = torch.ones(B, X, Y, Z, T) * 0.3

    return inputs, target, target.clone()


# ===================================================================
# Grid width extraction
# ===================================================================


class TestExtractGridWidths:
    """Tests for ExtractGridWidths."""

    def test_2d_extraction(self):
        """Verify 2D grid width extraction returns correct dx and dy vectors."""
        inputs, _, _ = _build_2d_reservoir()
        widths = _extract_grid_widths(inputs, spatial_ndim=2)
        assert len(widths) == 2
        assert torch.allclose(widths[0], DY_2D)  # axis-1 = H = dy
        assert torch.allclose(widths[1], DX_2D)  # axis-2 = W = dx

    def test_3d_extraction(self):
        """Verify 3D grid width extraction returns correct dx, dy, and dz vectors."""
        inputs, _, _ = _build_3d_reservoir()
        widths = _extract_grid_widths(inputs, spatial_ndim=3)
        assert len(widths) == 3
        assert torch.allclose(widths[0], DX_3D)  # axis-1 = X = dx
        assert torch.allclose(widths[1], DY_3D)  # axis-2 = Y = dy
        assert torch.allclose(widths[2], DZ_3D)  # axis-3 = Z = dz


# ===================================================================
# Cell volume computation
# ===================================================================


class TestCellVolumes:
    """Tests for CellVolumes."""

    def test_2d_total_area(self):
        """Total area of 2D reservoir should be 3300 m^2."""
        inputs, _, _ = _build_2d_reservoir()
        vol = compute_cell_volumes_from_widths(inputs, spatial_ndim=2)
        assert vol.shape == (4, 6)
        total = vol.sum().item()
        assert abs(total - TOTAL_AREA_2D) < 1e-3, (
            f"Expected {TOTAL_AREA_2D}, got {total}"
        )

    def test_2d_individual_cells(self):
        """Spot-check: cell [0,0] should have area dy[0]*dx[0] = 5*10 = 50."""
        inputs, _, _ = _build_2d_reservoir()
        vol = compute_cell_volumes_from_widths(inputs, spatial_ndim=2)
        assert torch.isclose(vol[0, 0], torch.tensor(50.0))
        # cell [1,2] = dy[1]*dx[2] = 10*30 = 300
        assert torch.isclose(vol[1, 2], torch.tensor(300.0))
        # cell [3,5] = dy[3]*dx[5] = 10*10 = 100
        assert torch.isclose(vol[3, 5], torch.tensor(100.0))

    def test_3d_total_volume(self):
        """Total volume of 3D reservoir should be 27000 m^3."""
        inputs, _, _ = _build_3d_reservoir()
        vol = compute_cell_volumes_from_widths(inputs, spatial_ndim=3)
        assert vol.shape == (3, 4, 2)
        total = vol.sum().item()
        assert abs(total - TOTAL_VOL_3D) < 1e-2, f"Expected {TOTAL_VOL_3D}, got {total}"

    def test_3d_individual_cells(self):
        """Spot-check: cell [0,0,0] = dx[0]*dy[0]*dz[0] = 10*5*8 = 400."""
        inputs, _, _ = _build_3d_reservoir()
        vol = compute_cell_volumes_from_widths(inputs, spatial_ndim=3)
        assert torch.isclose(vol[0, 0, 0], torch.tensor(400.0))
        # cell [1,2,1] = 20*5*12 = 1200
        assert torch.isclose(vol[1, 2, 1], torch.tensor(1200.0))


# ===================================================================
# Cell centre distance
# ===================================================================


class TestCellCentreDistance:
    """Tests for CellCentreDistance."""

    def test_uniform_grid(self):
        """Verify cell-centre distances are constant on a uniform grid."""
        widths = torch.tensor([2.0, 2.0, 2.0, 2.0])
        d = cell_centre_distance(widths)
        assert d.shape == (2,)
        # d[i] = w[i]/2 + w[i+1] + w[i+2]/2 = 1+2+1 = 4
        assert torch.allclose(d, torch.tensor([4.0, 4.0]))

    def test_non_uniform(self):
        """Verify cell-centre distances match hand-computed values on a non-uniform grid."""
        widths = DX_2D  # [10, 20, 30, 15, 25, 10]
        d = cell_centre_distance(widths)
        assert d.shape == (4,)
        # d[0] = 10/2 + 20 + 30/2 = 5 + 20 + 15 = 40
        assert torch.isclose(d[0], torch.tensor(40.0))
        # d[1] = 20/2 + 30 + 15/2 = 10 + 30 + 7.5 = 47.5
        assert torch.isclose(d[1], torch.tensor(47.5))


# ===================================================================
# Central difference
# ===================================================================


class TestCentralDifference:
    """Tests for CentralDifference."""

    def test_linear_field_2d(self):
        """Derivative of linear field f(w) = w should be 1/spacing."""
        B, H, W, T = 1, 4, 6, 2
        field = torch.arange(W, dtype=torch.float).view(1, 1, W, 1).expand(B, H, W, T)
        widths = torch.ones(W)
        spacing = cell_centre_distance(widths)  # all = 2.0

        deriv = central_difference(field, axis=2, spacing=spacing)
        # (f[i+2] - f[i]) / 2 = 2/2 = 1 for a linear field
        assert deriv.shape == (B, H, W - 2, T)
        assert torch.allclose(deriv, torch.ones_like(deriv))

    def test_non_uniform_spacing(self):
        """Known values with non-uniform grid."""
        _B, _H, _W, _T = 1, 1, 6, 1
        # f = [0, 1, 3, 6, 10, 15]
        field = torch.tensor([0.0, 1.0, 3.0, 6.0, 10.0, 15.0]).view(1, 1, 6, 1)
        widths = DX_2D  # [10, 20, 30, 15, 25, 10]
        spacing = cell_centre_distance(widths)  # [40, 47.5, 32.5, 27.5]

        deriv = central_difference(field, axis=2, spacing=spacing)
        # d[0] = (f[2]-f[0])/40 = 3/40 = 0.075
        assert torch.isclose(deriv[0, 0, 0, 0], torch.tensor(3.0 / 40.0))
        # d[1] = (f[3]-f[1])/47.5 = 5/47.5
        assert torch.isclose(deriv[0, 0, 1, 0], torch.tensor(5.0 / 47.5))

    def test_3d_axis(self):
        """Verify central difference output shape for a 3D field along axis 1."""
        B, X, Y, Z, T = 1, 6, 4, 3, 2
        field = torch.randn(B, X, Y, Z, T)
        spacing = cell_centre_distance(torch.ones(X))
        deriv = central_difference(field, axis=1, spacing=spacing)
        assert deriv.shape == (B, X - 2, Y, Z, T)


# ===================================================================
# Mass Conservation Loss — Analytic 2D Reservoir
# ===================================================================


class TestMassConservation2DReservoir:
    """Tests for MassConservation2DReservoir."""

    def test_zero_loss_identical_fields(self):
        """Verify mass conservation loss is zero when prediction equals target."""
        inputs, target, pred = _build_2d_reservoir()
        loss = MassConservationLoss(use_cell_volumes=True)(pred, target, inputs)
        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)

    def test_known_mass_imbalance(self):
        """Add extra saturation to one cell and verify the loss reflects it."""
        inputs, target, pred = _build_2d_reservoir(T=1)
        # target: all 0.5, so total mass at t=0 = 0.5 * 3300 = 1650
        # Add 1.0 to cell [0,0] (volume=50): pred mass = 1650 + 50 = 1700
        pred[0, 0, 0, 0] += 1.0

        loss_fn = MassConservationLoss(use_cell_volumes=True)
        loss = loss_fn(pred, target, inputs)

        # Analytic: M_true = [1650], M_pred = [1700]
        # L = |1650-1700| / (|1650| + eps) = 50/1650 ~ 0.03030
        expected = 50.0 / 1650.0
        assert abs(loss.item() - expected) < 1e-4, (
            f"Expected ~{expected:.4f}, got {loss.item():.4f}"
        )

    def test_uniform_vs_volume_weighted(self):
        """Volume weighting should change the loss when grid is non-uniform."""
        inputs, target, _ = _build_2d_reservoir(T=1)
        pred = target.clone()
        pred[0, 0, 0, 0] += 1.0  # cell [0,0] has small volume (50)
        pred_large = target.clone()
        pred_large[0, 1, 2, 0] += 1.0  # cell [1,2] has large volume (300)

        loss_fn_vol = MassConservationLoss(use_cell_volumes=True)
        loss_fn_uni = MassConservationLoss(use_cell_volumes=False)

        # With volume weighting, perturbation in larger cell has more impact
        loss_small_vol = loss_fn_vol(pred, target, inputs).item()
        loss_large_vol = loss_fn_vol(pred_large, target, inputs).item()
        assert loss_large_vol > loss_small_vol

        # Without volume weighting, both perturbations add same amount (+1.0 to sum)
        loss_small_uni = loss_fn_uni(pred, target, inputs).item()
        loss_large_uni = loss_fn_uni(pred_large, target, inputs).item()
        assert abs(loss_small_uni - loss_large_uni) < 1e-6

    def test_mass_conservation_multi_timestep(self):
        """Mass error at different timesteps contributes via L2 norm."""
        inputs, target, pred = _build_2d_reservoir(T=3)
        # Perturb only at t=0
        pred[0, 0, 0, 0] += 1.0
        loss_t0_only = MassConservationLoss(use_cell_volumes=True)(pred, target, inputs)

        # Now also perturb at t=1
        pred[0, 0, 0, 1] += 1.0
        loss_t01 = MassConservationLoss(use_cell_volumes=True)(pred, target, inputs)

        # L2 norm over time: more errors => higher loss
        assert loss_t01 > loss_t0_only

    def test_with_spatial_mask(self):
        """Mask out the perturbed cell => loss should be zero."""
        inputs, target, pred = _build_2d_reservoir(T=1)
        pred[0, 0, 0, 0] += 10.0

        mask = torch.ones(4, 6, dtype=torch.bool)
        mask[0, 0] = False  # mask out the perturbed cell

        loss = MassConservationLoss(use_cell_volumes=True)(
            pred, target, inputs, spatial_mask=mask
        )
        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-5)


# ===================================================================
# Mass Conservation Loss — Analytic 3D Reservoir
# ===================================================================


class TestMassConservation3DReservoir:
    """Tests for MassConservation3DReservoir."""

    def test_zero_loss_identical(self):
        """Verify 3D mass conservation loss is zero when prediction equals target."""
        inputs, target, pred = _build_3d_reservoir()
        loss = MassConservationLoss(use_cell_volumes=True)(pred, target, inputs)
        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)

    def test_known_mass_imbalance_3d(self):
        """Perturb one cell and check analytic loss."""
        inputs, target, pred = _build_3d_reservoir(T=1)
        # target: all 0.3, total mass = 0.3 * 27000 = 8100
        # Perturb cell [0,0,0] (volume=400): pred mass = 8100 + 400*1.0 = 8500
        pred[0, 0, 0, 0, 0] += 1.0

        loss = MassConservationLoss(use_cell_volumes=True)(pred, target, inputs)
        expected = 400.0 / 8100.0
        assert abs(loss.item() - expected) < 1e-3, (
            f"Expected ~{expected:.4f}, got {loss.item():.4f}"
        )

    def test_gradient_flow(self):
        """Verify gradients propagate through the 3D mass conservation loss."""
        inputs, target, _ = _build_3d_reservoir()
        pred = torch.randn_like(target, requires_grad=True)
        loss = MassConservationLoss(use_cell_volumes=True)(pred, target, inputs)
        loss.backward()
        assert pred.grad is not None

    def test_scale_invariance(self):
        """Verify relative mass conservation loss is invariant to uniform scaling."""
        inputs, target, _ = _build_3d_reservoir()
        pred = target + 0.1 * torch.randn_like(target)
        fn = MassConservationLoss(use_cell_volumes=True)
        l1 = fn(pred, target, inputs)
        # Reset cache for fresh volumes
        fn2 = MassConservationLoss(use_cell_volumes=True)
        l2 = fn2(pred * 5, target * 5, inputs)
        assert torch.isclose(l1, l2, rtol=1e-4)


# ===================================================================
# Derivative utilities — extract_grid_widths_for_axis
# ===================================================================


class TestExtractGridWidthsForAxis:
    """Tests for ExtractGridWidthsForAxis."""

    def test_2d_dx(self):
        """Verify dx extraction from 2D reservoir inputs matches expected widths."""
        inputs, _, _ = _build_2d_reservoir()
        w = extract_grid_widths_for_axis(inputs, spatial_ndim=2, dim_name="dx")
        assert torch.allclose(w, DX_2D)

    def test_2d_dy(self):
        """Verify dy extraction from 2D reservoir inputs matches expected widths."""
        inputs, _, _ = _build_2d_reservoir()
        w = extract_grid_widths_for_axis(inputs, spatial_ndim=2, dim_name="dy")
        assert torch.allclose(w, DY_2D)

    def test_3d_all(self):
        """Verify dx, dy, and dz extraction from 3D reservoir inputs."""
        inputs, _, _ = _build_3d_reservoir()
        assert torch.allclose(extract_grid_widths_for_axis(inputs, 3, "dx"), DX_3D)
        assert torch.allclose(extract_grid_widths_for_axis(inputs, 3, "dy"), DY_3D)
        assert torch.allclose(extract_grid_widths_for_axis(inputs, 3, "dz"), DZ_3D)

    def test_invalid_dim_raises(self):
        """Verify ValueError is raised for an unknown derivative dimension name."""
        inputs, _, _ = _build_2d_reservoir()
        with pytest.raises(ValueError, match="Unknown derivative dim"):
            extract_grid_widths_for_axis(inputs, 2, "dz")


# ===================================================================
# build_physics_losses
# ===================================================================


class TestBuildPhysicsLosses:
    """Tests for BuildPhysicsLosses."""

    def test_none_config(self):
        """Verify build_physics_losses returns empty dict for None config."""
        assert build_physics_losses(None) == {}

    def test_disabled(self):
        """Verify build_physics_losses returns empty dict when loss is disabled."""
        assert build_physics_losses({"mass_conservation": {"enabled": False}}) == {}

    def test_enabled(self):
        """Verify build_physics_losses creates a MassConservationLoss with correct weight."""
        result = build_physics_losses(
            {"mass_conservation": {"enabled": True, "weight": 0.5}}
        )
        assert "mass_conservation" in result
        mod, w = result["mass_conservation"]
        assert isinstance(mod, MassConservationLoss)
        assert w == 0.5

    def test_use_cell_volumes_flag(self):
        """Verify use_cell_volumes flag is propagated to MassConservationLoss."""
        cfg = {
            "mass_conservation": {
                "enabled": True,
                "weight": 1.0,
                "use_cell_volumes": True,
            }
        }
        mod, _ = build_physics_losses(cfg)["mass_conservation"]
        assert mod.use_cell_volumes is True

    def test_pressure_warning(self):
        """Verify a UserWarning is emitted when applying mass conservation to pressure."""
        cfg = {"mass_conservation": {"enabled": True, "weight": 1.0}}
        with pytest.warns(UserWarning, match="pressure"):
            build_physics_losses(cfg, variable="pressure")

    def test_zero_weight_skipped(self):
        """Verify a physics loss with zero weight is omitted from the result."""
        assert (
            build_physics_losses(
                {"mass_conservation": {"enabled": True, "weight": 0.0}}
            )
            == {}
        )


# ===================================================================
# Integration: UnifiedLoss + physics on dummy reservoir
# ===================================================================


class TestIntegrationReservoir:
    """Tests for IntegrationReservoir."""

    def test_full_pipeline_2d(self):
        """End-to-end: data loss + derivative + mass conservation on 2D reservoir."""
        B, T = 1, 3
        inputs, target, pred = _build_2d_reservoir(B=B, T=T)
        pred = pred + 0.01 * torch.randn_like(pred)

        fn = UnifiedLoss(
            types=["mse"],
            weights=[1.0],
            derivative_config={"enabled": True, "weight": 0.1, "dims": ["dx"]},
            physics_losses={"mc": (MassConservationLoss(use_cell_volumes=True), 0.5)},
        )
        loss = fn(pred, target, inputs)
        assert loss > 0
        assert not torch.isnan(loss)

    def test_full_pipeline_3d(self):
        """End-to-end on 3D reservoir with all derivative dims."""
        B, T = 1, 2
        inputs, target, pred = _build_3d_reservoir(B=B, T=T)
        pred = pred + 0.01 * torch.randn_like(pred)

        fn = UnifiedLoss(
            types=["relative_l2"],
            weights=[1.0],
            derivative_config={
                "enabled": True,
                "weight": 0.1,
                "dims": ["dx", "dy", "dz"],
            },
            physics_losses={"mc": (MassConservationLoss(use_cell_volumes=True), 0.5)},
        )
        loss = fn(pred, target, inputs)
        assert loss > 0
        assert not torch.isnan(loss)

    def test_gradient_full_pipeline(self):
        """Verify gradients propagate through the full unified loss pipeline."""
        inputs, target, _ = _build_2d_reservoir()
        pred = torch.randn_like(target, requires_grad=True)
        fn = UnifiedLoss(
            types=["relative_l2"],
            derivative_config={"enabled": True, "weight": 0.5, "dims": ["dx", "dy"]},
            physics_losses={"mc": (MassConservationLoss(use_cell_volumes=True), 1.0)},
        )
        loss = fn(pred, target, inputs)
        loss.backward()
        assert pred.grad is not None

    def test_factory_end_to_end(self):
        """Build from config dict like Hydra would produce."""
        cfg = {
            "types": ["relative_l2"],
            "weights": [1.0],
            "derivative": {"enabled": True, "weight": 0.3, "dims": ["dx"]},
            "physics": {
                "mass_conservation": {
                    "enabled": True,
                    "weight": 0.5,
                    "use_cell_volumes": True,
                },
            },
        }
        fn = get_loss_function(cfg, variable="saturation")
        inputs, target, pred = _build_2d_reservoir()
        pred = pred + 0.01 * torch.randn_like(pred)
        loss = fn(pred, target, inputs)
        assert loss > 0


# ===================================================================
# AR window compatibility
# ===================================================================


class TestARWindow:
    """Tests for ARWindow."""

    def test_single_timestep_mass(self):
        """Verify mass conservation loss works for a single-timestep AR window."""
        B, H, W, C = 1, 4, 6, 12
        inputs = torch.zeros(B, H, W, 1, C)
        inputs[..., -3] = DX_2D.view(1, 1, 6, 1)
        inputs[..., -2] = DY_2D.view(1, 4, 1, 1)
        inputs[..., -1] = 0.0

        target = torch.ones(B, H, W, 1) * 0.5
        pred = target.clone()
        pred[0, 0, 0, 0] += 1.0

        loss = MassConservationLoss(use_cell_volumes=True)(pred, target, inputs)
        assert not torch.isnan(loss)
        assert loss > 0


# ===================================================================
# Mass conservation metric options
# ===================================================================


class TestMassConservationMetrics:
    """Tests for configurable metric in MassConservationLoss."""

    @pytest.fixture
    def reservoir_data(self):
        """Simple 2D reservoir data for metric tests."""
        inputs, target, pred = _build_2d_reservoir(T=3)
        pred = pred + 0.05 * torch.randn_like(pred)
        return inputs, target, pred

    @pytest.mark.parametrize("metric", ["relative_l2", "mse", "l1", "huber"])
    def test_all_metrics_run(self, metric, reservoir_data):
        """Verify each supported metric produces a finite positive loss."""
        inputs, target, pred = reservoir_data
        fn = MassConservationLoss(metric=metric)
        loss = fn(pred, target, inputs)
        assert not torch.isnan(loss)
        assert loss > 0

    def test_invalid_metric_raises(self):
        """Verify ValueError is raised for an unsupported metric name."""
        with pytest.raises(ValueError, match="metric must be"):
            MassConservationLoss(metric="invalid")

    def test_default_metric_is_relative_l2(self):
        """Verify the default metric for MassConservationLoss is relative_l2."""
        fn = MassConservationLoss()
        assert fn.metric == "relative_l2"

    def test_mse_metric_value(self):
        """MSE metric on integrated quantities should match hand calculation."""
        inputs, target, _ = _build_2d_reservoir(T=1)
        pred = target.clone()
        pred[0, 0, 0, 0] += 1.0
        fn = MassConservationLoss(use_cell_volumes=True, metric="mse")
        loss = fn(pred, target, inputs)
        # M_true = 0.5 * 3300 = 1650, M_pred = 1650 + 50 = 1700
        # MSE over time (T=1): (1650 - 1700)^2 = 2500
        expected = 2500.0
        assert abs(loss.item() - expected) < 1.0

    def test_l1_metric_value(self):
        """Verify L1 metric returns the expected absolute mass difference."""
        inputs, target, _ = _build_2d_reservoir(T=1)
        pred = target.clone()
        pred[0, 0, 0, 0] += 1.0
        fn = MassConservationLoss(use_cell_volumes=True, metric="l1")
        loss = fn(pred, target, inputs)
        # L1: |1650 - 1700| = 50
        expected = 50.0
        assert abs(loss.item() - expected) < 0.1

    def test_different_metrics_give_different_values(self, reservoir_data):
        """Verify different metrics produce distinct loss values on the same data."""
        inputs, target, pred = reservoir_data
        losses = {}
        for metric in ["relative_l2", "mse", "l1"]:
            fn = MassConservationLoss(metric=metric)
            losses[metric] = fn(pred, target, inputs).item()
        assert losses["mse"] != losses["l1"]
        assert losses["relative_l2"] != losses["mse"]


class TestMetricInheritance:
    """Tests for metric inheritance from data loss via factory."""

    def test_null_inherits_first_data_loss(self):
        """Verify null metric inherits the provided default_metric."""
        cfg = {
            "mass_conservation": {"enabled": True, "weight": 1.0, "metric": None},
        }
        result = build_physics_losses(cfg, default_metric="mse")
        mod, _ = result["mass_conservation"]
        assert mod.metric == "mse"

    def test_null_inherits_relative_l2_by_default(self):
        """Verify omitted metric defaults to relative_l2 when no default given."""
        cfg = {
            "mass_conservation": {"enabled": True, "weight": 1.0},
        }
        result = build_physics_losses(cfg)
        mod, _ = result["mass_conservation"]
        assert mod.metric == "relative_l2"

    def test_explicit_metric_overrides_default(self):
        """Verify an explicit metric in config overrides the default_metric."""
        cfg = {
            "mass_conservation": {"enabled": True, "weight": 1.0, "metric": "l1"},
        }
        result = build_physics_losses(cfg, default_metric="mse")
        mod, _ = result["mass_conservation"]
        assert mod.metric == "l1"

    def test_factory_end_to_end_inheritance(self):
        """Verify factory inherits data loss type as physics metric end-to-end."""
        from training.losses import get_loss_function

        cfg = {
            "types": ["mse"],
            "weights": [1.0],
            "physics": {
                "mass_conservation": {"enabled": True, "weight": 0.5},
            },
        }
        fn = get_loss_function(cfg, variable="saturation")
        mod, _ = fn._physics_losses["mass_conservation"]
        assert mod.metric == "mse"

    def test_factory_explicit_override(self):
        """Verify factory respects an explicit physics metric override in config."""
        from training.losses import get_loss_function

        cfg = {
            "types": ["mse"],
            "weights": [1.0],
            "physics": {
                "mass_conservation": {
                    "enabled": True,
                    "weight": 0.5,
                    "metric": "relative_l2",
                },
            },
        }
        fn = get_loss_function(cfg, variable="saturation")
        mod, _ = fn._physics_losses["mass_conservation"]
        assert mod.metric == "relative_l2"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
