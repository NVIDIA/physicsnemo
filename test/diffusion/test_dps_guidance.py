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

"""Tests for DPS (Diffusion Posterior Sampling) guidance."""

import pytest
import torch
import torch._functorch.config as _functorch_config

from physicsnemo.diffusion.guidance import (
    DataConsistencyDPSGuidance,
    DPSScorePredictor,
    ModelConsistencyDPSGuidance,
)
from physicsnemo.diffusion.noise_schedulers import EDMNoiseScheduler

from .helpers import (
    Conv2dX0Predictor,
    Conv3dX0Predictor,
    FlatLinearX0Predictor,
    compare_outputs,
    instantiate_model_deterministic,
    load_or_create_reference,
    make_input,
)

# =============================================================================
# Constants and Configurations
# =============================================================================

REF_PREFIX = "test_dps_guidance_"
BATCH = 2

SPATIAL_CONFIGS = [
    ("1d", (BATCH, 3, 16), FlatLinearX0Predictor, {"features": 3 * 16}),
    ("2d", (BATCH, 3, 8, 6), Conv2dX0Predictor, {"channels": 3}),
    ("3d", (BATCH, 2, 4, 4, 4), Conv3dX0Predictor, {"channels": 2}),
]

# (config_name, description)
GUIDANCE_CONFIGS = [
    ("data_l2", "DataConsistency L2"),
    ("data_l2_sda", "DataConsistency L2 + SDA"),
    ("data_l1_sda", "DataConsistency L1 + SDA"),
    ("model_l2", "ModelConsistency L2"),
    ("model_l2_sda", "ModelConsistency L2 + SDA"),
    ("model_l1_sda", "ModelConsistency L1 + SDA"),
]

MULTI_GUIDANCE_CONFIGS = [
    ("two_data", "Two DataConsistency guidances"),
    ("data_and_model", "DataConsistency + ModelConsistency"),
]


# =============================================================================
# Helpers
# =============================================================================


def _nonlinear_obs_op(x):
    """Nonlinear observation operator: select first channel and square."""
    return x[:, :1] ** 2


def _make_mask(shape, device):
    """Create a boolean mask with a few observed locations."""
    mask = torch.zeros(shape, dtype=torch.bool, device=device)
    if len(shape) == 3:
        mask[:, :, 0] = True
        mask[:, :, 5] = True
    elif len(shape) == 4:
        mask[:, :, 2, 3] = True
        mask[:, :, 5, 1] = True
    else:
        mask[:, :, 1, 2, 1] = True
        mask[:, :, 0, 3, 2] = True
    return mask


def _make_guidance(config_name, shape, device, seed=310, create_graph=False):
    """Build a single guidance object from a config name."""
    scheduler = EDMNoiseScheduler()
    if config_name == "data_l2":
        mask = _make_mask(shape, device)
        y = make_input(shape, seed=seed, device=device)
        return DataConsistencyDPSGuidance(
            mask=mask,
            y=y,
            std_y=0.1,
            create_graph=create_graph,
        )
    elif config_name == "data_l2_sda":
        mask = _make_mask(shape, device)
        y = make_input(shape, seed=seed, device=device)
        return DataConsistencyDPSGuidance(
            mask=mask,
            y=y,
            std_y=0.1,
            gamma=1.0,
            sigma_fn=scheduler.sigma,
            alpha_fn=scheduler.alpha,
            create_graph=create_graph,
        )
    elif config_name == "data_l1_sda":
        mask = _make_mask(shape, device)
        y = make_input(shape, seed=seed, device=device)
        return DataConsistencyDPSGuidance(
            mask=mask,
            y=y,
            std_y=0.1,
            norm=1,
            gamma=1.0,
            sigma_fn=scheduler.sigma,
            alpha_fn=scheduler.alpha,
            create_graph=create_graph,
        )
    elif config_name == "model_l2":
        obs_shape = (*shape[:1], 1, *shape[2:])
        y = make_input(obs_shape, seed=seed, device=device)
        return ModelConsistencyDPSGuidance(
            observation_operator=_nonlinear_obs_op,
            y=y,
            std_y=0.1,
            create_graph=create_graph,
        )
    elif config_name == "model_l2_sda":
        obs_shape = (*shape[:1], 1, *shape[2:])
        y = make_input(obs_shape, seed=seed, device=device)
        return ModelConsistencyDPSGuidance(
            observation_operator=_nonlinear_obs_op,
            y=y,
            std_y=0.1,
            gamma=0.5,
            sigma_fn=scheduler.sigma,
            alpha_fn=scheduler.alpha,
            create_graph=create_graph,
        )
    elif config_name == "model_l1_sda":
        obs_shape = (*shape[:1], 1, *shape[2:])
        y = make_input(obs_shape, seed=seed, device=device)
        return ModelConsistencyDPSGuidance(
            observation_operator=_nonlinear_obs_op,
            y=y,
            std_y=0.1,
            norm=1,
            gamma=0.5,
            sigma_fn=scheduler.sigma,
            alpha_fn=scheduler.alpha,
            create_graph=create_graph,
        )
    raise ValueError(f"Unknown guidance config: {config_name}")


def _make_multi_guidance(
    config_name,
    shape,
    device,
    seed=420,
    create_graph=False,
    retain_all=False,
):
    """Build a list of guidance objects for multi-guidance tests.

    When create_graph=True (for gradient flow tests), retain_graph=True is set
    on ALL guidances because loss.backward() also needs to traverse the forward
    graph after all autograd.grad calls. When retain_all=True (for compile
    tests), all guidances share the same retain_graph=True to avoid torch.compile
    guard failures from differing attribute values. Otherwise, only non-last
    guidances need retain_graph=True.
    """
    scheduler = EDMNoiseScheduler()
    retain_last = create_graph or retain_all
    if config_name == "two_data":
        mask1 = torch.zeros(shape, dtype=torch.bool, device=device)
        mask2 = torch.zeros(shape, dtype=torch.bool, device=device)
        if len(shape) == 3:
            mask1[:, :, 0] = True
            mask2[:, :, 5] = True
        elif len(shape) == 4:
            mask1[:, :, 2, 3] = True
            mask2[:, :, 5, 1] = True
        else:
            mask1[:, :, 1, 2, 1] = True
            mask2[:, :, 0, 3, 2] = True
        y = make_input(shape, seed=seed, device=device)
        g1 = DataConsistencyDPSGuidance(
            mask=mask1,
            y=y,
            std_y=0.1,
            gamma=1.0,
            sigma_fn=scheduler.sigma,
            alpha_fn=scheduler.alpha,
            retain_graph=True,
            create_graph=create_graph,
        )
        g2 = DataConsistencyDPSGuidance(
            mask=mask2,
            y=y,
            std_y=0.1,
            gamma=1.0,
            sigma_fn=scheduler.sigma,
            alpha_fn=scheduler.alpha,
            retain_graph=retain_last,
            create_graph=create_graph,
        )
        return [g1, g2]
    elif config_name == "data_and_model":
        mask = _make_mask(shape, device)
        y_data = make_input(shape, seed=seed, device=device)
        g1 = DataConsistencyDPSGuidance(
            mask=mask,
            y=y_data,
            std_y=0.1,
            gamma=1.0,
            sigma_fn=scheduler.sigma,
            alpha_fn=scheduler.alpha,
            retain_graph=True,
            create_graph=create_graph,
        )
        obs_shape = (*shape[:1], 1, *shape[2:])
        y_model = make_input(obs_shape, seed=seed + 1, device=device)
        g2 = ModelConsistencyDPSGuidance(
            observation_operator=_nonlinear_obs_op,
            y=y_model,
            std_y=0.1,
            gamma=0.5,
            sigma_fn=scheduler.sigma,
            alpha_fn=scheduler.alpha,
            retain_graph=retain_last,
            create_graph=create_graph,
        )
        return [g1, g2]
    raise ValueError(f"Unknown multi-guidance config: {config_name}")


def _make_dps_score_predictor(
    predictor_cls,
    predictor_kwargs,
    guidances,
    device,
    seed=0,
):
    """Build a DPSScorePredictor from a model and guidances."""
    scheduler = EDMNoiseScheduler()
    model = instantiate_model_deterministic(
        predictor_cls,
        seed=seed,
        **predictor_kwargs,
    ).to(device)
    return DPSScorePredictor(
        x0_predictor=model,
        x0_to_score_fn=scheduler.x0_to_score,
        guidances=guidances,
    ), model


# =============================================================================
# Constructor Tests
# =============================================================================


class TestDPSScorePredictorConstructor:
    """Tests for DPSScorePredictor constructor."""

    def test_single_guidance_stored_as_list(self):
        def pred(x, t):
            return x * 0.9

        def score_fn(x0, x, t):
            return (x0 - x) / t.view(-1, 1) ** 2

        g = DataConsistencyDPSGuidance(
            mask=torch.ones(1, 3, 8, dtype=torch.bool),
            y=torch.randn(1, 3, 8),
            std_y=0.1,
        )
        dps = DPSScorePredictor(
            x0_predictor=pred,
            x0_to_score_fn=score_fn,
            guidances=g,
        )
        assert dps.x0_predictor is pred
        assert dps.x0_to_score_fn is score_fn
        assert isinstance(dps.guidances, list)
        assert len(dps.guidances) == 1

    def test_multiple_guidances_stored_as_list(self):
        def pred(x, t):
            return x * 0.9

        def score_fn(x0, x, t):
            return (x0 - x) / t.view(-1, 1) ** 2

        g1 = DataConsistencyDPSGuidance(
            mask=torch.ones(1, 3, 8, dtype=torch.bool),
            y=torch.randn(1, 3, 8),
            std_y=0.1,
            retain_graph=True,
        )
        g2 = DataConsistencyDPSGuidance(
            mask=torch.ones(1, 3, 8, dtype=torch.bool),
            y=torch.randn(1, 3, 8),
            std_y=0.1,
        )
        dps = DPSScorePredictor(
            x0_predictor=pred,
            x0_to_score_fn=score_fn,
            guidances=[g1, g2],
        )
        assert len(dps.guidances) == 2


class TestDataConsistencyDPSGuidanceConstructor:
    """Tests for DataConsistencyDPSGuidance constructor."""

    def test_default_attributes(self):
        mask = torch.ones(1, 3, 8, 8, dtype=torch.bool)
        y = torch.randn(1, 3, 8, 8)
        g = DataConsistencyDPSGuidance(mask=mask, y=y, std_y=0.1)
        assert g.std_y == pytest.approx(0.1)
        assert g.gamma == pytest.approx(0.0)
        assert g.retain_graph is False
        assert g.create_graph is False

    def test_custom_attributes(self):
        scheduler = EDMNoiseScheduler()
        mask = torch.ones(1, 3, 8, 8, dtype=torch.bool)
        y = torch.randn(1, 3, 8, 8)
        g = DataConsistencyDPSGuidance(
            mask=mask,
            y=y,
            std_y=0.5,
            norm=1,
            gamma=2.0,
            sigma_fn=scheduler.sigma,
            alpha_fn=scheduler.alpha,
            retain_graph=True,
            create_graph=True,
        )
        assert g.std_y == pytest.approx(0.5)
        assert g.gamma == pytest.approx(2.0)
        assert g.norm == 1
        assert g.retain_graph is True
        assert g.create_graph is True

    def test_gamma_requires_sigma_fn(self):
        mask = torch.ones(1, 3, 8, 8, dtype=torch.bool)
        y = torch.randn(1, 3, 8, 8)
        with pytest.raises(ValueError, match="sigma_fn"):
            DataConsistencyDPSGuidance(mask=mask, y=y, std_y=0.1, gamma=1.0)


class TestModelConsistencyDPSGuidanceConstructor:
    """Tests for ModelConsistencyDPSGuidance constructor."""

    def test_default_attributes(self):
        y = torch.randn(1, 1, 8, 8)
        g = ModelConsistencyDPSGuidance(
            observation_operator=_nonlinear_obs_op,
            y=y,
            std_y=0.1,
        )
        assert g.std_y == pytest.approx(0.1)
        assert g.gamma == pytest.approx(0.0)
        assert g.retain_graph is False
        assert g.create_graph is False

    def test_custom_attributes(self):
        scheduler = EDMNoiseScheduler()
        y = torch.randn(1, 1, 8, 8)
        g = ModelConsistencyDPSGuidance(
            observation_operator=_nonlinear_obs_op,
            y=y,
            std_y=0.5,
            norm=1,
            gamma=2.0,
            sigma_fn=scheduler.sigma,
            alpha_fn=scheduler.alpha,
            retain_graph=True,
            create_graph=True,
        )
        assert g.std_y == pytest.approx(0.5)
        assert g.gamma == pytest.approx(2.0)
        assert g.norm == 1
        assert g.retain_graph is True
        assert g.create_graph is True

    def test_gamma_requires_sigma_fn(self):
        y = torch.randn(1, 1, 8, 8)
        with pytest.raises(ValueError, match="sigma_fn"):
            ModelConsistencyDPSGuidance(
                observation_operator=_nonlinear_obs_op,
                y=y,
                std_y=0.1,
                gamma=1.0,
            )


# =============================================================================
# Non-Regression Tests (single guidance)
# =============================================================================


@pytest.mark.parametrize(
    "guidance_config,guidance_desc",
    GUIDANCE_CONFIGS,
    ids=[c[0] for c in GUIDANCE_CONFIGS],
)
@pytest.mark.parametrize(
    "spatial_name,shape,predictor_cls,predictor_kwargs",
    SPATIAL_CONFIGS,
    ids=[c[0] for c in SPATIAL_CONFIGS],
)
class TestGuidanceNonRegression:
    """Non-regression tests for all single-guidance configurations."""

    def test_guidance_call(
        self,
        deterministic_settings,
        device,
        tolerances,
        guidance_config,
        guidance_desc,
        spatial_name,
        shape,
        predictor_cls,
        predictor_kwargs,
    ):
        g = _make_guidance(guidance_config, shape, device, seed=310)
        x = make_input(shape, seed=311, device=device).requires_grad_(True)
        t = torch.tensor([1.0] * shape[0], device=device)
        x_0 = x * 0.9
        out = g(x, t, x_0)
        assert out.shape == shape
        assert torch.isfinite(out).all()

        ref_file = f"{REF_PREFIX}{guidance_config}_{spatial_name}.pth"
        ref = load_or_create_reference(ref_file, lambda: {"out": out.cpu()})
        compare_outputs(out, ref["out"], **tolerances)

    def test_dps_score_predictor(
        self,
        deterministic_settings,
        device,
        tolerances,
        guidance_config,
        guidance_desc,
        spatial_name,
        shape,
        predictor_cls,
        predictor_kwargs,
    ):
        g = _make_guidance(guidance_config, shape, device, seed=410)
        dps, _ = _make_dps_score_predictor(
            predictor_cls,
            predictor_kwargs,
            g,
            device,
        )
        x = make_input(shape, seed=411, device=device)
        t = torch.tensor([1.0] * shape[0], device=device)
        out = dps(x, t)
        assert out.shape == shape
        assert torch.isfinite(out).all()

        ref_file = f"{REF_PREFIX}dps_{guidance_config}_{spatial_name}.pth"
        ref = load_or_create_reference(ref_file, lambda: {"out": out.cpu()})
        compare_outputs(out, ref["out"], **tolerances)


# =============================================================================
# Multi-Guidance Non-Regression Tests
# =============================================================================


@pytest.mark.parametrize(
    "multi_config,multi_desc",
    MULTI_GUIDANCE_CONFIGS,
    ids=[c[0] for c in MULTI_GUIDANCE_CONFIGS],
)
@pytest.mark.parametrize(
    "spatial_name,shape,predictor_cls,predictor_kwargs",
    SPATIAL_CONFIGS,
    ids=[c[0] for c in SPATIAL_CONFIGS],
)
class TestMultiGuidanceNonRegression:
    """Non-regression tests for multi-guidance DPSScorePredictor."""

    def test_non_regression(
        self,
        deterministic_settings,
        device,
        tolerances,
        multi_config,
        multi_desc,
        spatial_name,
        shape,
        predictor_cls,
        predictor_kwargs,
    ):
        guidances = _make_multi_guidance(multi_config, shape, device, seed=420)
        dps, _ = _make_dps_score_predictor(
            predictor_cls,
            predictor_kwargs,
            guidances,
            device,
        )
        x = make_input(shape, seed=421, device=device)
        t = torch.tensor([1.0] * shape[0], device=device)
        out = dps(x, t)
        assert out.shape == shape
        assert torch.isfinite(out).all()

        ref_file = f"{REF_PREFIX}multi_{multi_config}_{spatial_name}.pth"
        ref = load_or_create_reference(ref_file, lambda: {"out": out.cpu()})
        compare_outputs(out, ref["out"], **tolerances)


# =============================================================================
# Gradient Flow Tests
# =============================================================================


@pytest.mark.parametrize(
    "multi_config,multi_desc",
    MULTI_GUIDANCE_CONFIGS,
    ids=[c[0] for c in MULTI_GUIDANCE_CONFIGS],
)
@pytest.mark.parametrize(
    "spatial_name,shape,predictor_cls,predictor_kwargs",
    SPATIAL_CONFIGS,
    ids=[c[0] for c in SPATIAL_CONFIGS],
)
class TestGradientFlow:
    """Tests that gradients flow through DPSScorePredictor to model parameters."""

    def test_backward_through_dps(
        self,
        device,
        multi_config,
        multi_desc,
        spatial_name,
        shape,
        predictor_cls,
        predictor_kwargs,
    ):
        """Backward through the DPS output reaches model parameters.

        Requires create_graph=True on all guidances so that autograd.grad
        creates a differentiable graph, and retain_graph=True on all guidances
        except the last so that multiple autograd.grad calls don't destroy the
        graph.
        """
        guidances = _make_multi_guidance(
            multi_config,
            shape,
            device,
            seed=440,
            create_graph=True,
        )
        dps, model = _make_dps_score_predictor(
            predictor_cls,
            predictor_kwargs,
            guidances,
            device,
        )
        x = make_input(shape, seed=441, device=device)
        t = torch.tensor([1.0] * shape[0], device=device)
        out = dps(x, t)
        loss = out.sum()
        loss.backward()

        has_grad = any(
            p.grad is not None and not torch.isnan(p.grad).any()
            for p in model.parameters()
        )
        assert has_grad


# =============================================================================
# Compile Tests
# =============================================================================


@pytest.mark.parametrize(
    "guidance_config,guidance_desc",
    GUIDANCE_CONFIGS,
    ids=[c[0] for c in GUIDANCE_CONFIGS],
)
@pytest.mark.parametrize(
    "spatial_name,shape,predictor_cls,predictor_kwargs",
    SPATIAL_CONFIGS,
    ids=[c[0] for c in SPATIAL_CONFIGS],
)
class TestCompileSingleGuidance:
    """torch.compile tests for DPSScorePredictor with single guidance."""

    def test_compile(
        self,
        device,
        guidance_config,
        guidance_desc,
        spatial_name,
        shape,
        predictor_cls,
        predictor_kwargs,
    ):
        torch._dynamo.config.error_on_recompile = True

        g = _make_guidance(guidance_config, shape, device, seed=500)
        dps, _ = _make_dps_score_predictor(
            predictor_cls,
            predictor_kwargs,
            g,
            device,
        )
        x = make_input(shape, seed=501, device=device)
        t = torch.tensor([1.0] * shape[0], device=device)

        # fullgraph=False because DPSScorePredictor uses
        # x.detach().requires_grad_(True) which is incompatible with fullgraph
        compiled_fn = torch.compile(lambda xi, ti: dps(xi, ti))

        out_eager = dps(x, t)
        out_compiled = compiled_fn(x, t)
        torch.testing.assert_close(out_eager, out_compiled, atol=1e-4, rtol=1e-4)

        out_compiled_2 = compiled_fn(x, t)
        torch.testing.assert_close(out_eager, out_compiled_2, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize(
    "multi_config,multi_desc",
    MULTI_GUIDANCE_CONFIGS,
    ids=[c[0] for c in MULTI_GUIDANCE_CONFIGS],
)
@pytest.mark.parametrize(
    "spatial_name,shape,predictor_cls,predictor_kwargs",
    SPATIAL_CONFIGS,
    ids=[c[0] for c in SPATIAL_CONFIGS],
)
class TestCompileMultiGuidance:
    """torch.compile tests for DPSScorePredictor with multiple guidances."""

    def test_compile(
        self,
        device,
        monkeypatch,
        multi_config,
        multi_desc,
        spatial_name,
        shape,
        predictor_cls,
        predictor_kwargs,
    ):
        torch._dynamo.config.error_on_recompile = True
        monkeypatch.setattr(_functorch_config, "donated_buffer", False)

        guidances = _make_multi_guidance(
            multi_config,
            shape,
            device,
            seed=510,
            retain_all=True,
        )
        dps, _ = _make_dps_score_predictor(
            predictor_cls,
            predictor_kwargs,
            guidances,
            device,
        )
        x = make_input(shape, seed=511, device=device)
        t = torch.tensor([1.0] * shape[0], device=device)

        compiled_fn = torch.compile(lambda xi, ti: dps(xi, ti))

        out_eager = dps(x, t)
        out_compiled = compiled_fn(x, t)
        torch.testing.assert_close(out_eager, out_compiled, atol=1e-4, rtol=1e-4)

        out_compiled_2 = compiled_fn(x, t)
        torch.testing.assert_close(out_eager, out_compiled_2, atol=1e-4, rtol=1e-4)
