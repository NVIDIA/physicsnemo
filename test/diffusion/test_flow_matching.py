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

"""Tests for FlowMatchingNoiseScheduler.

FlowMatchingNoiseScheduler is a LinearGaussianNoiseScheduler subclass, so it
is also covered by the generic non-regression suite in
test_noise_schedulers.py through the scheduler's inherited methods (add_noise,
score conversions, etc. — exercised indirectly via _MinimalScheduler-style
checks). This module focuses on behavior specific to the flow matching
(rectified flow) path: the closed-form velocity field, the t=0/t=1
singularities documented on the scheduler, and get_denoiser's
velocity_predictor support.
"""

import pytest
import torch

from physicsnemo.diffusion.noise_schedulers import (
    FlowMatchingNoiseScheduler,
    LinearGaussianNoiseScheduler,
    NoiseScheduler,
)
from physicsnemo.diffusion.samplers import sample

from .helpers import Conv2dX0Predictor, instantiate_model_deterministic, make_input

BATCH = 4
NUM_STEPS = 8
SHAPE = (BATCH, 3, 8, 6)


# =============================================================================
# Constructor Tests
# =============================================================================


class TestFlowMatchingNoiseSchedulerConstructor:
    """Tests for FlowMatchingNoiseScheduler constructor and attributes."""

    def test_default_attributes(self):
        s = FlowMatchingNoiseScheduler()
        assert s.t_min == pytest.approx(0.0)
        assert s.t_max == pytest.approx(1.0)

    def test_custom_attributes(self):
        s = FlowMatchingNoiseScheduler(t_min=1e-3, t_max=0.999)
        assert s.t_min == pytest.approx(1e-3)
        assert s.t_max == pytest.approx(0.999)

    def test_is_noise_scheduler(self):
        assert isinstance(FlowMatchingNoiseScheduler(), NoiseScheduler)

    def test_is_linear_gaussian(self):
        assert isinstance(FlowMatchingNoiseScheduler(), LinearGaussianNoiseScheduler)

    @pytest.mark.parametrize(
        "t_min,t_max",
        [(0.5, 0.5), (0.6, 0.4), (-0.1, 1.0), (0.0, 1.1)],
        ids=["equal", "reversed", "negative_t_min", "t_max_above_1"],
    )
    def test_invalid_time_range(self, t_min, t_max):
        with pytest.raises(ValueError, match="t_min and t_max"):
            FlowMatchingNoiseScheduler(t_min=t_min, t_max=t_max)


# =============================================================================
# Analytic Coefficient Tests
# =============================================================================


class TestFlowMatchingCoefficients:
    """Closed-form checks for the linear interpolation path coefficients."""

    def test_alpha_and_sigma_sum_to_one(self, device):
        s = FlowMatchingNoiseScheduler()
        t = torch.linspace(0.0, 1.0, 11, device=device)
        torch.testing.assert_close(s.alpha(t) + s.sigma(t), torch.ones_like(t))

    def test_sigma_is_identity(self, device):
        s = FlowMatchingNoiseScheduler()
        t = torch.linspace(0.0, 1.0, 11, device=device)
        torch.testing.assert_close(s.sigma(t), t)

    def test_sigma_inv_is_identity(self, device):
        s = FlowMatchingNoiseScheduler()
        sigma_val = torch.linspace(0.0, 1.0, 11, device=device)
        torch.testing.assert_close(s.sigma_inv(sigma_val), sigma_val)

    def test_alpha_is_one_minus_t(self, device):
        s = FlowMatchingNoiseScheduler()
        t = torch.linspace(0.0, 1.0, 11, device=device)
        torch.testing.assert_close(s.alpha(t), 1 - t)

    def test_sigma_dot_is_one(self, device):
        s = FlowMatchingNoiseScheduler()
        t = torch.linspace(0.0, 1.0, 11, device=device)
        torch.testing.assert_close(s.sigma_dot(t), torch.ones_like(t))

    def test_alpha_dot_is_minus_one(self, device):
        s = FlowMatchingNoiseScheduler()
        t = torch.linspace(0.0, 1.0, 11, device=device)
        torch.testing.assert_close(s.alpha_dot(t), -torch.ones_like(t))

    def test_loss_weight_is_one(self, device):
        s = FlowMatchingNoiseScheduler()
        t = torch.linspace(0.0, 1.0, 11, device=device)
        torch.testing.assert_close(s.loss_weight(t), torch.ones_like(t))


# =============================================================================
# Time Sampling and Discretization Tests
# =============================================================================


class TestFlowMatchingTimeSampling:
    """Tests for timesteps() and sample_time()."""

    def test_timesteps_shape_and_endpoints(self, device):
        s = FlowMatchingNoiseScheduler()
        t_steps = s.timesteps(NUM_STEPS, device=device)
        assert t_steps.shape == (NUM_STEPS + 1,)
        assert t_steps[0].item() == pytest.approx(1.0)
        assert t_steps[-1].item() == pytest.approx(0.0, abs=1e-7)

    def test_timesteps_are_decreasing(self, device):
        s = FlowMatchingNoiseScheduler()
        t_steps = s.timesteps(NUM_STEPS, device=device)
        diffs = t_steps[:-1] - t_steps[1:]
        assert (diffs >= -1e-7).all()

    def test_timesteps_respects_t_max(self, device):
        s = FlowMatchingNoiseScheduler(t_max=0.999)
        t_steps = s.timesteps(NUM_STEPS, device=device)
        assert t_steps[0].item() == pytest.approx(0.999)

    def test_sample_time_bounds(self, device):
        s = FlowMatchingNoiseScheduler(t_min=0.1, t_max=0.9)
        t = s.sample_time(1000, device=device)
        assert t.shape == (1000,)
        assert (t >= 0.1).all()
        assert (t <= 0.9).all()


# =============================================================================
# add_noise / init_latents Tests
# =============================================================================


class TestFlowMatchingSpatialMethods:
    """Tests for add_noise() and init_latents()."""

    def test_add_noise_shape_and_interpolation(self, device):
        s = FlowMatchingNoiseScheduler()
        x0 = make_input(SHAPE, seed=1, device=device)
        # At t=0, x_t should equal x0 exactly (alpha=1, sigma=0).
        t0 = torch.zeros(BATCH, device=device)
        x_t0 = s.add_noise(x0, t0)
        assert x_t0.shape == SHAPE
        torch.testing.assert_close(x_t0, x0)

    def test_init_latents_shape(self, device):
        s = FlowMatchingNoiseScheduler()
        tN = torch.ones(BATCH, device=device)
        xN = s.init_latents(SHAPE[1:], tN, device=device)
        assert xN.shape == SHAPE


# =============================================================================
# get_denoiser Tests
# =============================================================================


class TestFlowMatchingGetDenoiser:
    """Tests for get_denoiser() with the closed-form flow matching RHS."""

    def test_validates_multiple_predictors(self, device):
        s = FlowMatchingNoiseScheduler()
        pred = lambda x, t: x  # noqa: E731
        with pytest.raises(ValueError, match="Exactly one"):
            s.get_denoiser(velocity_predictor=pred, x0_predictor=pred)
        with pytest.raises(ValueError, match="Exactly one"):
            s.get_denoiser()

    def test_validates_denoising_type(self, device):
        s = FlowMatchingNoiseScheduler()
        pred = lambda x, t: x  # noqa: E731
        with pytest.raises(ValueError, match="denoising_type"):
            s.get_denoiser(velocity_predictor=pred, denoising_type="bad")

    def test_sde_rejects_default_t_max(self, device):
        """SDE sampling is singular at t=1, so it must be rejected when
        t_max=1 (the default) instead of silently producing inf/nan."""
        s = FlowMatchingNoiseScheduler()
        pred = lambda x, t: x  # noqa: E731
        with pytest.raises(ValueError, match="t_max"):
            s.get_denoiser(velocity_predictor=pred, denoising_type="sde")

    def test_sde_accepts_t_max_below_one(self, device):
        s = FlowMatchingNoiseScheduler(t_max=0.999)
        pred = lambda x, t: x  # noqa: E731
        # Should not raise.
        s.get_denoiser(velocity_predictor=pred, denoising_type="sde")

    def test_velocity_predictor_ode_rhs_is_the_prediction(self, device):
        """For a velocity_predictor, the ODE RHS is the velocity itself."""
        s = FlowMatchingNoiseScheduler()
        x = make_input(SHAPE, seed=2, device=device)
        t = torch.full((BATCH,), 0.5, device=device)
        velocity_pred = lambda x, t: -x  # noqa: E731
        denoiser = s.get_denoiser(velocity_predictor=velocity_pred)
        torch.testing.assert_close(denoiser(x, t), velocity_pred(x, t))

    def test_x0_predictor_matches_velocity_predictor(self, device):
        """An x0-predictor and its equivalent velocity-predictor agree."""
        s = FlowMatchingNoiseScheduler()
        x = make_input(SHAPE, seed=3, device=device)
        t = torch.full((BATCH,), 0.5, device=device)

        def x0_pred(x, t):
            return x * 0.9

        def velocity_pred(x, t):
            x0 = x0_pred(x, t)
            return s.x0_to_velocity(x0, x, t)

        denoiser_x0 = s.get_denoiser(x0_predictor=x0_pred)
        denoiser_v = s.get_denoiser(velocity_predictor=velocity_pred)
        torch.testing.assert_close(denoiser_x0(x, t), denoiser_v(x, t))

    def test_epsilon_predictor_matches_velocity_predictor(self, device):
        """An epsilon-predictor and its equivalent velocity-predictor agree.

        t_max is kept below 1 since the epsilon parameterization is singular
        at t=1 (see FlowMatchingNoiseScheduler.get_denoiser docstring).
        """
        s = FlowMatchingNoiseScheduler(t_max=0.999)
        x = make_input(SHAPE, seed=4, device=device)
        t = torch.full((BATCH,), 0.5, device=device)

        def eps_pred(x, t):
            return x * 0.1

        def velocity_pred(x, t):
            eps = eps_pred(x, t)
            x0 = s.epsilon_to_x0(eps, x, t)
            return s.x0_to_velocity(x0, x, t)

        denoiser_eps = s.get_denoiser(epsilon_predictor=eps_pred)
        denoiser_v = s.get_denoiser(velocity_predictor=velocity_pred)
        torch.testing.assert_close(denoiser_eps(x, t), denoiser_v(x, t))

    def test_score_predictor_matches_velocity_predictor(self, device):
        """A score-predictor and its equivalent velocity-predictor agree."""
        s = FlowMatchingNoiseScheduler(t_max=0.999)
        x = make_input(SHAPE, seed=5, device=device)
        t = torch.full((BATCH,), 0.5, device=device)

        def score_pred(x, t):
            return -x * 0.2

        def velocity_pred(x, t):
            score = score_pred(x, t)
            x0 = s.score_to_x0(score, x, t)
            return s.x0_to_velocity(x0, x, t)

        denoiser_score = s.get_denoiser(score_predictor=score_pred)
        denoiser_v = s.get_denoiser(velocity_predictor=velocity_pred)
        torch.testing.assert_close(denoiser_score(x, t), denoiser_v(x, t))

    def test_sde_denoiser_shape(self, device):
        s = FlowMatchingNoiseScheduler(t_max=0.999)
        x = make_input(SHAPE, seed=6, device=device)
        t = torch.full((BATCH,), 0.5, device=device)
        velocity_pred = lambda x, t: -x  # noqa: E731
        denoiser = s.get_denoiser(
            velocity_predictor=velocity_pred, denoising_type="sde"
        )
        out = denoiser(x, t)
        assert out.shape == x.shape
        assert torch.isfinite(out).all()

    def test_ode_regular_at_t_equals_one_for_velocity(self, device):
        """Unlike epsilon/score, velocity and x0 denoisers are regular at t=1."""
        s = FlowMatchingNoiseScheduler()
        x = make_input(SHAPE, seed=7, device=device)
        t = torch.ones(BATCH, device=device)
        velocity_pred = lambda x, t: -x  # noqa: E731
        denoiser = s.get_denoiser(velocity_predictor=velocity_pred)
        out = denoiser(x, t)
        assert torch.isfinite(out).all()


# =============================================================================
# End-to-End Sampling Test
# =============================================================================


class TestFlowMatchingSampling:
    """End-to-end smoke test: velocity-predictor -> denoiser -> sampler."""

    def test_sample_from_velocity_predictor(self, device):
        s = FlowMatchingNoiseScheduler()
        model = instantiate_model_deterministic(
            Conv2dX0Predictor, seed=0, channels=3
        ).to(device)
        denoiser = s.get_denoiser(velocity_predictor=model)
        t_steps = s.timesteps(NUM_STEPS, device=device)
        xN = s.init_latents(SHAPE[1:], t_steps[0].expand(BATCH), device=device)
        with torch.no_grad():
            samples = sample(denoiser, xN, s, num_steps=NUM_STEPS)
        assert samples.shape == SHAPE
        assert torch.isfinite(samples).all()
