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

r"""Check continuous inverse SNR and TrigFlow coordinates."""

import pytest
import torch

from physicsnemo.diffusion import (
    DiffusionExtraTimes,
    preconditioners,
)
from physicsnemo.diffusion import noise_schedulers as schedules

from .helpers import StructuredDiffusionModel

CONFIGS = (
    ("EDMNoiseScheduler", {}),
    ("EDMLogUniformNoiseScheduler", {}),
    ("VENoiseScheduler", {}),
    ("VPNoiseScheduler", {}),
    ("VPNoiseScheduler", {"beta_d": 0.0}),
    ("IDDPMNoiseScheduler", {"M": 32}),
    ("RectifiedFlowNoiseScheduler", {}),
    ("TrigFlowNoiseScheduler", {"sigma_data": 0.5}),
    ("TrigFlowNoiseScheduler", {"sigma_data": 1.4}),
)
IDS = (
    "edm",
    "edm-uniform",
    "ve",
    "vp",
    "vp-constant",
    "iddpm",
    "rf",
    "trig",
    "trig-scaled",
)


@pytest.mark.parametrize("name,kwargs", CONFIGS, ids=IDS)
class TestConsistency:
    def test_snr_inverse(self, name, kwargs, device):
        scheduler = getattr(schedules, name)(**kwargs)
        t = torch.tensor([0.1, 0.4, 0.8], dtype=torch.float64, device=device)
        recovered, tangent = torch.func.jvp(
            lambda t: scheduler.snr_inv(scheduler.snr(t)),
            (t,),
            (torch.ones_like(t),),
        )
        torch.testing.assert_close(recovered, t)
        torch.testing.assert_close(tangent, torch.ones_like(t))

    def test_coefficient_derivatives(self, name, kwargs, device):
        scheduler = getattr(schedules, name)(**kwargs)
        t = torch.tensor([0.1, 0.4, 0.8], dtype=torch.float64, device=device)
        for coefficient, derivative in (
            (scheduler.alpha, scheduler.alpha_dot),
            (scheduler.sigma, scheduler.sigma_dot),
        ):
            _, actual = torch.func.jvp(coefficient, (t,), (torch.ones_like(t),))
            torch.testing.assert_close(actual, derivative(t))


@pytest.mark.parametrize("sigma_data", (0.5, 1.4))
class TestGradientFlow:
    def test_trigflow(self, sigma_data):
        scheduler = schedules.TrigFlowNoiseScheduler(sigma_data=sigma_data)
        model = preconditioners.TrigFlowPreconditioner(
            StructuredDiffusionModel(), sigma_data=sigma_data
        ).double()
        x = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
        t = torch.tensor([0.4, 0.8], dtype=torch.float64, requires_grad=True)

        def forward(x, t):
            prediction, extras = model(
                x, (t, DiffusionExtraTimes(r=t / 2, batch_size=[2]))
            )
            return (
                prediction,
                extras.logvar,
                scheduler.snr_inv(scheduler.snr(t)),
            )

        assert torch.autograd.gradcheck(forward, (x, t))


class TestNumerics:
    @pytest.mark.parametrize(
        "dtype,ratio",
        (
            (torch.float16, 0.003),
            (torch.float32, 1e-20),
            (torch.float64, 1e-200),
        ),
    )
    def test_vp_small_ratio(self, dtype, ratio, device):
        scheduler = schedules.VPNoiseScheduler()
        q = torch.tensor([ratio], dtype=dtype, device=device, requires_grad=True)
        result = scheduler.snr_inv(q)
        (gradient,) = torch.autograd.grad(result.sum(), (q,))
        q64 = q.detach().double()
        a = torch.logaddexp(torch.zeros_like(q64), -2 * q64.log())
        root = (scheduler.beta_min**2 + 2 * scheduler.beta_d * a).sqrt()
        expected = 2 * a / (scheduler.beta_min + root)
        expected_gradient = -2 / (q64 * (1 + q64.square()) * root)
        torch.testing.assert_close(result, expected.to(dtype))
        torch.testing.assert_close(gradient, expected_gradient.to(dtype))


@pytest.mark.parametrize("sigma_data", (0.5, 1.4))
class TestConstructor:
    def test_trigflow(self, sigma_data, device):
        scheduler = schedules.TrigFlowNoiseScheduler(
            sigma_data=sigma_data, t_min=0.1, t_max=1.3
        )
        t = scheduler.sample_time(128, device=device, dtype=torch.float64)
        assert torch.all((t >= 0.1) & (t <= 1.3))
        grid = scheduler.timesteps(8, device=device)
        assert grid.shape == (9,)
        torch.testing.assert_close(grid, torch.linspace(1.3, 0, 9, device=device))
        model = preconditioners.TrigFlowPreconditioner(
            StructuredDiffusionModel(), sigma_data=sigma_data
        ).to(device)
        x = torch.randn(2, 3, device=device)
        t = torch.tensor([0.4, 0.8], device=device)
        result, extras = model(x, t)
        raw, raw_extras = model.model(x / sigma_data, t)
        expected = t.cos()[:, None] * x - sigma_data * t.sin()[:, None] * raw
        torch.testing.assert_close(result, expected)
        torch.testing.assert_close(extras.logvar, raw_extras.logvar)
        torch.testing.assert_close(
            scheduler.loss_weight(t), 1 / (sigma_data * t.sin()) ** 2
        )

    def test_zero_time(self, sigma_data):
        scheduler = schedules.TrigFlowNoiseScheduler(sigma_data=sigma_data)
        t = torch.zeros(2)
        torch.testing.assert_close(scheduler.alpha(t), torch.ones_like(t))
        torch.testing.assert_close(scheduler.sigma(t), t)
        torch.testing.assert_close(scheduler.alpha_dot(t), t)
        torch.testing.assert_close(
            scheduler.sigma_dot(t), torch.full_like(t, sigma_data)
        )


@pytest.mark.parametrize("name,kwargs", CONFIGS, ids=IDS)
class TestCompile:
    def test_inverse(self, name, kwargs, device):
        scheduler = getattr(schedules, name)(**kwargs)
        t = torch.tensor([0.2, 0.7], device=device, requires_grad=True)
        fn = torch.compile(
            lambda t: scheduler.snr_inv(scheduler.snr(t)),
            backend="aot_eager",
            fullgraph=True,
        )
        result = fn(t)
        with torch._dynamo.config.patch(error_on_recompile=True):
            torch.testing.assert_close(fn(t), result)
        torch.testing.assert_close(result, t)
        (gradient,) = torch.autograd.grad(result.sum(), (t,))
        torch.testing.assert_close(gradient, torch.ones_like(t))


class TestDiagnostics:
    @pytest.mark.parametrize(
        "kwargs",
        (
            {"sigma_data": 0},
            {"sigma_data": float("nan")},
            {"t_min": -0.1},
            {"t_max": 1.6},
            {"t_min": 0.7, "t_max": 0.5},
        ),
    )
    def test_invalid_scheduler(self, kwargs):
        with pytest.raises(ValueError):
            schedules.TrigFlowNoiseScheduler(**kwargs)

    @pytest.mark.parametrize("sigma_data", (0, -1, float("nan")))
    def test_invalid_preconditioner(self, sigma_data):
        with pytest.raises(ValueError):
            preconditioners.TrigFlowPreconditioner(
                StructuredDiffusionModel(), sigma_data
            )
