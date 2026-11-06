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

r"""Check training-only schedule translation against source operations."""

import pytest
import torch

pytest.importorskip("fastgen")

from physicsnemo.diffusion import noise_schedulers as schedules
from physicsnemo.diffusion.noise_schedulers._fastgen import _FastGenNoiseScheduler

from .helpers import CustomGaussianScheduler

CONFIGS = (
    ("EDMNoiseScheduler", {}),
    ("VENoiseScheduler", {}),
    ("VPNoiseScheduler", {}),
    ("RectifiedFlowNoiseScheduler", {}),
    ("TrigFlowNoiseScheduler", {"sigma_data": 1.4}),
)
IDS = ("edm", "ve", "vp", "rf", "trig")


@pytest.mark.parametrize("name,kwargs", CONFIGS, ids=IDS)
class TestConsistency:
    def test_mathematics(self, name, kwargs, device):
        source = getattr(schedules, name)(**kwargs)
        adapter = _FastGenNoiseScheduler(source)
        x = torch.randn(2, 3, 5, device=device, dtype=torch.float64)
        eps = torch.randn_like(x)
        t = torch.tensor([0.2, 0.7], device=device, dtype=x.dtype)
        t_bc = t[:, None, None]
        xt = source.alpha(t_bc) * x + source.sigma(t_bc) * eps
        torch.testing.assert_close(adapter.forward_process(x, eps, t), xt)
        torch.testing.assert_close(
            adapter.cond_velocity(x, eps, t),
            source.alpha_dot(t_bc) * x + source.sigma_dot(t_bc) * eps,
        )
        torch.testing.assert_close(adapter.x0_to_eps(xt, x, t), eps)
        torch.testing.assert_close(adapter.eps_to_x0(xt, eps, t), x)
        flow = source.x0_to_flow(x, xt, t)
        torch.testing.assert_close(adapter.x0_to_flow(xt, x, t), flow)
        torch.testing.assert_close(adapter.flow_to_x0(xt, flow, t), x)
        torch.testing.assert_close(adapter.rescale_t(t), t)
        torch.testing.assert_close(adapter.sqrt_snr(t), source.snr(t))

    def test_native_sampling(self, name, kwargs, device):
        source = getattr(schedules, name)(**kwargs)
        adapter = _FastGenNoiseScheduler(source)
        torch.manual_seed(7)
        expected = source.sample_time(8, device=device, dtype=adapter.t_precision)
        torch.manual_seed(7)
        torch.testing.assert_close(adapter.sample_t(8, device=device), expected)

    def test_live_methods(self, name, kwargs, monkeypatch):
        source = getattr(schedules, name)(**kwargs)
        adapter = _FastGenNoiseScheduler(source)
        monkeypatch.setattr(source, "alpha", lambda t: torch.full_like(t, 0.75))
        torch.testing.assert_close(adapter.alpha(torch.ones(2)), torch.full((2,), 0.75))


@pytest.mark.parametrize("name,kwargs", CONFIGS, ids=IDS)
class TestGradientFlow:
    def test_snr_coordinates(self, name, kwargs):
        adapter = _FastGenNoiseScheduler(getattr(schedules, name)(**kwargs))
        t = torch.tensor([0.2, 0.7], dtype=torch.float64, requires_grad=True)

        def fn(t):
            return adapter.sqrt_snr_to_t(adapter.sqrt_snr(t))

        assert torch.autograd.gradcheck(fn, (t,))
        _, tangent = torch.func.jvp(fn, (t,), (torch.ones_like(t),))
        torch.testing.assert_close(tangent, torch.ones_like(t))


@pytest.mark.parametrize("name,kwargs", CONFIGS, ids=IDS)
class TestCompile:
    def test_math(self, name, kwargs):
        adapter = _FastGenNoiseScheduler(getattr(schedules, name)(**kwargs))
        x = torch.randn(2, 3, requires_grad=True)
        eps = torch.randn_like(x)
        t = torch.tensor([0.2, 0.7], requires_grad=True)
        fn = torch.compile(adapter.forward_process, backend="aot_eager", fullgraph=True)
        expected = adapter.forward_process(x, eps, t)
        actual = fn(x, eps, t)
        with torch._dynamo.config.patch(error_on_recompile=True):
            torch.testing.assert_close(fn(x, eps, t), expected)
        torch.testing.assert_close(actual, expected)
        expected_grad = torch.autograd.grad(expected.sum(), (x, t))
        actual_grad = torch.autograd.grad(actual.sum(), (x, t))
        for actual, expected in zip(actual_grad, expected_grad):
            torch.testing.assert_close(actual, expected)


class TestConstructor:
    def test_custom_scheduler(self):
        source = CustomGaussianScheduler()
        adapter = _FastGenNoiseScheduler(source)
        t = torch.tensor([-0.5, 0.7], dtype=torch.float64, requires_grad=True)
        torch.testing.assert_close(adapter.sqrt_snr_to_t(adapter.sqrt_snr(t)), t)
        with pytest.raises(ValueError, match="training time_bounds"):
            _ = adapter.max_t
        adapter._configure_training(time_bounds=(-1.0, 2.0))
        assert adapter.min_t == -1.0
        assert adapter.max_t == 2.0
        torch.manual_seed(41)
        expected = source.sample_time(4, dtype=torch.float64)
        torch.manual_seed(41)
        torch.testing.assert_close(adapter.sample_t(4), expected)

    def test_source_bounds_intersection(self):
        source = schedules.EDMNoiseScheduler(sigma_min=0.1, sigma_max=2)
        adapter = _FastGenNoiseScheduler(source)
        adapter._configure_training(sample_bounds=(0, 4))
        times = adapter.sample_t(4096, "uniform")
        assert torch.all((times >= source.sigma_min) & (times <= source.sigma_max))

    def test_live_bounds(self):
        source = schedules.EDMNoiseScheduler(sigma_max=80)
        adapter = _FastGenNoiseScheduler(source)
        source.sigma_max = 40
        assert adapter.max_t == adapter.max_sigma == 40
        t = adapter.sample_t(1024, "lognormal", min_t=0.1, max_t=100)
        assert torch.all((t >= 0.1) & (t <= 40))
        assert not torch.any(t == 40)

    def test_training_grid(self, monkeypatch):
        source = schedules.RectifiedFlowNoiseScheduler()
        adapter = _FastGenNoiseScheduler(source)
        monkeypatch.setattr(
            source,
            "timesteps",
            lambda *args, **kwargs: pytest.fail("Inference grid used"),
        )
        torch.testing.assert_close(
            adapter.get_t_list(1), torch.tensor([0.99, 0], dtype=torch.float64)
        )
        adapter._configure_training(training_times=torch.tensor([0.9, 0.4, 0]))
        assert adapter.num_steps == 2
        torch.testing.assert_close(
            adapter.get_t_list(2), torch.tensor([0.9, 0.4, 0], dtype=torch.float64)
        )

    def test_zero_velocity(self):
        adapter = _FastGenNoiseScheduler(schedules.RectifiedFlowNoiseScheduler())
        x = torch.randn(2, 3)
        eps = torch.randn_like(x)
        torch.testing.assert_close(
            adapter.cond_velocity(x, eps, torch.zeros(2)), eps - x
        )


class TestDiagnostics:
    @pytest.mark.parametrize(
        "bounds", ((float("nan"), 1.0), (0.0, float("inf")), (2.0, 1.0))
    )
    def test_invalid_bounds(self, bounds):
        adapter = _FastGenNoiseScheduler(schedules.RectifiedFlowNoiseScheduler())
        adapter._configure_training(sample_bounds=bounds)
        with pytest.raises(ValueError, match="bounds"):
            adapter.sample_t(2, "uniform")

    def test_quantization(self):
        adapter = _FastGenNoiseScheduler(schedules.EDMNoiseScheduler())
        with pytest.raises(NotImplementedError, match="Quantization"):
            adapter.closest_sigma_idx(torch.ones(2))
        with pytest.raises(NotImplementedError, match="Quantization"):
            _ = adapter.sigmas

    def test_missing_grid(self):
        adapter = _FastGenNoiseScheduler(schedules.EDMNoiseScheduler())
        with pytest.raises(ValueError, match="training_times"):
            adapter.get_t_list(4)

    def test_empty_bounds(self):
        adapter = _FastGenNoiseScheduler(schedules.RectifiedFlowNoiseScheduler())
        with pytest.raises(ValueError, match="bounds"):
            adapter.sample_t(2, "uniform", min_t=2)
