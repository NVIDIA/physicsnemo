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

"""Tests for diffusion noise schedulers."""

import pytest
import torch

from physicsnemo.diffusion.noise_schedulers import (
    EDMNoiseScheduler,
    IDDPMNoiseScheduler,
    LinearGaussianNoiseScheduler,
    NoiseScheduler,
    StudentTEDMNoiseScheduler,
    VENoiseScheduler,
    VPNoiseScheduler,
)

from .conftest import GLOBAL_SEED
from .helpers import (
    Conv2dX0Predictor,
    Conv3dX0Predictor,
    FlatLinearX0Predictor,
    compare_outputs,
    gpu_rng_roundtrip,
    instantiate_model_deterministic,
    load_or_create_reference,
    make_input,
)

# =============================================================================
# Constants and Configurations
# =============================================================================

REF_PREFIX = "test_noise_schedulers_"

BATCH = 4
N_SAMPLES = 8
NUM_STEPS = 10

SCHEDULER_CONFIGS = [
    (EDMNoiseScheduler, {}, "edm"),
    (
        EDMNoiseScheduler,
        {"sigma_data": 1.0, "P_mean": -0.5, "P_std": 1.5},
        "edm_custom",
    ),
    (VENoiseScheduler, {}, "ve"),
    (VENoiseScheduler, {"sigma_min": 0.01, "sigma_max": 50.0}, "ve_custom"),
    (IDDPMNoiseScheduler, {}, "iddpm"),
    (IDDPMNoiseScheduler, {"C_1": 0.002, "C_2": 0.01, "M": 500}, "iddpm_custom"),
    (VPNoiseScheduler, {}, "vp"),
    (VPNoiseScheduler, {"beta_min": 0.05, "beta_d": 25.0}, "vp_custom"),
    (StudentTEDMNoiseScheduler, {"nu": 10}, "studentt"),
    (StudentTEDMNoiseScheduler, {"nu": 5, "sigma_data": 1.0}, "studentt_custom"),
]

SPATIAL_CONFIGS = [
    ("1d", (BATCH, 3, 16), FlatLinearX0Predictor, {"features": 3 * 16}),
    ("2d", (BATCH, 3, 8, 6), Conv2dX0Predictor, {"channels": 3}),
    ("3d", (BATCH, 2, 4, 4, 4), Conv3dX0Predictor, {"channels": 2}),
]


# =============================================================================
# Constructor and Attribute Tests
# =============================================================================


class TestEDMNoiseSchedulerConstructor:
    """Tests for EDMNoiseScheduler constructor and attributes."""

    def test_default_attributes(self):
        s = EDMNoiseScheduler()
        assert s.sigma_min == pytest.approx(0.002)
        assert s.sigma_max == pytest.approx(80.0)
        assert s.rho == pytest.approx(7.0)
        assert s.sigma_data == pytest.approx(0.5)
        assert s.P_mean == pytest.approx(-1.2)
        assert s.P_std == pytest.approx(1.2)

    def test_custom_attributes(self):
        s = EDMNoiseScheduler(sigma_min=0.01, sigma_max=100.0, rho=5.0, sigma_data=1.0)
        assert s.sigma_min == pytest.approx(0.01)
        assert s.sigma_max == pytest.approx(100.0)
        assert s.rho == pytest.approx(5.0)
        assert s.sigma_data == pytest.approx(1.0)

    def test_is_noise_scheduler(self):
        assert isinstance(EDMNoiseScheduler(), NoiseScheduler)

    def test_is_linear_gaussian(self):
        assert isinstance(EDMNoiseScheduler(), LinearGaussianNoiseScheduler)


class TestVENoiseSchedulerConstructor:
    """Tests for VENoiseScheduler constructor."""

    def test_default_attributes(self):
        s = VENoiseScheduler()
        assert s.sigma_min == pytest.approx(0.02)
        assert s.sigma_max == pytest.approx(100.0)

    def test_is_noise_scheduler(self):
        assert isinstance(VENoiseScheduler(), NoiseScheduler)


class TestIDDPMNoiseSchedulerConstructor:
    """Tests for IDDPMNoiseScheduler constructor."""

    def test_default_attributes(self):
        s = IDDPMNoiseScheduler()
        assert s.sigma_min == pytest.approx(0.002)
        assert s.sigma_max == pytest.approx(81.0)
        assert s.C_1 == pytest.approx(0.001)
        assert s.C_2 == pytest.approx(0.008)
        assert s.M == 1000
        assert s._u.shape == (1001,)

    def test_custom_attributes(self):
        s = IDDPMNoiseScheduler(M=500)
        assert s.M == 500
        assert s._u.shape == (501,)


class TestVPNoiseSchedulerConstructor:
    """Tests for VPNoiseScheduler constructor."""

    def test_default_attributes(self):
        s = VPNoiseScheduler()
        assert s.beta_min == pytest.approx(0.1)
        assert s.beta_d == pytest.approx(19.1)
        assert s.epsilon_s == pytest.approx(1e-3)
        assert s.t_max == pytest.approx(1.0)


class TestStudentTEDMNoiseSchedulerConstructor:
    """Tests for StudentTEDMNoiseScheduler constructor."""

    def test_default_attributes(self):
        s = StudentTEDMNoiseScheduler()
        assert s.nu == 10

    def test_invalid_nu(self):
        with pytest.raises(ValueError, match="nu must be > 2"):
            StudentTEDMNoiseScheduler(nu=2)


# =============================================================================
# LinearGaussianNoiseScheduler Abstract Base Class Tests
# =============================================================================


class _MinimalScheduler(LinearGaussianNoiseScheduler):
    """Minimal concrete subclass for testing the abstract base class."""

    def sigma(self, t):
        return t

    def sigma_inv(self, sigma):
        return sigma

    def sigma_dot(self, t):
        return torch.ones_like(t)

    def alpha(self, t):
        return torch.ones_like(t)

    def alpha_dot(self, t):
        return torch.zeros_like(t)

    def timesteps(self, num_steps, *, device=None, dtype=None):
        steps = torch.linspace(1, 0.01, num_steps, device=device, dtype=dtype)
        zero = torch.zeros(1, device=device, dtype=dtype)
        return torch.cat([steps, zero])

    def sample_time(self, N, *, device=None, dtype=None):
        return torch.rand(N, device=device, dtype=dtype) * 0.998 + 0.002

    def loss_weight(self, t):
        return 1 / t**2


class TestLinearGaussianNoiseScheduler:
    """Tests for the LinearGaussianNoiseScheduler abstract base class."""

    def test_cannot_instantiate_directly(self):
        """LinearGaussianNoiseScheduler is abstract."""
        with pytest.raises(TypeError, match="abstract method"):
            LinearGaussianNoiseScheduler()

    def test_incomplete_subclass_raises(self):
        """Subclass missing abstract methods cannot be instantiated."""

        class IncompleteScheduler(LinearGaussianNoiseScheduler):
            def sigma(self, t):
                return t

        with pytest.raises(TypeError, match="abstract method"):
            IncompleteScheduler()

    def test_minimal_subclass_satisfies_protocols(self):
        """A minimal complete subclass satisfies both protocols."""
        s = _MinimalScheduler()
        assert isinstance(s, NoiseScheduler)
        assert isinstance(s, LinearGaussianNoiseScheduler)

    def test_concrete_drift(self, device):
        """Default drift: f(x, t) = (alpha_dot / alpha) * x."""
        s = _MinimalScheduler()
        x = make_input((2, 3, 8, 8), seed=1, device=device)
        t = torch.tensor([1.0, 1.0], device=device)
        f = s.drift(x, t)
        assert f.shape == x.shape
        assert torch.allclose(f, torch.zeros_like(f), atol=1e-7)

    def test_concrete_diffusion(self, device):
        """Default diffusion: g^2 = 2*sigma_dot*sigma - 2*(alpha_dot/alpha)*sigma^2."""
        s = _MinimalScheduler()
        x = make_input((2, 3, 8, 8), seed=2, device=device)
        t = torch.tensor([1.0, 1.0], device=device)
        g_sq = s.diffusion(x, t)
        t_bc = t.reshape(-1, 1, 1, 1)
        expected = 2 * t_bc * torch.ones_like(x)
        assert torch.allclose(g_sq.expand_as(x), expected, atol=1e-6)

    def test_concrete_add_noise(self, device):
        s = _MinimalScheduler()
        x0 = make_input((2, 3, 8, 8), seed=3, device=device)
        t = torch.tensor([1.0, 1.0], device=device)
        x_noisy = s.add_noise(x0, t)
        assert x_noisy.shape == x0.shape
        assert not torch.allclose(x_noisy, x0)

    def test_concrete_init_latents(self, device):
        s = _MinimalScheduler()
        tN = torch.tensor([1.0, 1.0], device=device)
        xN = s.init_latents((3, 8, 8), tN, device=device)
        assert xN.shape == (2, 3, 8, 8)

    def test_concrete_get_denoiser(self, device):
        s = _MinimalScheduler()

        def pred(x, t):
            return x * 0.9

        denoiser = s.get_denoiser(x0_predictor=pred, denoising_type="ode")
        x = make_input((2, 3, 8, 8), seed=4, device=device)
        t = torch.tensor([1.0, 1.0], device=device)
        out = denoiser(x, t)
        assert out.shape == x.shape

    def test_concrete_x0_to_score_to_x0(self, device):
        s = _MinimalScheduler()
        x0 = make_input((2, 4), seed=5, device=device)
        x_t = make_input((2, 4), seed=6, device=device)
        t = torch.tensor([0.5, 0.5], device=device)
        score = s.x0_to_score(x0, x_t, t)
        x0_back = s.score_to_x0(score, x_t, t)
        assert torch.allclose(x0, x0_back, atol=1e-5)

    def test_custom_drift_override(self, device):
        """Overriding drift() changes get_denoiser behavior."""

        class CustomDriftScheduler(_MinimalScheduler):
            def drift(self, x, t):
                return -0.5 * x

        def pred(x, t):
            return x * 0.9

        s = CustomDriftScheduler()
        s_default = _MinimalScheduler()
        denoiser_custom = s.get_denoiser(x0_predictor=pred)
        denoiser_default = s_default.get_denoiser(x0_predictor=pred)
        x = make_input((2, 4), seed=7, device=device)
        t = torch.tensor([1.0, 1.0], device=device)
        out_custom = denoiser_custom(x, t)
        out_default = denoiser_default(x, t)
        assert not torch.allclose(out_custom, out_default)


# =============================================================================
# Non-Regression Tests (scalar methods)
# =============================================================================


@pytest.mark.parametrize(
    "sched_cls,sched_kwargs,sched_name",
    SCHEDULER_CONFIGS,
    ids=[c[2] for c in SCHEDULER_CONFIGS],
)
class TestMethodNonRegression:
    """Non-regression tests for scheduler methods with scalar/1D output."""

    def test_sigma(
        self,
        deterministic_settings,
        device,
        tolerances,
        sched_cls,
        sched_kwargs,
        sched_name,
    ):
        s = sched_cls(**sched_kwargs)
        t = make_input((N_SAMPLES,), seed=30, device=device).abs() + 0.1
        sigma_val = s.sigma(t)

        ref_file = f"{REF_PREFIX}{sched_name}_sigma.pth"
        ref = load_or_create_reference(ref_file, lambda: {"sigma": sigma_val.cpu()})
        compare_outputs(sigma_val, ref["sigma"], **tolerances)

    def test_alpha(
        self,
        deterministic_settings,
        device,
        tolerances,
        sched_cls,
        sched_kwargs,
        sched_name,
    ):
        s = sched_cls(**sched_kwargs)
        t = make_input((N_SAMPLES,), seed=30, device=device).abs() + 0.1
        alpha_val = s.alpha(t)

        ref_file = f"{REF_PREFIX}{sched_name}_alpha.pth"
        ref = load_or_create_reference(ref_file, lambda: {"alpha": alpha_val.cpu()})
        compare_outputs(alpha_val, ref["alpha"], **tolerances)

    def test_timesteps(
        self,
        deterministic_settings,
        device,
        tolerances,
        sched_cls,
        sched_kwargs,
        sched_name,
    ):
        s = sched_cls(**sched_kwargs)
        t_steps = s.timesteps(NUM_STEPS, device=device)
        assert t_steps.shape == (NUM_STEPS + 1,)
        assert t_steps[-1].item() == pytest.approx(0.0, abs=1e-7)
        diffs = t_steps[:-1] - t_steps[1:]
        assert (diffs >= -1e-7).all(), "timesteps should be in decreasing order"

        ref_file = f"{REF_PREFIX}{sched_name}_timesteps.pth"
        ref = load_or_create_reference(ref_file, lambda: {"t_steps": t_steps.cpu()})
        compare_outputs(t_steps, ref["t_steps"], **tolerances)

    def test_loss_weight(
        self,
        deterministic_settings,
        device,
        tolerances,
        sched_cls,
        sched_kwargs,
        sched_name,
    ):
        s = sched_cls(**sched_kwargs)
        t = make_input((N_SAMPLES,), seed=20, device=device).abs() + 0.1
        w = s.loss_weight(t)
        assert w.shape == (N_SAMPLES,)
        assert (w > 0).all()

        ref_file = f"{REF_PREFIX}{sched_name}_loss_weight.pth"
        ref = load_or_create_reference(ref_file, lambda: {"w": w.cpu()})
        compare_outputs(w, ref["w"], **tolerances)

    def test_sample_time(
        self,
        deterministic_settings,
        device,
        tolerances,
        sched_cls,
        sched_kwargs,
        sched_name,
    ):
        s = sched_cls(**sched_kwargs)
        if "cuda" in str(device):

            def fn():
                return s.sample_time(N_SAMPLES, device=device)

            result = gpu_rng_roundtrip(fn, GLOBAL_SEED, str(device))
            assert result.shape == (N_SAMPLES,)
            assert (result > 0).all()
            ref_file = f"{REF_PREFIX}{sched_name}_sample_time.pth"
            ref = load_or_create_reference(ref_file, None)
            assert result.shape == ref["t"].shape
        else:
            t = s.sample_time(N_SAMPLES, device=device)
            assert t.shape == (N_SAMPLES,)
            assert (t > 0).all()
            ref_file = f"{REF_PREFIX}{sched_name}_sample_time.pth"
            ref = load_or_create_reference(ref_file, lambda: {"t": t.cpu()})
            compare_outputs(t, ref["t"], **tolerances)

    def test_sigma_roundtrip(
        self,
        device,
        sched_cls,
        sched_kwargs,
        sched_name,
    ):
        """sigma(sigma_inv(sigma)) should recover the original value."""
        s = sched_cls(**sched_kwargs)
        t = make_input((N_SAMPLES,), seed=10, device=device).abs() * 0.3 + 0.1
        sigma_val = s.sigma(t)
        t_recovered = s.sigma_inv(sigma_val)
        compare_outputs(t_recovered, t, atol=5e-2, rtol=5e-2)


# =============================================================================
# Non-Regression Tests (spatial methods)
# =============================================================================


@pytest.mark.parametrize(
    "sched_cls,sched_kwargs,sched_name",
    SCHEDULER_CONFIGS,
    ids=[c[2] for c in SCHEDULER_CONFIGS],
)
@pytest.mark.parametrize(
    "spatial_name,shape,predictor_cls,predictor_kwargs",
    SPATIAL_CONFIGS,
    ids=[c[0] for c in SPATIAL_CONFIGS],
)
class TestSpatialMethodNonRegression:
    """Non-regression tests for scheduler methods that operate on spatial tensors."""

    def test_add_noise(
        self,
        deterministic_settings,
        device,
        tolerances,
        sched_cls,
        sched_kwargs,
        sched_name,
        spatial_name,
        shape,
        predictor_cls,
        predictor_kwargs,
    ):
        s = sched_cls(**sched_kwargs)
        x0 = make_input(shape, seed=50, device=device)
        t = make_input((shape[0],), seed=51, device=device).abs() + 0.5

        ref_file = f"{REF_PREFIX}{sched_name}_{spatial_name}_add_noise.pth"
        if "cuda" in str(device):

            def fn():
                return s.add_noise(x0, t)

            result = gpu_rng_roundtrip(fn, GLOBAL_SEED, str(device))
            assert result.shape == shape
            ref = load_or_create_reference(ref_file, None)
            assert result.shape == ref["x_noisy"].shape
        else:
            x_noisy = s.add_noise(x0, t)
            assert x_noisy.shape == shape
            assert not torch.allclose(x_noisy, x0)
            ref = load_or_create_reference(ref_file, lambda: {"x_noisy": x_noisy.cpu()})
            compare_outputs(x_noisy, ref["x_noisy"], **tolerances)

    def test_init_latents(
        self,
        deterministic_settings,
        device,
        tolerances,
        sched_cls,
        sched_kwargs,
        sched_name,
        spatial_name,
        shape,
        predictor_cls,
        predictor_kwargs,
    ):
        s = sched_cls(**sched_kwargs)
        t_steps = s.timesteps(NUM_STEPS, device=device)
        tN = t_steps[0].expand(shape[0])
        spatial_shape = shape[1:]

        ref_file = f"{REF_PREFIX}{sched_name}_{spatial_name}_init_latents.pth"
        if "cuda" in str(device):

            def fn():
                return s.init_latents(spatial_shape, tN, device=device)

            result = gpu_rng_roundtrip(fn, GLOBAL_SEED, str(device))
            assert result.shape == shape
            ref = load_or_create_reference(ref_file, None)
            assert result.shape == ref["xN"].shape
        else:
            xN = s.init_latents(spatial_shape, tN, device=device)
            assert xN.shape == shape
            ref = load_or_create_reference(ref_file, lambda: {"xN": xN.cpu()})
            compare_outputs(xN, ref["xN"], **tolerances)

    def test_drift_and_diffusion(
        self,
        deterministic_settings,
        device,
        tolerances,
        sched_cls,
        sched_kwargs,
        sched_name,
        spatial_name,
        shape,
        predictor_cls,
        predictor_kwargs,
    ):
        s = sched_cls(**sched_kwargs)
        x = make_input(shape, seed=92, device=device)
        t = make_input((shape[0],), seed=93, device=device).abs() + 0.5
        drift_val = s.drift(x, t)
        diff_val = s.diffusion(x, t)
        assert drift_val.shape == x.shape
        assert diff_val.shape[0] == x.shape[0]

        ref_file = f"{REF_PREFIX}{sched_name}_{spatial_name}_drift_diff.pth"
        ref = load_or_create_reference(
            ref_file,
            lambda: {"drift": drift_val.cpu(), "diff": diff_val.cpu()},
        )
        compare_outputs(drift_val, ref["drift"], **tolerances)
        compare_outputs(diff_val, ref["diff"], **tolerances)

    def test_get_denoiser_ode_x0(
        self,
        deterministic_settings,
        device,
        tolerances,
        sched_cls,
        sched_kwargs,
        sched_name,
        spatial_name,
        shape,
        predictor_cls,
        predictor_kwargs,
    ):
        s = sched_cls(**sched_kwargs)
        model = instantiate_model_deterministic(
            predictor_cls,
            seed=0,
            **predictor_kwargs,
        ).to(device)
        denoiser = s.get_denoiser(x0_predictor=model, denoising_type="ode")
        x = make_input(shape, seed=60, device=device)
        t = make_input((shape[0],), seed=61, device=device).abs() + 0.5
        out = denoiser(x, t)
        assert out.shape == x.shape

        ref_file = f"{REF_PREFIX}{sched_name}_{spatial_name}_ode_x0pred.pth"
        ref = load_or_create_reference(ref_file, lambda: {"out": out.cpu()})
        compare_outputs(out, ref["out"], **tolerances)

    def test_get_denoiser_sde_x0(
        self,
        deterministic_settings,
        device,
        tolerances,
        sched_cls,
        sched_kwargs,
        sched_name,
        spatial_name,
        shape,
        predictor_cls,
        predictor_kwargs,
    ):
        s = sched_cls(**sched_kwargs)
        model = instantiate_model_deterministic(
            predictor_cls,
            seed=0,
            **predictor_kwargs,
        ).to(device)
        denoiser = s.get_denoiser(x0_predictor=model, denoising_type="sde")
        x = make_input(shape, seed=60, device=device)
        t = make_input((shape[0],), seed=61, device=device).abs() + 0.5
        out = denoiser(x, t)
        assert out.shape == x.shape

    def test_get_denoiser_ode_score(
        self,
        deterministic_settings,
        device,
        tolerances,
        sched_cls,
        sched_kwargs,
        sched_name,
        spatial_name,
        shape,
        predictor_cls,
        predictor_kwargs,
    ):
        s = sched_cls(**sched_kwargs)
        model = instantiate_model_deterministic(
            predictor_cls,
            seed=0,
            **predictor_kwargs,
        ).to(device)
        denoiser = s.get_denoiser(score_predictor=model, denoising_type="ode")
        x = make_input(shape, seed=60, device=device)
        t = make_input((shape[0],), seed=61, device=device).abs() + 0.5
        out = denoiser(x, t)
        assert out.shape == x.shape

    def test_get_denoiser_validation(
        self,
        device,
        sched_cls,
        sched_kwargs,
        sched_name,
        spatial_name,
        shape,
        predictor_cls,
        predictor_kwargs,
    ):
        def pred(x, t):
            return x

        s = sched_cls(**sched_kwargs)
        with pytest.raises(ValueError, match="Exactly one"):
            s.get_denoiser(score_predictor=pred, x0_predictor=pred)
        with pytest.raises(ValueError, match="Exactly one"):
            s.get_denoiser()
        with pytest.raises(ValueError, match="denoising_type"):
            s.get_denoiser(x0_predictor=pred, denoising_type="bad")

    def test_x0_to_score_to_x0_roundtrip(
        self,
        device,
        sched_cls,
        sched_kwargs,
        sched_name,
        spatial_name,
        shape,
        predictor_cls,
        predictor_kwargs,
    ):
        s = sched_cls(**sched_kwargs)
        x0 = make_input(shape, seed=70, device=device)
        x_t = make_input(shape, seed=71, device=device)
        t = make_input((shape[0],), seed=72, device=device).abs() * 0.3 + 0.2
        score = s.x0_to_score(x0, x_t, t)
        x0_recovered = s.score_to_x0(score, x_t, t)
        compare_outputs(x0_recovered, x0, atol=1e-3, rtol=1e-3)

    def test_score_to_x0_to_score_roundtrip(
        self,
        device,
        sched_cls,
        sched_kwargs,
        sched_name,
        spatial_name,
        shape,
        predictor_cls,
        predictor_kwargs,
    ):
        s = sched_cls(**sched_kwargs)
        score = make_input(shape, seed=80, device=device)
        x_t = make_input(shape, seed=81, device=device)
        t = make_input((shape[0],), seed=82, device=device).abs() * 0.3 + 0.2
        x0 = s.score_to_x0(score, x_t, t)
        score_recovered = s.x0_to_score(x0, x_t, t)
        compare_outputs(score_recovered, score, atol=1e-3, rtol=1e-3)


# =============================================================================
# Compile Tests — Denoiser Closures
# =============================================================================

# Subset of schedulers for compile tests (avoid combinatorial explosion)
COMPILE_SCHEDULER_CONFIGS = [
    (EDMNoiseScheduler, {}, "edm"),
    (VPNoiseScheduler, {}, "vp"),
    (VENoiseScheduler, {}, "ve"),
]


@pytest.mark.parametrize(
    "denoising_type,predictor_kwarg",
    [("ode", "x0_predictor"), ("ode", "score_predictor"), ("sde", "x0_predictor")],
    ids=["ode_x0", "ode_score", "sde_x0"],
)
@pytest.mark.parametrize(
    "sched_cls,sched_kwargs,sched_name",
    COMPILE_SCHEDULER_CONFIGS,
    ids=[c[2] for c in COMPILE_SCHEDULER_CONFIGS],
)
@pytest.mark.parametrize(
    "spatial_name,shape,predictor_cls,predictor_kwargs",
    SPATIAL_CONFIGS,
    ids=[c[0] for c in SPATIAL_CONFIGS],
)
class TestDenoiserCompile:
    """Double-call compile tests for denoiser closures from get_denoiser()."""

    def test_compiled_denoiser(
        self,
        deterministic_settings,
        device,
        denoising_type,
        predictor_kwarg,
        sched_cls,
        sched_kwargs,
        sched_name,
        spatial_name,
        shape,
        predictor_cls,
        predictor_kwargs,
    ):
        """Compiled denoiser closure matches eager and graph is reused."""
        torch._dynamo.config.error_on_recompile = True

        s = sched_cls(**sched_kwargs)
        model = instantiate_model_deterministic(
            predictor_cls, seed=0, **predictor_kwargs
        ).to(device)
        denoiser = s.get_denoiser(
            **{predictor_kwarg: model}, denoising_type=denoising_type
        )

        x = make_input(shape, seed=60, device=device)
        t = make_input((shape[0],), seed=61, device=device).abs() + 0.5

        compiled_denoiser = torch.compile(denoiser, fullgraph=True)

        with torch.no_grad():
            out_eager = denoiser(x, t)
            out_compiled = compiled_denoiser(x, t)
        torch.testing.assert_close(out_eager, out_compiled)

        # Second call — must reuse the graph
        with torch.no_grad():
            out_compiled_2 = compiled_denoiser(x, t)
        torch.testing.assert_close(out_compiled, out_compiled_2)
